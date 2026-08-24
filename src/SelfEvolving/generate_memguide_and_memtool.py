#!/usr/bin/env python3

import argparse
import ast
import fcntl
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from MemTool.registry import get_tool_registry, load_tool_registry_from_dir, reload_tool_registry
from SelfEvolving.evolve_storage import (
    build_tool_definition_bundle,
    resolve_tool_summary_entries,
    sync_tool_summary,
)
from utils.api_cache import (
    build_cached_completion,
    build_llm_request_identity,
    load_llm_cache_entry,
    record_llm_actual_call,
    record_llm_cache_hit,
    serialize_message_payload,
    store_llm_cache_entry,
)
from utils.logger import log_block, log_info
from utils.timing_registry import timed_block
from utils.generated_paths import (
    add_generated_path_arguments,
    configure_generated_path_env_from_namespace,
    dynamic_path,
    ensure_mem_asset_bootstrap_from_namespace,
    resolve_memguide_dir,
    resolve_memtool_dir,
)


OPENAI_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
OPENAI_BASE_URL_ENV: Final[str] = "OPENAI_BASE_URL"
OPENAI_MODEL_ENV: Final[str] = "OPENAI_MODEL"
DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: Final[str] = "minimax/minimax-m1"
DEFAULT_TIMEOUT_SECONDS: Final[int] = 800
PROMPT_PLACEHOLDER: Final[str] = "<fill in the concrete requirement here>"
AUTHORING_COMMON_PLACEHOLDER: Final[str] = "{{AUTHORING_COMMON}}"
DESIGN_REQUIREMENT_PLACEHOLDER: Final[str] = "{{DESIGN_REQUIREMENT}}"
EXISTING_TOOL_DEFINITIONS_PLACEHOLDER: Final[str] = "{{EXISTING_TOOL_DEFINITIONS}}"
REUSABLE_TOOL_DEFINITIONS_PLACEHOLDER: Final[str] = "{{REUSABLE_TOOL_DEFINITIONS}}"
REUSABLE_TOOL_SOURCE_CODE_PLACEHOLDER: Final[str] = "{{REUSABLE_TOOL_SOURCE_CODE}}"
GUIDE_CATEGORY_REPRESENTATIVES_PLACEHOLDER: Final[str] = "{{GUIDE_CATEGORY_REPRESENTATIVES}}"
NEW_GUIDE_CANDIDATE_PLACEHOLDER: Final[str] = "{{NEW_GUIDE_CANDIDATE}}"
CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```([A-Za-z0-9_+-]+)[ \t]*\n(.*?)\n```",
    re.DOTALL,
)

SRC_DIR: Final[Path] = Path(__file__).resolve().parents[1]
SELF_EVOLVING_DIR: Final[Path] = SRC_DIR / "SelfEvolving"
PROMPT_DIR: Final[Path] = SELF_EVOLVING_DIR / "prompt"
MEMGUIDE_DIR = dynamic_path(resolve_memguide_dir)
MEMTOOL_DIR = dynamic_path(resolve_memtool_dir)
DEFAULT_PROMPT_PATH: Final[Path] = PROMPT_DIR / "mem_evolution_authoring_prompt.md"
DEFAULT_REUSE_SELECTION_PROMPT_PATH: Final[Path] = PROMPT_DIR / "mem_evolution_reuse_selection_prompt.md"
DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH: Final[Path] = PROMPT_DIR / "mem_evolution_guide_classification_prompt.md"
DEFAULT_EXPLORATION_PROMPT_PATH: Final[Path] = PROMPT_DIR / "mem_evolution_exploration_prompt.md"
MEM_ASSET_GENERATION_LOCK_PATH: Final[Path] = SELF_EVOLVING_DIR / ".mem_asset_generation.lock"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class ParsedResponse:
    guide_json_text: str
    tool_code_texts: list[str]


@dataclass(frozen=True)
class ValidatedGuide:
    guide_object: dict[str, Any]
    formatted_json_text: str
    guide_name: str
    tool_names: list[str]


@dataclass(frozen=True)
class ValidatedTool:
    tool_name: str
    code_text: str


@dataclass(frozen=True)
class ReuseSelection:
    candidate_tool_names: list[str]
    analysis: str


@dataclass(frozen=True)
class GuideClassificationSelection:
    matched_representative_guide_file: str | None
    analysis: str


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                continue
            part_type = getattr(part, "type", None)
            if part_type == "text":
                text_parts.append(str(getattr(part, "text", "") or ""))
        return "".join(text_parts)
    return str(content or "")


def _resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    api_key = (
        str(args.api_key or "").strip()
        or str(os.getenv(OPENAI_API_KEY_ENV, "")).strip()
        or str(os.getenv(OPENROUTER_API_KEY_ENV, "")).strip()
    )
    if not api_key:
        raise ValueError(
            "Missing API key. Pass --api_key or set OPENAI_API_KEY / OPENROUTER_API_KEY."
        )

    base_url = (
        str(args.base_url or "").strip()
        or str(os.getenv(OPENAI_BASE_URL_ENV, "")).strip()
        or DEFAULT_BASE_URL
    )
    model = (
        str(args.model or "").strip()
        or str(os.getenv(OPENAI_MODEL_ENV, "")).strip()
        or DEFAULT_MODEL
    )
    return LLMConfig(api_key=api_key, base_url=base_url, model=model)


def _load_design_requirement(args: argparse.Namespace) -> str:
    if args.design_requirement:
        requirement = str(args.design_requirement).strip()
    else:
        requirement = Path(args.design_requirement_file).read_text(encoding="utf-8").strip()

    if not requirement:
        raise ValueError("The design requirement must be non-empty.")
    return requirement


def _resolve_authoring_common_prompt_path(prompt_path: Path) -> Path:
    if prompt_path.name.endswith("-cn.md"):
        return PROMPT_DIR / "mem_evolution_authoring_common-cn.md"
    return PROMPT_DIR / "mem_evolution_authoring_common.md"


def _render_prompt_and_lengths(
    prompt_path: Path,
    *,
    design_requirement: str = "",
    existing_tool_definitions: str = "",
    reusable_tool_definitions: str = "",
    reusable_tool_source_code: str = "",
    guide_category_representatives: str = "",
    new_guide_candidate: str = "",
) -> tuple[str, dict[str, int]]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    authoring_common_text = ""
    expanded_template = prompt_template
    if AUTHORING_COMMON_PLACEHOLDER in expanded_template:
        common_prompt_path = _resolve_authoring_common_prompt_path(prompt_path)
        authoring_common_text = common_prompt_path.read_text(encoding="utf-8")
        expanded_template = expanded_template.replace(
            AUTHORING_COMMON_PLACEHOLDER,
            authoring_common_text,
            1,
        )

    prompt = expanded_template
    if DESIGN_REQUIREMENT_PLACEHOLDER in prompt:
        if not design_requirement:
            raise ValueError("Prompt template requires a non-empty design requirement.")
        prompt = prompt.replace(DESIGN_REQUIREMENT_PLACEHOLDER, design_requirement, 1)
    elif PROMPT_PLACEHOLDER in prompt:
        if not design_requirement:
            raise ValueError("Prompt template requires a non-empty design requirement.")
        prompt = prompt.replace(PROMPT_PLACEHOLDER, design_requirement, 1)

    if EXISTING_TOOL_DEFINITIONS_PLACEHOLDER in prompt:
        prompt = prompt.replace(
            EXISTING_TOOL_DEFINITIONS_PLACEHOLDER,
            existing_tool_definitions or "None.",
            1,
        )
    if REUSABLE_TOOL_DEFINITIONS_PLACEHOLDER in prompt:
        prompt = prompt.replace(
            REUSABLE_TOOL_DEFINITIONS_PLACEHOLDER,
            reusable_tool_definitions or "None.",
            1,
        )
    if REUSABLE_TOOL_SOURCE_CODE_PLACEHOLDER in prompt:
        prompt = prompt.replace(
            REUSABLE_TOOL_SOURCE_CODE_PLACEHOLDER,
            reusable_tool_source_code or "None.",
            1,
        )
    if GUIDE_CATEGORY_REPRESENTATIVES_PLACEHOLDER in prompt:
        prompt = prompt.replace(
            GUIDE_CATEGORY_REPRESENTATIVES_PLACEHOLDER,
            guide_category_representatives or "None.",
            1,
        )
    if NEW_GUIDE_CANDIDATE_PLACEHOLDER in prompt:
        prompt = prompt.replace(
            NEW_GUIDE_CANDIDATE_PLACEHOLDER,
            new_guide_candidate or "None.",
            1,
        )
    prompt_lengths = {
        "total_prompt_chars": len(prompt),
        "template_chars": len(prompt_template),
        "authoring_common_chars": len(authoring_common_text),
        "design_requirement_chars": len(design_requirement),
        "existing_tool_definitions_chars": len(existing_tool_definitions),
        "reusable_tool_definitions_chars": len(reusable_tool_definitions),
        "reusable_tool_source_code_chars": len(reusable_tool_source_code),
        "guide_category_representatives_chars": len(guide_category_representatives),
        "new_guide_candidate_chars": len(new_guide_candidate),
    }
    return prompt, prompt_lengths


def _extract_json_object_text(response_text: str) -> str:
    normalized_text = str(response_text or "").strip()
    json_start = normalized_text.find("{")
    json_end = normalized_text.rfind("}")
    if json_start == -1 or json_end == -1 or json_end < json_start:
        raise ValueError("The LLM response must contain a JSON object.")
    return normalized_text[json_start : json_end + 1]


def _request_llm_response(prompt: str, llm_config: LLMConfig, *, stage: str) -> str:
    from openai import OpenAI

    messages = [{"role": "user", "content": prompt}]
    request_identity = build_llm_request_identity(
        base_url=llm_config.base_url,
        model=llm_config.model,
        messages=messages,
        tools=None,
        tool_choice=None,
    )
    log_info(
        "self_evolving",
        f"LLM request start | model={llm_config.model} | base_url={llm_config.base_url}",
    )
    # log_block("self_evolving", "Prompt sent to LLM", prompt)
    cache_key, cache_entry = load_llm_cache_entry(request_identity)
    if cache_entry is not None:
        record_llm_cache_hit()
        log_info("self_evolving", f"LLM cache hit | stage={stage} | cache_key={cache_key}")
        completion = build_cached_completion(cache_entry.get("response", {}).get("message", {}))
        message = completion.choices[0].message
        text = _extract_message_text(message).strip()
        log_block("self_evolving", "LLM response", text)
        if not text:
            raise ValueError("The LLM returned an empty response.")
        return text
    log_info("self_evolving", f"LLM cache miss | stage={stage} | cache_key={cache_key}")
    client = OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    with timed_block(
        "llm.self_evolving_generation",
        stage,
        kind="llm",
        metadata={
            "model": llm_config.model,
            "base_url": llm_config.base_url,
        },
    ):
        record_llm_actual_call()
        completion = client.chat.completions.create(
            model=llm_config.model,
            messages=messages,
        )
    message = completion.choices[0].message
    text = _extract_message_text(message).strip()
    log_block("self_evolving", "LLM response", text)
    if not text:
        raise ValueError("The LLM returned an empty response.")
    store_llm_cache_entry(
        cache_key=cache_key,
        request_identity=request_identity,
        message_payload=serialize_message_payload(
            content=text,
            tool_calls=list(getattr(message, "tool_calls", None) or []),
        ),
    )
    return text


def _parse_reuse_selection_response(
    response_text: str,
    *,
    available_tool_names: list[str],
) -> ReuseSelection:
    try:
        payload = json.loads(_extract_json_object_text(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reuse-selection JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("The reuse-selection response must be a JSON object.")

    raw_candidates = payload.get("candidate_tool_names", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate_tool_names must be a list.")

    available_set = set(available_tool_names)
    candidate_tool_names: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_candidates:
        tool_name = str(raw_name or "").strip()
        if not tool_name or tool_name not in available_set or tool_name in seen:
            continue
        seen.add(tool_name)
        candidate_tool_names.append(tool_name)
        if len(candidate_tool_names) >= 3:
            break

    analysis = str(payload.get("analysis", "") or "").strip()
    return ReuseSelection(
        candidate_tool_names=candidate_tool_names,
        analysis=analysis,
    )


def _parse_guide_classification_response(
    response_text: str,
    *,
    available_representative_guide_files: list[str],
) -> GuideClassificationSelection:
    try:
        payload = json.loads(_extract_json_object_text(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid guide-classification JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("The guide-classification response must be a JSON object.")

    raw_match = payload.get("matched_representative_guide_file")
    matched_representative_guide_file: str | None
    if raw_match is None:
        matched_representative_guide_file = None
    else:
        candidate = str(raw_match or "").strip()
        if not candidate or candidate.lower() in {"null", "none"}:
            matched_representative_guide_file = None
        elif candidate not in set(available_representative_guide_files):
            matched_representative_guide_file = None
        else:
            matched_representative_guide_file = candidate

    analysis = str(payload.get("analysis", "") or "").strip()
    if not analysis:
        raise ValueError("The guide-classification response must define a non-empty analysis.")

    return GuideClassificationSelection(
        matched_representative_guide_file=matched_representative_guide_file,
        analysis=analysis,
    )


def _parse_llm_response(response_text: str) -> ParsedResponse:
    matches = list(CODE_BLOCK_PATTERN.finditer(response_text))
    if not matches:
        raise ValueError("The LLM response must contain fenced code blocks.")

    cursor = 0
    blocks: list[tuple[str, str]] = []
    for match in matches:
        if response_text[cursor:match.start()].strip():
            raise ValueError("The LLM response contains non-code text outside fenced blocks.")
        language = match.group(1).strip().lower()
        code_text = match.group(2)
        blocks.append((language, code_text))
        cursor = match.end()

    if response_text[cursor:].strip():
        raise ValueError("The LLM response contains trailing non-code text outside fenced blocks.")

    first_language, first_code = blocks[0]
    if first_language != "json":
        raise ValueError("The first fenced block must be a json MemGuide block.")

    tool_code_texts: list[str] = []
    for language, code_text in blocks[1:]:
        if language != "python":
            raise ValueError("All fenced blocks after the MemGuide block must be python.")
        tool_code_texts.append(code_text)

    return ParsedResponse(
        guide_json_text=first_code,
        tool_code_texts=tool_code_texts,
    )


def _validate_guide(guide_json_text: str) -> ValidatedGuide:
    try:
        guide_object = json.loads(guide_json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MemGuide JSON: {exc}") from exc

    if not isinstance(guide_object, dict):
        raise ValueError("The MemGuide must be a JSON object.")

    guide_name = str(guide_object.get("guide_name", "") or "").strip()
    prompt = str(guide_object.get("prompt", "") or "").strip()
    raw_tool_names = guide_object.get("tool_names")

    if not guide_name:
        raise ValueError("The MemGuide must define a non-empty guide_name.")
    if not prompt:
        raise ValueError("The MemGuide must define a non-empty prompt.")
    if not isinstance(raw_tool_names, list) or not raw_tool_names:
        raise ValueError("The MemGuide must define a non-empty tool_names list.")

    tool_names: list[str] = []
    for raw_name in raw_tool_names:
        tool_name = str(raw_name or "").strip()
        if not tool_name:
            raise ValueError("Every MemGuide tool_names entry must be a non-empty string.")
        tool_names.append(tool_name)

    return ValidatedGuide(
        guide_object=guide_object,
        formatted_json_text=json.dumps(guide_object, ensure_ascii=False, indent=2) + "\n",
        guide_name=guide_name,
        tool_names=tool_names,
    )


def _get_top_level_assignments(module: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments[node.target.id] = node.value
    return assignments


def _eval_simple_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unsupported name reference during validation: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Dict):
        return {
            _eval_simple_ast(key_node, env): _eval_simple_ast(value_node, env)
            for key_node, value_node in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_eval_simple_ast(element, env) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_simple_ast(element, env) for element in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_simple_ast(element, env) for element in node.elts}
    raise ValueError(f"Unsupported AST expression during validation: {ast.dump(node, include_attributes=False)}")


def _validate_tool_code(tool_code_text: str) -> ValidatedTool:
    try:
        module = ast.parse(tool_code_text)
    except SyntaxError as exc:
        raise ValueError(f"Invalid MemTool Python syntax: {exc}") from exc

    assignments = _get_top_level_assignments(module)
    required_exports = {"TOOL_NAME", "TOOL_SPEC", "__all__"}
    missing_exports = sorted(name for name in required_exports if name not in assignments)
    if missing_exports:
        raise ValueError(f"MemTool is missing required top-level assignments: {', '.join(missing_exports)}")

    function_names = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_functions = sorted(name for name in ("build_runner_kwargs", "run_tool") if name not in function_names)
    if missing_functions:
        raise ValueError(f"MemTool is missing required functions: {', '.join(missing_functions)}")

    env: dict[str, Any] = {}
    tool_name = _eval_simple_ast(assignments["TOOL_NAME"], env)
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("MemTool TOOL_NAME must be a non-empty string.")
    env["TOOL_NAME"] = tool_name

    tool_spec = _eval_simple_ast(assignments["TOOL_SPEC"], env)
    if not isinstance(tool_spec, dict):
        raise ValueError(f"{tool_name} TOOL_SPEC must evaluate to a dict.")
    if tool_spec.get("type") != "function":
        raise ValueError(f"{tool_name} TOOL_SPEC must have type='function'.")
    function_spec = tool_spec.get("function")
    if not isinstance(function_spec, dict):
        raise ValueError(f"{tool_name} TOOL_SPEC.function must be a dict.")
    if function_spec.get("name") != tool_name:
        raise ValueError(f"{tool_name} TOOL_SPEC.function.name must match TOOL_NAME.")
    parameters = function_spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{tool_name} TOOL_SPEC.function.parameters must be a JSON Schema object.")

    exports = _eval_simple_ast(assignments["__all__"], env)
    if not isinstance(exports, (list, tuple)):
        raise ValueError(f"{tool_name} __all__ must be a list or tuple.")
    export_set = {str(value) for value in exports}
    required_all_exports = {"TOOL_NAME", "TOOL_SPEC", "build_runner_kwargs", "run_tool"}
    if not required_all_exports.issubset(export_set):
        raise ValueError(
            f"{tool_name} __all__ must include TOOL_NAME, TOOL_SPEC, build_runner_kwargs, and run_tool."
        )

    return ValidatedTool(tool_name=tool_name, code_text=tool_code_text)


def _build_line_offsets(text: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _node_to_span(text: str, node: ast.AST) -> tuple[int, int]:
    if (
        getattr(node, "lineno", None) is None
        or getattr(node, "col_offset", None) is None
        or getattr(node, "end_lineno", None) is None
        or getattr(node, "end_col_offset", None) is None
    ):
        raise ValueError("Cannot rewrite tool code because AST node positions are unavailable.")

    line_offsets = _build_line_offsets(text)
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _replace_spans(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    updated = text
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _find_tool_spec_function_name_node(tool_spec_node: ast.AST) -> ast.AST | None:
    if not isinstance(tool_spec_node, ast.Dict):
        return None

    function_node: ast.AST | None = None
    for key_node, value_node in zip(tool_spec_node.keys, tool_spec_node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == "function":
            function_node = value_node
            break
    if not isinstance(function_node, ast.Dict):
        return None

    for key_node, value_node in zip(function_node.keys, function_node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == "name":
            return value_node
    return None


def _rename_tool_code(tool: ValidatedTool, new_tool_name: str) -> ValidatedTool:
    if tool.tool_name == new_tool_name:
        return tool

    module = ast.parse(tool.code_text)
    assignments = _get_top_level_assignments(module)
    tool_name_node = assignments.get("TOOL_NAME")
    tool_spec_node = assignments.get("TOOL_SPEC")
    if tool_name_node is None or tool_spec_node is None:
        raise ValueError("MemTool is missing TOOL_NAME or TOOL_SPEC during rename.")

    replacements: list[tuple[int, int, str]] = []
    tool_name_start, tool_name_end = _node_to_span(tool.code_text, tool_name_node)
    replacements.append((tool_name_start, tool_name_end, json.dumps(new_tool_name)))

    function_name_node = _find_tool_spec_function_name_node(tool_spec_node)
    if function_name_node is None:
        raise ValueError("MemTool TOOL_SPEC.function.name is missing during rename.")
    if not (isinstance(function_name_node, ast.Name) and function_name_node.id == "TOOL_NAME"):
        function_name_start, function_name_end = _node_to_span(tool.code_text, function_name_node)
        replacements.append((function_name_start, function_name_end, json.dumps(new_tool_name)))

    renamed_code_text = _replace_spans(tool.code_text, replacements)
    renamed_tool = _validate_tool_code(renamed_code_text)
    if renamed_tool.tool_name != new_tool_name:
        raise ValueError(
            f"Failed to rename MemTool from {tool.tool_name} to {new_tool_name}."
        )
    return renamed_tool


def _next_unique_tool_name(base_name: str, reserved_names: set[str]) -> str:
    if base_name not in reserved_names:
        reserved_names.add(base_name)
        return base_name

    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in reserved_names:
            reserved_names.add(candidate)
            return candidate
        suffix += 1


def _rename_guide_tool_names(
    guide: ValidatedGuide,
    renamed_tools_by_original_name: dict[str, list[str]],
) -> ValidatedGuide:
    single_replacement_map = {
        tool_name: names[0]
        for tool_name, names in renamed_tools_by_original_name.items()
        if len(names) == 1 and names[0] != tool_name
    }

    def _replace_tool_mentions_in_string(text: str) -> str:
        updated_text = text
        for original_name, renamed_name in single_replacement_map.items():
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(original_name)}(?![A-Za-z0-9_])"
            )
            updated_text = pattern.sub(renamed_name, updated_text)
        return updated_text

    def _rewrite_guide_value(value: Any) -> Any:
        if isinstance(value, str):
            return _replace_tool_mentions_in_string(value)
        if isinstance(value, list):
            return [_rewrite_guide_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _rewrite_guide_value(item) for key, item in value.items()}
        return value

    updated_tool_names: list[str] = []
    remaining_names = {
        tool_name: list(names)
        for tool_name, names in renamed_tools_by_original_name.items()
    }

    for tool_name in guide.tool_names:
        pending_names = remaining_names.get(tool_name)
        if pending_names:
            updated_tool_names.append(pending_names.pop(0))
            if not pending_names:
                remaining_names.pop(tool_name, None)
            continue
        updated_tool_names.append(tool_name)

    updated_guide_object = _rewrite_guide_value(dict(guide.guide_object))
    updated_guide_object["tool_names"] = updated_tool_names
    return _validate_guide(json.dumps(updated_guide_object, ensure_ascii=False))


def _assign_unique_tool_names(
    guide: ValidatedGuide,
    tools: list[ValidatedTool],
) -> tuple[ValidatedGuide, list[ValidatedTool], dict[str, list[str]]]:
    reserved_names = set(get_tool_registry())
    renamed_tools: list[ValidatedTool] = []
    renamed_tools_by_original_name: dict[str, list[str]] = {}
    renamed_tool_names: dict[str, list[str]] = {}

    for tool in tools:
        new_tool_name = _next_unique_tool_name(tool.tool_name, reserved_names)
        renamed_tool = _rename_tool_code(tool, new_tool_name)
        renamed_tools.append(renamed_tool)
        renamed_tools_by_original_name.setdefault(tool.tool_name, []).append(new_tool_name)
        if new_tool_name != tool.tool_name:
            renamed_tool_names.setdefault(tool.tool_name, []).append(new_tool_name)

    renamed_guide = _rename_guide_tool_names(guide, renamed_tools_by_original_name)
    return renamed_guide, renamed_tools, renamed_tool_names


def _validate_cross_references(guide: ValidatedGuide, tools: list[ValidatedTool]) -> None:
    existing_tool_names = set(get_tool_registry())
    tool_summary_payload = sync_tool_summary()
    valid_existing_tool_names = set(tool_summary_payload.get("tools", {}).keys())
    new_tool_names = [tool.tool_name for tool in tools]

    duplicate_new_names = sorted({name for name in new_tool_names if new_tool_names.count(name) > 1})
    if duplicate_new_names:
        raise ValueError(f"Duplicate TOOL_NAME values in new MemTools: {', '.join(duplicate_new_names)}")

    conflicting_names = sorted(name for name in new_tool_names if name in existing_tool_names)
    if conflicting_names:
        raise ValueError(f"New MemTool TOOL_NAME conflicts with existing registry: {', '.join(conflicting_names)}")

    available_tool_names = valid_existing_tool_names | set(new_tool_names)
    missing_tool_names = sorted(name for name in guide.tool_names if name not in available_tool_names)
    if missing_tool_names:
        raise ValueError(f"MemGuide tool_names reference unavailable tools: {', '.join(missing_tool_names)}")


def _find_next_numbered_path(directory: Path, prefix: str, suffix: str) -> Path:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    used_numbers: set[int] = set()
    for child in directory.iterdir():
        if not child.is_file():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        used_numbers.add(int(match.group(1)))

    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return directory / f"{prefix}_{next_number}{suffix}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_file.write(text)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = Path(tmp_file.name)
    os.replace(temp_path, path)


def _plan_output_paths(tool_count: int) -> tuple[Path, list[Path]]:
    MEMGUIDE_DIR.mkdir(parents=True, exist_ok=True)
    MEMTOOL_DIR.mkdir(parents=True, exist_ok=True)
    guide_path = _find_next_numbered_path(MEMGUIDE_DIR, "guide", ".json")

    reserved_tool_paths: list[Path] = []
    temp_dir = MEMTOOL_DIR
    used_numbers: set[int] = set()
    pattern = re.compile(r"^tool_(\d+)\.py$")
    for child in temp_dir.iterdir():
        if not child.is_file():
            continue
        match = pattern.match(child.name)
        if match is not None:
            used_numbers.add(int(match.group(1)))

    next_number = 1
    while len(reserved_tool_paths) < tool_count:
        if next_number not in used_numbers:
            reserved_tool_paths.append(MEMTOOL_DIR / f"tool_{next_number}.py")
            used_numbers.add(next_number)
        next_number += 1

    return guide_path, reserved_tool_paths


def _write_staged_assets(
    guide: ValidatedGuide,
    tools: list[ValidatedTool],
    *,
    stage_dir: Path,
) -> tuple[Path, list[Path], Path, list[Path]]:
    guide_path, tool_paths = _plan_output_paths(len(tools))
    staged_guide_path = stage_dir / guide_path.name
    staged_tool_paths = [stage_dir / tool_path.name for tool_path in tool_paths]
    _atomic_write_text(staged_guide_path, guide.formatted_json_text)

    written_tool_paths: list[Path] = []
    try:
        for tool, staged_tool_path in zip(tools, staged_tool_paths):
            _atomic_write_text(staged_tool_path, tool.code_text)
            written_tool_paths.append(staged_tool_path)
    except Exception:
        if staged_guide_path.exists():
            staged_guide_path.unlink()
        for path in written_tool_paths:
            if path.exists():
                path.unlink()
        raise

    return guide_path, tool_paths, staged_guide_path, staged_tool_paths


def _build_validation_memtool_dir(
    *,
    stage_dir: Path,
    final_tool_paths: list[Path],
    staged_tool_paths: list[Path],
) -> Path:
    validation_memtool_dir = stage_dir / "validation_memtool"
    validation_memtool_dir.mkdir(parents=True, exist_ok=True)

    for child in MEMTOOL_DIR.iterdir():
        if not child.is_file():
            continue
        if child.suffix != ".py":
            continue
        if not child.stem.startswith("tool_"):
            continue
        shutil.copy2(child, validation_memtool_dir / child.name)

    for final_tool_path, staged_tool_path in zip(final_tool_paths, staged_tool_paths):
        shutil.copy2(staged_tool_path, validation_memtool_dir / final_tool_path.name)

    return validation_memtool_dir


def _validate_staged_assets(
    *,
    guide_path: Path,
    tool_paths: list[Path],
    stage_dir: Path,
    staged_tool_paths: list[Path],
) -> None:
    try:
        validation_memtool_dir = _build_validation_memtool_dir(
            stage_dir=stage_dir,
            final_tool_paths=tool_paths,
            staged_tool_paths=staged_tool_paths,
        )
        load_tool_registry_from_dir(validation_memtool_dir)
    except Exception as exc:
        raise ValueError(
            "Generated MemGuide/MemTool assets failed staged registry validation: "
            f"guide={guide_path.name} | tools={[path.name for path in tool_paths]} | error={exc}"
        ) from exc


def _publish_staged_assets(
    *,
    guide_path: Path,
    tool_paths: list[Path],
    staged_guide_path: Path,
    staged_tool_paths: list[Path],
) -> None:
    for staged_tool_path, tool_path in zip(staged_tool_paths, tool_paths):
        os.replace(staged_tool_path, tool_path)
    os.replace(staged_guide_path, guide_path)
    reload_tool_registry()


def generate_assets_from_response_text(response_text: str) -> dict[str, Any]:
    MEM_ASSET_GENERATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEM_ASSET_GENERATION_LOCK_PATH.touch(exist_ok=True)

    with MEM_ASSET_GENERATION_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            reload_tool_registry()
            parsed = _parse_llm_response(response_text)
            guide = _validate_guide(parsed.guide_json_text)
            tools = [_validate_tool_code(code_text) for code_text in parsed.tool_code_texts]
            guide, tools, renamed_tool_names = _assign_unique_tool_names(guide, tools)
            if renamed_tool_names:
                log_info(
                    "self_evolving",
                    f"Auto-renamed duplicate TOOL_NAME values | mappings={renamed_tool_names}",
                )
            _validate_cross_references(guide, tools)
            MEMTOOL_DIR.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=MEMTOOL_DIR.parent,
                prefix=".mem-asset-stage-",
            ) as stage_dir_str:
                stage_dir = Path(stage_dir_str)
                guide_path, tool_paths, staged_guide_path, staged_tool_paths = _write_staged_assets(
                    guide,
                    tools,
                    stage_dir=stage_dir,
                )
                _validate_staged_assets(
                    guide_path=guide_path,
                    tool_paths=tool_paths,
                    stage_dir=stage_dir,
                    staged_tool_paths=staged_tool_paths,
                )
                _publish_staged_assets(
                    guide_path=guide_path,
                    tool_paths=tool_paths,
                    staged_guide_path=staged_guide_path,
                    staged_tool_paths=staged_tool_paths,
                )
            return {
                "guide_path": str(guide_path),
                "tool_paths": [str(path) for path in tool_paths],
                "guide_name": guide.guide_name,
                "tool_names": [tool.tool_name for tool in tools],
                "renamed_tool_names": renamed_tool_names,
            }
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def select_reusable_tools_from_design_requirement(
    *,
    design_requirement: str,
    tool_summary_payload: dict[str, Any],
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_REUSE_SELECTION_PROMPT_PATH,
) -> dict[str, Any]:
    normalized_requirement = str(design_requirement or "").strip()
    if not normalized_requirement:
        raise ValueError("The design requirement must be non-empty.")

    resolved_prompt_path = Path(prompt_path)
    tool_summary_tools = tool_summary_payload.get("tools", {})
    if not isinstance(tool_summary_tools, dict):
        raise ValueError("Invalid tool summary payload: missing tools dict.")

    all_tool_names = list(tool_summary_tools.keys())
    existing_tool_definitions = build_tool_definition_bundle(all_tool_names, tool_summary_payload)
    prompt, prompt_lengths = _render_prompt_and_lengths(
        resolved_prompt_path,
        design_requirement=normalized_requirement,
        existing_tool_definitions=existing_tool_definitions,
    )
    response_text = _request_llm_response(
        prompt,
        llm_config,
        stage="reuse_tool_selection",
    )
    parsed = _parse_reuse_selection_response(
        response_text,
        available_tool_names=all_tool_names,
    )
    candidate_tool_entries = resolve_tool_summary_entries(
        parsed.candidate_tool_names,
        tool_summary_payload,
    )
    return {
        "prompt_path": str(resolved_prompt_path),
        "prompt_lengths": prompt_lengths,
        "response_text": response_text,
        "analysis": parsed.analysis,
        "candidate_tool_names": parsed.candidate_tool_names,
        "candidate_tool_entries": candidate_tool_entries,
    }


def classify_guide_with_representatives(
    *,
    representative_guide_files: list[str],
    guide_category_representatives_bundle: str,
    new_guide_candidate_bundle: str,
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH,
) -> dict[str, Any]:
    resolved_prompt_path = Path(prompt_path)
    normalized_representative_guide_files = [
        str(guide_file or "").strip()
        for guide_file in representative_guide_files
        if str(guide_file or "").strip()
    ]
    prompt, prompt_lengths = _render_prompt_and_lengths(
        resolved_prompt_path,
        guide_category_representatives=guide_category_representatives_bundle,
        new_guide_candidate=new_guide_candidate_bundle,
    )
    response_text = _request_llm_response(
        prompt,
        llm_config,
        stage="guide_classification",
    )
    parsed = _parse_guide_classification_response(
        response_text,
        available_representative_guide_files=normalized_representative_guide_files,
    )
    return {
        "prompt_path": str(resolved_prompt_path),
        "prompt_lengths": prompt_lengths,
        "response_text": response_text,
        "analysis": parsed.analysis,
        "matched_representative_guide_file": parsed.matched_representative_guide_file,
    }


def generate_assets_from_design_requirement(
    *,
    design_requirement: str,
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    reusable_tool_entries: list[dict[str, Any]] | None = None,
    reusable_tool_source_bundle: str = "",
) -> dict[str, Any]:
    normalized_requirement = str(design_requirement or "").strip()
    if not normalized_requirement:
        raise ValueError("The design requirement must be non-empty.")

    resolved_prompt_path = Path(prompt_path)
    normalized_reusable_entries = list(reusable_tool_entries or [])
    reusable_tool_definitions = build_tool_definition_bundle(
        [str(entry.get("tool_name", "") or "").strip() for entry in normalized_reusable_entries if str(entry.get("tool_name", "") or "").strip()],
        {
            "tools": {
                str(entry.get("tool_name", "") or "").strip(): entry
                for entry in normalized_reusable_entries
                if str(entry.get("tool_name", "") or "").strip()
            }
        },
    ) if normalized_reusable_entries else ""
    prompt, prompt_lengths = _render_prompt_and_lengths(
        resolved_prompt_path,
        design_requirement=normalized_requirement,
        reusable_tool_definitions=reusable_tool_definitions,
        reusable_tool_source_code=reusable_tool_source_bundle,
    )
    response_text = _request_llm_response(
        prompt,
        llm_config,
        stage="design_requirement_generation",
    )
    summary = generate_assets_from_response_text(response_text)
    summary["design_requirement"] = normalized_requirement
    summary["prompt_path"] = str(resolved_prompt_path)
    summary["prompt_lengths"] = prompt_lengths
    summary["response_text"] = response_text
    summary["reusable_tool_entries"] = normalized_reusable_entries
    summary["reusable_tool_source_bundle"] = reusable_tool_source_bundle
    return summary


def generate_assets_from_exploration_context(
    *,
    guide_category_representatives_bundle: str,
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_EXPLORATION_PROMPT_PATH,
) -> dict[str, Any]:
    resolved_prompt_path = Path(prompt_path)
    prompt, prompt_lengths = _render_prompt_and_lengths(
        resolved_prompt_path,
        guide_category_representatives=guide_category_representatives_bundle,
    )
    response_text = _request_llm_response(
        prompt,
        llm_config,
        stage="exploration_generation",
    )
    summary = generate_assets_from_response_text(response_text)
    summary["prompt_path"] = str(resolved_prompt_path)
    summary["prompt_lengths"] = prompt_lengths
    summary["response_text"] = response_text
    summary["guide_category_representatives_bundle"] = guide_category_representatives_bundle
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate new MemGuide and MemTool assets with an LLM.")
    add_generated_path_arguments(parser)
    requirement_group = parser.add_mutually_exclusive_group(required=True)
    requirement_group.add_argument(
        "--design_requirement",
        help="Inline design requirement text to inject into the prompt template.",
    )
    requirement_group.add_argument(
        "--design_requirement_file",
        help="UTF-8 text file containing the design requirement to inject into the prompt template.",
    )
    parser.add_argument(
        "--prompt_path",
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to the MemGuide/MemTool authoring prompt template.",
    )
    parser.add_argument("--api_key", "-k", default=None, help="OpenAI-compatible API key.")
    parser.add_argument(
        "--base_url",
        "-u",
        default=None,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name to use for generation.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    configure_generated_path_env_from_namespace(args)
    bootstrap_actions = ensure_mem_asset_bootstrap_from_namespace(args)
    reload_tool_registry()

    try:
        if bootstrap_actions:
            log_info(
                "self_evolving",
                f"Bootstrapped initial mem assets | actions={bootstrap_actions}",
            )
        llm_config = _resolve_llm_config(args)
        design_requirement = _load_design_requirement(args)
        log_block("self_evolving", "DESIGN_REQUIREMENT", design_requirement)
        summary = generate_assets_from_design_requirement(
            design_requirement=design_requirement,
            llm_config=llm_config,
            prompt_path=Path(args.prompt_path),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Guide path: {summary['guide_path']}")
    print(f"Tool paths: {summary['tool_paths']}")
    print(f"Guide name: {summary['guide_name']}")
    print(f"New tool names: {summary['tool_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
