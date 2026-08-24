#!/usr/bin/env python3

from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from MemTool.registry import get_tool_specs as get_memtool_specs
from core.rollout import record_subagent_interaction, register_rollout
from utils.api_cache import (
    build_cached_completion,
    build_llm_request_identity,
    load_llm_cache_entry,
    record_llm_actual_call,
    record_llm_cache_hit,
    serialize_message_payload,
    store_llm_cache_entry,
)
from utils.logger import format_json, log_block, log_info
from utils.timing_registry import timed_block
from utils.generated_paths import MEMTOOL_DIR_ENV, resolve_memguide_dir, resolve_memtool_dir


TOOL_NAME: Final[str] = "privacy_data_process"
IS_PUBLIC_TOOL: Final[bool] = False
TOOL_SPEC: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Private tool that reads raw_data.json files and writes final_data.txt outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directories that contain raw_data.json files.",
                }
            },
            "required": ["item_dirs"],
            "additionalProperties": False,
        },
    },
}
DEFAULT_INPUT_FILENAME: Final[str] = "raw_data.json"
DEFAULT_OUTPUT_FILENAME: Final[str] = "final_data.txt"
DEFAULT_GUIDE_FILENAME: Final[str] = "guide_initial.json"
DEFAULT_LLM_TIMEOUT: Final[int] = 800
MEMTOOL_SANDBOX_TIMEOUT_SECONDS: Final[int] = 30
BWRAP_BINARY: Final[str] = "bwrap"


@dataclass(frozen=True)
class _GuideDefinition:
    guide_name: str
    prompt: str
    tool_names: list[str]


def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:

    raw_item_dirs = arguments.get("item_dirs")
    if not isinstance(raw_item_dirs, list) or not raw_item_dirs:
        raise ValueError(f"{TOOL_NAME} requires a non-empty item_dirs list")

    item_dirs: list[str] = []
    for item_dir in raw_item_dirs:
        normalized = str(item_dir).strip()
        if not normalized:
            raise ValueError(f"{TOOL_NAME} requires every item_dirs entry to be a non-empty string")
        item_dirs.append(normalized)

    llm_api_key = str(getattr(config, "llm_api_key", "") or "").strip()
    if not llm_api_key:
        raise ValueError(f"{TOOL_NAME} requires a non-empty llm_api_key")
    llm_base_url = str(getattr(config, "llm_base_url", "") or "").strip()
    if not llm_base_url:
        raise ValueError(f"{TOOL_NAME} requires a non-empty llm_base_url")
    llm_model = str(getattr(config, "llm_model", "") or "").strip()
    if not llm_model:
        raise ValueError(f"{TOOL_NAME} requires a non-empty llm_model")
    mem_guide = str(getattr(config, "mem_guide", "") or "").strip()
    subagent_max_turns = int(getattr(config, "subagent_max_turns", 10))
    if subagent_max_turns < 1:
        raise ValueError(f"{TOOL_NAME} requires subagent_max_turns >= 1")
    return {
        "item_dirs": item_dirs,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "subagent_max_turns": subagent_max_turns,
        "input_filename": DEFAULT_INPUT_FILENAME,
        "output_filename": DEFAULT_OUTPUT_FILENAME,
        "guide_path": str(_resolve_guide_path(mem_guide)),
        "parent_task_id": str(getattr(runtime_context, "task_id", "") or "").strip(),
        "problem_statement": str(getattr(runtime_context, "problem_statement", "") or "").strip(),
        "task_requirements": str(getattr(runtime_context, "task_requirements", "") or "").strip(),
    }


def run_tool(
    *,
    item_dirs: list[str],
    llm_api_key: str,
    llm_base_url: str,
    llm_model: str,
    subagent_max_turns: int,
    input_filename: str,
    output_filename: str,
    guide_path: str,
    parent_task_id: str,
    problem_statement: str,
    task_requirements: str,
) -> str:
    register_rollout(
        parent_task_id,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
    )
    normalized_item_dirs: list[str] = []
    processed_contents: list[str] = []

    guide = _load_guide(Path(guide_path))
    client = OpenAI(
        base_url=llm_base_url,
        api_key=llm_api_key,
        timeout=DEFAULT_LLM_TIMEOUT,
    )

    for raw_item_dir in item_dirs:
        item_dir = Path(raw_item_dir)
        if not item_dir.is_dir():
            raise ValueError(f"{TOOL_NAME} could not find item_dir: {item_dir}")

        log_info("privacy_data_process", f"Subagent start | item_dir={item_dir}")
        _run_item_subagent(
            client=client,
            base_url=llm_base_url,
            model=llm_model,
            guide=guide,
            max_turns=subagent_max_turns,
            item_dir=item_dir,
            input_filename=input_filename,
            output_filename=output_filename,
            parent_task_id=parent_task_id,
        )

        final_data_path = item_dir / output_filename
        if not final_data_path.exists():
            raise ValueError(f"{TOOL_NAME} expected {output_filename} under {item_dir}")

        normalized_item_dirs.append(str(item_dir))
        processed_contents.append(final_data_path.read_text(encoding="utf-8"))
        log_info("privacy_data_process", f"Subagent end | item_dir={item_dir}")

    return json.dumps(
        {
            "item_dirs": normalized_item_dirs,
            "processed_contents": processed_contents,
        },
        ensure_ascii=False,
    )


def _run_item_subagent(
    *,
    client: OpenAI,
    base_url: str,
    model: str,
    guide: _GuideDefinition,
    max_turns: int,
    item_dir: Path,
    input_filename: str,
    output_filename: str,
    parent_task_id: str,
) -> None:
    strategy_tools = get_memtool_specs(guide.tool_names)
    final_data_path = item_dir / output_filename
    if final_data_path.exists():
        final_data_path.unlink()

    runtime_context = SimpleNamespace(
        item_dir=str(item_dir),
        input_filename=input_filename,
        output_filename=output_filename,
    )
    config = SimpleNamespace()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": guide.prompt},
        {
            "role": "user",
            "content": (
                f"Current item_dir: {item_dir}\n"
                f"Input filename: {input_filename}\n"
                f"Output filename: {output_filename}\n"
                f"Create {output_filename} for this item_dir."
            ),
        },
    ]
    logged_subagent_tools = False

    for turn in range(1, max_turns + 1):
        log_info(
            "privacy_data_process",
            f"Subagent turn start | item_dir={item_dir} | turn={turn}/{max_turns}",
        )
        messages_for_turn = deepcopy(messages)
        tools_for_turn = deepcopy(strategy_tools)
        completion = _create_subagent_completion(
            client=client,
            base_url=base_url,
            model=model,
            messages=messages_for_turn,
            tools=tools_for_turn,
            log_tools=not logged_subagent_tools,
        )
        logged_subagent_tools = True
        message = completion.choices[0].message
        text = _extract_message_text(message)
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        assistant_message = _serialize_assistant_message(message, text)
        if len(assistant_message) > 1:
            messages.append(assistant_message)

        if tool_calls:
            log_block(
                "privacy_data_process",
                "Subagent tool calls",
                format_json(
                    [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ]
                ),
            )
            for tool_call in tool_calls:
                tool_result = _execute_memtool_call(
                    tool_call=tool_call,
                    allowed_tool_names=guide.tool_names,
                    config=config,
                    runtime_context=runtime_context,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
            record_subagent_interaction(
                task_id=parent_task_id,
                item_dir=str(item_dir),
                guide_name=guide.guide_name,
                messages=messages,
                tools=strategy_tools,
            )
            if final_data_path.exists():
                return
            continue

        record_subagent_interaction(
            task_id=parent_task_id,
            item_dir=str(item_dir),
            guide_name=guide.guide_name,
            messages=messages,
            tools=strategy_tools,
        )
        if final_data_path.exists():
            return

        if turn < max_turns:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{output_filename} has not been created yet. "
                        "Use the available tool to finish the task."
                    ),
                }
            )

    if final_data_path.exists():
        return
    raise ValueError(f"{TOOL_NAME} subagent did not create {output_filename} under {item_dir}")


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def _completion_with_backoff(client: OpenAI, **kwargs):
    record_llm_actual_call()
    return client.chat.completions.create(**kwargs)


def _create_subagent_completion(
    *,
    client: OpenAI,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    log_tools: bool,
):
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    log_block("privacy_data_process", "Subagent messages", format_json(messages))
    if log_tools:
        log_block("privacy_data_process", "Subagent tools", format_json(tools))
    request_identity = build_llm_request_identity(
        base_url=base_url,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    cache_key, cache_entry = load_llm_cache_entry(request_identity)
    if cache_entry is not None:
        record_llm_cache_hit()
        log_info("privacy_data_process", f"Subagent LLM cache hit | cache_key={cache_key}")
        return build_cached_completion(cache_entry.get("response", {}).get("message", {}))
    log_info("privacy_data_process", f"Subagent LLM cache miss | cache_key={cache_key}")
    with timed_block(
        "llm.data_process_subagent",
        "subagent_completion",
        kind="llm",
        metadata={
            "model": model,
            "tool_count": len(tools),
        },
    ):
        completion = _completion_with_backoff(client, **payload)
    message = completion.choices[0].message
    response_text = _extract_message_text(message)
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if response_text or tool_calls:
        store_llm_cache_entry(
            cache_key=cache_key,
            request_identity=request_identity,
            message_payload=serialize_message_payload(
                content=response_text,
                tool_calls=tool_calls,
            ),
        )
    log_block(
        "privacy_data_process",
        "Subagent response text",
        response_text,
    )
    return completion


def _execute_memtool_call(*, tool_call, allowed_tool_names: list[str], config: Any, runtime_context: Any) -> str:
    function = getattr(tool_call, "function", None)
    if function is None:
        raise ValueError("MemTool call is missing function data")
    if function.name not in allowed_tool_names:
        raise ValueError(f"MemTool call is not allowed by the current guide: {function.name}")

    raw_arguments = getattr(function, "arguments", "") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MemTool arguments: {raw_arguments}") from exc

    item_dir = Path(str(getattr(runtime_context, "item_dir", "") or "")).resolve()
    if not item_dir.is_dir():
        raise ValueError(f"{TOOL_NAME} sandbox requires an existing item_dir: {item_dir}")

    sandbox_payload = {
        "tool_name": function.name,
        "arguments": arguments,
        "config": _namespace_to_dict(config),
        "runtime_context": _namespace_to_dict(runtime_context),
    }
    command = _build_memtool_sandbox_command(item_dir=item_dir)
    log_block("privacy_data_process", "Sandbox command", format_json(command))

    try:
        with timed_block(
            "memtool.sandbox",
            function.name,
            kind="memtool",
            metadata={
                "timeout_seconds": MEMTOOL_SANDBOX_TIMEOUT_SECONDS,
                "item_dir": str(item_dir),
            },
        ):
            completed = subprocess.run(
                command,
                input=json.dumps(sandbox_payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=MEMTOOL_SANDBOX_TIMEOUT_SECONDS,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"{TOOL_NAME} sandbox timed out after {MEMTOOL_SANDBOX_TIMEOUT_SECONDS}s for {function.name}"
        ) from exc

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "unknown sandbox error"
        raise ValueError(
            f"{TOOL_NAME} sandbox failed for {function.name} with exit code {completed.returncode}: {detail}"
        )
    if stderr:
        log_block("privacy_data_process", "Sandbox stderr", stderr)
    return stdout


def _load_guide(path: Path) -> _GuideDefinition:
    if not path.exists():
        raise ValueError(f"{TOOL_NAME} could not find guide file: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{TOOL_NAME} guide file is invalid JSON: {path}") from exc

    guide_name = str(raw_data.get("guide_name", "") or "").strip()
    if not guide_name:
        raise ValueError(f"{TOOL_NAME} guide file requires a non-empty guide_name: {path}")
    prompt = str(raw_data.get("prompt", "") or "").strip()
    if not prompt:
        raise ValueError(f"{TOOL_NAME} guide file requires a non-empty prompt: {path}")

    raw_tool_names = raw_data.get("tool_names")
    if not isinstance(raw_tool_names, list) or not raw_tool_names:
        raise ValueError(f"{TOOL_NAME} guide file requires a non-empty tool_names list: {path}")
    tool_names = [str(tool_name).strip() for tool_name in raw_tool_names if str(tool_name).strip()]
    if len(tool_names) != len(raw_tool_names):
        raise ValueError(f"{TOOL_NAME} guide file has empty tool_names entries: {path}")

    get_memtool_specs(tool_names)
    return _GuideDefinition(
        guide_name=guide_name,
        prompt=prompt,
        tool_names=tool_names,
    )


def _extract_message_text(message) -> str:
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        part_type = getattr(item, "type", None)
        if part_type == "text":
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                parts.append(text_value)
            elif hasattr(text_value, "value"):
                parts.append(str(text_value.value))
    return "".join(parts).strip()


def _serialize_assistant_message(message, text: str) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {"role": "assistant"}
    if text:
        assistant_message["content"] = text

    tool_calls = []
    for tool_call in list(getattr(message, "tool_calls", None) or []):
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def _default_guide_path() -> Path:
    return resolve_memguide_dir() / DEFAULT_GUIDE_FILENAME


def _resolve_guide_path(raw_value: str) -> Path:
    normalized = str(raw_value or "").strip()
    if not normalized:
        return _default_guide_path()

    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    memguide_dir = resolve_memguide_dir()
    if candidate.suffix == ".json":
        return memguide_dir / candidate.name

    return memguide_dir / f"{candidate.name}.json"


def _namespace_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return {key: getattr(value, key) for key in vars(value)}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _build_memtool_sandbox_command(*, item_dir: Path) -> list[str]:
    bwrap_path = shutil.which(BWRAP_BINARY)
    if not bwrap_path:
        raise ValueError(f"{TOOL_NAME} requires {BWRAP_BINARY} to sandbox MemTool calls")

    repo_src = Path(__file__).resolve().parent.parent
    repo_root = repo_src.parent
    python_executable = Path(sys.executable).resolve()
    memtool_dir = resolve_memtool_dir()
    python_prefixes = _python_runtime_prefixes(python_executable)
    command: list[str] = [
        bwrap_path,
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        f"{python_executable.parent}:/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        str(repo_src),
        "--setenv",
        MEMTOOL_DIR_ENV,
        str(memtool_dir),
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "LANG",
        os.environ.get("LANG", "C.UTF-8"),
        "--setenv",
        "LC_ALL",
        os.environ.get("LC_ALL", "C.UTF-8"),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]

    for system_path in _system_readonly_mounts():
        command.extend(["--ro-bind", str(system_path), str(system_path)])
    for prefix in python_prefixes:
        command.extend(["--ro-bind", str(prefix), str(prefix)])

    command.extend(["--ro-bind", str(repo_root), str(repo_root)])
    if not _is_within_root(memtool_dir, repo_root):
        command.extend(["--ro-bind", str(memtool_dir), str(memtool_dir)])
    command.extend(["--bind", str(item_dir), str(item_dir)])
    command.extend(["--chdir", str(item_dir)])
    command.extend([str(python_executable), "-m", "MemTool.sandbox_runner"])
    return command


def _system_readonly_mounts() -> list[Path]:
    mounts: list[Path] = []
    for candidate in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        path = Path(candidate)
        if path.exists():
            mounts.append(path)
    return mounts


def _python_runtime_prefixes(python_executable: Path) -> list[Path]:
    candidates = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        python_executable.parent.resolve(),
        python_executable.parent.parent.resolve(),
    }
    prefixes = [
        candidate
        for candidate in sorted(candidates, key=lambda path: len(str(path)))
        if candidate.exists()
    ]
    # Drop nested prefixes so we do not add redundant ro-bind mounts.
    normalized: list[Path] = []
    for candidate in prefixes:
        if any(candidate == existing or candidate.is_relative_to(existing) for existing in normalized):
            continue
        normalized.append(candidate)
    return normalized


__all__ = ["TOOL_NAME", "IS_PUBLIC_TOOL", "TOOL_SPEC", "build_runner_kwargs", "run_tool"]
