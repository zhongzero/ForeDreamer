#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from typing import Any

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
from SelfEvolving.generate_memguide_and_memtool import LLMConfig


DEFAULT_TIMEOUT_SECONDS = 800
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
CRITIC_GUIDE_JSON_PLACEHOLDER = "{{GUIDE_JSON}}"
CRITIC_TOOL_SOURCE_PLACEHOLDER = "{{TOOL_SOURCE_CODE}}"
CRITIC_ROLLOUT_SUMMARY_PLACEHOLDER = "{{ROLLOUT_SUMMARY}}"


def extract_message_text(message: Any) -> str:
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


def extract_json_object_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        raise ValueError("The critique model returned an empty response.")

    fence_match = JSON_FENCE_RE.match(normalized)
    if fence_match is not None:
        normalized = fence_match.group(1).strip()

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The critique model response did not contain a JSON object.")
    return normalized[start : end + 1]


def parse_critique_response(response_text: str) -> dict[str, Any]:
    critique = json.loads(extract_json_object_text(response_text))
    if not isinstance(critique, dict):
        raise ValueError("The critique model response must be a JSON object.")

    should_evolve = critique.get("should_evolve")
    if not isinstance(should_evolve, bool):
        raise ValueError("The critique model response must define boolean should_evolve.")

    analysis = str(critique.get("analysis", "") or "").strip()
    design_requirement = str(critique.get("design_requirement", "") or "").strip()
    if not analysis:
        raise ValueError("The critique model response must define a non-empty analysis.")
    if should_evolve and not design_requirement:
        raise ValueError("design_requirement must be non-empty when should_evolve is true.")
    if not should_evolve:
        design_requirement = ""

    return {
        "should_evolve": should_evolve,
        "analysis": analysis,
        "design_requirement": design_requirement,
    }


def request_llm_response(prompt: str, llm_config: LLMConfig, *, log_tag: str) -> str:
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
        log_tag,
        f"LLM request start | model={llm_config.model} | base_url={llm_config.base_url}",
    )
    cache_key, cache_entry = load_llm_cache_entry(request_identity)
    if cache_entry is not None:
        record_llm_cache_hit()
        log_info(log_tag, f"LLM cache hit | cache_key={cache_key}")
        completion = build_cached_completion(cache_entry.get("response", {}).get("message", {}))
        message = completion.choices[0].message
        text = extract_message_text(message).strip()
        log_block(log_tag, "LLM response", text)
        if not text:
            raise ValueError("The LLM returned an empty response.")
        return text
    log_info(log_tag, f"LLM cache miss | cache_key={cache_key}")
    client = OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    with timed_block(
        "llm.self_evolving_critic",
        "critique_generation",
        kind="llm",
        metadata={
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "log_tag": log_tag,
        },
    ):
        record_llm_actual_call()
        completion = client.chat.completions.create(
            model=llm_config.model,
            messages=messages,
        )
    message = completion.choices[0].message
    text = extract_message_text(message).strip()
    log_block(log_tag, "LLM response", text)
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


def build_critic_prompt_and_lengths(
    *,
    critic_prompt_path,
    guide_object: dict[str, Any],
    tool_source_bundle: str,
    rollout_summary: str,
    rollout_summary_breakdown: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    guide_json_text = json.dumps(guide_object, ensure_ascii=False, indent=2)
    prompt_template = critic_prompt_path.read_text(encoding="utf-8")
    for placeholder in (
        CRITIC_GUIDE_JSON_PLACEHOLDER,
        CRITIC_TOOL_SOURCE_PLACEHOLDER,
        CRITIC_ROLLOUT_SUMMARY_PLACEHOLDER,
    ):
        if placeholder not in prompt_template:
            raise ValueError(f"Critic prompt template is missing placeholder: {placeholder}")

    prompt = (
        prompt_template
        .replace(CRITIC_GUIDE_JSON_PLACEHOLDER, guide_json_text, 1)
        .replace(CRITIC_TOOL_SOURCE_PLACEHOLDER, tool_source_bundle, 1)
        .replace(CRITIC_ROLLOUT_SUMMARY_PLACEHOLDER, rollout_summary, 1)
    )
    prompt_lengths = {
        "total_prompt_chars": len(prompt),
        "template_chars": len(prompt_template),
        "guide_json_chars": len(guide_json_text),
        "tool_source_code_chars": len(tool_source_bundle),
        "rollout_summary_chars": len(rollout_summary),
        "rollout_summary_char_breakdown": rollout_summary_breakdown,
    }
    return prompt, prompt_lengths
