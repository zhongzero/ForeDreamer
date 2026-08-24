#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from experience_bank import (
    iso_now,
)
from prediction.runtime import build_dataset_runner, run_dataset_events_sync
from SelfEvolving.evolve_summary import truncate_text
from SelfEvolving.evolve_storage import load_validation_results, upsert_validation_result
from SelfEvolving.evolve_validation import (
    build_validation_key,
    can_reuse_validation_entry,
    compute_validation_result,
    get_validation_result_entry_for_experience,
    is_validation_entry_valid,
)
from SelfEvolving.generate_memguide_and_memtool import LLMConfig
from utils.logger import log_info
from utils.timing_registry import timed_block


SRC_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = SRC_DIR / "SelfEvolving" / "prompt"
DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH = PROMPT_DIR / "experience_evolution_prompt.md"
DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH_CN = PROMPT_DIR / "experience_evolution_prompt-cn.md"

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
CURRENT_EXPERIENCE_BANK_PLACEHOLDER = "{{CURRENT_EXPERIENCE_BANK}}"
ROLLOUT_SUMMARY_PLACEHOLDER = "{{ROLLOUT_SUMMARY}}"
RUN_RESULT_PLACEHOLDER = "{{RUN_RESULT}}"
SELECTED_GUIDE_CONTEXT_PLACEHOLDER = "{{SELECTED_GUIDE_CONTEXT}}"
MAX_SUGGESTIONS_PLACEHOLDER = "{{MAX_SUGGESTIONS}}"


def extract_json_object_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        raise ValueError("The experience evolution model returned an empty response.")

    fence_match = JSON_FENCE_RE.match(normalized)
    if fence_match is not None:
        normalized = fence_match.group(1).strip()

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The experience evolution model response did not contain a JSON object.")
    return normalized[start : end + 1]


def resolve_experience_evolution_prompt_path(generation_prompt_path: Path) -> Path:
    if generation_prompt_path.name.endswith("-cn.md"):
        return DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH_CN
    return DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH


def _render_section(title: str, body: str, max_chars: int) -> str:
    return f"## {title}\n{truncate_text(body, max_chars)}"


def _build_prompt_experience_bank_payload(experience_bank_payload: dict[str, Any]) -> dict[str, Any]:
    experiences = list(experience_bank_payload.get("experiences", []) or [])
    simplified_experiences: list[dict[str, Any]] = []
    for entry in experiences:
        if not isinstance(entry, dict):
            continue
        experience_id = str(entry.get("experience_id", "") or "").strip()
        text = str(entry.get("text", "") or "").strip()
        if not experience_id or not text:
            continue
        simplified_experiences.append(
            {
                "experience_id": experience_id,
                "text": text,
            }
        )

    return {
        "experience_count": len(simplified_experiences),
        "experiences": simplified_experiences,
    }


def build_experience_evolution_summary(
    *,
    experience_bank_payload: dict[str, Any],
    selected_guide_file: str,
    selected_guide_name: str,
    selected_guide_validation: dict[str, Any] | None,
    run_result: dict[str, Any],
    rollout_summary: str,
    summary_max_chars: int,
) -> tuple[str, dict[str, Any]]:
    prompt_experience_bank_payload = _build_prompt_experience_bank_payload(experience_bank_payload)
    bank_section = _render_section(
        "Current Experience Bank",
        json.dumps(prompt_experience_bank_payload, ensure_ascii=False, indent=2),
        max(1200, summary_max_chars // 4),
    )
    guide_section = _render_section(
        "Current Best Guide",
        json.dumps(
            {
                "guide_file": selected_guide_file,
                "guide_name": selected_guide_name,
                "validation": selected_guide_validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        max(800, summary_max_chars // 8),
    )
    run_result_section = _render_section(
        "Run Result",
        json.dumps(run_result, ensure_ascii=False, indent=2),
        max(1000, summary_max_chars // 5),
    )
    rollout_section = _render_section(
        "Rollout Summary",
        rollout_summary,
        max(2000, summary_max_chars // 2),
    )
    combined = "\n\n".join([bank_section, guide_section, run_result_section, rollout_section])
    final_summary = truncate_text(combined, summary_max_chars)
    breakdown = {
        "total_chars": len(final_summary),
        "pre_truncation_total_chars": len(combined),
        "bank_section_chars": len(bank_section),
        "guide_section_chars": len(guide_section),
        "run_result_section_chars": len(run_result_section),
        "rollout_section_chars": len(rollout_section),
    }
    return final_summary, breakdown


def build_experience_evolution_prompt_and_lengths(
    *,
    prompt_path: Path,
    experience_bank_payload: dict[str, Any],
    selected_guide_file: str,
    selected_guide_name: str,
    selected_guide_validation: dict[str, Any] | None,
    run_result: dict[str, Any],
    rollout_summary: str,
    max_suggestions: int,
) -> tuple[str, dict[str, Any]]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    for placeholder in (
        CURRENT_EXPERIENCE_BANK_PLACEHOLDER,
        ROLLOUT_SUMMARY_PLACEHOLDER,
        RUN_RESULT_PLACEHOLDER,
        SELECTED_GUIDE_CONTEXT_PLACEHOLDER,
        MAX_SUGGESTIONS_PLACEHOLDER,
    ):
        if placeholder not in prompt_template:
            raise ValueError(f"Experience evolution prompt template is missing placeholder: {placeholder}")

    prompt_experience_bank_payload = _build_prompt_experience_bank_payload(experience_bank_payload)
    bank_json = json.dumps(prompt_experience_bank_payload, ensure_ascii=False, indent=2)
    selected_guide_context = json.dumps(
        {
            "guide_file": selected_guide_file,
            "guide_name": selected_guide_name,
            "validation": selected_guide_validation,
        },
        ensure_ascii=False,
        indent=2,
    )
    run_result_json = json.dumps(run_result, ensure_ascii=False, indent=2)

    prompt = (
        prompt_template
        .replace(CURRENT_EXPERIENCE_BANK_PLACEHOLDER, bank_json, 1)
        .replace(ROLLOUT_SUMMARY_PLACEHOLDER, rollout_summary, 1)
        .replace(RUN_RESULT_PLACEHOLDER, run_result_json, 1)
        .replace(SELECTED_GUIDE_CONTEXT_PLACEHOLDER, selected_guide_context, 1)
        .replace(MAX_SUGGESTIONS_PLACEHOLDER, str(max_suggestions), 1)
    )
    prompt_lengths = {
        "total_prompt_chars": len(prompt),
        "template_chars": len(prompt_template),
        "experience_bank_chars": len(bank_json),
        "selected_guide_context_chars": len(selected_guide_context),
        "run_result_chars": len(run_result_json),
        "rollout_summary_chars": len(rollout_summary),
    }
    return prompt, prompt_lengths


def parse_experience_evolution_response(
    response_text: str,
    *,
    max_suggestions: int,
    current_experience_ids: set[str],
) -> dict[str, Any]:
    payload = json.loads(extract_json_object_text(response_text))
    if not isinstance(payload, dict):
        raise ValueError("The experience evolution response must be a JSON object.")

    raw_suggestions = payload.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        raise ValueError("The experience evolution response must define suggestions as a list.")

    parsed_suggestions: list[dict[str, Any]] = []
    rejected_suggestions: list[dict[str, Any]] = []

    for index, raw_suggestion in enumerate(raw_suggestions, start=1):
        if not isinstance(raw_suggestion, dict):
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": "Suggestion must be a JSON object.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue

        try:
            priority = int(raw_suggestion.get("priority"))
        except (TypeError, ValueError):
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": "Suggestion priority must be an integer.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue

        operation = str(raw_suggestion.get("operation", "") or "").strip().lower()
        target_experience_id = str(raw_suggestion.get("target_experience_id", "") or "").strip() or None
        new_text = str(raw_suggestion.get("new_text", "") or "").strip() or None
        analysis = str(raw_suggestion.get("analysis", "") or "").strip()
        generality_assessment = str(raw_suggestion.get("generality_assessment", "") or "").strip()
        expected_benefit = str(raw_suggestion.get("expected_benefit", "") or "").strip()

        if operation not in {"add", "remove", "modify"}:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": f"Unsupported operation: {operation}",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue
        if not analysis:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": "Suggestion analysis must be non-empty.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue
        if not generality_assessment:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": "Suggestion generality_assessment must be non-empty.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue
        if not expected_benefit:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": "Suggestion expected_benefit must be non-empty.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue

        if operation in {"remove", "modify"} and not target_experience_id:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": f"Suggestion operation {operation} requires target_experience_id.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue
        if operation in {"remove", "modify"} and target_experience_id not in current_experience_ids:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": f"Target experience_id does not exist: {target_experience_id}",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue
        if operation in {"add", "modify"} and not new_text:
            rejected_suggestions.append(
                {
                    "index": index,
                    "reason": f"Suggestion operation {operation} requires non-empty new_text.",
                    "raw_suggestion": raw_suggestion,
                }
            )
            continue

        parsed_suggestions.append(
            {
                "priority": priority,
                "operation": operation,
                "target_experience_id": target_experience_id,
                "new_text": new_text,
                "analysis": analysis,
                "generality_assessment": generality_assessment,
                "expected_benefit": expected_benefit,
            }
        )

    parsed_suggestions.sort(key=lambda item: (item["priority"], item["operation"]))
    return {
        "suggestions": parsed_suggestions[:max(0, max_suggestions)],
        "rejected_suggestions": rejected_suggestions,
    }


def build_experience_validation_key(
    *,
    dataset_type: str,
    val_data_path: Path,
    llm_config: LLMConfig,
    base_strategy_config: Any,
    use_sources: bool,
    use_market_data: bool,
    guide_file: str,
) -> tuple[str, dict[str, Any]]:
    del guide_file
    return build_validation_key(
        dataset_type=dataset_type,
        val_data_path=val_data_path,
        llm_config=llm_config,
        base_strategy_config=base_strategy_config,
        use_sources=use_sources,
        use_market_data=use_market_data,
    )


def _build_validation_run_label(base_run_label: str, guide_file: str, bank_hash: str) -> str:
    guide_stem = Path(guide_file).stem
    short_hash = str(bank_hash or "")[:12]
    timestamp = datetime_now_compact()
    return (
        f"{str(base_run_label or '').strip()}-experience-validation-{guide_stem}-{short_hash}-{timestamp}"
    )


def datetime_now_compact() -> str:
    return datetime_stamp(iso_now())


def datetime_stamp(value: str) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))[:14] or "unknown"


def validate_experience_bank_candidate(
    *,
    dataset_type: str,
    bank_payload: dict[str, Any],
    guide_file: str,
    guide_name: str,
    val_records: list[dict[str, Any]],
    llm_config: LLMConfig,
    base_strategy_config: Any,
    use_sources: bool,
    use_market_data: bool,
    val_data_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    validation_config = replace(
        base_strategy_config,
        mem_guide=guide_file,
        save_rollout=False,
        experience_entries=tuple(bank_payload.get("experiences", [])),
        experience_bank_hash=str(bank_payload.get("bank_hash", "") or "").strip() or None,
        factual_memory_run_label=_build_validation_run_label(
            str(getattr(base_strategy_config, "factual_memory_run_label", "") or ""),
            guide_file,
            str(bank_payload.get("bank_hash", "") or ""),
        ),
    )
    validation_key, validation_key_payload = build_experience_validation_key(
        dataset_type=dataset_type,
        val_data_path=val_data_path,
        llm_config=llm_config,
        base_strategy_config=validation_config,
        use_sources=use_sources,
        use_market_data=use_market_data,
        guide_file=guide_file,
    )
    existing_results = load_validation_results()
    existing_entry = get_validation_result_entry_for_experience(
        existing_results,
        guide_file=guide_file,
        validation_key=validation_key,
        experience_bank_hash=str(bank_payload.get("bank_hash", "") or ""),
    )
    if can_reuse_validation_entry(existing_entry):
        return existing_entry, validation_key, validation_key_payload

    runner = build_dataset_runner(
        dataset_type,
        model=llm_config.model,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )
    try:
        with timed_block(
            "experience.validation.rollout_batch",
            "run_dataset_events_sync",
            kind="validation",
            metadata={
                "guide_file": guide_file,
                "dataset_type": dataset_type,
                "num_samples": len(val_records),
                "experience_bank_hash": bank_payload.get("bank_hash"),
            },
        ):
            results = run_dataset_events_sync(
                dataset_type,
                val_records,
                runner,
                strategy="web_search_loop",
                strategy_config=validation_config,
                use_sources=use_sources,
                use_market_data=use_market_data,
            )
        with timed_block(
            "experience.validation.metrics",
            "compute_validation_result",
            kind="validation",
            metadata={
                "guide_file": guide_file,
                "dataset_type": dataset_type,
                "num_samples": len(val_records),
                "experience_bank_hash": bank_payload.get("bank_hash"),
            },
        ):
            result_entry = compute_validation_result(
                dataset_type=dataset_type,
                records=val_records,
                results=results,
                model_name=llm_config.model,
            )
    except Exception as exc:
        ranking_metric_name = "avg_brier" if dataset_type == "prophet_arena" else "avg_accuracy"
        result_entry = {
            "status": "error",
            "validated_at": iso_now(),
            "num_samples": len(val_records),
            "ranking_metric_name": ranking_metric_name,
            "ranking_metric_value": None,
            "details": {
                ranking_metric_name: None,
                "success_count": 0,
                "error_count": len(val_records),
                "is_valid": False,
                "validity_status": "invalid",
            },
            "samples": [],
            "error": {
                "message": str(exc),
            },
        }
    result_entry["experience_bank"] = {
        "version_id": str(bank_payload.get("version_id", "") or "").strip() or None,
        "bank_hash": str(bank_payload.get("bank_hash", "") or "").strip() or None,
        "experience_count": len(list(bank_payload.get("experiences", []) or [])),
        "experience_ids": [
            str(entry.get("experience_id", "") or "").strip()
            for entry in list(bank_payload.get("experiences", []) or [])
            if isinstance(entry, dict) and str(entry.get("experience_id", "") or "").strip()
        ],
    }
    upsert_validation_result(
        guide_file=guide_file,
        guide_name=guide_name,
        validation_key=validation_key,
        validation_key_payload=validation_key_payload,
        result_entry=result_entry,
    )
    log_info(
        "experience_evolving",
        (
            f"Experience bank validation | guide={guide_file} | "
            f"bank_hash={bank_payload.get('bank_hash')} | "
            f"status={result_entry.get('status')}"
        ),
    )
    return result_entry, validation_key, validation_key_payload


def is_candidate_validation_better(
    *,
    dataset_type: str,
    baseline_entry: dict[str, Any],
    candidate_entry: dict[str, Any],
) -> bool:
    if not is_validation_entry_valid(candidate_entry):
        return False
    if str(baseline_entry.get("status", "") or "").strip() != "success":
        return True

    baseline_value = baseline_entry.get("ranking_metric_value")
    candidate_value = candidate_entry.get("ranking_metric_value")
    if not isinstance(candidate_value, (int, float)):
        return False
    if not isinstance(baseline_value, (int, float)):
        return True

    if dataset_type == "prophet_arena":
        return float(candidate_value) < float(baseline_value)
    return float(candidate_value) > float(baseline_value)


__all__ = [
    "DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH",
    "DEFAULT_EXPERIENCE_EVOLUTION_PROMPT_PATH_CN",
    "_build_prompt_experience_bank_payload",
    "build_experience_evolution_prompt_and_lengths",
    "build_experience_evolution_summary",
    "build_experience_validation_key",
    "is_candidate_validation_better",
    "parse_experience_evolution_response",
    "resolve_experience_evolution_prompt_path",
    "validate_experience_bank_candidate",
]
