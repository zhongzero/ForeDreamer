#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from prediction.metrics import compute_prophet_arena_evaluation
from prediction.runtime import build_dataset_runner, run_dataset_events_sync
from utils.logger import log_info
from utils.search_provider import normalize_search_provider
from utils.serialization import parse_serialized_data
from utils.timing_registry import timed_block
from SelfEvolving.evolve_storage import (
    load_validation_results,
    upsert_validation_result,
)
from SelfEvolving.generate_memguide_and_memtool import LLMConfig


UNIFORM_RANDOM_SELECTION = "uniform_random"
ZIPF_BY_VALIDATION_RANK_SELECTION = "zipf_by_validation_rank"
NO_ELIGIBLE_BEST_GUIDE_SELECTION_ERROR = (
    "No eligible guide nodes are available for deterministic best-guide selection."
)
NO_EXPERIENCE_VALIDATION_CONTEXT = "__no_experience_bank__"


class NoEligibleBestGuideSelectionError(ValueError):
    """Raised when no validation-qualified guide is available for deterministic selection."""


def serialize_validation_key(validation_key_payload: dict[str, Any]) -> str:
    return json.dumps(validation_key_payload, ensure_ascii=False, sort_keys=True)


def build_validation_key(
    *,
    dataset_type: str,
    val_data_path: Path,
    llm_config: LLMConfig,
    base_strategy_config: Any,
    use_sources: bool,
    use_market_data: bool,
) -> tuple[str, dict[str, Any]]:
    normalized_search_provider = normalize_search_provider(
        getattr(base_strategy_config, "search_provider", "tavily")
    )
    key_payload = {
        "dataset_type": dataset_type,
        "val_data_path": str(val_data_path.resolve()),
        "model": llm_config.model,
        "strategy": "web_search_loop",
        "max_turns": int(getattr(base_strategy_config, "max_turns", 0)),
        "subagent_max_turns": int(getattr(base_strategy_config, "subagent_max_turns", 0)),
        "search_max_results": int(getattr(base_strategy_config, "search_max_results", 0)),
        "search_max_chars_per_result": int(getattr(base_strategy_config, "search_max_chars_per_result", 0)),
        "search_max_total_chars": int(getattr(base_strategy_config, "search_max_total_chars", 0)),
        "use_tavilty_raw_context": bool(getattr(base_strategy_config, "use_tavilty_raw_context", False)),
        "use_market_data_in_prophet_arena": bool(use_market_data),
        "use_source_in_prophet_arena": bool(use_sources),
    }
    if normalized_search_provider != "tavily":
        key_payload["search_provider"] = normalized_search_provider
    return serialize_validation_key(key_payload), key_payload


def get_validation_result_entry(
    validation_results: dict[str, Any],
    *,
    guide_file: str,
    validation_key: str,
) -> dict[str, Any] | None:
    guides = validation_results.get("guides", {})
    guide_bucket = guides.get(guide_file)
    if not isinstance(guide_bucket, dict):
        return None
    results = guide_bucket.get("results", {})
    if not isinstance(results, dict):
        return None
    entry = results.get(validation_key)
    if not isinstance(entry, dict):
        return None
    experience_results = _extract_validation_experience_results(entry, fallback_context_key=validation_key)
    if not experience_results:
        return None
    selected_context_key, selected_entry = _select_primary_validation_entry(experience_results)
    if selected_entry is None:
        return None
    selected_payload = dict(selected_entry)
    selected_payload["selected_experience_context_key"] = selected_context_key
    selected_payload["available_experience_validation_count"] = len(experience_results)
    return selected_payload


def get_validation_result_entry_for_experience(
    validation_results: dict[str, Any],
    *,
    guide_file: str,
    validation_key: str,
    experience_bank_hash: str | None,
) -> dict[str, Any] | None:
    guides = validation_results.get("guides", {})
    guide_bucket = guides.get(guide_file)
    if not isinstance(guide_bucket, dict):
        return None
    results = guide_bucket.get("results", {})
    if not isinstance(results, dict):
        return None
    entry = results.get(validation_key)
    if not isinstance(entry, dict):
        return None
    experience_results = _extract_validation_experience_results(entry, fallback_context_key=validation_key)
    if not experience_results:
        return None
    target_context_key = _validation_experience_context_key(
        bank_hash=str(experience_bank_hash or "").strip() or None,
        version_id=None,
        fallback_key=validation_key,
    )
    selected_entry = experience_results.get(target_context_key)
    if not isinstance(selected_entry, dict):
        return None
    payload = dict(selected_entry)
    payload["selected_experience_context_key"] = target_context_key
    payload["available_experience_validation_count"] = len(experience_results)
    return payload


def build_validation_run_label(base_run_label: str, guide_file: str) -> str:
    guide_stem = Path(guide_file).stem
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{str(base_run_label or '').strip()}-validation-{guide_stem}-{timestamp}"


def _validation_experience_context_key(
    *,
    bank_hash: str | None,
    version_id: str | None,
    fallback_key: str | None = None,
) -> str:
    normalized_hash = str(bank_hash or "").strip()
    if normalized_hash:
        return normalized_hash
    normalized_version_id = str(version_id or "").strip()
    if normalized_version_id:
        return f"version:{normalized_version_id}"
    normalized_fallback = str(fallback_key or "").strip()
    if normalized_fallback:
        return f"legacy:{normalized_fallback}"
    return NO_EXPERIENCE_VALIDATION_CONTEXT


def _strip_validation_result_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.pop("experience_results", None)
    normalized.pop("selected_experience_context_key", None)
    normalized.pop("available_experience_validation_count", None)
    return normalized


def _extract_validation_experience_results(
    entry: dict[str, Any],
    *,
    fallback_context_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    experience_results = entry.get("experience_results")
    if isinstance(experience_results, dict):
        normalized_results: dict[str, dict[str, Any]] = {}
        for context_key, experience_entry in experience_results.items():
            if not isinstance(experience_entry, dict):
                continue
            normalized_results[str(context_key)] = _strip_validation_result_entry(experience_entry)
        return normalized_results

    return {
        _validation_experience_context_key(
            bank_hash=(
                str(((entry.get("experience_bank") or {}).get("bank_hash", "")) or "").strip()
                if isinstance(entry.get("experience_bank"), dict)
                else str(entry.get("experience_bank_hash", "") or "").strip()
            )
            or None,
            version_id=(
                str(((entry.get("experience_bank") or {}).get("version_id", "")) or "").strip()
                if isinstance(entry.get("experience_bank"), dict)
                else str(entry.get("experience_bank_version_id", "") or "").strip()
            )
            or None,
            fallback_key=fallback_context_key,
        ): _strip_validation_result_entry(entry)
    }


def _dataset_type_from_validation_entry(entry: dict[str, Any]) -> str | None:
    payload = entry.get("validation_key_payload")
    if not isinstance(payload, dict):
        return None
    dataset_type = str(payload.get("dataset_type", "") or "").strip()
    return dataset_type or None


def _validation_entry_metric_value(entry: dict[str, Any]) -> float | None:
    value = entry.get("ranking_metric_value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _select_primary_validation_entry(
    experience_results: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if not experience_results:
        return NO_EXPERIENCE_VALIDATION_CONTEXT, None

    selected_context_key = ""
    selected_entry: dict[str, Any] | None = None
    for context_key, candidate_entry in sorted(experience_results.items()):
        if selected_entry is None:
            selected_context_key = context_key
            selected_entry = candidate_entry
            continue

        selected_is_valid = is_validation_entry_valid(selected_entry)
        candidate_is_valid = is_validation_entry_valid(candidate_entry)
        if candidate_is_valid and not selected_is_valid:
            selected_context_key = context_key
            selected_entry = candidate_entry
            continue
        if selected_is_valid and not candidate_is_valid:
            continue

        if candidate_is_valid and selected_is_valid:
            dataset_type = _dataset_type_from_validation_entry(candidate_entry) or _dataset_type_from_validation_entry(
                selected_entry
            )
            selected_metric = _validation_entry_metric_value(selected_entry)
            candidate_metric = _validation_entry_metric_value(candidate_entry)
            if (
                dataset_type == "prophet_arena"
                and candidate_metric is not None
                and selected_metric is not None
                and candidate_metric < selected_metric
            ):
                selected_context_key = context_key
                selected_entry = candidate_entry
                continue
            if (
                dataset_type == "futurex"
                and candidate_metric is not None
                and selected_metric is not None
                and candidate_metric > selected_metric
            ):
                selected_context_key = context_key
                selected_entry = candidate_entry
                continue

        selected_validated_at = str(selected_entry.get("validated_at", "") or "")
        candidate_validated_at = str(candidate_entry.get("validated_at", "") or "")
        if candidate_validated_at > selected_validated_at:
            selected_context_key = context_key
            selected_entry = candidate_entry

    return selected_context_key or NO_EXPERIENCE_VALIDATION_CONTEXT, selected_entry


def _best_effort_parse_serialized(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if not normalized:
        return None

    try:
        return parse_serialized_data(normalized, field_name)
    except Exception:
        return normalized


def _extract_validation_ground_truth(*, dataset_type: str, event_data: dict[str, Any], result: dict[str, Any]) -> Any:
    if dataset_type == "futurex":
        return _best_effort_parse_serialized(
            result.get("ground_truth", event_data.get("ground_truth")),
            "ground_truth",
        )
    return _best_effort_parse_serialized(event_data.get("market_outcome"), "market_outcome")


def _extract_validation_prediction_result(*, dataset_type: str, result: dict[str, Any]) -> Any:
    if dataset_type == "futurex":
        response_text = str(result.get("response_text", "") or "").strip() or None
        extracted_rationale = str(result.get("extracted_rationale", "") or "").strip() or None
        parsed_answer = _best_effort_parse_serialized(result.get("parsed_answer"), "parsed_answer")
        prediction_result = {
            "parsed_answer": parsed_answer,
            "extracted_rationale": extracted_rationale,
            "response_text": response_text,
        }
        if not any(value is not None for value in prediction_result.values()):
            return None
        return prediction_result

    parsed_prediction = _best_effort_parse_serialized(result.get("prediction"), "prediction")
    if parsed_prediction is not None:
        return parsed_prediction

    rationale = str(result.get("rationale", "") or "").strip() or None
    return {"rationale": rationale} if rationale is not None else None


def is_validation_entry_valid(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    if str(entry.get("status", "") or "").strip() != "success":
        return False

    details = entry.get("details")
    if isinstance(details, dict):
        is_valid = details.get("is_valid")
        if isinstance(is_valid, bool):
            return is_valid

        success_count = details.get("success_count")
        error_count = details.get("error_count")
        num_samples = entry.get("num_samples")
        if isinstance(success_count, int) and isinstance(error_count, int) and isinstance(num_samples, int):
            return error_count == 0 and success_count == num_samples

    return False


def _validation_entry_has_detailed_samples(entry: dict[str, Any]) -> bool:
    samples = entry.get("samples")
    if not isinstance(samples, list) or not samples:
        return False
    details = entry.get("details")
    if not isinstance(details, dict) or "is_valid" not in details:
        return False
    for sample in samples:
        if not isinstance(sample, dict):
            return False
        if "ground_truth" not in sample or "prediction_result" not in sample:
            return False
    return True


def can_reuse_validation_entry(entry: dict[str, Any] | None) -> bool:
    return (
        isinstance(entry, dict)
        and str(entry.get("status", "") or "").strip() == "success"
        and _validation_entry_has_detailed_samples(entry)
    )


def build_validation_experience_bank_metadata(base_strategy_config: Any) -> dict[str, Any]:
    experience_entries = list(getattr(base_strategy_config, "experience_entries", ()) or ())
    experience_ids = [
        str(entry.get("experience_id", "") or "").strip()
        for entry in experience_entries
        if isinstance(entry, dict) and str(entry.get("experience_id", "") or "").strip()
    ]
    return {
        "version_id": str(getattr(base_strategy_config, "experience_bank_version_id", "") or "").strip() or None,
        "bank_hash": str(getattr(base_strategy_config, "experience_bank_hash", "") or "").strip() or None,
        "experience_count": len(experience_entries),
        "experience_ids": experience_ids,
    }


def _build_validation_sample_record(
    *,
    dataset_type: str,
    event_data: dict[str, Any],
    result: dict[str, Any],
    status: str,
    error_message: str | None,
) -> dict[str, Any]:
    if dataset_type == "futurex":
        sample_identifier = str(event_data.get("id", "") or "")
        sample_title = str(event_data.get("title", "") or "")
    else:
        sample_identifier = str(event_data.get("submission_id", "") or "")
        sample_title = str(event_data.get("title", "") or "")

    return {
        "sample_identifier": sample_identifier,
        "sample_title": sample_title,
        "status": str(status or "").strip() or "unknown",
        "error_message": str(error_message or "").strip() or None,
        "ground_truth": _extract_validation_ground_truth(
            dataset_type=dataset_type,
            event_data=event_data,
            result=result,
        ),
        "prediction_result": _extract_validation_prediction_result(
            dataset_type=dataset_type,
            result=result,
        ),
    }


def compute_validation_result(
    *,
    dataset_type: str,
    records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    if len(records) != len(results):
        raise ValueError(
            f"Validation returned mismatched result count: expected {len(records)}, got {len(results)}"
        )

    total_count = len(records)
    if total_count == 0:
        raise ValueError("Validation dataset is empty.")
    success_count = 0
    error_count = 0
    validation_samples: list[dict[str, Any]] = []

    if dataset_type == "futurex":
        correct_count = 0
        for event_data, result in zip(records, results):
            if result.get("status") == "success":
                success_count += 1
                if bool(result.get("exact_match", False)):
                    correct_count += 1
                validation_samples.append(
                    _build_validation_sample_record(
                        dataset_type=dataset_type,
                        event_data=event_data,
                        result=result,
                        status="success",
                        error_message=result.get("error_message"),
                    )
                )
            else:
                error_count += 1
                validation_samples.append(
                    _build_validation_sample_record(
                        dataset_type=dataset_type,
                        event_data=event_data,
                        result=result,
                        status=str(result.get("status", "") or "error"),
                        error_message=result.get("error_message"),
                    )
                )
        avg_accuracy = correct_count / total_count
        is_valid = error_count == 0 and success_count == total_count
        return {
            "status": "success",
            "validated_at": datetime.now().isoformat(timespec="seconds"),
            "num_samples": total_count,
            "ranking_metric_name": "avg_accuracy",
            "ranking_metric_value": avg_accuracy,
            "details": {
                "avg_accuracy": avg_accuracy,
                "success_count": success_count,
                "error_count": error_count,
                "is_valid": is_valid,
                "validity_status": "valid" if is_valid else "invalid",
            },
            "samples": validation_samples,
            "error": None,
        }

    total_brier = 0.0
    total_average_return = 0.0
    for event_data, result in zip(records, results):
        if result.get("status") == "success":
            try:
                prediction = json.loads(str(result.get("prediction", "") or ""))
                evaluation = compute_prophet_arena_evaluation(
                    event_data=event_data,
                    probabilities=list(prediction.get("probabilities", [])),
                    model_name=model_name,
                )
                total_brier += float(evaluation["brier"])
                total_average_return += float(evaluation["average_return"])
                success_count += 1
                validation_samples.append(
                    _build_validation_sample_record(
                        dataset_type=dataset_type,
                        event_data=event_data,
                        result=result,
                        status="success",
                        error_message=result.get("error_message"),
                    )
                )
            except Exception as exc:
                error_count += 1
                total_brier += 1.0
                total_average_return += 0.0
                validation_samples.append(
                    _build_validation_sample_record(
                        dataset_type=dataset_type,
                        event_data=event_data,
                        result=result,
                        status="error",
                        error_message=f"Validation metric computation failed: {exc}",
                    )
                )
        else:
            error_count += 1
            total_brier += 1.0
            total_average_return += 0.0
            validation_samples.append(
                _build_validation_sample_record(
                    dataset_type=dataset_type,
                    event_data=event_data,
                    result=result,
                    status=str(result.get("status", "") or "error"),
                    error_message=result.get("error_message"),
                )
            )

    avg_brier = total_brier / total_count
    avg_average_return = total_average_return / total_count
    is_valid = error_count == 0 and success_count == total_count
    return {
        "status": "success",
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": total_count,
        "ranking_metric_name": "avg_brier",
        "ranking_metric_value": avg_brier,
        "details": {
            "avg_brier": avg_brier,
            "avg_average_return": avg_average_return,
            "success_count": success_count,
            "error_count": error_count,
            "is_valid": is_valid,
            "validity_status": "valid" if is_valid else "invalid",
        },
        "samples": validation_samples,
        "error": None,
    }


def validate_guide_if_needed(
    *,
    guide_file: str,
    guide_name: str,
    dataset_type: str,
    val_records: list[dict[str, Any]],
    llm_config: LLMConfig,
    base_strategy_config: Any,
    use_sources: bool,
    use_market_data: bool,
    validation_key: str,
    validation_key_payload: dict[str, Any],
) -> dict[str, Any]:
    existing_results = load_validation_results()
    existing_entry = get_validation_result_entry(
        existing_results,
        guide_file=guide_file,
        validation_key=validation_key,
    )
    if can_reuse_validation_entry(existing_entry):
        return existing_entry

    validation_config = replace(
        base_strategy_config,
        mem_guide=guide_file,
        save_rollout=False,
        factual_memory_run_label=build_validation_run_label(
            str(getattr(base_strategy_config, "factual_memory_run_label", "") or ""),
            guide_file,
        ),
    )
    validation_experience_bank = build_validation_experience_bank_metadata(validation_config)
    runner = build_dataset_runner(
        dataset_type,
        model=llm_config.model,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )

    try:
        with timed_block(
            "validation.rollout_batch",
            "run_dataset_events_sync",
            kind="validation",
            metadata={
                "guide_file": guide_file,
                "dataset_type": dataset_type,
                "num_samples": len(val_records),
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
            "validation.metrics",
            "compute_validation_result",
            kind="validation",
            metadata={
                "guide_file": guide_file,
                "dataset_type": dataset_type,
                "num_samples": len(val_records),
            },
        ):
            result_entry = compute_validation_result(
                dataset_type=dataset_type,
                records=val_records,
                results=results,
                model_name=llm_config.model,
            )
        result_entry["experience_bank"] = validation_experience_bank
        upsert_validation_result(
            guide_file=guide_file,
            guide_name=guide_name,
            validation_key=validation_key,
            validation_key_payload=validation_key_payload,
            result_entry=result_entry,
        )
        log_info(
            "self_evolving",
            (
                f"Guide validation | guide={guide_file} | status=success | "
                f"ranking_metric_name={result_entry['ranking_metric_name']} | "
                f"ranking_metric_value={result_entry['ranking_metric_value']}"
            ),
        )
        return result_entry
    except Exception as exc:
        ranking_metric_name = "avg_brier" if dataset_type == "prophet_arena" else "avg_accuracy"
        result_entry = {
            "status": "error",
            "validated_at": datetime.now().isoformat(timespec="seconds"),
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
        result_entry["experience_bank"] = validation_experience_bank
        upsert_validation_result(
            guide_file=guide_file,
            guide_name=guide_name,
            validation_key=validation_key,
            validation_key_payload=validation_key_payload,
            result_entry=result_entry,
        )
        log_info(
            "self_evolving",
            f"Guide validation | guide={guide_file} | status=error | error={exc}",
        )
        return result_entry


def ensure_tree_validation_results(
    *,
    tree: dict[str, Any],
    dataset_type: str,
    val_records: list[dict[str, Any]],
    llm_config: LLMConfig,
    base_strategy_config: Any,
    use_sources: bool,
    use_market_data: bool,
    validation_key: str,
    validation_key_payload: dict[str, Any],
) -> dict[str, Any]:
    nodes = tree.get("nodes", {})
    for guide_file, node in sorted(nodes.items()):
        validate_guide_if_needed(
            guide_file=guide_file,
            guide_name=str(node.get("guide_name", "") or "").strip(),
            dataset_type=dataset_type,
            val_records=val_records,
            llm_config=llm_config,
            base_strategy_config=base_strategy_config,
            use_sources=use_sources,
            use_market_data=use_market_data,
            validation_key=validation_key,
            validation_key_payload=validation_key_payload,
        )
    return load_validation_results()


def build_ranked_guide_candidates(
    *,
    tree: dict[str, Any],
    dataset_type: str,
    validation_results: dict[str, Any] | None,
    validation_key: str | None,
    experience_bank_hash: str | None = None,
) -> list[dict[str, Any]]:
    nodes = tree.get("nodes", {})
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("The evolving tree has no nodes.")

    if validation_results is None or validation_key is None:
        return [
            {
                "guide_file": guide_file,
                "guide_name": str(node.get("guide_name", "") or "").strip(),
            }
            for guide_file, node in sorted(nodes.items())
        ]

    ranked: list[dict[str, Any]] = []
    for guide_file, node in nodes.items():
        if experience_bank_hash is not None:
            entry = get_validation_result_entry_for_experience(
                validation_results,
                guide_file=guide_file,
                validation_key=validation_key,
                experience_bank_hash=experience_bank_hash,
            )
        else:
            entry = get_validation_result_entry(
                validation_results,
                guide_file=guide_file,
                validation_key=validation_key,
            )
        if not is_validation_entry_valid(entry):
            continue
        ranked.append(
            {
                "guide_file": guide_file,
                "guide_name": str(node.get("guide_name", "") or "").strip(),
                "validation": entry,
            }
        )

    if dataset_type == "prophet_arena":
        ranked.sort(
            key=lambda item: (
                float(item["validation"]["ranking_metric_value"]),
                str(item["guide_file"]),
            )
        )
    else:
        ranked.sort(
            key=lambda item: (
                -float(item["validation"]["ranking_metric_value"]),
                str(item["guide_file"]),
            )
        )

    harmonic = sum(1.0 / rank for rank in range(1, len(ranked) + 1))
    for rank, candidate in enumerate(ranked, start=1):
        probability = (1.0 / rank) / harmonic if harmonic > 0 else 0.0
        candidate["rank"] = rank
        candidate["probability"] = probability
    return ranked


def choose_tree_node(
    *,
    tree: dict[str, Any],
    dataset_type: str,
    rng: random.Random,
    validation_enabled: bool,
    validation_results: dict[str, Any] | None,
    validation_key: str | None,
    experience_bank_hash: str | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    candidates = build_ranked_guide_candidates(
        tree=tree,
        dataset_type=dataset_type,
        validation_results=validation_results,
        validation_key=validation_key,
        experience_bank_hash=experience_bank_hash,
    )
    if not candidates:
        raise ValueError("No eligible guide nodes are available for selection.")

    if not validation_enabled:
        selected = rng.choice(candidates)
        log_info(
            "self_evolving",
            (
                f"Guide selection | strategy={UNIFORM_RANDOM_SELECTION} | "
                f"candidate_count={len(candidates)} | selected_guide={selected['guide_file']}"
            ),
        )
        return selected["guide_file"], UNIFORM_RANDOM_SELECTION, None

    threshold = rng.random()
    cumulative = 0.0
    selected = candidates[-1]
    for candidate in candidates:
        cumulative += float(candidate["probability"])
        if threshold <= cumulative:
            selected = candidate
            break

    selection_info = {
        "guide_file": selected["guide_file"],
        "guide_name": selected["guide_name"],
        "rank": int(selected["rank"]),
        "probability": float(selected["probability"]),
        "ranking_metric_name": str(selected["validation"]["ranking_metric_name"]),
        "ranking_metric_value": selected["validation"]["ranking_metric_value"],
    }
    log_info(
        "self_evolving",
        (
            f"Guide selection | strategy={ZIPF_BY_VALIDATION_RANK_SELECTION} | "
            f"candidate_count={len(candidates)} | selected_guide={selected['guide_file']} | "
            f"rank={selection_info['rank']} | probability={selection_info['probability']} | "
            f"ranking_metric_name={selection_info['ranking_metric_name']} | "
            f"ranking_metric_value={selection_info['ranking_metric_value']}"
        ),
    )
    return selected["guide_file"], ZIPF_BY_VALIDATION_RANK_SELECTION, selection_info


def choose_best_tree_node(
    *,
    tree: dict[str, Any],
    dataset_type: str,
    validation_enabled: bool,
    validation_results: dict[str, Any] | None,
    validation_key: str | None,
    experience_bank_hash: str | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    candidates = build_ranked_guide_candidates(
        tree=tree,
        dataset_type=dataset_type,
        validation_results=validation_results,
        validation_key=validation_key,
        experience_bank_hash=experience_bank_hash,
    )
    if not candidates:
        raise NoEligibleBestGuideSelectionError(NO_ELIGIBLE_BEST_GUIDE_SELECTION_ERROR)

    selected = candidates[0]
    if not validation_enabled:
        selection_info = {
            "guide_file": selected["guide_file"],
            "guide_name": selected["guide_name"],
        }
        log_info(
            "self_evolving",
            (
                f"Guide selection | strategy=best_available | "
                f"candidate_count={len(candidates)} | selected_guide={selected['guide_file']}"
            ),
        )
        return selected["guide_file"], "best_available", selection_info

    selection_info = {
        "guide_file": selected["guide_file"],
        "guide_name": selected["guide_name"],
        "rank": int(selected["rank"]),
        "probability": float(selected["probability"]),
        "ranking_metric_name": str(selected["validation"]["ranking_metric_name"]),
        "ranking_metric_value": selected["validation"]["ranking_metric_value"],
    }
    log_info(
        "self_evolving",
        (
            f"Guide selection | strategy=best_validation_rank | "
            f"candidate_count={len(candidates)} | selected_guide={selected['guide_file']} | "
            f"rank={selection_info['rank']} | "
            f"ranking_metric_name={selection_info['ranking_metric_name']} | "
            f"ranking_metric_value={selection_info['ranking_metric_value']}"
        ),
    )
    return selected["guide_file"], "best_validation_rank", selection_info
