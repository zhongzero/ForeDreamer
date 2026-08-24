#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from utils.generated_paths import (
    resolve_experience_bank_dir,
    resolve_history_experience_evolution_dir,
)


CURRENT_BANK_FILENAME = "current.json"
VALIDATION_RESULTS_FILENAME = "validation_results.json"
VERSIONS_DIRNAME = "versions"


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_dataset_type(dataset_type: str) -> str:
    normalized = str(dataset_type or "").strip()
    if not normalized:
        raise ValueError("dataset_type must be non-empty")
    return normalized


def _bank_dir() -> Path:
    return resolve_experience_bank_dir()


def _versions_dir(dataset_type: str) -> Path:
    del dataset_type
    return _bank_dir() / VERSIONS_DIRNAME


def _current_bank_path(dataset_type: str) -> Path:
    del dataset_type
    return _bank_dir() / CURRENT_BANK_FILENAME


def _experience_bank_lock_path(dataset_type: str) -> Path:
    del dataset_type
    return _bank_dir() / ".experience_bank.lock"


def _validation_results_path(dataset_type: str) -> Path:
    del dataset_type
    return _bank_dir() / VALIDATION_RESULTS_FILENAME


def _validation_results_lock_path(dataset_type: str) -> Path:
    del dataset_type
    return _bank_dir() / ".validation_results.lock"


def _history_dir(dataset_type: str) -> Path:
    _normalize_dataset_type(dataset_type)
    return resolve_history_experience_evolution_dir()


def _history_attempt_lock_path(dataset_type: str) -> Path:
    return _history_dir(dataset_type) / ".attempt.lock"


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_unlocked(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.write("\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = Path(tmp_file.name)
    os.replace(temp_path, path)


def _next_numbered_path(directory: Path, prefix: str, suffix: str) -> Path:
    used_numbers: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
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


def _normalize_experience_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Experience entry must be a JSON object")

    experience_id = str(entry.get("experience_id", "") or "").strip()
    text = str(entry.get("text", "") or "").strip()
    if not experience_id:
        raise ValueError("Experience entry requires a non-empty experience_id")
    if not text:
        raise ValueError(f"Experience entry {experience_id} requires non-empty text")

    normalized = dict(entry)
    normalized["experience_id"] = experience_id
    normalized["text"] = text
    normalized["created_at"] = str(entry.get("created_at", "") or "").strip() or iso_now()
    normalized["updated_at"] = str(entry.get("updated_at", "") or "").strip() or normalized["created_at"]
    normalized["source_attempt_file"] = str(entry.get("source_attempt_file", "") or "").strip() or None
    normalized["source_rollout_file"] = str(entry.get("source_rollout_file", "") or "").strip() or None
    normalized["source_guide_file"] = str(entry.get("source_guide_file", "") or "").strip() or None
    normalized["source_sample_identifier"] = (
        str(entry.get("source_sample_identifier", "") or "").strip() or None
    )
    return normalized


def normalize_experience_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_entry in entries:
        entry = _normalize_experience_entry(raw_entry)
        if entry["experience_id"] in seen_ids:
            raise ValueError(f"Duplicate experience_id in bank: {entry['experience_id']}")
        seen_ids.add(entry["experience_id"])
        normalized_entries.append(entry)
    return normalized_entries


def compute_experience_bank_hash(experience_entries: list[dict[str, Any]]) -> str:
    normalized_for_hash = [
        {
            "experience_id": str(entry.get("experience_id", "") or "").strip(),
            "text": str(entry.get("text", "") or "").strip(),
        }
        for entry in normalize_experience_entries(list(experience_entries))
    ]
    serialized = json.dumps(normalized_for_hash, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_bank_payload(
    *,
    dataset_type: str,
    experience_entries: list[dict[str, Any]],
    version_id: str | None,
    base_version_id: str | None,
    applied_suggestion: dict[str, Any] | None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    normalized_entries = normalize_experience_entries(experience_entries)
    now = iso_now()
    resolved_created_at = str(created_at or "").strip() or now
    resolved_updated_at = str(updated_at or "").strip() or resolved_created_at
    version_name = str(version_id or "").strip() or None
    return {
        "dataset_type": _normalize_dataset_type(dataset_type),
        "version_id": version_name,
        "version_file": f"{version_name}.json" if version_name else None,
        "base_version_id": str(base_version_id or "").strip() or None,
        "bank_hash": compute_experience_bank_hash(normalized_entries),
        "experience_count": len(normalized_entries),
        "experiences": normalized_entries,
        "applied_suggestion": applied_suggestion if isinstance(applied_suggestion, dict) else None,
        "created_at": resolved_created_at,
        "updated_at": resolved_updated_at,
    }


def bootstrap_experience_bank(dataset_type: str) -> dict[str, Any]:
    normalized_dataset_type = _normalize_dataset_type(dataset_type)
    bank_dir = _bank_dir()
    bank_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = _versions_dir(normalized_dataset_type)
    versions_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _experience_bank_lock_path(normalized_dataset_type)
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            current_path = _current_bank_path(normalized_dataset_type)
            if current_path.exists():
                return load_current_experience_bank(normalized_dataset_type)

            version_path = _next_numbered_path(versions_dir, "experience_bank", ".json")
            version_id = version_path.stem
            payload = _build_bank_payload(
                dataset_type=normalized_dataset_type,
                experience_entries=[],
                version_id=version_id,
                base_version_id=None,
                applied_suggestion=None,
            )
            _write_json_unlocked(version_path, payload)
            _write_json_unlocked(current_path, payload)
            return payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_current_experience_bank(dataset_type: str) -> dict[str, Any]:
    normalized_dataset_type = _normalize_dataset_type(dataset_type)
    current_path = _current_bank_path(normalized_dataset_type)
    if not current_path.exists():
        return bootstrap_experience_bank(normalized_dataset_type)

    payload = _load_json_file(current_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid experience bank file: {current_path}")
    experiences = payload.get("experiences", [])
    if not isinstance(experiences, list):
        raise ValueError(f"Experience bank requires list experiences: {current_path}")

    normalized_payload = _build_bank_payload(
        dataset_type=str(payload.get("dataset_type", "") or "").strip() or normalized_dataset_type,
        experience_entries=experiences,
        version_id=str(payload.get("version_id", "") or "").strip() or None,
        base_version_id=str(payload.get("base_version_id", "") or "").strip() or None,
        applied_suggestion=payload.get("applied_suggestion"),
        created_at=str(payload.get("created_at", "") or "").strip() or None,
        updated_at=str(payload.get("updated_at", "") or "").strip() or None,
    )
    return normalized_payload


def load_experience_bank_from_file(dataset_type: str, experience_file: str) -> dict[str, Any]:
    normalized_dataset_type = _normalize_dataset_type(dataset_type)
    normalized_file = str(experience_file or "").strip()
    if not normalized_file:
        raise ValueError("experience_file must be non-empty")

    candidate_path = Path(normalized_file).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (_bank_dir() / candidate_path).resolve()

    current_path = _current_bank_path(normalized_dataset_type)
    if (
        candidate_path == current_path.resolve()
        and not candidate_path.exists()
    ):
        return load_current_experience_bank(normalized_dataset_type)

    if not candidate_path.exists():
        raise ValueError(f"Experience bank file does not exist: {candidate_path}")
    if not candidate_path.is_file():
        raise ValueError(f"Experience bank path is not a file: {candidate_path}")

    payload = _load_json_file(candidate_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid experience bank file: {candidate_path}")
    experiences = payload.get("experiences", [])
    if not isinstance(experiences, list):
        raise ValueError(f"Experience bank requires list experiences: {candidate_path}")

    normalized_payload = _build_bank_payload(
        dataset_type=str(payload.get("dataset_type", "") or "").strip() or normalized_dataset_type,
        experience_entries=experiences,
        version_id=str(payload.get("version_id", "") or "").strip() or None,
        base_version_id=str(payload.get("base_version_id", "") or "").strip() or None,
        applied_suggestion=payload.get("applied_suggestion"),
        created_at=str(payload.get("created_at", "") or "").strip() or None,
        updated_at=str(payload.get("updated_at", "") or "").strip() or None,
    )
    normalized_payload["source_file"] = str(candidate_path)
    return normalized_payload


def load_current_experience_prompt_context(dataset_type: str) -> tuple[list[dict[str, Any]], str]:
    payload = load_current_experience_bank(dataset_type)
    return list(payload.get("experiences", [])), str(payload.get("bank_hash", "") or "")


def render_experience_bank_prompt_section(experience_entries: list[dict[str, Any]]) -> str:
    normalized_entries = normalize_experience_entries(list(experience_entries))
    if not normalized_entries:
        return ""

    lines = [
        "Experience Bank:",
        (
            "These are reusable prior experiences for the main agent. "
            "Use them when relevant, but do not let them override direct evidence from the current task."
        ),
    ]
    for entry in normalized_entries:
        lines.append(f"- [{entry['experience_id']}] {entry['text']}")
    return "\n".join(lines)


def _next_experience_id(experience_entries: list[dict[str, Any]]) -> str:
    used_numbers: set[int] = set()
    for entry in experience_entries:
        experience_id = str(entry.get("experience_id", "") or "").strip()
        if experience_id.startswith("experience_") and experience_id[len("experience_") :].isdigit():
            used_numbers.add(int(experience_id[len("experience_") :]))
    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"experience_{next_number}"


def build_candidate_experience_bank(
    *,
    bank_payload: dict[str, Any],
    suggestion: dict[str, Any],
    source_attempt_file: str | None,
    source_rollout_file: str | None,
    source_guide_file: str | None,
    source_sample_identifier: str | None,
) -> dict[str, Any]:
    dataset_type = _normalize_dataset_type(str(bank_payload.get("dataset_type", "") or ""))
    experience_entries = normalize_experience_entries(list(bank_payload.get("experiences", [])))
    operation = str(suggestion.get("operation", "") or "").strip().lower()
    if operation not in {"add", "remove", "modify"}:
        raise ValueError(f"Unsupported experience suggestion operation: {operation}")

    suggestion_analysis = str(suggestion.get("analysis", "") or "").strip()
    suggestion_generality = str(suggestion.get("generality_assessment", "") or "").strip()
    suggestion_benefit = str(suggestion.get("expected_benefit", "") or "").strip()
    applied_suggestion = {
        "priority": suggestion.get("priority"),
        "operation": operation,
        "analysis": suggestion_analysis,
        "generality_assessment": suggestion_generality,
        "expected_benefit": suggestion_benefit,
        "target_experience_id": str(suggestion.get("target_experience_id", "") or "").strip() or None,
        "new_text": str(suggestion.get("new_text", "") or "").strip() or None,
        "source_attempt_file": str(source_attempt_file or "").strip() or None,
        "source_rollout_file": str(source_rollout_file or "").strip() or None,
        "source_guide_file": str(source_guide_file or "").strip() or None,
        "source_sample_identifier": str(source_sample_identifier or "").strip() or None,
    }

    next_entries = list(experience_entries)
    now = iso_now()
    if operation == "add":
        new_text = str(suggestion.get("new_text", "") or "").strip()
        if not new_text:
            raise ValueError("Experience add suggestion requires non-empty new_text")
        new_experience_id = _next_experience_id(next_entries)
        next_entries.append(
            {
                "experience_id": new_experience_id,
                "text": new_text,
                "created_at": now,
                "updated_at": now,
                "source_attempt_file": str(source_attempt_file or "").strip() or None,
                "source_rollout_file": str(source_rollout_file or "").strip() or None,
                "source_guide_file": str(source_guide_file or "").strip() or None,
                "source_sample_identifier": str(source_sample_identifier or "").strip() or None,
            }
        )
        applied_suggestion["resolved_experience_id"] = new_experience_id
    elif operation == "remove":
        target_experience_id = str(suggestion.get("target_experience_id", "") or "").strip()
        if not target_experience_id:
            raise ValueError("Experience remove suggestion requires target_experience_id")
        removed = False
        remaining_entries: list[dict[str, Any]] = []
        for entry in next_entries:
            if entry["experience_id"] == target_experience_id:
                removed = True
                continue
            remaining_entries.append(entry)
        if not removed:
            raise ValueError(f"Experience remove suggestion target does not exist: {target_experience_id}")
        next_entries = remaining_entries
        applied_suggestion["resolved_experience_id"] = target_experience_id
    else:
        target_experience_id = str(suggestion.get("target_experience_id", "") or "").strip()
        new_text = str(suggestion.get("new_text", "") or "").strip()
        if not target_experience_id:
            raise ValueError("Experience modify suggestion requires target_experience_id")
        if not new_text:
            raise ValueError("Experience modify suggestion requires non-empty new_text")
        modified = False
        updated_entries: list[dict[str, Any]] = []
        for entry in next_entries:
            if entry["experience_id"] == target_experience_id:
                updated_entry = dict(entry)
                updated_entry["text"] = new_text
                updated_entry["updated_at"] = now
                updated_entry["source_attempt_file"] = str(source_attempt_file or "").strip() or None
                updated_entry["source_rollout_file"] = str(source_rollout_file or "").strip() or None
                updated_entry["source_guide_file"] = str(source_guide_file or "").strip() or None
                updated_entry["source_sample_identifier"] = (
                    str(source_sample_identifier or "").strip() or None
                )
                updated_entries.append(updated_entry)
                modified = True
            else:
                updated_entries.append(entry)
        if not modified:
            raise ValueError(f"Experience modify suggestion target does not exist: {target_experience_id}")
        next_entries = updated_entries
        applied_suggestion["resolved_experience_id"] = target_experience_id

    return _build_bank_payload(
        dataset_type=dataset_type,
        experience_entries=next_entries,
        version_id=None,
        base_version_id=str(bank_payload.get("version_id", "") or "").strip() or None,
        applied_suggestion=applied_suggestion,
        created_at=str(bank_payload.get("created_at", "") or "").strip() or None,
        updated_at=now,
    )


def save_experience_bank_version(
    *,
    dataset_type: str,
    experience_entries: list[dict[str, Any]],
    base_version_id: str | None,
    applied_suggestion: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_dataset_type = _normalize_dataset_type(dataset_type)
    bank_dir = _bank_dir()
    bank_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = _versions_dir(normalized_dataset_type)
    versions_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _experience_bank_lock_path(normalized_dataset_type)
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            version_path = _next_numbered_path(versions_dir, "experience_bank", ".json")
            version_id = version_path.stem
            payload = _build_bank_payload(
                dataset_type=normalized_dataset_type,
                experience_entries=experience_entries,
                version_id=version_id,
                base_version_id=base_version_id,
                applied_suggestion=applied_suggestion,
            )
            _write_json_unlocked(version_path, payload)
            _write_json_unlocked(_current_bank_path(normalized_dataset_type), payload)
            return payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_experience_validation_results(dataset_type: str) -> dict[str, Any]:
    path = _validation_results_path(dataset_type)
    if not path.exists():
        return {
            "updated_at": None,
            "banks": {},
        }

    payload = _load_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid experience validation results file: {path}")
    banks = payload.get("banks")
    if not isinstance(banks, dict):
        payload["banks"] = {}
    payload.setdefault("updated_at", None)
    return payload


def get_experience_validation_result_entry(
    validation_results: dict[str, Any],
    *,
    bank_hash: str,
    validation_key: str,
) -> dict[str, Any] | None:
    banks = validation_results.get("banks", {})
    bank_bucket = banks.get(bank_hash)
    if not isinstance(bank_bucket, dict):
        return None
    results = bank_bucket.get("results", {})
    if not isinstance(results, dict):
        return None
    entry = results.get(validation_key)
    return entry if isinstance(entry, dict) else None


def upsert_experience_validation_result(
    *,
    dataset_type: str,
    bank_payload: dict[str, Any],
    guide_file: str,
    guide_name: str,
    validation_key: str,
    validation_key_payload: dict[str, Any],
    result_entry: dict[str, Any],
) -> None:
    bank_dir = _bank_dir()
    bank_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _validation_results_lock_path(dataset_type)
    lock_path.touch(exist_ok=True)

    bank_hash = str(bank_payload.get("bank_hash", "") or "").strip()
    if not bank_hash:
        raise ValueError("Experience bank payload requires bank_hash for validation persistence")

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_experience_validation_results(dataset_type)
            banks = payload.setdefault("banks", {})
            bank_bucket = banks.setdefault(
                bank_hash,
                {
                    "dataset_type": _normalize_dataset_type(dataset_type),
                    "bank_hash": bank_hash,
                    "version_id": str(bank_payload.get("version_id", "") or "").strip() or None,
                    "results": {},
                },
            )
            bank_bucket["dataset_type"] = _normalize_dataset_type(dataset_type)
            bank_bucket["bank_hash"] = bank_hash
            bank_bucket["version_id"] = str(bank_payload.get("version_id", "") or "").strip() or None
            bank_bucket.setdefault("results", {})
            bank_bucket["results"][validation_key] = {
                **result_entry,
                "guide_file": guide_file,
                "guide_name": guide_name,
                "validation_key_payload": validation_key_payload,
                "experience_bank_hash": bank_hash,
                "experience_bank_version_id": str(bank_payload.get("version_id", "") or "").strip() or None,
            }
            payload["updated_at"] = iso_now()
            _write_json_unlocked(_validation_results_path(dataset_type), payload)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def create_experience_attempt_record(
    *,
    dataset_type: str,
    initial_payload: dict[str, Any],
    formatter: Callable[[dict[str, Any], Path | None], dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    history_dir = _history_dir(dataset_type)
    history_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _history_attempt_lock_path(dataset_type)
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            attempt_path = _next_numbered_path(history_dir, "attempt", ".json")
            attempt_payload = dict(initial_payload)
            attempt_payload["attempt_index"] = int(attempt_path.stem.split("_")[1])
            _write_json_unlocked(attempt_path, formatter(attempt_payload, attempt_path))
            return attempt_path, attempt_payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_experience_attempt_record(
    *,
    dataset_type: str,
    attempt_path: Path,
    payload: dict[str, Any],
    formatter: Callable[[dict[str, Any], Path | None], dict[str, Any]],
) -> None:
    history_dir = _history_dir(dataset_type)
    history_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _history_attempt_lock_path(dataset_type)
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _write_json_unlocked(attempt_path, formatter(payload, attempt_path))
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = [
    "bootstrap_experience_bank",
    "build_candidate_experience_bank",
    "compute_experience_bank_hash",
    "create_experience_attempt_record",
    "get_experience_validation_result_entry",
    "iso_now",
    "load_current_experience_bank",
    "load_experience_bank_from_file",
    "load_current_experience_prompt_context",
    "load_experience_validation_results",
    "normalize_experience_entries",
    "render_experience_bank_prompt_section",
    "save_experience_bank_version",
    "upsert_experience_validation_result",
    "write_experience_attempt_record",
]
