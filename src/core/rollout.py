#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import fcntl

from utils.generated_paths import dynamic_path, resolve_history_rollout_dir


_TRACE_LOCK = Lock()
_TASK_ROLLOUTS: dict[str, dict[str, Any]] = {}
_HISTORY_ROLLOUT_DIR = dynamic_path(resolve_history_rollout_dir)
_HISTORY_ROLLOUT_LOCK_PATH = dynamic_path(lambda: resolve_history_rollout_dir() / ".rollout.lock")


def _ensure_rollout_entry(task_id: str) -> dict[str, Any]:
    if task_id not in _TASK_ROLLOUTS:
        _TASK_ROLLOUTS[task_id] = {
            "task_id": task_id,
            "sample_identifier": "",
            "problem_statement": "",
            "task_requirements": "",
            "ground_truth": None,
            "evaluation": {},
            "main_agent_interactions": [],
            "subagent_interactions": [],
        }
    return _TASK_ROLLOUTS[task_id]


def register_rollout(
    task_id: str,
    *,
    problem_statement: str = "",
    task_requirements: str = "",
    sample_identifier: str = "",
) -> None:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    with _TRACE_LOCK:
        entry = _ensure_rollout_entry(normalized_task_id)
        if sample_identifier:
            entry["sample_identifier"] = str(sample_identifier)
        if problem_statement:
            entry["problem_statement"] = str(problem_statement)
        if task_requirements:
            entry["task_requirements"] = str(task_requirements)


def update_rollout_outcome(
    task_id: str,
    *,
    ground_truth: Any,
    evaluation: dict[str, Any],
) -> None:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    with _TRACE_LOCK:
        entry = _ensure_rollout_entry(normalized_task_id)
        entry["ground_truth"] = deepcopy(ground_truth)
        entry["evaluation"] = deepcopy(evaluation)


def record_main_agent_interaction(
    *,
    task_id: str,
    messages: list[dict[str, Any]],
    available_tools: list[dict[str, Any]] | None,
) -> None:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    with _TRACE_LOCK:
        entry = _ensure_rollout_entry(normalized_task_id)
        entry["main_agent_interactions"] = [
            {
                "available_tools": deepcopy(available_tools) if available_tools is not None else [],
                "messages": deepcopy(messages),
            }
        ]


def record_subagent_interaction(
    *,
    task_id: str,
    item_dir: str,
    guide_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    with _TRACE_LOCK:
        entry = _ensure_rollout_entry(normalized_task_id)
        interaction = {
            "item_dir": item_dir,
            "guide_name": guide_name,
            "item_dir_files": _collect_item_dir_files(Path(item_dir)),
            "available_tools": deepcopy(tools),
            "messages": deepcopy(messages),
        }
        existing_interactions = entry["subagent_interactions"]
        for index, existing_interaction in enumerate(existing_interactions):
            if existing_interaction.get("item_dir") == item_dir:
                existing_interactions[index] = interaction
                break
        else:
            existing_interactions.append(interaction)


def pop_rollout(task_id: str) -> dict[str, Any]:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return {
            "task_id": "",
            "sample_identifier": "",
            "problem_statement": "",
            "task_requirements": "",
            "ground_truth": None,
            "evaluation": {},
            "main_agent_interactions": [],
            "subagent_interactions": [],
        }

    with _TRACE_LOCK:
        entry = _TASK_ROLLOUTS.pop(normalized_task_id, None)

    if entry is None:
        return {
            "task_id": normalized_task_id,
            "sample_identifier": "",
            "problem_statement": "",
            "task_requirements": "",
            "ground_truth": None,
            "evaluation": {},
            "main_agent_interactions": [],
            "subagent_interactions": [],
        }
    return deepcopy(entry)


def save_rollout(rollout: dict[str, Any]) -> str:
    _HISTORY_ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    _HISTORY_ROLLOUT_LOCK_PATH.touch(exist_ok=True)

    with _HISTORY_ROLLOUT_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            next_path = _next_rollout_path()
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=_HISTORY_ROLLOUT_DIR,
                prefix=".rollout_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(rollout, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.write("\n")
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                temp_path = Path(tmp_file.name)

            os.replace(temp_path, next_path)
            return str(next_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _next_rollout_path() -> Path:
    used_numbers: set[int] = set()
    for child in _HISTORY_ROLLOUT_DIR.iterdir():
        if not child.is_file():
            continue
        if not child.name.startswith("rollout_") or child.suffix != ".json":
            continue
        stem = child.stem
        prefix, _, suffix = stem.partition("_")
        if prefix != "rollout" or not suffix.isdigit():
            continue
        used_numbers.add(int(suffix))

    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return _HISTORY_ROLLOUT_DIR / f"rollout_{next_number}.json"


def _collect_item_dir_files(item_dir: Path) -> list[dict[str, str]]:
    if not item_dir.exists() or not item_dir.is_dir():
        return []

    file_paths = [path for path in item_dir.rglob("*") if path.is_file()]
    file_paths.sort(key=_item_dir_file_sort_key)

    collected: list[dict[str, str]] = []
    for file_path in file_paths:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        collected.append(
            {
                "path": file_path.relative_to(item_dir).as_posix(),
                "content": content,
            }
        )
    return collected


def _item_dir_file_sort_key(file_path: Path) -> tuple[int, str]:
    name = file_path.name
    relative_path = file_path.as_posix()
    if name == "raw_data.json":
        return (0, relative_path)
    if name == "final_data.txt":
        return (2, relative_path)
    return (1, relative_path)
