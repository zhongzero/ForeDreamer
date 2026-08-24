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

from MemTool.registry import get_tool_registry, reload_tool_registry
from utils.generated_paths import (
    dynamic_path,
    resolve_history_evolution_dir,
    resolve_history_guide_and_tool_evolution_dir,
    resolve_memguide_dir,
    resolve_memtool_dir,
)
MEMGUIDE_DIR = dynamic_path(resolve_memguide_dir)
MEMTOOL_DIR = dynamic_path(resolve_memtool_dir)
HISTORY_EVOLUTION_ROOT_DIR = dynamic_path(resolve_history_evolution_dir)
HISTORY_GUIDE_AND_TOOL_EVOLUTION_DIR = dynamic_path(resolve_history_guide_and_tool_evolution_dir)
EVOLVING_TREE_PATH = dynamic_path(lambda: resolve_memguide_dir() / "evolving_tree.json")
EVOLVING_TREE_LOCK_PATH = dynamic_path(lambda: resolve_memguide_dir() / ".evolving_tree.lock")
ATTEMPT_LOCK_PATH = dynamic_path(lambda: resolve_history_guide_and_tool_evolution_dir() / ".attempt.lock")
VALIDATION_RESULTS_PATH = dynamic_path(lambda: resolve_memguide_dir() / "validation_results.json")
VALIDATION_RESULTS_LOCK_PATH = dynamic_path(
    lambda: resolve_memguide_dir() / ".validation_results.lock"
)
GUIDE_SUMMARY_PATH = dynamic_path(lambda: resolve_memguide_dir() / "guide_summary.json")
GUIDE_SUMMARY_LOCK_PATH = dynamic_path(lambda: resolve_memguide_dir() / ".guide_summary.lock")
TOOL_SUMMARY_PATH = dynamic_path(lambda: resolve_memtool_dir() / "tool_summary.json")
TOOL_SUMMARY_LOCK_PATH = dynamic_path(lambda: resolve_memtool_dir() / ".tool_summary.lock")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any], lock_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
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
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_json_unlocked(path: Path, payload: dict[str, Any]) -> None:
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


def next_numbered_path(directory: Path, prefix: str, suffix: str) -> Path:
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


def create_attempt_record(
    initial_payload: dict[str, Any],
    formatter: Callable[[dict[str, Any], Path | None], dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    HISTORY_GUIDE_AND_TOOL_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT_LOCK_PATH.touch(exist_ok=True)

    with ATTEMPT_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            attempt_path = next_numbered_path(HISTORY_GUIDE_AND_TOOL_EVOLUTION_DIR, "attempt", ".json")
            attempt_payload = dict(initial_payload)
            attempt_payload["attempt_index"] = int(attempt_path.stem.split("_")[1])
            write_json_unlocked(attempt_path, formatter(attempt_payload, attempt_path))
            return attempt_path, attempt_payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_attempt_record(
    attempt_path: Path,
    payload: dict[str, Any],
    formatter: Callable[[dict[str, Any], Path | None], dict[str, Any]],
) -> None:
    atomic_write_json(attempt_path, formatter(payload, attempt_path), ATTEMPT_LOCK_PATH)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_tool_definitions() -> dict[str, dict[str, Any]]:
    reload_tool_registry()
    definitions: dict[str, dict[str, Any]] = {}
    for tool_name, tool_definition in get_tool_registry().items():
        tool_path = tool_definition.module_path
        code_text = tool_path.read_text(encoding="utf-8")
        definitions[tool_name] = {
            "tool_name": tool_name,
            "tool_file": tool_path.name,
            "tool_spec": tool_definition.spec,
            "code_text": code_text,
            "code_hash": _code_hash(code_text),
        }
    return definitions


def load_guide_object(guide_file: str) -> dict[str, Any]:
    guide_path = MEMGUIDE_DIR / guide_file
    if not guide_path.exists():
        raise ValueError(f"Missing guide file: {guide_path}")
    guide_object = load_json_file(guide_path)
    if not isinstance(guide_object, dict):
        raise ValueError(f"Guide file must contain a JSON object: {guide_path}")
    return guide_object


def load_tool_file_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tool_name, definition in load_tool_definitions().items():
        mapping[tool_name] = str(definition["tool_file"])
    return mapping


def load_tool_summary() -> dict[str, Any]:
    if not TOOL_SUMMARY_PATH.exists():
        return {
            "updated_at": None,
            "tools": {},
            "invalid_tools": {},
        }
    payload = load_json_file(TOOL_SUMMARY_PATH)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tool summary file: {TOOL_SUMMARY_PATH}")
    tools = payload.get("tools")
    if not isinstance(tools, dict):
        payload["tools"] = {}
    invalid_tools = payload.get("invalid_tools")
    if not isinstance(invalid_tools, dict):
        payload["invalid_tools"] = {}
    payload.setdefault("updated_at", None)
    return payload


def _build_tool_summary_entry(
    definition: dict[str, Any],
    existing_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": str(definition["tool_name"]),
        "tool_file": str(definition["tool_file"]),
        "tool_spec": definition["tool_spec"],
        "code_hash": str(definition["code_hash"]),
        "updated_at": (
            existing_entry.get("updated_at")
            if isinstance(existing_entry, dict)
            and existing_entry.get("tool_file") == definition["tool_file"]
            and existing_entry.get("code_hash") == definition["code_hash"]
            and existing_entry.get("tool_spec") == definition["tool_spec"]
            else iso_now()
        ),
    }


def sync_tool_summary() -> dict[str, Any]:
    current_definitions = load_tool_definitions()
    TOOL_SUMMARY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOOL_SUMMARY_LOCK_PATH.touch(exist_ok=True)

    with TOOL_SUMMARY_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_tool_summary()
            existing_tools = payload.get("tools", {})
            existing_invalid_tools = payload.get("invalid_tools", {})
            next_tools: dict[str, Any] = {}
            next_invalid_tools: dict[str, Any] = {}
            changed = False
            invalid_tool_names = {
                str(tool_name)
                for tool_name in existing_invalid_tools.keys()
                if str(tool_name or "").strip()
            }

            for tool_name, definition in current_definitions.items():
                existing_entry = existing_tools.get(tool_name)
                existing_invalid_entry = existing_invalid_tools.get(tool_name)
                tool_entry = _build_tool_summary_entry(
                    definition,
                    existing_entry if isinstance(existing_entry, dict) else existing_invalid_entry,
                )
                if tool_name in invalid_tool_names:
                    invalid_entry = {
                        **tool_entry,
                        "invalidated_at": (
                            existing_invalid_entry.get("invalidated_at")
                            if isinstance(existing_invalid_entry, dict)
                            else iso_now()
                        ),
                        "invalid_reason": (
                            existing_invalid_entry.get("invalid_reason")
                            if isinstance(existing_invalid_entry, dict)
                            else None
                        ),
                        "invalid_source_guide_file": (
                            existing_invalid_entry.get("invalid_source_guide_file")
                            if isinstance(existing_invalid_entry, dict)
                            else None
                        ),
                        "invalid_source_attempt_file": (
                            existing_invalid_entry.get("invalid_source_attempt_file")
                            if isinstance(existing_invalid_entry, dict)
                            else None
                        ),
                        "invalid_source_validation_key": (
                            existing_invalid_entry.get("invalid_source_validation_key")
                            if isinstance(existing_invalid_entry, dict)
                            else None
                        ),
                    }
                    next_invalid_tools[tool_name] = invalid_entry
                    if existing_invalid_entry != invalid_entry:
                        changed = True
                else:
                    next_tools[tool_name] = tool_entry
                    if existing_entry != tool_entry:
                        changed = True

            for tool_name, invalid_entry in existing_invalid_tools.items():
                if tool_name not in next_invalid_tools:
                    next_invalid_tools[tool_name] = invalid_entry

            if set(existing_tools.keys()) != set(next_tools.keys()):
                changed = True
            if set(existing_invalid_tools.keys()) != set(next_invalid_tools.keys()):
                changed = True

            payload["tools"] = next_tools
            payload["invalid_tools"] = next_invalid_tools
            if changed:
                payload["updated_at"] = iso_now()
                write_json_unlocked(TOOL_SUMMARY_PATH, payload)
            return payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def mark_tools_invalid(
    *,
    tool_names: list[str],
    invalid_reason: str,
    source_guide_file: str | None = None,
    source_attempt_file: str | None = None,
    source_validation_key: str | None = None,
) -> dict[str, Any]:
    normalized_tool_names = [
        str(tool_name or "").strip()
        for tool_name in tool_names
        if str(tool_name or "").strip()
    ]
    if not normalized_tool_names:
        return load_tool_summary()

    current_definitions = load_tool_definitions()
    TOOL_SUMMARY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOOL_SUMMARY_LOCK_PATH.touch(exist_ok=True)

    with TOOL_SUMMARY_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_tool_summary()
            tools = payload.setdefault("tools", {})
            invalid_tools = payload.setdefault("invalid_tools", {})
            changed = False

            for tool_name in normalized_tool_names:
                existing_valid_entry = tools.pop(tool_name, None)
                existing_invalid_entry = invalid_tools.get(tool_name)
                definition = current_definitions.get(tool_name)

                if definition is not None:
                    base_entry = _build_tool_summary_entry(
                        definition,
                        existing_valid_entry if isinstance(existing_valid_entry, dict) else existing_invalid_entry,
                    )
                elif isinstance(existing_valid_entry, dict):
                    base_entry = dict(existing_valid_entry)
                elif isinstance(existing_invalid_entry, dict):
                    base_entry = {
                        key: existing_invalid_entry.get(key)
                        for key in ("tool_name", "tool_file", "tool_spec", "code_hash", "updated_at")
                    }
                else:
                    base_entry = {
                        "tool_name": tool_name,
                        "tool_file": "",
                        "tool_spec": {},
                        "code_hash": "",
                        "updated_at": iso_now(),
                    }

                invalid_entry = {
                    **base_entry,
                    "invalidated_at": (
                        existing_invalid_entry.get("invalidated_at")
                        if isinstance(existing_invalid_entry, dict)
                        else iso_now()
                    ),
                    "invalid_reason": invalid_reason,
                    "invalid_source_guide_file": source_guide_file,
                    "invalid_source_attempt_file": source_attempt_file,
                    "invalid_source_validation_key": source_validation_key,
                }
                if existing_invalid_entry != invalid_entry or isinstance(existing_valid_entry, dict):
                    changed = True
                invalid_tools[tool_name] = invalid_entry

            if changed:
                payload["updated_at"] = iso_now()
                write_json_unlocked(TOOL_SUMMARY_PATH, payload)
            return payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def resolve_tool_summary_entries(tool_names: list[str], tool_summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools = tool_summary_payload.get("tools", {})
    if not isinstance(tools, dict):
        raise ValueError("Invalid tool summary payload: missing tools dict.")

    entries: list[dict[str, Any]] = []
    for tool_name in tool_names:
        entry = tools.get(tool_name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing tool summary entry for TOOL_NAME={tool_name}")
        entries.append(
            {
                "tool_name": str(entry.get("tool_name", "") or tool_name),
                "tool_file": str(entry.get("tool_file", "") or ""),
                "tool_spec": entry.get("tool_spec", {}),
                "code_hash": str(entry.get("code_hash", "") or ""),
                "updated_at": entry.get("updated_at"),
            }
        )
    return entries


def build_tool_definition_bundle(tool_names: list[str], tool_summary_payload: dict[str, Any]) -> str:
    bundles: list[str] = []
    for entry in resolve_tool_summary_entries(tool_names, tool_summary_payload):
        bundles.append(
            "### {tool_file} | TOOL_NAME={tool_name}\nTOOL_SPEC:\n{tool_spec}".format(
                tool_file=entry["tool_file"],
                tool_name=entry["tool_name"],
                tool_spec=json.dumps(entry["tool_spec"], ensure_ascii=False, indent=2),
            )
        )
    return "\n\n".join(bundles)


def resolve_tool_files(tool_names: list[str], tool_file_map: dict[str, str]) -> list[str]:
    tool_files: list[str] = []
    for tool_name in tool_names:
        tool_file = tool_file_map.get(tool_name)
        if tool_file is None:
            raise ValueError(f"Could not resolve tool file for TOOL_NAME={tool_name}")
        tool_files.append(tool_file)
    return tool_files


def load_tool_source_bundle(tool_names: list[str], tool_file_map: dict[str, str]) -> str:
    bundles: list[str] = []
    for tool_name in tool_names:
        tool_file = tool_file_map.get(tool_name)
        if tool_file is None:
            raise ValueError(f"Could not resolve tool file for TOOL_NAME={tool_name}")
        tool_path = MEMTOOL_DIR / tool_file
        bundles.append(
            f"### {tool_file} | TOOL_NAME={tool_name}\n{tool_path.read_text(encoding='utf-8')}"
        )
    return "\n\n".join(bundles)


def build_root_tree(tool_file_map: dict[str, str]) -> dict[str, Any]:
    root_guide_file = "guide_initial.json"
    root_guide = load_guide_object(root_guide_file)
    tool_names = [str(name) for name in root_guide.get("tool_names", [])]
    return {
        "root_guide_file": root_guide_file,
        "created_at": iso_now(),
        "nodes": {
            root_guide_file: {
                "guide_file": root_guide_file,
                "guide_name": str(root_guide.get("guide_name", "") or "").strip(),
                "parent_guide_file": None,
                "source_rollout_file": None,
                "source_attempt_file": None,
                "tool_names": tool_names,
                "tool_files": resolve_tool_files(tool_names, tool_file_map),
                "new_tool_files": [],
                "exploration_from_guide_file": None,
                "child_guide_files": [],
                "created_at": iso_now(),
            }
        },
    }


def load_or_initialize_tree(tool_file_map: dict[str, str]) -> dict[str, Any]:
    if EVOLVING_TREE_PATH.exists():
        tree = load_json_file(EVOLVING_TREE_PATH)
        if not isinstance(tree, dict) or not isinstance(tree.get("nodes"), dict):
            raise ValueError(f"Invalid evolving tree file: {EVOLVING_TREE_PATH}")
        return tree

    tree = build_root_tree(tool_file_map)
    atomic_write_json(EVOLVING_TREE_PATH, tree, EVOLVING_TREE_LOCK_PATH)
    return tree


def load_current_tree() -> dict[str, Any]:
    if not EVOLVING_TREE_PATH.exists():
        raise ValueError(f"Missing evolving tree file: {EVOLVING_TREE_PATH}")
    tree = load_json_file(EVOLVING_TREE_PATH)
    if not isinstance(tree, dict) or not isinstance(tree.get("nodes"), dict):
        raise ValueError(f"Invalid evolving tree file: {EVOLVING_TREE_PATH}")
    return tree


def load_validation_results() -> dict[str, Any]:
    if not VALIDATION_RESULTS_PATH.exists():
        return {
            "updated_at": None,
            "guides": {},
        }
    payload = load_json_file(VALIDATION_RESULTS_PATH)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid validation results file: {VALIDATION_RESULTS_PATH}")
    guides = payload.get("guides")
    if not isinstance(guides, dict):
        payload["guides"] = {}
    payload.setdefault("updated_at", None)
    return payload


def load_guide_summary() -> dict[str, Any]:
    if not GUIDE_SUMMARY_PATH.exists():
        return {
            "updated_at": None,
            "categories": {},
            "guide_to_category": {},
            "invalid_guides": {},
            "validation_context": None,
        }
    payload = load_json_file(GUIDE_SUMMARY_PATH)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid guide summary file: {GUIDE_SUMMARY_PATH}")
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        payload["categories"] = {}
    guide_to_category = payload.get("guide_to_category")
    if not isinstance(guide_to_category, dict):
        payload["guide_to_category"] = {}
    invalid_guides = payload.get("invalid_guides")
    if not isinstance(invalid_guides, dict):
        payload["invalid_guides"] = {}
    validation_context = payload.get("validation_context")
    if validation_context is not None and not isinstance(validation_context, dict):
        payload["validation_context"] = None
    payload.setdefault("validation_context", None)
    payload.setdefault("updated_at", None)
    return payload


def _validation_experience_context_key_for_storage(
    entry: dict[str, Any],
    *,
    fallback_key: str,
) -> str:
    experience_bank = entry.get("experience_bank")
    if isinstance(experience_bank, dict):
        bank_hash = str(experience_bank.get("bank_hash", "") or "").strip()
        if bank_hash:
            return bank_hash
        version_id = str(experience_bank.get("version_id", "") or "").strip()
        if version_id:
            return f"version:{version_id}"

    bank_hash = str(entry.get("experience_bank_hash", "") or "").strip()
    if bank_hash:
        return bank_hash
    version_id = str(entry.get("experience_bank_version_id", "") or "").strip()
    if version_id:
        return f"version:{version_id}"
    return f"legacy:{fallback_key}"


def _strip_validation_result_entry_for_storage(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.pop("experience_results", None)
    normalized.pop("selected_experience_context_key", None)
    normalized.pop("available_experience_validation_count", None)
    return normalized


def upsert_validation_result(
    *,
    guide_file: str,
    guide_name: str,
    validation_key: str,
    validation_key_payload: dict[str, Any],
    result_entry: dict[str, Any],
) -> None:
    MEMGUIDE_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_RESULTS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_RESULTS_LOCK_PATH.touch(exist_ok=True)

    with VALIDATION_RESULTS_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_validation_results()
            guides = payload.setdefault("guides", {})
            guide_bucket = guides.setdefault(
                guide_file,
                {
                    "guide_file": guide_file,
                    "guide_name": guide_name,
                    "results": {},
                },
            )
            guide_bucket["guide_name"] = guide_name
            guide_bucket.setdefault("results", {})
            existing_record = guide_bucket["results"].get(validation_key)
            experience_results: dict[str, Any] = {}
            if isinstance(existing_record, dict):
                existing_experience_results = existing_record.get("experience_results")
                if isinstance(existing_experience_results, dict):
                    for context_key, experience_entry in existing_experience_results.items():
                        if isinstance(experience_entry, dict):
                            experience_results[str(context_key)] = _strip_validation_result_entry_for_storage(
                                experience_entry
                            )
                else:
                    context_key = _validation_experience_context_key_for_storage(
                        existing_record,
                        fallback_key=validation_key,
                    )
                    experience_results[context_key] = _strip_validation_result_entry_for_storage(existing_record)

            normalized_result_entry = {
                **_strip_validation_result_entry_for_storage(result_entry),
                "guide_file": guide_file,
                "guide_name": guide_name,
                "validation_key_payload": validation_key_payload,
            }
            current_context_key = _validation_experience_context_key_for_storage(
                normalized_result_entry,
                fallback_key=validation_key,
            )
            experience_results[current_context_key] = normalized_result_entry
            guide_bucket["results"][validation_key] = {
                "guide_file": guide_file,
                "guide_name": guide_name,
                "validation_key_payload": validation_key_payload,
                "experience_results": experience_results,
            }
            payload["updated_at"] = iso_now()
            write_json_unlocked(VALIDATION_RESULTS_PATH, payload)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_tree_child_locked(
    *,
    parent_guide_file: str | None,
    child_guide_file: str,
    source_rollout_file: str | None,
    source_attempt_file: str,
    new_tool_files: list[str],
    tool_file_map: dict[str, str],
    exploration_from_guide_file: list[str] | None = None,
) -> dict[str, Any]:
    def _append_tree_child(tree: dict[str, Any]) -> dict[str, Any]:
        nodes = tree.get("nodes")
        if not isinstance(nodes, dict):
            raise ValueError("The evolving tree is missing nodes.")
        if parent_guide_file is not None and parent_guide_file not in nodes:
            raise ValueError(f"Parent guide is not present in the evolving tree: {parent_guide_file}")
        if child_guide_file in nodes:
            raise ValueError(f"Child guide already exists in the evolving tree: {child_guide_file}")

        child_guide = load_guide_object(child_guide_file)
        tool_names = [str(name) for name in child_guide.get("tool_names", [])]
        child_node = {
            "guide_file": child_guide_file,
            "guide_name": str(child_guide.get("guide_name", "") or "").strip(),
            "parent_guide_file": parent_guide_file,
            "source_rollout_file": source_rollout_file,
            "source_attempt_file": source_attempt_file,
            "tool_names": tool_names,
            "tool_files": resolve_tool_files(tool_names, tool_file_map),
            "new_tool_files": list(new_tool_files),
            "exploration_from_guide_file": list(exploration_from_guide_file) if exploration_from_guide_file else None,
            "child_guide_files": [],
            "created_at": iso_now(),
        }
        nodes[child_guide_file] = child_node
        if parent_guide_file is not None:
            parent_children = nodes[parent_guide_file].setdefault("child_guide_files", [])
            if child_guide_file not in parent_children:
                parent_children.append(child_guide_file)
        return child_node

    MEMGUIDE_DIR.mkdir(parents=True, exist_ok=True)
    EVOLVING_TREE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVOLVING_TREE_LOCK_PATH.touch(exist_ok=True)

    with EVOLVING_TREE_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if EVOLVING_TREE_PATH.exists():
                tree = load_json_file(EVOLVING_TREE_PATH)
            else:
                tree = build_root_tree(tool_file_map)
            child_node = _append_tree_child(tree)
            write_json_unlocked(EVOLVING_TREE_PATH, tree)
            return child_node
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
