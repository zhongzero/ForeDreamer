#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from utils.logger import log_info
from SelfEvolving.evolve_storage import (
    GUIDE_SUMMARY_LOCK_PATH,
    GUIDE_SUMMARY_PATH,
    load_current_tree,
    load_guide_object,
    load_guide_summary,
    load_tool_definitions,
    write_json_unlocked,
    iso_now,
)
from SelfEvolving.evolve_validation import (
    get_validation_result_entry,
    is_validation_entry_valid,
)
from SelfEvolving.generate_memguide_and_memtool import (
    DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH,
    LLMConfig,
    classify_guide_with_representatives,
)


def _empty_guide_summary() -> dict[str, Any]:
    return {
        "updated_at": None,
        "categories": {},
        "guide_to_category": {},
        "invalid_guides": {},
        "validation_context": None,
    }


def _ordered_tree_guide_files(tree: dict[str, Any]) -> list[str]:
    nodes = tree.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("The evolving tree is missing nodes.")

    root_guide_file = str(tree.get("root_guide_file", "guide_initial.json") or "guide_initial.json")
    ordered: list[str] = []
    if root_guide_file in nodes:
        ordered.append(root_guide_file)

    remaining = [
        (guide_file, node)
        for guide_file, node in nodes.items()
        if guide_file != root_guide_file and isinstance(node, dict)
    ]
    remaining.sort(key=lambda item: (str(item[1].get("created_at", "") or ""), item[0]))
    ordered.extend(guide_file for guide_file, _ in remaining)
    return ordered


def _tree_guide_set(tree: dict[str, Any]) -> set[str]:
    nodes = tree.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("The evolving tree is missing nodes.")
    return {str(guide_file) for guide_file in nodes}


def _build_invalid_guide_entry(
    *,
    guide_file: str,
    guide_name: str,
    validation_entry: dict[str, Any] | None,
    invalid_reason: str,
    existing_entry: dict[str, Any] | None = None,
    source_attempt_file: str | None = None,
    source_validation_key: str | None = None,
    in_tree: bool,
) -> dict[str, Any]:
    details = validation_entry.get("details") if isinstance(validation_entry, dict) else {}
    if not isinstance(details, dict):
        details = {}

    return {
        "guide_file": guide_file,
        "guide_name": guide_name,
        "invalid_reason": invalid_reason,
        "invalidated_at": (
            existing_entry.get("invalidated_at")
            if isinstance(existing_entry, dict) and existing_entry.get("invalidated_at")
            else iso_now()
        ),
        "source_attempt_file": source_attempt_file,
        "source_validation_key": source_validation_key,
        "in_tree": bool(in_tree),
        "validation_status": (
            str(validation_entry.get("status", "") or "").strip()
            if isinstance(validation_entry, dict)
            else None
        ),
        "ranking_metric_name": (
            validation_entry.get("ranking_metric_name")
            if isinstance(validation_entry, dict)
            else None
        ),
        "ranking_metric_value": (
            validation_entry.get("ranking_metric_value")
            if isinstance(validation_entry, dict)
            else None
        ),
        "num_samples": (
            validation_entry.get("num_samples")
            if isinstance(validation_entry, dict)
            else None
        ),
        "success_count": details.get("success_count"),
        "error_count": details.get("error_count"),
        "validity_status": details.get("validity_status", "invalid"),
    }


def _collect_invalid_tree_guides(
    *,
    tree: dict[str, Any],
    validation_results: dict[str, Any] | None,
    validation_key: str | None,
    existing_invalid_guides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if validation_results is None or validation_key is None:
        return {}

    nodes = tree.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("The evolving tree is missing nodes.")

    invalid_guides: dict[str, Any] = {}
    for guide_file, node in nodes.items():
        entry = get_validation_result_entry(
            validation_results,
            guide_file=guide_file,
            validation_key=validation_key,
        )
        if entry is None or is_validation_entry_valid(entry):
            continue
        existing_entry = (
            existing_invalid_guides.get(guide_file)
            if isinstance(existing_invalid_guides, dict)
            else None
        )
        invalid_guides[guide_file] = _build_invalid_guide_entry(
            guide_file=str(guide_file),
            guide_name=str(node.get("guide_name", "") or "").strip(),
            validation_entry=entry,
            invalid_reason="validation_failed",
            existing_entry=existing_entry if isinstance(existing_entry, dict) else None,
            source_validation_key=validation_key,
            in_tree=True,
        )
    return invalid_guides


def _is_guide_summary_valid_for_tree(
    payload: dict[str, Any],
    tree: dict[str, Any],
    *,
    validation_results: dict[str, Any] | None = None,
    validation_key: str | None = None,
) -> bool:
    categories = payload.get("categories")
    guide_to_category = payload.get("guide_to_category")
    invalid_guides = payload.get("invalid_guides")
    if not isinstance(categories, dict) or not isinstance(guide_to_category, dict) or not isinstance(invalid_guides, dict):
        return False

    tree_guides = _tree_guide_set(tree)
    invalid_tree_guides = set(
        _collect_invalid_tree_guides(
            tree=tree,
            validation_results=validation_results,
            validation_key=validation_key,
            existing_invalid_guides=invalid_guides,
        ).keys()
    )
    validation_context = payload.get("validation_context")
    if validation_key is not None:
        if not isinstance(validation_context, dict):
            return False
        if validation_context.get("validation_key") != validation_key:
            return False
        context_invalid_guides = validation_context.get("invalid_tree_guide_files")
        if not isinstance(context_invalid_guides, list):
            return False
        if {str(item) for item in context_invalid_guides} != invalid_tree_guides:
            return False
    elif validation_context is not None:
        return False

    valid_tree_guides = tree_guides - invalid_tree_guides
    if set(guide_to_category.keys()) != valid_tree_guides:
        return False

    seen_members: set[str] = set()
    for representative_guide_file, category in categories.items():
        if representative_guide_file not in valid_tree_guides:
            return False
        if not isinstance(category, dict):
            return False
        member_guide_files = category.get("member_guide_files")
        if not isinstance(member_guide_files, list) or not member_guide_files:
            return False
        if representative_guide_file not in member_guide_files:
            return False
        if guide_to_category.get(representative_guide_file) != representative_guide_file:
            return False
        for guide_file in member_guide_files:
            if guide_file in seen_members or guide_file not in valid_tree_guides:
                return False
            if guide_to_category.get(guide_file) != representative_guide_file:
                return False
            seen_members.add(guide_file)

    for guide_file in invalid_tree_guides:
        if guide_file not in invalid_guides:
            return False

    return seen_members == valid_tree_guides


def _load_guide_details(guide_file: str, tool_definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    guide_object = load_guide_object(guide_file)
    tool_names = [str(name) for name in guide_object.get("tool_names", [])]
    tool_details: list[dict[str, Any]] = []
    for tool_name in tool_names:
        definition = tool_definitions.get(tool_name)
        if definition is None:
            raise ValueError(f"Missing tool definition for TOOL_NAME={tool_name}")
        tool_details.append(
            {
                "tool_name": tool_name,
                "tool_spec": definition["tool_spec"],
            }
        )
    return {
        "guide_file": guide_file,
        "guide_name": str(guide_object.get("guide_name", "") or "").strip(),
        "prompt": str(guide_object.get("prompt", "") or "").strip(),
        "tool_names": tool_names,
        "tools": tool_details,
    }


def _build_single_guide_bundle(guide_file: str, tool_definitions: dict[str, dict[str, Any]]) -> str:
    details = _load_guide_details(guide_file, tool_definitions)
    tool_blocks = []
    for tool_detail in details["tools"]:
        tool_blocks.append(
            "TOOL_NAME={tool_name}\nTOOL_SPEC:\n{tool_spec}".format(
                tool_name=tool_detail["tool_name"],
                tool_spec=json.dumps(tool_detail["tool_spec"], ensure_ascii=False, indent=2),
            )
        )

    return "\n".join(
        [
            f"guide_file: {details['guide_file']}",
            f"guide_name: {details['guide_name']}",
            f"tool_names: {details['tool_names']}",
            "prompt:",
            details["prompt"],
            "tool_definitions:",
            "\n\n".join(tool_blocks) if tool_blocks else "None.",
        ]
    )


def build_guide_category_representatives_bundle(
    representative_guide_files: list[str],
) -> str:
    tool_definitions = load_tool_definitions()
    bundles = [
        f"### Representative Guide\n{_build_single_guide_bundle(guide_file, tool_definitions)}"
        for guide_file in representative_guide_files
    ]
    return "\n\n".join(bundles)


def _build_new_guide_candidate_bundle(
    guide_file: str,
    *,
    tool_definitions: dict[str, dict[str, Any]],
) -> str:
    return _build_single_guide_bundle(guide_file, tool_definitions)


def _append_guide_to_category_payload(
    payload: dict[str, Any],
    *,
    guide_file: str,
    matched_representative_guide_file: str | None,
) -> dict[str, Any]:
    categories = payload.setdefault("categories", {})
    guide_to_category = payload.setdefault("guide_to_category", {})
    created_at = iso_now()

    representative_guide_file = matched_representative_guide_file
    category_created = False
    if representative_guide_file is None or representative_guide_file not in categories:
        representative_guide_file = guide_file
        category_created = True
        categories[representative_guide_file] = {
            "representative_guide_file": representative_guide_file,
            "member_guide_files": [guide_file],
            "created_at": created_at,
            "updated_at": created_at,
        }
    else:
        category = categories[representative_guide_file]
        member_guide_files = category.setdefault("member_guide_files", [])
        if guide_file not in member_guide_files:
            member_guide_files.append(guide_file)
        category["updated_at"] = created_at

    guide_to_category[guide_file] = representative_guide_file
    payload["updated_at"] = created_at
    return {
        "matched_representative_guide_file": matched_representative_guide_file,
        "assigned_representative_guide_file": representative_guide_file,
        "category_created": category_created,
    }


def _classify_guide_unlocked(
    payload: dict[str, Any],
    *,
    guide_file: str,
    llm_config: LLMConfig,
    prompt_path: Path,
) -> dict[str, Any]:
    representative_guide_files = list(payload.get("categories", {}).keys())
    if not representative_guide_files:
        assignment = _append_guide_to_category_payload(
            payload,
            guide_file=guide_file,
            matched_representative_guide_file=None,
        )
        return {
            "prompt_path": str(prompt_path),
            "prompt_lengths": None,
            "response_text": None,
            "analysis": "No existing guide category representative was available; created a new category.",
            **assignment,
        }

    tool_definitions = load_tool_definitions()
    representatives_bundle = build_guide_category_representatives_bundle(representative_guide_files)
    new_guide_candidate_bundle = _build_new_guide_candidate_bundle(
        guide_file,
        tool_definitions=tool_definitions,
    )
    classification_summary = classify_guide_with_representatives(
        representative_guide_files=representative_guide_files,
        guide_category_representatives_bundle=representatives_bundle,
        new_guide_candidate_bundle=new_guide_candidate_bundle,
        llm_config=llm_config,
        prompt_path=prompt_path,
    )
    assignment = _append_guide_to_category_payload(
        payload,
        guide_file=guide_file,
        matched_representative_guide_file=classification_summary["matched_representative_guide_file"],
    )
    return {
        **classification_summary,
        **assignment,
    }


def _rebuild_guide_summary_unlocked(
    tree: dict[str, Any],
    *,
    llm_config: LLMConfig,
    prompt_path: Path,
    validation_results: dict[str, Any] | None = None,
    validation_key: str | None = None,
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_guide_files = _ordered_tree_guide_files(tree)
    payload = _empty_guide_summary()
    tree_guides = _tree_guide_set(tree)
    existing_invalid_guides = (
        existing_payload.get("invalid_guides")
        if isinstance(existing_payload, dict)
        and isinstance(existing_payload.get("invalid_guides"), dict)
        else {}
    )
    preserved_off_tree_invalid_guides = {
        guide_file: entry
        for guide_file, entry in existing_invalid_guides.items()
        if guide_file not in tree_guides and isinstance(entry, dict)
    }
    payload["invalid_guides"].update(preserved_off_tree_invalid_guides)

    invalid_tree_guides = _collect_invalid_tree_guides(
        tree=tree,
        validation_results=validation_results,
        validation_key=validation_key,
        existing_invalid_guides=existing_invalid_guides,
    )
    payload["invalid_guides"].update(invalid_tree_guides)
    payload["validation_context"] = (
        {
            "validation_key": validation_key,
            "invalid_tree_guide_files": sorted(invalid_tree_guides.keys()),
        }
        if validation_key is not None
        else None
    )

    valid_ordered_guide_files = [
        guide_file for guide_file in ordered_guide_files if guide_file not in invalid_tree_guides
    ]
    if not valid_ordered_guide_files:
        payload["updated_at"] = iso_now()
        write_json_unlocked(GUIDE_SUMMARY_PATH, payload)
        return payload

    root_guide_file = valid_ordered_guide_files[0]
    _append_guide_to_category_payload(
        payload,
        guide_file=root_guide_file,
        matched_representative_guide_file=None,
    )
    for guide_file in valid_ordered_guide_files[1:]:
        _classify_guide_unlocked(
            payload,
            guide_file=guide_file,
            llm_config=llm_config,
            prompt_path=prompt_path,
        )

    payload["updated_at"] = iso_now()
    write_json_unlocked(GUIDE_SUMMARY_PATH, payload)
    return payload


def sync_guide_summary(
    *,
    tree: dict[str, Any],
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH,
    validation_results: dict[str, Any] | None = None,
    validation_key: str | None = None,
) -> dict[str, Any]:
    resolved_prompt_path = Path(prompt_path)
    GUIDE_SUMMARY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDE_SUMMARY_LOCK_PATH.touch(exist_ok=True)

    with GUIDE_SUMMARY_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_guide_summary()
            if _is_guide_summary_valid_for_tree(
                payload,
                tree,
                validation_results=validation_results,
                validation_key=validation_key,
            ):
                return payload
            rebuilt_payload = _rebuild_guide_summary_unlocked(
                tree,
                llm_config=llm_config,
                prompt_path=resolved_prompt_path,
                validation_results=validation_results,
                validation_key=validation_key,
                existing_payload=payload,
            )
            log_info(
                "self_evolving",
                (
                    f"Guide summary sync | guide_summary_path={GUIDE_SUMMARY_PATH} | "
                    f"category_count={len(rebuilt_payload.get('categories', {}))} | "
                    f"guide_count={len(rebuilt_payload.get('guide_to_category', {}))} | "
                    f"invalid_guide_count={len(rebuilt_payload.get('invalid_guides', {}))}"
                ),
            )
            return rebuilt_payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def categorize_tree_guide(
    *,
    guide_file: str,
    llm_config: LLMConfig,
    prompt_path: Path | str = DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH,
    validation_results: dict[str, Any] | None = None,
    validation_key: str | None = None,
) -> dict[str, Any]:
    resolved_prompt_path = Path(prompt_path)
    GUIDE_SUMMARY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDE_SUMMARY_LOCK_PATH.touch(exist_ok=True)

    with GUIDE_SUMMARY_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            tree = load_current_tree()
            payload = load_guide_summary()
            if not _is_guide_summary_valid_for_tree(
                payload,
                tree,
                validation_results=validation_results,
                validation_key=validation_key,
            ):
                payload = _rebuild_guide_summary_unlocked(
                    tree,
                    llm_config=llm_config,
                    prompt_path=resolved_prompt_path,
                    validation_results=validation_results,
                    validation_key=validation_key,
                    existing_payload=payload,
                )

            assigned_representative_guide_file = payload.get("guide_to_category", {}).get(guide_file)
            if isinstance(assigned_representative_guide_file, str) and assigned_representative_guide_file:
                return {
                    "prompt_path": str(resolved_prompt_path),
                    "prompt_lengths": None,
                    "response_text": None,
                    "analysis": "Guide was already present in the current guide summary.",
                    "matched_representative_guide_file": assigned_representative_guide_file,
                    "assigned_representative_guide_file": assigned_representative_guide_file,
                    "category_created": assigned_representative_guide_file == guide_file,
                }

            classification_summary = _classify_guide_unlocked(
                payload,
                guide_file=guide_file,
                llm_config=llm_config,
                prompt_path=resolved_prompt_path,
            )
            write_json_unlocked(GUIDE_SUMMARY_PATH, payload)
            log_info(
                "self_evolving",
                (
                    f"Guide category assignment | guide={guide_file} | "
                    f"matched_representative={classification_summary['matched_representative_guide_file']} | "
                    f"assigned_representative={classification_summary['assigned_representative_guide_file']} | "
                    f"category_created={classification_summary['category_created']}"
                ),
            )
            return classification_summary
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def record_invalid_guide(
    *,
    guide_file: str,
    guide_name: str,
    validation_entry: dict[str, Any] | None,
    source_attempt_file: str | None = None,
    source_validation_key: str | None = None,
) -> dict[str, Any]:
    GUIDE_SUMMARY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDE_SUMMARY_LOCK_PATH.touch(exist_ok=True)

    with GUIDE_SUMMARY_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = load_guide_summary()
            invalid_guides = payload.setdefault("invalid_guides", {})
            existing_entry = invalid_guides.get(guide_file)
            invalid_guides[guide_file] = _build_invalid_guide_entry(
                guide_file=guide_file,
                guide_name=guide_name,
                validation_entry=validation_entry,
                invalid_reason="validation_failed",
                existing_entry=existing_entry if isinstance(existing_entry, dict) else None,
                source_attempt_file=source_attempt_file,
                source_validation_key=source_validation_key,
                in_tree=False,
            )
            payload["updated_at"] = iso_now()
            write_json_unlocked(GUIDE_SUMMARY_PATH, payload)
            return payload
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
