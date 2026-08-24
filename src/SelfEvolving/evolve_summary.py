#!/usr/bin/env python3

from __future__ import annotations

import json
from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    normalized = str(text or "")
    if max_chars <= 0:
        return "[TRUNCATED]"
    if len(normalized) <= max_chars:
        return normalized
    removed_chars = len(normalized) - max_chars
    marker = f"\n...[TRUNCATED {removed_chars} CHARS]"
    head_chars = max(0, max_chars - len(marker))
    return normalized[:head_chars] + marker


def build_section_text(title: str, body: str, section_limit: int) -> str:
    return f"## {title}\n{truncate_text(body, section_limit)}"


def format_item_dir_files(files: list[dict[str, Any]], summary_max_chars: int) -> str:
    important_cap = max(1200, summary_max_chars // 5)
    other_cap = max(300, summary_max_chars // 14)
    rendered_parts: list[str] = []
    for file_info in files:
        path = str(file_info.get("path", "") or "")
        content = str(file_info.get("content", "") or "")
        cap = important_cap if path in {"raw_data.json", "final_data.txt"} else other_cap
        rendered_parts.append(f"[FILE] {path}\n{truncate_text(content, cap)}")
    return "\n\n".join(rendered_parts)


def build_item_dir_file_char_breakdown(
    files: list[dict[str, Any]],
    summary_max_chars: int,
) -> tuple[int, list[dict[str, Any]]]:
    important_cap = max(1200, summary_max_chars // 5)
    other_cap = max(300, summary_max_chars // 14)
    per_file: list[dict[str, Any]] = []
    total_chars = 0
    for file_info in files:
        path = str(file_info.get("path", "") or "")
        content = str(file_info.get("content", "") or "")
        cap = important_cap if path in {"raw_data.json", "final_data.txt"} else other_cap
        rendered = f"[FILE] {path}\n{truncate_text(content, cap)}"
        rendered_chars = len(rendered)
        total_chars += rendered_chars
        per_file.append(
            {
                "path": path,
                "chars": rendered_chars,
            }
        )
    return total_chars, per_file


def build_rollout_summary(
    rollout: dict[str, Any],
    summary_max_chars: int,
    *,
    include_subagents: bool = True,
) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    breakdown: dict[str, Any] = {
        "task_metadata_chars": 0,
        "main_agent": None,
        "subagents": [],
    }
    metadata = {
        "task_id": rollout.get("task_id"),
        "problem_statement": rollout.get("problem_statement"),
        "task_requirements": rollout.get("task_requirements"),
        "ground_truth": rollout.get("ground_truth"),
        "evaluation": rollout.get("evaluation"),
    }
    metadata_section = build_section_text(
        "Task Metadata",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        max(1200, summary_max_chars // 5),
    )
    parts.append(metadata_section)
    breakdown["task_metadata_chars"] = len(metadata_section)

    main_interactions = rollout.get("main_agent_interactions", [])
    if main_interactions:
        main_interaction = main_interactions[-1]
        main_tools_section = build_section_text(
            "Main Agent Available Tools",
            json.dumps(main_interaction.get("available_tools", []), ensure_ascii=False, indent=2),
            max(1000, summary_max_chars // 6),
        )
        main_messages_section = build_section_text(
            "Main Agent Messages",
            json.dumps(main_interaction.get("messages", []), ensure_ascii=False, indent=2),
            max(1800, summary_max_chars // 4),
        )
        parts.extend([main_tools_section, main_messages_section])
        breakdown["main_agent"] = {
            "total_chars": len(main_tools_section) + len(main_messages_section),
            "available_tools_chars": len(main_tools_section),
            "messages_chars": len(main_messages_section),
        }

    if include_subagents:
        subagent_interactions = rollout.get("subagent_interactions", [])
        for index, interaction in enumerate(subagent_interactions, start=1):
            header_lines = [
                f"Item Dir: {interaction.get('item_dir', '')}",
                f"Guide Name: {interaction.get('guide_name', '')}",
            ]
            metadata_section = build_section_text(
                f"Subagent {index} Metadata",
                "\n".join(header_lines),
                600,
            )
            tools_section = build_section_text(
                f"Subagent {index} Available Tools",
                json.dumps(interaction.get("available_tools", []), ensure_ascii=False, indent=2),
                max(900, summary_max_chars // 8),
            )
            item_dir_files_section = build_section_text(
                f"Subagent {index} Item Dir Files",
                format_item_dir_files(interaction.get("item_dir_files", []), summary_max_chars),
                max(2400, summary_max_chars // 3),
            )
            _, item_dir_files_char_breakdown = build_item_dir_file_char_breakdown(
                interaction.get("item_dir_files", []),
                summary_max_chars,
            )
            messages_section = build_section_text(
                f"Subagent {index} Messages",
                json.dumps(interaction.get("messages", []), ensure_ascii=False, indent=2),
                max(1500, summary_max_chars // 5),
            )
            parts.extend([metadata_section, tools_section, item_dir_files_section, messages_section])
            breakdown["subagents"].append(
                {
                    "index": index,
                    "total_chars": (
                        len(metadata_section)
                        + len(tools_section)
                        + len(item_dir_files_section)
                        + len(messages_section)
                    ),
                    "metadata_chars": len(metadata_section),
                    "available_tools_chars": len(tools_section),
                    "item_dir_files_chars": len(item_dir_files_section),
                    "item_dir_files_char_breakdown": item_dir_files_char_breakdown,
                    "messages_chars": len(messages_section),
                }
            )

    summary_text = "\n\n".join(parts)
    final_summary_text = truncate_text(summary_text, summary_max_chars)
    breakdown["pre_truncation_total_chars"] = len(summary_text)
    breakdown["total_chars"] = len(final_summary_text)
    return final_summary_text, breakdown
