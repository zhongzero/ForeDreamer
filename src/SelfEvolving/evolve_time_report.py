#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.timing_registry import export_timing_snapshot
from utils.generated_paths import dynamic_path, resolve_history_guide_and_tool_evolution_dir


SRC_DIR = Path(__file__).resolve().parents[1]
HISTORY_EVOLUTION_DIR = dynamic_path(resolve_history_guide_and_tool_evolution_dir)
REPORT_BASENAME = "total_time_cost"
REPORT_SUFFIX = ".md"


def _format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.000"


def _summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (
            str(event.get("kind", "") or ""),
            str(event.get("component", "") or ""),
            str(event.get("event_name", "") or ""),
        )
        bucket = grouped.setdefault(
            key,
            {
                "kind": key[0],
                "component": key[1],
                "event_name": key[2],
                "count": 0,
                "total_seconds": 0.0,
                "max_seconds": 0.0,
            },
        )
        duration = float(event.get("duration_seconds", 0.0) or 0.0)
        bucket["count"] += 1
        bucket["total_seconds"] += duration
        bucket["max_seconds"] = max(bucket["max_seconds"], duration)

    summary = list(grouped.values())
    for item in summary:
        item["avg_seconds"] = item["total_seconds"] / item["count"] if item["count"] else 0.0
    summary.sort(
        key=lambda item: (
            -float(item["total_seconds"]),
            item["kind"],
            item["component"],
            item["event_name"],
        )
    )
    return summary


def _attempt_sort_key(attempt_key: str) -> tuple[int, str]:
    prefix = "attempt_"
    if attempt_key.startswith(prefix):
        suffix = attempt_key[len(prefix) :]
        if suffix.isdigit():
            return (int(suffix), attempt_key)
    return (10**9, attempt_key)


def _find_report_path() -> Path:
    HISTORY_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    default_path = HISTORY_EVOLUTION_DIR / f"{REPORT_BASENAME}{REPORT_SUFFIX}"
    if not default_path.exists():
        return default_path

    next_number = 2
    while True:
        candidate = HISTORY_EVOLUTION_DIR / f"{REPORT_BASENAME}-{next_number}{REPORT_SUFFIX}"
        if not candidate.exists():
            return candidate
        next_number += 1


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _collect_attempt_details(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    attempt_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempt_totals: dict[str, dict[str, Any]] = {}

    for event in events:
        context = event.get("context", {})
        if not isinstance(context, dict):
            continue
        attempt_file = str(context.get("attempt_file", "") or "").strip()
        execution_id = str(context.get("execution_id", "") or "").strip()
        attempt_key = attempt_file or execution_id
        if not attempt_key:
            continue
        attempt_events[attempt_key].append(event)
        if event.get("kind") == "attempt":
            attempt_totals[attempt_key] = event

    ordered_keys = sorted(attempt_events.keys(), key=_attempt_sort_key)
    ordered_attempts: list[dict[str, Any]] = []
    for attempt_key in ordered_keys:
        total_event = attempt_totals.get(attempt_key)
        context = (total_event or attempt_events[attempt_key][0]).get("context", {})
        metadata = (total_event or {}).get("metadata", {})
        ordered_attempts.append(
            {
                "attempt_key": attempt_key,
                "context": context,
                "metadata": metadata,
                "total_event": total_event,
            }
        )
    return ordered_attempts, attempt_events


def build_timing_report_markdown(snapshot: dict[str, Any] | None = None) -> str:
    snapshot = snapshot or export_timing_snapshot()
    run = snapshot.get("run") if isinstance(snapshot, dict) else None
    events = list(snapshot.get("events", [])) if isinstance(snapshot, dict) else []

    lines: list[str] = ["# Time Cost Report", ""]
    lines.append(
        "Note: submodule totals are wall-clock measurements for nested scopes and can overlap with parent phases, so they should not be summed back to the run total."
    )
    lines.append("")

    if isinstance(run, dict):
        metadata = run.get("metadata", {})
        lines.extend(
            [
                "## Run Summary",
                "",
                f"- Run name: `{run.get('run_name', '')}`",
                f"- Status: `{run.get('status', '')}`",
                f"- Started at: `{run.get('started_at', '')}`",
                f"- Finished at: `{run.get('finished_at', '')}`",
                f"- Total wall time: `{_format_seconds(run.get('duration_seconds'))} s`",
                f"- Dataset: `{metadata.get('dataset_type', 'unknown')}`",
                f"- Train data: `{metadata.get('train_data_path', 'unknown')}`",
                f"- Val data: `{metadata.get('val_data_path', 'None')}`",
                f"- Iterations requested: `{metadata.get('num_iterations', 'unknown')}`",
                f"- Parallelism: `{metadata.get('parallelism', 'unknown')}`",
                f"- Model: `{metadata.get('model', 'unknown')}`",
            ]
        )
        if metadata.get("optimization_encourage_exploration_enabled") is not None:
            lines.append(
                f"- Exploration optimization: `{metadata.get('optimization_encourage_exploration_enabled')}`"
            )
        if metadata.get("optimization_reuse_duplicated_tool_enabled") is not None:
            lines.append(
                f"- Tool reuse optimization: `{metadata.get('optimization_reuse_duplicated_tool_enabled')}`"
            )
        if run.get("error"):
            lines.append(f"- Run error: `{run.get('error')}`")
        lines.append("")

    attempts, attempt_events = _collect_attempt_details(events)
    attempt_total_events = [item["total_event"] for item in attempts if item.get("total_event")]
    mode_buckets: dict[str, list[float]] = defaultdict(list)
    for event in attempt_total_events:
        context = event.get("context", {})
        mode = str(context.get("evolution_mode", "") or "unknown")
        mode_buckets[mode].append(float(event.get("duration_seconds", 0.0) or 0.0))

    lines.append("## Mode Totals")
    lines.append("")
    if mode_buckets:
        mode_rows: list[list[str]] = []
        for mode, durations in sorted(mode_buckets.items()):
            total = sum(durations)
            avg = total / len(durations)
            mode_rows.append(
                [
                    mode,
                    str(len(durations)),
                    _format_seconds(total),
                    _format_seconds(avg),
                    _format_seconds(min(durations)),
                    _format_seconds(max(durations)),
                ]
            )
        lines.extend(
            _build_markdown_table(
                ["mode", "attempt_count", "total_s", "avg_s", "min_s", "max_s"],
                mode_rows,
            )
        )
    else:
        lines.append("No completed attempt timing events were recorded.")
    lines.append("")

    run_level_phase_events = [
        event
        for event in events
        if event.get("kind") == "phase"
        and not str((event.get("context") or {}).get("attempt_file", "") or "").strip()
        and not str((event.get("context") or {}).get("execution_id", "") or "").strip()
    ]
    lines.append("## Run-Level Phases")
    lines.append("")
    if run_level_phase_events:
        phase_rows = [
            [
                item["component"],
                item["event_name"],
                str(item["count"]),
                _format_seconds(item["total_seconds"]),
                _format_seconds(item["avg_seconds"]),
            ]
            for item in _summarize_events(run_level_phase_events)
        ]
        lines.extend(
            _build_markdown_table(
                ["component", "event", "count", "total_s", "avg_s"],
                phase_rows,
            )
        )
    else:
        lines.append("No run-level phase timings were recorded.")
    lines.append("")

    aggregate_submodule_events = [
        event for event in events if event.get("kind") not in {"attempt", "phase", "run"}
    ]
    lines.append("## Aggregate Submodule Totals")
    lines.append("")
    if aggregate_submodule_events:
        aggregate_rows = [
            [
                item["kind"],
                item["component"],
                item["event_name"],
                str(item["count"]),
                _format_seconds(item["total_seconds"]),
                _format_seconds(item["avg_seconds"]),
                _format_seconds(item["max_seconds"]),
            ]
            for item in _summarize_events(aggregate_submodule_events)
        ]
        lines.extend(
            _build_markdown_table(
                ["kind", "component", "event", "count", "total_s", "avg_s", "max_s"],
                aggregate_rows,
            )
        )
    else:
        lines.append("No submodule timing events were recorded.")
    lines.append("")

    lines.append("## Attempt Details")
    lines.append("")
    if not attempts:
        lines.append("No attempt timing events were recorded.")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    for attempt in attempts:
        attempt_key = attempt["attempt_key"]
        total_event = attempt.get("total_event")
        context = attempt.get("context", {})
        metadata = attempt.get("metadata", {})
        mode = str(context.get("evolution_mode", "") or "unknown")
        duration_seconds = total_event.get("duration_seconds", 0.0) if total_event else 0.0

        lines.append(f"### {attempt_key}")
        lines.append("")
        lines.append(f"- Mode: `{mode}`")
        lines.append(f"- Iteration: `{context.get('iteration', 'unknown')}`")
        lines.append(f"- Execution id: `{context.get('execution_id', 'unknown')}`")
        lines.append(f"- Status: `{metadata.get('status', 'unknown')}`")
        lines.append(f"- Duration: `{_format_seconds(duration_seconds)} s`")
        if metadata.get("selected_guide_file"):
            lines.append(f"- Selected guide: `{metadata.get('selected_guide_file')}`")
        if metadata.get("sample_id"):
            lines.append(f"- Sample id: `{metadata.get('sample_id')}`")
        if metadata.get("new_guide_file"):
            lines.append(f"- New guide: `{metadata.get('new_guide_file')}`")
        if metadata.get("error_message"):
            lines.append(f"- Error: `{metadata.get('error_message')}`")
        lines.append("")

        phase_events = [
            event
            for event in attempt_events.get(attempt_key, [])
            if event.get("kind") == "phase"
        ]
        lines.append("Top-level and stage phases:")
        if phase_events:
            phase_rows = [
                [
                    item["component"],
                    item["event_name"],
                    str(item["count"]),
                    _format_seconds(item["total_seconds"]),
                    _format_seconds(item["avg_seconds"]),
                ]
                for item in _summarize_events(phase_events)
            ]
            lines.extend(_build_markdown_table(["component", "event", "count", "total_s", "avg_s"], phase_rows))
        else:
            lines.append("No phase timings were recorded for this attempt.")
        lines.append("")

        submodule_events = [
            event
            for event in attempt_events.get(attempt_key, [])
            if event.get("kind") not in {"attempt", "phase", "run"}
        ]
        lines.append("LLM/search/submodule timings:")
        if submodule_events:
            submodule_rows = [
                [
                    item["kind"],
                    item["component"],
                    item["event_name"],
                    str(item["count"]),
                    _format_seconds(item["total_seconds"]),
                    _format_seconds(item["avg_seconds"]),
                    _format_seconds(item["max_seconds"]),
                ]
                for item in _summarize_events(submodule_events)
            ]
            lines.extend(
                _build_markdown_table(
                    ["kind", "component", "event", "count", "total_s", "avg_s", "max_s"],
                    submodule_rows,
                )
            )
        else:
            lines.append("No LLM/search/submodule timings were recorded for this attempt.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_timing_report(snapshot: dict[str, Any] | None = None) -> Path:
    resolved_snapshot = snapshot or export_timing_snapshot()
    report_path = _find_report_path()
    report_path.write_text(build_timing_report_markdown(resolved_snapshot), encoding="utf-8")
    return report_path
