#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from typing import Any


_CURRENT_TIMING_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "current_timing_context",
    default={},
)
_TIMING_LOCK = threading.Lock()
_TIMING_STATE: dict[str, Any] = {
    "run": None,
    "events": [],
    "next_event_id": 1,
}


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key)] = _normalize_value(item)
        return normalized
    return str(value)


def _normalize_mapping(values: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        normalized[str(key)] = _normalize_value(value)
    return normalized


def reset_timing_registry() -> None:
    with _TIMING_LOCK:
        _TIMING_STATE["run"] = None
        _TIMING_STATE["events"] = []
        _TIMING_STATE["next_event_id"] = 1
    _CURRENT_TIMING_CONTEXT.set({})


def start_timing_run(*, run_name: str, metadata: dict[str, Any] | None = None) -> None:
    with _TIMING_LOCK:
        _TIMING_STATE["run"] = {
            "run_name": str(run_name).strip() or "self_evolving",
            "started_at": _iso_now(),
            "start_perf_counter": time.perf_counter(),
            "status": "running",
            "metadata": _normalize_mapping(metadata),
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
        }


def update_run_metadata(**updates: Any) -> None:
    normalized_updates = _normalize_mapping(updates)
    if not normalized_updates:
        return

    with _TIMING_LOCK:
        run = _TIMING_STATE.get("run")
        if not isinstance(run, dict):
            return
        metadata = run.setdefault("metadata", {})
        metadata.update(normalized_updates)


def finish_timing_run(*, status: str, error: str | None = None) -> None:
    with _TIMING_LOCK:
        run = _TIMING_STATE.get("run")
        if not isinstance(run, dict):
            return
        finished_at = _iso_now()
        duration_seconds = time.perf_counter() - float(run["start_perf_counter"])
        run["status"] = str(status or "unknown")
        run["finished_at"] = finished_at
        run["duration_seconds"] = duration_seconds
        run["error"] = str(error) if error else None


def export_timing_snapshot() -> dict[str, Any]:
    with _TIMING_LOCK:
        run = _TIMING_STATE.get("run")
        if isinstance(run, dict):
            run_snapshot = {
                key: value
                for key, value in run.items()
                if key != "start_perf_counter"
            }
            if run_snapshot.get("duration_seconds") is None:
                run_snapshot["duration_seconds"] = time.perf_counter() - float(run["start_perf_counter"])
        else:
            run_snapshot = None

        return {
            "run": run_snapshot,
            "events": [dict(event) for event in _TIMING_STATE.get("events", [])],
        }


def get_current_timing_context() -> dict[str, Any]:
    return dict(_CURRENT_TIMING_CONTEXT.get({}))


@contextmanager
def push_timing_context(**updates: Any):
    current = get_current_timing_context()
    normalized_updates = _normalize_mapping(updates)
    merged = dict(current)
    merged.update(normalized_updates)
    token: Token = _CURRENT_TIMING_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _CURRENT_TIMING_CONTEXT.reset(token)


def _append_event(event: dict[str, Any]) -> None:
    with _TIMING_LOCK:
        event["event_id"] = int(_TIMING_STATE["next_event_id"])
        _TIMING_STATE["next_event_id"] += 1
        _TIMING_STATE["events"].append(event)


@dataclass
class TimedBlock:
    component: str
    event_name: str
    kind: str = "phase"
    metadata: dict[str, Any] = field(default_factory=dict)
    _context: dict[str, Any] = field(init=False, default_factory=dict)
    _started_at: str = field(init=False, default="")
    _start_perf_counter: float = field(init=False, default=0.0)

    def __enter__(self) -> "TimedBlock":
        self._context = get_current_timing_context()
        self.metadata = _normalize_mapping(self.metadata)
        self._started_at = _iso_now()
        self._start_perf_counter = time.perf_counter()
        return self

    def set_metadata(self, **updates: Any) -> None:
        self.metadata.update(_normalize_mapping(updates))

    def __exit__(self, exc_type, exc, tb) -> bool:
        finished_at = _iso_now()
        duration_seconds = time.perf_counter() - self._start_perf_counter
        event_metadata = dict(self.metadata)
        if exc is not None:
            event_metadata.setdefault("status", "error")
            event_metadata.setdefault("error_message", str(exc))
        event = {
            "kind": str(self.kind or "phase"),
            "component": str(self.component).strip(),
            "event_name": str(self.event_name).strip(),
            "started_at": self._started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "context": dict(self._context),
            "metadata": event_metadata,
        }
        _append_event(event)
        return False


def timed_block(
    component: str,
    event_name: str,
    *,
    kind: str = "phase",
    metadata: dict[str, Any] | None = None,
) -> TimedBlock:
    return TimedBlock(
        component=str(component).strip(),
        event_name=str(event_name).strip(),
        kind=str(kind).strip() or "phase",
        metadata=dict(metadata or {}),
    )

