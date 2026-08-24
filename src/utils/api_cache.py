#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import fcntl

from utils.generated_paths import is_api_cache_enabled, resolve_cache_dir
from utils.search_provider import FIRECRAWL_SEARCH_PROVIDER, TAVILY_SEARCH_PROVIDER


LLM_CACHE_SUBDIR: Final[str] = "LLM_api_return"
SEARCH_CACHE_SUBDIR: Final[str] = "search_api_return"
CACHE_STATS_FILENAME: Final[str] = "cache_stats.json"
CACHE_STATS_LOCK_FILENAME: Final[str] = ".cache_stats.lock"


@dataclass(frozen=True)
class CachedMessagePayload:
    role: str
    content: str
    tool_calls: tuple[dict[str, Any], ...] = ()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _cache_key_for_identity(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def compute_api_cache_key(identity: dict[str, Any]) -> str:
    return _cache_key_for_identity(identity)


def _ensure_cache_layout() -> Path:
    cache_dir = resolve_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / LLM_CACHE_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cache_dir / SEARCH_CACHE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_file_path(subdir: str, cache_key: str) -> Path:
    return _ensure_cache_layout() / subdir / f"{cache_key}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _empty_stats_payload() -> dict[str, Any]:
    return {
        "llm_api": {
            "actual_calls": 0,
            "cache_hits": 0,
        },
        "search_api": {
            "actual_calls": 0,
            "cache_hits": 0,
            "lossy_cache_hits": 0,
            "providers": {
                TAVILY_SEARCH_PROVIDER: {
                    "actual_calls": 0,
                    "cache_hits": 0,
                    "lossy_cache_hits": 0,
                },
                FIRECRAWL_SEARCH_PROVIDER: {
                    "actual_calls": 0,
                    "cache_hits": 0,
                    "lossy_cache_hits": 0,
                },
            },
        },
        "updated_at": iso_now(),
    }


def _stats_path() -> Path:
    return _ensure_cache_layout() / CACHE_STATS_FILENAME


def _stats_lock_path() -> Path:
    return _ensure_cache_layout() / CACHE_STATS_LOCK_FILENAME


def _load_stats_payload_unlocked() -> dict[str, Any]:
    payload = _load_json_file(_stats_path())
    if payload is None:
        return _empty_stats_payload()
    llm_api = payload.setdefault("llm_api", {})
    llm_api.setdefault("actual_calls", 0)
    llm_api.setdefault("cache_hits", 0)
    search_api = payload.setdefault("search_api", {})
    search_api.setdefault("actual_calls", 0)
    search_api.setdefault("cache_hits", 0)
    search_api.setdefault("lossy_cache_hits", 0)
    providers = search_api.setdefault("providers", {})
    for provider in (TAVILY_SEARCH_PROVIDER, FIRECRAWL_SEARCH_PROVIDER):
        provider_payload = providers.setdefault(provider, {})
        provider_payload.setdefault("actual_calls", 0)
        provider_payload.setdefault("cache_hits", 0)
        provider_payload.setdefault("lossy_cache_hits", 0)
    payload.setdefault("updated_at", iso_now())
    return payload


def _update_stats(*, channel: str, metric: str, provider: str | None = None, increment: int = 1) -> None:
    if increment < 1:
        raise ValueError("increment must be >= 1")
    lock_path = _stats_lock_path()
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            payload = _load_stats_payload_unlocked()
            section = payload[channel]
            section[metric] = int(section.get(metric, 0)) + increment
            if channel == "search_api" and provider:
                provider_section = section.setdefault("providers", {}).setdefault(
                    provider,
                    {"actual_calls": 0, "cache_hits": 0},
                )
                provider_section[metric] = int(provider_section.get(metric, 0)) + increment
            payload["updated_at"] = iso_now()
            _write_json_atomic(_stats_path(), payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def record_llm_actual_call(increment: int = 1) -> None:
    if not is_api_cache_enabled():
        return
    _update_stats(channel="llm_api", metric="actual_calls", increment=increment)


def record_llm_cache_hit(increment: int = 1) -> None:
    if not is_api_cache_enabled():
        return
    _update_stats(channel="llm_api", metric="cache_hits", increment=increment)


def record_search_actual_call(provider: str, increment: int = 1) -> None:
    if not is_api_cache_enabled():
        return
    _update_stats(channel="search_api", metric="actual_calls", provider=provider, increment=increment)


def record_search_cache_hit(provider: str, increment: int = 1) -> None:
    if not is_api_cache_enabled():
        return
    _update_stats(channel="search_api", metric="cache_hits", provider=provider, increment=increment)


def record_search_lossy_cache_hit(provider: str, increment: int = 1) -> None:
    if not is_api_cache_enabled():
        return
    _update_stats(channel="search_api", metric="lossy_cache_hits", provider=provider, increment=increment)


def build_llm_request_identity(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    reasoning: Any = None,
    extra_body: Any = None,
) -> dict[str, Any]:
    return {
        "base_url": str(base_url or "").strip(),
        "model": str(model or "").strip(),
        "messages": _normalize_json_value(messages),
        "tools": _normalize_json_value(tools),
        "tool_choice": _normalize_json_value(tool_choice),
        "reasoning": _normalize_json_value(reasoning),
        "extra_body": _normalize_json_value(extra_body),
    }


def build_search_request_identity(
    *,
    provider: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provider": str(provider or "").strip().lower(),
        "endpoint": str(endpoint or "").strip(),
        "payload": _normalize_json_value(payload),
    }


def build_lossy_search_identity(
    *,
    provider: str,
    endpoint: str,
    payload: dict[str, Any],
    dataset_type: str,
    sample_identifier: str,
    search_turn: int,
) -> dict[str, Any]:
    normalized_payload = dict(_normalize_json_value(payload))
    normalized_payload.pop("query", None)
    return {
        "provider": str(provider or "").strip().lower(),
        "endpoint": str(endpoint or "").strip(),
        "dataset_type": str(dataset_type or "").strip(),
        "sample_identifier": str(sample_identifier or "").strip(),
        "search_turn": int(search_turn),
        "payload_without_query": normalized_payload,
    }


def load_llm_cache_entry(request_identity: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    cache_key = _cache_key_for_identity(request_identity)
    if not is_api_cache_enabled():
        return cache_key, None
    entry = _load_json_file(_cache_file_path(LLM_CACHE_SUBDIR, cache_key))
    return cache_key, entry


def load_search_cache_entry(request_identity: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    cache_key = _cache_key_for_identity(request_identity)
    if not is_api_cache_enabled():
        return cache_key, None
    entry = _load_json_file(_cache_file_path(SEARCH_CACHE_SUBDIR, cache_key))
    return cache_key, entry


def find_latest_lossy_search_cache_entry(lossy_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not is_api_cache_enabled():
        return None
    search_dir = _ensure_cache_layout() / SEARCH_CACHE_SUBDIR
    if not search_dir.exists():
        return None

    target_lossy_key = _cache_key_for_identity(lossy_identity)
    matched_entry: dict[str, Any] | None = None
    matched_created_at = ""
    for path in sorted(search_dir.glob("*.json")):
        entry = _load_json_file(path)
        if not entry:
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("lossy_cache_key", "") or "").strip() != target_lossy_key:
            continue
        created_at = str(entry.get("created_at", "") or "")
        if matched_entry is None or created_at > matched_created_at:
            matched_entry = entry
            matched_created_at = created_at
    return matched_entry


def serialize_tool_calls(tool_calls: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for tool_call in list(tool_calls or []):
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        serialized.append(
            {
                "id": str(getattr(tool_call, "id", "") or "").strip(),
                "type": str(getattr(tool_call, "type", "function") or "function"),
                "function": {
                    "name": str(getattr(function, "name", "") or "").strip(),
                    "arguments": str(getattr(function, "arguments", "") or ""),
                },
            }
        )
    return serialized


def serialize_message_payload(
    *,
    content: str,
    tool_calls: list[Any] | tuple[Any, ...] | None,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": str(content or ""),
        "tool_calls": serialize_tool_calls(tool_calls),
    }


def store_llm_cache_entry(
    *,
    cache_key: str,
    request_identity: dict[str, Any],
    message_payload: dict[str, Any],
) -> None:
    if not is_api_cache_enabled():
        return
    payload = {
        "cache_key": cache_key,
        "created_at": iso_now(),
        "request": _normalize_json_value(request_identity),
        "response": {
            "message": _normalize_json_value(message_payload),
        },
    }
    _write_json_atomic(_cache_file_path(LLM_CACHE_SUBDIR, cache_key), payload)


def store_search_cache_entry(
    *,
    cache_key: str,
    request_identity: dict[str, Any],
    raw_response_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    if not is_api_cache_enabled():
        return
    payload = {
        "cache_key": cache_key,
        "created_at": iso_now(),
        "request": _normalize_json_value(request_identity),
        "response": _normalize_json_value(raw_response_payload),
    }
    if metadata:
        payload["metadata"] = _normalize_json_value(metadata)
    _write_json_atomic(_cache_file_path(SEARCH_CACHE_SUBDIR, cache_key), payload)


def build_cached_completion(message_payload: dict[str, Any]) -> SimpleNamespace:
    tool_call_objects = []
    for tool_call in list(message_payload.get("tool_calls", []) or []):
        function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        tool_call_objects.append(
            SimpleNamespace(
                id=str(tool_call.get("id", "") or ""),
                type=str(tool_call.get("type", "function") or "function"),
                function=SimpleNamespace(
                    name=str(function_payload.get("name", "") or ""),
                    arguments=str(function_payload.get("arguments", "") or ""),
                ),
            )
        )
    message = SimpleNamespace(
        role=str(message_payload.get("role", "assistant") or "assistant"),
        content=str(message_payload.get("content", "") or ""),
        tool_calls=tool_call_objects,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])
