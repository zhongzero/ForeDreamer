#!/usr/bin/env python3

import asyncio
import json
import os
import re
import threading
import time
import weakref
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Optional

import httpx

from utils.api_cache import (
    build_lossy_search_identity,
    build_search_request_identity,
    compute_api_cache_key,
    find_latest_lossy_search_cache_entry,
    load_search_cache_entry,
    record_search_actual_call,
    record_search_cache_hit,
    record_search_lossy_cache_hit,
    store_search_cache_entry,
)
from utils.logger import format_json, log_block, log_info
from utils.timing_registry import timed_block
from utils.generated_paths import resolve_factual_memory_dir
from utils.search_provider import (
    FIRECRAWL_SEARCH_PROVIDER,
    TAVILY_SEARCH_PROVIDER,
    default_search_api_env_var,
    normalize_search_provider,
)


TOOL_NAME: Final[str] = "privacy_web_search"
IS_PUBLIC_TOOL: Final[bool] = False
TOOL_SPEC: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Private tool for web search, normalization, and raw_data.json creation.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to issue.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
TAVILY_SEARCH_URL: Final[str] = "https://api.tavily.com/search"
FIRECRAWL_SEARCH_URL: Final[str] = "https://api.firecrawl.dev/v2/search"
TAVILY_MAX_QUERY_LENGTH: Final[int] = 400
ASYNC_TAVILY_CONCURRENCY_LIMIT: Final[int] = max(
    1,
    int(os.getenv("TAVILY_CONCURRENCY_LIMIT", "8")),
)
TAVILY_HTTP_429_MAX_ATTEMPTS: Final[int] = 3
TAVILY_HTTP_429_RETRY_DELAY_SECONDS: Final[float] = 3.0
FIRECRAWL_HTTP_429_MAX_ATTEMPTS: Final[int] = 3
FIRECRAWL_HTTP_429_RETRY_DELAY_SECONDS: Final[float] = 3.0


class TavilySearchError(Exception):
    """Raised when Tavily search fails or returns unusable data."""

    def __init__(self, message: str, *, http_status: Optional[int] = None):
        super().__init__(message)
        self.http_status = http_status


class _TavilyKeyPool:
    def __init__(self, configured_keys: tuple[str, ...]):
        self.configured_keys = configured_keys
        self.active_keys = list(configured_keys)
        self.lock = threading.Lock()


_TAVILY_KEY_POOLS_LOCK = threading.Lock()
_TAVILY_KEY_POOLS: dict[tuple[str, ...], _TavilyKeyPool] = {}
_ASYNC_TAVILY_SEMAPHORES_LOCK = threading.Lock()
_ASYNC_TAVILY_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Semaphore,
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class _SearchResult:
    title: str
    url: str
    content: str
    published_date: Optional[str] = None
    score: Optional[float] = None


def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError(f"{TOOL_NAME} requires a non-empty query")

    search_provider = normalize_search_provider(getattr(config, "search_provider", TAVILY_SEARCH_PROVIDER))
    api_key = getattr(config, "search_api_key", None) or getattr(config, "api_key", None)
    if not api_key:
        env_var_name = default_search_api_env_var(search_provider)
        api_key = str(os.getenv(env_var_name, "") or "").strip() or None
    if not api_key:
        raise ValueError(
            f"{TOOL_NAME} requires search_api_key in config for provider={search_provider}"
        )

    task_id = str(getattr(runtime_context, "task_id", "")).strip()
    if not task_id:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime task_id")
    search_turn = int(getattr(runtime_context, "search_turn", 0))
    if search_turn < 1:
        raise ValueError(f"{TOOL_NAME} requires runtime search_turn >= 1")
    problem_statement = str(getattr(runtime_context, "problem_statement", "") or "").strip()
    task_requirements = str(getattr(runtime_context, "task_requirements", "") or "").strip()
    sample_identifier = str(getattr(runtime_context, "sample_identifier", "") or "").strip()

    factual_memory_run_label = str(getattr(config, "factual_memory_run_label", "")).strip()
    if not factual_memory_run_label:
        raise ValueError(f"{TOOL_NAME} requires factual_memory_run_label in config")
    factual_memory_dataset_name = str(getattr(config, "factual_memory_dataset_name", "")).strip()
    if not factual_memory_dataset_name:
        raise ValueError(f"{TOOL_NAME} requires factual_memory_dataset_name in config")

    return {
        "query": query,
        "search_provider": search_provider,
        "api_key": api_key,
        "task_id": task_id,
        "search_turn": search_turn,
        "problem_statement": problem_statement,
        "task_requirements": task_requirements,
        "sample_identifier": sample_identifier,
        "factual_memory_run_label": factual_memory_run_label,
        "factual_memory_dataset_name": factual_memory_dataset_name,
        "search_before": getattr(config, "search_before", None),
        "max_results": getattr(config, "search_max_results", 5),
        "max_chars_per_result": getattr(config, "search_max_chars_per_result", 700),
        "max_total_chars": getattr(config, "search_max_total_chars", 2500),
        "use_tavilty_raw_context": getattr(config, "use_tavilty_raw_context", False),
        "endpoint": getattr(
            config,
            "endpoint",
            FIRECRAWL_SEARCH_URL if search_provider == FIRECRAWL_SEARCH_PROVIDER else TAVILY_SEARCH_URL,
        ),
    }


def run_tool(
    *,
    query: str,
    search_provider: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    use_tavilty_raw_context: bool = False,
    endpoint: str = TAVILY_SEARCH_URL,
) -> str:
    normalized_search_provider = normalize_search_provider(search_provider)
    if normalized_search_provider == FIRECRAWL_SEARCH_PROVIDER:
        return _run_firecrawl_tool(
            query=query,
            api_key=api_key,
            task_id=task_id,
            search_turn=search_turn,
            problem_statement=problem_statement,
            task_requirements=task_requirements,
            sample_identifier=sample_identifier,
            factual_memory_run_label=factual_memory_run_label,
            factual_memory_dataset_name=factual_memory_dataset_name,
            search_before=search_before,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            max_total_chars=max_total_chars,
            endpoint=endpoint,
        )
    return _run_tavily_tool(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=use_tavilty_raw_context,
        endpoint=endpoint,
    )


async def run_tool_async(
    *,
    query: str,
    search_provider: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    use_tavilty_raw_context: bool = False,
    endpoint: str = TAVILY_SEARCH_URL,
) -> str:
    normalized_search_provider = normalize_search_provider(search_provider)
    if normalized_search_provider == FIRECRAWL_SEARCH_PROVIDER:
        return await _run_firecrawl_tool_async(
            query=query,
            api_key=api_key,
            task_id=task_id,
            search_turn=search_turn,
            problem_statement=problem_statement,
            task_requirements=task_requirements,
            sample_identifier=sample_identifier,
            factual_memory_run_label=factual_memory_run_label,
            factual_memory_dataset_name=factual_memory_dataset_name,
            search_before=search_before,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            max_total_chars=max_total_chars,
            endpoint=endpoint,
        )
    return await _run_tavily_tool_async(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=use_tavilty_raw_context,
        endpoint=endpoint,
    )


def _run_tavily_tool(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    use_tavilty_raw_context: bool = False,
    endpoint: str = TAVILY_SEARCH_URL,
) -> str:
    configured_keys, normalized_query, payload = _prepare_tavily_search_execution(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
        use_tavilty_raw_context=use_tavilty_raw_context,
    )

    with timed_block(
        "search.tavily",
        "search_with_fallback",
        kind="search",
        metadata={
            "provider": TAVILY_SEARCH_PROVIDER,
            "max_results": max_results,
            "has_search_before": bool(search_before),
            "use_tavilty_raw_context": use_tavilty_raw_context,
        },
    ):
        cache_metadata = _build_search_cache_metadata(
            provider=TAVILY_SEARCH_PROVIDER,
            endpoint=endpoint,
            payload=payload,
            dataset_type=factual_memory_dataset_name,
            sample_identifier=sample_identifier,
            search_turn=search_turn,
        )
        raw_results = _post_search_with_fallback(
            payload=payload,
            configured_keys=configured_keys,
            endpoint=endpoint,
            cache_metadata=cache_metadata,
        ).get("results", [])
    return _finalize_search_response(
        raw_results=raw_results,
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        query=normalized_query,
        search_before=search_before,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=use_tavilty_raw_context,
        content_source="raw_content (fallback to content)" if use_tavilty_raw_context else "content",
        log_tag="tavily",
    )


async def _run_tavily_tool_async(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    use_tavilty_raw_context: bool = False,
    endpoint: str = TAVILY_SEARCH_URL,
) -> str:
    configured_keys, normalized_query, payload = _prepare_tavily_search_execution(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
        use_tavilty_raw_context=use_tavilty_raw_context,
    )

    with timed_block(
        "search.tavily",
        "search_with_fallback",
        kind="search",
        metadata={
            "provider": TAVILY_SEARCH_PROVIDER,
            "max_results": max_results,
            "has_search_before": bool(search_before),
            "use_tavilty_raw_context": use_tavilty_raw_context,
        },
    ):
        cache_metadata = _build_search_cache_metadata(
            provider=TAVILY_SEARCH_PROVIDER,
            endpoint=endpoint,
            payload=payload,
            dataset_type=factual_memory_dataset_name,
            sample_identifier=sample_identifier,
            search_turn=search_turn,
        )
        raw_results = (
            await _post_search_with_fallback_async(
                payload=payload,
                configured_keys=configured_keys,
                endpoint=endpoint,
                cache_metadata=cache_metadata,
            )
        ).get("results", [])
    return _finalize_search_response(
        raw_results=raw_results,
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        query=normalized_query,
        search_before=search_before,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=use_tavilty_raw_context,
        content_source="raw_content (fallback to content)" if use_tavilty_raw_context else "content",
        log_tag="tavily",
    )


def _run_firecrawl_tool(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    endpoint: str = FIRECRAWL_SEARCH_URL,
) -> str:
    configured_keys, normalized_query, payload = _prepare_firecrawl_search_execution(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
    )
    with timed_block(
        "search.firecrawl",
        "search_with_fallback",
        kind="search",
        metadata={
            "provider": FIRECRAWL_SEARCH_PROVIDER,
            "max_results": max_results,
            "has_search_before": bool(search_before),
        },
    ):
        cache_metadata = _build_search_cache_metadata(
            provider=FIRECRAWL_SEARCH_PROVIDER,
            endpoint=endpoint,
            payload=payload,
            dataset_type=factual_memory_dataset_name,
            sample_identifier=sample_identifier,
            search_turn=search_turn,
        )
        response_payload = _post_firecrawl_search_with_fallback(
            payload=payload,
            configured_keys=configured_keys,
            endpoint=endpoint,
            cache_metadata=cache_metadata,
        )
        raw_results = _extract_firecrawl_results(response_payload)
    return _finalize_search_response(
        raw_results=_convert_firecrawl_results(raw_results),
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        query=normalized_query,
        search_before=search_before,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=False,
        content_source="markdown (fallback to description)",
        log_tag="firecrawl",
    )


async def _run_firecrawl_tool_async(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str] = None,
    max_results: int = 5,
    max_chars_per_result: int = 700,
    max_total_chars: int = 2500,
    endpoint: str = FIRECRAWL_SEARCH_URL,
) -> str:
    configured_keys, normalized_query, payload = _prepare_firecrawl_search_execution(
        query=query,
        api_key=api_key,
        task_id=task_id,
        search_turn=search_turn,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        search_before=search_before,
        max_results=max_results,
    )
    with timed_block(
        "search.firecrawl",
        "search_with_fallback",
        kind="search",
        metadata={
            "provider": FIRECRAWL_SEARCH_PROVIDER,
            "max_results": max_results,
            "has_search_before": bool(search_before),
        },
    ):
        cache_metadata = _build_search_cache_metadata(
            provider=FIRECRAWL_SEARCH_PROVIDER,
            endpoint=endpoint,
            payload=payload,
            dataset_type=factual_memory_dataset_name,
            sample_identifier=sample_identifier,
            search_turn=search_turn,
        )
        response_payload = await _post_firecrawl_search_with_fallback_async(
            payload=payload,
            configured_keys=configured_keys,
            endpoint=endpoint,
            cache_metadata=cache_metadata,
        )
        raw_results = _extract_firecrawl_results(response_payload)
    return _finalize_search_response(
        raw_results=_convert_firecrawl_results(raw_results),
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        query=normalized_query,
        search_before=search_before,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=False,
        content_source="markdown (fallback to description)",
        log_tag="firecrawl",
    )


def _prepare_tavily_search_execution(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str],
    max_results: int,
    use_tavilty_raw_context: bool,
) -> tuple[tuple[str, ...], str, dict[str, Any]]:
    configured_keys = _parse_search_api_keys(api_key, provider=TAVILY_SEARCH_PROVIDER)
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty query")
    if not task_id or not str(task_id).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty task_id")
    if search_turn < 1:
        raise TavilySearchError(f"{TOOL_NAME} requires search_turn >= 1")
    if not factual_memory_run_label or not str(factual_memory_run_label).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty factual_memory_run_label")
    if not factual_memory_dataset_name or not str(factual_memory_dataset_name).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty factual_memory_dataset_name")

    payload = _build_tavily_payload(
        normalized_query=normalized_query,
        search_before=search_before,
        max_results=max_results,
        use_tavilty_raw_context=use_tavilty_raw_context,
    )
    log_info("tavily", "Search start")
    log_block("tavily", "Search payload", format_json(payload))
    return configured_keys, normalized_query, payload


def _build_tavily_payload(
    *,
    normalized_query: str,
    search_before: Optional[str],
    max_results: int,
    use_tavilty_raw_context: bool,
) -> dict[str, Any]:
    truncated_query = normalized_query[:TAVILY_MAX_QUERY_LENGTH]
    payload = {
        "query": truncated_query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": use_tavilty_raw_context,
        "include_images": False,
        "include_favicon": False,
    }
    if search_before:
        payload["end_date"] = search_before
    return payload


def _prepare_firecrawl_search_execution(
    *,
    query: str,
    api_key: str,
    task_id: str,
    search_turn: int,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    search_before: Optional[str],
    max_results: int,
) -> tuple[tuple[str, ...], str, dict[str, Any]]:
    configured_keys = _parse_search_api_keys(api_key, provider=FIRECRAWL_SEARCH_PROVIDER)
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty query")
    if not task_id or not str(task_id).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty task_id")
    if search_turn < 1:
        raise TavilySearchError(f"{TOOL_NAME} requires search_turn >= 1")
    if not factual_memory_run_label or not str(factual_memory_run_label).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty factual_memory_run_label")
    if not factual_memory_dataset_name or not str(factual_memory_dataset_name).strip():
        raise TavilySearchError(f"{TOOL_NAME} requires a non-empty factual_memory_dataset_name")

    payload = _build_firecrawl_payload(
        normalized_query=normalized_query,
        search_before=search_before,
        max_results=max_results,
    )
    log_info("firecrawl", "Search start")
    log_block("firecrawl", "Search payload", format_json(payload))
    return configured_keys, normalized_query, payload


def _build_firecrawl_payload(
    *,
    normalized_query: str,
    search_before: Optional[str],
    max_results: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": normalized_query,
        "limit": max_results,
        "sources": ["web"],
        "timeout": 60000,
        "ignoreInvalidURLs": False,
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if search_before:
        tbs_value = _build_firecrawl_tbs(search_before)
        if tbs_value:
            payload["tbs"] = tbs_value
    return payload


def _build_search_cache_metadata(
    *,
    provider: str,
    endpoint: str,
    payload: dict[str, Any],
    dataset_type: str,
    sample_identifier: str,
    search_turn: int,
) -> dict[str, Any]:
    lossy_identity = build_lossy_search_identity(
        provider=provider,
        endpoint=endpoint,
        payload=payload,
        dataset_type=dataset_type,
        sample_identifier=sample_identifier,
        search_turn=search_turn,
    )
    return {
        "dataset_type": str(dataset_type or "").strip(),
        "sample_identifier": str(sample_identifier or "").strip(),
        "search_turn": int(search_turn),
        "query": str(payload.get("query", "") or "").strip(),
        "lossy_cache_key": compute_api_cache_key(lossy_identity),
    }


def resolve_lossy_cached_search_query(
    *,
    query: str,
    search_provider: str,
    search_before: Optional[str],
    max_results: int,
    use_tavilty_raw_context: bool,
    endpoint: Optional[str],
    factual_memory_dataset_name: str,
    sample_identifier: str,
    search_turn: int,
    enable_lossy_search_cache: bool,
) -> str | None:
    if not enable_lossy_search_cache:
        return None
    normalized_sample_identifier = str(sample_identifier or "").strip()
    if not normalized_sample_identifier or int(search_turn) < 1:
        return None
    normalized_provider = normalize_search_provider(search_provider)
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return None
    resolved_endpoint = str(
        endpoint
        or (FIRECRAWL_SEARCH_URL if normalized_provider == FIRECRAWL_SEARCH_PROVIDER else TAVILY_SEARCH_URL)
    ).strip()
    if normalized_provider == FIRECRAWL_SEARCH_PROVIDER:
        payload = _build_firecrawl_payload(
            normalized_query=normalized_query,
            search_before=search_before,
            max_results=max_results,
        )
    else:
        payload = _build_tavily_payload(
            normalized_query=normalized_query,
            search_before=search_before,
            max_results=max_results,
            use_tavilty_raw_context=use_tavilty_raw_context,
        )
    lossy_identity = build_lossy_search_identity(
        provider=normalized_provider,
        endpoint=resolved_endpoint,
        payload=payload,
        dataset_type=factual_memory_dataset_name,
        sample_identifier=normalized_sample_identifier,
        search_turn=search_turn,
    )
    entry = find_latest_lossy_search_cache_entry(lossy_identity)
    if not entry:
        return None
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return None
    cached_query = str(metadata.get("query", "") or "").strip()
    if cached_query and cached_query != normalized_query:
        record_search_lossy_cache_hit(normalized_provider)
    return cached_query or None


def _finalize_search_response(
    *,
    raw_results: list[dict[str, Any]],
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    query: str,
    search_before: Optional[str],
    max_chars_per_result: int,
    max_total_chars: int,
    use_tavilty_raw_context: bool,
    content_source: str,
    log_tag: str,
) -> str:
    normalized = _normalize_results(
        raw_results=raw_results,
        search_before=search_before,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        use_tavilty_raw_context=use_tavilty_raw_context,
    )
    log_block(
        log_tag,
        "Normalized search results",
        format_json([result.__dict__ for result in normalized]),
    )
    log_info(log_tag, f"Search end with {len(normalized)} normalized results")
    item_dirs = _write_raw_data_files(
        task_id=task_id,
        search_turn=search_turn,
        problem_statement=problem_statement,
        task_requirements=task_requirements,
        sample_identifier=sample_identifier,
        factual_memory_run_label=factual_memory_run_label,
        factual_memory_dataset_name=factual_memory_dataset_name,
        query=query,
        search_before=search_before,
        content_source=content_source,
        normalized=normalized,
    )

    return json.dumps(
        {
            "normalized": [asdict(result) for result in normalized],
            "item_dirs": [str(path) for path in item_dirs],
            "content_source": content_source,
        },
        ensure_ascii=False,
    )


def _post_search(*, payload: dict[str, Any], api_key: str, endpoint: str) -> dict[str, Any]:
    try:
        record_search_actual_call(TAVILY_SEARCH_PROVIDER)
        with timed_block(
            "search.tavily_http",
            "http_request",
            kind="search",
            metadata={
                "endpoint": endpoint,
                "max_results": payload.get("max_results"),
            },
        ):
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                response.raise_for_status()
                return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise TavilySearchError(
            f"Tavily search failed with HTTP {exc.response.status_code}: {detail}",
            http_status=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise TavilySearchError(f"Tavily search failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TavilySearchError("Tavily search returned invalid JSON") from exc


async def _post_search_async(*, payload: dict[str, Any], api_key: str, endpoint: str) -> dict[str, Any]:
    semaphore = _get_async_tavily_semaphore()
    try:
        async with semaphore:
            record_search_actual_call(TAVILY_SEARCH_PROVIDER)
            with timed_block(
                "search.tavily_http",
                "http_request",
                kind="search",
                metadata={
                    "endpoint": endpoint,
                    "max_results": payload.get("max_results"),
                },
            ):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        endpoint,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    response.raise_for_status()
                    return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise TavilySearchError(
            f"Tavily search failed with HTTP {exc.response.status_code}: {detail}",
            http_status=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise TavilySearchError(f"Tavily search failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TavilySearchError("Tavily search returned invalid JSON") from exc


def _post_firecrawl_search(*, payload: dict[str, Any], api_key: str, endpoint: str) -> dict[str, Any]:
    try:
        record_search_actual_call(FIRECRAWL_SEARCH_PROVIDER)
        with timed_block(
            "search.firecrawl_http",
            "http_request",
            kind="search",
            metadata={
                "endpoint": endpoint,
                "limit": payload.get("limit"),
            },
        ):
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                response.raise_for_status()
                return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise TavilySearchError(
            f"Firecrawl search failed with HTTP {exc.response.status_code}: {detail}",
            http_status=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise TavilySearchError(f"Firecrawl search failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TavilySearchError("Firecrawl search returned invalid JSON") from exc


async def _post_firecrawl_search_async(
    *,
    payload: dict[str, Any],
    api_key: str,
    endpoint: str,
) -> dict[str, Any]:
    semaphore = _get_async_tavily_semaphore()
    try:
        async with semaphore:
            record_search_actual_call(FIRECRAWL_SEARCH_PROVIDER)
            with timed_block(
                "search.firecrawl_http",
                "http_request",
                kind="search",
                metadata={
                    "endpoint": endpoint,
                    "limit": payload.get("limit"),
                },
            ):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        endpoint,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    response.raise_for_status()
                    return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise TavilySearchError(
            f"Firecrawl search failed with HTTP {exc.response.status_code}: {detail}",
            http_status=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise TavilySearchError(f"Firecrawl search failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TavilySearchError("Firecrawl search returned invalid JSON") from exc


def _parse_search_api_keys(raw_value: Any, *, provider: str) -> tuple[str, ...]:
    parts = [str(part).strip() for part in str(raw_value or "").split(",")]
    configured_keys: list[str] = []
    seen_keys: set[str] = set()
    for part in parts:
        if not part or part in seen_keys:
            continue
        configured_keys.append(part)
        seen_keys.add(part)

    if not configured_keys:
        raise TavilySearchError(f"{TOOL_NAME} requires at least one {provider} API key")
    return tuple(configured_keys)


def _get_tavily_key_pool(configured_keys: tuple[str, ...]) -> _TavilyKeyPool:
    with _TAVILY_KEY_POOLS_LOCK:
        pool = _TAVILY_KEY_POOLS.get(configured_keys)
        if pool is None:
            pool = _TavilyKeyPool(configured_keys)
            _TAVILY_KEY_POOLS[configured_keys] = pool
        return pool


def _snapshot_active_tavily_keys(pool: _TavilyKeyPool) -> list[str]:
    with pool.lock:
        return list(pool.active_keys)


def _retire_tavily_key(pool: _TavilyKeyPool, api_key: str) -> tuple[bool, int]:
    with pool.lock:
        if api_key not in pool.active_keys:
            return False, len(pool.active_keys)
        pool.active_keys = [candidate for candidate in pool.active_keys if candidate != api_key]
        return True, len(pool.active_keys)


def _post_search_with_fallback(
    *,
    payload: dict[str, Any],
    configured_keys: tuple[str, ...],
    endpoint: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pool = _get_tavily_key_pool(configured_keys)
    active_keys = _snapshot_active_tavily_keys(pool)
    if not active_keys:
        raise TavilySearchError(
            "No active Tavily API keys remain after previous HTTP 432 failures."
        )

    failures: list[tuple[int, TavilySearchError]] = []
    total_keys = len(active_keys)
    for key_index, candidate_key in enumerate(active_keys, start=1):
        log_info(
            "tavily",
            (
                f"Search key attempt | key_index={key_index}/{total_keys} | "
                f"active_keys_snapshot={total_keys}"
            ),
        )
        try:
            response = _post_search_with_same_key_429_retries(
                payload=payload,
                api_key=candidate_key,
                endpoint=endpoint,
                key_index=key_index,
                total_keys=total_keys,
                cache_metadata=cache_metadata,
            )
            if key_index > 1:
                log_info(
                    "tavily",
                    f"Search succeeded after fallback | key_index={key_index}/{total_keys}",
                )
            return response
        except TavilySearchError as exc:
            failures.append((key_index, exc))
            if _is_tavily_usage_limit_error(exc):
                retired, remaining_count = _retire_tavily_key(pool, candidate_key)
                if retired:
                    log_info(
                        "tavily",
                        (
                            f"Retired Tavily API key due to HTTP 432 | "
                            f"key_index={key_index}/{total_keys} | active_keys_remaining={remaining_count}"
                        ),
                    )
            if key_index < total_keys:
                log_info(
                    "tavily",
                    (
                        f"Search key failed | key_index={key_index}/{total_keys} | "
                        f"error={exc} | trying_next_key=True"
                    ),
                )

    raise _build_tavily_exhausted_error(failures)


async def _post_search_with_fallback_async(
    *,
    payload: dict[str, Any],
    configured_keys: tuple[str, ...],
    endpoint: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pool = _get_tavily_key_pool(configured_keys)
    active_keys = _snapshot_active_tavily_keys(pool)
    if not active_keys:
        raise TavilySearchError(
            "No active Tavily API keys remain after previous HTTP 432 failures."
        )

    failures: list[tuple[int, TavilySearchError]] = []
    total_keys = len(active_keys)
    for key_index, candidate_key in enumerate(active_keys, start=1):
        log_info(
            "tavily",
            (
                f"Search key attempt | key_index={key_index}/{total_keys} | "
                f"active_keys_snapshot={total_keys}"
            ),
        )
        try:
            response = await _post_search_with_same_key_429_retries_async(
                payload=payload,
                api_key=candidate_key,
                endpoint=endpoint,
                key_index=key_index,
                total_keys=total_keys,
                cache_metadata=cache_metadata,
            )
            if key_index > 1:
                log_info(
                    "tavily",
                    f"Search succeeded after fallback | key_index={key_index}/{total_keys}",
                )
            return response
        except TavilySearchError as exc:
            failures.append((key_index, exc))
            if _is_tavily_usage_limit_error(exc):
                retired, remaining_count = _retire_tavily_key(pool, candidate_key)
                if retired:
                    log_info(
                        "tavily",
                        (
                            f"Retired Tavily API key due to HTTP 432 | "
                            f"key_index={key_index}/{total_keys} | active_keys_remaining={remaining_count}"
                        ),
                    )
            if key_index < total_keys:
                log_info(
                    "tavily",
                    (
                        f"Search key failed | key_index={key_index}/{total_keys} | "
                        f"error={exc} | trying_next_key=True"
                    ),
                )

    raise _build_tavily_exhausted_error(failures)


def _is_tavily_usage_limit_error(exc: TavilySearchError) -> bool:
    http_status = getattr(exc, "http_status", None)
    if http_status == 432:
        return True
    return "Tavily search failed with HTTP 432" in str(exc)


def _is_tavily_too_many_requests_error(exc: TavilySearchError) -> bool:
    http_status = getattr(exc, "http_status", None)
    if http_status == 429:
        return True
    return "Tavily search failed with HTTP 429" in str(exc)


def _post_search_with_same_key_429_retries(
    *,
    payload: dict[str, Any],
    api_key: str,
    endpoint: str,
    key_index: int,
    total_keys: int,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_identity = build_search_request_identity(
        provider=TAVILY_SEARCH_PROVIDER,
        endpoint=endpoint,
        payload=payload,
    )
    cache_key, cache_entry = load_search_cache_entry(request_identity)
    if cache_entry is not None:
        record_search_cache_hit(TAVILY_SEARCH_PROVIDER)
        log_info("tavily", f"Search cache hit | cache_key={cache_key}")
        return dict(cache_entry.get("response", {}) or {})
    log_info("tavily", f"Search cache miss | cache_key={cache_key}")
    last_error: TavilySearchError | None = None
    for retry_attempt in range(1, TAVILY_HTTP_429_MAX_ATTEMPTS + 1):
        try:
            response = _post_search(payload=payload, api_key=api_key, endpoint=endpoint)
            store_search_cache_entry(
                cache_key=cache_key,
                request_identity=request_identity,
                raw_response_payload=response,
                metadata=cache_metadata,
            )
            return response
        except TavilySearchError as exc:
            last_error = exc
            if not _is_tavily_too_many_requests_error(exc):
                raise
            if retry_attempt >= TAVILY_HTTP_429_MAX_ATTEMPTS:
                raise
            log_info(
                "tavily",
                (
                    f"Received HTTP 429 | key_index={key_index}/{total_keys} | "
                    f"retry_attempt={retry_attempt}/{TAVILY_HTTP_429_MAX_ATTEMPTS} | "
                    f"sleep_seconds={TAVILY_HTTP_429_RETRY_DELAY_SECONDS}"
                ),
            )
            time.sleep(TAVILY_HTTP_429_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise last_error
    raise TavilySearchError("Unreachable Tavily retry state")


async def _post_search_with_same_key_429_retries_async(
    *,
    payload: dict[str, Any],
    api_key: str,
    endpoint: str,
    key_index: int,
    total_keys: int,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_identity = build_search_request_identity(
        provider=TAVILY_SEARCH_PROVIDER,
        endpoint=endpoint,
        payload=payload,
    )
    cache_key, cache_entry = load_search_cache_entry(request_identity)
    if cache_entry is not None:
        record_search_cache_hit(TAVILY_SEARCH_PROVIDER)
        log_info("tavily", f"Search cache hit | cache_key={cache_key}")
        return dict(cache_entry.get("response", {}) or {})
    log_info("tavily", f"Search cache miss | cache_key={cache_key}")
    last_error: TavilySearchError | None = None
    for retry_attempt in range(1, TAVILY_HTTP_429_MAX_ATTEMPTS + 1):
        try:
            response = await _post_search_async(payload=payload, api_key=api_key, endpoint=endpoint)
            store_search_cache_entry(
                cache_key=cache_key,
                request_identity=request_identity,
                raw_response_payload=response,
                metadata=cache_metadata,
            )
            return response
        except TavilySearchError as exc:
            last_error = exc
            if not _is_tavily_too_many_requests_error(exc):
                raise
            if retry_attempt >= TAVILY_HTTP_429_MAX_ATTEMPTS:
                raise
            log_info(
                "tavily",
                (
                    f"Received HTTP 429 | key_index={key_index}/{total_keys} | "
                    f"retry_attempt={retry_attempt}/{TAVILY_HTTP_429_MAX_ATTEMPTS} | "
                    f"sleep_seconds={TAVILY_HTTP_429_RETRY_DELAY_SECONDS}"
                ),
            )
            await asyncio.sleep(TAVILY_HTTP_429_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise last_error
    raise TavilySearchError("Unreachable Tavily retry state")


def _build_tavily_exhausted_error(
    failures: list[tuple[int, TavilySearchError]],
) -> TavilySearchError:
    if not failures:
        return TavilySearchError("All configured Tavily API keys failed.")

    reasons = "; ".join(f"key#{key_index}: {exc}" for key_index, exc in failures)
    if all(_is_tavily_usage_limit_error(exc) for _, exc in failures):
        return TavilySearchError(
            f"All configured Tavily API keys are exhausted due to HTTP 432 failures: {reasons}"
        )
    return TavilySearchError(f"All configured Tavily API keys failed: {reasons}")


def _post_firecrawl_search_with_fallback(
    *,
    payload: dict[str, Any],
    configured_keys: tuple[str, ...],
    endpoint: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[tuple[int, TavilySearchError]] = []
    total_keys = len(configured_keys)
    for key_index, candidate_key in enumerate(configured_keys, start=1):
        log_info(
            "firecrawl",
            f"Search key attempt | key_index={key_index}/{total_keys}",
        )
        try:
            return _post_firecrawl_search_with_same_key_429_retries(
                payload=payload,
                api_key=candidate_key,
                endpoint=endpoint,
                key_index=key_index,
                total_keys=total_keys,
                cache_metadata=cache_metadata,
            )
        except TavilySearchError as exc:
            failures.append((key_index, exc))
            if key_index < total_keys:
                log_info(
                    "firecrawl",
                    (
                        f"Search key failed | key_index={key_index}/{total_keys} | "
                        f"error={exc} | trying_next_key=True"
                    ),
                )
    return _build_provider_exhausted_error(FIRECRAWL_SEARCH_PROVIDER, failures)


async def _post_firecrawl_search_with_fallback_async(
    *,
    payload: dict[str, Any],
    configured_keys: tuple[str, ...],
    endpoint: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[tuple[int, TavilySearchError]] = []
    total_keys = len(configured_keys)
    for key_index, candidate_key in enumerate(configured_keys, start=1):
        log_info(
            "firecrawl",
            f"Search key attempt | key_index={key_index}/{total_keys}",
        )
        try:
            return await _post_firecrawl_search_with_same_key_429_retries_async(
                payload=payload,
                api_key=candidate_key,
                endpoint=endpoint,
                key_index=key_index,
                total_keys=total_keys,
                cache_metadata=cache_metadata,
            )
        except TavilySearchError as exc:
            failures.append((key_index, exc))
            if key_index < total_keys:
                log_info(
                    "firecrawl",
                    (
                        f"Search key failed | key_index={key_index}/{total_keys} | "
                        f"error={exc} | trying_next_key=True"
                    ),
                )
    return _build_provider_exhausted_error(FIRECRAWL_SEARCH_PROVIDER, failures)


def _is_firecrawl_too_many_requests_error(exc: TavilySearchError) -> bool:
    http_status = getattr(exc, "http_status", None)
    if http_status == 429:
        return True
    return "Firecrawl search failed with HTTP 429" in str(exc)


def _post_firecrawl_search_with_same_key_429_retries(
    *,
    payload: dict[str, Any],
    api_key: str,
    endpoint: str,
    key_index: int,
    total_keys: int,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_identity = build_search_request_identity(
        provider=FIRECRAWL_SEARCH_PROVIDER,
        endpoint=endpoint,
        payload=payload,
    )
    cache_key, cache_entry = load_search_cache_entry(request_identity)
    if cache_entry is not None:
        record_search_cache_hit(FIRECRAWL_SEARCH_PROVIDER)
        log_info("firecrawl", f"Search cache hit | cache_key={cache_key}")
        return dict(cache_entry.get("response", {}) or {})
    log_info("firecrawl", f"Search cache miss | cache_key={cache_key}")
    last_error: TavilySearchError | None = None
    for retry_attempt in range(1, FIRECRAWL_HTTP_429_MAX_ATTEMPTS + 1):
        try:
            response = _post_firecrawl_search(payload=payload, api_key=api_key, endpoint=endpoint)
            store_search_cache_entry(
                cache_key=cache_key,
                request_identity=request_identity,
                raw_response_payload=response,
                metadata=cache_metadata,
            )
            return response
        except TavilySearchError as exc:
            last_error = exc
            if not _is_firecrawl_too_many_requests_error(exc):
                raise
            if retry_attempt >= FIRECRAWL_HTTP_429_MAX_ATTEMPTS:
                raise
            log_info(
                "firecrawl",
                (
                    f"Received HTTP 429 | key_index={key_index}/{total_keys} | "
                    f"retry_attempt={retry_attempt}/{FIRECRAWL_HTTP_429_MAX_ATTEMPTS} | "
                    f"sleep_seconds={FIRECRAWL_HTTP_429_RETRY_DELAY_SECONDS}"
                ),
            )
            time.sleep(FIRECRAWL_HTTP_429_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise last_error
    raise TavilySearchError("Unreachable Firecrawl retry state")


async def _post_firecrawl_search_with_same_key_429_retries_async(
    *,
    payload: dict[str, Any],
    api_key: str,
    endpoint: str,
    key_index: int,
    total_keys: int,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_identity = build_search_request_identity(
        provider=FIRECRAWL_SEARCH_PROVIDER,
        endpoint=endpoint,
        payload=payload,
    )
    cache_key, cache_entry = load_search_cache_entry(request_identity)
    if cache_entry is not None:
        record_search_cache_hit(FIRECRAWL_SEARCH_PROVIDER)
        log_info("firecrawl", f"Search cache hit | cache_key={cache_key}")
        return dict(cache_entry.get("response", {}) or {})
    log_info("firecrawl", f"Search cache miss | cache_key={cache_key}")
    last_error: TavilySearchError | None = None
    for retry_attempt in range(1, FIRECRAWL_HTTP_429_MAX_ATTEMPTS + 1):
        try:
            response = await _post_firecrawl_search_async(payload=payload, api_key=api_key, endpoint=endpoint)
            store_search_cache_entry(
                cache_key=cache_key,
                request_identity=request_identity,
                raw_response_payload=response,
                metadata=cache_metadata,
            )
            return response
        except TavilySearchError as exc:
            last_error = exc
            if not _is_firecrawl_too_many_requests_error(exc):
                raise
            if retry_attempt >= FIRECRAWL_HTTP_429_MAX_ATTEMPTS:
                raise
            log_info(
                "firecrawl",
                (
                    f"Received HTTP 429 | key_index={key_index}/{total_keys} | "
                    f"retry_attempt={retry_attempt}/{FIRECRAWL_HTTP_429_MAX_ATTEMPTS} | "
                    f"sleep_seconds={FIRECRAWL_HTTP_429_RETRY_DELAY_SECONDS}"
                ),
            )
            await asyncio.sleep(FIRECRAWL_HTTP_429_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise last_error
    raise TavilySearchError("Unreachable Firecrawl retry state")


def _build_provider_exhausted_error(
    provider: str,
    failures: list[tuple[int, TavilySearchError]],
) -> TavilySearchError:
    provider_label = "Tavily" if provider == TAVILY_SEARCH_PROVIDER else "Firecrawl"
    if not failures:
        return TavilySearchError(f"All configured {provider_label} API keys failed.")
    reasons = "; ".join(f"key#{key_index}: {exc}" for key_index, exc in failures)
    return TavilySearchError(f"All configured {provider_label} API keys failed: {reasons}")


def _extract_firecrawl_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TavilySearchError("Firecrawl search returned a non-object payload")
    if payload.get("success") is False:
        raise TavilySearchError(f"Firecrawl search failed: {payload}")
    data = payload.get("data", {})
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        web_results = data.get("web", [])
        if isinstance(web_results, list):
            return [item for item in web_results if isinstance(item, dict)]
    raise TavilySearchError("Firecrawl search returned an unexpected response shape")


def _convert_firecrawl_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in raw_results:
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        converted.append(
            {
                "title": item.get("title") or metadata.get("title") or "",
                "url": item.get("url") or metadata.get("sourceURL") or metadata.get("url") or "",
                "content": item.get("markdown") or item.get("description") or "",
                "raw_content": item.get("markdown") or item.get("description") or "",
                "published_date": (
                    item.get("date")
                    or metadata.get("publishedDate")
                    or metadata.get("published_date")
                    or ""
                ),
                "score": item.get("score"),
            }
        )
    return converted


def _build_firecrawl_tbs(search_before: str) -> str:
    cutoff = _parse_date(search_before)
    if cutoff is None:
        return ""
    return f"cdr:1,cd_min:1/1/1900,cd_max:{cutoff.month}/{cutoff.day}/{cutoff.year}"


def _normalize_results(
    *,
    raw_results: list[dict[str, Any]],
    search_before: Optional[str],
    max_chars_per_result: int,
    max_total_chars: int,
    use_tavilty_raw_context: bool = False,
) -> list[_SearchResult]:
    cutoff_date = _parse_date(search_before) if search_before else None
    remaining_chars = max_total_chars
    results: list[_SearchResult] = []

    for item in raw_results:
        published_date = _extract_date(item)
        if cutoff_date and published_date:
            item_date = _parse_date(published_date)
            if item_date and item_date > cutoff_date:
                continue

        raw_or_content = (
            item.get("raw_content") or item.get("content") or ""
            if use_tavilty_raw_context
            else item.get("content") or item.get("raw_content") or ""
        )
        content = _clean_text(raw_or_content)
        if max_chars_per_result > 0:
            content = content[:max_chars_per_result].rstrip()
        if remaining_chars <= 0:
            content = ""
        elif len(content) > remaining_chars:
            content = content[:remaining_chars].rstrip()
        remaining_chars -= len(content)

        result = _SearchResult(
            title=_clean_text(item.get("title") or ""),
            url=_clean_text(item.get("url") or ""),
            content=content,
            published_date=published_date,
            score=item.get("score"),
        )
        if result.title or result.url or result.content:
            results.append(result)

        if remaining_chars <= 0:
            break

    return results


def _write_raw_data_files(
    *,
    task_id: str,
    search_turn: int,
    problem_statement: str,
    task_requirements: str,
    sample_identifier: str,
    factual_memory_run_label: str,
    factual_memory_dataset_name: str,
    query: str,
    search_before: Optional[str],
    content_source: str,
    normalized: list[_SearchResult],
) -> list[Path]:
    safe_run_label = _sanitize_path_component(factual_memory_run_label, "factual_memory_run_label")
    safe_dataset_name = _sanitize_path_component(
        factual_memory_dataset_name,
        "factual_memory_dataset_name",
    )
    safe_task_id = _sanitize_task_id(task_id)
    run_dir_name = f"{safe_run_label}-{safe_dataset_name}"
    factual_memory_root = resolve_factual_memory_dir() / run_dir_name / safe_task_id
    factual_memory_root.mkdir(parents=True, exist_ok=True)

    item_dirs: list[Path] = []
    for item_rank, item in enumerate(normalized, start=1):
        item_dir = factual_memory_root / f"search-turn-{search_turn}-item-{item_rank}"
        item_dir.mkdir(parents=True, exist_ok=True)
        item_dirs.append(item_dir)

        raw_data = {
            "task_id": task_id,
            "sample_identifier": sample_identifier,
            "run_label": factual_memory_run_label,
            "dataset_name": factual_memory_dataset_name,
            "run_dir_name": run_dir_name,
            "task_dir_name": safe_task_id,
            "search_turn": search_turn,
            "result_count": len(normalized),
            "problem_statement": problem_statement,
            "task_requirements": task_requirements,
            "item_rank": item_rank,
            "query": query,
            "search_before": search_before,
            "content_source": content_source,
            "item": {
                "title": item.title,
                "url": item.url,
                "content": item.content,
                "published_date": item.published_date,
                "score": item.score,
            },
        }
        raw_data_path = item_dir / "raw_data.json"
        raw_data_path.write_text(
            json.dumps(raw_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return item_dirs


def _clean_text(text: Any) -> str:
    return " ".join(str(text).split())


def _extract_date(item: dict[str, Any]) -> Optional[str]:
    for key in ("published_date", "published_at", "date", "publish_date", "last_updated"):
        value = item.get(key)
        if value:
            return str(value)[:10]
    return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    normalized = str(value).strip()[:10]
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None


def _sanitize_task_id(task_id: str) -> str:
    return _sanitize_path_component(task_id, "task_id")


def _sanitize_path_component(value: str, field_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise TavilySearchError(f"Could not derive a safe directory name from {field_name}={value!r}")
    return sanitized


def _get_async_tavily_semaphore() -> asyncio.Semaphore:
    running_loop = asyncio.get_running_loop()
    with _ASYNC_TAVILY_SEMAPHORES_LOCK:
        semaphore = _ASYNC_TAVILY_SEMAPHORES.get(running_loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(ASYNC_TAVILY_CONCURRENCY_LIMIT)
            _ASYNC_TAVILY_SEMAPHORES[running_loop] = semaphore
        return semaphore


__all__ = ["TOOL_NAME", "IS_PUBLIC_TOOL", "TOOL_SPEC", "build_runner_kwargs", "run_tool", "run_tool_async"]
