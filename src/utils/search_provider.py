#!/usr/bin/env python3

from __future__ import annotations

from typing import Final


TAVILY_SEARCH_PROVIDER: Final[str] = "tavily"
FIRECRAWL_SEARCH_PROVIDER: Final[str] = "firecrawl"
DEFAULT_SEARCH_PROVIDER: Final[str] = TAVILY_SEARCH_PROVIDER
SUPPORTED_SEARCH_PROVIDERS: Final[tuple[str, ...]] = (
    TAVILY_SEARCH_PROVIDER,
    FIRECRAWL_SEARCH_PROVIDER,
)


def normalize_search_provider(raw_value: str | None) -> str:
    normalized = str(raw_value or "").strip().lower() or DEFAULT_SEARCH_PROVIDER
    if normalized not in SUPPORTED_SEARCH_PROVIDERS:
        supported = ", ".join(SUPPORTED_SEARCH_PROVIDERS)
        raise ValueError(
            f"Unsupported search provider: {raw_value!r}. Expected one of: {supported}"
        )
    return normalized


def default_search_api_env_var(provider: str) -> str:
    normalized = normalize_search_provider(provider)
    if normalized == TAVILY_SEARCH_PROVIDER:
        return "TAVILY_API_KEY"
    if normalized == FIRECRAWL_SEARCH_PROVIDER:
        return "FIRECRAWL_API_KEY"
    raise AssertionError(f"Unhandled search provider: {normalized}")


__all__ = [
    "DEFAULT_SEARCH_PROVIDER",
    "FIRECRAWL_SEARCH_PROVIDER",
    "SUPPORTED_SEARCH_PROVIDERS",
    "TAVILY_SEARCH_PROVIDER",
    "default_search_api_env_var",
    "normalize_search_provider",
]
