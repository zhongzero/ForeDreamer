#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Callable


SRC_DIR = Path(__file__).resolve().parents[1]

FACTUAL_MEMORY_DIR_ENV = "FACTUAL_MEMORY_DIR"
HISTORY_EVOLUTION_DIR_ENV = "HISTORY_EVOLUTION_DIR"
HISTORY_ROLLOUT_DIR_ENV = "HISTORY_ROLLOUT_DIR"
HISTORY_EXPERIENCE_EVOLUTION_DIR_ENV = "HISTORY_EXPERIENCE_EVOLUTION_DIR"
MEMGUIDE_DIR_ENV = "MEMGUIDE_DIR"
MEMTOOL_DIR_ENV = "MEMTOOL_DIR"
EXPERIENCE_BANK_DIR_ENV = "EXPERIENCE_BANK_DIR"
CACHE_DIR_ENV = "CACHE_DIR"
ENABLE_API_CACHE_ENV = "ENABLE_API_CACHE"
GUIDE_AND_TOOL_HISTORY_SUBDIR = "guide_and_tool"
EXPERIENCE_HISTORY_SUBDIR = "experience"

DEFAULT_FACTUAL_MEMORY_DIR = SRC_DIR / "FactualMemory"
DEFAULT_HISTORY_EVOLUTION_DIR = SRC_DIR / "HistoryEvolution"
DEFAULT_HISTORY_GUIDE_AND_TOOL_EVOLUTION_DIR = (
    DEFAULT_HISTORY_EVOLUTION_DIR / GUIDE_AND_TOOL_HISTORY_SUBDIR
)
DEFAULT_HISTORY_ROLLOUT_DIR = SRC_DIR / "HistoryRollout"
DEFAULT_HISTORY_EXPERIENCE_EVOLUTION_DIR = DEFAULT_HISTORY_EVOLUTION_DIR / EXPERIENCE_HISTORY_SUBDIR
DEFAULT_MEMGUIDE_DIR = SRC_DIR / "MemGuide"
DEFAULT_MEMTOOL_DIR = SRC_DIR / "MemTool"
DEFAULT_EXPERIENCE_BANK_DIR = SRC_DIR / "ExperienceBank"
DEFAULT_CACHE_DIR = SRC_DIR / "cache"
DEFAULT_GUIDE_INITIAL_PATH = DEFAULT_MEMGUIDE_DIR / "guide_initial.json"
DEFAULT_TOOL_INITIAL_PATH = DEFAULT_MEMTOOL_DIR / "tool_initial.py"


def _normalize_path(raw_value: Any, *, default_path: Path) -> Path:
    normalized = str(raw_value or "").strip()
    if not normalized:
        return default_path

    candidate = Path(normalized).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_directory(env_name: str, *, default_path: Path) -> Path:
    return _normalize_path(os.getenv(env_name, ""), default_path=default_path)


def _infer_save_dir_from_generated_paths() -> Path | None:
    candidate_envs = (
        EXPERIENCE_BANK_DIR_ENV,
        FACTUAL_MEMORY_DIR_ENV,
        HISTORY_EVOLUTION_DIR_ENV,
        HISTORY_ROLLOUT_DIR_ENV,
        MEMGUIDE_DIR_ENV,
        MEMTOOL_DIR_ENV,
    )
    for env_name in candidate_envs:
        raw_value = str(os.getenv(env_name, "") or "").strip()
        if not raw_value:
            continue
        try:
            resolved = _normalize_path(raw_value, default_path=DEFAULT_CACHE_DIR)
        except Exception:
            continue
        return resolved.parent
    return None


def resolve_factual_memory_dir() -> Path:
    return _resolve_directory(
        FACTUAL_MEMORY_DIR_ENV,
        default_path=DEFAULT_FACTUAL_MEMORY_DIR,
    )


def resolve_history_evolution_dir() -> Path:
    return _resolve_directory(
        HISTORY_EVOLUTION_DIR_ENV,
        default_path=DEFAULT_HISTORY_EVOLUTION_DIR,
    )


def resolve_history_guide_and_tool_evolution_dir() -> Path:
    return resolve_history_evolution_dir() / GUIDE_AND_TOOL_HISTORY_SUBDIR


def resolve_history_rollout_dir() -> Path:
    return _resolve_directory(
        HISTORY_ROLLOUT_DIR_ENV,
        default_path=DEFAULT_HISTORY_ROLLOUT_DIR,
    )


def resolve_history_experience_evolution_dir() -> Path:
    raw_value = os.getenv(HISTORY_EXPERIENCE_EVOLUTION_DIR_ENV, "")
    if str(raw_value or "").strip():
        return _normalize_path(
            raw_value,
            default_path=DEFAULT_HISTORY_EXPERIENCE_EVOLUTION_DIR,
        )
    return resolve_history_evolution_dir() / EXPERIENCE_HISTORY_SUBDIR


def resolve_memguide_dir() -> Path:
    return _resolve_directory(
        MEMGUIDE_DIR_ENV,
        default_path=DEFAULT_MEMGUIDE_DIR,
    )


def resolve_memtool_dir() -> Path:
    return _resolve_directory(
        MEMTOOL_DIR_ENV,
        default_path=DEFAULT_MEMTOOL_DIR,
    )


def resolve_experience_bank_dir() -> Path:
    return _resolve_directory(
        EXPERIENCE_BANK_DIR_ENV,
        default_path=DEFAULT_EXPERIENCE_BANK_DIR,
    )


def resolve_cache_dir() -> Path:
    raw_value = str(os.getenv(CACHE_DIR_ENV, "") or "").strip()
    if raw_value:
        return _normalize_path(raw_value, default_path=DEFAULT_CACHE_DIR)
    inferred_save_dir = _infer_save_dir_from_generated_paths()
    if inferred_save_dir is not None:
        return inferred_save_dir / "cache"
    return DEFAULT_CACHE_DIR


def _normalize_bool_flag(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def is_api_cache_enabled(raw_value: Any | None = None) -> bool:
    if raw_value is not None:
        return _normalize_bool_flag(raw_value)
    return _normalize_bool_flag(os.getenv(ENABLE_API_CACHE_ENV, ""))


def is_experience_bank_disabled(raw_value: Any | None = None) -> bool:
    if raw_value is not None:
        return not str(raw_value).strip()
    return EXPERIENCE_BANK_DIR_ENV in os.environ and not str(os.environ.get(EXPERIENCE_BANK_DIR_ENV, "")).strip()


class DynamicPath(os.PathLike[str]):
    def __init__(self, resolver: Callable[[], Path]):
        self._resolver = resolver

    def _path(self) -> Path:
        return self._resolver()

    @property
    def path(self) -> Path:
        return self._path()

    def __fspath__(self) -> str:
        return os.fspath(self._path())

    def __str__(self) -> str:
        return str(self._path())

    def __repr__(self) -> str:
        return f"DynamicPath({self._path()!r})"

    def __truediv__(self, other: Any) -> Path:
        return self._path() / other

    def __rtruediv__(self, other: Any) -> Path:
        return Path(other) / self._path()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._path(), name)


def dynamic_path(resolver: Callable[[], Path]) -> DynamicPath:
    return DynamicPath(resolver)


def configure_generated_path_env(
    *,
    factual_memory_dir: str | None = None,
    history_evolution_dir: str | None = None,
    history_rollout_dir: str | None = None,
    memguide_dir: str | None = None,
    memtool_dir: str | None = None,
    experience_bank_dir: str | None = None,
    cache_dir: str | None = None,
    enable_api_cache: bool | None = None,
    enable_lossy_search_cache: bool | None = None,
) -> dict[str, str]:
    configured: dict[str, str] = {}
    values = (
        (FACTUAL_MEMORY_DIR_ENV, factual_memory_dir, DEFAULT_FACTUAL_MEMORY_DIR),
        (HISTORY_EVOLUTION_DIR_ENV, history_evolution_dir, DEFAULT_HISTORY_EVOLUTION_DIR),
        (HISTORY_ROLLOUT_DIR_ENV, history_rollout_dir, DEFAULT_HISTORY_ROLLOUT_DIR),
        (MEMGUIDE_DIR_ENV, memguide_dir, DEFAULT_MEMGUIDE_DIR),
        (MEMTOOL_DIR_ENV, memtool_dir, DEFAULT_MEMTOOL_DIR),
        (EXPERIENCE_BANK_DIR_ENV, experience_bank_dir, DEFAULT_EXPERIENCE_BANK_DIR),
        (CACHE_DIR_ENV, cache_dir, DEFAULT_CACHE_DIR),
    )
    for env_name, raw_value, default_path in values:
        if raw_value is None:
            continue
        if env_name == EXPERIENCE_BANK_DIR_ENV and is_experience_bank_disabled(raw_value):
            os.environ[env_name] = ""
            configured[env_name] = ""
            continue
        resolved = _normalize_path(raw_value, default_path=default_path)
        os.environ[env_name] = str(resolved)
        configured[env_name] = str(resolved)
    effective_enable_api_cache: bool | None = None
    if enable_api_cache is not None or enable_lossy_search_cache is not None:
        effective_enable_api_cache = bool(enable_api_cache) or bool(enable_lossy_search_cache)
    if effective_enable_api_cache is not None:
        normalized_flag = "1" if effective_enable_api_cache else "0"
        os.environ[ENABLE_API_CACHE_ENV] = normalized_flag
        configured[ENABLE_API_CACHE_ENV] = normalized_flag
    return configured


def configure_generated_path_env_from_namespace(args: Any) -> dict[str, str]:
    return configure_generated_path_env(
        factual_memory_dir=getattr(args, "factual_memory_dir", None),
        history_evolution_dir=getattr(args, "history_evolution_dir", None),
        history_rollout_dir=getattr(args, "history_rollout_dir", None),
        memguide_dir=getattr(args, "memguide_dir", None),
        memtool_dir=getattr(args, "memtool_dir", None),
        experience_bank_dir=getattr(args, "experience_bank_dir", None),
        cache_dir=getattr(args, "cache_dir", None),
        enable_api_cache=getattr(args, "enable_api_cache", None),
        enable_lossy_search_cache=getattr(args, "enable_lossy_search_cache", None),
    )


def ensure_mem_asset_bootstrap(
    *,
    memguide_dir: Path | None = None,
    memtool_dir: Path | None = None,
) -> dict[str, str]:
    resolved_memguide_dir = Path(memguide_dir) if memguide_dir is not None else resolve_memguide_dir()
    resolved_memtool_dir = Path(memtool_dir) if memtool_dir is not None else resolve_memtool_dir()

    resolved_memguide_dir.mkdir(parents=True, exist_ok=True)
    resolved_memtool_dir.mkdir(parents=True, exist_ok=True)

    actions: dict[str, str] = {}
    guide_initial_target = resolved_memguide_dir / DEFAULT_GUIDE_INITIAL_PATH.name
    tool_initial_target = resolved_memtool_dir / DEFAULT_TOOL_INITIAL_PATH.name

    if not guide_initial_target.exists():
        if not DEFAULT_GUIDE_INITIAL_PATH.exists():
            raise ValueError(f"Missing default guide bootstrap file: {DEFAULT_GUIDE_INITIAL_PATH}")
        if guide_initial_target.resolve() != DEFAULT_GUIDE_INITIAL_PATH.resolve():
            shutil.copy2(DEFAULT_GUIDE_INITIAL_PATH, guide_initial_target)
            actions["guide_initial"] = str(guide_initial_target)

    if not tool_initial_target.exists():
        if not DEFAULT_TOOL_INITIAL_PATH.exists():
            raise ValueError(f"Missing default tool bootstrap file: {DEFAULT_TOOL_INITIAL_PATH}")
        if tool_initial_target.resolve() != DEFAULT_TOOL_INITIAL_PATH.resolve():
            shutil.copy2(DEFAULT_TOOL_INITIAL_PATH, tool_initial_target)
            actions["tool_initial"] = str(tool_initial_target)

    return actions


def ensure_mem_asset_bootstrap_from_namespace(args: Any) -> dict[str, str]:
    return ensure_mem_asset_bootstrap(
        memguide_dir=_normalize_path(
            getattr(args, "memguide_dir", None),
            default_path=DEFAULT_MEMGUIDE_DIR,
        ),
        memtool_dir=_normalize_path(
            getattr(args, "memtool_dir", None),
            default_path=DEFAULT_MEMTOOL_DIR,
        ),
    )


def add_generated_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--factual_memory_dir",
        default=None,
        help=f"Directory for FactualMemory outputs (default: {DEFAULT_FACTUAL_MEMORY_DIR})",
    )
    parser.add_argument(
        "--history_evolution_dir",
        default=None,
        help=(
            "Root directory for evolution history "
            f"(guide/tool records default to {DEFAULT_HISTORY_GUIDE_AND_TOOL_EVOLUTION_DIR})"
        ),
    )
    parser.add_argument(
        "--history_rollout_dir",
        default=None,
        help=f"Directory for HistoryRollout records (default: {DEFAULT_HISTORY_ROLLOUT_DIR})",
    )
    parser.add_argument(
        "--memguide_dir",
        default=None,
        help=f"Directory for MemGuide assets (default: {DEFAULT_MEMGUIDE_DIR})",
    )
    parser.add_argument(
        "--memtool_dir",
        default=None,
        help=f"Directory for MemTool assets (default: {DEFAULT_MEMTOOL_DIR})",
    )
    parser.add_argument(
        "--experience_bank_dir",
        default=None,
        help=f"Directory for ExperienceBank assets (default: {DEFAULT_EXPERIENCE_BANK_DIR})",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=f"Directory for shared API cache assets (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--enable_api_cache",
        action="store_true",
        help="Enable shared API caching under --cache_dir for LLM and search requests.",
    )
