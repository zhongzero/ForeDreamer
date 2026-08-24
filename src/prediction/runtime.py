#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.shared_llm_client import PromptStrategyConfig
from experience_bank import compute_experience_bank_hash, load_current_experience_bank
from prediction.datasets.futurex import load_futurex_dataset
from prediction.datasets.prophet_arena import load_events_from_csv
from prediction.runners.futurex import FutureXRunner, process_futurex_event_async, process_futurex_events_async
from prediction.runners.prophet_arena import (
    ProphetArenaRunner,
    process_events_async,
    process_prophet_arena_event_async,
)
from utils.search_provider import normalize_search_provider
FACTUAL_MEMORY_RUN_LABEL_ENV = "FACTUAL_MEMORY_RUN_LABEL"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATHS = {
    "prophet_arena": REPO_ROOT / "data" / "Prophet-arena" / "subset_data_1200.csv",
    "futurex": REPO_ROOT / "data" / "FutureX" / "train.parquet",
}


def ensure_factual_memory_run_label(run_label: Optional[str] = None) -> str:
    normalized = str(run_label or "").strip()
    if normalized:
        os.environ[FACTUAL_MEMORY_RUN_LABEL_ENV] = normalized
        return normalized

    existing = str(os.getenv(FACTUAL_MEMORY_RUN_LABEL_ENV, "")).strip()
    if existing:
        return existing

    generated = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ[FACTUAL_MEMORY_RUN_LABEL_ENV] = generated
    return generated


def build_prompt_strategy_config(
    *,
    dataset_type: str,
    max_turns: int = 4,
    subagent_max_turns: int = 10,
    save_rollout: bool = False,
    search_provider: str = "tavily",
    search_max_results: int = 5,
    search_max_chars_per_result: int = 700,
    search_max_total_chars: int = 2500,
    search_api_key: Optional[str] = None,
    use_tavilty_raw_context: bool = False,
    enable_lossy_search_cache: bool = False,
    disable_main_agent_final_answer_cache: bool = False,
    factual_memory_run_label: Optional[str] = None,
    mem_guide: Optional[str] = None,
    experience_entries: Optional[list[dict[str, Any]]] = None,
    experience_bank_hash: Optional[str] = None,
    experience_bank_version_id: Optional[str] = None,
    enable_experience_bank: bool = True,
) -> PromptStrategyConfig:
    resolved_run_label = ensure_factual_memory_run_label(factual_memory_run_label)
    normalized_search_provider = normalize_search_provider(search_provider)
    if not enable_experience_bank:
        return PromptStrategyConfig(
            max_turns=max_turns,
            subagent_max_turns=subagent_max_turns,
            save_rollout=save_rollout,
            search_provider=normalized_search_provider,
            search_max_results=search_max_results,
            search_max_chars_per_result=search_max_chars_per_result,
            search_max_total_chars=search_max_total_chars,
            search_api_key=search_api_key,
            use_tavilty_raw_context=use_tavilty_raw_context,
            enable_lossy_search_cache=enable_lossy_search_cache,
            disable_main_agent_final_answer_cache=disable_main_agent_final_answer_cache,
            factual_memory_run_label=resolved_run_label,
            factual_memory_dataset_name=dataset_type,
            mem_guide=mem_guide,
            experience_entries=tuple(),
            experience_bank_hash=None,
            experience_bank_version_id=None,
        )
    resolved_experience_entries = experience_entries
    resolved_experience_bank_hash = str(experience_bank_hash or "").strip() or None
    resolved_experience_bank_version_id = str(experience_bank_version_id or "").strip() or None
    if resolved_experience_entries is not None and resolved_experience_bank_hash is None:
        resolved_experience_bank_hash = compute_experience_bank_hash(resolved_experience_entries)
    if (
        resolved_experience_entries is None
        or resolved_experience_bank_hash is None
        or resolved_experience_bank_version_id is None
    ):
        latest_bank_payload = load_current_experience_bank(dataset_type)
        if resolved_experience_entries is None:
            resolved_experience_entries = list(latest_bank_payload.get("experiences", []))
        if resolved_experience_bank_hash is None:
            resolved_experience_bank_hash = str(latest_bank_payload.get("bank_hash", "") or "").strip() or None
        if resolved_experience_bank_version_id is None:
            resolved_experience_bank_version_id = (
                str(latest_bank_payload.get("version_id", "") or "").strip() or None
            )
    return PromptStrategyConfig(
        max_turns=max_turns,
        subagent_max_turns=subagent_max_turns,
        save_rollout=save_rollout,
        search_provider=normalized_search_provider,
        search_max_results=search_max_results,
        search_max_chars_per_result=search_max_chars_per_result,
        search_max_total_chars=search_max_total_chars,
        search_api_key=search_api_key,
        use_tavilty_raw_context=use_tavilty_raw_context,
        enable_lossy_search_cache=enable_lossy_search_cache,
        disable_main_agent_final_answer_cache=disable_main_agent_final_answer_cache,
        factual_memory_run_label=resolved_run_label,
        factual_memory_dataset_name=dataset_type,
        mem_guide=mem_guide,
        experience_entries=tuple(resolved_experience_entries or ()),
        experience_bank_hash=resolved_experience_bank_hash,
        experience_bank_version_id=resolved_experience_bank_version_id,
    )


def resolve_dataset_path(dataset_type: str, input_path: Optional[str] = None) -> Path:
    if input_path:
        return Path(input_path)
    try:
        return DEFAULT_DATASET_PATHS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}") from exc


def load_dataset_records(
    dataset_type: str,
    input_path: str,
    run_specific: Optional[str] = None,
) -> List[Dict]:
    if dataset_type == "futurex":
        return load_futurex_dataset(input_path, run_specific)
    if dataset_type == "prophet_arena":
        return load_events_from_csv(input_path, run_specific)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def build_dataset_runner(
    dataset_type: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
) -> Any:
    if dataset_type == "futurex":
        return FutureXRunner(model=model, base_url=base_url, api_key=api_key)
    if dataset_type == "prophet_arena":
        return ProphetArenaRunner(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


async def run_dataset_events_async(
    dataset_type: str,
    events_data: List[Dict],
    runner: Any,
    *,
    strategy: str,
    strategy_config: Optional[PromptStrategyConfig] = None,
    use_sources: bool = False,
    use_market_data: bool = False,
) -> List[Dict]:
    if dataset_type == "futurex":
        return await process_futurex_events_async(
            events_data,
            runner,
            strategy=strategy,
            strategy_config=strategy_config,
        )
    if dataset_type == "prophet_arena":
        return await process_events_async(
            events_data,
            runner,
            strategy=strategy,
            strategy_config=strategy_config,
            use_sources=use_sources,
            use_market_data=use_market_data,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def run_dataset_events_sync(
    dataset_type: str,
    events_data: List[Dict],
    runner: Any,
    *,
    strategy: str,
    strategy_config: Optional[PromptStrategyConfig] = None,
    use_sources: bool = False,
    use_market_data: bool = False,
) -> List[Dict]:
    return asyncio.run(
        run_dataset_events_async(
            dataset_type,
            events_data,
            runner,
            strategy=strategy,
            strategy_config=strategy_config,
            use_sources=use_sources,
            use_market_data=use_market_data,
        )
    )


def run_single_event_with_rollout_sync(
    dataset_type: str,
    event_data: Dict,
    runner: Any,
    *,
    strategy: str,
    strategy_config: Optional[PromptStrategyConfig] = None,
    use_sources: bool = False,
    use_market_data: bool = False,
    execution_id: Optional[str] = None,
    factual_memory_run_label_override: Optional[str] = None,
) -> Dict[str, Any]:
    base_config = strategy_config or PromptStrategyConfig()
    forced_config = replace(
        base_config,
        save_rollout=True,
        factual_memory_run_label=(
            factual_memory_run_label_override
            if factual_memory_run_label_override is not None
            else base_config.factual_memory_run_label
        ),
    )

    async def _run_single() -> Dict[str, Any]:
        if dataset_type == "futurex":
            return await process_futurex_event_async(
                event_data=event_data,
                runner=runner,
                strategy=strategy,
                strategy_config=forced_config,
                should_save_rollout=True,
                execution_id_override=execution_id,
            )
        if dataset_type == "prophet_arena":
            return await process_prophet_arena_event_async(
                event_data=event_data,
                predictor=runner,
                strategy=strategy,
                strategy_config=forced_config,
                use_sources=use_sources,
                use_market_data=use_market_data,
                should_save_rollout=True,
                execution_id_override=execution_id,
            )
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    result = asyncio.run(_run_single())
    rollout_path = str(result.get("rollout_path", "") or "").strip()
    if not rollout_path:
        raise ValueError("Expected a saved rollout_path for the single-event execution")

    raw_rollout = result.get("rollout")
    if raw_rollout:
        rollout = json.loads(raw_rollout)
    else:
        rollout = json.loads(Path(rollout_path).read_text(encoding="utf-8"))

    return {
        "result": result,
        "rollout_path": rollout_path,
        "rollout": rollout,
    }
