#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Any

from experience_bank import (
    build_candidate_experience_bank,
    create_experience_attempt_record,
    load_current_experience_bank,
    save_experience_bank_version,
    write_experience_attempt_record,
)
from prediction.runtime import build_dataset_runner, run_single_event_with_rollout_sync
from utils.logger import log_info
from utils.timing_registry import push_timing_context, timed_block
from MemTool.registry import reload_tool_registry
from SelfEvolving.evolve_critic import (
    build_critic_prompt_and_lengths,
    parse_critique_response,
    request_llm_response,
)
from SelfEvolving.evolve_guide_summary import (
    build_guide_category_representatives_bundle,
    categorize_tree_guide,
    record_invalid_guide,
    sync_guide_summary,
)
from SelfEvolving.evolve_experience import (
    build_experience_evolution_prompt_and_lengths,
    build_experience_evolution_summary,
    is_candidate_validation_better,
    parse_experience_evolution_response,
    validate_experience_bank_candidate,
)
from SelfEvolving.evolve_storage import (
    append_tree_child_locked,
    build_tool_definition_bundle,
    create_attempt_record,
    iso_now,
    load_current_tree,
    load_guide_object,
    load_tool_file_map,
    load_tool_source_bundle,
    load_validation_results,
    mark_tools_invalid,
    resolve_tool_files,
    sync_tool_summary,
    write_attempt_record,
)
from SelfEvolving.evolve_summary import build_rollout_summary
from SelfEvolving.evolve_validation import (
    NO_ELIGIBLE_BEST_GUIDE_SELECTION_ERROR,
    NoEligibleBestGuideSelectionError,
    UNIFORM_RANDOM_SELECTION,
    ZIPF_BY_VALIDATION_RANK_SELECTION,
    build_validation_key,
    choose_best_tree_node,
    choose_tree_node,
    ensure_tree_validation_results,
    get_validation_result_entry_for_experience,
    is_validation_entry_valid,
    validate_guide_if_needed,
)
from SelfEvolving.generate_memguide_and_memtool import (
    LLMConfig,
    generate_assets_from_design_requirement,
    generate_assets_from_exploration_context,
    select_reusable_tools_from_design_requirement,
)


@dataclass(frozen=True)
class EvolutionRuntime:
    dataset_type: str
    train_data_path: Path
    train_records: list[dict[str, Any]]
    val_data_path: Path | None
    val_records: list[dict[str, Any]] | None
    llm_config: LLMConfig
    critic_prompt_path: Path
    generation_prompt_path: Path
    experience_evolution_prompt_path: Path
    reuse_selection_prompt_path: Path
    guide_classification_prompt_path: Path
    exploration_prompt_path: Path
    base_strategy_config: Any
    use_sources: bool
    use_market_data: bool
    summary_max_chars: int
    experience_max_suggestions: int
    validation_key: str | None
    validation_key_payload: dict[str, Any] | None
    optimization_reuse_duplicated_tool_enabled: bool
    optimization_encourage_exploration_enabled: bool
    exploration_over_expansion: str | None


@dataclass(frozen=True)
class CurrentValidationContext:
    experience_bank_payload: dict[str, Any]
    strategy_config: Any
    validation_key: str | None
    validation_key_payload: dict[str, Any] | None
    validation_results: dict[str, Any] | None


def choose_training_sample(dataset_type: str, records: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No training records available for dataset_type={dataset_type}")
    return rng.choice(records)


def sample_identifier(dataset_type: str, event_data: dict[str, Any]) -> str:
    if dataset_type == "futurex":
        return str(event_data.get("id", "") or "")
    return str(event_data.get("submission_id", "") or "")


def sample_title(dataset_type: str, event_data: dict[str, Any]) -> str:
    if dataset_type == "futurex":
        return str(event_data.get("title", "") or "")
    return str(event_data.get("title", "") or "")


def derive_execution_id(attempt_index: int) -> str:
    return f"attempt_{attempt_index}"


def derive_attempt_run_label(base_run_label: str, execution_id: str) -> str:
    normalized_base = str(base_run_label or "").strip()
    normalized_execution_id = str(execution_id or "").strip()
    if not normalized_base:
        return normalized_execution_id
    if not normalized_execution_id:
        return normalized_base
    return f"{normalized_base}-{normalized_execution_id}"


def _validation_enabled(runtime: EvolutionRuntime) -> bool:
    return runtime.val_data_path is not None and runtime.val_records is not None


def _resolve_current_validation_context(
    *,
    runtime: EvolutionRuntime,
    tree: dict[str, Any] | None = None,
) -> CurrentValidationContext:
    experience_bank_payload = load_current_experience_bank(runtime.dataset_type)
    strategy_config = replace(
        runtime.base_strategy_config,
        experience_entries=tuple(experience_bank_payload.get("experiences", [])),
        experience_bank_hash=str(experience_bank_payload.get("bank_hash", "") or "").strip() or None,
        experience_bank_version_id=str(experience_bank_payload.get("version_id", "") or "").strip() or None,
    )

    if not _validation_enabled(runtime):
        return CurrentValidationContext(
            experience_bank_payload=experience_bank_payload,
            strategy_config=strategy_config,
            validation_key=None,
            validation_key_payload=None,
            validation_results=None,
        )

    resolved_tree = tree if tree is not None else load_current_tree()
    validation_key, validation_key_payload = build_validation_key(
        dataset_type=runtime.dataset_type,
        val_data_path=runtime.val_data_path or runtime.train_data_path,
        llm_config=runtime.llm_config,
        base_strategy_config=strategy_config,
        use_sources=runtime.use_sources,
        use_market_data=runtime.use_market_data,
    )
    validation_results = ensure_tree_validation_results(
        tree=resolved_tree,
        dataset_type=runtime.dataset_type,
        val_records=runtime.val_records or [],
        llm_config=runtime.llm_config,
        base_strategy_config=strategy_config,
        use_sources=runtime.use_sources,
        use_market_data=runtime.use_market_data,
        validation_key=validation_key,
        validation_key_payload=validation_key_payload,
    )
    return CurrentValidationContext(
        experience_bank_payload=experience_bank_payload,
        strategy_config=strategy_config,
        validation_key=validation_key,
        validation_key_payload=validation_key_payload,
        validation_results=validation_results,
    )


def format_attempt_record(attempt_record: dict[str, Any], attempt_path: Path | None) -> dict[str, Any]:
    generation = attempt_record.get("generation") if isinstance(attempt_record.get("generation"), dict) else None
    critique = attempt_record.get("critique") if isinstance(attempt_record.get("critique"), dict) else None
    prompt_lengths = attempt_record.get("prompt_lengths")
    if not isinstance(prompt_lengths, dict):
        prompt_lengths = {
            "critic": attempt_record.get("critic_prompt_lengths"),
            "generation": generation.get("prompt_lengths") if generation else None,
        }

    parent_guide_file = attempt_record.get("selected_guide_file")
    parent_guide_name = attempt_record.get("selected_guide_name")
    parent_tool_names = list(attempt_record.get("selected_tool_names") or [])
    parent_tool_files = list(attempt_record.get("selected_tool_files") or [])

    child_guide_file = generation.get("guide_file") if generation else None
    child_guide_name = generation.get("guide_name") if generation else None
    child_tool_names = list(generation.get("tool_names") or []) if generation else []
    child_tool_files = list(generation.get("tool_files") or []) if generation else []

    return {
        "attempt_index": attempt_record.get("attempt_index"),
        "iteration": attempt_record.get("iteration"),
        "execution_id": attempt_record.get("execution_id"),
        "created_at": attempt_record.get("created_at"),
        "status": attempt_record.get("status"),
        "dataset_type": attempt_record.get("dataset_type"),
        "train_data_path": attempt_record.get("train_data_path"),
        "val_data_path": attempt_record.get("val_data_path"),
        "validation_enabled": attempt_record.get("validation_enabled"),
        "optimization_reuse_duplicated_tool_enabled": attempt_record.get(
            "optimization_reuse_duplicated_tool_enabled"
        ),
        "optimization_encourage_exploration_enabled": attempt_record.get(
            "optimization_encourage_exploration_enabled"
        ),
        "exploration_over_expansion": attempt_record.get("exploration_over_expansion"),
        "evolution_mode": attempt_record.get("evolution_mode"),
        "guide_selection_strategy": attempt_record.get("guide_selection_strategy"),
        "sample_identifier": attempt_record.get("sample_identifier"),
        "sample_title": attempt_record.get("sample_title"),
        "experience_bank_version_id": attempt_record.get("experience_bank_version_id"),
        "experience_bank_hash": attempt_record.get("experience_bank_hash"),
        "selected_guide_validation": attempt_record.get("selected_guide_validation"),
        "tree_relation": {
            "parent_guide_file": parent_guide_file,
            "parent_guide_name": parent_guide_name,
            "parent_tool_names": parent_tool_names,
            "parent_tool_files": parent_tool_files,
            "child_guide_file": child_guide_file,
            "child_guide_name": child_guide_name,
            "child_tool_names": child_tool_names,
            "child_tool_files": child_tool_files,
            "source_rollout_file": attempt_record.get("rollout_file"),
            "source_attempt_file": attempt_path.name if attempt_path is not None else None,
        },
        "rollout_file": attempt_record.get("rollout_file"),
        "rollout_path": attempt_record.get("rollout_path"),
        "prompt_lengths": prompt_lengths,
        "processed_rollout_summary": attempt_record.get("processed_rollout_summary"),
        "run_result": attempt_record.get("run_result"),
        "critique": critique,
        "generation": generation,
        "error": attempt_record.get("error"),
    }


def _create_base_attempt_record(
    *,
    iteration: int,
    runtime: EvolutionRuntime,
    validation_enabled: bool,
    evolution_mode: str,
) -> dict[str, Any]:
    return {
        "created_at": iso_now(),
        "iteration": iteration,
        "execution_id": None,
        "dataset_type": runtime.dataset_type,
        "train_data_path": str(runtime.train_data_path),
        "val_data_path": str(runtime.val_data_path) if runtime.val_data_path is not None else None,
        "validation_enabled": validation_enabled,
        "optimization_reuse_duplicated_tool_enabled": runtime.optimization_reuse_duplicated_tool_enabled,
        "optimization_encourage_exploration_enabled": runtime.optimization_encourage_exploration_enabled,
        "exploration_over_expansion": runtime.exploration_over_expansion,
        "evolution_mode": evolution_mode,
        "guide_selection_strategy": (
            ZIPF_BY_VALIDATION_RANK_SELECTION if validation_enabled and evolution_mode == "rollout_expansion"
            else (UNIFORM_RANDOM_SELECTION if evolution_mode == "rollout_expansion" else None)
        ),
        "selected_guide_file": None,
        "selected_guide_name": None,
        "selected_tool_names": [],
        "selected_tool_files": [],
        "sample_identifier": None,
        "sample_title": None,
        "experience_bank_version_id": None,
        "experience_bank_hash": None,
        "selected_guide_validation": None,
        "rollout_file": None,
        "rollout_path": None,
        "prompt_lengths": {
            "critic": None,
            "experience_evolution": None,
            "reuse_tool_selection": None,
            "generation": None,
            "guide_category": None,
        },
        "processed_rollout_summary": "",
        "run_result": None,
        "critique": None,
        "generation": None,
        "status": "running",
        "error": None,
    }


def format_experience_attempt_record(attempt_record: dict[str, Any], attempt_path: Path | None) -> dict[str, Any]:
    prompt_lengths = attempt_record.get("prompt_lengths")
    if not isinstance(prompt_lengths, dict):
        prompt_lengths = {"experience_evolution": None}

    return {
        "attempt_index": attempt_record.get("attempt_index"),
        "iteration": attempt_record.get("iteration"),
        "execution_id": attempt_record.get("execution_id"),
        "created_at": attempt_record.get("created_at"),
        "status": attempt_record.get("status"),
        "dataset_type": attempt_record.get("dataset_type"),
        "train_data_path": attempt_record.get("train_data_path"),
        "val_data_path": attempt_record.get("val_data_path"),
        "validation_enabled": attempt_record.get("validation_enabled"),
        "evolution_mode": attempt_record.get("evolution_mode"),
        "guide_selection_strategy": attempt_record.get("guide_selection_strategy"),
        "sample_identifier": attempt_record.get("sample_identifier"),
        "sample_title": attempt_record.get("sample_title"),
        "experience_bank_version_id": attempt_record.get("experience_bank_version_id"),
        "experience_bank_hash": attempt_record.get("experience_bank_hash"),
        "selected_guide_file": attempt_record.get("selected_guide_file"),
        "selected_guide_name": attempt_record.get("selected_guide_name"),
        "selected_tool_names": list(attempt_record.get("selected_tool_names") or []),
        "selected_tool_files": list(attempt_record.get("selected_tool_files") or []),
        "selected_guide_validation": attempt_record.get("selected_guide_validation"),
        "rollout_file": attempt_record.get("rollout_file"),
        "rollout_path": attempt_record.get("rollout_path"),
        "prompt_lengths": prompt_lengths,
        "processed_rollout_summary": attempt_record.get("processed_rollout_summary"),
        "run_result": attempt_record.get("run_result"),
        "experience_evolution": attempt_record.get("experience_evolution"),
        "error": attempt_record.get("error"),
        "source_attempt_file": attempt_path.name if attempt_path is not None else None,
    }


def _create_base_experience_attempt_record(
    *,
    iteration: int,
    runtime: EvolutionRuntime,
    validation_enabled: bool,
) -> dict[str, Any]:
    return {
        "created_at": iso_now(),
        "iteration": iteration,
        "execution_id": None,
        "dataset_type": runtime.dataset_type,
        "train_data_path": str(runtime.train_data_path),
        "val_data_path": str(runtime.val_data_path) if runtime.val_data_path is not None else None,
        "validation_enabled": validation_enabled,
        "evolution_mode": "experience_evolution",
        "guide_selection_strategy": None,
        "sample_identifier": None,
        "sample_title": None,
        "experience_bank_version_id": None,
        "experience_bank_hash": None,
        "selected_guide_file": None,
        "selected_guide_name": None,
        "selected_tool_names": [],
        "selected_tool_files": [],
        "selected_guide_validation": None,
        "rollout_file": None,
        "rollout_path": None,
        "prompt_lengths": {
            "experience_evolution": None,
        },
        "processed_rollout_summary": "",
        "run_result": None,
        "experience_evolution": None,
        "status": "running",
        "error": None,
    }


def _finalize_generation(
    *,
    iteration: int,
    total_iterations: int,
    execution_id: str,
    evolution_mode: str,
    attempt_path: Path,
    attempt_record: dict[str, Any],
    generation_summary: dict[str, Any],
    runtime: EvolutionRuntime,
    validation_enabled: bool,
    current_strategy_config: Any,
    current_validation_key: str | None,
    current_validation_key_payload: dict[str, Any] | None,
    rollout_path: Path | None = None,
    selected_guide_file: str = "",
    sample_id: str = "",
    reuse_tool_selection_record: dict[str, Any] | None = None,
    exploration_from_guide_file: list[str] | None = None,
) -> None:
    with timed_block(
        "self_evolving.attempt_phase",
        "finalize_generation",
        kind="phase",
        metadata={"evolution_mode": evolution_mode},
    ):
        with timed_block(
            "self_evolving.attempt_phase",
            "reload_registry_for_generation",
            kind="phase",
        ):
            reload_tool_registry()
            tool_file_map = load_tool_file_map()
        if runtime.optimization_reuse_duplicated_tool_enabled:
            with timed_block(
                "self_evolving.attempt_phase",
                "sync_tool_summary_for_generation",
                kind="phase",
            ):
                sync_tool_summary()

        generation_record = {
            "guide_file": Path(generation_summary["guide_path"]).name,
            "guide_path": generation_summary["guide_path"],
            "guide_name": generation_summary["guide_name"],
            "tool_files": [Path(path).name for path in generation_summary["tool_paths"]],
            "tool_paths": generation_summary["tool_paths"],
            "tool_names": generation_summary["tool_names"],
            "design_requirement": generation_summary.get("design_requirement"),
            "prompt_path": generation_summary["prompt_path"],
            "prompt_lengths": generation_summary["prompt_lengths"],
            "response_text": generation_summary["response_text"],
            "reusable_tool_entries": generation_summary.get("reusable_tool_entries", []),
            "reusable_tool_source_bundle": generation_summary.get("reusable_tool_source_bundle", ""),
            "reuse_tool_selection": reuse_tool_selection_record,
            "guide_category_representatives_bundle": generation_summary.get(
                "guide_category_representatives_bundle",
                "",
            ),
            "exploration_from_guide_file": list(exploration_from_guide_file) if exploration_from_guide_file else None,
            "validation": None,
            "guide_category_assignment": None,
        }
        attempt_record["generation"] = generation_record
        attempt_record["prompt_lengths"]["generation"] = generation_summary["prompt_lengths"]

        with push_timing_context(new_guide_file=generation_record["guide_file"]):
            if validation_enabled:
                with timed_block(
                    "self_evolving.attempt_phase",
                    "validate_generated_guide",
                    kind="phase",
                    metadata={"guide_file": generation_record["guide_file"]},
                ):
                    validated_entry = validate_guide_if_needed(
                        guide_file=generation_record["guide_file"],
                        guide_name=generation_record["guide_name"],
                        dataset_type=runtime.dataset_type,
                        val_records=runtime.val_records or [],
                        llm_config=runtime.llm_config,
                        base_strategy_config=current_strategy_config,
                        use_sources=runtime.use_sources,
                        use_market_data=runtime.use_market_data,
                        validation_key=current_validation_key or "",
                        validation_key_payload=current_validation_key_payload or {},
                    )
                generation_record["validation"] = validated_entry
                if not is_validation_entry_valid(validated_entry):
                    record_invalid_guide(
                        guide_file=generation_record["guide_file"],
                        guide_name=generation_record["guide_name"],
                        validation_entry=validated_entry,
                        source_attempt_file=attempt_path.name,
                        source_validation_key=current_validation_key,
                    )
                    if generation_record["tool_names"]:
                        mark_tools_invalid(
                            tool_names=generation_record["tool_names"],
                            invalid_reason="generated_under_invalid_guide",
                            source_guide_file=generation_record["guide_file"],
                            source_attempt_file=attempt_path.name,
                            source_validation_key=current_validation_key,
                        )
                    attempt_record["status"] = "validation_failed"
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                    log_info(
                        "self_evolving",
                        (
                            f"Iteration end | iteration={iteration}/{total_iterations} | "
                            f"execution_id={execution_id} | mode={evolution_mode} | "
                            f"guide={selected_guide_file or 'none'} | sample_id={sample_id or 'none'} "
                            f"| evolved=false | validation_failed=true | new_guide={generation_record['guide_file']}"
                        ),
                    )
                    return

            with timed_block(
                "self_evolving.attempt_phase",
                "append_tree_child",
                kind="phase",
                metadata={"guide_file": generation_record["guide_file"]},
            ):
                child_node = append_tree_child_locked(
                    parent_guide_file=selected_guide_file or None,
                    child_guide_file=generation_record["guide_file"],
                    source_rollout_file=rollout_path.name if rollout_path is not None else None,
                    source_attempt_file=attempt_path.name,
                    new_tool_files=generation_record["tool_files"],
                    tool_file_map=tool_file_map,
                    exploration_from_guide_file=exploration_from_guide_file,
                )

            with timed_block(
                "self_evolving.attempt_phase",
                "guide_category_assignment",
                kind="phase",
                metadata={"guide_file": generation_record["guide_file"]},
            ):
                guide_category_assignment = categorize_tree_guide(
                    guide_file=generation_record["guide_file"],
                    llm_config=runtime.llm_config,
                    prompt_path=runtime.guide_classification_prompt_path,
                    validation_results=load_validation_results() if validation_enabled else None,
                    validation_key=current_validation_key,
                )
            generation_record["guide_category_assignment"] = guide_category_assignment
            attempt_record["prompt_lengths"]["guide_category"] = guide_category_assignment.get("prompt_lengths")

            attempt_record["status"] = "evolved"
            attempt_record["generation"]["tree_node"] = child_node
            write_attempt_record(attempt_path, attempt_record, format_attempt_record)
            log_info(
                "self_evolving",
                (
                    f"Iteration end | iteration={iteration}/{total_iterations} | "
                    f"execution_id={execution_id} | mode={evolution_mode} | "
                    f"guide={selected_guide_file or 'none'} | sample_id={sample_id or 'none'} "
                    f"| evolved=true | new_guide={generation_record['guide_file']}"
                ),
            )


def run_single_evolution_attempt(
    *,
    iteration: int,
    total_iterations: int,
    seed: int,
    runtime: EvolutionRuntime,
) -> None:
    local_rng = random.Random(seed + iteration)
    selected_guide_file = ""
    sample_id = ""
    execution_id = ""
    validation_enabled = _validation_enabled(runtime)
    attempt_path: Path | None = None
    attempt_record = _create_base_attempt_record(
        iteration=iteration,
        runtime=runtime,
        validation_enabled=validation_enabled,
        evolution_mode="rollout_expansion",
    )

    attempt_path, attempt_record = create_attempt_record(attempt_record, format_attempt_record)
    execution_id = derive_execution_id(int(attempt_record["attempt_index"]))
    attempt_record["execution_id"] = execution_id
    write_attempt_record(attempt_path, attempt_record, format_attempt_record)

    with push_timing_context(
        iteration=iteration,
        total_iterations=total_iterations,
        execution_id=execution_id,
        evolution_mode="rollout_expansion",
        attempt_index=attempt_record.get("attempt_index"),
        attempt_file=attempt_path.name if attempt_path is not None else None,
    ):
        with timed_block(
            "self_evolving.attempt",
            "rollout_expansion_total",
            kind="attempt",
            metadata={"status": "running"},
        ) as attempt_scope:
            try:
                with timed_block(
                    "self_evolving.attempt_phase",
                    "state_preparation",
                    kind="phase",
                ):
                    reload_tool_registry()
                    tool_file_map = load_tool_file_map()
                    tree = load_current_tree()
                    current_context = _resolve_current_validation_context(
                        runtime=runtime,
                        tree=tree,
                    )
                    current_validation_results = current_context.validation_results
                    selected_guide_file, guide_selection_strategy, selected_guide_validation = choose_tree_node(
                        tree=tree,
                        dataset_type=runtime.dataset_type,
                        rng=local_rng,
                        validation_enabled=validation_enabled,
                        validation_results=current_validation_results,
                        validation_key=current_context.validation_key,
                    )
                    selected_guide = load_guide_object(selected_guide_file)
                    selected_tool_names = [str(name) for name in selected_guide.get("tool_names", [])]
                    selected_tool_files = resolve_tool_files(selected_tool_names, tool_file_map)
                    event_data = choose_training_sample(runtime.dataset_type, runtime.train_records, local_rng)
                    sample_id = sample_identifier(runtime.dataset_type, event_data)

                attempt_record.update(
                    {
                        "guide_selection_strategy": guide_selection_strategy,
                        "selected_guide_file": selected_guide_file,
                        "selected_guide_name": str(selected_guide.get("guide_name", "") or "").strip(),
                        "selected_tool_names": selected_tool_names,
                        "selected_tool_files": selected_tool_files,
                        "sample_identifier": sample_id,
                        "sample_title": sample_title(runtime.dataset_type, event_data),
                        "experience_bank_version_id": current_context.experience_bank_payload.get("version_id"),
                        "experience_bank_hash": current_context.experience_bank_payload.get("bank_hash"),
                        "selected_guide_validation": selected_guide_validation,
                    }
                )
                write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                attempt_scope.set_metadata(
                    selected_guide_file=selected_guide_file,
                    sample_id=sample_id,
                )

                log_info(
                    "self_evolving",
                    (
                        f"Iteration start | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id} | mode=rollout_expansion | "
                        f"guide={selected_guide_file} | sample_id={sample_id}"
                    ),
                )

                with push_timing_context(
                    selected_guide_file=selected_guide_file,
                    sample_id=sample_id,
                ):
                    runner = build_dataset_runner(
                        runtime.dataset_type,
                        model=runtime.llm_config.model,
                        base_url=runtime.llm_config.base_url,
                        api_key=runtime.llm_config.api_key,
                    )
                    attempt_run_label = derive_attempt_run_label(
                        str(getattr(runtime.base_strategy_config, "factual_memory_run_label", "") or ""),
                        execution_id,
                    )
                    iteration_strategy_config = replace(
                        current_context.strategy_config,
                        mem_guide=selected_guide_file,
                        factual_memory_run_label=attempt_run_label,
                    )
                    with timed_block(
                        "self_evolving.attempt_phase",
                        "rollout_execution",
                        kind="phase",
                    ):
                        execution = run_single_event_with_rollout_sync(
                            runtime.dataset_type,
                            event_data,
                            runner,
                            strategy="web_search_loop",
                            strategy_config=iteration_strategy_config,
                            use_sources=runtime.use_sources,
                            use_market_data=runtime.use_market_data,
                            execution_id=execution_id,
                            factual_memory_run_label_override=attempt_run_label,
                        )
                    rollout_path = Path(execution["rollout_path"])
                    rollout = execution["rollout"]
                    with timed_block(
                        "self_evolving.attempt_phase",
                        "rollout_summary_build",
                        kind="phase",
                    ):
                        rollout_summary, rollout_summary_breakdown = build_rollout_summary(
                            rollout,
                            runtime.summary_max_chars,
                        )
                    attempt_record.update(
                        {
                            "rollout_file": rollout_path.name,
                            "rollout_path": str(rollout_path),
                            "processed_rollout_summary": rollout_summary,
                            "run_result": {
                                key: value
                                for key, value in execution["result"].items()
                                if key not in {"rollout", "rollout_path"}
                            },
                        }
                    )
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)

                    with timed_block(
                        "self_evolving.attempt_phase",
                        "critic_prompt_build",
                        kind="phase",
                    ):
                        tool_source_bundle = load_tool_source_bundle(selected_tool_names, tool_file_map)
                        critic_prompt, critic_prompt_lengths = build_critic_prompt_and_lengths(
                            critic_prompt_path=runtime.critic_prompt_path,
                            guide_object=selected_guide,
                            tool_source_bundle=tool_source_bundle,
                            rollout_summary=rollout_summary,
                            rollout_summary_breakdown=rollout_summary_breakdown,
                        )
                    attempt_record["prompt_lengths"]["critic"] = critic_prompt_lengths
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                    critique_response_text = request_llm_response(
                        critic_prompt,
                        runtime.llm_config,
                        log_tag="self_evolving.critic",
                    )
                    critique = parse_critique_response(critique_response_text)
                    critique["raw_response"] = critique_response_text
                    attempt_record["critique"] = critique
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)

                    if not critique["should_evolve"]:
                        attempt_record["status"] = "no_change"
                        write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                        attempt_scope.set_metadata(status="no_change")
                        log_info(
                            "self_evolving",
                            (
                                f"Iteration end | iteration={iteration}/{total_iterations} | "
                                f"execution_id={execution_id} | mode=rollout_expansion | "
                                f"guide={selected_guide_file} | sample_id={sample_id} | evolved=false"
                            ),
                        )
                        return

                    reuse_tool_selection_record = None
                    reusable_tool_entries: list[dict[str, Any]] = []
                    reusable_tool_source_bundle = ""
                    if runtime.optimization_reuse_duplicated_tool_enabled:
                        with timed_block(
                            "self_evolving.attempt_phase",
                            "reuse_tool_selection",
                            kind="phase",
                        ):
                            tool_summary_payload = sync_tool_summary()
                            reuse_selection_summary = select_reusable_tools_from_design_requirement(
                                design_requirement=critique["design_requirement"],
                                tool_summary_payload=tool_summary_payload,
                                llm_config=runtime.llm_config,
                                prompt_path=runtime.reuse_selection_prompt_path,
                            )
                            reusable_tool_entries = list(reuse_selection_summary["candidate_tool_entries"])
                            reusable_tool_names = list(reuse_selection_summary["candidate_tool_names"])
                            if reusable_tool_names:
                                reusable_tool_source_bundle = load_tool_source_bundle(
                                    reusable_tool_names,
                                    load_tool_file_map(),
                                )
                            reuse_tool_selection_record = {
                                "prompt_path": reuse_selection_summary["prompt_path"],
                                "prompt_lengths": reuse_selection_summary["prompt_lengths"],
                                "response_text": reuse_selection_summary["response_text"],
                                "analysis": reuse_selection_summary["analysis"],
                                "candidate_tool_names": reusable_tool_names,
                                "candidate_tool_files": [
                                    str(entry.get("tool_file", "") or "")
                                    for entry in reusable_tool_entries
                                ],
                                "candidate_tool_entries": reusable_tool_entries,
                                "candidate_tool_definition_bundle": build_tool_definition_bundle(
                                    reusable_tool_names,
                                    {"tools": {entry["tool_name"]: entry for entry in reusable_tool_entries}},
                                ) if reusable_tool_entries else "",
                                "candidate_tool_source_bundle": reusable_tool_source_bundle,
                            }
                        attempt_record["prompt_lengths"]["reuse_tool_selection"] = reuse_selection_summary["prompt_lengths"]
                        attempt_record["generation"] = {"reuse_tool_selection": reuse_tool_selection_record}
                        write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                        log_info(
                            "self_evolving",
                            (
                                f"Reuse tool selection | iteration={iteration}/{total_iterations} | "
                                f"execution_id={execution_id} | candidate_tools={reusable_tool_names}"
                            ),
                        )

                    with timed_block(
                        "self_evolving.attempt_phase",
                        "asset_generation",
                        kind="phase",
                    ):
                        generation_summary = generate_assets_from_design_requirement(
                            design_requirement=critique["design_requirement"],
                            llm_config=runtime.llm_config,
                            prompt_path=runtime.generation_prompt_path,
                            reusable_tool_entries=reusable_tool_entries,
                            reusable_tool_source_bundle=reusable_tool_source_bundle,
                        )
                    _finalize_generation(
                        iteration=iteration,
                        total_iterations=total_iterations,
                        execution_id=execution_id,
                        evolution_mode="rollout_expansion",
                        attempt_path=attempt_path,
                        attempt_record=attempt_record,
                        generation_summary=generation_summary,
                        runtime=runtime,
                        validation_enabled=validation_enabled,
                        current_strategy_config=current_context.strategy_config,
                        current_validation_key=current_context.validation_key,
                        current_validation_key_payload=current_context.validation_key_payload,
                        rollout_path=rollout_path,
                        selected_guide_file=selected_guide_file,
                        sample_id=sample_id,
                        reuse_tool_selection_record=reuse_tool_selection_record,
                    )
                    attempt_scope.set_metadata(
                        status=str(attempt_record.get("status", "unknown") or "unknown"),
                        new_guide_file=(
                            str(
                                (
                                    attempt_record.get("generation", {}) or {}
                                ).get("guide_file", "")
                                or ""
                            ).strip()
                            or None
                        ),
                    )
            except Exception as exc:
                attempt_record["status"] = "error"
                attempt_record["error"] = {
                    "message": str(exc),
                }
                attempt_scope.set_metadata(status="error", error_message=str(exc))
                if attempt_path is not None:
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                log_info(
                    "self_evolving",
                    (
                        f"Iteration failed | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id or 'unknown'} | mode=rollout_expansion | "
                        f"guide={selected_guide_file or 'unknown'} "
                        f"| sample_id={sample_id or 'unknown'} | error={exc}"
                    ),
                )


def run_single_experience_evolution_attempt(
    *,
    iteration: int,
    total_iterations: int,
    seed: int,
    runtime: EvolutionRuntime,
) -> None:
    if not _validation_enabled(runtime):
        raise ValueError("Experience evolution requires a validation dataset.")

    local_rng = random.Random(seed + iteration)
    selected_guide_file = ""
    sample_id = ""
    execution_id = ""
    attempt_path: Path | None = None
    attempt_record = _create_base_experience_attempt_record(
        iteration=iteration,
        runtime=runtime,
        validation_enabled=True,
    )
    attempt_path, attempt_record = create_experience_attempt_record(
        dataset_type=runtime.dataset_type,
        initial_payload=attempt_record,
        formatter=format_experience_attempt_record,
    )
    execution_id = derive_execution_id(int(attempt_record["attempt_index"]))
    attempt_record["execution_id"] = execution_id
    write_experience_attempt_record(
        dataset_type=runtime.dataset_type,
        attempt_path=attempt_path,
        payload=attempt_record,
        formatter=format_experience_attempt_record,
    )

    with push_timing_context(
        iteration=iteration,
        total_iterations=total_iterations,
        execution_id=execution_id,
        evolution_mode="experience_evolution",
        attempt_index=attempt_record.get("attempt_index"),
        attempt_file=attempt_path.name if attempt_path is not None else None,
    ):
        with timed_block(
            "experience_evolving.attempt",
            "experience_evolution_total",
            kind="attempt",
            metadata={"status": "running"},
        ) as attempt_scope:
            try:
                with timed_block(
                    "experience_evolving.attempt_phase",
                    "state_preparation",
                    kind="phase",
                ):
                    reload_tool_registry()
                    tool_file_map = load_tool_file_map()
                    tree = load_current_tree()
                    current_context = _resolve_current_validation_context(
                        runtime=runtime,
                        tree=tree,
                    )
                    (
                        selected_guide_file,
                        guide_selection_strategy,
                        selected_guide_selection,
                    ) = choose_best_tree_node(
                        tree=tree,
                        dataset_type=runtime.dataset_type,
                        validation_enabled=True,
                        validation_results=current_context.validation_results,
                        validation_key=current_context.validation_key,
                        experience_bank_hash=(
                            str(current_context.experience_bank_payload.get("bank_hash", "") or "").strip() or None
                        ),
                    )
                    selected_guide = load_guide_object(selected_guide_file)
                    selected_tool_names = [str(name) for name in selected_guide.get("tool_names", [])]
                    selected_tool_files = resolve_tool_files(selected_tool_names, tool_file_map)
                    baseline_validation_entry = get_validation_result_entry_for_experience(
                        current_context.validation_results or {},
                        guide_file=selected_guide_file,
                        validation_key=current_context.validation_key or "",
                        experience_bank_hash=(
                            str(current_context.experience_bank_payload.get("bank_hash", "") or "").strip() or None
                        ),
                    )
                    if baseline_validation_entry is None:
                        raise ValueError(
                            "Missing baseline validation entry for guide "
                            f"{selected_guide_file} under current experience bank "
                            f"{current_context.experience_bank_payload.get('bank_hash')}"
                        )
                    event_data = choose_training_sample(runtime.dataset_type, runtime.train_records, local_rng)
                    sample_id = sample_identifier(runtime.dataset_type, event_data)

                attempt_record.update(
                    {
                        "guide_selection_strategy": guide_selection_strategy,
                        "sample_identifier": sample_id,
                        "sample_title": sample_title(runtime.dataset_type, event_data),
                        "experience_bank_version_id": current_context.experience_bank_payload.get("version_id"),
                        "experience_bank_hash": current_context.experience_bank_payload.get("bank_hash"),
                        "selected_guide_file": selected_guide_file,
                        "selected_guide_name": str(selected_guide.get("guide_name", "") or "").strip(),
                        "selected_tool_names": selected_tool_names,
                        "selected_tool_files": selected_tool_files,
                        "selected_guide_validation": baseline_validation_entry,
                        "experience_evolution": {
                            "prompt_path": str(runtime.experience_evolution_prompt_path),
                            "prompt_lengths": None,
                            "response_text": "",
                            "selected_guide_selection": selected_guide_selection,
                            "baseline_validation": baseline_validation_entry,
                            "suggestions": [],
                            "rejected_suggestions": [],
                            "candidate_validations": [],
                            "accepted_suggestion": None,
                            "adopted_bank": None,
                        },
                    }
                )
                write_experience_attempt_record(
                    dataset_type=runtime.dataset_type,
                    attempt_path=attempt_path,
                    payload=attempt_record,
                    formatter=format_experience_attempt_record,
                )
                attempt_scope.set_metadata(
                    selected_guide_file=selected_guide_file,
                    sample_id=sample_id,
                    experience_bank_hash=current_context.experience_bank_payload.get("bank_hash"),
                )
                log_info(
                    "experience_evolving",
                    (
                        f"Iteration start | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id} | mode=experience_evolution | "
                        f"guide={selected_guide_file} | sample_id={sample_id}"
                    ),
                )

                runner = build_dataset_runner(
                    runtime.dataset_type,
                    model=runtime.llm_config.model,
                    base_url=runtime.llm_config.base_url,
                    api_key=runtime.llm_config.api_key,
                )
                attempt_run_label = derive_attempt_run_label(
                    str(getattr(runtime.base_strategy_config, "factual_memory_run_label", "") or ""),
                    execution_id,
                )
                iteration_strategy_config = replace(
                    current_context.strategy_config,
                    mem_guide=selected_guide_file,
                    factual_memory_run_label=attempt_run_label,
                )
                with timed_block(
                    "experience_evolving.attempt_phase",
                    "rollout_execution",
                    kind="phase",
                ):
                    execution = run_single_event_with_rollout_sync(
                        runtime.dataset_type,
                        event_data,
                        runner,
                        strategy="web_search_loop",
                        strategy_config=iteration_strategy_config,
                        use_sources=runtime.use_sources,
                        use_market_data=runtime.use_market_data,
                        execution_id=execution_id,
                        factual_memory_run_label_override=attempt_run_label,
                    )
                rollout_path = Path(execution["rollout_path"])
                rollout = execution["rollout"]
                run_result = {
                    key: value
                    for key, value in execution["result"].items()
                    if key not in {"rollout", "rollout_path"}
                }
                rollout_summary, _ = build_rollout_summary(
                    rollout,
                    runtime.summary_max_chars,
                    include_subagents=False,
                )
                experience_summary, experience_summary_breakdown = build_experience_evolution_summary(
                    experience_bank_payload=current_context.experience_bank_payload,
                    selected_guide_file=selected_guide_file,
                    selected_guide_name=str(selected_guide.get("guide_name", "") or "").strip(),
                    selected_guide_validation=baseline_validation_entry,
                    run_result=run_result,
                    rollout_summary=rollout_summary,
                    summary_max_chars=runtime.summary_max_chars,
                )
                attempt_record.update(
                    {
                        "rollout_file": rollout_path.name,
                        "rollout_path": str(rollout_path),
                        "processed_rollout_summary": experience_summary,
                        "run_result": run_result,
                    }
                )
                prompt, prompt_lengths = build_experience_evolution_prompt_and_lengths(
                    prompt_path=runtime.experience_evolution_prompt_path,
                    experience_bank_payload=current_context.experience_bank_payload,
                    selected_guide_file=selected_guide_file,
                    selected_guide_name=str(selected_guide.get("guide_name", "") or "").strip(),
                    selected_guide_validation=baseline_validation_entry,
                    run_result=run_result,
                    rollout_summary=experience_summary,
                    max_suggestions=runtime.experience_max_suggestions,
                )
                attempt_record["prompt_lengths"]["experience_evolution"] = {
                    **prompt_lengths,
                    "experience_summary_breakdown": experience_summary_breakdown,
                }
                response_text = request_llm_response(
                    prompt,
                    runtime.llm_config,
                    log_tag="experience_evolving.llm",
                )
                parsed_response = parse_experience_evolution_response(
                    response_text,
                    max_suggestions=runtime.experience_max_suggestions,
                    current_experience_ids={
                        str(entry.get("experience_id", "") or "").strip()
                        for entry in current_context.experience_bank_payload.get("experiences", [])
                    },
                )
                attempt_record["experience_evolution"] = {
                    **(attempt_record.get("experience_evolution", {}) or {}),
                    "prompt_lengths": attempt_record["prompt_lengths"]["experience_evolution"],
                    "response_text": response_text,
                    "suggestions": parsed_response["suggestions"],
                    "rejected_suggestions": parsed_response["rejected_suggestions"],
                }
                write_experience_attempt_record(
                    dataset_type=runtime.dataset_type,
                    attempt_path=attempt_path,
                    payload=attempt_record,
                    formatter=format_experience_attempt_record,
                )

                if not parsed_response["suggestions"]:
                    attempt_record["status"] = "no_change"
                    write_experience_attempt_record(
                        dataset_type=runtime.dataset_type,
                        attempt_path=attempt_path,
                        payload=attempt_record,
                        formatter=format_experience_attempt_record,
                    )
                    attempt_scope.set_metadata(status="no_change")
                    log_info(
                        "experience_evolving",
                        (
                            f"Iteration end | iteration={iteration}/{total_iterations} | "
                            f"execution_id={execution_id} | mode=experience_evolution | "
                            f"guide={selected_guide_file} | sample_id={sample_id} | evolved=false"
                        ),
                    )
                    return

                for suggestion in parsed_response["suggestions"]:
                    candidate_record: dict[str, Any] = {
                        "suggestion": suggestion,
                        "candidate_bank_hash": None,
                        "candidate_bank_version_id": None,
                        "validation_entry": None,
                        "validation_key": None,
                        "validation_key_payload": None,
                        "accepted": False,
                        "rejection_reason": None,
                        "error": None,
                    }
                    try:
                        candidate_bank_payload = build_candidate_experience_bank(
                            bank_payload=current_context.experience_bank_payload,
                            suggestion=suggestion,
                            source_attempt_file=attempt_path.name if attempt_path is not None else None,
                            source_rollout_file=rollout_path.name,
                            source_guide_file=selected_guide_file,
                            source_sample_identifier=sample_id,
                        )
                        candidate_record["candidate_bank_hash"] = candidate_bank_payload.get("bank_hash")
                        (
                            candidate_validation_entry,
                            candidate_validation_key,
                            candidate_validation_key_payload,
                        ) = validate_experience_bank_candidate(
                            dataset_type=runtime.dataset_type,
                            bank_payload=candidate_bank_payload,
                            guide_file=selected_guide_file,
                            guide_name=str(selected_guide.get("guide_name", "") or "").strip(),
                            val_records=runtime.val_records or [],
                            llm_config=runtime.llm_config,
                            base_strategy_config=current_context.strategy_config,
                            use_sources=runtime.use_sources,
                            use_market_data=runtime.use_market_data,
                            val_data_path=runtime.val_data_path or runtime.train_data_path,
                        )
                        candidate_record["validation_entry"] = candidate_validation_entry
                        candidate_record["validation_key"] = candidate_validation_key
                        candidate_record["validation_key_payload"] = candidate_validation_key_payload

                        if is_candidate_validation_better(
                            dataset_type=runtime.dataset_type,
                            baseline_entry=baseline_validation_entry,
                            candidate_entry=candidate_validation_entry,
                        ):
                            saved_bank_payload = save_experience_bank_version(
                                dataset_type=runtime.dataset_type,
                                experience_entries=list(candidate_bank_payload.get("experiences", [])),
                                base_version_id=current_context.experience_bank_payload.get("version_id"),
                                applied_suggestion=candidate_bank_payload.get("applied_suggestion"),
                            )
                            candidate_record["accepted"] = True
                            candidate_record["candidate_bank_version_id"] = saved_bank_payload.get("version_id")
                            attempt_record["experience_evolution"] = {
                                **(attempt_record.get("experience_evolution", {}) or {}),
                                "accepted_suggestion": suggestion,
                                "adopted_bank": {
                                    "version_id": saved_bank_payload.get("version_id"),
                                    "bank_hash": saved_bank_payload.get("bank_hash"),
                                    "base_version_id": saved_bank_payload.get("base_version_id"),
                                },
                                "candidate_validations": list(
                                    (
                                        attempt_record.get("experience_evolution", {}) or {}
                                    ).get("candidate_validations", [])
                                )
                                + [candidate_record],
                            }
                            attempt_record["status"] = "evolved"
                            write_experience_attempt_record(
                                dataset_type=runtime.dataset_type,
                                attempt_path=attempt_path,
                                payload=attempt_record,
                                formatter=format_experience_attempt_record,
                            )
                            attempt_scope.set_metadata(
                                status="evolved",
                                adopted_bank_hash=saved_bank_payload.get("bank_hash"),
                            )
                            log_info(
                                "experience_evolving",
                                (
                                    f"Iteration end | iteration={iteration}/{total_iterations} | "
                                    f"execution_id={execution_id} | mode=experience_evolution | "
                                    f"guide={selected_guide_file} | sample_id={sample_id} | "
                                    f"evolved=true | adopted_bank={saved_bank_payload.get('version_id')}"
                                ),
                            )
                            return

                        candidate_record["rejection_reason"] = "candidate_validation_not_better"
                    except Exception as exc:
                        candidate_record["rejection_reason"] = "candidate_validation_error"
                        candidate_record["error"] = {"message": str(exc)}

                    attempt_record["experience_evolution"] = {
                        **(attempt_record.get("experience_evolution", {}) or {}),
                        "candidate_validations": list(
                            ((attempt_record.get("experience_evolution", {}) or {}).get("candidate_validations", []))
                        )
                        + [candidate_record],
                    }
                    write_experience_attempt_record(
                        dataset_type=runtime.dataset_type,
                        attempt_path=attempt_path,
                        payload=attempt_record,
                        formatter=format_experience_attempt_record,
                    )

                attempt_record["status"] = "no_change"
                write_experience_attempt_record(
                    dataset_type=runtime.dataset_type,
                    attempt_path=attempt_path,
                    payload=attempt_record,
                    formatter=format_experience_attempt_record,
                )
                attempt_scope.set_metadata(status="no_change")
                log_info(
                    "experience_evolving",
                    (
                        f"Iteration end | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id} | mode=experience_evolution | "
                        f"guide={selected_guide_file} | sample_id={sample_id} | evolved=false"
                    ),
                )
            except Exception as exc:
                attempt_record["status"] = "error"
                attempt_record["error"] = {
                    "message": str(exc),
                }
                attempt_scope.set_metadata(status="error", error_message=str(exc))
                if attempt_path is not None:
                    write_experience_attempt_record(
                        dataset_type=runtime.dataset_type,
                        attempt_path=attempt_path,
                        payload=attempt_record,
                        formatter=format_experience_attempt_record,
                    )
                log_info(
                    "experience_evolving",
                    (
                        f"Iteration failed | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id or 'unknown'} | mode=experience_evolution | "
                        f"guide={selected_guide_file or 'unknown'} | sample_id={sample_id or 'unknown'} "
                        f"| error={exc}"
                    ),
                )
                if isinstance(exc, NoEligibleBestGuideSelectionError):
                    log_info(
                        "experience_evolving",
                        (
                            "Fatal experience evolution error | "
                            f"reason={NO_ELIGIBLE_BEST_GUIDE_SELECTION_ERROR} | "
                            "action=abort_run"
                        ),
                    )
                    raise


def run_single_exploration_attempt(
    *,
    iteration: int,
    total_iterations: int,
    runtime: EvolutionRuntime,
) -> None:
    execution_id = ""
    validation_enabled = _validation_enabled(runtime)
    attempt_path: Path | None = None
    attempt_record = _create_base_attempt_record(
        iteration=iteration,
        runtime=runtime,
        validation_enabled=validation_enabled,
        evolution_mode="exploration",
    )

    attempt_path, attempt_record = create_attempt_record(attempt_record, format_attempt_record)
    execution_id = derive_execution_id(int(attempt_record["attempt_index"]))
    attempt_record["execution_id"] = execution_id
    write_attempt_record(attempt_path, attempt_record, format_attempt_record)

    with push_timing_context(
        iteration=iteration,
        total_iterations=total_iterations,
        execution_id=execution_id,
        evolution_mode="exploration",
        attempt_index=attempt_record.get("attempt_index"),
        attempt_file=attempt_path.name if attempt_path is not None else None,
    ):
        with timed_block(
            "self_evolving.attempt",
            "exploration_total",
            kind="attempt",
            metadata={"status": "running"},
        ) as attempt_scope:
            try:
                with timed_block(
                    "self_evolving.attempt_phase",
                    "sync_guide_summary",
                    kind="phase",
                ):
                    tree = load_current_tree()
                    current_context = _resolve_current_validation_context(
                        runtime=runtime,
                        tree=tree,
                    )
                    guide_summary_payload = sync_guide_summary(
                        tree=tree,
                        llm_config=runtime.llm_config,
                        prompt_path=runtime.guide_classification_prompt_path,
                        validation_results=current_context.validation_results if validation_enabled else None,
                        validation_key=current_context.validation_key,
                    )
                attempt_record["experience_bank_version_id"] = current_context.experience_bank_payload.get(
                    "version_id"
                )
                attempt_record["experience_bank_hash"] = current_context.experience_bank_payload.get("bank_hash")
                write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                representative_guide_files = [
                    str(guide_file)
                    for guide_file in guide_summary_payload.get("categories", {}).keys()
                    if str(guide_file or "").strip()
                ]
                with timed_block(
                    "self_evolving.attempt_phase",
                    "build_guide_representatives_bundle",
                    kind="phase",
                ):
                    representatives_bundle = build_guide_category_representatives_bundle(
                        representative_guide_files,
                    )
                attempt_scope.set_metadata(
                    representative_guide_count=len(representative_guide_files),
                )
                log_info(
                    "self_evolving",
                    (
                        f"Iteration start | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id} | mode=exploration | "
                        f"category_count={len(representative_guide_files)}"
                    ),
                )

                with timed_block(
                    "self_evolving.attempt_phase",
                    "exploration_asset_generation",
                    kind="phase",
                ):
                    generation_summary = generate_assets_from_exploration_context(
                        guide_category_representatives_bundle=representatives_bundle,
                        llm_config=runtime.llm_config,
                        prompt_path=runtime.exploration_prompt_path,
                    )
                _finalize_generation(
                    iteration=iteration,
                    total_iterations=total_iterations,
                    execution_id=execution_id,
                    evolution_mode="exploration",
                    attempt_path=attempt_path,
                    attempt_record=attempt_record,
                    generation_summary=generation_summary,
                    runtime=runtime,
                    validation_enabled=validation_enabled,
                    current_strategy_config=current_context.strategy_config,
                    current_validation_key=current_context.validation_key,
                    current_validation_key_payload=current_context.validation_key_payload,
                    exploration_from_guide_file=representative_guide_files,
                )
                attempt_scope.set_metadata(
                    status=str(attempt_record.get("status", "unknown") or "unknown"),
                    new_guide_file=(
                        str(((attempt_record.get("generation", {}) or {}).get("guide_file", "")) or "").strip()
                        or None
                    ),
                )
            except Exception as exc:
                attempt_record["status"] = "error"
                attempt_record["error"] = {
                    "message": str(exc),
                }
                attempt_scope.set_metadata(status="error", error_message=str(exc))
                if attempt_path is not None:
                    write_attempt_record(attempt_path, attempt_record, format_attempt_record)
                log_info(
                    "self_evolving",
                    (
                        f"Iteration failed | iteration={iteration}/{total_iterations} | "
                        f"execution_id={execution_id or 'unknown'} | mode=exploration | error={exc}"
                    ),
                )
