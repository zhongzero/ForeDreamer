#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import os
from pathlib import Path
from typing import Final

from prediction.runtime import build_prompt_strategy_config, load_dataset_records
from utils.logger import log_info
from utils.timing_registry import (
    finish_timing_run,
    reset_timing_registry,
    start_timing_run,
    timed_block,
    update_run_metadata,
)
from MemTool.registry import reload_tool_registry
from SelfEvolving.evolve_storage import (
    GUIDE_SUMMARY_PATH,
    TOOL_SUMMARY_PATH,
    load_or_initialize_tree,
    load_tool_file_map,
    sync_tool_summary,
)
from SelfEvolving.evolve_experience import resolve_experience_evolution_prompt_path
from SelfEvolving.evolve_guide_summary import sync_guide_summary
from SelfEvolving.evolve_time_report import write_timing_report
from SelfEvolving.evolve_validation import (
    UNIFORM_RANDOM_SELECTION,
    ZIPF_BY_VALIDATION_RANK_SELECTION,
    build_validation_key,
    ensure_tree_validation_results,
)
from SelfEvolving.evolve_worker import (
    EvolutionRuntime,
    run_single_experience_evolution_attempt,
    run_single_evolution_attempt,
    run_single_exploration_attempt,
)
from SelfEvolving.generate_memguide_and_memtool import (
    DEFAULT_EXPLORATION_PROMPT_PATH,
    DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH,
    DEFAULT_PROMPT_PATH as DEFAULT_GENERATION_PROMPT_PATH,
    DEFAULT_REUSE_SELECTION_PROMPT_PATH,
    LLMConfig,
)
from utils.generated_paths import (
    add_generated_path_arguments,
    configure_generated_path_env_from_namespace,
    ensure_mem_asset_bootstrap_from_namespace,
)
from utils.search_provider import DEFAULT_SEARCH_PROVIDER, SUPPORTED_SEARCH_PROVIDERS


OPENAI_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
OPENAI_BASE_URL_ENV: Final[str] = "OPENAI_BASE_URL"
OPENAI_MODEL_ENV: Final[str] = "OPENAI_MODEL"
DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: Final[str] = "minimax/minimax-m1"
DEFAULT_SUMMARY_MAX_CHARS: Final[int] = 12000

SRC_DIR: Final[Path] = Path(__file__).resolve().parents[1]
SELF_EVOLVING_DIR: Final[Path] = SRC_DIR / "SelfEvolving"
PROMPT_DIR: Final[Path] = SELF_EVOLVING_DIR / "prompt"
DEFAULT_CRITIC_PROMPT_PATH: Final[Path] = PROMPT_DIR / "mem_evolution_critic_prompt.md"


def _resolve_reuse_selection_prompt_path(generation_prompt_path: Path) -> Path:
    if generation_prompt_path.name.endswith("-cn.md"):
        return PROMPT_DIR / "mem_evolution_reuse_selection_prompt-cn.md"
    return DEFAULT_REUSE_SELECTION_PROMPT_PATH


def _resolve_guide_classification_prompt_path(generation_prompt_path: Path) -> Path:
    if generation_prompt_path.name.endswith("-cn.md"):
        return PROMPT_DIR / "mem_evolution_guide_classification_prompt-cn.md"
    return DEFAULT_GUIDE_CLASSIFICATION_PROMPT_PATH


def _resolve_exploration_prompt_path(generation_prompt_path: Path) -> Path:
    if generation_prompt_path.name.endswith("-cn.md"):
        return PROMPT_DIR / "mem_evolution_exploration_prompt-cn.md"
    return DEFAULT_EXPLORATION_PROMPT_PATH


def _parse_exploration_over_expansion(value: str) -> tuple[int, int]:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError("--exploration_over_expansion must be a non-empty ratio like 3:4")

    parts = normalized_value.split(":")
    if len(parts) != 2:
        raise ValueError("--exploration_over_expansion must be in A:B format, for example 3:4")

    try:
        exploration_count = int(parts[0])
        expansion_count = int(parts[1])
    except ValueError as exc:
        raise ValueError("--exploration_over_expansion must contain integer counts like 3:4") from exc

    if exploration_count < 1 or expansion_count < 1:
        raise ValueError("--exploration_over_expansion must use positive integers like 3:4")
    return exploration_count, expansion_count


def _build_iteration_modes(
    *,
    total_iterations: int,
    optimization_encourage_exploration_enabled: bool,
    exploration_over_expansion: tuple[int, int] | None,
) -> list[str]:
    if not optimization_encourage_exploration_enabled:
        return ["rollout_expansion"] * total_iterations

    if exploration_over_expansion is None:
        raise ValueError("Missing exploration_over_expansion ratio.")

    exploration_count, expansion_count = exploration_over_expansion
    phase_cycle = (["exploration"] * exploration_count) + (["rollout_expansion"] * expansion_count)
    modes: list[str] = []
    while len(modes) < total_iterations:
        modes.extend(phase_cycle)
    return modes[:total_iterations]


def _build_cycle_phase_sequence(
    *,
    total_iterations: int,
    experience_rounds_per_cycle: int,
    memguide_rounds_per_cycle: int,
    experience_before_memguide: bool,
) -> list[str]:
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")
    if experience_rounds_per_cycle < 0:
        raise ValueError("--experience_rounds_per_cycle must be >= 0")
    if memguide_rounds_per_cycle < 0:
        raise ValueError("--memguide_rounds_per_cycle must be >= 0")
    if experience_rounds_per_cycle == 0 and memguide_rounds_per_cycle == 0:
        raise ValueError(
            "At least one of --experience_rounds_per_cycle or --memguide_rounds_per_cycle must be > 0"
        )

    if experience_before_memguide:
        cycle = (["experience"] * experience_rounds_per_cycle) + (["memguide"] * memguide_rounds_per_cycle)
    else:
        cycle = (["memguide"] * memguide_rounds_per_cycle) + (["experience"] * experience_rounds_per_cycle)
    phases: list[str] = []
    while len(phases) < total_iterations:
        phases.extend(cycle)
    return phases[:total_iterations]


def _run_rollout_expansion_phase(
    *,
    executor: ThreadPoolExecutor,
    phase_iterations: list[int],
    total_iterations: int,
    seed: int,
    runtime: EvolutionRuntime,
    parallelism: int,
) -> None:
    max_workers = min(parallelism, len(phase_iterations))
    pending: dict[Future[None], int] = {}
    next_index = 0

    while next_index < len(phase_iterations) and len(pending) < max_workers:
        scheduled_iteration = phase_iterations[next_index]
        future = executor.submit(
            run_single_evolution_attempt,
            iteration=scheduled_iteration,
            total_iterations=total_iterations,
            seed=seed,
            runtime=runtime,
        )
        pending[future] = scheduled_iteration
        next_index += 1

    while pending:
        completed, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
        for future in completed:
            pending.pop(future)
            future.result()
            if next_index < len(phase_iterations):
                scheduled_iteration = phase_iterations[next_index]
                next_future = executor.submit(
                    run_single_evolution_attempt,
                    iteration=scheduled_iteration,
                    total_iterations=total_iterations,
                    seed=seed,
                    runtime=runtime,
                )
                pending[next_future] = scheduled_iteration
                next_index += 1


def _resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    api_key = (
        str(args.api_key or "").strip()
        or str(os.getenv(OPENAI_API_KEY_ENV, "")).strip()
        or str(os.getenv(OPENROUTER_API_KEY_ENV, "")).strip()
    )
    if not api_key:
        raise ValueError(
            "Missing API key. Pass --api_key or set OPENAI_API_KEY / OPENROUTER_API_KEY."
        )

    base_url = (
        str(args.base_url or "").strip()
        or str(os.getenv(OPENAI_BASE_URL_ENV, "")).strip()
        or DEFAULT_BASE_URL
    )
    model = (
        str(args.model or "").strip()
        or str(os.getenv(OPENAI_MODEL_ENV, "")).strip()
        or DEFAULT_MODEL
    )
    return LLMConfig(api_key=api_key, base_url=base_url, model=model)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grow the MemGuide tree through self-evolution.")
    add_generated_path_arguments(parser)
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=["prophet_arena", "futurex"],
        help="Dataset used for training-time guide evolution.",
    )
    parser.add_argument(
        "--train_data_path",
        required=True,
        help="Training dataset used to sample rollout questions for guide evolution.",
    )
    parser.add_argument(
        "--val_data_path",
        default=None,
        help="Optional validation dataset. If provided, guide selection uses Zipf sampling over validation rank.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Total number of evolution attempts to run across rollout expansion and optional exploration phases.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Maximum number of independent evolution attempts to run concurrently.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for guide-node and training-sample selection.",
    )
    parser.add_argument(
        "--summary_max_chars",
        type=int,
        default=DEFAULT_SUMMARY_MAX_CHARS,
        help="Maximum number of characters allowed in the processed rollout summary passed to the critique model.",
    )
    parser.add_argument(
        "--critic_prompt_path",
        default=str(DEFAULT_CRITIC_PROMPT_PATH),
        help="Prompt template used to critique one rollout against the selected guide and tools.",
    )
    parser.add_argument(
        "--generation_prompt_path",
        default=str(DEFAULT_GENERATION_PROMPT_PATH),
        help="Prompt template used to generate a new guide and optional tools from a design requirement.",
    )
    parser.add_argument(
        "--experience_evolution_prompt_path",
        default=None,
        help="Prompt template used to evolve the main-agent experience bank.",
    )
    parser.add_argument(
        "--optimization_reuse_duplicated_tool",
        action="store_true",
        help="Enable reuse-oriented generation by selecting up to 3 existing tools before final guide/tool generation.",
    )
    parser.add_argument(
        "--optimization_encourage_exploration",
        action="store_true",
        help="Enable alternating exploration evolves that generate category-diverse guides without rollout feedback.",
    )
    parser.add_argument(
        "--exploration_over_expansion",
        default=None,
        help='Exploration/expansion cycle ratio in A:B form, for example "3:4". Only used when --optimization_encourage_exploration is enabled.',
    )
    parser.add_argument("--api_key", "-k", default=None, help="OpenAI-compatible API key.")
    parser.add_argument(
        "--base_url",
        "-u",
        default=None,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name used for answering, critique, and generation.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=4,
        help="Maximum assistant turns for the main agent web_search_loop.",
    )
    parser.add_argument(
        "--subagent_max_turns",
        type=int,
        default=10,
        help="Maximum assistant turns for each data-process subagent.",
    )
    parser.add_argument(
        "--search_max_results",
        type=int,
        default=5,
        help="Maximum number of search results to return to the model.",
    )
    parser.add_argument(
        "--search_max_chars_per_result",
        type=int,
        default=700,
        help="Maximum characters per search result snippet.",
    )
    parser.add_argument(
        "--search_max_total_chars",
        type=int,
        default=2500,
        help="Maximum total characters across all search result snippets.",
    )
    parser.add_argument(
        "--search_provider",
        choices=sorted(SUPPORTED_SEARCH_PROVIDERS),
        default=os.getenv("SEARCH_PROVIDER", DEFAULT_SEARCH_PROVIDER),
        help="Search provider to use for web_search_loop (default: tavily or $SEARCH_PROVIDER).",
    )
    parser.add_argument(
        "--search_api_key",
        default=None,
        help=(
            "Optional search API key override for web_search_loop. "
            "Tavily supports a comma-separated list; Firecrawl typically uses a single key."
        ),
    )
    parser.add_argument(
        "--experience_rounds_per_cycle",
        type=int,
        default=10,
        help="Number of experience-evolution iterations to run in each cycle.",
    )
    parser.add_argument(
        "--memguide_rounds_per_cycle",
        type=int,
        default=10,
        help="Number of memguide iterations to run in each cycle.",
    )
    parser.add_argument(
        "--experience_before_memguide",
        action="store_true",
        help=(
            "Run each cycle as experience -> memguide. "
            "By default, cycles run as memguide -> experience."
        ),
    )
    parser.add_argument(
        "--experience_max_suggestions",
        type=int,
        default=2,
        help="Maximum number of experience suggestions to validate in one experience-evolution attempt.",
    )
    parser.add_argument(
        "--use_tavilty_raw_context",
        action="store_true",
        help="Use Tavily raw_content instead of the default content snippets when available.",
    )
    parser.add_argument(
        "--enable_lossy_search_cache",
        action="store_true",
        help=(
            "Enable lossy search-cache reuse keyed by dataset sample identifier and main-agent turn. "
            "Automatically enables --enable_api_cache."
        ),
    )
    parser.add_argument(
        "--disable_main_agent_final_answer_cache",
        action="store_true",
        help=(
            "Bypass shared LLM cache for the main agent's final answer during rollout/evaluation "
            "so final-response failures are not replayed from cache."
        ),
    )
    parser.add_argument(
        "--use_market_data_in_prophet_arena",
        action="store_true",
        help="Include prediction-market data in Prophet-Arena prompts during evolution runs.",
    )
    parser.add_argument(
        "--use_source_in_prophet_arena",
        action="store_true",
        help="Include curated source summaries in Prophet-Arena prompts during evolution runs.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.enable_lossy_search_cache:
        args.enable_api_cache = True
    configure_generated_path_env_from_namespace(args)
    bootstrap_actions = ensure_mem_asset_bootstrap_from_namespace(args)
    reload_tool_registry()
    reset_timing_registry()
    start_timing_run(
        run_name="evolve_memguide_and_memtool",
        metadata={
            "script": "src/evolve_experience_memguide_and_memtool.py",
        },
    )

    exit_code = 0
    run_error: str | None = None

    try:
        if bootstrap_actions:
            log_info(
                "self_evolving",
                f"Bootstrapped initial mem assets | actions={bootstrap_actions}",
            )
        if args.parallelism < 1:
            raise ValueError("--parallelism must be >= 1")
        if args.num_iterations < 1:
            raise ValueError("--num_iterations must be >= 1")
        if args.experience_max_suggestions < 1:
            raise ValueError("--experience_max_suggestions must be >= 1")
        if args.experience_rounds_per_cycle < 0:
            raise ValueError("--experience_rounds_per_cycle must be >= 0")
        if args.memguide_rounds_per_cycle < 0:
            raise ValueError("--memguide_rounds_per_cycle must be >= 0")
        if args.experience_rounds_per_cycle == 0 and args.memguide_rounds_per_cycle == 0:
            raise ValueError(
                "At least one of --experience_rounds_per_cycle or --memguide_rounds_per_cycle must be > 0"
            )

        llm_config = _resolve_llm_config(args)
        update_run_metadata(
            dataset_type=args.dataset_type,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            num_iterations=args.num_iterations,
            parallelism=args.parallelism,
            model=llm_config.model,
            base_url=llm_config.base_url,
            max_turns=args.max_turns,
            subagent_max_turns=args.subagent_max_turns,
            search_max_results=args.search_max_results,
            search_max_chars_per_result=args.search_max_chars_per_result,
            search_max_total_chars=args.search_max_total_chars,
            experience_rounds_per_cycle=args.experience_rounds_per_cycle,
            memguide_rounds_per_cycle=args.memguide_rounds_per_cycle,
            experience_before_memguide=bool(args.experience_before_memguide),
            experience_max_suggestions=args.experience_max_suggestions,
            optimization_reuse_duplicated_tool_enabled=bool(args.optimization_reuse_duplicated_tool),
            optimization_encourage_exploration_enabled=bool(args.optimization_encourage_exploration),
            exploration_over_expansion=args.exploration_over_expansion,
        )
        critic_prompt_path = Path(args.critic_prompt_path)
        generation_prompt_path = Path(args.generation_prompt_path)
        reuse_selection_prompt_path = _resolve_reuse_selection_prompt_path(generation_prompt_path)
        guide_classification_prompt_path = _resolve_guide_classification_prompt_path(generation_prompt_path)
        exploration_prompt_path = _resolve_exploration_prompt_path(generation_prompt_path)
        experience_evolution_prompt_path = (
            Path(args.experience_evolution_prompt_path)
            if str(args.experience_evolution_prompt_path or "").strip()
            else resolve_experience_evolution_prompt_path(generation_prompt_path)
        )
        optimization_reuse_duplicated_tool_enabled = bool(args.optimization_reuse_duplicated_tool)
        optimization_encourage_exploration_enabled = bool(args.optimization_encourage_exploration)
        if optimization_encourage_exploration_enabled and args.exploration_over_expansion is None:
            raise ValueError(
                "--exploration_over_expansion is required when --optimization_encourage_exploration is enabled"
            )
        exploration_over_expansion = (
            _parse_exploration_over_expansion(args.exploration_over_expansion)
            if optimization_encourage_exploration_enabled
            else None
        )

        train_data_path = Path(args.train_data_path)
        if not train_data_path.exists():
            raise ValueError(f"Missing train_data_path: {train_data_path}")
        with timed_block(
            "self_evolving.run_phase",
            "load_training_records",
            kind="phase",
        ):
            train_records = load_dataset_records(args.dataset_type, str(train_data_path))
        if not train_records:
            raise ValueError(f"Training dataset is empty: {train_data_path}")

        val_data_path = Path(args.val_data_path) if args.val_data_path else None
        if val_data_path is not None and not val_data_path.exists():
            raise ValueError(f"Missing val_data_path: {val_data_path}")
        with timed_block(
            "self_evolving.run_phase",
            "load_validation_records",
            kind="phase",
            metadata={"enabled": val_data_path is not None},
        ):
            val_records = (
                load_dataset_records(args.dataset_type, str(val_data_path))
                if val_data_path is not None
                else None
            )
        validation_enabled = val_data_path is not None
        if validation_enabled and not val_records:
            raise ValueError(f"Validation dataset is empty: {val_data_path}")
        if args.experience_rounds_per_cycle > 0 and not validation_enabled:
            raise ValueError("Experience evolution requires --val_data_path.")

        with timed_block(
            "self_evolving.run_phase",
            "build_base_strategy_config",
            kind="phase",
        ):
            base_strategy_config = build_prompt_strategy_config(
                dataset_type=args.dataset_type,
                max_turns=args.max_turns,
                subagent_max_turns=args.subagent_max_turns,
                save_rollout=True,
                search_provider=args.search_provider,
                search_max_results=args.search_max_results,
                search_max_chars_per_result=args.search_max_chars_per_result,
                search_max_total_chars=args.search_max_total_chars,
                search_api_key=args.search_api_key,
                use_tavilty_raw_context=args.use_tavilty_raw_context,
                enable_lossy_search_cache=args.enable_lossy_search_cache,
                disable_main_agent_final_answer_cache=args.disable_main_agent_final_answer_cache,
                mem_guide="guide_initial.json",
            )

        with timed_block(
            "self_evolving.run_phase",
            "load_tool_registry_and_tree",
            kind="phase",
        ):
            reload_tool_registry()
            tool_file_map = load_tool_file_map()
            tree = load_or_initialize_tree(tool_file_map)
        if optimization_reuse_duplicated_tool_enabled:
            with timed_block(
                "self_evolving.run_phase",
                "sync_tool_summary",
                kind="phase",
            ):
                tool_summary_payload = sync_tool_summary()
            log_info(
                "self_evolving",
                (
                    f"Tool reuse optimization | enabled=true | tool_summary_path={TOOL_SUMMARY_PATH} "
                    f"| tool_count={len(tool_summary_payload.get('tools', {}))} "
                    f"| reuse_selection_prompt_path={reuse_selection_prompt_path}"
                ),
            )
        else:
            log_info("self_evolving", "Tool reuse optimization | enabled=false")

        if optimization_encourage_exploration_enabled:
            exploration_count, expansion_count = exploration_over_expansion or (0, 0)
            log_info(
                "self_evolving",
                (
                    f"Exploration optimization | enabled=true | "
                    f"exploration_over_expansion={exploration_count}:{expansion_count} | "
                    f"exploration_prompt_path={exploration_prompt_path} | "
                    f"guide_classification_prompt_path={guide_classification_prompt_path}"
                ),
            )
        else:
            log_info("self_evolving", "Exploration optimization | enabled=false")

        log_info(
            "self_evolving",
            (
                f"cycle_phase_order={'experience_then_memguide' if args.experience_before_memguide else 'memguide_then_experience'} | "
                f"Experience evolution config | prompt_path={experience_evolution_prompt_path} | "
                f"experience_rounds_per_cycle={args.experience_rounds_per_cycle} | "
                f"memguide_rounds_per_cycle={args.memguide_rounds_per_cycle} | "
                f"experience_max_suggestions={args.experience_max_suggestions}"
            ),
        )

        guide_selection_strategy = (
            ZIPF_BY_VALIDATION_RANK_SELECTION if validation_enabled else UNIFORM_RANDOM_SELECTION
        )
        log_info(
            "self_evolving",
            (
                f"Validation config | train_data_path={train_data_path} | "
                f"val_data_path={str(val_data_path) if val_data_path is not None else 'None'} | "
                f"validation_enabled={validation_enabled} | "
                f"guide_selection_strategy={guide_selection_strategy}"
            ),
        )

        validation_key = None
        validation_key_payload = None
        current_validation_results = None
        if validation_enabled:
            with timed_block(
                "self_evolving.run_phase",
                "validation_bootstrap",
                kind="phase",
            ):
                validation_key, validation_key_payload = build_validation_key(
                    dataset_type=args.dataset_type,
                    val_data_path=val_data_path,
                    llm_config=llm_config,
                    base_strategy_config=base_strategy_config,
                    use_sources=args.use_source_in_prophet_arena,
                    use_market_data=args.use_market_data_in_prophet_arena,
                )
                current_validation_results = ensure_tree_validation_results(
                    tree=tree,
                    dataset_type=args.dataset_type,
                    val_records=val_records or [],
                    llm_config=llm_config,
                    base_strategy_config=base_strategy_config,
                    use_sources=args.use_source_in_prophet_arena,
                    use_market_data=args.use_market_data_in_prophet_arena,
                    validation_key=validation_key,
                    validation_key_payload=validation_key_payload,
                )

        with timed_block(
            "self_evolving.run_phase",
            "guide_summary_sync",
            kind="phase",
        ):
            guide_summary_payload = sync_guide_summary(
                tree=tree,
                llm_config=llm_config,
                prompt_path=guide_classification_prompt_path,
                validation_results=current_validation_results,
                validation_key=validation_key,
            )
        log_info(
            "self_evolving",
            (
                f"Guide summary ready | guide_summary_path={GUIDE_SUMMARY_PATH} | "
                f"category_count={len(guide_summary_payload.get('categories', {}))} | "
                f"guide_count={len(guide_summary_payload.get('guide_to_category', {}))}"
            ),
        )

        runtime = EvolutionRuntime(
            dataset_type=args.dataset_type,
            train_data_path=train_data_path,
            train_records=train_records,
            val_data_path=val_data_path,
            val_records=val_records,
            llm_config=llm_config,
            critic_prompt_path=critic_prompt_path,
            generation_prompt_path=generation_prompt_path,
            experience_evolution_prompt_path=experience_evolution_prompt_path,
            reuse_selection_prompt_path=reuse_selection_prompt_path,
            guide_classification_prompt_path=guide_classification_prompt_path,
            exploration_prompt_path=exploration_prompt_path,
            base_strategy_config=base_strategy_config,
            use_sources=args.use_source_in_prophet_arena,
            use_market_data=args.use_market_data_in_prophet_arena,
            summary_max_chars=args.summary_max_chars,
            experience_max_suggestions=args.experience_max_suggestions,
            validation_key=validation_key,
            validation_key_payload=validation_key_payload,
            optimization_reuse_duplicated_tool_enabled=optimization_reuse_duplicated_tool_enabled,
            optimization_encourage_exploration_enabled=optimization_encourage_exploration_enabled,
            exploration_over_expansion=(
                f"{exploration_over_expansion[0]}:{exploration_over_expansion[1]}"
                if exploration_over_expansion is not None
                else None
            ),
        )
        phase_sequence = _build_cycle_phase_sequence(
            total_iterations=args.num_iterations,
            experience_rounds_per_cycle=args.experience_rounds_per_cycle,
            memguide_rounds_per_cycle=args.memguide_rounds_per_cycle,
            experience_before_memguide=bool(args.experience_before_memguide),
        )
        max_workers = min(args.parallelism, args.num_iterations)
        with timed_block(
            "self_evolving.run_phase",
            "execute_iterations",
            kind="phase",
        ):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                iteration_index = 0
                while iteration_index < len(phase_sequence):
                    phase_mode = phase_sequence[iteration_index]
                    phase_iterations: list[int] = []
                    while (
                        iteration_index < len(phase_sequence)
                        and phase_sequence[iteration_index] == phase_mode
                    ):
                        phase_iterations.append(iteration_index + 1)
                        iteration_index += 1

                    log_info(
                        "self_evolving",
                        (
                            f"Evolution phase start | mode={phase_mode} | "
                            f"iteration_range={phase_iterations[0]}-{phase_iterations[-1]} | "
                            f"count={len(phase_iterations)}"
                        ),
                    )

                    if phase_mode == "experience":
                        for scheduled_iteration in phase_iterations:
                            run_single_experience_evolution_attempt(
                                iteration=scheduled_iteration,
                                total_iterations=args.num_iterations,
                                seed=args.seed,
                                runtime=runtime,
                            )
                    else:
                        memguide_modes = _build_iteration_modes(
                            total_iterations=len(phase_iterations),
                            optimization_encourage_exploration_enabled=optimization_encourage_exploration_enabled,
                            exploration_over_expansion=exploration_over_expansion,
                        )
                        memguide_index = 0
                        while memguide_index < len(memguide_modes):
                            memguide_mode = memguide_modes[memguide_index]
                            mode_iterations: list[int] = []
                            while (
                                memguide_index < len(memguide_modes)
                                and memguide_modes[memguide_index] == memguide_mode
                            ):
                                mode_iterations.append(phase_iterations[memguide_index])
                                memguide_index += 1

                            if memguide_mode == "exploration":
                                for scheduled_iteration in mode_iterations:
                                    run_single_exploration_attempt(
                                        iteration=scheduled_iteration,
                                        total_iterations=args.num_iterations,
                                        runtime=runtime,
                                    )
                            else:
                                _run_rollout_expansion_phase(
                                    executor=executor,
                                    phase_iterations=mode_iterations,
                                    total_iterations=args.num_iterations,
                                    seed=args.seed,
                                    runtime=runtime,
                                    parallelism=args.parallelism,
                                )
    except Exception as exc:
        exit_code = 1
        run_error = str(exc)
        print(f"Error: {exc}")
    finally:
        finish_timing_run(
            status="success" if exit_code == 0 else "error",
            error=run_error,
        )
        try:
            report_path = write_timing_report()
            log_info(
                "self_evolving",
                f"Timing report written | path={report_path}",
            )
        except Exception as report_exc:
            log_info(
                "self_evolving",
                f"Failed to write timing report | error={report_exc}",
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
