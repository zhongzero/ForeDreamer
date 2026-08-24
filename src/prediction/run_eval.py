#!/usr/bin/env python3
"""
Standalone prediction script that can run predictions on event data from CSV or parquet files.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experience_bank import load_experience_bank_from_file
from core.shared_llm_client import SUPPORTED_PROMPT_STRATEGIES
from prediction.runtime import (
    build_dataset_runner,
    build_prompt_strategy_config,
    load_dataset_records,
    run_dataset_events_sync,
)
from utils.generated_paths import (
    add_generated_path_arguments,
    configure_generated_path_env_from_namespace,
    ensure_mem_asset_bootstrap_from_namespace,
)
from utils.logger import log_info
from utils.search_provider import DEFAULT_SEARCH_PROVIDER, SUPPORTED_SEARCH_PROVIDERS


VALID_RESUME_STATUSES = {"success", "error"}


def _normalize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _resume_key_field(dataset_type: str) -> str:
    if dataset_type == "futurex":
        return "id"
    if dataset_type == "prophet_arena":
        return "submission_id"
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def _input_record_signature(dataset_type: str, event_data: Dict) -> tuple[str, ...]:
    if dataset_type == "futurex":
        return (
            _normalize_value(event_data.get("id")),
            _normalize_value(event_data.get("title")),
            _normalize_value(event_data.get("end_time")),
            _normalize_value(event_data.get("level")),
        )

    return (
        _normalize_value(event_data.get("submission_id")),
        _normalize_value(event_data.get("event_ticker")),
        _normalize_value(event_data.get("title")),
        _normalize_value(event_data.get("category")),
    )


def _output_record_signature(dataset_type: str, row: pd.Series) -> tuple[str, ...]:
    if dataset_type == "futurex":
        return (
            _normalize_value(row.get("id")),
            _normalize_value(row.get("title")),
            _normalize_value(row.get("end_time")),
            _normalize_value(row.get("level")),
        )

    return (
        _normalize_value(row.get("submission_id")),
        _normalize_value(row.get("event_ticker")),
        _normalize_value(row.get("title")),
        _normalize_value(row.get("category")),
    )


def _build_event_map(dataset_type: str, events_data: List[Dict]) -> Dict[str, Dict]:
    key_field = _resume_key_field(dataset_type)
    event_map: Dict[str, Dict] = {}
    for event_data in events_data:
        key = _normalize_value(event_data.get(key_field))
        if not key:
            raise ValueError(f"Missing key field `{key_field}` in input data")
        if key in event_map:
            raise ValueError(f"Duplicate key `{key}` found in input data for `{key_field}`")
        event_map[key] = event_data
    return event_map


def _validate_resume_output(
    dataset_type: str,
    events_data: List[Dict],
    existing_df: pd.DataFrame,
) -> tuple[bool, str]:
    key_field = _resume_key_field(dataset_type)
    if key_field not in existing_df.columns:
        return False, f"Existing output is missing required key column `{key_field}`"
    if "status" not in existing_df.columns:
        return False, "Existing output is missing required `status` column"

    event_map = _build_event_map(dataset_type, events_data)
    if len(existing_df) != len(event_map):
        return (
            False,
            f"Existing output row count ({len(existing_df)}) does not match input row count ({len(event_map)})",
        )

    seen_output_keys: set[str] = set()
    invalid_statuses: set[str] = set()
    for _, row in existing_df.iterrows():
        key = _normalize_value(row.get(key_field))
        if not key:
            return False, f"Existing output contains empty `{key_field}` values"
        if key in seen_output_keys:
            return False, f"Existing output contains duplicate `{key_field}` value `{key}`"
        seen_output_keys.add(key)

        status = _normalize_value(row.get("status"))
        if status not in VALID_RESUME_STATUSES:
            invalid_statuses.add(status or "<empty>")

        expected_event = event_map.get(key)
        if expected_event is None:
            return False, f"Existing output contains `{key_field}`={key} which is not present in the input data"

        if _output_record_signature(dataset_type, row) != _input_record_signature(dataset_type, expected_event):
            return False, f"Existing output row for `{key_field}`={key} does not match the current input data"

    if invalid_statuses:
        invalid_display = ", ".join(sorted(invalid_statuses))
        return False, (
            "Existing output contains invalid status values. "
            f"Expected only `success` or `error`, got: {invalid_display}"
        )

    missing_keys = set(event_map) - seen_output_keys
    if missing_keys:
        preview = ", ".join(sorted(missing_keys)[:5])
        return False, f"Existing output is missing {len(missing_keys)} input rows, e.g. {preview}"

    return True, ""


def _run_events(
    *,
    args: argparse.Namespace,
    events_data: List[Dict],
    runner,
    strategy_config,
) -> List[Dict]:
    if not events_data:
        return []

    if args.run_all and len(events_data) > 1:
        if args.dataset_type == "futurex":
            print("Running async processing for multiple FutureX samples...")
        else:
            print("Running async processing for multiple events...")
        return run_dataset_events_sync(
            args.dataset_type,
            events_data,
            runner,
            strategy=args.strategy,
            strategy_config=strategy_config,
            use_sources=args.use_source_in_prophet_arena,
            use_market_data=args.use_market_data_in_prophet_arena,
        )

    print("Processing...")
    results = []
    for i, event_data in enumerate(events_data):
        if args.dataset_type == "futurex":
            print(f"Processing sample {i + 1}/{len(events_data)}: {event_data['id']}")
        else:
            print(f"Processing event {i + 1}/{len(events_data)}: {event_data['event_ticker']}")
        result = run_dataset_events_sync(
            args.dataset_type,
            [event_data],
            runner,
            strategy=args.strategy,
            strategy_config=strategy_config,
            use_sources=args.use_source_in_prophet_arena,
            use_market_data=args.use_market_data_in_prophet_arena,
        )
        results.extend(result)
    return results


def _merge_resumed_results(
    dataset_type: str,
    existing_df: pd.DataFrame,
    rerun_results: List[Dict],
) -> pd.DataFrame:
    if not rerun_results:
        return existing_df

    key_field = _resume_key_field(dataset_type)
    existing_clean = existing_df.drop(columns=["rollout", "rollout_path"], errors="ignore").copy()
    replacement_df = pd.DataFrame(rerun_results).drop(columns=["rollout", "rollout_path"], errors="ignore")

    replacement_records = {}
    for record in replacement_df.to_dict("records"):
        replacement_records[_normalize_value(record.get(key_field))] = record

    all_columns = list(dict.fromkeys([*existing_clean.columns.tolist(), *replacement_df.columns.tolist()]))
    merged_records = []
    for _, row in existing_clean.iterrows():
        existing_record = row.to_dict()
        key = _normalize_value(existing_record.get(key_field))
        record = replacement_records.get(key, existing_record)
        merged_records.append({column: record.get(column, pd.NA) for column in all_columns})

    return pd.DataFrame(merged_records, columns=all_columns)


def main():
    parser = argparse.ArgumentParser(description="Standalone event prediction script")
    parser.add_argument("--input_csv", "-i", required=True, help="Input CSV or parquet file with event data")
    parser.add_argument("--output_csv", "-o", required=True, help="Output CSV file for predictions")
    parser.add_argument(
        "--resume_from_output_csv",
        action="store_true",
        help=(
            "Resume from an existing output CSV. If the file exists and matches the current input data, "
            "only rows with status=error will be re-run and updated."
        ),
    )
    parser.add_argument(
        "--api_key",
        "-k",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        help="OpenAI-compatible API key (default: $OPENAI_API_KEY or $OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--base_url",
        "-u",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (default: $OPENAI_BASE_URL or OpenAI)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("OPENAI_MODEL"),
        help="Model to use for predictions (default: $OPENAI_MODEL)",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["prophet_arena", "futurex"],
        default="prophet_arena",
        help="Dataset format to load (default: prophet_arena)",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(SUPPORTED_PROMPT_STRATEGIES),
        default="single_call",
        help="Prompt processing strategy to use",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=4,
        help="Maximum assistant turns for multi-turn strategies (default: 4)",
    )
    parser.add_argument(
        "--subagent_max_turns",
        type=int,
        default=10,
        help="Maximum assistant turns for each data-process subagent (default: 10)",
    )
    parser.add_argument(
        "--save_rollout",
        action="store_true",
        help="Save per-question rollout JSON files under src/HistoryRollout",
    )
    parser.add_argument(
        "--search_max_results",
        type=int,
        default=5,
        help="Maximum number of search results to return to the model (default: 5)",
    )
    parser.add_argument(
        "--search_max_chars_per_result",
        type=int,
        default=700,
        help="Maximum characters per search result snippet (default: 700)",
    )
    parser.add_argument(
        "--search_max_total_chars",
        type=int,
        default=2500,
        help="Maximum total characters across all search result snippets (default: 2500)",
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
        "--use_tavilty_raw_context",
        action="store_true",
        help="Use Tavily raw_content instead of the default content snippets when available",
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
            "Bypass shared LLM cache for the main agent's final answer so final-response failures "
            "are not repeatedly replayed from cache."
        ),
    )
    parser.add_argument(
        "--use_market_data_in_prophet_arena",
        action="store_true",
        help="Include prediction-market data in Prophet-Arena prompts",
    )
    parser.add_argument(
        "--use_source_in_prophet_arena",
        action="store_true",
        help="Include curated source summaries in Prophet-Arena prompts",
    )
    parser.add_argument(
        "--mem_guide",
        default="guide_initial.json",
        help="MemGuide filename under src/MemGuide/ or a path to a guide JSON file",
    )
    parser.add_argument(
        "--experience",
        nargs="?",
        const="current.json",
        default=None,
        help=(
            "Experience bank file to use under --experience_bank_dir. "
            "If provided without a value, defaults to current.json. "
            "If omitted, run_eval will not inject any ExperienceBank content."
        ),
    )
    add_generated_path_arguments(parser)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--run_all", action="store_true", help="Run predictions for all rows in the dataset")
    mode_group.add_argument(
        "--run_specific",
        type=str,
        help="Run predictions for specific event tickers or FutureX sample ids (comma-separated)",
    )

    args = parser.parse_args()
    if args.enable_lossy_search_cache:
        args.enable_api_cache = True
    if not str(args.api_key or "").strip():
        parser.error("Missing API key: set OPENAI_API_KEY/OPENROUTER_API_KEY or pass --api_key")
    if not str(args.model or "").strip():
        parser.error("Missing model: set OPENAI_MODEL or pass --model")
    if args.experience is not None and not str(args.experience_bank_dir or "").strip():
        parser.error("--experience requires an explicit --experience_bank_dir")
    configure_generated_path_env_from_namespace(args)
    bootstrap_actions = ensure_mem_asset_bootstrap_from_namespace(args)
    if bootstrap_actions:
        log_info("run_eval", f"Bootstrapped initial mem assets | actions={bootstrap_actions}")
    selected_experience_payload = None
    if args.experience is not None:
        selected_experience_payload = load_experience_bank_from_file(
            args.dataset_type,
            args.experience,
        )
    strategy_config = build_prompt_strategy_config(
        dataset_type=args.dataset_type,
        max_turns=args.max_turns,
        subagent_max_turns=args.subagent_max_turns,
        save_rollout=args.save_rollout,
        search_provider=args.search_provider,
        search_max_results=args.search_max_results,
        search_max_chars_per_result=args.search_max_chars_per_result,
        search_max_total_chars=args.search_max_total_chars,
        search_api_key=args.search_api_key,
        use_tavilty_raw_context=args.use_tavilty_raw_context,
        enable_lossy_search_cache=args.enable_lossy_search_cache,
        disable_main_agent_final_answer_cache=args.disable_main_agent_final_answer_cache,
        mem_guide=args.mem_guide,
        experience_entries=(
            list(selected_experience_payload.get("experiences", []))
            if selected_experience_payload is not None
            else None
        ),
        experience_bank_hash=(
            str(selected_experience_payload.get("bank_hash", "") or "").strip() or None
            if selected_experience_payload is not None
            else None
        ),
        experience_bank_version_id=(
            str(selected_experience_payload.get("version_id", "") or "").strip() or None
            if selected_experience_payload is not None
            else None
        ),
        enable_experience_bank=args.experience is not None,
    )

    events_data = load_dataset_records(args.dataset_type, args.input_csv, args.run_specific)

    if not events_data:
        print("No matching events found!")
        return

    runner = build_dataset_runner(
        args.dataset_type,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_question_count = len(events_data)

    results_df: pd.DataFrame
    results: List[Dict]
    if args.resume_from_output_csv and output_path.exists():
        log_info("run_eval.resume", f"Resume requested | checking existing output at {output_path}")
        existing_df = pd.read_csv(output_path)
        try:
            is_valid, validation_message = _validate_resume_output(args.dataset_type, events_data, existing_df)
        except ValueError as exc:
            log_info("run_eval.resume", f"Resume check failed | {exc}")
            raise SystemExit(1) from exc
        if not is_valid:
            log_info("run_eval.resume", f"Resume check failed | {validation_message}")
            raise SystemExit(1)

        key_field = _resume_key_field(args.dataset_type)
        existing_df = existing_df.drop(columns=["rollout", "rollout_path"], errors="ignore")
        status_series = existing_df["status"].apply(_normalize_value)
        error_keys = {
            _normalize_value(value)
            for value in existing_df.loc[status_series == "error", key_field].tolist()
        }
        success_count_existing = int((status_series == "success").sum())
        error_count_existing = int((status_series == "error").sum())
        log_info(
            "run_eval.resume",
            (
                f"Resume progress | completed={success_count_existing}/{total_question_count} "
                f"| pending={error_count_existing}"
            ),
        )

        if not error_keys:
            log_info(
                "run_eval.resume",
                (
                    f"Resume matched existing output | success={success_count_existing} "
                    f"| error={error_count_existing} | no rows need re-run"
                ),
            )
            results_df = existing_df
            results = results_df.to_dict("records")
        else:
            log_info(
                "run_eval.resume",
                (
                    f"Resume matched existing output | success={success_count_existing} "
                    f"| error={error_count_existing} | re-running error rows only"
                ),
            )
            events_to_rerun = [
                event_data
                for event_data in events_data
                if _normalize_value(event_data.get(key_field)) in error_keys
            ]
            rerun_results = _run_events(
                args=args,
                events_data=events_to_rerun,
                runner=runner,
                strategy_config=strategy_config,
            )
            results_df = _merge_resumed_results(args.dataset_type, existing_df, rerun_results)
            results = results_df.to_dict("records")
    else:
        if args.resume_from_output_csv:
            log_info(
                "run_eval.resume",
                f"Resume requested but {output_path} does not exist; starting from scratch",
            )
            log_info(
                "run_eval.resume",
                f"Resume progress | completed=0/{total_question_count} | pending={total_question_count}",
            )
        results = _run_events(
            args=args,
            events_data=events_data,
            runner=runner,
            strategy_config=strategy_config,
        )
        results_df = pd.DataFrame(results)
        results_df = results_df.drop(columns=["rollout", "rollout_path"], errors="ignore")

    results_df.to_csv(output_path, index=False)

    status_series = (
        results_df["status"].apply(_normalize_value)
        if "status" in results_df.columns
        else pd.Series([], dtype=object)
    )
    successful = int((status_series == "success").sum())
    log_info("run_eval", f"Results saved to {output_path}")
    log_info("run_eval", f"Successfully processed: {successful}/{len(results_df)} events")


if __name__ == "__main__":
    main()
