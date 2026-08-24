#!/usr/bin/env python3

"""Create a compact, standalone score summary from ForeDreamer test output."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=("futurex", "prophet_arena"),
    )
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--input_path")
    parser.add_argument(
        "--train_data_path",
        help="Training CSV/parquet used to report scores after removing train overlap",
    )
    parser.add_argument("--mem_guide")
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value in exact_match: {value!r}")


def _models(frame: pd.DataFrame) -> list[str]:
    if "model" not in frame.columns:
        return []
    return sorted(
        {
            str(model).strip()
            for model in frame["model"].dropna().tolist()
            if str(model).strip()
        }
    )


def _counts(frame: pd.DataFrame) -> dict[str, int | float]:
    total = len(frame)
    if "status" in frame.columns:
        success = int(frame["status"].astype(str).str.lower().eq("success").sum())
    else:
        success = total
    return {
        "total": total,
        "success": success,
        "error": total - success,
        "success_rate": success / total if total else 0.0,
    }


def _futurex_scores(frame: pd.DataFrame) -> dict[str, int | float]:
    if "exact_match" not in frame.columns:
        raise ValueError("FutureX predictions are missing the exact_match column")
    exact_matches = int(frame["exact_match"].map(_as_bool).sum())
    total = len(frame)
    return {
        "exact_match_accuracy": exact_matches / total if total else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_total": total,
    }


def _prophet_arena_scores(frame: pd.DataFrame) -> dict[str, int | float | None]:
    missing = {"brier", "average_return"} - set(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Prophet Arena predictions are missing score columns: {columns}")

    brier = pd.to_numeric(frame["brier"], errors="coerce").dropna()
    average_return = pd.to_numeric(frame["average_return"], errors="coerce").dropna()
    return {
        "mean_brier": float(brier.mean()) if not brier.empty else None,
        "brier_scored_count": int(len(brier)),
        "mean_average_return": (
            float(average_return.mean()) if not average_return.empty else None
        ),
        "average_return_scored_count": int(len(average_return)),
    }


def _score_scope(frame: pd.DataFrame, dataset_type: str) -> dict[str, Any]:
    scores = (
        _futurex_scores(frame)
        if dataset_type == "futurex"
        else _prophet_arena_scores(frame)
    )
    return {
        "counts": _counts(frame),
        "scores": scores,
    }


def _load_data_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _normalized_keys(series: pd.Series) -> pd.Series:
    return series.map(lambda value: "" if pd.isna(value) else str(value).strip())


def build_summary(
    frame: pd.DataFrame,
    *,
    dataset_type: str,
    predictions_csv: Path,
    input_path: str | None,
    mem_guide: str | None,
    train_frame: pd.DataFrame | None = None,
    train_data_path: str | None = None,
) -> dict[str, Any]:
    all_test = _score_scope(frame, dataset_type)
    summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": dataset_type,
        "input_path": str(Path(input_path).resolve()) if input_path else None,
        "train_data_path": (
            str(Path(train_data_path).resolve()) if train_data_path else None
        ),
        "predictions_csv": str(predictions_csv.resolve()),
        "mem_guide": mem_guide,
        "models": _models(frame),
        # Backward-compatible aliases: these always describe the complete test input.
        "counts": all_test["counts"],
        "scores": all_test["scores"],
        "all_test": all_test,
        "train_overlap": None,
        "test_without_train": None,
    }

    if train_frame is None:
        return summary

    key_field = "id" if dataset_type == "futurex" else "submission_id"
    if key_field not in frame.columns:
        raise ValueError(f"Predictions are missing the dataset key column: {key_field}")
    if key_field not in train_frame.columns:
        raise ValueError(f"Training data are missing the dataset key column: {key_field}")

    train_keys = set(_normalized_keys(train_frame[key_field]))
    prediction_keys = _normalized_keys(frame[key_field])
    overlap_mask = prediction_keys.isin(train_keys)
    held_out_frame = frame.loc[~overlap_mask].copy()
    summary["train_overlap"] = {
        "key_field": key_field,
        "train_rows": int(len(train_frame)),
        "train_unique_keys": int(len(train_keys)),
        "test_rows_in_train": int(overlap_mask.sum()),
        "test_rows_without_train": int((~overlap_mask).sum()),
    }
    summary["test_without_train"] = _score_scope(held_out_frame, dataset_type)
    return summary


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")

    frame = pd.read_csv(predictions_csv)
    train_frame = None
    if args.train_data_path:
        train_frame = _load_data_frame(Path(args.train_data_path))
    summary = build_summary(
        frame,
        dataset_type=args.dataset_type,
        predictions_csv=predictions_csv,
        input_path=args.input_path,
        mem_guide=args.mem_guide,
        train_frame=train_frame,
        train_data_path=args.train_data_path,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output_json)

    print(f"Test score summary: {output_json}")
    if args.dataset_type == "futurex":
        scores = summary["all_test"]["scores"]
        print(
            "FutureX all test exact match: "
            f"{scores['exact_match_count']}/{scores['exact_match_total']} "
            f"({scores['exact_match_accuracy']:.6f})"
        )
        if summary["test_without_train"] is not None:
            held_out_scores = summary["test_without_train"]["scores"]
            print(
                "FutureX test without train exact match: "
                f"{held_out_scores['exact_match_count']}/"
                f"{held_out_scores['exact_match_total']} "
                f"({held_out_scores['exact_match_accuracy']:.6f})"
            )
    else:
        scores = summary["all_test"]["scores"]
        print(
            "Prophet Arena all test: "
            f"mean_brier={scores['mean_brier']} "
            f"mean_average_return={scores['mean_average_return']}"
        )
        if summary["test_without_train"] is not None:
            held_out_scores = summary["test_without_train"]["scores"]
            print(
                "Prophet Arena test without train: "
                f"mean_brier={held_out_scores['mean_brier']} "
                f"mean_average_return={held_out_scores['mean_average_return']}"
            )
    if summary["train_overlap"] is not None:
        overlap = summary["train_overlap"]
        print(
            "Train/test overlap: "
            f"{overlap['test_rows_in_train']} removed, "
            f"{overlap['test_rows_without_train']} held out"
        )


if __name__ == "__main__":
    main()
