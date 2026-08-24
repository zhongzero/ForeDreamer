#!/usr/bin/env python3

import ast
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.logger import log_info


def parse_ground_truth(value):
    """Parse FutureX ground-truth strings into a normalized list of options."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return []
        try:
            parsed = ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            return [normalized]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def load_futurex_dataset(input_path: str, run_specific: str | None = None) -> List[Dict]:
    """Load FutureX rows from parquet/csv and apply optional sample-id filtering."""
    dataset_path = Path(input_path)
    log_info("data.futurex", f"Reading FutureX data from {dataset_path}")

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    log_info("data.futurex", f"Loaded {len(df)} samples")

    if run_specific:
        sample_ids = [item.strip() for item in run_specific.split(",")]
        df = df[df["id"].astype(str).isin(sample_ids)]
        log_info("data.futurex", f"Filtered to {len(df)} specific samples: {sample_ids}")
        if df.empty:
            return []

    df = df.copy()
    df["ground_truth"] = df["ground_truth"].apply(parse_ground_truth)
    return df.to_dict("records")
