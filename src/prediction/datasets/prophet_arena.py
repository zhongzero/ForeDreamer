#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.logger import log_info
from utils.serialization import parse_serialized_data


def load_events_from_csv(input_csv: str, run_specific: str | None = None) -> List[Dict]:
    """Load Prophet Arena event rows from CSV and apply optional ticker filtering."""
    input_path = Path(input_csv)
    log_info("data.prophet_arena", f"Reading events from {input_path}")
    df = pd.read_csv(input_path)
    log_info("data.prophet_arena", f"Loaded {len(df)} events")

    if run_specific:
        event_tickers = [ticker.strip() for ticker in run_specific.split(",")]
        df = df[df["event_ticker"].isin(event_tickers)]
        log_info("data.prophet_arena", f"Filtered to {len(df)} specific events: {event_tickers}")

        if df.empty:
            return []

    if "market_data" in df.columns:
        df.rename(columns={"market_data": "market_info"}, inplace=True)
        log_info("data.prophet_arena", "Backward compatibility: renamed `market_data` column to `market_info`")

    return df.to_dict("records")
