#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.serialization import parse_serialized_data


def compute_futurex_evaluation(parsed_answer: list[str], ground_truth: list[str]) -> dict[str, Any]:
    normalized_pred = sorted(str(value).strip() for value in parsed_answer if str(value).strip())
    normalized_gt = sorted(str(value).strip() for value in ground_truth if str(value).strip())
    return {
        "dataset": "futurex",
        "accuracy": normalized_pred == normalized_gt,
    }


def compute_prophet_arena_evaluation(
    *,
    event_data: dict[str, Any],
    probabilities: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    from pm_rank.data.base import ForecastProblem, ProphetArenaForecastEvent
    from pm_rank.data.loaders import ProphetArenaChallengeLoader
    from pm_rank.model.average_return import AverageReturn, AverageReturnConfig
    from pm_rank.model.scoring_rule import BrierScoringRule

    markets = event_data["markets"]
    if isinstance(markets, str):
        markets = parse_serialized_data(markets, "markets")

    market_info = event_data.get("market_info", {})
    if isinstance(market_info, str):
        market_info = parse_serialized_data(market_info, "market_info")

    market_outcome = event_data["market_outcome"]
    if isinstance(market_outcome, str):
        market_outcome = parse_serialized_data(market_outcome, "market_outcome")

    problem_option_keys = list(market_info.keys())
    correct_option_idx = [
        i for i, key in enumerate(problem_option_keys)
        if market_outcome.get(key, 0) == 1
    ]

    probs_dict = {str(item["market"]): float(item["probability"]) for item in probabilities}
    unnormalized_probs = [max(0.0, min(1.0, probs_dict.get(opt, 0.0))) for opt in problem_option_keys]
    normalized_probs = ProphetArenaChallengeLoader._get_normalized_probs(unnormalized_probs)
    odds = ProphetArenaChallengeLoader._calculate_implied_probs_for_problem(
        market_info,
        problem_option_keys,
        use_bid_for_odds=False,
        yes_contract=True,
    )
    no_odds = ProphetArenaChallengeLoader._calculate_implied_probs_for_problem(
        market_info,
        problem_option_keys,
        use_bid_for_odds=False,
        yes_contract=False,
    )

    close_time_raw = event_data.get("close_time")
    end_time = (
        datetime.fromisoformat(str(close_time_raw).replace("Z", "+00:00"))
        if close_time_raw
        else datetime.now()
    )
    forecast = ProphetArenaForecastEvent(
        forecast_id=str(event_data.get("submission_id", event_data.get("event_ticker", ""))),
        problem_id=str(event_data["event_ticker"]),
        submission_id=str(event_data["submission_id"]),
        username=model_name,
        timestamp=datetime.now(),
        probs=normalized_probs,
        unnormalized_probs=unnormalized_probs,
        odds=odds,
        no_odds=no_odds,
    )
    problem = ForecastProblem(
        title=str(event_data["title"]),
        problem_id=str(event_data["event_ticker"]),
        options=problem_option_keys,
        correct_option_idx=correct_option_idx,
        forecasts=[forecast],
        end_time=end_time,
        num_forecasters=1,
        url=None,
        category=event_data.get("category"),
    )

    brier_keep_higher = BrierScoringRule().fit([problem], include_scores=True)[0][model_name]
    brier = 1 - float(brier_keep_higher)

    average_return_config = AverageReturnConfig(
        num_money_per_round=1,
        use_approximate=True,
        risk_aversion=0.0,
        use_binary_reduction=True,
    )
    average_return = float(
        AverageReturn(config=average_return_config).fit([problem], include_scores=True)[0][model_name]
    )

    return {
        "dataset": "prophet_arena",
        "brier": brier,
        "average_return": average_return,
    }
