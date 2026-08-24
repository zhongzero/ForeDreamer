#!/usr/bin/env python3

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from core.rollout import pop_rollout, register_rollout, save_rollout
from core.rollout import update_rollout_outcome
from core.shared_llm_client import (
    PromptStrategyConfig,
    SharedLLMClient,
    ToolRuntimeContext,
    WEB_SEARCH_LOOP_STRATEGY,
    normalize_search_before_exclusive_date,
)
from experience_bank import render_experience_bank_prompt_section
from prediction.datasets.prophet_arena import parse_serialized_data
from prediction.metrics import compute_prophet_arena_evaluation
from utils.logger import log_info


def normalize_market_text(text: str) -> str:
    """Normalize market text to keep apostrophe handling consistent."""
    return text.replace("'", "\u2019")


@dataclass
class MarketPrediction:
    market: str
    probability: float


@dataclass
class PredictionOutput:
    probabilities: List[MarketPrediction]
    rationale: str


@dataclass
class ProphetArenaPromptParts:
    question_context: str
    sources_context: str
    market_context: str
    final_answer_requirements: str


class PredictionPrompts:
    """Prompts for Prophet-Arena market prediction tasks."""

    @staticmethod
    def create_question_context(event_title: str, market_names: List[str]) -> str:
        market_list_str = "\n".join(f"- {market}" for market in market_names)
        return (
            f'Forecast the future event: "{event_title}".\n\n'
            "Possible outcomes:\n"
            f"{market_list_str}"
        )

    @staticmethod
    def create_sources_context(sources: str) -> str:
        sections = [
            "You are given curated sources with summaries and rankings.",
            "Lower ranking numbers should generally be weighted more heavily.",
            "Sources:\n" + sources,
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def create_market_context(market_stats: Optional[dict] = None) -> str:
        sections = []
        if market_stats:
            sections.append(
                "Prediction-market context (useful but not sufficient on its own):\n"
                + json.dumps(market_stats, indent=2)
            )
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def create_final_answer_requirements(
        market_names: List[str],
        strategy: str = "single_call",
    ) -> str:
        json_example = ",\n    ".join(
            f'"{market}": <probability_value_from_0_to_1>' for market in market_names
        )
        rationale_placeholder = (
            "<a detailed explanation that cites and evaluates the most relevant evidence>"
            if strategy == WEB_SEARCH_LOOP_STRATEGY
            else "<three concise sentences>"
        )
        rationale_requirement = (
            "The rationale should evaluate the most relevant evidence returned by web_search_and_process, explicitly cite the key pieces of evidence, explain why they are relevant or limited, discuss uncertainties or conflicting signals, and justify the final probability assignment in detail."
            if strategy == WEB_SEARCH_LOOP_STRATEGY
            else "The rationale should explain the key considerations, uncertainties, and the probability assignment."
        )
        return (
            "Return exactly one JSON object with this schema and no markdown fences:\n"
            "{\n"
            f'  "rationale": "{rationale_placeholder}",\n'
            '  "probabilities": {\n'
            f"    {json_example}\n"
            "  }\n"
            "}\n\n"
            "Requirements:\n"
            "- Use exactly the listed outcome names, case-sensitive.\n"
            "- Do not invent additional outcomes.\n"
            "- Every listed outcome must appear exactly once.\n"
            "- Every probability must be a number between 0 and 1.\n"
            f"- {rationale_requirement}"
        )

    @staticmethod
    def build_prompt(
        parts: ProphetArenaPromptParts,
        strategy: str,
        strategy_config: Optional[PromptStrategyConfig] = None,
        retry_instruction: Optional[str] = None,
        use_sources: bool = False,
        use_market_data: bool = False,
    ) -> str:
        has_context = use_sources or use_market_data
        if strategy == WEB_SEARCH_LOOP_STRATEGY:
            search_note = (
                f"If you call web_search_and_process, results are restricted to sources published on or before {strategy_config.search_before}."
                if strategy_config and strategy_config.search_before
                else "If you call web_search_and_process, there is no publication-date cutoff."
            )
            strategy_context = (
                "You are an AI forecasting assistant. You may reason over multiple turns and use web_search_and_process "
                "to gather additional evidence when useful. The web_search_and_process tool returns processed evidence, not raw search results. "
                f"{search_note} In your final rationale, evaluate and cite the most relevant processed evidence instead of giving only a brief generic explanation. Only your final response should be the required JSON object."
            )
        else:
            if has_context:
                strategy_context = (
                    "You are an AI forecasting assistant. Use the provided evidence to estimate the probability of each outcome "
                    "and return the final JSON answer directly."
                )
            else:
                strategy_context = (
                    "You are an AI forecasting assistant. Estimate the probability of each outcome and return the final JSON answer directly."
                )

        sections = [
            strategy_context,
        ]
        experience_section = render_experience_bank_prompt_section(
            list(getattr(strategy_config, "experience_entries", ()) or ())
        )
        if experience_section:
            sections.append(experience_section)
        sections.append("Question:\n" + parts.question_context)
        if use_market_data and parts.market_context:
            sections.append("Market context:\n" + parts.market_context)
        if use_sources and parts.sources_context:
            sections.append("Evidence:\n" + parts.sources_context)
        sections.append("Final answer requirements:\n" + parts.final_answer_requirements)
        if retry_instruction:
            sections.append(retry_instruction)
        return "\n\n".join(section for section in sections if section)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""


class ProphetArenaRunner(SharedLLMClient):
    """Prophet-Arena predictor using the shared prompt-based LLM client."""

    def _validate_response(self, dynamic_out: dict, expected_markets: list) -> tuple[bool, str]:
        if not isinstance(dynamic_out, dict):
            return False, "Response is not a valid dictionary"
        if "probabilities" not in dynamic_out:
            return False, "Missing 'probabilities' field in response"
        if "rationale" not in dynamic_out:
            return False, "Missing 'rationale' field in response"

        probs = dynamic_out["probabilities"]
        if not isinstance(probs, dict):
            return False, "Probabilities field is not a dictionary"

        response_markets = set(normalize_market_text(market) for market in probs.keys())
        expected_markets_set = set(normalize_market_text(market) for market in expected_markets)

        extra_markets = response_markets - expected_markets_set
        if extra_markets:
            return False, f"Model hallucinated extra markets: {list(extra_markets)}"

        missing_markets = expected_markets_set - response_markets
        if missing_markets:
            return False, f"Model failed to provide probabilities for markets: {list(missing_markets)}"

        for market, prob in probs.items():
            if not isinstance(prob, (int, float)):
                return False, f"Invalid probability type for {market}: {type(prob)}"
            if not (0 <= prob <= 1):
                return False, f"Probability for {market} out of range [0,1]: {prob}"

        return True, ""

    async def predict_event_async(
        self,
        event_title: str,
        markets: List[str],
        sources: List[Dict],
        market_stats: Dict = None,
        strategy: str = "single_call",
        strategy_config: Optional[PromptStrategyConfig] = None,
        use_sources: bool = False,
        use_market_data: bool = False,
        tool_runtime_context: Optional[ToolRuntimeContext] = None,
    ) -> PredictionOutput:
        parts = ProphetArenaPromptParts(
            question_context=PredictionPrompts.create_question_context(event_title, markets),
            sources_context=(
                PredictionPrompts.create_sources_context(self._format_sources(sources))
                if use_sources
                else ""
            ),
            market_context=(
                PredictionPrompts.create_market_context(market_stats)
                if use_market_data
                else ""
            ),
            final_answer_requirements=PredictionPrompts.create_final_answer_requirements(
                markets,
                strategy=strategy,
            ),
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                retry_instruction = None
                # if attempt > 0:
                #     retry_instruction = (
                #         "The previous response had validation errors. Return a corrected final answer that strictly obeys the required JSON schema "
                #         "and includes exactly the listed markets."
                #     )

                prompt = PredictionPrompts.build_prompt(
                    parts=parts,
                    strategy=strategy,
                    strategy_config=strategy_config,
                    retry_instruction=retry_instruction,
                    use_sources=use_sources,
                    use_market_data=use_market_data,
                )
                response_text = await super().run_prompt_async(
                    prompt,
                    strategy=strategy,
                    strategy_config=strategy_config,
                    tool_runtime_context=tool_runtime_context,
                )

                json_start = response_text.find("{")
                if json_start > 0:
                    response_text = response_text[json_start:]

                json_end = response_text.rfind("}")
                if json_end != -1 and json_end < len(response_text) - 1:
                    response_text = response_text[: json_end + 1]

                dynamic_out = json.loads(response_text)
                is_valid, error_msg = self._validate_response(dynamic_out, markets)
                if not is_valid:
                    if attempt < max_retries - 1:
                        continue
                    raise LLMError(f"Validation failed after {max_retries} attempts: {error_msg}")

                flat_probs: dict[str, float] = dynamic_out["probabilities"]
                preds = [
                    MarketPrediction(market=market_name, probability=probability)
                    for market_name, probability in flat_probs.items()
                ]
                return PredictionOutput(
                    probabilities=preds,
                    rationale=dynamic_out["rationale"],
                )

            except json.JSONDecodeError as exc:
                if attempt < max_retries - 1:
                    continue
                raise LLMError(
                    f"Failed to parse JSON response after {max_retries} attempts: {str(exc)}"
                ) from exc
            except Exception as exc:
                if attempt < max_retries - 1:
                    continue
                raise LLMError(f"Failed to get prediction: {str(exc)}") from exc

    def _format_sources(self, sources: List[Dict]) -> str:
        if not sources:
            return "No sources available for this event."

        formatted_sources = []
        for source in sources:
            source_text = f"Source {source.get('ranking', 'N/A')}: {source.get('title', 'No title')}\n"
            source_text += f"URL: {source.get('url', 'No URL')}\n"
            source_text += f"Summary: {source.get('summary', 'No summary')}\n"
            formatted_sources.append(source_text)

        return "\n---\n".join(formatted_sources)


def extract_market_stats(event_data: Dict) -> Dict | None:
    if "market_info" not in event_data or not event_data["market_info"]:
        return None

    try:
        market_info_raw = event_data["market_info"]
        market_info = (
            parse_serialized_data(market_info_raw, "market_info")
            if isinstance(market_info_raw, str)
            else market_info_raw
        )
        if not market_info:
            return None

        market_stats = {}
        for market_title, market_data in market_info.items():
            market_stats[market_title] = {
                "last_price": market_data.get("last_price"),
                "yes_ask": market_data.get("yes_ask"),
                "no_ask": market_data.get("no_ask"),
            }
        return market_stats
    except (ValueError, TypeError):
        return None


def resolve_prophet_arena_strategy_config(
    event_data: Dict,
    strategy_config: Optional[PromptStrategyConfig] = None,
) -> PromptStrategyConfig:
    base_config = strategy_config or PromptStrategyConfig()
    dataset_cutoff = normalize_search_before_exclusive_date(event_data.get("snapshot_time"))
    if not dataset_cutoff:
        raise ValueError(
            "Missing or invalid snapshot_time for submission "
            f"{event_data.get('submission_id', '')}: {event_data.get('snapshot_time', '')}"
        )
    log_info(
        "prophet_arena.search",
        (
            f"Resolved search cutoff | submission_id={event_data.get('submission_id', '')} "
            f"| snapshot_time={event_data.get('snapshot_time', '')} | exclusive_search_before={dataset_cutoff}"
        ),
    )
    return replace(base_config, search_before=dataset_cutoff)


async def process_events_async(
    events_data: List[Dict],
    predictor: ProphetArenaRunner,
    strategy: str = "single_call",
    strategy_config: Optional[PromptStrategyConfig] = None,
    use_sources: bool = False,
    use_market_data: bool = False,
) -> List[Dict]:
    should_save_rollout = bool((strategy_config or PromptStrategyConfig()).save_rollout)
    tasks = [
        process_prophet_arena_event_async(
            event_data=event,
            predictor=predictor,
            strategy=strategy,
            strategy_config=strategy_config,
            use_sources=use_sources,
            use_market_data=use_market_data,
            should_save_rollout=should_save_rollout,
        )
        for event in events_data
    ]
    return await asyncio.gather(*tasks)


async def process_prophet_arena_event_async(
    *,
    event_data: Dict,
    predictor: ProphetArenaRunner,
    strategy: str = "single_call",
    strategy_config: Optional[PromptStrategyConfig] = None,
    use_sources: bool = False,
    use_market_data: bool = False,
    should_save_rollout: Optional[bool] = None,
    execution_id_override: Optional[str] = None,
) -> Dict:
    markets = []
    sample_id = str(event_data["submission_id"])
    execution_id = str(execution_id_override or sample_id).strip()
    if not execution_id:
        raise ValueError("Prophet Arena execution_id must be non-empty")

    resolved_should_save_rollout = bool(
        (strategy_config or PromptStrategyConfig()).save_rollout
        if should_save_rollout is None
        else should_save_rollout
    )

    result: Dict
    try:
        markets = event_data["markets"]
        if isinstance(markets, str):
            markets = parse_serialized_data(markets, "markets")

        sources = event_data.get("sources", [])
        if isinstance(sources, str):
            sources = parse_serialized_data(sources, "sources")

        market_stats = extract_market_stats(event_data)
        event_strategy_config = resolve_prophet_arena_strategy_config(event_data, strategy_config)
        problem_statement = PredictionPrompts.create_question_context(event_data["title"], markets)
        task_requirements = PredictionPrompts.create_final_answer_requirements(
            markets,
            strategy=strategy,
        )
        register_rollout(
            execution_id,
            sample_identifier=sample_id,
            problem_statement=problem_statement,
            task_requirements=task_requirements,
        )
        prediction = await predictor.predict_event_async(
            event_title=event_data["title"],
            markets=markets,
            sources=sources,
            market_stats=market_stats,
            strategy=strategy,
            strategy_config=event_strategy_config,
            use_sources=use_sources,
            use_market_data=use_market_data,
            tool_runtime_context=ToolRuntimeContext(
                task_id=execution_id,
                problem_statement=problem_statement,
                task_requirements=task_requirements,
                sample_identifier=sample_id,
            ),
        )

        complete_prediction = {
            "probabilities": [
                {"market": pred.market, "probability": pred.probability}
                for pred in prediction.probabilities
            ],
            "rationale": prediction.rationale,
        }
        ground_truth = event_data.get("market_outcome", {})
        if isinstance(ground_truth, str):
            ground_truth = parse_serialized_data(ground_truth, "market_outcome")
        evaluation = compute_prophet_arena_evaluation(
            event_data=event_data,
            probabilities=complete_prediction["probabilities"],
            model_name=predictor.model,
        )
        update_rollout_outcome(
            execution_id,
            ground_truth=ground_truth,
            evaluation=evaluation,
        )
        log_info(
            "metric.prophet_arena",
            (
                f"Task metrics | task_id={execution_id} | "
                f"brier={evaluation['brier']:.6f} | average_return={evaluation['average_return']:.6f}"
            ),
        )

        result = {
            "event_ticker": event_data["event_ticker"],
            "submission_id": event_data["submission_id"],
            "title": event_data["title"],
            "category": event_data.get("category", ""),
            "markets": json.dumps(markets),
            "prediction": json.dumps(complete_prediction),
            "brier": evaluation["brier"],
            "average_return": evaluation["average_return"],
            "model": predictor.model,
            "status": "success",
        }
    except Exception as exc:
        ground_truth = event_data.get("market_outcome", {})
        if isinstance(ground_truth, str):
            ground_truth = parse_serialized_data(ground_truth, "market_outcome")
        evaluation = {
            "dataset": "prophet_arena",
            "brier": None,
            "average_return": None,
        }
        update_rollout_outcome(
            execution_id,
            ground_truth=ground_truth,
            evaluation=evaluation,
        )
        log_info(
            "metric.prophet_arena",
            f"Task metrics | task_id={execution_id} | brier=None | average_return=None",
        )
        result = {
            "event_ticker": event_data["event_ticker"],
            "submission_id": event_data["submission_id"],
            "title": event_data.get("title", ""),
            "category": event_data.get("category", ""),
            "markets": json.dumps(markets) if markets else "",
            "prediction": "",
            "rationale": "",
            "brier": None,
            "average_return": None,
            "model": predictor.model,
            "status": "error",
            "error_message": str(exc),
        }
    rollout = pop_rollout(execution_id)
    if resolved_should_save_rollout:
        rollout_path = save_rollout(rollout)
        log_info("rollout", f'Saved rollout | task_id={execution_id} | path="{rollout_path}"')
        result["rollout"] = json.dumps(rollout, ensure_ascii=False)
        result["rollout_path"] = rollout_path
    return result
