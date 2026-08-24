#!/usr/bin/env python3

import asyncio
import re
from dataclasses import replace
from dataclasses import dataclass
from typing import Dict, List, Optional

import json

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
from prediction.metrics import compute_futurex_evaluation
from utils.logger import log_block, log_info


FUTUREX_PREFIX = "You are an agent that can predict future events. The event to be predicted:"
FUTUREX_IMPORTANT_MARKER = "IMPORTANT: Your final answer MUST end with this exact format:"
FUTUREX_FINAL_CONSTRAINT_MARKER = "Do not use any other format."


class FutureXError(Exception):
    """Exception raised for FutureX inference errors."""


@dataclass(frozen=True)
class FutureXPromptParts:
    question_text: str
    answer_requirements: str
    final_constraints: str


BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^}]*)\}")


def parse_boxed_answer(response_text: str) -> List[str]:
    if not response_text:
        return []

    matches = BOXED_ANSWER_RE.findall(response_text)
    if not matches:
        return []

    raw = matches[-1].strip()
    if not raw:
        return []

    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_rationale_text(response_text: str) -> str:
    if not response_text:
        return ""

    matches = list(BOXED_ANSWER_RE.finditer(response_text))
    if not matches:
        return response_text.strip()

    prefix = response_text[: matches[-1].start()].strip()
    prefix = re.sub(r"^\s*\**(?:reasoning|rationale)\**\s*:\s*", "", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"(?:final\s+answer|answer)\s*:\s*$", "", prefix, flags=re.IGNORECASE).strip()
    return prefix


def normalize_answer_list(values: List[str]) -> List[str]:
    return sorted(str(value).strip() for value in values if str(value).strip())


def parse_futurex_prompt(prompt: str) -> FutureXPromptParts:
    normalized = prompt.strip()
    if not normalized.startswith(FUTUREX_PREFIX):
        raise FutureXError("Unexpected FutureX prompt prefix")

    body = normalized[len(FUTUREX_PREFIX) :].strip()
    important_idx = body.find(FUTUREX_IMPORTANT_MARKER)
    if important_idx == -1:
        raise FutureXError("Could not find FutureX answer-format marker")

    question_text = body[:important_idx].strip()
    if question_text.startswith('"'):
        question_text = question_text[1:]
    if question_text.endswith('"'):
        question_text = question_text[:-1]
    question_text = question_text.strip()

    remainder = body[important_idx + len(FUTUREX_IMPORTANT_MARKER) :].strip()
    final_constraint_idx = remainder.find(FUTUREX_FINAL_CONSTRAINT_MARKER)
    if final_constraint_idx == -1:
        raise FutureXError("Could not find FutureX final-constraint marker")

    answer_requirements = remainder[:final_constraint_idx].strip()
    final_constraints = remainder[final_constraint_idx:].strip()
    return FutureXPromptParts(
        question_text=question_text,
        answer_requirements=answer_requirements,
        final_constraints=final_constraints,
    )


def build_futurex_prompt(
    parts: FutureXPromptParts,
    strategy: str,
    strategy_config: Optional[PromptStrategyConfig] = None,
) -> str:
    experience_section = render_experience_bank_prompt_section(
        list(getattr(strategy_config, "experience_entries", ()) or ())
    )
    if strategy == WEB_SEARCH_LOOP_STRATEGY:
        search_note = (
            f"If you call web_search_and_process, results are restricted to sources published on or before {strategy_config.search_before}."
            if strategy_config and strategy_config.search_before
            else "If you call web_search_and_process, there is no publication-date cutoff."
        )
        intro = (
            "You are an agent that predicts future events. You may reason over multiple turns and use web_search_and_process "
            "to gather supporting evidence when helpful. The web_search_and_process tool returns processed evidence, not raw search results. "
            f"{search_note} In your final response, provide a detailed rationale section that evaluates and cites the most relevant processed evidence, explains why that evidence is relevant or limited, discusses uncertainties or conflicting signals, and then end with the boxed final answer."
        )
    else:
        intro = (
            "You are an agent that predicts future events. Analyze the question, first provide a brief rationale section, "
            "then end with the boxed final answer."
        )

    return "\n\n".join(
        [
            section
            for section in [
                intro,
                experience_section,
                "Event to be predicted:\n" + parts.question_text,
                "Final answer requirements:\n" + parts.answer_requirements,
                (
                    "Response format for this run:\n"
                    "Rationale: <a detailed explanation that cites and evaluates the most relevant evidence, explains its limitations, and justifies the final answer>\n"
                    "\\boxed{<final answer>}\n"
                    "Do not put any text after the boxed answer."
                ),
                "Constraint for the final boxed answer line only:\n" + parts.final_constraints,
            ]
            if section
        ]
    )


class FutureXRunner(SharedLLMClient):
    """LLM runner for FutureX prompts."""

    async def run_futurex_prompt_async(
        self,
        raw_prompt: str,
        strategy: str = "single_call",
        strategy_config: Optional[PromptStrategyConfig] = None,
        tool_runtime_context: Optional[ToolRuntimeContext] = None,
    ) -> str:
        try:
            prompt_parts = parse_futurex_prompt(raw_prompt)
            log_block("futurex.prompt", "Extracted question_text", prompt_parts.question_text)
            log_block("futurex.prompt", "Extracted answer_requirements", prompt_parts.answer_requirements)
            prompt = build_futurex_prompt(prompt_parts, strategy, strategy_config)
            return await super().run_prompt_async(
                prompt,
                strategy=strategy,
                strategy_config=strategy_config,
                tool_runtime_context=tool_runtime_context,
            )
        except ValueError as exc:
            raise FutureXError(str(exc)) from exc


def resolve_futurex_strategy_config(
    event_data: Dict,
    strategy_config: Optional[PromptStrategyConfig] = None,
) -> PromptStrategyConfig:
    base_config = strategy_config or PromptStrategyConfig()
    dataset_cutoff = normalize_search_before_exclusive_date(event_data.get("end_time"))
    if not dataset_cutoff:
        raise FutureXError(
            f"Missing or invalid end_time for sample {event_data.get('id', '')}: {event_data.get('end_time', '')}"
        )
    log_info(
        "futurex.search",
        (
            f"Resolved search cutoff | sample_id={event_data.get('id', '')} "
            f"| end_time={event_data.get('end_time', '')} | exclusive_search_before={dataset_cutoff}"
        ),
    )
    return replace(base_config, search_before=dataset_cutoff)


async def process_futurex_events_async(
    events_data: List[Dict],
    runner: FutureXRunner,
    strategy: str = "single_call",
    strategy_config: Optional[PromptStrategyConfig] = None,
) -> List[Dict]:
    should_save_rollout = bool((strategy_config or PromptStrategyConfig()).save_rollout)

    tasks = [
        process_futurex_event_async(
            event_data=event,
            runner=runner,
            strategy=strategy,
            strategy_config=strategy_config,
            should_save_rollout=should_save_rollout,
        )
        for event in events_data
    ]
    return await asyncio.gather(*tasks)


async def process_futurex_event_async(
    *,
    event_data: Dict,
    runner: FutureXRunner,
    strategy: str = "single_call",
    strategy_config: Optional[PromptStrategyConfig] = None,
    should_save_rollout: Optional[bool] = None,
    execution_id_override: Optional[str] = None,
) -> Dict:
    sample_id = str(event_data["id"])
    execution_id = str(execution_id_override or sample_id).strip()
    if not execution_id:
        raise ValueError("FutureX execution_id must be non-empty")

    resolved_should_save_rollout = bool(
        (strategy_config or PromptStrategyConfig()).save_rollout
        if should_save_rollout is None
        else should_save_rollout
    )

    result: Dict
    try:
        event_strategy_config = resolve_futurex_strategy_config(event_data, strategy_config)
        prompt_parts = parse_futurex_prompt(event_data["prompt"])
        task_requirements = (
            "Answer requirements:\n"
            + prompt_parts.answer_requirements
            + "\n\nFinal answer constraints:\n"
            + prompt_parts.final_constraints
        )
        register_rollout(
            execution_id,
            sample_identifier=sample_id,
            problem_statement=prompt_parts.question_text,
            task_requirements=task_requirements,
        )
        response_text = await runner.run_futurex_prompt_async(
            event_data["prompt"],
            strategy=strategy,
            strategy_config=event_strategy_config,
            tool_runtime_context=ToolRuntimeContext(
                task_id=execution_id,
                problem_statement=prompt_parts.question_text,
                task_requirements=task_requirements,
                sample_identifier=sample_id,
            ),
        )
        parsed_answer = parse_boxed_answer(response_text)
        extracted_rationale = parse_rationale_text(response_text)
        ground_truth = event_data.get("ground_truth", [])
        evaluation = compute_futurex_evaluation(parsed_answer, ground_truth)
        update_rollout_outcome(
            execution_id,
            ground_truth=ground_truth,
            evaluation=evaluation,
        )
        log_info(
            "metric.futurex",
            f"Task metric | task_id={execution_id} | accuracy={evaluation['accuracy']}",
        )
        result = {
            "id": event_data["id"],
            "title": event_data["title"],
            "end_time": event_data.get("end_time", ""),
            "level": event_data.get("level", ""),
            "ground_truth": str(ground_truth),
            "response_text": response_text,
            "extracted_rationale": extracted_rationale,
            "parsed_answer": str(parsed_answer),
            "exact_match": evaluation["accuracy"],
            "model": runner.model,
            "status": "success",
        }
    except Exception as exc:
        ground_truth = event_data.get("ground_truth", [])
        evaluation = {
            "dataset": "futurex",
            "accuracy": False,
        }
        update_rollout_outcome(
            execution_id,
            ground_truth=ground_truth,
            evaluation=evaluation,
        )
        log_info(
            "metric.futurex",
            f"Task metric | task_id={execution_id} | accuracy={evaluation['accuracy']}",
        )
        result = {
            "id": event_data["id"],
            "title": event_data.get("title", ""),
            "end_time": event_data.get("end_time", ""),
            "level": event_data.get("level", ""),
            "ground_truth": str(event_data.get("ground_truth", [])),
            "response_text": "",
            "extracted_rationale": "",
            "parsed_answer": "",
            "exact_match": False,
            "model": runner.model,
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
