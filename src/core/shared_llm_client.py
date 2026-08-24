#!/usr/bin/env python3

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Final, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from DefaultTool.registry import (
    PUBLIC_TOOL_REGISTRY,
    execute_registered_tool_async,
    get_public_tool_specs,
)
from core.rollout import record_main_agent_interaction, register_rollout
from utils.api_cache import (
    build_cached_completion,
    build_llm_request_identity,
    load_llm_cache_entry,
    record_llm_actual_call,
    record_llm_cache_hit,
    serialize_message_payload,
    store_llm_cache_entry,
)
from utils.logger import format_json, log_block, log_info
from utils.timing_registry import timed_block


SINGLE_CALL_STRATEGY: Final[str] = "single_call"
WEB_SEARCH_LOOP_STRATEGY: Final[str] = "web_search_loop"
SUPPORTED_PROMPT_STRATEGIES: Final[set[str]] = {
    SINGLE_CALL_STRATEGY,
    WEB_SEARCH_LOOP_STRATEGY,
}
STRATEGY_TO_TOOL_NAMES: Final[dict[str, tuple[str, ...]]] = {
    WEB_SEARCH_LOOP_STRATEGY: ("web_search_and_process",),
}


@dataclass(frozen=True)
class PromptStrategyConfig:
    max_turns: int = 4
    subagent_max_turns: int = 10
    save_rollout: bool = False
    search_before: Optional[str] = None
    search_provider: str = "tavily"
    search_max_results: int = 5
    search_max_chars_per_result: int = 700
    search_max_total_chars: int = 2500
    search_api_key: Optional[str] = None
    use_tavilty_raw_context: bool = False
    enable_lossy_search_cache: bool = False
    disable_main_agent_final_answer_cache: bool = False
    factual_memory_run_label: Optional[str] = None
    factual_memory_dataset_name: Optional[str] = None
    mem_guide: Optional[str] = None
    experience_entries: Optional[tuple[dict[str, Any], ...]] = None
    experience_bank_hash: Optional[str] = None
    experience_bank_version_id: Optional[str] = None


@dataclass(frozen=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolRuntimeContext:
    task_id: str
    search_turn: int = 0
    problem_statement: str = ""
    task_requirements: str = ""
    sample_identifier: str = ""


def normalize_search_before_date(raw_value: Any) -> Optional[str]:
    if raw_value is None:
        return None

    normalized = str(raw_value).strip()
    if not normalized:
        return None

    candidate = normalized[:10]
    try:
        return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.date().isoformat()


def normalize_search_before_exclusive_date(raw_value: Any) -> Optional[str]:
    normalized = normalize_search_before_date(raw_value)
    if normalized is None:
        return None
    parsed = datetime.strptime(normalized, "%Y-%m-%d").date()
    return (parsed - timedelta(days=1)).isoformat()


class SharedLLMClient:
    """Shared OpenAI-compatible chat client with prompt-based execution strategies."""

    def __init__(
        self,
        model: str = "minimax/minimax-m1",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str = None,
        reasoning: str = None,
    ):
        if not api_key:
            raise ValueError("API key is required")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=800,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning = reasoning

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def completion_with_backoff(self, **kwargs):
        record_llm_actual_call()
        return self.client.chat.completions.create(**kwargs)

    async def async_completion_with_backoff(self, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.completion_with_backoff(**kwargs),
        )

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = dict(model=self.model, messages=messages)
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if self.reasoning and self.base_url == "https://openrouter.ai/api/v1":
            payload["extra_body"] = dict(reasoning={"effort": self.reasoning})
        elif self.reasoning:
            payload["reasoning"] = {"effort": self.reasoning}
        return payload

    async def _create_completion_async(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        log_tag: str = "llm",
        task_id: Optional[str] = None,
        use_cache: bool = True,
        disable_cache_for_text_only_response: bool = False,
    ):
        payload = self._build_payload(messages, tools=tools, tool_choice=tool_choice)
        request_identity = build_llm_request_identity(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=payload.get("reasoning"),
            extra_body=payload.get("extra_body"),
        )
        log_info(
            log_tag,
            f"LLM request start | model={self.model} | tools_enabled={tools is not None} | tool_choice={tool_choice}",
        )
        log_block(log_tag, "Messages sent to LLM", format_json(messages))
        if tools is not None:
            log_block(log_tag, "Tool spec sent to LLM", format_json(tools))
        cache_key = ""
        cache_entry = None
        if use_cache:
            cache_key, cache_entry = load_llm_cache_entry(request_identity)
            if cache_entry is not None:
                cached_completion = build_cached_completion(cache_entry.get("response", {}).get("message", {}))
                cached_message = cached_completion.choices[0].message
                cached_text = self._extract_message_text(cached_message)
                cached_tool_calls = list(getattr(cached_message, "tool_calls", None) or [])
                if disable_cache_for_text_only_response and cached_text and not cached_tool_calls:
                    log_info(
                        log_tag,
                        f"LLM cache bypassed for text-only response | cache_key={cache_key}",
                    )
                else:
                    record_llm_cache_hit()
                    log_info(log_tag, f"LLM cache hit | cache_key={cache_key}")
                    return cached_completion
            log_info(log_tag, f"LLM cache miss | cache_key={cache_key}")
        else:
            log_info(log_tag, "LLM cache bypassed | cache_disabled_for_this_request=true")
        with timed_block(
            "llm.main_agent",
            log_tag,
            kind="llm",
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "tools_enabled": tools is not None,
                "tool_choice": tool_choice,
            },
        ):
            completion = await self.async_completion_with_backoff(**payload)
        message = completion.choices[0].message
        response_text = self._extract_message_text(message)
        native_tool_calls = list(getattr(message, "tool_calls", None) or [])
        log_info(log_tag, f"LLM request end | native_tool_calls={len(native_tool_calls)}")
        if native_tool_calls:
            log_block(
                log_tag,
                "Native tool calls returned by LLM",
                format_json(
                    [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in native_tool_calls
                    ]
                ),
            )
        log_block(log_tag, "LLM response text", response_text)
        should_store_cache_entry = bool(response_text or native_tool_calls)
        if (
            should_store_cache_entry
            and use_cache
            and disable_cache_for_text_only_response
            and response_text
            and not native_tool_calls
        ):
            should_store_cache_entry = False
            log_info(log_tag, "LLM cache store skipped for text-only response")
        if should_store_cache_entry and use_cache:
            store_llm_cache_entry(
                cache_key=cache_key,
                request_identity=request_identity,
                message_payload=serialize_message_payload(
                    content=response_text,
                    tool_calls=native_tool_calls,
                ),
            )
        return completion

    async def _run_messages_once_async(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        log_tag: str = "llm",
        task_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        completion = await self._create_completion_async(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            log_tag=log_tag,
            task_id=task_id,
            use_cache=use_cache,
        )
        message = completion.choices[0].message
        content = self._extract_message_text(message)
        if not content:
            raise ValueError("Model returned an empty response")
        finalized_messages = deepcopy(messages)
        assistant_message = self._serialize_assistant_message(message, content)
        if len(assistant_message) > 1:
            finalized_messages.append(assistant_message)
        record_main_agent_interaction(
            task_id=str(task_id or "").strip(),
            messages=finalized_messages,
            available_tools=tools,
        )
        return content

    async def _run_single_call_async(
        self,
        prompt: str,
        config: PromptStrategyConfig,
        tool_runtime_context: Optional[ToolRuntimeContext] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self._run_messages_once_async(
            messages,
            log_tag="llm.single_call",
            task_id=str(getattr(tool_runtime_context, "task_id", "") or "").strip(),
            use_cache=not config.disable_main_agent_final_answer_cache,
        )

    async def _run_web_search_loop_async(
        self,
        prompt: str,
        config: PromptStrategyConfig,
        tool_runtime_context: ToolRuntimeContext,
    ) -> str:
        tool_names = self._get_tool_names_for_strategy(WEB_SEARCH_LOOP_STRATEGY)
        strategy_tools = get_public_tool_specs(list(tool_names))
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_web_search_system_prompt(config, tool_names),
            },
            {"role": "user", "content": prompt},
        ]

        log_info(
            "llm.web_search_loop",
            (
                f"Starting multi-turn loop | max_turns={config.max_turns} | "
                f"search_before={config.search_before} | tools={list(tool_names)}"
            ),
        )

        for turn in range(1, config.max_turns + 1):
            final_turn = turn == config.max_turns
            log_info(
                "llm.web_search_loop",
                f"Turn {turn}/{config.max_turns} start | final_turn={final_turn} | native_tools=True",
            )
            if final_turn:
                messages.append({"role": "user", "content": self._build_final_turn_instruction()})

            tools = strategy_tools if not final_turn else None
            tool_choice = "auto" if tools is not None else None

            completion = await self._create_completion_async(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                log_tag=f"llm.web_search_loop.turn_{turn}",
                task_id=tool_runtime_context.task_id,
                use_cache=not (config.disable_main_agent_final_answer_cache and final_turn),
                disable_cache_for_text_only_response=bool(
                    config.disable_main_agent_final_answer_cache and not final_turn and tools is not None
                ),
            )

            message = completion.choices[0].message
            text = self._extract_message_text(message)
            native_tool_calls = list(getattr(message, "tool_calls", None) or [])
            if len(native_tool_calls) > 1:
                kept_tool_call = native_tool_calls[0]
                ignored_tool_calls = native_tool_calls[1:]
                log_block(
                    "tool",
                    f"Ignoring extra native tool calls on turn {turn}",
                    format_json(
                        {
                            "returned_count": len(native_tool_calls),
                            "kept": {
                                "id": kept_tool_call.id,
                                "type": kept_tool_call.type,
                                "function": {
                                    "name": kept_tool_call.function.name,
                                    "arguments": kept_tool_call.function.arguments,
                                },
                            },
                            "ignored": [
                                {
                                    "id": tool_call.id,
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in ignored_tool_calls
                            ],
                        }
                    ),
                )
                native_tool_calls = [kept_tool_call]

            if native_tool_calls:
                native_tool_calls = [
                    self._maybe_rewrite_native_tool_call_for_lossy_search_cache(
                        tool_call,
                        config,
                        replace(tool_runtime_context, search_turn=turn),
                    )
                    for tool_call in native_tool_calls
                ]

            if final_turn and native_tool_calls:
                raise ValueError("Model attempted tool use on the final turn")

            if native_tool_calls:
                messages.append(
                    self._serialize_assistant_message(
                        message,
                        text,
                        tool_calls_override=native_tool_calls,
                    )
                )
                for tool_call in native_tool_calls:
                    log_block(
                        "tool",
                        f"Executing native tool call on turn {turn}",
                        format_json(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ),
                    )
                    turn_runtime_context = replace(tool_runtime_context, search_turn=turn)
                    tool_result = await self._execute_native_tool_call(
                        tool_call,
                        config,
                        turn_runtime_context,
                    )
                    # log_block("tool", f"Tool result on turn {turn}", tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                record_main_agent_interaction(
                    task_id=tool_runtime_context.task_id,
                    messages=messages,
                    available_tools=strategy_tools,
                )
                continue

            if text:
                assistant_message = self._serialize_assistant_message(message, text)
                if len(assistant_message) > 1:
                    messages.append(assistant_message)
                record_main_agent_interaction(
                    task_id=tool_runtime_context.task_id,
                    messages=messages,
                    available_tools=strategy_tools,
                )
                log_info("llm.web_search_loop", f"Completed multi-turn loop on turn {turn}")
                return text

            raise ValueError("Model returned an empty response")

        raise ValueError(f"Model did not produce a final answer within {config.max_turns} turns")

    def _get_strategy_config(self, strategy_config: Optional[PromptStrategyConfig]) -> PromptStrategyConfig:
        config = strategy_config or PromptStrategyConfig()
        if config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if config.search_max_results < 1:
            raise ValueError("search_max_results must be at least 1")
        if config.search_max_chars_per_result < 1:
            raise ValueError("search_max_chars_per_result must be at least 1")
        if config.search_max_total_chars < 1:
            raise ValueError("search_max_total_chars must be at least 1")
        if config.search_before is not None and normalize_search_before_date(config.search_before) is None:
            raise ValueError(f"search_before must be a valid YYYY-MM-DD date, got: {config.search_before}")
        return config

    def _maybe_rewrite_native_tool_call_for_lossy_search_cache(
        self,
        tool_call: Any,
        config: PromptStrategyConfig,
        runtime_context: ToolRuntimeContext,
    ) -> Any:
        function = getattr(tool_call, "function", None)
        if function is None:
            return tool_call
        if str(getattr(function, "name", "") or "").strip() != "web_search_and_process":
            return tool_call

        raw_arguments = getattr(function, "arguments", "") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return tool_call

        original_query = str(arguments.get("query", "") or "").strip()
        if not original_query:
            return tool_call

        from DefaultTool.privacy_tool_web_search import resolve_lossy_cached_search_query

        cached_query = resolve_lossy_cached_search_query(
            query=original_query,
            search_provider=config.search_provider,
            search_before=config.search_before,
            max_results=config.search_max_results,
            use_tavilty_raw_context=config.use_tavilty_raw_context,
            endpoint=getattr(config, "endpoint", None),
            factual_memory_dataset_name=config.factual_memory_dataset_name or "",
            sample_identifier=runtime_context.sample_identifier,
            search_turn=runtime_context.search_turn,
            enable_lossy_search_cache=config.enable_lossy_search_cache,
        )
        if not cached_query or cached_query == original_query:
            return tool_call

        rewritten_arguments = dict(arguments)
        rewritten_arguments["query"] = cached_query
        log_info(
            "tool",
            (
                f"Lossy search cache rewrite | turn={runtime_context.search_turn} | "
                f"sample_identifier={runtime_context.sample_identifier} | "
                f"from_query={original_query} | to_query={cached_query}"
            ),
        )
        return SimpleNamespace(
            id=getattr(tool_call, "id", ""),
            type=getattr(tool_call, "type", "function"),
            function=SimpleNamespace(
                name=getattr(function, "name", ""),
                arguments=json.dumps(rewritten_arguments, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _get_tool_names_for_strategy(strategy: str) -> tuple[str, ...]:
        tool_names = STRATEGY_TO_TOOL_NAMES.get(strategy, ())
        for tool_name in tool_names:
            if tool_name not in PUBLIC_TOOL_REGISTRY:
                raise ValueError(
                    f"Strategy {strategy} requires unregistered public tool: {tool_name}"
                )
        return tool_names

    def _build_web_search_system_prompt(
        self,
        config: PromptStrategyConfig,
        tool_names: tuple[str, ...],
    ) -> str:
        if len(tool_names) != 1:
            raise ValueError(
                f"{WEB_SEARCH_LOOP_STRATEGY} expects exactly one configured tool, got {list(tool_names)}"
            )
        tool_name = tool_names[0]
        cutoff_text = (
            f"If you use {tool_name}, results are restricted to sources published on or before {config.search_before}."
            if config.search_before
            else f"If you use {tool_name}, there is no publication-date cutoff."
        )
        return (
            f"You may solve the task over multiple turns and can use the {tool_name} tool when helpful. "
            f"{cutoff_text} The {tool_name} tool searches the web and returns processed evidence rather than raw results. "
            "That processed evidence may still be truncated, so extract only the most relevant information. "
            f"You may issue at most one {tool_name} tool call in a single assistant turn. "
            "Do not batch multiple search requests in one turn. "
            "If you need more evidence, use a later turn instead of issuing multiple tool calls at once. "
            "Only your final answer should follow the task-specific output format."
        )

    @staticmethod
    def _build_final_turn_instruction() -> str:
        return (
            "This is your final allowed assistant turn. Do not call any tool. "
            "Produce the final answer now using only the required task-specific answer format."
        )

    async def _execute_native_tool_call(
        self,
        tool_call,
        config: PromptStrategyConfig,
        runtime_context: ToolRuntimeContext,
    ) -> str:
        function = getattr(tool_call, "function", None)
        if function is None:
            raise ValueError("Tool call is missing function data")

        raw_arguments = getattr(function, "arguments", "") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool arguments: {raw_arguments}") from exc

        request = ToolCallRequest(name=function.name, arguments=arguments)
        return await self._execute_tool_call(request, config, runtime_context)

    async def _execute_tool_call(
        self,
        tool_call: ToolCallRequest,
        config: PromptStrategyConfig,
        runtime_context: ToolRuntimeContext,
    ) -> str:
        query = str(tool_call.arguments.get("query", "")).strip()
        tool_config = self._build_tool_config_view(config)

        log_info("tool", f"Tool start | name={tool_call.name} | query={query}")
        result = await execute_registered_tool_async(
            tool_call.name,
            tool_call.arguments,
            tool_config,
            runtime_context,
            public_only=True,
        )
        log_info("tool", f"Tool end | name={tool_call.name} | query={query}")
        return result

    def _build_tool_config_view(self, config: PromptStrategyConfig) -> Any:
        config_data = asdict(config)
        config_data.update(
            {
                "llm_api_key": self.api_key,
                "llm_base_url": self.base_url,
                "llm_model": self.model,
            }
        )
        return SimpleNamespace(**config_data)

    @staticmethod
    def _serialize_assistant_message(
        message,
        text: str,
        tool_calls_override: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if text:
            assistant_message["content"] = text
        tool_calls = []
        source_tool_calls = (
            list(tool_calls_override)
            if tool_calls_override is not None
            else list(getattr(message, "tool_calls", None) or [])
        )
        for tool_call in source_tool_calls[:1]:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    @staticmethod
    def _extract_message_text(message) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            part_type = getattr(item, "type", None)
            if part_type == "text":
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
                elif hasattr(text_value, "value"):
                    parts.append(str(text_value.value))
        return "".join(parts).strip()

    async def run_prompt_async(
        self,
        prompt: str,
        strategy: str = SINGLE_CALL_STRATEGY,
        strategy_config: Optional[PromptStrategyConfig] = None,
        tool_runtime_context: Optional[ToolRuntimeContext] = None,
    ) -> str:
        config = self._get_strategy_config(strategy_config)
        log_info("llm", f"run_prompt_async start | strategy={strategy} | model={self.model}")
        if tool_runtime_context is not None and str(tool_runtime_context.task_id).strip():
            register_rollout(
                tool_runtime_context.task_id,
                problem_statement=tool_runtime_context.problem_statement,
                task_requirements=tool_runtime_context.task_requirements,
                sample_identifier=tool_runtime_context.sample_identifier,
            )

        if strategy == SINGLE_CALL_STRATEGY:
            result = await self._run_single_call_async(
                prompt,
                config=config,
                tool_runtime_context=tool_runtime_context,
            )
            log_info("llm", f"run_prompt_async end | strategy={strategy}")
            return result
        if strategy == WEB_SEARCH_LOOP_STRATEGY:
            if tool_runtime_context is None or not str(tool_runtime_context.task_id).strip():
                raise ValueError("web_search_loop requires a non-empty tool runtime task_id")
            result = await self._run_web_search_loop_async(
                prompt=prompt,
                config=config,
                tool_runtime_context=tool_runtime_context,
            )
            log_info("llm", f"run_prompt_async end | strategy={strategy}")
            return result
        raise ValueError(
            f"Unsupported prompt strategy: {strategy}. Supported strategies: {sorted(SUPPORTED_PROMPT_STRATEGIES)}"
        )
