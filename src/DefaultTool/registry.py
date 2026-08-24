#!/usr/bin/env python3

import asyncio
import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Final


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    spec: dict[str, Any]
    runner: Callable[..., str]
    async_runner: Callable[..., Any] | None
    build_runner_kwargs: Callable[[dict[str, Any], Any, Any], dict[str, Any]]
    is_public: bool


def _iter_tool_module_names(prefixes: tuple[str, ...]) -> list[str]:
    package_dir = Path(__file__).resolve().parent
    module_names: list[str] = []
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.ispkg:
            continue
        if not module_info.name.startswith(prefixes):
            continue
        module_names.append(module_info.name)
    return sorted(module_names)


def _load_tool_module(module_name: str) -> ModuleType:
    return importlib.import_module(f"DefaultTool.{module_name}")


def _build_tool_definition(module: ModuleType, module_name: str) -> ToolDefinition:
    tool_name = getattr(module, "TOOL_NAME", None)
    tool_spec = getattr(module, "TOOL_SPEC", None)
    runner = getattr(module, "run_tool", None)
    async_runner = getattr(module, "run_tool_async", None)
    build_runner_kwargs = getattr(module, "build_runner_kwargs", None)
    is_public = getattr(module, "IS_PUBLIC_TOOL", None)

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"{module.__name__} must define a non-empty TOOL_NAME")
    if not isinstance(tool_spec, dict):
        raise ValueError(f"{module.__name__} must define TOOL_SPEC as a dict")
    if not callable(runner):
        raise ValueError(f"{module.__name__} must define callable run_tool")
    if async_runner is not None and not callable(async_runner):
        raise ValueError(f"{module.__name__} must define callable run_tool_async when present")
    if not callable(build_runner_kwargs):
        raise ValueError(f"{module.__name__} must define callable build_runner_kwargs")
    if not isinstance(is_public, bool):
        raise ValueError(f"{module.__name__} must define boolean IS_PUBLIC_TOOL")

    if module_name.startswith("tool_") and not is_public:
        raise ValueError(f"{module.__name__} must set IS_PUBLIC_TOOL=True")
    if module_name.startswith("privacy_tool_") and is_public:
        raise ValueError(f"{module.__name__} must set IS_PUBLIC_TOOL=False")

    return ToolDefinition(
        name=tool_name,
        spec=tool_spec,
        runner=runner,
        async_runner=async_runner,
        build_runner_kwargs=build_runner_kwargs,
        is_public=is_public,
    )


def _load_all_tool_registry() -> dict[str, ToolDefinition]:
    registry: dict[str, ToolDefinition] = {}
    module_names = _iter_tool_module_names(("tool_", "privacy_tool_"))
    for module_name in module_names:
        tool_definition = _build_tool_definition(_load_tool_module(module_name), module_name)
        if tool_definition.name in registry:
            raise ValueError(f"Duplicate TOOL_NAME detected: {tool_definition.name}")
        registry[tool_definition.name] = tool_definition
    return registry


def _build_public_tool_registry(all_tools: dict[str, ToolDefinition]) -> dict[str, ToolDefinition]:
    return {
        tool_name: tool_definition
        for tool_name, tool_definition in all_tools.items()
        if tool_definition.is_public
    }


def get_public_tool_specs(tool_names: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool_name in tool_names:
        tool_definition = PUBLIC_TOOL_REGISTRY.get(tool_name)
        if tool_definition is None:
            raise ValueError(f"Public tool is not registered: {tool_name}")
        specs.append(tool_definition.spec)
    return specs


def _get_tool_definition(tool_name: str, *, public_only: bool) -> ToolDefinition:
    registry = PUBLIC_TOOL_REGISTRY if public_only else ALL_TOOL_REGISTRY
    tool_definition = registry.get(tool_name)
    if tool_definition is None:
        registry_name = "public" if public_only else "all"
        raise ValueError(f"Tool is not registered in {registry_name} registry: {tool_name}")
    return tool_definition


def execute_registered_tool(
    tool_name: str,
    arguments: dict[str, Any],
    config: Any,
    runtime_context: Any,
    *,
    public_only: bool = False,
) -> str:
    tool_definition = _get_tool_definition(tool_name, public_only=public_only)
    runner_kwargs = tool_definition.build_runner_kwargs(arguments, config, runtime_context)
    return tool_definition.runner(**runner_kwargs)


async def execute_registered_tool_async(
    tool_name: str,
    arguments: dict[str, Any],
    config: Any,
    runtime_context: Any,
    *,
    public_only: bool = False,
) -> str:
    tool_definition = _get_tool_definition(tool_name, public_only=public_only)
    runner_kwargs = tool_definition.build_runner_kwargs(arguments, config, runtime_context)

    if tool_definition.async_runner is not None:
        async_result = tool_definition.async_runner(**runner_kwargs)
        if inspect.isawaitable(async_result):
            return await async_result
        if isinstance(async_result, str):
            return async_result
        raise ValueError(
            f"Async tool runner returned unsupported result type for {tool_name}: {type(async_result)}"
        )

    return await asyncio.to_thread(tool_definition.runner, **runner_kwargs)


ALL_TOOL_REGISTRY: Final[dict[str, ToolDefinition]] = _load_all_tool_registry()
PUBLIC_TOOL_REGISTRY: Final[dict[str, ToolDefinition]] = _build_public_tool_registry(ALL_TOOL_REGISTRY)
PUBLIC_TOOL_SPECS: Final[list[dict[str, Any]]] = [
    tool_definition.spec for tool_definition in PUBLIC_TOOL_REGISTRY.values()
]
