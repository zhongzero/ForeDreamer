#!/usr/bin/env python3

import importlib
import importlib.util
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from utils.generated_paths import resolve_memtool_dir


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    spec: dict[str, Any]
    runner: Callable[..., str]
    build_runner_kwargs: Callable[[dict[str, Any], Any, Any], dict[str, Any]]
    module_path: Path


def _iter_tool_module_names() -> list[str]:
    package_dir = resolve_memtool_dir()
    return [module_path.stem for module_path in _iter_tool_module_paths(package_dir)]


def _iter_tool_module_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    module_paths: list[Path] = []
    for module_path in directory.iterdir():
        if not module_path.is_file():
            continue
        if module_path.suffix != ".py":
            continue
        if not module_path.stem.startswith("tool_"):
            continue
        module_paths.append(module_path)
    return sorted(module_paths)


def _tool_module_path(module_name: str) -> Path:
    return resolve_memtool_dir() / f"{module_name}.py"


def _tool_module_cache_name(module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()
    return f"_dynamic_memtool_{digest}"


def _load_tool_module(module_name: str) -> ModuleType:
    importlib.invalidate_caches()
    module_path = _tool_module_path(module_name)
    return _load_tool_module_from_path(module_path)


def _load_tool_module_from_path(module_path: Path) -> ModuleType:
    importlib.invalidate_caches()
    if not module_path.exists():
        raise ValueError(f"Missing MemTool module file: {module_path}")

    qualified_name = _tool_module_cache_name(module_path)
    sys.modules.pop(qualified_name, None)
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load MemTool module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _build_tool_definition(module: ModuleType) -> ToolDefinition:
    tool_name = getattr(module, "TOOL_NAME", None)
    tool_spec = getattr(module, "TOOL_SPEC", None)
    runner = getattr(module, "run_tool", None)
    build_runner_kwargs = getattr(module, "build_runner_kwargs", None)
    module_file = getattr(module, "__file__", None)

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"{module.__name__} must define a non-empty TOOL_NAME")
    if not isinstance(tool_spec, dict):
        raise ValueError(f"{module.__name__} must define TOOL_SPEC as a dict")
    if not callable(runner):
        raise ValueError(f"{module.__name__} must define callable run_tool")
    if not callable(build_runner_kwargs):
        raise ValueError(f"{module.__name__} must define callable build_runner_kwargs")
    if not isinstance(module_file, str) or not module_file.strip():
        raise ValueError(f"{module.__name__} must define a valid __file__")

    return ToolDefinition(
        name=tool_name,
        spec=tool_spec,
        runner=runner,
        build_runner_kwargs=build_runner_kwargs,
        module_path=Path(module_file).resolve(),
    )


def _load_tool_registry() -> dict[str, ToolDefinition]:
    return _load_tool_registry_from_dir(resolve_memtool_dir())


def _load_tool_registry_from_dir(directory: Path) -> dict[str, ToolDefinition]:
    registry: dict[str, ToolDefinition] = {}
    for module_path in _iter_tool_module_paths(directory):
        tool_definition = _build_tool_definition(_load_tool_module_from_path(module_path))
        if tool_definition.name in registry:
            raise ValueError(f"Duplicate TOOL_NAME detected: {tool_definition.name}")
        registry[tool_definition.name] = tool_definition
    return registry


def load_tool_registry_from_dir(directory: str | Path) -> dict[str, ToolDefinition]:
    return _load_tool_registry_from_dir(Path(directory).resolve())


def get_tool_specs(tool_names: list[str]) -> list[dict[str, Any]]:
    registry = get_tool_registry()
    specs: list[dict[str, Any]] = []
    for tool_name in tool_names:
        tool_definition = registry.get(tool_name)
        if tool_definition is None:
            raise ValueError(f"MemTool is not registered: {tool_name}")
        specs.append(tool_definition.spec)
    return specs


def execute_registered_tool(
    tool_name: str,
    arguments: dict[str, Any],
    config: Any,
    runtime_context: Any,
) -> str:
    tool_definition = get_tool_registry().get(tool_name)
    if tool_definition is None:
        raise ValueError(f"MemTool is not registered: {tool_name}")

    runner_kwargs = tool_definition.build_runner_kwargs(arguments, config, runtime_context)
    return tool_definition.runner(**runner_kwargs)


def reload_tool_registry() -> dict[str, ToolDefinition]:
    global TOOL_REGISTRY, TOOL_REGISTRY_SOURCE_DIR
    TOOL_REGISTRY = _load_tool_registry()
    TOOL_REGISTRY_SOURCE_DIR = str(resolve_memtool_dir())
    return TOOL_REGISTRY


def get_tool_registry() -> dict[str, ToolDefinition]:
    global TOOL_REGISTRY, TOOL_REGISTRY_SOURCE_DIR
    current_source_dir = str(resolve_memtool_dir())
    if not TOOL_REGISTRY or TOOL_REGISTRY_SOURCE_DIR != current_source_dir:
        TOOL_REGISTRY = _load_tool_registry()
        TOOL_REGISTRY_SOURCE_DIR = current_source_dir
    return TOOL_REGISTRY


TOOL_REGISTRY: dict[str, ToolDefinition] = _load_tool_registry()
TOOL_REGISTRY_SOURCE_DIR: str = str(resolve_memtool_dir())
