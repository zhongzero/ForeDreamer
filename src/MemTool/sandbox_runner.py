#!/usr/bin/env python3

import json
import sys
from types import SimpleNamespace
from typing import Any

from MemTool.registry import execute_registered_tool


def _as_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    return value


def main() -> int:
    raw_payload = sys.stdin.read()
    if not raw_payload.strip():
        print("sandbox_runner requires a JSON payload on stdin", file=sys.stderr)
        return 2

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        print(f"sandbox_runner received invalid JSON: {exc}", file=sys.stderr)
        return 2

    tool_name = str(payload.get("tool_name", "") or "").strip()
    if not tool_name:
        print("sandbox_runner payload requires a non-empty tool_name", file=sys.stderr)
        return 2

    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        print("sandbox_runner payload requires arguments to be an object", file=sys.stderr)
        return 2

    try:
        result = execute_registered_tool(
            tool_name,
            arguments,
            _as_namespace(payload.get("config", {})),
            _as_namespace(payload.get("runtime_context", {})),
        )
    except Exception as exc:
        print(f"sandbox_runner failed: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
