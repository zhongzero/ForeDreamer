#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Final


TOOL_NAME: Final[str] = "tool_initial"
TOOL_SPEC: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Read the current item_dir input file, extract the relevant content, and write the output file."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:

    item_dir = str(getattr(runtime_context, "item_dir", "") or "").strip()
    if not item_dir:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime item_dir")
    input_filename = str(getattr(runtime_context, "input_filename", "") or "").strip()
    if not input_filename:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime input_filename")
    output_filename = str(getattr(runtime_context, "output_filename", "") or "").strip()
    if not output_filename:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime output_filename")

    return {
        "item_dir": item_dir,
        "input_filename": input_filename,
        "output_filename": output_filename,
    }


def run_tool(*, item_dir: str, input_filename: str, output_filename: str) -> str:
    item_dir_path = Path(item_dir)
    input_path = item_dir_path / input_filename
    output_path = item_dir_path / output_filename
    if not input_path.exists():
        raise ValueError(f"{TOOL_NAME} could not find {input_filename} under {item_dir_path}")

    raw_data = json.loads(input_path.read_text(encoding="utf-8"))
    item = raw_data.get("item", {})
    title = str(item.get("title", "") or "").strip()
    url = str(item.get("url", "") or "").strip()
    content = str(item.get("content", "") or "").strip()

    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    if url:
        lines.append(f"URL: {url}")
    if content:
        if lines:
            lines.append("")
        lines.append(content)
    final_data_text = "\n".join(lines)
    output_path.write_text(final_data_text, encoding="utf-8")
    return f"{TOOL_NAME} wrote {output_filename}"


__all__ = ["TOOL_NAME", "TOOL_SPEC", "build_runner_kwargs", "run_tool"]
