#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(stage: str, message: str) -> None:
    print(f"[{_timestamp()}] [{stage}] {message}", flush=True)


def log_block(stage: str, title: str, content: Any) -> None:
    body = "" if content is None else str(content)
    print(f"[{_timestamp()}] [{stage}] {title}", flush=True)
    if body:
        print(body, flush=True)
    print(f"[{_timestamp()}] [{stage}] End {title}", flush=True)


def format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
