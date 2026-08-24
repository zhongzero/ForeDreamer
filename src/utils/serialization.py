#!/usr/bin/env python3

import ast
import json
import re
from typing import Any


def parse_serialized_data(value: Any, field_name: str):
    """Parse JSON strings and Python literal strings from serialized fields."""
    if not isinstance(value, str):
        return value

    normalized = re.sub(r"UUID\('([^']+)'\)", r"'\1'", value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is empty")

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(normalized)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Failed to parse {field_name}: {exc}") from exc

