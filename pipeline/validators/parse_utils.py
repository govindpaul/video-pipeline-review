"""
Shared JSON parsing utilities for all validators.

Multi-strategy parser with increasing fallback tolerance:
1. Direct json.loads
2. Extract from markdown code fence
3. Find outermost { } in text
4. Regex-based field extraction for known schema fields

Parse failure returns None — callers must handle as VALIDATOR_ERROR.
"""

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)


def parse_validator_json(
    response: str,
    expected_fields: list[str] | None = None,
) -> Optional[dict]:
    """Parse JSON from an LLM validator response.

    Tries multiple strategies in order of strictness.
    Returns None only if all strategies fail.

    Args:
        response: Raw LLM response text (thinking tags already stripped)
        expected_fields: Optional list of field names to look for in regex fallback
    """
    text = response.strip()
    if not text:
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find outermost { } pair (skip nested)
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace by counting nesting
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break

    # Strategy 4: Regex field extraction for known fields
    if expected_fields:
        result = _regex_extract(text, expected_fields)
        if result:
            return result

    return None


def _regex_extract(text: str, fields: list[str]) -> Optional[dict]:
    """Last-resort: extract known fields from text using regex.

    Handles cases where the model outputs valid key-value pairs but
    wraps them in prose or malformed JSON.
    """
    result = {}
    found_any = False

    for field_name in fields:
        # Look for "field_name": value patterns
        # Handle: bool, int, float, string, array
        pattern = rf'"{field_name}"\s*:\s*'

        match = re.search(pattern, text)
        if not match:
            # Try without quotes (some models omit them)
            pattern = rf'\b{field_name}\s*:\s*'
            match = re.search(pattern, text)

        if match:
            rest = text[match.end():].strip()

            # Try to parse the value
            if rest.startswith(("true", "false")):
                result[field_name] = rest.startswith("true")
                found_any = True
            elif rest.startswith('"'):
                str_match = re.match(r'"((?:[^"\\]|\\.)*)"', rest)
                if str_match:
                    result[field_name] = str_match.group(1)
                    found_any = True
            elif rest.startswith("["):
                bracket_end = rest.find("]")
                if bracket_end >= 0:
                    try:
                        result[field_name] = json.loads(rest[: bracket_end + 1])
                        found_any = True
                    except json.JSONDecodeError:
                        pass
            elif rest[0].isdigit() or rest[0] == "-":
                num_match = re.match(r"-?\d+\.?\d*", rest)
                if num_match:
                    val = num_match.group()
                    result[field_name] = float(val) if "." in val else int(val)
                    found_any = True

    return result if found_any else None
