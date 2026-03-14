"""
Rubric loader — reads versioned YAML rubric files.
"""

from pathlib import Path
from typing import Optional

import yaml

RUBRICS_DIR = Path(__file__).parent


def load_rubric(name: str, version: int = 1) -> dict:
    """Load a rubric YAML file.

    Args:
        name: Rubric name (scene, prompt, image, pairwise)
        version: Rubric version number

    Returns:
        Parsed YAML dict with criteria and prompt_template
    """
    path = RUBRICS_DIR / f"{name}_rubric_v{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def format_rubric_text(rubric: dict) -> str:
    """Format rubric criteria into text for LLM prompt insertion."""
    lines = []
    for c in rubric.get("criteria", []):
        weight = c.get("weight", "")
        weight_tag = f" [{weight.upper()}]" if weight else ""
        lines.append(f"- {c['id']} {c['name']}{weight_tag}: {c['description']}")
    return "\n".join(lines)
