"""
Scene validator — checks story scene structure for short-form video.

Layer A: runs AFTER gate (structural) checks pass.
Deterministic checks first, then LLM semantic validation.
FAIL → rewrite story structure (not just prompts).
"""

import json
import logging
import re

from pipeline.llm import local as llm
from pipeline.rubrics import load_rubric, format_rubric_text
from pipeline.story.parser import StoryData
from pipeline.validators.deterministic import check_scenes_deterministic
from pipeline.validators.schema import SceneValidation, ValidationState

log = logging.getLogger(__name__)


def validate_scenes(story: StoryData, config: dict) -> SceneValidation:
    """Validate story scene structure.

    Runs deterministic checks first, then LLM semantic validation.

    Args:
        story: Parsed story data
        config: Pipeline config dict

    Returns:
        SceneValidation with state and details
    """
    # --- Deterministic checks first ---
    det_state, det_issues = check_scenes_deterministic(story, config)
    if det_state == ValidationState.FAIL:
        log.warning(f"Scene deterministic check FAILED: {det_issues}")
        return SceneValidation(
            state=ValidationState.FAIL,
            issues=det_issues,
            confidence=1.0,
            raw_response="(deterministic check)",
        )

    # --- LLM semantic validation ---
    rubric = load_rubric("scene")
    rubric_text = format_rubric_text(rubric)

    # Build scenes text
    # Keep scenes text compact — titles and durations only.
    # Full prompts are validated separately by the prompt validator.
    scenes_text = ""
    for s in story.scenes:
        scenes_text += f"Scene {s.number}: {s.title} ({s.duration_s}s)\n"

    estimated_runtime = sum(s.duration_s for s in story.scenes)

    prompt = rubric["prompt_template"].format(
        title=story.title,
        summary=story.summary,
        scene_count=len(story.scenes),
        estimated_runtime=estimated_runtime,
        scenes_text=scenes_text,
        rubric_text=rubric_text,
    )

    model = config["models"]["story_llm"]["model"]

    try:
        response = llm.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            format_json=False,
            max_tokens=8192,  # Qwen3.5 uses ~2-4k tokens for <think>, needs room for JSON
        )

        from pipeline.validators.parse_utils import parse_validator_json
        data = parse_validator_json(response, expected_fields=[
            "state", "hook_present", "payoff_present", "confidence",
            "silent_understandable", "pacing", "issues",
        ])
        if data is None:
            log.warning("Scene validator: could not parse LLM response")
            return SceneValidation(
                state=ValidationState.VALIDATOR_ERROR,
                issues=["Validator could not parse its own response"],
                raw_response=response[:500],
            )

        state_str = data.get("state", "").upper()
        if state_str == "PASS":
            state = ValidationState.PASS
        elif state_str == "FAIL":
            state = ValidationState.FAIL
        elif state_str == "INCONCLUSIVE":
            state = ValidationState.INCONCLUSIVE
        else:
            state = ValidationState.INCONCLUSIVE  # Unknown state → benefit of doubt

        result = SceneValidation(
            state=state,
            hook_present=data.get("hook_present", True),
            hook_by_scene=data.get("hook_by_scene", 1),
            payoff_present=data.get("payoff_present", True),
            payoff_scene=data.get("payoff_scene", len(story.scenes)),
            silent_understandable=data.get("silent_understandable", True),
            pacing=data.get("pacing", "good"),
            each_scene_has_purpose=data.get("each_scene_has_purpose", True),
            no_redundant_scenes=data.get("no_redundant_scenes", True),
            redundant_scenes=data.get("redundant_scenes", []),
            confusing_scenes=data.get("confusing_scenes", []),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            confidence=data.get("confidence", 0.0),
            raw_response=response[:1000],
        )

        log.info(
            f"Scene validation: {result.state.value} "
            f"(confidence={result.confidence:.2f}, issues={len(result.issues)})"
        )
        if result.issues:
            for issue in result.issues[:3]:
                log.info(f"  - {issue}")

        return result

    except Exception as e:
        log.error(f"Scene validator error: {e}")
        return SceneValidation(
            state=ValidationState.VALIDATOR_ERROR,
            issues=[str(e)],
            raw_response=str(e),
        )


def _parse_json(response: str) -> dict | None:
    """Parse JSON from LLM response. Returns None on failure."""
    text = response.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None
