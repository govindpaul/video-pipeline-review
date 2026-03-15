"""
Prompt validator — checks image prompts for semantic correctness.

Layer B: runs AFTER scene validation passes.
Deterministic checks first, then LLM semantic validation.
FAIL → rewrite prompt only (not story structure).
"""

import json
import logging
import re

from pipeline.llm import local as llm
from pipeline.rubrics import load_rubric, format_rubric_text
from pipeline.story.parser import StoryData, Scene
from pipeline.validators.deterministic import check_prompt_deterministic
from pipeline.validators.schema import PromptValidation, ValidationState

log = logging.getLogger(__name__)


def validate_prompt(
    scene: Scene,
    story: StoryData,
    config: dict,
) -> PromptValidation:
    """Validate a single scene's image prompt.

    Runs deterministic checks first, then LLM semantic validation.

    Args:
        scene: The scene to validate
        story: Full story data (for context)
        config: Pipeline config dict

    Returns:
        PromptValidation with state and details
    """
    # --- Deterministic checks first ---
    det_state, det_issues = check_prompt_deterministic(scene, story)
    if det_state == ValidationState.FAIL:
        log.warning(f"Scene {scene.number} prompt deterministic FAIL: {det_issues}")
        return PromptValidation(
            scene_num=scene.number,
            state=ValidationState.FAIL,
            issues=det_issues,
            confidence=1.0,
            raw_response="(deterministic check)",
        )

    # --- LLM semantic validation ---
    rubric = load_rubric("prompt")
    rubric_text = format_rubric_text(rubric)

    # Build DNA context
    char_dna = "\n".join(f"[{k}]: {v}" for k, v in story.character_dna.items()) or "(none)"
    loc_dna = "\n".join(f"[{k}]: {v}" for k, v in story.location_dna.items()) or "(none)"
    obj_dna = "\n".join(f"[{k}]: {v}" for k, v in story.object_dna.items()) or "(none)"

    prompt = rubric["prompt_template"].format(
        title=story.title,
        summary=story.summary,
        scene_num=scene.number,
        total_scenes=len(story.scenes),
        scene_title=scene.title,
        scene_purpose=scene.title,  # Scene title serves as purpose summary
        character_dna=char_dna,
        location_dna=loc_dna,
        object_dna=obj_dna,
        image_prompt=scene.image_prompt,
        video_prompt=scene.video_prompt,
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
            "state", "matches_story_beat", "character_dna_present",
            "describes_opening_frame", "no_contradictions", "prompt_clarity",
            "issues", "fix_notes", "replacement_prompt", "confidence",
        ])
        if data is None:
            log.warning(f"Scene {scene.number}: prompt validator parse failure")
            return PromptValidation(
                scene_num=scene.number,
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

        result = PromptValidation(
            scene_num=scene.number,
            state=state,
            matches_story_beat=data.get("matches_story_beat", True),
            character_dna_present=data.get("character_dna_present", True),
            location_dna_present=data.get("location_dna_present", True),
            object_dna_present=data.get("object_dna_present", True),
            describes_opening_frame=data.get("describes_opening_frame", True),
            framing_appropriate=data.get("framing_appropriate", True),
            no_contradictions=data.get("no_contradictions", True),
            not_overloaded=data.get("not_overloaded", True),
            prompt_clarity=data.get("prompt_clarity", "clear"),
            issues=data.get("issues", []),
            fix_notes=data.get("fix_notes", data.get("suggested_fix", "")),
            replacement_prompt=data.get("replacement_prompt", ""),
            confidence=data.get("confidence", 0.0),
            raw_response=response[:1000],
        )

        log.info(
            f"Scene {scene.number} prompt: {result.state.value} "
            f"(clarity={result.prompt_clarity}, confidence={result.confidence:.2f})"
        )
        if result.issues:
            for issue in result.issues[:2]:
                log.info(f"  - {issue}")

        return result

    except Exception as e:
        log.error(f"Scene {scene.number} prompt validator error: {e}")
        return PromptValidation(
            scene_num=scene.number,
            state=ValidationState.VALIDATOR_ERROR,
            issues=[str(e)],
            raw_response=str(e),
        )


def validate_all_prompts(
    story: StoryData, config: dict
) -> list[PromptValidation]:
    """Validate all scene prompts. Returns list of PromptValidation."""
    results = []
    for scene in story.scenes:
        result = validate_prompt(scene, story, config)
        results.append(result)

    passed = sum(1 for r in results if r.state == ValidationState.PASS)
    failed = sum(1 for r in results if r.state == ValidationState.FAIL)
    errors = sum(1 for r in results if r.state in (
        ValidationState.VALIDATOR_ERROR, ValidationState.INCONCLUSIVE
    ))

    log.info(
        f"Prompt validation: {passed} PASS, {failed} FAIL, {errors} ERROR/INCONCLUSIVE "
        f"out of {len(results)} scenes"
    )
    return results


    # Dead code removed — all validators now use parse_utils.parse_validator_json()
