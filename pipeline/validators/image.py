"""
Image validator — visually inspects generated images against prompts.

Layer D: runs AFTER image generation.
Deterministic checks first, then LLM vision validation.

Parse failure = VALIDATOR_ERROR (not FAIL).
VALIDATOR_ERROR / INCONCLUSIVE = keep image (benefit of doubt).
Only FAIL triggers challenger generation.
"""

import json
import logging
import re
from pathlib import Path

from pipeline.llm import local as llm
from pipeline.rubrics import load_rubric, format_rubric_text
from pipeline.story.parser import StoryData, Scene
from pipeline.validators.deterministic import (
    check_image_deterministic,
    check_image_duplicates,
)
from pipeline.validators.schema import ImageValidation, ValidationState

log = logging.getLogger(__name__)


def validate_image(
    image_path: Path,
    scene: Scene,
    story: StoryData,
    config: dict,
    original_prompt: str | None = None,
) -> ImageValidation:
    """Validate a single generated image.

    Deterministic checks first, then LLM vision validation.

    Args:
        image_path: Path to generated image
        scene: The scene this image is for
        story: Full story data
        config: Pipeline config
        original_prompt: Original prompt from story file (not rewritten).
                        If None, uses scene.image_prompt.

    Returns:
        ImageValidation with explicit state
    """
    prompt_text = original_prompt if original_prompt else scene.image_prompt

    # --- Deterministic checks ---
    img_cfg = config["models"]["image"]
    det_state, det_issues = check_image_deterministic(
        image_path,
        expected_width=img_cfg.get("width", 704),
        expected_height=img_cfg.get("height", 1248),
    )
    if det_state == ValidationState.FAIL:
        log.warning(f"Scene {scene.number} image deterministic FAIL: {det_issues}")
        return ImageValidation(
            scene_num=scene.number,
            state=ValidationState.FAIL,
            score=0,
            findings=det_issues,
            confidence=1.0,
            raw_response="(deterministic check)",
        )

    # --- LLM vision validation ---
    rubric = load_rubric("image")
    rubric_text = format_rubric_text(rubric)

    prompt = rubric["prompt_template"].format(
        title=story.title,
        scene_num=scene.number,
        total_scenes=len(story.scenes),
        scene_title=scene.title,
        image_prompt=prompt_text,
        rubric_text=rubric_text,
    )

    model = config["models"]["vision_llm"]["model"]

    try:
        response = llm.vision(
            model=model,
            image_path=str(image_path),
            prompt=prompt,
            temperature=config["models"]["vision_llm"].get("temperature", 0.3),
            format_json=False,  # Don't use Ollama JSON mode — conflicts with thinking
        )

        from pipeline.validators.parse_utils import parse_validator_json
        data = parse_validator_json(response, expected_fields=[
            "state", "matches_prompt", "characters_present", "setting_correct",
            "score", "findings", "confidence",
        ])
        if data is None:
            log.warning(
                f"Scene {scene.number}: image validator PARSE FAILURE — "
                f"returning VALIDATOR_ERROR (image is KEPT)"
            )
            return ImageValidation(
                scene_num=scene.number,
                state=ValidationState.VALIDATOR_ERROR,
                findings=["Validator could not parse its own response"],
                raw_response=response[:1000],
            )

        state_str = data.get("state", "").upper()
        if state_str == "PASS":
            state = ValidationState.PASS
        elif state_str == "FAIL":
            state = ValidationState.FAIL
        else:
            state = ValidationState.INCONCLUSIVE

        result = ImageValidation(
            scene_num=scene.number,
            state=state,
            matches_prompt=data.get("matches_prompt", True),
            characters_present=data.get("characters_present", True),
            objects_present=data.get("objects_present", True),
            setting_correct=data.get("setting_correct", True),
            matches_scene_intent=data.get("matches_scene_intent", True),
            matches_story=data.get("matches_story", True),
            composition_acceptable=data.get("composition_acceptable", True),
            visually_readable=data.get("visually_readable", True),
            strong_opening_frame=data.get("strong_opening_frame", True),
            score=data.get("score", 50),
            findings=data.get("findings", []),
            confidence=data.get("confidence", 0.0),
            raw_response=response[:1000],
        )

        log.info(
            f"Scene {scene.number}: {result.state.value} "
            f"(score={result.score}, confidence={result.confidence:.2f})"
        )
        if result.findings:
            for f in result.findings[:2]:
                log.info(f"  - {f}")

        return result

    except Exception as e:
        log.error(f"Scene {scene.number} image validator error: {e}")
        return ImageValidation(
            scene_num=scene.number,
            state=ValidationState.VALIDATOR_ERROR,
            findings=[str(e)],
            raw_response=str(e),
        )


def validate_all_images(
    images: list[Path],
    story: StoryData,
    config: dict,
    original_prompts: dict[int, str] | None = None,
) -> list[ImageValidation]:
    """Validate all scene images. Returns list of ImageValidation.

    Also runs duplicate detection across all images.
    """
    results = []

    # Build image path lookup
    image_map = {}
    for img in images:
        if img.stem.startswith("scene_"):
            try:
                num = int(img.stem.split("_")[1])
                image_map[num] = img
            except (IndexError, ValueError):
                pass

    for scene in story.scenes:
        img_path = image_map.get(scene.number)
        if not img_path or not img_path.exists():
            log.warning(f"Scene {scene.number}: no image found")
            results.append(ImageValidation(
                scene_num=scene.number,
                state=ValidationState.FAIL,
                findings=["Image file not found"],
            ))
            continue

        orig_prompt = None
        if original_prompts:
            orig_prompt = original_prompts.get(scene.number)

        result = validate_image(img_path, scene, story, config, orig_prompt)
        results.append(result)

    # Duplicate detection
    dupes = check_image_duplicates(images)
    if dupes:
        log.warning(f"Near-duplicate images detected: {dupes}")
        for a, b in dupes:
            # Add finding to both scenes
            for r in results:
                if r.scene_num in (a, b):
                    r.findings.append(f"Near-duplicate with scene {b if r.scene_num == a else a}")

    passed = sum(1 for r in results if r.state == ValidationState.PASS)
    failed = sum(1 for r in results if r.state == ValidationState.FAIL)
    errors = sum(1 for r in results if r.state in (
        ValidationState.VALIDATOR_ERROR, ValidationState.INCONCLUSIVE
    ))

    log.info(
        f"Image validation: {passed} PASS, {failed} FAIL, {errors} ERROR/INCONCLUSIVE "
        f"out of {len(results)} scenes"
    )
    return results


    # Dead code removed — all validators now use parse_utils.parse_validator_json()
