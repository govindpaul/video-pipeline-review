"""
Pairwise comparison judge — baseline vs challenger.

Final authority for image promotion decisions.
Sees BOTH images side by side and decides which better serves the scene.

TIE or INCONCLUSIVE → keep baseline (incumbent advantage).
Only CHALLENGER wins → promote.
"""

import json
import logging
import re
from pathlib import Path

from pipeline.llm import local as llm
from pipeline.rubrics import load_rubric, format_rubric_text
from pipeline.story.parser import Scene, StoryData
from pipeline.validators.schema import PairwiseDecision, ValidationState

log = logging.getLogger(__name__)


def compare_images(
    baseline_path: Path,
    challenger_path: Path,
    scene: Scene,
    story: StoryData,
    config: dict,
    original_prompt: str | None = None,
) -> PairwiseDecision:
    """Compare baseline vs challenger image for the same scene.

    The vision model sees both images and the prompt, then decides
    which better matches the scene's purpose.

    Args:
        baseline_path: Path to current accepted image
        challenger_path: Path to challenger image
        scene: Scene being evaluated
        story: Full story data
        config: Pipeline config
        original_prompt: Original prompt (always use this, not rewritten)

    Returns:
        PairwiseDecision with winner and per-criterion preferences
    """
    prompt_text = original_prompt if original_prompt else scene.image_prompt

    rubric = load_rubric("pairwise")
    rubric_text = format_rubric_text(rubric)

    # Build the comparison prompt
    # We send baseline as the image in the vision call, and describe the setup
    compare_prompt = rubric["prompt_template"].format(
        title=story.title,
        scene_num=scene.number,
        total_scenes=len(story.scenes),
        scene_title=scene.title,
        image_prompt=prompt_text,
    )

    model = config["models"]["vision_llm"]["model"]

    # Qwen3-VL via Ollama supports multiple images in a single request
    # Send both images for direct comparison
    try:
        import base64

        with open(baseline_path, "rb") as f:
            baseline_b64 = base64.b64encode(f.read()).decode()
        with open(challenger_path, "rb") as f:
            challenger_b64 = base64.b64encode(f.read()).decode()

        # Use raw Ollama API to send both images
        import requests
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You will see two images. The FIRST image is Image A (BASELINE). "
                        "The SECOND image is Image B (CHALLENGER).\n\n"
                        + compare_prompt
                    ),
                    "images": [baseline_b64, challenger_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            },
        }

        log.info(f"Scene {scene.number}: Pairwise comparison (baseline vs challenger)")

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        response = result["message"]["content"]

        # Strip thinking tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        from pipeline.validators.parse_utils import parse_validator_json
        data = parse_validator_json(response, expected_fields=[
            "winner", "prompt_adherence", "scene_intent",
            "story_continuity", "visual_readability", "reason", "confidence",
        ])
        if data is None:
            log.warning(
                f"Scene {scene.number}: pairwise judge PARSE FAILURE — "
                f"keeping BASELINE (incumbent advantage)"
            )
            return PairwiseDecision(
                scene_num=scene.number,
                winner="BASELINE",
                reason="Pairwise judge could not parse response — keeping baseline",
                raw_response=response[:1000],
            )

        winner = data.get("winner", "TIE").upper()
        # Normalize winner value
        if winner not in ("BASELINE", "CHALLENGER", "TIE"):
            if "a" in winner.lower() or "baseline" in winner.lower():
                winner = "BASELINE"
            elif "b" in winner.lower() or "challenger" in winner.lower():
                winner = "CHALLENGER"
            else:
                winner = "TIE"

        decision = PairwiseDecision(
            scene_num=scene.number,
            winner=winner,
            prompt_adherence=_normalize_preference(data.get("prompt_adherence", "TIE")),
            scene_intent=_normalize_preference(data.get("scene_intent", "TIE")),
            story_continuity=_normalize_preference(data.get("story_continuity", "TIE")),
            visual_readability=_normalize_preference(data.get("visual_readability", "TIE")),
            reason=data.get("reason", ""),
            confidence=data.get("confidence", 0.0),
            raw_response=response[:1000],
        )

        log.info(
            f"Scene {scene.number}: Pairwise winner = {decision.winner} "
            f"(confidence={decision.confidence:.2f}, reason: {decision.reason[:80]})"
        )

        return decision

    except Exception as e:
        log.error(f"Scene {scene.number}: pairwise comparison error: {e}")
        return PairwiseDecision(
            scene_num=scene.number,
            winner="BASELINE",
            reason=f"Pairwise judge error: {e} — keeping baseline",
            raw_response=str(e),
        )


def _normalize_preference(value: str) -> str:
    """Normalize preference value to BASELINE/CHALLENGER/TIE."""
    v = value.upper().strip()
    if v in ("BASELINE", "A", "IMAGE A"):
        return "BASELINE"
    if v in ("CHALLENGER", "B", "IMAGE B"):
        return "CHALLENGER"
    return "TIE"


    # Dead code removed — all validators now use parse_utils.parse_validator_json()
