"""
Prompt rewriting based on validation feedback.

Uses the same Qwen3-VL-30B-A3B vision model that did validation —
no extra model load needed. Always rewrites from the ORIGINAL prompt,
never from a previously rewritten prompt, to prevent hallucination cascades.

When the generated image is completely wrong (score < 30), the rewriter
skips vision analysis and makes minor adjustments to the original prompt
instead — the wrong image provides no useful signal.
"""

import logging
from pathlib import Path

from pipeline.llm import local as llm
from pipeline.story.parser import Scene

log = logging.getLogger(__name__)

REWRITE_PROMPT = """\
You are rewriting an image generation prompt to fix specific issues found by a validator.

ORIGINAL PROMPT:
{original_prompt}

SCENE CONTEXT:
Scene {scene_num}: {scene_title}
This image is the starting frame for a short video scene.

VALIDATOR FINDINGS:
- Score: {score}/100
- Issues found: {findings}

RULES FOR REWRITING:
1. Keep the same pipe-separated format: Subject: | Pose: | Camera: | Environment: | Lighting: | Mood:
2. PRESERVE everything the validator marked as CORRECT
3. FIX what's WRONG by finding an ALTERNATIVE approach (don't just add more words)
4. ADD what's MISSING in a natural way
5. Keep the same story beat — the scene must still serve the same narrative purpose
6. End Mood field with "photograph"
7. If gaze direction was wrong, use physical face-angle enforcement ("face tilted downward, crown of head visible")
8. Keep it concise — overly detailed prompts cause the model to lose focus
9. The SUBJECT must remain the same as in the original prompt — do NOT change who/what the scene is about

Write ONLY the new prompt. No explanation, no commentary.
"""

# When the image is completely wrong, don't show it to the vision model —
# it provides no useful signal and causes hallucination. Just tweak the original.
BLIND_REWRITE_PROMPT = """\
You are rewriting an image generation prompt. The previous generation produced a completely wrong image (unrelated to the prompt). Rewrite the prompt with minor adjustments to help the model generate correctly this time.

ORIGINAL PROMPT:
{original_prompt}

SCENE CONTEXT:
Scene {scene_num}: {scene_title}

RULES:
1. Keep the same pipe-separated format: Subject: | Pose: | Camera: | Environment: | Lighting: | Mood:
2. Keep the same subject, same scene, same story beat
3. Simplify slightly — shorter prompts sometimes generate better
4. Make the Subject field more explicit and front-loaded
5. End Mood field with "photograph"

Write ONLY the new prompt. No explanation, no commentary.
"""

# Score below which we skip vision analysis (image is too wrong to be useful)
BLIND_REWRITE_THRESHOLD = 30


def rewrite_prompt(
    scene: Scene,
    result,  # ImageValidation or ValidationResult or None
    image_path: Path,
    config: dict,
    original_prompt: str | None = None,
) -> str:
    """Rewrite an image prompt based on validation feedback.

    Always rewrites from the original prompt to prevent hallucination cascades.
    When the image is completely wrong (score < 30), skips vision analysis
    and does a blind rewrite from the original prompt.

    Args:
        scene: The scene being fixed
        result: Validation result with findings
        image_path: Path to the generated image
        config: Pipeline config dict
        original_prompt: The original prompt from the story file.
                        If provided, this is used instead of scene.image_prompt.

    Returns:
        Rewritten image prompt string
    """
    model = config["models"]["vision_llm"]["model"]
    source_prompt = original_prompt if original_prompt else scene.image_prompt

    if result is None or result.score < BLIND_REWRITE_THRESHOLD:
        # Image is completely wrong — don't show it to the vision model
        score_str = str(result.score) if result else "N/A"
        log.info(
            f"Scene {scene.number}: Score {score_str} — "
            f"blind rewrite (not showing image to model)"
        )
        prompt = BLIND_REWRITE_PROMPT.format(
            original_prompt=source_prompt,
            scene_num=scene.number,
            scene_title=scene.title,
        )

        response = llm.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config["models"]["vision_llm"].get("temperature", 0.3),
            max_tokens=1024,
        )
    else:
        # Image has some correct elements — show it for targeted fixes
        log.info(f"Scene {scene.number}: Rewriting prompt (score was {result.score})")
        # Support both old ValidationResult (correct/wrong/missing) and new ImageValidation (findings)
        findings = getattr(result, "findings", [])
        if not findings:
            # Fallback for old schema
            parts = []
            for attr in ("correct", "wrong", "missing"):
                val = getattr(result, attr, None)
                if val:
                    parts.extend(val)
            findings = parts

        prompt = REWRITE_PROMPT.format(
            original_prompt=source_prompt,
            scene_num=scene.number,
            scene_title=scene.title,
            score=result.score,
            findings="; ".join(findings) if findings else "no specific findings",
        )

        response = llm.vision(
            model=model,
            image_path=str(image_path),
            prompt=prompt,
            temperature=config["models"]["vision_llm"].get("temperature", 0.3),
            max_tokens=1024,
        )

    new_prompt = _clean_response(response)
    log.info(f"Scene {scene.number}: Rewritten prompt: {new_prompt[:150]}...")
    return new_prompt


def _clean_response(response: str) -> str:
    """Clean up LLM response — remove wrapping quotes, code fences."""
    text = response.strip()

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return text
