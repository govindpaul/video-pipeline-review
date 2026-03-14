"""
Pre-generation gate checks — pure code validation, no LLM.

Catches structural issues before wasting GPU time on image/video generation.
Returns (passed, failures) where failures is a list of specific error strings.
"""

import re
import logging
from pipeline.story.parser import StoryData

log = logging.getLogger(__name__)

# Required pipe-separated fields in image prompts
IMAGE_PROMPT_FIELDS = ["Subject:", "Pose:", "Camera:", "Environment:", "Lighting:", "Mood:"]


def gate_check(story: StoryData, config: dict) -> tuple[bool, list[str]]:
    """Run all gate checks on a parsed story.

    Args:
        story: Parsed StoryData
        config: Pipeline config dict

    Returns:
        (passed, failures) — passed is True if all gates pass
    """
    failures = []

    failures.extend(_check_dna_completeness(story))
    failures.extend(_check_scene_count(story, config))
    failures.extend(_check_image_prompt_format(story))
    failures.extend(_check_dna_references(story))
    failures.extend(_check_video_prompts(story))
    failures.extend(_check_timing(story, config))
    failures.extend(_check_no_text_logos(story))

    passed = len(failures) == 0
    if passed:
        log.info("All gates passed")
    else:
        log.warning(f"Gate check failed: {len(failures)} issues")
        for f in failures:
            log.warning(f"  - {f}")

    return passed, failures


def _check_dna_completeness(story: StoryData) -> list[str]:
    """Every story needs at least 1 character DNA and 1 location DNA."""
    failures = []
    if not story.character_dna:
        failures.append("No character DNA blocks defined")
    if not story.location_dna:
        failures.append("No location DNA blocks defined")
    # Object DNA is optional
    return failures


def _check_scene_count(story: StoryData, config: dict) -> list[str]:
    """Scene count must be within configured range."""
    story_cfg = config.get("story", {})
    min_scenes = story_cfg.get("scene_count_min", 5)
    max_scenes = story_cfg.get("scene_count_max", 8)

    if len(story.scenes) < min_scenes:
        return [f"Too few scenes: {len(story.scenes)} (min {min_scenes})"]
    if len(story.scenes) > max_scenes:
        return [f"Too many scenes: {len(story.scenes)} (max {max_scenes})"]
    return []


def _check_image_prompt_format(story: StoryData) -> list[str]:
    """Image prompts must use pipe-separated structured format."""
    failures = []
    for scene in story.scenes:
        prompt = scene.image_prompt
        if not prompt:
            failures.append(f"Scene {scene.number}: Empty image prompt")
            continue

        # Check for pipe separators
        if "|" not in prompt:
            failures.append(
                f"Scene {scene.number}: Image prompt missing pipe separators (|). "
                f"Must use format: Subject: ... | Pose: ... | Camera: ..."
            )
            continue

        # Check for required fields
        missing = []
        for field_name in IMAGE_PROMPT_FIELDS:
            if field_name not in prompt:
                missing.append(field_name)
        if missing:
            failures.append(
                f"Scene {scene.number}: Image prompt missing fields: {', '.join(missing)}"
            )

        # Check Subject is first
        if not prompt.strip().startswith("Subject:"):
            # Allow for shot type prefixes like "WIDE SHOT, eye-level composition of..."
            # which is also a valid pattern from old pipeline
            if "Subject:" in prompt:
                pass  # Subject exists but not first — acceptable for shot-type prefix style
            else:
                failures.append(f"Scene {scene.number}: Image prompt must contain 'Subject:'")

    return failures


def _check_dna_references(story: StoryData) -> list[str]:
    """DNA tags must be referenced in scene prompts or their DNA descriptions expanded inline."""
    failures = []

    # Collect all DNA tags
    all_tags = set()
    all_tags.update(story.character_dna.keys())
    all_tags.update(story.location_dna.keys())

    char_tags = set(story.character_dna.keys())
    loc_tags = set(story.location_dna.keys())

    for scene in story.scenes:
        prompt = scene.image_prompt

        # Check character DNA referenced (as [TAG], bare TAG:, or key features inlined)
        char_found = False
        for tag in char_tags:
            if f"[{tag}]" in prompt or f"{tag}:" in prompt:
                char_found = True
                break
            # Check if key features from DNA description appear in prompt
            desc = story.character_dna[tag]
            desc_words = desc.split()[:10]
            matches = sum(1 for w in desc_words if w.lower() in prompt.lower())
            if matches >= 3:
                char_found = True
                break

        if not char_found and char_tags:
            failures.append(
                f"Scene {scene.number}: No character DNA referenced in image prompt"
            )

        # Check location DNA referenced
        loc_found = False
        for tag in loc_tags:
            if f"[{tag}]" in prompt or f"{tag}:" in prompt:
                loc_found = True
                break
            desc = story.location_dna[tag]
            desc_words = desc.split()[:10]
            matches = sum(1 for w in desc_words if w.lower() in prompt.lower())
            if matches >= 3:
                loc_found = True
                break

        if not loc_found and loc_tags:
            failures.append(
                f"Scene {scene.number}: No location DNA referenced in image prompt"
            )

    return failures


def _check_video_prompts(story: StoryData) -> list[str]:
    """Video prompts should NOT contain DNA tags (use pronouns instead)."""
    failures = []
    all_tags = set()
    all_tags.update(story.character_dna.keys())
    all_tags.update(story.location_dna.keys())
    all_tags.update(story.object_dna.keys())

    for scene in story.scenes:
        for tag in all_tags:
            if f"[{tag}]" in scene.video_prompt or f"{tag}:" in scene.video_prompt:
                failures.append(
                    f"Scene {scene.number}: Video prompt contains DNA tag [{tag}] — "
                    f"use pronouns (he/she/they) instead"
                )
    return failures


def _check_timing(story: StoryData, config: dict) -> list[str]:
    """Check scene durations and trim values are sane."""
    failures = []
    story_cfg = config.get("story", {})
    min_dur = story_cfg.get("scene_duration_min", 3)
    max_dur = story_cfg.get("scene_duration_max", 5)

    for scene in story.scenes:
        if scene.duration_s < min_dur or scene.duration_s > max_dur:
            failures.append(
                f"Scene {scene.number}: Duration {scene.duration_s}s outside range "
                f"[{min_dur}-{max_dur}s]"
            )
        if scene.trim_ms < 0 or scene.trim_ms > 1000:
            failures.append(
                f"Scene {scene.number}: Trim {scene.trim_ms}ms outside range [0-1000ms]"
            )
    return failures


def _check_no_text_logos(story: StoryData) -> list[str]:
    """Human character clothing DNA should include 'no text no logos'."""
    failures = []
    # Skip non-human characters (animals, objects)
    animal_keywords = ["cat", "dog", "bird", "animal", "tabby", "kitten", "puppy"]

    for tag, desc in story.character_dna.items():
        desc_lower = desc.lower()
        # Skip animals
        if any(kw in desc_lower for kw in animal_keywords):
            continue
        if "no text" not in desc_lower and "no logos" not in desc_lower:
            failures.append(
                f"Character [{tag}]: Clothing DNA should include 'no text no logos' "
                f"to prevent AI rendering text on clothes"
            )
    return failures
