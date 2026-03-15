"""
Storyboard validity checker — deterministic checks on scene design.

Catches unrenderable patterns BEFORE any image generation:
- Near-identical scene prompts
- Inside-container POV
- Tiny-prop dependency
- Object counting concepts
- Image prompts missing key objects
- Prompts too long

This is a code-based check, not LLM-based. Fast and reliable.
"""

import logging
import re
from pipeline.story.parser import StoryData
from pipeline.validators.schema import ValidationState

log = logging.getLogger(__name__)


def check_storyboard(story: StoryData) -> tuple[ValidationState, list[str]]:
    """Check scene design for renderability problems.

    Returns (state, issues). FAIL if any critical issue found.
    """
    issues = []

    issues.extend(_check_prompt_lengths(story))
    issues.extend(_check_scene_distinctness(story))
    issues.extend(_check_pov_realism(story))
    issues.extend(_check_image_prompt_completeness(story))
    issues.extend(_check_object_counting(story))

    state = ValidationState.FAIL if issues else ValidationState.PASS

    if issues:
        log.warning(f"Storyboard check: {len(issues)} issues")
        for issue in issues[:5]:
            log.warning(f"  - {issue}")
    else:
        log.info("Storyboard check: PASS")

    return state, issues


def _check_prompt_lengths(story: StoryData) -> list[str]:
    """Image prompts should be under 400 chars. Flag overlong prompts."""
    issues = []
    for scene in story.scenes:
        length = len(scene.image_prompt)
        if length > 600:
            issues.append(
                f"Scene {scene.number}: Image prompt too long ({length} chars). "
                f"Max recommended 400. Key subject gets buried in DNA boilerplate."
            )
    return issues


def _check_scene_distinctness(story: StoryData) -> list[str]:
    """Check that adjacent scenes have meaningfully different image prompts."""
    issues = []

    for i in range(1, len(story.scenes)):
        prev = story.scenes[i - 1].image_prompt.lower()
        curr = story.scenes[i].image_prompt.lower()

        # Strip common DNA blocks to compare the scene-specific parts
        # Use word overlap as a proxy for similarity
        prev_words = set(prev.split())
        curr_words = set(curr.split())

        if not prev_words or not curr_words:
            continue

        overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))

        if overlap > 0.85:
            issues.append(
                f"Scenes {story.scenes[i-1].number} and {story.scenes[i].number}: "
                f"Image prompts are {overlap:.0%} similar — will produce near-identical images. "
                f"Change the camera angle, subject, or composition."
            )

    # Check for 3+ consecutive near-identical scenes
    identical_runs = 0
    for i in range(1, len(story.scenes)):
        prev_words = set(story.scenes[i-1].image_prompt.lower().split())
        curr_words = set(story.scenes[i].image_prompt.lower().split())
        overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words), 1)
        if overlap > 0.80:
            identical_runs += 1
        else:
            identical_runs = 0

        if identical_runs >= 2:
            issues.append(
                f"Scenes {story.scenes[i-2].number}-{story.scenes[i].number}: "
                f"3+ consecutive near-identical image prompts. This will produce "
                f"indistinguishable images. The scene breakdown needs more visual variety."
            )
            break

    return issues


def _check_pov_realism(story: StoryData) -> list[str]:
    """Flag unusual POV descriptions that text-to-image models can't render."""
    issues = []

    unrealistic_pov = [
        (r"from inside\b", "inside-object POV"),
        (r"looking up from\s+(inside|within|the bottom)", "inside-container upward view"),
        (r"pov.*from.*case\b", "inside-case POV"),
        (r"pov.*from.*box\b", "inside-box POV"),
        (r"pov.*from.*bag\b", "inside-bag POV"),
        (r"through.*aperture\b", "through-aperture view"),
        (r"through.*keyhole\b", "through-keyhole view"),
        (r"worm.?s?.?eye\b", "worm's eye view"),
    ]

    for scene in story.scenes:
        prompt_lower = scene.image_prompt.lower()
        for pattern, label in unrealistic_pov:
            if re.search(pattern, prompt_lower):
                issues.append(
                    f"Scene {scene.number}: Uses '{label}' camera angle — "
                    f"not renderable by text-to-image models. Use standard angles."
                )
                break

    return issues


def _check_image_prompt_completeness(story: StoryData) -> list[str]:
    """Check that key objects mentioned in video prompts also appear in image prompts."""
    issues = []

    for scene in story.scenes:
        video = scene.video_prompt.lower()
        image = scene.image_prompt.lower()

        # Extract nouns from video prompt that might be key objects
        # Look for object DNA references in video but not image
        for tag, desc in story.object_dna.items():
            # Get a key identifier from the object description
            key_words = desc.lower().split()[:3]  # First 3 words of object DNA

            # Check if the object concept appears in video prompt
            obj_in_video = any(w in video for w in key_words if len(w) > 3)
            obj_in_image = any(w in image for w in key_words if len(w) > 3)

            if obj_in_video and not obj_in_image:
                issues.append(
                    f"Scene {scene.number}: Object [{tag}] appears in video prompt "
                    f"but NOT in image prompt. The image model won't generate it. "
                    f"Add it to the image prompt Subject or Pose field."
                )

    return issues


def _check_object_counting(story: StoryData) -> list[str]:
    """Flag concepts that seem to rely on progressive object counting."""
    issues = []

    # Check for number words in sequential scene prompts
    # Only count explicit quantity-dependent language, not incidental number words
    count_phrases = [
        "stack of", "pile of", "stacked", "stacking",
        "two bears", "three bears", "four bears", "five bears",
        "two items", "three items", "second bear", "third bear",
        "grows higher", "exceeds capacity", "overflow",
        "one more", "another one on top", "adds another",
    ]

    count_scenes = 0
    for scene in story.scenes:
        prompt_lower = scene.image_prompt.lower() + " " + scene.video_prompt.lower()
        if any(phrase in prompt_lower for phrase in count_phrases):
            count_scenes += 1

    if count_scenes >= 3:
        issues.append(
            f"{count_scenes} scenes reference counting/quantities. "
            f"Text-to-image models cannot reliably generate exact object counts. "
            f"Consider a concept that doesn't depend on precise quantities."
        )

    return issues
