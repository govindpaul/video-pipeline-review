"""
Deterministic (code-based) validators.

Run BEFORE LLM validators at each layer. Fast, reliable, no GPU needed.
These catch obvious structural problems so the LLM doesn't waste time
evaluating fundamentally broken inputs.
"""

import hashlib
import logging
import re
import struct
from pathlib import Path
from typing import Optional

from pipeline.story.parser import StoryData, Scene
from pipeline.validators.schema import ValidationState

log = logging.getLogger(__name__)

# Motion verbs that belong in video prompts, not image prompts
MOTION_VERBS = [
    "runs", "running", "walks", "walking", "jumps", "jumping",
    "dances", "dancing", "falls", "falling", "throws", "throwing",
    "catches", "catching", "climbs", "climbing", "swims", "swimming",
    "flies", "flying", "kicks", "kicking", "punches", "punching",
    "spins", "spinning", "turns around", "moves toward",
]


# =============================================================================
# SCENE-LEVEL DETERMINISTIC CHECKS
# =============================================================================


def check_scenes_deterministic(
    story: StoryData, config: dict
) -> tuple[ValidationState, list[str]]:
    """Fast structural checks on scene structure.

    Returns (state, issues). State is PASS if no issues, FAIL if any critical issue.
    """
    issues = []
    story_cfg = config.get("story", {})
    min_scenes = story_cfg.get("scene_count_min", 5)
    max_scenes = story_cfg.get("scene_count_max", 8)

    # Scene count
    n = len(story.scenes)
    if n < min_scenes:
        issues.append(f"Too few scenes: {n} (min {min_scenes})")
    if n > max_scenes:
        issues.append(f"Too many scenes: {n} (max {max_scenes})")

    # Estimated runtime
    total_seconds = sum(s.duration_s for s in story.scenes)
    if total_seconds < 10:
        issues.append(f"Total runtime too short: {total_seconds}s (min 10s)")
    if total_seconds > 30:
        issues.append(f"Total runtime too long: {total_seconds}s (max ~25s)")

    # Each scene has both prompts
    for scene in story.scenes:
        if not scene.image_prompt.strip():
            issues.append(f"Scene {scene.number}: empty image prompt")
        if not scene.video_prompt.strip():
            issues.append(f"Scene {scene.number}: empty video prompt")

    # Duplicate scene titles (exact match = likely copy-paste)
    titles = [s.title.strip().lower() for s in story.scenes]
    seen = set()
    for i, t in enumerate(titles):
        if t in seen:
            issues.append(f"Scene {story.scenes[i].number}: duplicate title '{story.scenes[i].title}'")
        seen.add(t)

    state = ValidationState.FAIL if issues else ValidationState.PASS
    return state, issues


# =============================================================================
# PROMPT-LEVEL DETERMINISTIC CHECKS
# =============================================================================


def check_prompt_deterministic(
    scene: Scene, story: StoryData
) -> tuple[ValidationState, list[str]]:
    """Fast structural checks on a single image prompt.

    Returns (state, issues).
    """
    issues = []
    prompt = scene.image_prompt

    # Required pipe-separated fields
    required_fields = ["Subject:", "Pose:", "Camera:", "Environment:", "Lighting:", "Mood:"]
    if "|" not in prompt:
        issues.append("No pipe separators — must use Subject: | Pose: | Camera: | ... format")
    else:
        for field_name in required_fields:
            if field_name not in prompt:
                issues.append(f"Missing field: {field_name}")

    # Character DNA referenced (for scenes with human characters)
    char_tags = set(story.character_dna.keys())
    if char_tags:
        char_found = False
        for tag in char_tags:
            if f"[{tag}]" in prompt or f"{tag}:" in prompt:
                char_found = True
                break
            # Check key features inlined
            desc = story.character_dna[tag]
            desc_words = desc.split()[:10]
            if sum(1 for w in desc_words if w.lower() in prompt.lower()) >= 3:
                char_found = True
                break
        if not char_found:
            issues.append("No character DNA referenced in prompt")

    # Location DNA referenced
    loc_tags = set(story.location_dna.keys())
    if loc_tags:
        loc_found = False
        for tag in loc_tags:
            if f"[{tag}]" in prompt or f"{tag}:" in prompt:
                loc_found = True
                break
            desc = story.location_dna[tag]
            desc_words = desc.split()[:10]
            if sum(1 for w in desc_words if w.lower() in prompt.lower()) >= 3:
                loc_found = True
                break
        if not loc_found:
            issues.append("No location DNA referenced in prompt")

    # Motion verbs in image prompt (should be in video prompt only)
    prompt_lower = prompt.lower()
    found_motion = [v for v in MOTION_VERBS if v in prompt_lower]
    if found_motion:
        issues.append(
            f"Motion verbs in image prompt (belong in video prompt): {found_motion[:3]}"
        )

    # Prompt length check
    if len(prompt) < 50:
        issues.append(f"Prompt too short ({len(prompt)} chars) — likely incomplete")
    if len(prompt) > 1500:
        issues.append(
            f"Prompt very long ({len(prompt)} chars) — model may lose focus on key details"
        )

    state = ValidationState.FAIL if issues else ValidationState.PASS
    return state, issues


# =============================================================================
# IMAGE-LEVEL DETERMINISTIC CHECKS
# =============================================================================


def check_image_deterministic(
    image_path: Path,
    expected_width: int = 704,
    expected_height: int = 1248,
) -> tuple[ValidationState, list[str]]:
    """Fast checks on a generated image file.

    Returns (state, issues).
    """
    issues = []

    # File exists
    if not image_path.exists():
        return ValidationState.FAIL, [f"Image file not found: {image_path}"]

    # File size
    size_bytes = image_path.stat().st_size
    if size_bytes == 0:
        return ValidationState.FAIL, ["Image file is empty (0 bytes)"]
    if size_bytes < 10000:  # < 10KB is suspiciously small for a 704x1248 image
        issues.append(f"Image suspiciously small: {size_bytes} bytes")

    # Valid PNG header
    with open(image_path, "rb") as f:
        header = f.read(8)
    png_signature = b"\x89PNG\r\n\x1a\n"
    if header != png_signature:
        return ValidationState.FAIL, ["File is not a valid PNG"]

    # Read dimensions from PNG IHDR chunk
    width, height = _read_png_dimensions(image_path)
    if width and height:
        if width != expected_width or height != expected_height:
            issues.append(
                f"Wrong dimensions: {width}x{height} (expected {expected_width}x{expected_height})"
            )

    state = ValidationState.FAIL if issues else ValidationState.PASS
    return state, issues


def check_image_duplicates(
    images: list[Path],
    threshold: float = 0.95,
) -> list[tuple[int, int]]:
    """Detect IDENTICAL or byte-level-similar scene images via file hash.

    This is NOT visual near-duplicate detection. It only catches:
    - Identical files (same bytes)
    - Trivially similar files (same partial hash)

    For true visual similarity detection, a perceptual hash or
    embedding-based comparison would be needed (not implemented).

    Returns list of (scene_a, scene_b) pairs that are byte-level duplicates.
    """
    duplicates = []

    # Build fingerprints: (file_size, hash of first 8KB + last 8KB)
    fingerprints = {}
    for img in images:
        if not img.exists():
            continue
        try:
            scene_num = int(img.stem.split("_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue

        size = img.stat().st_size
        with open(img, "rb") as f:
            head = f.read(8192)
            f.seek(max(0, size - 8192))
            tail = f.read(8192)
        h = hashlib.md5(head + tail).hexdigest()
        fingerprints[scene_num] = (size, h)

    # Compare pairs
    scene_nums = sorted(fingerprints.keys())
    for i, a in enumerate(scene_nums):
        for b in scene_nums[i + 1 :]:
            size_a, hash_a = fingerprints[a]
            size_b, hash_b = fingerprints[b]
            # Same hash = very likely duplicate
            if hash_a == hash_b:
                duplicates.append((a, b))
            # Very similar file size (within 5%) could also indicate near-duplicate
            elif size_a > 0 and abs(size_a - size_b) / size_a < 0.05:
                # Only flag if sizes are suspiciously close AND both are same rough size
                pass  # File size alone isn't enough — skip for now

    return duplicates


def _read_png_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Read width and height from PNG IHDR chunk."""
    try:
        with open(path, "rb") as f:
            f.read(8)  # Skip PNG signature
            f.read(4)  # IHDR chunk length
            chunk_type = f.read(4)
            if chunk_type == b"IHDR":
                width = struct.unpack(">I", f.read(4))[0]
                height = struct.unpack(">I", f.read(4))[0]
                return width, height
    except Exception:
        pass
    return None, None
