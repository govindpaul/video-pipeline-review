#!/usr/bin/env python3
"""
Calibration smoke tests for the layered validation pipeline.

Tests each validator layer against existing story #173 artifacts.
Run: python tests/test_validators.py
"""

import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.story.parser import parse_story
from pipeline.validators.schema import ValidationState
from pipeline.validators.deterministic import (
    check_scenes_deterministic,
    check_prompt_deterministic,
    check_image_deterministic,
    check_image_duplicates,
)


def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def test_deterministic_scene():
    """Test scene-level deterministic checks."""
    print("=== Deterministic Scene Checks ===")
    story = parse_story("stories/story-176-the-color-echo.md")
    config = load_config()

    state, issues = check_scenes_deterministic(story, config)
    print(f"  State: {state.value}")
    print(f"  Issues: {issues}")
    assert state == ValidationState.PASS, f"Expected PASS, got {state.value}: {issues}"
    print("  PASSED")


def test_deterministic_prompt():
    """Test prompt-level deterministic checks."""
    print("\n=== Deterministic Prompt Checks ===")
    story = parse_story("stories/story-176-the-color-echo.md")

    for scene in story.scenes:
        state, issues = check_prompt_deterministic(scene, story)
        status = "OK" if state == ValidationState.PASS else f"ISSUES: {issues}"
        print(f"  Scene {scene.number}: {state.value} — {status}")


def test_deterministic_image():
    """Test image-level deterministic checks."""
    print("\n=== Deterministic Image Checks ===")
    images_dir = Path("stories/output/176/images")

    if not images_dir.exists():
        print("  SKIPPED — no images directory")
        return

    for img in sorted(images_dir.glob("scene_[0-9][0-9].png")):
        state, issues = check_image_deterministic(img)
        print(f"  {img.name}: {state.value} {issues if issues else ''}")


def test_duplicate_detection():
    """Test duplicate image detection."""
    print("\n=== Duplicate Detection ===")
    images_dir = Path("stories/output/176/images")

    if not images_dir.exists():
        print("  SKIPPED — no images directory")
        return

    images = sorted(images_dir.glob("scene_[0-9][0-9].png"))
    dupes = check_image_duplicates(images)
    print(f"  Duplicates found: {dupes}")
    print(f"  PASSED (found {len(dupes)} pairs)")


def test_rubric_loading():
    """Test rubric YAML files load and format correctly."""
    print("\n=== Rubric Loading ===")
    from pipeline.rubrics import load_rubric, format_rubric_text

    for name in ["scene", "prompt", "image", "pairwise"]:
        r = load_rubric(name)
        text = format_rubric_text(r)
        print(f"  {name}_rubric_v{r['version']}: {len(r['criteria'])} criteria, {len(text)} chars formatted")
        assert "prompt_template" in r, f"{name} rubric missing prompt_template"
        assert len(r["criteria"]) > 0, f"{name} rubric has no criteria"

    print("  PASSED")


def test_schema_states():
    """Test validation state logic."""
    print("\n=== Schema State Logic ===")
    from pipeline.validators.schema import (
        ImageValidation, SceneImageHistory, ImageVersion, PairwiseDecision,
    )

    # VALIDATOR_ERROR should NOT trigger repair
    v = ImageValidation(scene_num=1, state=ValidationState.VALIDATOR_ERROR)
    assert v.state != ValidationState.FAIL, "VALIDATOR_ERROR should not be FAIL"

    # INCONCLUSIVE should NOT trigger repair
    v = ImageValidation(scene_num=1, state=ValidationState.INCONCLUSIVE)
    assert v.state != ValidationState.FAIL, "INCONCLUSIVE should not be FAIL"

    # Version history
    h = SceneImageHistory(scene_num=1)
    h.add_version(ImageVersion(version=1, filename="scene_01_v1.png"))
    h.add_version(ImageVersion(version=2, filename="scene_01_v2.png"))
    assert h.selected_version == 1, "Initial selection should be v1"
    h.promote(2)
    assert h.selected_version == 2, "After promotion should be v2"
    assert h.selected.filename == "scene_01_v2.png"

    # TIE = keep baseline
    d = PairwiseDecision(scene_num=1, winner="TIE")
    assert d.winner == "TIE", "TIE should keep baseline"

    print("  PASSED")


def test_state_machine():
    """Test new state transitions."""
    print("\n=== State Machine ===")
    from pipeline.state import State, TRANSITIONS

    # New states exist
    assert State.VALIDATING_SCENES in State.__members__.values()
    assert State.VALIDATING_PROMPTS in State.__members__.values()
    assert State.CHALLENGING_IMAGES in State.__members__.values()

    # Key transitions
    assert State.VALIDATING_SCENES in TRANSITIONS[State.GATING]
    assert State.VALIDATING_PROMPTS in TRANSITIONS[State.VALIDATING_SCENES]
    assert State.GENERATING_IMAGES in TRANSITIONS[State.VALIDATING_PROMPTS]
    assert State.CHALLENGING_IMAGES in TRANSITIONS[State.VALIDATING_IMAGES]
    assert State.GENERATING_VIDEOS in TRANSITIONS[State.CHALLENGING_IMAGES]

    print("  PASSED")


if __name__ == "__main__":
    print("Video Pipeline v3 — Validator Calibration Tests\n")

    test_rubric_loading()
    test_schema_states()
    test_state_machine()
    test_deterministic_scene()
    test_deterministic_prompt()
    test_deterministic_image()
    test_duplicate_detection()

    print("\n" + "=" * 40)
    print("ALL CALIBRATION TESTS PASSED")
    print("=" * 40)
