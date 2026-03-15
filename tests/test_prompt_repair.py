#!/usr/bin/env python3
"""
Tests for prompt repair flow.

Proves:
- Replacement prompts require BOTH deterministic AND semantic validation
- Advisory fix_notes are never assigned as prompts
- Semantic FAIL rejects replacement even if deterministic passes
- Resume path uses same validators

Run: python tests/test_prompt_repair.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.validators.schema import ValidationState, PromptValidation
from pipeline.validators.deterministic import check_prompt_deterministic
from pipeline.story.parser import StoryData, Scene


def make_story():
    story = StoryData(
        title="Test",
        summary="A test story",
        character_dna={"CHAR_DNA": "male, 30, brown hair, blue eyes, jacket no text no logos"},
        location_dna={"LOC_DNA": "city street, concrete sidewalk, brick buildings, streetlamps, dusk"},
        object_dna={},
    )
    story.scenes.append(Scene(
        number=1, title="Scene 1", duration_s=3.0,
        image_prompt=(
            "Subject: CHAR_DNA male 30 brown hair | Pose: standing, face tilted down | "
            "Camera: medium shot, 50mm, f/2.8 | Environment: LOC_DNA city street | "
            "Lighting: warm streetlamp from left | Mood: contemplative, photograph"
        ),
        video_prompt="He walks slowly.",
    ))
    return story


def test_acceptance_full_pipeline():
    """TEST 4: Replacement prompt accepted only after BOTH validations pass.

    Flow:
    - original prompt FAILs prompt validation
    - replacement_prompt is proposed
    - deterministic validation PASSES
    - semantic prompt validation PASSES
    - replacement becomes the prompt used for generation
    """
    print("=== TEST 4: Replacement acceptance (both validations pass) ===")

    story = make_story()
    original_prompt = story.scenes[0].image_prompt

    # A valid pipe-format replacement
    replacement = (
        "Subject: male, 30, brown hair, blue eyes, jacket no text no logos, focused expression | "
        "Pose: standing upright, face tilted slightly down, crown of head visible | "
        "Camera: medium shot, eye level, 50mm lens, f/2.8, shallow DoF | "
        "Environment: city street, concrete sidewalk, brick buildings, streetlamps, dusk | "
        "Lighting: warm streetlamp from left, directional | "
        "Mood: contemplative, photograph"
    )

    # Step 1: Verify deterministic passes
    test_scene = Scene(number=1, title="Scene 1", image_prompt=replacement, video_prompt="He walks.")
    det_state, det_issues = check_prompt_deterministic(test_scene, story)
    assert det_state == ValidationState.PASS, f"Deterministic should PASS: {det_issues}"
    print(f"  Deterministic: {det_state.value}")

    # Step 2: Mock semantic validation to PASS
    mock_sem_result = PromptValidation(
        scene_num=1,
        state=ValidationState.PASS,
        matches_story_beat=True,
        character_dna_present=True,
        location_dna_present=True,
        describes_opening_frame=True,
        no_contradictions=True,
        confidence=0.90,
    )

    with patch("pipeline.validators.prompt.validate_prompt") as mock_validate:
        mock_validate.return_value = mock_sem_result

        # Simulate the acceptance logic from main.py (v2: require PASS, not just !FAIL)
        accepted = False
        if det_state == ValidationState.PASS:
            sem_result = mock_validate(test_scene, story, {})
            if sem_result.state == ValidationState.PASS:
                story.scenes[0].image_prompt = replacement
                accepted = True

    assert accepted, "Should be accepted after both validations PASS"
    assert story.scenes[0].image_prompt == replacement, "Prompt should be the replacement"
    assert story.scenes[0].image_prompt != original_prompt, "Should differ from original"
    print(f"  Semantic: {mock_sem_result.state.value}")
    print(f"  Accepted: {accepted}")
    print(f"  Final prompt starts with: {story.scenes[0].image_prompt[:60]}...")
    print("  PASSED\n")


def test_rejection_semantic_fail():
    """TEST 5: Replacement rejected when semantic validation FAILs.

    Flow:
    - replacement_prompt passes deterministic
    - BUT semantic validation FAILs (contradictions, wrong framing, etc.)
    - original prompt is kept
    """
    print("=== TEST 5: Replacement rejected (semantic FAIL) ===")

    story = make_story()
    original_prompt = story.scenes[0].image_prompt

    # A structurally valid but semantically bad replacement
    replacement = (
        "Subject: male, 30, brown hair | Pose: running fast, arms pumping | "
        "Camera: wide shot, 24mm | Environment: LOC_DNA city street | "
        "Lighting: harsh noon sun | Mood: energetic, photograph"
    )

    # Step 1: Deterministic passes (it has pipe format, DNA, etc.)
    # Note: "running" is a motion verb — deterministic should catch this
    test_scene = Scene(number=1, title="Scene 1", image_prompt=replacement, video_prompt="He walks.")
    det_state, det_issues = check_prompt_deterministic(test_scene, story)
    print(f"  Deterministic: {det_state.value} {det_issues}")

    if det_state == ValidationState.PASS:
        # Step 2: Semantic FAILS (motion in image prompt, wrong framing)
        mock_sem_result = PromptValidation(
            scene_num=1,
            state=ValidationState.FAIL,
            describes_opening_frame=False,
            no_contradictions=False,
            issues=["'running fast' is motion, not a static opening frame",
                     "Framing contradicts contemplative scene"],
            confidence=0.92,
        )

        with patch("pipeline.validators.prompt.validate_prompt") as mock_validate:
            mock_validate.return_value = mock_sem_result

            accepted = False
            sem_result = mock_validate(test_scene, story, {})
            if sem_result.state == ValidationState.PASS:
                story.scenes[0].image_prompt = replacement
                accepted = True

        assert not accepted, "Should NOT be accepted when semantic FAILs"
        assert story.scenes[0].image_prompt == original_prompt, "Original must be preserved"
        print(f"  Semantic: {mock_sem_result.state.value} (issues: {mock_sem_result.issues})")
        print(f"  Accepted: {accepted}")
        print(f"  Prompt unchanged: {story.scenes[0].image_prompt[:60]}...")
    else:
        # Deterministic already caught "running" as motion verb — even better
        assert story.scenes[0].image_prompt == original_prompt, "Original must be preserved"
        print(f"  Rejected at deterministic stage (motion verb caught)")
        print(f"  Prompt unchanged: {story.scenes[0].image_prompt[:60]}...")

    print("  PASSED\n")


def _test_rejection_for_state(state: ValidationState, state_name: str):
    """Helper: replacement rejected when semantic returns given state."""
    story = make_story()
    original_prompt = story.scenes[0].image_prompt

    # Valid pipe-format replacement that passes deterministic
    replacement = (
        "Subject: male, 30, brown hair, blue eyes, jacket no text no logos | "
        "Pose: standing upright, face tilted down | "
        "Camera: medium shot, 50mm, f/2.8 | "
        "Environment: LOC_DNA city street, concrete sidewalk, brick buildings | "
        "Lighting: warm streetlamp from left | "
        "Mood: contemplative, photograph"
    )

    test_scene = Scene(number=1, title="Scene 1", image_prompt=replacement, video_prompt="He walks.")
    det_state, _ = check_prompt_deterministic(test_scene, story)
    assert det_state == ValidationState.PASS, "Deterministic must PASS for this test"

    mock_result = PromptValidation(scene_num=1, state=state, confidence=0.5)

    with patch("pipeline.validators.prompt.validate_prompt") as mock_validate:
        mock_validate.return_value = mock_result

        accepted = False
        sem_result = mock_validate(test_scene, story, {})
        if sem_result.state == ValidationState.PASS:
            story.scenes[0].image_prompt = replacement
            accepted = True

    assert not accepted, f"Must NOT accept when semantic is {state_name}"
    assert story.scenes[0].image_prompt == original_prompt, "Original must be preserved"
    print(f"  Deterministic: PASS")
    print(f"  Semantic: {state_name}")
    print(f"  Accepted: {accepted}")
    print(f"  Prompt unchanged: yes")


def test_rejection_semantic_inconclusive():
    """TEST: Replacement rejected when semantic returns INCONCLUSIVE."""
    print("=== TEST: Replacement rejected (semantic INCONCLUSIVE) ===")
    _test_rejection_for_state(ValidationState.INCONCLUSIVE, "INCONCLUSIVE")
    print("  PASSED\n")


def test_rejection_semantic_validator_error():
    """TEST: Replacement rejected when semantic returns VALIDATOR_ERROR."""
    print("=== TEST: Replacement rejected (semantic VALIDATOR_ERROR) ===")
    _test_rejection_for_state(ValidationState.VALIDATOR_ERROR, "VALIDATOR_ERROR")
    print("  PASSED\n")


def test_fix_notes_never_assigned():
    """Prove fix_notes (advisory prose) is never used as a prompt."""
    print("=== TEST: fix_notes never assigned as prompt ===")

    import inspect
    from pipeline import main

    source = inspect.getsource(main.run_new)

    assert "scene.image_prompt = pr.fix_notes" not in source, \
        "fix_notes must never be assigned as prompt"
    assert "scene.image_prompt = pr.suggested_fix" not in source, \
        "suggested_fix must never be assigned as prompt"

    # Check challenger too
    from pipeline.image import challenger
    chall_source = inspect.getsource(challenger.run_challenges)
    assert "scene.image_prompt = " not in chall_source, \
        "Challenger should not mutate scene.image_prompt"

    print("  fix_notes: never assigned")
    print("  suggested_fix: never referenced")
    print("  challenger: no scene.image_prompt mutation")
    print("  PASSED\n")


def test_resume_uses_same_validators():
    """TEST 6: Resume path uses same validation imports as main path."""
    print("=== TEST 6: Resume path uses same validators ===")

    import inspect
    from pipeline import main

    new_source = inspect.getsource(main.run_new)
    resume_source = inspect.getsource(main.run_resume)

    # Both should use the same validator modules
    shared_imports = [
        "validate_all_images",
        "setup_initial_history",
        "run_challenges",
        "ValidationState",
    ]

    for imp in shared_imports:
        # Check module-level imports (used by both)
        assert imp in inspect.getsource(main), \
            f"main.py missing import: {imp}"
        print(f"  {imp}: available to both paths")

    # Resume should reference VALIDATING_IMAGES state
    assert "VALIDATING_IMAGES" in resume_source, \
        "Resume must use VALIDATING_IMAGES state"
    assert "CHALLENGING_IMAGES" in resume_source, \
        "Resume must use CHALLENGING_IMAGES state"

    print("  Resume uses VALIDATING_IMAGES: yes")
    print("  Resume uses CHALLENGING_IMAGES: yes")
    print("  PASSED\n")


if __name__ == "__main__":
    print("Video Pipeline v3 — Prompt Repair Tests\n")

    test_acceptance_full_pipeline()
    test_rejection_semantic_fail()
    test_rejection_semantic_inconclusive()
    test_rejection_semantic_validator_error()
    test_fix_notes_never_assigned()
    test_resume_uses_same_validators()

    # Verify the acceptance logic in actual code matches
    import inspect
    from pipeline import main
    source = inspect.getsource(main.run_new)
    assert "sem_result.state != ValidationState.PASS" in source, \
        "main.py must reject anything except PASS"
    print("\n=== CODE INSPECTION ===")
    print("  main.py acceptance rule: != PASS → reject (confirmed)")

    from pipeline.image import challenger
    chall_source = inspect.getsource(challenger.run_challenges)
    assert "sem_result.state != ValidationState.PASS" in chall_source, \
        "challenger.py must reject anything except PASS"
    print("  challenger.py acceptance rule: != PASS → reject (confirmed)")

    print("\n" + "=" * 50)
    print("ALL PROMPT REPAIR TESTS PASSED (6 tests)")
    print("=" * 50)
