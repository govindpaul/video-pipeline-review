#!/usr/bin/env python3
"""
Fixture-based validator tests.

Tests scene and prompt validators with known inputs and mocked LLM responses.
Identifies whether failures come from: rubric, prompt template, parser, or model output.

Run: python tests/test_validator_fixtures.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.validators.schema import ValidationState
from pipeline.validators.scene import validate_scenes, _parse_json
from pipeline.validators.prompt import validate_prompt, _parse_json as prompt_parse_json
from pipeline.story.parser import StoryData, Scene


def make_story(scenes=5, title="Test Story", summary="A test story"):
    """Create a minimal StoryData for testing."""
    story = StoryData(
        title=title,
        summary=summary,
        character_dna={"CHAR_DNA": "male, 30, brown hair, blue eyes, jacket no text no logos"},
        location_dna={"LOC_DNA": "city street, concrete sidewalk, brick buildings, streetlamps, dusk"},
        object_dna={"OBJ_DNA": "red umbrella, worn handle, partially open"},
    )
    for i in range(1, scenes + 1):
        story.scenes.append(Scene(
            number=i,
            title=f"Scene {i} Title",
            duration_s=3.0,
            image_prompt=(
                f"Subject: [CHAR_DNA] expression focused | Pose: standing, face tilted down | "
                f"Camera: medium shot, 50mm, f/2.8 | Environment: [LOC_DNA] | "
                f"Lighting: warm streetlamp from left | Mood: contemplative, photograph"
            ),
            video_prompt="He walks slowly. Subsequently, he stops. Then, he looks up. Finally, he smiles.",
        ))
    return story


CONFIG = {
    "story": {"scene_count_min": 5, "scene_count_max": 8},
    "models": {"story_llm": {"model": "qwen3.5:35b-a3b"}},
}


# =============================================================================
# PARSER TESTS — does the JSON parser handle all expected formats?
# =============================================================================


def test_parser_clean_json():
    """Parser handles clean JSON."""
    data = _parse_json('{"state": "PASS", "confidence": 0.9}')
    assert data is not None
    assert data["state"] == "PASS"
    print("  clean JSON: OK")


def test_parser_json_in_code_fence():
    """Parser extracts JSON from markdown code fence."""
    data = _parse_json('```json\n{"state": "PASS", "confidence": 0.9}\n```')
    assert data is not None
    assert data["state"] == "PASS"
    print("  code fence JSON: OK")


def test_parser_json_with_preamble():
    """Parser finds JSON in text with preamble."""
    data = _parse_json('Here is my analysis:\n\n{"state": "FAIL", "issues": ["too short"]}')
    assert data is not None
    assert data["state"] == "FAIL"
    print("  JSON with preamble: OK")


def test_parser_malformed():
    """Parser returns None for unparseable text."""
    assert _parse_json("This is just text") is None
    assert _parse_json("") is None
    assert _parse_json('{"broken": ') is None
    print("  malformed returns None: OK")


def test_parser_thinking_stripped():
    """Response after _strip_thinking should be parseable."""
    from pipeline.llm.local import _strip_thinking
    raw = '<think>lots of reasoning here about the story</think>{"state": "PASS"}'
    cleaned = _strip_thinking(raw)
    data = _parse_json(cleaned)
    assert data is not None
    assert data["state"] == "PASS"
    print("  thinking-stripped JSON: OK")


# =============================================================================
# SCENE VALIDATOR FIXTURES
# =============================================================================


def test_scene_validator_pass():
    """Scene validator returns PASS for well-formed LLM response."""
    story = make_story(scenes=6)

    mock_response = json.dumps({
        "state": "PASS",
        "hook_present": True,
        "hook_by_scene": 1,
        "payoff_present": True,
        "payoff_scene": 6,
        "silent_understandable": True,
        "pacing": "good",
        "each_scene_has_purpose": True,
        "no_redundant_scenes": True,
        "redundant_scenes": [],
        "confusing_scenes": [],
        "issues": [],
        "suggestions": [],
        "confidence": 0.92,
    })

    with patch("pipeline.validators.scene.llm") as mock_llm:
        mock_llm.chat.return_value = mock_response
        result = validate_scenes(story, CONFIG)

    assert result.state == ValidationState.PASS
    assert result.confidence == 0.92
    assert result.hook_present is True
    print("  PASS response: correctly parsed as PASS")


def test_scene_validator_fail():
    """Scene validator returns FAIL for failing LLM response."""
    story = make_story(scenes=6)

    mock_response = json.dumps({
        "state": "FAIL",
        "hook_present": False,
        "hook_by_scene": 0,
        "payoff_present": False,
        "payoff_scene": 0,
        "silent_understandable": False,
        "pacing": "too_fast",
        "each_scene_has_purpose": False,
        "no_redundant_scenes": False,
        "redundant_scenes": [3, 4],
        "confusing_scenes": [2],
        "issues": ["No hook", "No payoff", "Scenes 3-4 redundant"],
        "suggestions": ["Add a hook in scene 1"],
        "confidence": 0.88,
    })

    with patch("pipeline.validators.scene.llm") as mock_llm:
        mock_llm.chat.return_value = mock_response
        result = validate_scenes(story, CONFIG)

    assert result.state == ValidationState.FAIL
    assert result.hook_present is False
    assert len(result.redundant_scenes) == 2
    print("  FAIL response: correctly parsed as FAIL with details")


def test_scene_validator_parse_failure():
    """Scene validator returns VALIDATOR_ERROR on garbage response."""
    story = make_story(scenes=6)

    with patch("pipeline.validators.scene.llm") as mock_llm:
        mock_llm.chat.return_value = "I cannot parse this into JSON sorry"
        result = validate_scenes(story, CONFIG)

    assert result.state == ValidationState.VALIDATOR_ERROR
    print("  garbage response: correctly returns VALIDATOR_ERROR")


def test_scene_validator_too_few_scenes():
    """Scene validator catches too few scenes deterministically."""
    story = make_story(scenes=3)  # Below minimum of 5

    # LLM should not even be called — deterministic check fails first
    with patch("pipeline.validators.scene.llm") as mock_llm:
        result = validate_scenes(story, CONFIG)
        mock_llm.chat.assert_not_called()

    assert result.state == ValidationState.FAIL
    assert any("Too few" in i for i in result.issues)
    print("  too few scenes: deterministic FAIL (LLM not called)")


# =============================================================================
# PROMPT VALIDATOR FIXTURES
# =============================================================================


def test_prompt_validator_pass():
    """Prompt validator returns PASS for well-formed response."""
    story = make_story(scenes=5)
    scene = story.scenes[0]

    mock_response = json.dumps({
        "state": "PASS",
        "matches_story_beat": True,
        "character_dna_present": True,
        "location_dna_present": True,
        "object_dna_present": True,
        "describes_opening_frame": True,
        "framing_appropriate": True,
        "no_contradictions": True,
        "not_overloaded": True,
        "prompt_clarity": "clear",
        "issues": [],
        "suggested_fix": "",
        "confidence": 0.90,
    })

    with patch("pipeline.validators.prompt.llm") as mock_llm:
        mock_llm.chat.return_value = mock_response
        result = validate_prompt(scene, story, CONFIG)

    assert result.state == ValidationState.PASS
    assert result.prompt_clarity == "clear"
    print("  PASS response: correctly parsed")


def test_prompt_validator_fail_overloaded():
    """Prompt validator detects overloaded prompt."""
    story = make_story(scenes=5)
    scene = story.scenes[0]

    mock_response = json.dumps({
        "state": "FAIL",
        "matches_story_beat": True,
        "character_dna_present": True,
        "location_dna_present": True,
        "object_dna_present": True,
        "describes_opening_frame": True,
        "framing_appropriate": True,
        "no_contradictions": True,
        "not_overloaded": False,
        "prompt_clarity": "overloaded",
        "issues": ["Prompt contains 800+ chars of DNA description that dilutes the main subject"],
        "suggested_fix": "Subject: man, 30, brown hair, jacket | Pose: standing | Camera: medium, 50mm...",
        "confidence": 0.85,
    })

    with patch("pipeline.validators.prompt.llm") as mock_llm:
        mock_llm.chat.return_value = mock_response
        result = validate_prompt(scene, story, CONFIG)

    assert result.state == ValidationState.FAIL
    assert result.prompt_clarity == "overloaded"
    assert result.suggested_fix != ""
    print("  FAIL overloaded: correctly parsed with suggested fix")


def test_prompt_validator_parse_failure():
    """Prompt validator returns VALIDATOR_ERROR on garbage."""
    story = make_story(scenes=5)
    scene = story.scenes[0]

    with patch("pipeline.validators.prompt.llm") as mock_llm:
        mock_llm.chat.return_value = ""
        result = validate_prompt(scene, story, CONFIG)

    assert result.state == ValidationState.VALIDATOR_ERROR
    print("  empty response: correctly returns VALIDATOR_ERROR")


def test_prompt_validator_missing_dna():
    """Prompt validator catches missing DNA deterministically."""
    story = make_story(scenes=5)
    scene = story.scenes[0]
    # Remove all DNA references from prompt
    scene.image_prompt = "A beautiful landscape at sunset, photograph"

    # Deterministic check should fail before LLM
    with patch("pipeline.validators.prompt.llm") as mock_llm:
        result = validate_prompt(scene, story, CONFIG)
        mock_llm.chat.assert_not_called()

    assert result.state == ValidationState.FAIL
    assert any("character DNA" in i.lower() or "pipe" in i.lower() for i in result.issues)
    print("  missing DNA: deterministic FAIL (LLM not called)")


if __name__ == "__main__":
    print("Video Pipeline v3 — Validator Fixture Tests\n")

    print("=== Parser Tests ===")
    test_parser_clean_json()
    test_parser_json_in_code_fence()
    test_parser_json_with_preamble()
    test_parser_malformed()
    test_parser_thinking_stripped()

    print("\n=== Scene Validator Fixtures ===")
    test_scene_validator_pass()
    test_scene_validator_fail()
    test_scene_validator_parse_failure()
    test_scene_validator_too_few_scenes()

    print("\n=== Prompt Validator Fixtures ===")
    test_prompt_validator_pass()
    test_prompt_validator_fail_overloaded()
    test_prompt_validator_parse_failure()
    test_prompt_validator_missing_dna()

    print("\n" + "=" * 50)
    print("ALL FIXTURE TESTS PASSED (13/13)")
    print("=" * 50)
