"""
State machine for the video pipeline.

Single state file (stories/state.json), resumable from any state after crash.
State is written BEFORE entering each state. Each stage checks for existing
artifacts and skips completed work.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    CONCEIVING = "CONCEIVING"
    WRITING = "WRITING"
    GATING = "GATING"
    VALIDATING_SCENES = "VALIDATING_SCENES"       # NEW: semantic scene validation
    VALIDATING_PROMPTS = "VALIDATING_PROMPTS"      # NEW: semantic prompt validation
    GENERATING_IMAGES = "GENERATING_IMAGES"
    VALIDATING_IMAGES = "VALIDATING_IMAGES"
    CHALLENGING_IMAGES = "CHALLENGING_IMAGES"       # NEW: replaces REWRITING + REGENERATING
    GENERATING_VIDEOS = "GENERATING_VIDEOS"
    COMBINING = "COMBINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    # Old states kept for backward compatibility with existing state.json files
    REWRITING_PROMPTS = "REWRITING_PROMPTS"
    REGENERATING_IMAGES = "REGENERATING_IMAGES"


# Valid state transitions
TRANSITIONS = {
    State.IDLE: [State.CONCEIVING],
    State.CONCEIVING: [State.WRITING, State.FAILED],
    State.WRITING: [State.GATING, State.FAILED],
    State.GATING: [State.VALIDATING_SCENES, State.WRITING, State.FAILED],
    State.VALIDATING_SCENES: [State.VALIDATING_PROMPTS, State.WRITING, State.FAILED],
    State.VALIDATING_PROMPTS: [State.GENERATING_IMAGES, State.FAILED],
    State.GENERATING_IMAGES: [State.VALIDATING_IMAGES, State.FAILED],
    State.VALIDATING_IMAGES: [
        State.GENERATING_VIDEOS,
        State.CHALLENGING_IMAGES,
        State.FAILED,
    ],
    State.CHALLENGING_IMAGES: [State.GENERATING_VIDEOS, State.FAILED],
    State.GENERATING_VIDEOS: [State.COMBINING, State.FAILED],
    State.COMBINING: [State.COMPLETED, State.FAILED],
    State.COMPLETED: [State.IDLE],
    State.FAILED: [State.IDLE],
    # Old states — allow transitions for backward compat
    State.REWRITING_PROMPTS: [State.REGENERATING_IMAGES, State.FAILED],
    State.REGENERATING_IMAGES: [State.VALIDATING_IMAGES, State.FAILED],
}


class PipelineState:
    """Manages pipeline state with JSON persistence."""

    def __init__(self, state_dir: Path):
        self.state_file = Path(state_dir) / "state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return self._default_state()

    def _default_state(self) -> dict:
        return {
            "version": 1,
            "story_number": None,
            "state": State.IDLE.value,
            "started_at": None,
            "updated_at": None,
            "concept": None,
            "story_file": None,
            "output_dir": None,
            "current_stage": {},
            "history": [],
            "error": None,
        }

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def state(self) -> State:
        return State(self._data["state"])

    @property
    def story_number(self) -> Optional[int]:
        return self._data.get("story_number")

    @property
    def story_file(self) -> Optional[str]:
        return self._data.get("story_file")

    @property
    def output_dir(self) -> Optional[str]:
        return self._data.get("output_dir")

    @property
    def current_stage(self) -> dict:
        return self._data.get("current_stage", {})

    def transition(self, new_state: State):
        """Transition to a new state. Validates the transition is allowed."""
        current = self.state
        allowed = TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        now = datetime.now().isoformat()
        self._data["state"] = new_state.value
        self._data["updated_at"] = now
        self._data["history"].append({"state": new_state.value, "at": now})
        self._data["error"] = None
        log.info(f"State: {current.value} -> {new_state.value}")
        self._save()

    def start_new_story(self, story_number: int):
        """Initialize state for a new story."""
        now = datetime.now().isoformat()
        self._data = self._default_state()
        self._data["story_number"] = story_number
        self._data["started_at"] = now
        self._data["updated_at"] = now
        self._data["output_dir"] = f"stories/output/{story_number}"
        self._data["history"] = [{"state": State.IDLE.value, "at": now}]
        self._save()
        log.info(f"New story #{story_number}")

    def set_story_file(self, path: str):
        self._data["story_file"] = path
        self._save()

    def set_concept(self, concept: str):
        self._data["concept"] = concept
        self._save()

    def update_stage(self, **kwargs):
        """Update current_stage metadata (round, scenes_pending, etc.)."""
        self._data["current_stage"].update(kwargs)
        self._data["updated_at"] = datetime.now().isoformat()
        self._save()

    def fail(self, error: str):
        """Transition to FAILED state with error message."""
        now = datetime.now().isoformat()
        self._data["state"] = State.FAILED.value
        self._data["updated_at"] = now
        self._data["error"] = error
        self._data["history"].append({"state": State.FAILED.value, "at": now})
        log.error(f"Pipeline FAILED: {error}")
        self._save()

    def reset(self):
        """Reset to IDLE state."""
        self._data = self._default_state()
        self._save()

    def get_next_story_number(self, start: int = 172) -> int:
        """Get the next story number based on existing output directories."""
        output_base = self.state_file.parent / "output"
        if not output_base.exists():
            return start
        existing = []
        for d in output_base.iterdir():
            if d.is_dir() and d.name.isdigit():
                existing.append(int(d.name))
        if not existing:
            return start
        return max(existing) + 1

    def __repr__(self) -> str:
        return (
            f"PipelineState(story={self.story_number}, "
            f"state={self.state.value}, "
            f"stage={self.current_stage})"
        )
