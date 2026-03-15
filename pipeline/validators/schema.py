"""
Validation schemas for the layered validation pipeline.

All validators return structured dataclasses with explicit states.
Parse failure is VALIDATOR_ERROR, never FAIL.
VALIDATOR_ERROR and INCONCLUSIVE keep the current artifact.
Only FAIL triggers repair at the appropriate abstraction layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ValidationState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    VALIDATOR_ERROR = "VALIDATOR_ERROR"


# =============================================================================
# SCENE VALIDATION (Layer A)
# =============================================================================


@dataclass
class SceneValidation:
    """Result of scene-level validation (story structure)."""

    state: ValidationState = ValidationState.INCONCLUSIVE

    # Rubric criteria
    scene_count_ok: bool = True
    runtime_fits_format: bool = True
    hook_present: bool = True
    hook_by_scene: int = 0              # Which scene contains the hook
    payoff_present: bool = True
    payoff_scene: int = 0               # Which scene contains the payoff
    silent_understandable: bool = True
    pacing: str = "good"                # "good" | "too_fast" | "too_slow"
    each_scene_has_purpose: bool = True
    no_redundant_scenes: bool = True

    # Details
    redundant_scenes: list[int] = field(default_factory=list)
    confusing_scenes: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Confidence + debugging
    confidence: float = 0.0             # 0.0-1.0, LLM self-reported
    raw_response: str = ""


# =============================================================================
# PROMPT VALIDATION (Layer B)
# =============================================================================


@dataclass
class PromptValidation:
    """Result of prompt-level validation (per scene)."""

    scene_num: int = 0
    state: ValidationState = ValidationState.INCONCLUSIVE

    # Rubric criteria
    matches_story_beat: bool = True
    character_dna_present: bool = True
    location_dna_present: bool = True
    object_dna_present: bool = True
    describes_opening_frame: bool = True  # Not motion/action
    framing_appropriate: bool = True
    no_contradictions: bool = True
    not_overloaded: bool = True           # Prompt isn't too noisy

    # Details
    prompt_clarity: str = "clear"         # "clear"|"overloaded"|"vague"|"contradictory"
    issues: list[str] = field(default_factory=list)
    fix_notes: str = ""                   # Advisory prose (NOT a replacement prompt)
    replacement_prompt: str = ""          # Full valid pipe-format prompt (if provided)

    # Confidence + debugging
    confidence: float = 0.0
    raw_response: str = ""


# =============================================================================
# IMAGE VALIDATION (Layer D)
# =============================================================================


@dataclass
class ImageValidation:
    """Result of image-level validation (per scene)."""

    scene_num: int = 0
    state: ValidationState = ValidationState.INCONCLUSIVE

    # Rubric criteria — prompt adherence
    matches_prompt: bool = True
    characters_present: bool = True
    objects_present: bool = True
    setting_correct: bool = True

    # Rubric criteria — scene/story alignment
    matches_scene_intent: bool = True
    matches_story: bool = True
    composition_acceptable: bool = True

    # Rubric criteria — short-form readability
    visually_readable: bool = True        # Clear at phone size, not cluttered
    strong_opening_frame: bool = True     # Works as first frame of a 3-5s clip

    # Score (meaningful only for PASS/FAIL, not for VALIDATOR_ERROR/INCONCLUSIVE)
    score: int = 0

    # Details
    findings: list[str] = field(default_factory=list)

    # Confidence + debugging
    confidence: float = 0.0
    raw_response: str = ""


# =============================================================================
# PAIRWISE COMPARISON (Layer F)
# =============================================================================


@dataclass
class PairwiseDecision:
    """Result of pairwise baseline vs challenger comparison."""

    scene_num: int = 0

    # Overall decision
    winner: str = "BASELINE"              # "BASELINE" | "CHALLENGER" | "TIE"

    # Per-criterion preference
    prompt_adherence: str = "TIE"         # "BASELINE" | "CHALLENGER" | "TIE"
    scene_intent: str = "TIE"
    story_continuity: str = "TIE"
    visual_readability: str = "TIE"

    # Details
    reason: str = ""

    # Confidence + debugging
    confidence: float = 0.0
    raw_response: str = ""


# =============================================================================
# IMAGE VERSION TRACKING
# =============================================================================


@dataclass
class ImageVersion:
    """Metadata for one version of a scene image."""

    version: int = 1                      # v1, v2, v3...
    filename: str = ""                    # scene_01_v1.png
    prompt_used: str = ""                 # The prompt that generated this version
    validation: Optional[ImageValidation] = None
    comparison: Optional[PairwiseDecision] = None  # Only for v2+ (challenger)


@dataclass
class SceneImageHistory:
    """Full version history for one scene's images."""

    scene_num: int = 0
    selected_version: int = 1             # Currently selected version number
    versions: list[ImageVersion] = field(default_factory=list)

    @property
    def selected(self) -> Optional[ImageVersion]:
        for v in self.versions:
            if v.version == self.selected_version:
                return v
        return None

    def add_version(self, version: ImageVersion):
        self.versions.append(version)

    def promote(self, version_num: int):
        self.selected_version = version_num


# =============================================================================
# PROMOTION DECISION LOG
# =============================================================================


@dataclass
class PromotionLog:
    """Log entry for a promotion decision."""

    scene_num: int = 0
    round: int = 0
    baseline_version: int = 0
    challenger_version: int = 0
    baseline_state: ValidationState = ValidationState.INCONCLUSIVE
    baseline_score: int = 0
    challenger_state: ValidationState = ValidationState.INCONCLUSIVE
    challenger_score: int = 0
    pairwise: Optional[PairwiseDecision] = None
    promoted: bool = False                # True = challenger became selected
    reason: str = ""
