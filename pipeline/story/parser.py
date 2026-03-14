"""
Story markdown parser.

Parses story files into structured data: DNA blocks, scenes, prompts.
Story format follows the established convention from 171 stories.
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Scene:
    number: int
    title: str
    duration_s: float = 3.0
    trim_ms: int = 500
    image_prompt: str = ""
    video_prompt: str = ""


@dataclass
class StoryData:
    title: str = ""
    story_number: int = 0
    genre: str = ""
    duration: str = ""
    theme: str = ""
    summary: str = ""
    character_dna: dict[str, str] = field(default_factory=dict)  # {TAG: description}
    location_dna: dict[str, str] = field(default_factory=dict)
    object_dna: dict[str, str] = field(default_factory=dict)
    scenes: list[Scene] = field(default_factory=list)
    raw_text: str = ""


def parse_story(path: str | Path) -> StoryData:
    """Parse a story markdown file into StoryData.

    Expected format:
        # Story #172: Title
        **Genre:** ...
        **Duration:** ...
        **Theme:** ...
        ## Story Summary
        ...
        ## Character DNA Blocks
        [NAME_DNA]: description
        ## Location DNA Blocks
        [PLACE_LOC]: description
        ## Object DNA Blocks
        [ITEM_OBJ]: description
        ## Scenes
        ### Scene 1: Title (3s, trim: 500ms)
        **Image Prompt:**
        ...
        **Video Prompt:**
        ...
        **Trim Notes:** 500ms
    """
    path = Path(path)
    text = path.read_text()
    story = StoryData(raw_text=text)

    # Parse title and number
    title_match = re.search(r"#\s*Story\s*#(\d+):\s*(.+)", text)
    if title_match:
        story.story_number = int(title_match.group(1))
        story.title = title_match.group(2).strip()

    # Parse metadata
    for key, attr in [("Genre", "genre"), ("Duration", "duration"), ("Theme", "theme")]:
        m = re.search(rf"\*\*{key}:\*\*\s*(.+)", text)
        if m:
            setattr(story, attr, m.group(1).strip())

    # Parse summary
    summary_match = re.search(
        r"##\s*Story\s*Summary.*?\n(.+?)(?=\n---|\n##)", text, re.DOTALL
    )
    if summary_match:
        story.summary = summary_match.group(1).strip()

    # Parse DNA blocks
    story.character_dna = _parse_dna_section(text, "Character DNA")
    story.location_dna = _parse_dna_section(text, "Location DNA")
    story.object_dna = _parse_dna_section(text, "Object DNA")

    # Parse scenes
    story.scenes = _parse_scenes(text)

    log.info(
        f"Parsed: {story.title} — "
        f"{len(story.character_dna)} chars, {len(story.location_dna)} locs, "
        f"{len(story.object_dna)} objs, {len(story.scenes)} scenes"
    )

    return story


def _parse_dna_section(text: str, section_name: str) -> dict[str, str]:
    """Extract DNA blocks from a named section.

    DNA format: [TAG_SUFFIX]: description (can span multiple lines until next [ or ##)
    """
    # Find section
    pattern = rf"##\s*{section_name}.*?\n(.*?)(?=\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {}

    section = match.group(1)
    dna = {}

    # Find all DNA entries — supports both [TAG]: and TAG: formats
    # Use findall with MULTILINE to match tags at start of line
    for m in re.finditer(
        r"^[ \t]*\[?([A-Z][A-Z0-9_]{2,})\]?:\s*(.+?)(?=\n[ \t]*\[?[A-Z][A-Z0-9_]{2,}\]?:|\Z)",
        section,
        re.DOTALL | re.MULTILINE,
    ):
        tag = m.group(1)
        desc = m.group(2).strip()
        # Normalize whitespace (multi-line descriptions)
        desc = re.sub(r"\s+", " ", desc)
        dna[tag] = desc

    return dna


def _parse_scenes(text: str) -> list[Scene]:
    """Parse all scenes from story text."""
    scenes = []

    # Split on scene headers: ### Scene N: Title (Xs, trim: XXXms)
    scene_splits = re.split(r"(?=###\s*Scene\s+\d+)", text)

    for block in scene_splits:
        block = block.strip()
        if not block.startswith("### Scene"):
            continue

        # Parse header
        header_match = re.match(
            r"###\s*Scene\s+(\d+):\s*(.+?)(?:\((\d+)s(?:,\s*trim:\s*(\d+)ms)?\))?\s*$",
            block.split("\n")[0],
        )
        if not header_match:
            # Try simpler format: ### Scene N: Title
            header_match = re.match(
                r"###\s*Scene\s+(\d+):\s*(.+)",
                block.split("\n")[0],
            )
            if not header_match:
                continue
            scene = Scene(
                number=int(header_match.group(1)),
                title=header_match.group(2).strip().rstrip(")"),
            )
        else:
            scene = Scene(
                number=int(header_match.group(1)),
                title=header_match.group(2).strip(),
                duration_s=float(header_match.group(3) or 3),
                trim_ms=int(header_match.group(4) or 500),
            )

        # Parse image prompt
        img_match = re.search(
            r"\*\*Image Prompt:\*\*\s*\n(.+?)(?=\n\*\*Video Prompt|\n\*\*Trim|\n---|\n###|\Z)",
            block,
            re.DOTALL,
        )
        if img_match:
            scene.image_prompt = img_match.group(1).strip()

        # Parse video prompt
        vid_match = re.search(
            r"\*\*Video Prompt:\*\*\s*\n(.+?)(?=\n\*\*Trim|\n---|\n###|\Z)",
            block,
            re.DOTALL,
        )
        if vid_match:
            scene.video_prompt = vid_match.group(1).strip()

        # Parse trim notes (overrides header trim)
        trim_match = re.search(r"\*\*Trim Notes?:\*\*\s*(\d+)\s*ms", block)
        if trim_match:
            scene.trim_ms = int(trim_match.group(1))

        scenes.append(scene)

    scenes.sort(key=lambda s: s.number)
    return scenes


def expand_dna(prompt: str, story: StoryData) -> str:
    """Replace DNA tags in a prompt with their descriptions.

    [NAME_DNA] → full character description
    [PLACE_LOC] → full location description
    [ITEM_OBJ] → full object description
    """
    all_dna = {}
    all_dna.update(story.character_dna)
    all_dna.update(story.location_dna)
    all_dna.update(story.object_dna)

    result = prompt
    for tag, desc in all_dna.items():
        # Replace both [TAG] and bare TAG references
        result = result.replace(f"[{tag}]", desc)
        # Also handle inline DNA expansion where the LLM copy-pasted "TAG: full desc"
        # by replacing "TAG: <existing desc>" with just the desc
        result = re.sub(rf"\b{re.escape(tag)}:\s*{re.escape(desc[:30])}[^|]*", desc, result)

    return result
