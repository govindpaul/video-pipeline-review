"""
Story conception and writing via Qwen3.5-35B-A3B.

Stage 1 (CONCEIVE): Generate concept ideas, pick the best one
Stage 2 (WRITE): Write full story markdown with DNA blocks + scenes
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.llm import local as llm

log = logging.getLogger(__name__)

# --- Concept History ---

HISTORY_FILE = Path("stories/learnings/concept-history.json")


def load_concept_history() -> list[dict]:
    """Load recent concept history to avoid repeats."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        return data.get("stories", [])
    return []


def save_concept(story_number: int, title: str, concept: str, setting: str):
    """Append a concept to history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_concept_history()
    history.append({
        "number": story_number,
        "title": title,
        "concept": concept,
        "setting": setting,
        "date": datetime.now().strftime("%Y-%m-%d"),
    })
    # Keep last 50
    history = history[-50:]
    with open(HISTORY_FILE, "w") as f:
        json.dump({"stories": history}, f, indent=2)


# --- System Prompts ---

CONCEIVE_SYSTEM = """\
/no_think
You are a visual storytelling concept generator for TikTok-format short videos (9:16, 10-25 seconds).

Your job: Propose 3 story concepts and select the strongest one.

REQUIREMENTS:
- Each concept must be a single, clear VISUAL moment — not a plot summary
- Must work as a SILENT video (no dialogue, no narration, no text)
- Must be understandable to a stranger in 10 seconds
- Must have a clear hook (what stops the scroll?) and payoff (what's satisfying?)
- Vary settings: indoor, outdoor, home, workplace, nature, city, etc.
- Vary character types: age, gender, occupation, relationships
- Vary emotional tones: comedy, wonder, tension, warmth, melancholy, surprise

OUTPUT FORMAT (JSON):
{
  "concepts": [
    {"title": "...", "concept": "one sentence", "setting": "location", "hook": "why it stops scrolling", "tone": "emotional tone"},
    {"title": "...", "concept": "...", "setting": "...", "hook": "...", "tone": "..."},
    {"title": "...", "concept": "...", "setting": "...", "hook": "...", "tone": "..."}
  ],
  "selected": 0,
  "reason": "why this concept is strongest"
}
"""

WRITE_SYSTEM = """\
/no_think
You are a visual storyteller writing short stories for TikTok-format videos (9:16, 10-25 seconds total).

OUTPUT FORMAT — write EXACTLY this markdown structure:

# Story #[NUMBER]: [Title]

**Genre:** [Genre]
**Duration:** [total seconds]s
**Theme:** [One sentence]

---

## Story Summary
[One sentence describing ONLY what you SEE, not feelings]

---

## Character DNA Blocks
[NAME_DNA]: age, gender, skin tone, face shape, jawline, nose, lips, eye shape/color/lashes, cheekbones, eyebrows, distinctive marks, hair, clothing (include "no text no logos"), accessories, build

## Location DNA Blocks
[PLACE_LOC]: type of place, architectural features, floor/ground material + color, wall material + color, key furniture/fixed objects, lighting source + direction + quality, color palette, atmosphere/weather, time of day, unique details (minimum 30 words)

## Object DNA Blocks
[ITEM_OBJ]: color, material, condition, visible damage, exact size cues (be specific and detailed)

---

## Scenes

### Scene 1: Title (3s, trim: 500ms)

**Image Prompt:**
Subject: [character DNA description, expression] | Pose: [body position, gesture, gaze direction with face-angle enforcement] | Camera: [shot type, angle, lens spec, depth of field] | Environment: [location DNA expanded] | Lighting: [source, direction, quality, color temperature] | Mood: [2-4 words], photograph

**Video Prompt:**
[Lighting/color palette], [shot size], [composition].

[Character starting position and expression.]

Subsequently, [action 1]. [emotional shift]

Then, [action 2]. [intermediate state]

Finally, [action 3/conclusion], [final state].

[Background/atmosphere, camera movement if any.]

**Trim Notes:** 500ms

---

### Scene 2: Title (3s, trim: 500ms)
[... repeat for each scene ...]

---

CRITICAL RULES:
1. Write 5-8 scenes, each 3-5 seconds duration
2. Image prompts MUST use pipe-separated format: Subject: | Pose: | Camera: | Environment: | Lighting: | Mood:
3. Video prompts use paragraph narrative with structural words (subsequently, then, finally)
4. Character DNA MUST appear in EVERY scene's image prompt (copy key features, don't just use [TAG])
5. Location DNA MUST appear in EVERY scene's Environment field
6. Object DNA descriptions MUST be copy-pasted exactly in every scene they appear
7. Image = START state (static before-state, no motion), Video = JOURNEY (motion/action)
8. Clothing MUST include "no text no logos" to prevent AI rendering text
9. Use real camera specs: 85mm, 50mm, 35mm lens, f/2.8, shallow/deep DoF
10. Video prompts: NO [DNA_TAG] references — use pronouns (he/she/they) and descriptive terms
11. Trim: 500ms for face-visible scenes, 0ms for profile/object-only/no-face shots
12. Keep it purely visual — no internal thoughts, no dialogue, no voiceover cues
13. Face-angle enforcement in Pose: physically describe head tilt ("face tilted downward, crown of head visible") not just "looking down"
14. End with "photograph" in the Mood field (not "photorealistic" or "3d render")
"""


# --- Conceive ---

def conceive(
    config: dict,
    concept_seed: Optional[str] = None,
) -> dict:
    """Generate and select a story concept.

    Args:
        config: Pipeline config dict
        concept_seed: Optional user-provided concept idea

    Returns:
        Selected concept dict with title, concept, setting, hook, tone
    """
    model = config["models"]["story_llm"]["model"]
    history = load_concept_history()

    # Build context about recent stories
    history_text = ""
    if history:
        recent = history[-20:]
        history_text = "RECENT STORIES (avoid repeating these concepts/settings):\n"
        for h in recent:
            history_text += f"  #{h['number']}: {h['title']} — {h['concept']} ({h['setting']})\n"

    user_msg = history_text + "\n"
    if concept_seed:
        user_msg += f"USER CONCEPT SEED: {concept_seed}\n\nBuild on this seed. Propose 3 variations and select the strongest."
    else:
        user_msg += "Propose 3 fresh, original story concepts. Avoid anything similar to the recent stories listed above."

    messages = [
        {"role": "system", "content": CONCEIVE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    response = llm.chat(
        model=model,
        messages=messages,
        temperature=config["models"]["story_llm"].get("temperature", 0.7),
        format_json=False,  # Ollama JSON mode conflicts with Qwen3.5 thinking
        max_tokens=config["models"]["story_llm"].get("max_tokens", 16384),
    )

    data = _parse_json_response(response)
    selected_idx = data.get("selected", 0)
    concept = data["concepts"][selected_idx]

    log.info(f"Concept selected: {concept['title']} — {concept['concept']}")
    return concept


# --- Write ---

def write_story(
    concept: dict,
    story_number: int,
    config: dict,
) -> Path:
    """Write a full story markdown file from a concept.

    Args:
        concept: Concept dict from conceive()
        story_number: Story number (e.g. 172)
        config: Pipeline config dict

    Returns:
        Path to the written story file
    """
    model = config["models"]["story_llm"]["model"]

    user_msg = (
        f"Write Story #{story_number}.\n\n"
        f"CONCEPT: {concept['concept']}\n"
        f"TITLE: {concept['title']}\n"
        f"SETTING: {concept['setting']}\n"
        f"TONE: {concept.get('tone', 'dramatic')}\n"
        f"HOOK: {concept.get('hook', '')}\n\n"
        f"Write the complete story markdown now. Follow the format EXACTLY."
    )

    messages = [
        {"role": "system", "content": WRITE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    response = llm.chat(
        model=model,
        messages=messages,
        temperature=config["models"]["story_llm"].get("temperature", 0.7),
        max_tokens=config["models"]["story_llm"].get("max_tokens", 4096),
    )

    # Clean up response — extract just the markdown
    story_text = _extract_markdown(response)

    # Generate filename slug
    slug = re.sub(r"[^a-z0-9]+", "-", concept["title"].lower()).strip("-")[:40]
    filename = f"story-{story_number}-{slug}.md"
    story_path = Path("stories") / filename
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text(story_text)

    # Save to concept history
    save_concept(story_number, concept["title"], concept["concept"], concept["setting"])

    log.info(f"Story written: {story_path}")
    return story_path


def _extract_markdown(response: str) -> str:
    """Extract markdown content from LLM response, removing any wrapping."""
    text = response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```markdown or ```)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip()


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response, handling code fences and other wrapping."""
    text = response.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    import re
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")
