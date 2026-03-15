"""
Story conception and writing — v2 (upstream creative planning).

Key changes from v1:
- Seed fidelity scoring at concept selection
- Renderability gate before story approval
- Self-contained image prompts (no leaving key objects for video prompt only)
- Scene distinctness enforcement
- Negative ruleset from real failures (stories 179-181)
- Full creative trace logging
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
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        return data.get("stories", [])
    return []


def save_concept(story_number: int, title: str, concept: str, setting: str):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_concept_history()
    history.append({
        "number": story_number,
        "title": title,
        "concept": concept,
        "setting": setting,
        "date": datetime.now().strftime("%Y-%m-%d"),
    })
    history = history[-50:]
    with open(HISTORY_FILE, "w") as f:
        json.dump({"stories": history}, f, indent=2)


# =============================================================================
# CONCEPT SELECTION PROMPT (v2 — with seed fidelity + renderability)
# =============================================================================

CONCEIVE_SYSTEM = """\
/no_think
You are a visual storytelling concept generator for TikTok-format short videos (9:16, 10-25 seconds).

Your job: Propose 3 story concepts and select the strongest one.

REQUIREMENTS:
- Each concept must be a single, clear VISUAL moment — not a plot summary
- Must work as a SILENT video (no dialogue, no narration, no text)
- Must be understandable to a stranger in 10 seconds
- Must have a clear hook (what stops the scroll?) and payoff (what's satisfying?)

SEED FIDELITY — when a user seed is provided:
- You MUST preserve the core hook of the seed (the specific thing that makes it interesting)
- You MUST preserve the emotional/narrative core (mystery, warmth, humor, surprise)
- You MUST preserve key characters and relationships described in the seed
- Do NOT replace the seed with a completely different concept
- Variations should explore different angles on the SAME core idea, not invent new ideas

RENDERABILITY — every concept must pass these checks:
- Can each scene be represented by ONE clear still photograph? If it needs continuous motion, reject it.
- Are key story objects large enough to be visible in a portrait photo? Tiny props (< 15cm) that carry the whole story will fail.
- Does the concept use standard camera angles? No inside-container POV, no through-aperture views.
- Can the scenes be visually distinguished as still photographs? If 4+ scenes would look identical, reject it.
- Does the concept require precise object counting (1 item → 2 items → 3 items)? If yes, reject it — the image model cannot count.
- Does the concept require continuous state across scenes (stacking, filling, assembling)? If yes, reject it — each scene is generated independently.

OUTPUT FORMAT (JSON):
{
  "concepts": [
    {
      "title": "...",
      "concept": "one sentence describing the visual story",
      "setting": "location",
      "hook": "what stops scrolling",
      "tone": "emotional tone",
      "seed_fidelity": "how this preserves the seed's core hook and emotion",
      "renderability_notes": "why each scene can be a distinct, clear still photograph"
    }
  ],
  "selected": 0,
  "reason": "why this concept is strongest for BOTH story quality AND renderability"
}
"""


# =============================================================================
# STORY WRITING PROMPT (v2 — self-contained prompts, scene distinctness)
# =============================================================================

WRITE_SYSTEM = """\
/no_think
You are a visual storyteller writing short stories for TikTok-format videos (9:16, 10-25 seconds total).

Each story is rendered as INDEPENDENT still images that are then animated into video clips.
The image model generates each scene separately — it has NO memory of previous scenes.
The image prompt is the ONLY instruction the image model receives.

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
[NAME_DNA]: age, gender, skin tone, face shape, jawline, nose, lips, eye shape/color, cheekbones, eyebrows, marks, hair, clothing (include "no text no logos"), build

## Location DNA Blocks
[PLACE_LOC]: type, key features, floor, walls, furniture, lighting, color palette, atmosphere, time of day (KEEP UNDER 60 WORDS — only the essential visual anchors)

## Object DNA Blocks
[ITEM_OBJ]: color, material, condition, size

---

## Scenes

### Scene N: Title (Xs, trim: XXXms)

**Scene Purpose:** [One sentence: what new visual information does this scene introduce?]

**Image Prompt:**
Subject: [KEY character features — 30-50 words max, not full DNA dump] | Pose: [position, gesture, gaze] | Camera: [standard angle, lens] | Environment: [KEY location features — 20-30 words max] | Lighting: [source, direction] | Mood: [2-3 words], photograph

**Video Prompt:**
[4-6 sentence paragraph describing motion from the still image starting point]

**Trim Notes:** 500ms

---

CRITICAL RULES:

IMAGE PROMPT RULES:
1. Image prompts MUST be UNDER 400 CHARACTERS TOTAL. Front-load the subject.
2. Do NOT dump the entire DNA block into every prompt. Use only the KEY identifying features (age, hair, clothing color, 1-2 distinctive marks).
3. Do NOT dump the entire location DNA. Use only the KEY setting anchors (3-5 words).
4. Every object that is VISIBLE and IMPORTANT in this scene MUST be named in the image prompt Subject or Pose field. If the scene is about a bottle on the sand, "bottle on sand" must be in the image prompt.
5. Image = START state. It must be a complete, self-contained still photograph. A human should understand the scene from the image prompt alone without reading the video prompt.
6. Use STANDARD camera angles only: eye-level, low angle, high angle, close-up, medium, wide. No inside-container POV, no through-object views.

SCENE RULES:
7. Each scene MUST introduce new visual information visible as a STILL PHOTOGRAPH.
8. Adjacent scenes MUST be visually distinguishable: different subject, different angle, different composition, or different key object state.
9. Do NOT write 3+ scenes that are the same person from the same angle with tiny pose changes. Change the camera, the framing, or the visible content.
10. If a scene's key difference from the previous scene is only in motion (hand moves, object falls), merge it with the adjacent scene or find a still-frame-visible difference.

STORY RULES:
11. 5-6 scenes maximum. Fewer is better if each scene is distinct.
12. Video prompts describe motion FROM the still image. Do not describe objects in the video prompt that are not visible in the image prompt.
13. End with "photograph" in the Mood field.
14. Clothing MUST include "no text no logos".
"""


# =============================================================================
# CONCEIVE (v2)
# =============================================================================

def conceive(
    config: dict,
    concept_seed: Optional[str] = None,
) -> dict:
    """Generate and select a story concept with seed fidelity + renderability scoring."""
    model = config["models"]["story_llm"]["model"]
    history = load_concept_history()

    history_text = ""
    if history:
        recent = history[-20:]
        history_text = "RECENT STORIES (avoid repeating these concepts/settings):\n"
        for h in recent:
            history_text += f"  #{h['number']}: {h['title']} — {h['concept']} ({h['setting']})\n"

    user_msg = history_text + "\n"
    if concept_seed:
        user_msg += (
            f"USER CONCEPT SEED: {concept_seed}\n\n"
            f"Build on this seed. Propose 3 variations that PRESERVE the seed's core hook.\n"
            f"Do NOT replace the seed with a different concept.\n"
            f"Each variation must be renderable as 5-6 distinct still photographs."
        )
    else:
        user_msg += "Propose 3 fresh, original story concepts."

    messages = [
        {"role": "system", "content": CONCEIVE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    response = llm.chat(
        model=model,
        messages=messages,
        temperature=config["models"]["story_llm"].get("temperature", 0.7),
        format_json=False,
        max_tokens=config["models"]["story_llm"].get("max_tokens", 16384),
    )

    data = _parse_json_response(response)
    selected_idx = data.get("selected", 0)
    concept = data["concepts"][selected_idx]

    # Save full creative trace
    _save_creative_trace(config, data, concept_seed)

    log.info(f"Concept selected: {concept['title']} — {concept['concept']}")
    if concept.get("seed_fidelity"):
        log.info(f"  Seed fidelity: {concept['seed_fidelity'][:80]}")
    if concept.get("renderability_notes"):
        log.info(f"  Renderability: {concept['renderability_notes'][:80]}")

    return concept


# =============================================================================
# WRITE (v2)
# =============================================================================

def write_story(
    concept: dict,
    story_number: int,
    config: dict,
) -> Path:
    """Write a full story markdown file with self-contained prompts."""
    model = config["models"]["story_llm"]["model"]

    user_msg = (
        f"Write Story #{story_number}.\n\n"
        f"CONCEPT: {concept['concept']}\n"
        f"TITLE: {concept['title']}\n"
        f"SETTING: {concept['setting']}\n"
        f"TONE: {concept.get('tone', 'dramatic')}\n"
        f"HOOK: {concept.get('hook', '')}\n\n"
        f"REMEMBER:\n"
        f"- Image prompts must be UNDER 400 characters. Front-load the subject.\n"
        f"- Every key object VISIBLE in the scene must be named in the image prompt.\n"
        f"- Each scene must be visually distinct from adjacent scenes as a still photo.\n"
        f"- Use standard camera angles only. No inside-container POV.\n"
        f"- Location DNA under 60 words. Character features in prompt under 50 words.\n\n"
        f"Write the complete story markdown now."
    )

    messages = [
        {"role": "system", "content": WRITE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    response = llm.chat(
        model=model,
        messages=messages,
        temperature=config["models"]["story_llm"].get("temperature", 0.7),
        max_tokens=config["models"]["story_llm"].get("max_tokens", 16384),
    )

    story_text = _extract_markdown(response)

    slug = re.sub(r"[^a-z0-9]+", "-", concept["title"].lower()).strip("-")[:40]
    filename = f"story-{story_number}-{slug}.md"
    story_path = Path("stories") / filename
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text(story_text)

    save_concept(story_number, concept["title"], concept["concept"], concept["setting"])

    log.info(f"Story written: {story_path}")
    return story_path


# =============================================================================
# CREATIVE TRACE LOGGING
# =============================================================================

def _save_creative_trace(config: dict, data: dict, seed: Optional[str]):
    """Save the full concept selection trace for later inspection."""
    trace_dir = Path("stories/creative_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "all_concepts": data.get("concepts", []),
        "selected_index": data.get("selected", 0),
        "selection_reason": data.get("reason", ""),
    }

    # Save with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = trace_dir / f"trace_{ts}.json"
    with open(trace_file, "w") as f:
        json.dump(trace, f, indent=2)

    log.info(f"Creative trace saved: {trace_file}")


# =============================================================================
# HELPERS
# =============================================================================

def _extract_markdown(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _parse_json_response(response: str) -> dict:
    text = response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")
