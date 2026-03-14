# Video Pipeline v3 — Review Repository

Automated pipeline that generates short-form vertical video stories (TikTok/Reels format, 10-25 seconds) from a concept seed. Takes a text concept, writes a story, generates images, generates videos, and combines them into a final MP4.

## Pipeline Stages

```
CONCEIVE → WRITE → GATE → VALIDATE SCENES → VALIDATE PROMPTS
                                                    ↓
                                            GENERATE IMAGES
                                                    ↓
                                            VALIDATE IMAGES
                                                    ↓
                                        CHALLENGE (if FAIL scenes)
                                                    ↓
                                            GENERATE VIDEOS
                                                    ↓
                                              COMBINE → DONE
```

### Stage Details

| # | Stage | Model | VRAM | What it does |
|---|-------|-------|------|-------------|
| 1 | **CONCEIVE** | Qwen3.5-35B-A3B (Ollama) | ~10 GB | Brainstorm 3 concepts, pick strongest |
| 2 | **WRITE** | Qwen3.5-35B-A3B | ~10 GB | Write full story markdown with DNA blocks + prompts |
| 3 | **GATE** | None (code) | 0 | Deterministic structural checks (scene count, format) |
| 4 | **VALIDATE SCENES** | Qwen3.5-35B-A3B | ~10 GB | Semantic scene structure check (hook, payoff, pacing) |
| 5 | **VALIDATE PROMPTS** | Qwen3.5-35B-A3B | ~10 GB | Semantic prompt check (DNA present, no contradictions) |
| 6 | **GENERATE IMAGES** | Qwen-Image-2512 (ComfyUI) | ~23 GB | One image per scene via GGUF Q8 |
| 7 | **VALIDATE IMAGES** | Qwen3-VL-30B-A3B (Ollama) | ~10 GB | Vision check against prompt + story intent |
| 8 | **CHALLENGE** | Both (batched) | Sequential | Rewrite prompt → generate challenger → pairwise judge |
| 9 | **GENERATE VIDEOS** | LTX-2.3 22B (ComfyUI) | ~31 GB | Image-to-video, 1088x1920, with audio |
| 10 | **COMBINE** | ffmpeg | 0 | Trim + concatenate scene videos |

Models run sequentially, never concurrently (32 GB GPU, 16 GB RAM).

## State Machine

```python
IDLE → CONCEIVING → WRITING → GATING
  → VALIDATING_SCENES       # Semantic scene check
  → VALIDATING_PROMPTS      # Semantic prompt check
  → GENERATING_IMAGES
  → VALIDATING_IMAGES       # Vision-based image check
  → CHALLENGING_IMAGES      # Challenger generation + pairwise judge
  → GENERATING_VIDEOS
  → COMBINING → COMPLETED
```

State persisted to `stories/state.json` before each transition. Resumable from any state after crash.

## Layered Validation Architecture

Each validation layer protects the next expensive stage. Each layer only repairs at its own abstraction level:

| Layer | What it checks | On FAIL | On VALIDATOR_ERROR |
|-------|---------------|---------|-------------------|
| **Gate** (code) | Structural format | Rewrite story | N/A |
| **Scene validator** (LLM) | Narrative structure, hook, payoff | Rewrite story | Keep, proceed |
| **Prompt validator** (LLM) | DNA present, no contradictions, static frame | Rewrite prompt only | Keep, proceed |
| **Image validator** (LLM vision) | Subject, setting, composition match | Generate challenger | Keep, proceed |
| **Pairwise judge** (LLM vision) | Side-by-side baseline vs challenger | Promote if CHALLENGER wins | Keep baseline |

### Validation States

```python
PASS              # Artifact is good, proceed
FAIL              # Artifact has major issues, trigger repair
INCONCLUSIVE      # Can't determine — keep artifact (benefit of doubt)
VALIDATOR_ERROR   # Validator itself broke (parse failure) — keep artifact
```

**Critical rule**: Parse failure = `VALIDATOR_ERROR`, never `FAIL`. A broken validator must not destroy a good artifact.

### Challenger System (Non-Destructive Image Retry)

When the image validator returns FAIL for a scene:

1. Original image is backed up as `scene_XX_v1.png` (never deleted)
2. Prompt is rewritten from the ORIGINAL prompt (not previous rewrite)
3. Challenger is generated to `scene_XX_v2.png` (separate file)
4. Challenger is validated independently
5. Pairwise judge sees BOTH images and decides: BASELINE, CHALLENGER, or TIE
6. TIE → keep baseline (incumbent advantage)
7. Only CHALLENGER wins → safe promotion via `tempfile + os.replace()`
8. Full version history tracked in `image_history.json`

VRAM-aware batched flow:
- Phase 1: Rewrite all prompts (Ollama LLM)
- Phase 2: Generate all challengers (ComfyUI)
- Phase 3: Validate + pairwise compare (Ollama LLM)

Never runs LLM + ComfyUI simultaneously.

### Rubrics

Stored as versioned YAML files in `pipeline/rubrics/`:

| Rubric | Criteria | Used by |
|--------|----------|---------|
| `scene_rubric_v1.yaml` | 8 criteria (hook, payoff, pacing, etc.) | Scene validator |
| `prompt_rubric_v1.yaml` | 8 criteria (DNA present, static frame, etc.) | Prompt validator |
| `image_rubric_v1.yaml` | 8 criteria (subject, setting, readability) | Image validator |
| `pairwise_rubric_v1.yaml` | 4 criteria (adherence, intent, continuity) | Pairwise judge |

Each rubric contains criteria definitions and an LLM prompt template. Treated as versioned software artifacts.

### Deterministic Checks (Before LLM)

At each layer, fast code-based checks run before the LLM:

- **Scene**: count, runtime, duplicate titles, missing prompts
- **Prompt**: pipe-separated fields, DNA references, motion verbs, length
- **Image**: file exists, valid PNG, correct dimensions, not suspiciously small
- **Duplicate detection**: hash-based near-duplicate across scene images

## Benchmark Results (Scene Validator, max_tokens=8192)

```
Parse success rate: 25/25 (100%)
Agreement rate:     16/20 (80%)
False FAIL rate:    1/10
False PASS rate:    3/10
```

## 3-Story Verification Results

| Metric | Story 1 | Story 2 | Story 3 |
|--------|---------|---------|---------|
| Scene validator | PASS (0.85) | PASS (0.95) | PASS (0.95) |
| Image validator | 3P/1F/1E | 1P/2F/2E | 2P/2F/1E |
| Challengers | 1 | 3 | 4 |
| Promotions | 1 | 2 | 1 |
| Output | 14.7 MB | 13.7 MB | 18.6 MB |
| Runtime | 22 min | 25 min | 29 min |

Zero images destroyed across all 3 runs. All originals preserved as `_v1.png`.

## Legacy Code

The following files exist in the working repo but are **NOT imported by `main.py`**:

- `pipeline/image/validator.py` — old image validator with destructive overwrite logic. Replaced by `pipeline/validators/image.py`.
- State machine states `REWRITING_PROMPTS` and `REGENERATING_IMAGES` — defined for backward compatibility but never entered by the new orchestrator.

## File Structure

```
pipeline/
  main.py                    # Orchestrator (stages 1-10)
  state.py                   # State machine with JSON persistence
  __main__.py                # python -m pipeline support
  story/
    creator.py               # Conceive + Write via Qwen3.5
    parser.py                # Story markdown parser (DNA blocks, scenes)
    gates.py                 # Deterministic structural checks
  validators/
    schema.py                # ValidationState enum + all dataclasses
    deterministic.py         # Code-based checks (scene, prompt, image)
    scene.py                 # LLM scene validator
    prompt.py                # LLM prompt validator
    image.py                 # LLM vision image validator
    pairwise.py              # Pairwise baseline vs challenger judge
  image/
    generator.py             # Qwen-Image-2512 via ComfyUI
    challenger.py            # Challenger gen + promotion + version history
    rewriter.py              # Prompt rewriting (blind or vision-guided)
  video/
    generator.py             # LTX-2.3 22B via ComfyUI
    combiner.py              # ffmpeg trim + concat
  rubrics/
    scene_rubric_v1.yaml     # Versioned rubric files
    prompt_rubric_v1.yaml
    image_rubric_v1.yaml
    pairwise_rubric_v1.yaml
  llm/
    local.py                 # Ollama API wrapper
  utils/
    comfyui.py               # ComfyUI lifecycle management
    gpu.py                   # VRAM monitoring
    files.py                 # Nextcloud upload
  notify/
    telegram.py              # Telegram notifications
tests/
  test_validators.py         # Calibration tests
  test_failure_modes.py      # Safety proof tests
  test_validator_fixtures.py # 13 fixture tests with mocked LLM
  benchmarks/validators/
    run_benchmark.py          # Benchmark runner (25 cases)
    scene_benchmark.json      # 10 good, 10 bad, 5 ambiguous
artifacts/
  story_176/                 # Complete run artifacts
  raw_model_outputs/         # Exact LLM outputs for each validator
```

## Usage

```bash
python -m pipeline new "a janitor discovers a painting changes every night"
python -m pipeline new --story-only "concept"   # Stop after prompt validation
python -m pipeline resume                        # Resume from crash
python -m pipeline status                        # Show current state
```

## Hardware

- GPU: NVIDIA RTX 5090 (32 GB VRAM)
- RAM: 16 GB
- Models run sequentially via Ollama (LLM) and ComfyUI (image/video)
