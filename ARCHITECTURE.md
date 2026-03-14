# Import Graph and Execution Map

## What `main.py` imports and calls for stages 3-8

```
main.py
  │
  ├── pipeline.state (PipelineState, State)
  │     State transitions: GATING → VALIDATING_SCENES → VALIDATING_PROMPTS
  │                        → GENERATING_IMAGES → VALIDATING_IMAGES
  │                        → CHALLENGING_IMAGES → GENERATING_VIDEOS
  │
  ├── pipeline.story.gates (gate_check)
  │     Stage 3: GATE
  │     Pure code validation, no imports beyond parser
  │
  ├── pipeline.validators.scene (validate_scenes)
  │     Stage 4: VALIDATE_SCENES
  │     ├── validators.deterministic.check_scenes_deterministic
  │     ├── pipeline.rubrics.load_rubric("scene")
  │     └── pipeline.llm.local.chat (Qwen3.5, max_tokens=8192)
  │
  ├── pipeline.validators.prompt (validate_all_prompts)
  │     Stage 5: VALIDATE_PROMPTS
  │     ├── validators.deterministic.check_prompt_deterministic
  │     ├── pipeline.rubrics.load_rubric("prompt")
  │     └── pipeline.llm.local.chat (Qwen3.5, max_tokens=8192)
  │
  ├── pipeline.image.generator (generate_scene_images)
  │     Stage 6: GENERATE_IMAGES
  │     ├── pipeline.utils.comfyui (start, stop, submit, wait, get_output)
  │     └── EXTERNAL: video_gen_web/workflows/qwen_image.py (build_workflow)
  │
  ├── pipeline.validators.image (validate_all_images)
  │     Stage 7: VALIDATE_IMAGES
  │     ├── validators.deterministic.check_image_deterministic
  │     ├── validators.deterministic.check_image_duplicates
  │     ├── pipeline.rubrics.load_rubric("image")
  │     └── pipeline.llm.local.vision (Qwen3-VL)
  │
  ├── pipeline.image.challenger (run_challenges, setup_initial_history)
  │     Stage 8: CHALLENGING_IMAGES
  │     ├── pipeline.image.rewriter.rewrite_prompt
  │     │     └── pipeline.llm.local.chat or .vision
  │     ├── pipeline.utils.comfyui (generate challenger images)
  │     │     └── EXTERNAL: video_gen_web/workflows/qwen_image.py
  │     ├── pipeline.validators.image.validate_image (validate challenger)
  │     └── pipeline.validators.pairwise.compare_images (side-by-side judge)
  │           └── pipeline.llm.local (Ollama API, two images)
  │
  └── pipeline.video.generator (generate_scene_videos)
        Stage 9: GENERATE_VIDEOS
        ├── pipeline.utils.comfyui (start, restart between scenes, stop)
        └── EXTERNAL: tests/benchmarks/.../ltx23_i2v_twostage_v2.py (build_workflow)
```

## VRAM Sequencing in Stages 3-8

```
Stage 3-5:  Ollama loaded (Qwen3.5 ~10GB)
            ↓ llm.stop_all()
Stage 6:    ComfyUI loaded (Qwen-Image ~23GB)
            ↓ comfyui.stop()
Stage 7:    Ollama loaded (Qwen3-VL ~10GB)
            ↓ (if challengers needed)
Stage 8:    Batched phases:
              Phase 1: Ollama (rewrite)     → stop
              Phase 2: ComfyUI (generate)   → stop
              Phase 3: Ollama (validate+compare) → stop
            ↓
Stage 9:    ComfyUI loaded (LTX-2.3 ~31GB, restart between scenes)
```

Never runs Ollama + ComfyUI simultaneously. Each transition verified by
checking VRAM < 500 MB before starting next model.

## External Dependencies

Two workflow files are imported from outside this repo:

1. **Qwen-Image workflow**: `/path/to/video_gen_web/workflows/qwen_image.py`
   - `build_workflow(prompt, seed, ...)` → ComfyUI API JSON
   - 11 nodes: UnetLoaderGGUF → LoraLoader → ModelSamplingAuraFlow → KSampler → VAEDecode → SaveImage

2. **LTX-2.3 I2V workflow**: `/path/to/tests/benchmarks/.../ltx23_i2v_twostage_v2.py`
   - `build_workflow(source_image, prompt, ...)` → ComfyUI API JSON
   - 33 nodes: Two-stage spatial upscale, distilled 8-step, with audio

These are imported via `sys.path.insert()` — not copied, so fixes to the workflow
files apply everywhere.

## Legacy Files (not imported by new main.py)

| File | Status | Why it exists |
|------|--------|--------------|
| `pipeline/image/validator.py` | **DEAD CODE** | Old validator with destructive overwrite. Replaced by `pipeline/validators/image.py`. Not imported anywhere in the new flow. |
| `State.REWRITING_PROMPTS` | **DEAD STATE** | Defined in state.py for backward compat with old state.json files. Never entered by `run_new()`. |
| `State.REGENERATING_IMAGES` | **DEAD STATE** | Same as above. |

Verified by `test_failure_modes.py::test_old_destructive_path_unreachable()` which
inspects `main.run_new` source code and confirms no references to old states or old imports.
