"""
Video Pipeline v3 — CLI entry point and orchestrator.

Layered validation pipeline:
  1. CONCEIVE  (Ollama Qwen3.5 ~10GB)
  2. WRITE     (Ollama Qwen3.5 ~10GB)
  3. GATE      (deterministic structural checks)
  4. VALIDATE SCENES  (deterministic + LLM semantic) ← NEW
  5. VALIDATE PROMPTS (deterministic + LLM semantic) ← NEW
  6. [stop Ollama, start ComfyUI]
  7. GENERATE IMAGES  (ComfyUI + Qwen-Image ~23GB)
  8. [stop ComfyUI, start Ollama]
  9. VALIDATE IMAGES  (deterministic + LLM vision) ← REDESIGNED
  10. CHALLENGE IMAGES (rewrite + generate challenger + pairwise judge) ← REDESIGNED
  11. [stop Ollama, start ComfyUI]
  12. GENERATE VIDEOS  (ComfyUI + LTX-2.3 ~28-31GB)
  13. COMBINE + TRIM   (ffmpeg, no GPU)
  14. OUTPUT + NOTIFY   (Telegram, Nextcloud)

Key design rules:
  - Parse failure = VALIDATOR_ERROR, not image FAIL
  - VALIDATOR_ERROR / INCONCLUSIVE = keep current artifact
  - Only FAIL triggers repair at the appropriate layer
  - Scene fail → rewrite story. Prompt fail → rewrite prompt. Image fail → challenger.
  - Never delete original. Challenger is a separate file. Promote only if pairwise wins.
  - TIE = keep baseline (incumbent advantage).

Usage:
  python -m pipeline new "comedy about a chef who can't stop seasoning"
  python -m pipeline new --story-only "concept here"
  python -m pipeline resume
  python -m pipeline status
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from pipeline.state import PipelineState, State
from pipeline.llm import local as llm
from pipeline.story.creator import conceive, write_story
from pipeline.story.parser import parse_story, expand_dna
from pipeline.story.gates import gate_check
from pipeline.validators.scene import validate_scenes
from pipeline.validators.prompt import validate_all_prompts, validate_prompt
from pipeline.validators.image import validate_all_images
from pipeline.validators.schema import ValidationState
from pipeline.image.generator import generate_scene_images
from pipeline.image.challenger import run_challenges, setup_initial_history
from pipeline.image.rewriter import rewrite_prompt
from pipeline.video.generator import generate_scene_videos
from pipeline.video.combiner import combine_videos
from pipeline.notify.telegram import send_message, send_video
from pipeline.utils import comfyui
from pipeline.utils.files import upload_to_nextcloud

log = logging.getLogger("pipeline")

BASE_DIR = Path("/home/bbnlabs5/video-pipeline")


def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(story_number: int = 0):
    handlers = [logging.StreamHandler(sys.stdout)]
    if story_number:
        log_dir = BASE_DIR / "stories" / "output" / str(story_number)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "run.log", mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def run_new(concept_seed: str = None, story_only: bool = False):
    """Run the full pipeline for a new story."""
    config = load_config()
    state = PipelineState(BASE_DIR / "stories")

    story_number = state.get_next_story_number(
        start=config["project"].get("story_start", 172)
    )
    state.start_new_story(story_number)
    setup_logging(story_number)

    output_dir = BASE_DIR / "stories" / "output" / str(story_number)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'='*60}")
    log.info(f"  VIDEO PIPELINE v3 — Story #{story_number}")
    log.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*60}")

    start_time = time.time()

    try:
        # =================================================================
        # Stage 1: CONCEIVE
        # =================================================================
        state.transition(State.CONCEIVING)
        llm.ensure_running()

        if concept_seed:
            log.info(f"Concept seed: {concept_seed}")

        concept = conceive(config, concept_seed=concept_seed)
        state.set_concept(concept["concept"])
        log.info(f"Concept: {concept['title']} — {concept['concept']}")

        # =================================================================
        # Stage 2: WRITE
        # =================================================================
        state.transition(State.WRITING)
        story_path = write_story(concept, story_number, config)
        state.set_story_file(str(story_path))
        log.info(f"Story written: {story_path}")

        # =================================================================
        # Stage 3: GATE (deterministic structural checks)
        # =================================================================
        state.transition(State.GATING)
        story = parse_story(story_path)
        passed, failures = gate_check(story, config)

        if not passed:
            for attempt in range(2):
                log.warning(f"Gate failed (attempt {attempt + 1}), rewriting story...")
                state.transition(State.WRITING)
                story_path = write_story(concept, story_number, config)
                state.set_story_file(str(story_path))
                state.transition(State.GATING)
                story = parse_story(story_path)
                passed, failures = gate_check(story, config)
                if passed:
                    break

            if not passed:
                state.fail(f"Gate check failed after 3 attempts: {failures[:3]}")
                _notify_failure(story_number, failures, config)
                return

        # =================================================================
        # Stage 4: VALIDATE SCENES (deterministic + LLM semantic)
        # Repair layer: rewrite STORY
        # =================================================================
        state.transition(State.VALIDATING_SCENES)
        log.info("Validating scene structure...")

        scene_result = validate_scenes(story, config)

        if scene_result.state == ValidationState.FAIL:
            log.warning(f"Scene validation FAILED: {scene_result.issues}")
            # Retry: rewrite story (scene-level repair)
            for attempt in range(2):
                log.info(f"Rewriting story for scene issues (attempt {attempt + 1})...")
                state.transition(State.WRITING)
                story_path = write_story(concept, story_number, config)
                state.set_story_file(str(story_path))
                state.transition(State.GATING)
                story = parse_story(story_path)
                passed, _ = gate_check(story, config)
                if not passed:
                    continue
                state.transition(State.VALIDATING_SCENES)
                scene_result = validate_scenes(story, config)
                if scene_result.state != ValidationState.FAIL:
                    break

            if scene_result.state == ValidationState.FAIL:
                log.warning("Scene validation still failing — proceeding with best effort")
                # Don't halt — proceed with what we have

        log.info(f"Scene validation: {scene_result.state.value}")

        # =================================================================
        # Stage 5: VALIDATE PROMPTS (deterministic + LLM semantic)
        # Repair layer: rewrite PROMPT only
        # =================================================================
        state.transition(State.VALIDATING_PROMPTS)
        log.info("Validating image prompts...")

        # Store original prompts BEFORE any rewriting
        original_prompts = {s.number: s.image_prompt for s in story.scenes}

        prompt_results = validate_all_prompts(story, config)

        # Rewrite failing prompts (prompt-level repair, not story-level)
        failed_prompts = [
            r for r in prompt_results if r.state == ValidationState.FAIL
        ]
        if failed_prompts:
            log.info(f"Rewriting {len(failed_prompts)} failing prompts...")
            for pr in failed_prompts:
                scene = next(
                    (s for s in story.scenes if s.number == pr.scene_num), None
                )
                if not scene:
                    continue

                if pr.suggested_fix:
                    # Use the validator's suggested fix
                    log.info(
                        f"Scene {scene.number}: Applying suggested fix: "
                        f"{pr.suggested_fix[:80]}..."
                    )
                    scene.image_prompt = pr.suggested_fix
                else:
                    # Ask LLM for a rewrite
                    log.info(f"Scene {scene.number}: LLM rewriting prompt...")
                    rewritten = rewrite_prompt(
                        scene, None, None, config,
                        original_prompt=original_prompts[scene.number],
                    )
                    scene.image_prompt = rewritten

            # Re-validate rewritten prompts
            prompt_results = validate_all_prompts(story, config)
            still_failing = sum(
                1 for r in prompt_results if r.state == ValidationState.FAIL
            )
            if still_failing:
                log.warning(
                    f"{still_failing} prompts still failing after rewrite — "
                    f"proceeding with best effort"
                )

        if story_only:
            log.info("--story-only: Stopping after prompt validation")
            return

        # =================================================================
        # VRAM transition: Stop LLM before image gen
        # =================================================================
        log.info("Stopping Ollama for image generation...")
        llm.stop_all()
        time.sleep(2)

        # =================================================================
        # Stage 6: GENERATE IMAGES
        # =================================================================
        state.transition(State.GENERATING_IMAGES)
        log.info("Starting ComfyUI for image generation (Qwen-Image-2512)...")

        comfyui.clean_output("scene_")

        if not comfyui.start(mode="image"):
            state.fail("Failed to start ComfyUI for image generation")
            return

        images = generate_scene_images(story, output_dir, config)
        comfyui.stop()

        if not images:
            state.fail("No images were generated")
            _notify_failure(story_number, ["No images generated"], config)
            return

        # =================================================================
        # Stage 7: VALIDATE IMAGES (deterministic + LLM vision)
        # Only FAIL triggers challenger. VALIDATOR_ERROR keeps image.
        # =================================================================
        state.transition(State.VALIDATING_IMAGES)
        log.info("Validating generated images...")

        llm.ensure_running()
        image_results = validate_all_images(
            images, story, config,
            original_prompts=original_prompts,
        )

        # Save validation results
        _save_validation(output_dir, image_results)

        # Setup version history (backup originals as v1)
        histories = setup_initial_history(images, original_prompts, image_results)

        # Identify scenes that need challengers (FAIL only, not ERROR/INCONCLUSIVE)
        failed_images = [
            r for r in image_results if r.state == ValidationState.FAIL
        ]
        non_fail = [
            r for r in image_results if r.state != ValidationState.FAIL
        ]

        log.info(
            f"Image validation: {len(non_fail)} accepted, "
            f"{len(failed_images)} need challengers"
        )

        # =================================================================
        # Stage 8: CHALLENGE IMAGES (rewrite + generate + pairwise judge)
        # Only runs for FAIL scenes. Never touches PASS/ERROR/INCONCLUSIVE.
        # =================================================================
        if failed_images:
            state.transition(State.CHALLENGING_IMAGES)
            log.info(f"Challenging {len(failed_images)} failed scenes...")

            # run_challenges manages its own VRAM lifecycle internally:
            # Phase 1: LLM rewrite (Ollama), Phase 2: generate (ComfyUI),
            # Phase 3: validate+compare (Ollama). Never runs both simultaneously.
            histories = run_challenges(
                failed_scenes=failed_images,
                story=story,
                output_dir=output_dir,
                config=config,
                original_prompts=original_prompts,
                histories=histories,
            )
        else:
            log.info("All images accepted — skipping challenger phase")

        llm.stop_all()
        time.sleep(2)

        # =================================================================
        # Stage 9: GENERATE VIDEOS
        # =================================================================
        state.transition(State.GENERATING_VIDEOS)
        log.info("Starting video generation (LTX-2.3 22B distilled)...")

        comfyui.clean_output("scene_")

        if not comfyui.start(mode="video"):
            state.fail("Failed to start ComfyUI for video generation")
            return

        videos = generate_scene_videos(story, output_dir, config)
        comfyui.stop()

        if not videos:
            state.fail("No videos were generated")
            _notify_failure(story_number, ["No videos generated"], config)
            return

        # =================================================================
        # Stage 10: COMBINE + TRIM
        # =================================================================
        state.transition(State.COMBINING)
        combined = combine_videos(videos, story, output_dir)

        if not combined.exists():
            state.fail("Failed to create combined.mp4")
            return

        # =================================================================
        # Stage 11: COMPLETED
        # =================================================================
        state.transition(State.COMPLETED)

        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        log.info(f"{'='*60}")
        log.info(f"  Story #{story_number} COMPLETED in {elapsed_min:.1f} minutes")
        log.info(f"  Output: {combined}")
        log.info(f"{'='*60}")

        _notify_success(story_number, story, combined, elapsed_min, config)

    except Exception as e:
        log.exception(f"Pipeline error: {e}")
        state.fail(str(e))
        _notify_failure(story_number, [str(e)], config)


# =============================================================================
# RESUME
# =============================================================================


def run_resume():
    """Resume pipeline from last saved state."""
    config = load_config()
    state = PipelineState(BASE_DIR / "stories")

    if state.state == State.IDLE:
        log.info("No story in progress. Use 'new' to start one.")
        return
    if state.state == State.COMPLETED:
        log.info(f"Story #{state.story_number} already completed.")
        return
    if state.state == State.FAILED:
        log.info(f"Story #{state.story_number} failed: {state._data.get('error')}")
        return

    story_number = state.story_number
    setup_logging(story_number)
    log.info(f"Resuming story #{story_number} from state: {state.state.value}")

    output_dir = BASE_DIR / "stories" / "output" / str(story_number)

    story_file = state.story_file
    if not story_file or not Path(story_file).exists():
        log.error(f"Story file not found: {story_file}")
        return

    story = parse_story(story_file)
    current = state.state

    # For early stages, re-run from scratch
    if current in (
        State.CONCEIVING, State.WRITING, State.GATING,
        State.VALIDATING_SCENES, State.VALIDATING_PROMPTS,
    ):
        concept_seed = state._data.get("concept")
        run_new(concept_seed=concept_seed)
        return

    # For image stages, collect existing images and continue
    if current in (State.GENERATING_IMAGES, State.VALIDATING_IMAGES, State.CHALLENGING_IMAGES):
        images_dir = output_dir / "images"
        images = sorted(images_dir.glob("scene_[0-9][0-9].png"))
        if not images:
            log.error("No images found to resume from")
            return

        original_prompts = {s.number: s.image_prompt for s in story.scenes}

        if current == State.GENERATING_IMAGES:
            if not comfyui.start(mode="image"):
                state.fail("Failed to start ComfyUI")
                return
            images = generate_scene_images(story, output_dir, config)
            comfyui.stop()

        # Continue with validation + challenge + video gen
        state.transition(State.VALIDATING_IMAGES)
        llm.ensure_running()
        image_results = validate_all_images(images, story, config, original_prompts=original_prompts)
        histories = setup_initial_history(images, original_prompts, image_results)

        failed = [r for r in image_results if r.state == ValidationState.FAIL]
        if failed:
            state.transition(State.CHALLENGING_IMAGES)
            if not comfyui.start(mode="image"):
                state.fail("ComfyUI failed")
                return
            histories = run_challenges(failed, story, output_dir, config, original_prompts, histories)
            comfyui.stop()

        llm.stop_all()
        time.sleep(2)

        state.transition(State.GENERATING_VIDEOS)
        comfyui.clean_output("scene_")
        if not comfyui.start(mode="video"):
            state.fail("ComfyUI failed")
            return
        videos = generate_scene_videos(story, output_dir, config)
        comfyui.stop()
        _finish_combine(state, story, videos, output_dir, config)
        return

    if current == State.GENERATING_VIDEOS:
        if not comfyui.start(mode="video"):
            state.fail("ComfyUI failed")
            return
        videos = generate_scene_videos(story, output_dir, config)
        comfyui.stop()
        _finish_combine(state, story, videos, output_dir, config)
        return

    if current == State.COMBINING:
        videos_dir = output_dir / "videos"
        videos = sorted(videos_dir.glob("scene_*.mp4"))
        _finish_combine(state, story, videos, output_dir, config)
        return

    log.warning(f"Don't know how to resume from: {current.value}")


def _finish_combine(state, story, videos, output_dir, config):
    state.transition(State.COMBINING)
    combined = combine_videos(videos, story, output_dir)
    if combined.exists():
        state.transition(State.COMPLETED)
        log.info(f"Story #{state.story_number} completed: {combined}")
        _notify_success(state.story_number, story, combined, 0, config)
    else:
        state.fail("combined.mp4 not created")


# =============================================================================
# HELPERS
# =============================================================================


def _save_validation(output_dir: Path, results: list):
    """Save validation results to validation.json."""
    val_file = output_dir / "validation.json"
    data = []
    for r in results:
        entry = {
            "scene": r.scene_num,
            "state": r.state.value,
            "score": r.score,
            "findings": r.findings,
            "confidence": r.confidence,
        }
        data.append(entry)
    with open(val_file, "w") as f:
        json.dump(data, f, indent=2)


def show_status():
    config = load_config()
    state = PipelineState(BASE_DIR / "stories")
    print(f"Pipeline State: {state.state.value}")
    print(f"Story Number:   {state.story_number or 'None'}")
    print(f"Story File:     {state.story_file or 'None'}")
    print(f"Output Dir:     {state.output_dir or 'None'}")
    if state.current_stage:
        print(f"Current Stage:  {state.current_stage}")
    if state._data.get("error"):
        print(f"Error:          {state._data['error']}")
    history = state._data.get("history", [])
    if history:
        print(f"\nRecent History:")
        for h in history[-5:]:
            print(f"  {h['at']}: {h['state']}")


def _notify_success(story_number, story, combined_path, elapsed_min, config):
    if not config.get("output", {}).get("notify_telegram"):
        return
    size_mb = combined_path.stat().st_size / (1024 * 1024) if combined_path.exists() else 0
    caption = f"Story #{story_number}: {story.title}\n{len(story.scenes)} scenes | {size_mb:.1f}MB"
    if elapsed_min > 0:
        caption += f" | {elapsed_min:.0f}min"
    send_video(combined_path, caption, config)
    if config.get("output", {}).get("upload_nextcloud"):
        upload_to_nextcloud(combined_path, config)


def _notify_failure(story_number, failures, config):
    if not config.get("output", {}).get("notify_telegram"):
        return
    text = f"Story #{story_number} FAILED:\n" + "\n".join(f"- {f}" for f in failures[:5])
    send_message(text, config)


# =============================================================================
# CLI
# =============================================================================


def cli():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Video Pipeline v3 — Story to video generation",
    )
    sub = parser.add_subparsers(dest="command")

    new_cmd = sub.add_parser("new", help="Create and run a new story")
    new_cmd.add_argument("concept", nargs="?", help="Optional concept seed")
    new_cmd.add_argument("--story-only", action="store_true", help="Stop after prompt validation")

    sub.add_parser("resume", help="Resume from where we left off")
    sub.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()

    if args.command == "new":
        run_new(concept_seed=args.concept, story_only=args.story_only)
    elif args.command == "resume":
        run_resume()
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
