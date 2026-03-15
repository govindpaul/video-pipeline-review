"""
Challenger system — generates competitor images and promotes only if better.

Key invariants:
- Original image (v1) is NEVER deleted
- Challengers are written to separate versioned files
- Promotion uses pairwise comparison, not just scores
- TIE → keep baseline (incumbent advantage)
- scene_XX.png is always the currently selected version (safe copy via temp + os.replace)
- Full version history tracked in image_history.json
"""

import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from pipeline.image.rewriter import rewrite_prompt
from pipeline.story.parser import StoryData, Scene, expand_dna
from pipeline.utils import comfyui
from pipeline.validators.image import validate_image
from pipeline.validators.pairwise import compare_images
from pipeline.validators.schema import (
    ImageValidation,
    ImageVersion,
    PairwiseDecision,
    PromotionLog,
    SceneImageHistory,
    ValidationState,
)

log = logging.getLogger(__name__)

MAX_CHALLENGE_ROUNDS = 2


def run_challenges(
    failed_scenes: list[ImageValidation],
    story: StoryData,
    output_dir: Path,
    config: dict,
    original_prompts: dict[int, str],
    histories: dict[int, SceneImageHistory],
) -> dict[int, SceneImageHistory]:
    """Run challenger generation for all FAIL scenes.

    VRAM-aware batched flow (cannot run LLM + ComfyUI simultaneously):
      Phase 1: Rewrite all prompts (LLM — Ollama)
      Phase 2: Generate all challengers (ComfyUI)
      Phase 3: Validate challengers + pairwise compare (LLM — Ollama)
      Phase 4: Promote winners

    Max 2 challenge rounds per scene.

    Args:
        failed_scenes: List of ImageValidation results with state=FAIL
        story: Parsed story data
        output_dir: Story output directory
        config: Pipeline config
        original_prompts: Original prompts from story file
        histories: Current version histories per scene

    Returns:
        Updated histories dict
    """
    from pipeline.llm import local as llm

    images_dir = output_dir / "images"
    promotion_log = []

    for round_num in range(1, MAX_CHALLENGE_ROUNDS + 1):
        if not failed_scenes:
            break

        log.info(f"Challenge round {round_num}/{MAX_CHALLENGE_ROUNDS}: {len(failed_scenes)} scenes")

        # === Phase 1: Rewrite prompts (LLM) ===
        log.info("Phase 1: Rewriting prompts...")
        llm.ensure_running()

        rewrite_tasks = []  # (scene, result, new_prompt, version_num, challenger_path)
        for result in failed_scenes:
            scene = next((s for s in story.scenes if s.number == result.scene_num), None)
            if not scene:
                continue

            scene_num = scene.number
            history = histories.get(scene_num, SceneImageHistory(scene_num=scene_num))
            new_version_num = len(history.versions) + 1

            original_prompt = original_prompts[scene_num]
            new_prompt = rewrite_prompt(
                scene, result,
                images_dir / f"scene_{scene_num:02d}.png",
                config,
                original_prompt=original_prompt,
            )

            # Two-stage revalidation before using rewritten prompt:
            # 1. Deterministic (format) 2. Semantic (LLM)
            from pipeline.validators.deterministic import check_prompt_deterministic
            from pipeline.validators.prompt import validate_prompt as _validate_prompt
            from pipeline.story.parser import Scene as _Scene
            test_scene = _Scene(
                number=scene_num, title=scene.title,
                image_prompt=new_prompt, video_prompt=scene.video_prompt,
            )
            det_state, det_issues = check_prompt_deterministic(test_scene, story)
            if det_state != ValidationState.PASS:
                log.warning(
                    f"Scene {scene_num}: Rewritten prompt failed deterministic: "
                    f"{det_issues[:2]}. Using original."
                )
                new_prompt = original_prompt
            else:
                sem_result = _validate_prompt(test_scene, story, config)
                if sem_result.state != ValidationState.PASS:
                    log.warning(
                        f"Scene {scene_num}: Rewritten prompt rejected "
                        f"(semantic {sem_result.state.value}: "
                        f"{sem_result.issues[:2] if sem_result.issues else 'no details'}). "
                        f"Using original."
                    )
                    new_prompt = original_prompt
                else:
                    log.info(
                        f"Scene {scene_num}: Rewritten prompt accepted "
                        f"(deterministic PASS + semantic PASS)"
                    )

            challenger_filename = f"scene_{scene_num:02d}_v{new_version_num}.png"
            challenger_path = images_dir / challenger_filename

            rewrite_tasks.append((scene, result, new_prompt, new_version_num, challenger_path))

        llm.stop_all()
        import time
        time.sleep(2)

        # === Phase 2: Generate challengers (ComfyUI) ===
        log.info(f"Phase 2: Generating {len(rewrite_tasks)} challengers...")
        if not comfyui.is_running():
            if not comfyui.start(mode="image"):
                log.error("Failed to start ComfyUI for challenger generation")
                break

        gen_results = []  # (task, success)
        for scene, result, new_prompt, version_num, challenger_path in rewrite_tasks:
            prefix = f"scene_{scene.number:02d}_v{version_num}"
            comfyui.clean_output(prefix)

            log.info(f"Scene {scene.number}: Generating challenger v{version_num}")

            success = _generate_challenger(
                prompt=expand_dna(new_prompt, story),
                seed=random.randint(0, 2**32 - 1),
                prefix=prefix,
                output_path=challenger_path,
                config=config,
            )
            gen_results.append((scene, result, new_prompt, version_num, challenger_path, success))

        comfyui.stop()
        time.sleep(2)

        # === Phase 3: Validate + pairwise compare (LLM) ===
        log.info("Phase 3: Validating challengers + pairwise comparison...")
        llm.ensure_running()

        still_failed = []
        for scene, result, new_prompt, version_num, challenger_path, gen_ok in gen_results:
            scene_num = scene.number
            history = histories.get(scene_num, SceneImageHistory(scene_num=scene_num))

            if not gen_ok:
                log.error(f"Scene {scene_num}: Challenger generation failed, keeping baseline")
                still_failed.append(result)
                continue

            # Validate challenger
            original_prompt = original_prompts[scene_num]
            challenger_val = validate_image(
                challenger_path, scene, story, config,
                original_prompt=original_prompt,
            )

            # Record version
            challenger_version = ImageVersion(
                version=version_num,
                filename=challenger_path.name,
                prompt_used=new_prompt,
                validation=challenger_val,
            )

            # Pairwise comparison
            baseline_path = images_dir / f"scene_{scene_num:02d}.png"
            decision = compare_images(
                baseline_path=baseline_path,
                challenger_path=challenger_path,
                scene=scene, story=story, config=config,
                original_prompt=original_prompt,
            )
            challenger_version.comparison = decision
            history.add_version(challenger_version)

            # Promotion decision
            promoted = decision.winner == "CHALLENGER"

            promo = PromotionLog(
                scene_num=scene_num,
                round=round_num,
                baseline_version=history.selected_version,
                challenger_version=version_num,
                baseline_state=result.state,
                baseline_score=result.score,
                challenger_state=challenger_val.state,
                challenger_score=challenger_val.score,
                pairwise=decision,
                promoted=promoted,
                reason=(
                    f"Pairwise: {decision.winner} "
                    f"(baseline={result.score}, challenger={challenger_val.score}, "
                    f"reason: {decision.reason[:80]})"
                ),
            )
            promotion_log.append(promo)

            if promoted:
                _safe_promote(challenger_path, baseline_path)
                history.promote(version_num)
                log.info(
                    f"Scene {scene_num}: PROMOTED v{version_num} "
                    f"(score {result.score} -> {challenger_val.score})"
                )
            else:
                log.info(
                    f"Scene {scene_num}: Keeping baseline "
                    f"(pairwise: {decision.winner}, reason: {decision.reason[:60]})"
                )
                # Still failed — eligible for next round
                if challenger_val.state == ValidationState.FAIL:
                    still_failed.append(result)

            histories[scene_num] = history

        llm.stop_all()
        time.sleep(2)

        # Next round only challenges scenes that are still FAIL and weren't promoted
        failed_scenes = still_failed

    # Save logs
    _save_promotion_log(output_dir, promotion_log)
    _save_histories(output_dir, histories)

    return histories


def setup_initial_history(
    images: list[Path],
    original_prompts: dict[int, str],
    validations: list[ImageValidation],
) -> dict[int, SceneImageHistory]:
    """Create initial version history from first-generation images.

    Copies each scene_XX.png → scene_XX_v1.png (backup, never deleted).
    """
    histories = {}
    val_map = {v.scene_num: v for v in validations}

    for img in images:
        try:
            scene_num = int(img.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        images_dir = img.parent

        # Backup: copy to v1 (never deleted)
        v1_path = images_dir / f"scene_{scene_num:02d}_v1.png"
        if not v1_path.exists():
            shutil.copy2(str(img), str(v1_path))

        history = SceneImageHistory(scene_num=scene_num, selected_version=1)
        history.add_version(ImageVersion(
            version=1,
            filename=f"scene_{scene_num:02d}_v1.png",
            prompt_used=original_prompts.get(scene_num, ""),
            validation=val_map.get(scene_num),
        ))

        histories[scene_num] = history

    return histories


def _generate_challenger(
    prompt: str,
    seed: int,
    prefix: str,
    output_path: Path,
    config: dict,
) -> bool:
    """Generate a single challenger image."""
    import sys
    _comfyui_parent = str(Path(__file__).resolve().parent.parent.parent)
    try:
        import yaml as _yaml
        with open(Path(_comfyui_parent) / "config.yaml") as _f:
            _cfg = _yaml.safe_load(_f)
        _wf_dir = str(Path(_cfg["paths"]["comfyui_dir"]).parent)
    except Exception:
        _wf_dir = "/home/bbnlabs5/video_gen_web"
    sys.path.insert(0, _wf_dir)
    from workflows.qwen_image import build_workflow as build_qwen_image_workflow

    img_cfg = config["models"]["image"]
    comfyui.clean_output(prefix)

    workflow = build_qwen_image_workflow(
        prompt=prompt,
        seed=seed,
        filename_prefix=prefix,
        negative_prompt=img_cfg.get("negative_prompt"),
        width=img_cfg.get("width", 704),
        height=img_cfg.get("height", 1248),
        steps=img_cfg.get("steps", 8),
        cfg=img_cfg.get("cfg", 1.0),
        sampler=img_cfg.get("sampler", "euler"),
        scheduler=img_cfg.get("scheduler", "beta"),
        shift=img_cfg.get("shift", 3.0),
    )

    prompt_id = comfyui.submit_workflow(workflow)
    if not prompt_id:
        return False

    success, _ = comfyui.wait_for_completion(prompt_id, timeout=180)
    if not success:
        return False

    output_file = comfyui.get_output(prefix, "png")
    if output_file and output_file.exists():
        shutil.move(str(output_file), str(output_path))
        return True
    return False


def _safe_promote(source: Path, target: Path):
    """Atomically replace target with source using temp file + os.replace."""
    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(target.parent), suffix=".tmp"
    )
    os.close(temp_fd)
    try:
        shutil.copy2(str(source), temp_path)
        os.replace(temp_path, str(target))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _save_promotion_log(output_dir: Path, log_entries: list[PromotionLog]):
    """Append promotion decisions to promotion_log.json."""
    log_file = output_dir / "promotion_log.json"
    existing = []
    if log_file.exists():
        with open(log_file) as f:
            existing = json.load(f)

    for entry in log_entries:
        record = {
            "scene_num": entry.scene_num,
            "round": entry.round,
            "baseline_version": entry.baseline_version,
            "challenger_version": entry.challenger_version,
            "baseline_score": entry.baseline_score,
            "challenger_score": entry.challenger_score,
            "promoted": entry.promoted,
            "reason": entry.reason,
        }
        if entry.pairwise:
            record["pairwise_winner"] = entry.pairwise.winner
            record["pairwise_confidence"] = entry.pairwise.confidence
        existing.append(record)

    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2)


def _save_histories(output_dir: Path, histories: dict[int, SceneImageHistory]):
    """Save version histories to image_history.json."""
    hist_file = output_dir / "image_history.json"
    data = {}
    for scene_num, history in histories.items():
        key = f"scene_{scene_num:02d}"
        data[key] = {
            "selected_version": history.selected_version,
            "versions": [],
        }
        for v in history.versions:
            vdata = {
                "version": v.version,
                "filename": v.filename,
                "prompt_used": v.prompt_used[:200] + "..." if len(v.prompt_used) > 200 else v.prompt_used,
            }
            if v.validation:
                vdata["validation_state"] = v.validation.state.value
                vdata["validation_score"] = v.validation.score
            if v.comparison:
                vdata["comparison_winner"] = v.comparison.winner
                vdata["comparison_confidence"] = v.comparison.confidence
            data[key]["versions"].append(vdata)

    with open(hist_file, "w") as f:
        json.dump(data, f, indent=2)
