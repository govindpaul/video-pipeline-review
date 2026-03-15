"""
Video generation via LTX-2.3 22B distilled + ComfyUI.

Imports the proven workflow from tests/benchmarks/video_models/workflows/ltx23_i2v_twostage_v2.py.
Uses production config (Test 91): LoRA 0.5, i2v=1.0, 8 steps, 544x960→1088x1920, with audio.
ComfyUI is restarted between each scene for VRAM cleanup.
"""

import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

from pipeline.story.parser import StoryData, expand_dna
from pipeline.utils import comfyui

log = logging.getLogger(__name__)

# Import LTX-2.3 workflow from benchmarks
_base = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_base / "tests" / "benchmarks" / "video_models" / "workflows"))
from ltx23_i2v_twostage_v2 import build_workflow as build_ltx_workflow


def generate_scene_videos(
    story: StoryData,
    output_dir: Path,
    config: dict,
    seed_base: Optional[int] = None,
) -> list[Path]:
    """Generate videos for all scenes from their validated images.

    ComfyUI is restarted between each scene to prevent VRAM accumulation.

    Args:
        story: Parsed story data
        output_dir: Output directory (will create videos/ subdirectory)
        config: Pipeline config dict
        seed_base: Base seed for reproducibility

    Returns:
        List of paths to generated video files
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    vid_cfg = config["models"]["video"]
    if seed_base is None:
        seed_base = random.randint(0, 2**32 - 1)

    generated = []

    for i, scene in enumerate(story.scenes):
        video_file = videos_dir / f"scene_{scene.number:02d}.mp4"

        # Skip if already exists (resumability)
        if video_file.exists():
            log.info(f"Scene {scene.number}: Video exists, skipping")
            generated.append(video_file)
            continue

        image_file = images_dir / f"scene_{scene.number:02d}.png"
        if not image_file.exists():
            log.error(f"Scene {scene.number}: Source image not found: {image_file}")
            continue

        # Expand DNA in video prompt
        prompt = expand_dna(scene.video_prompt, story)
        seed = seed_base + scene.number

        log.info(f"Scene {scene.number}/{len(story.scenes)}: Generating video (seed={seed})")
        log.info(f"  Prompt: {prompt[:120]}...")

        # Restart ComfyUI between scenes for VRAM cleanup (except first)
        if i == 0:
            if not comfyui.is_running():
                if not comfyui.start(mode="video"):
                    log.error("Failed to start ComfyUI for video generation")
                    break
        else:
            if not comfyui.restart(mode="video"):
                log.error("Failed to restart ComfyUI between scenes")
                break

        # Copy image to ComfyUI input
        input_name = f"scene_{scene.number:02d}_input.png"
        comfyui.copy_to_input(image_file, input_name)

        # Build workflow with production config (Test 91)
        prefix = f"scene_{scene.number:02d}"
        workflow = build_ltx_workflow(
            source_image_filename=input_name,
            prompt=prompt,
            negative_prompt=vid_cfg.get("negative_prompt", ""),
            seed=seed,
            width=vid_cfg.get("width", 544),
            height=vid_cfg.get("height", 960),
            num_frames=vid_cfg.get("num_frames", 121),
            fps=vid_cfg.get("fps", 25),
            img_compression=vid_cfg.get("img_compression", 35),
            lora_strength_stage1=vid_cfg.get("lora_strength_stage1", 0.5),
            lora_strength_stage2=vid_cfg.get("lora_strength_stage2", 0.5),
            i2v_strength_stage1=vid_cfg.get("i2v_strength", 1.0),
            i2v_strength_stage2=vid_cfg.get("i2v_strength", 1.0),
            stage1_sampler=vid_cfg.get("stage1_sampler", "euler_ancestral_cfg_pp"),
            stage1_sigmas=vid_cfg.get("stage1_sigmas"),
            stage1_cfg=vid_cfg.get("stage1_cfg", 1.0),
            stage1_stg_scale=vid_cfg.get("stage1_stg_scale", 0.0),
            stage2_sigmas=vid_cfg.get("stage2_sigmas"),
            stage2_sampler=vid_cfg.get("stage2_sampler", "euler_cfg_pp"),
            stage2_cfg=vid_cfg.get("stage2_cfg", 1.0),
            filename_prefix=prefix,
        )

        prompt_id = comfyui.submit_workflow(workflow)
        if not prompt_id:
            log.error(f"Scene {scene.number}: Failed to submit workflow")
            continue

        success, _ = comfyui.wait_for_completion(
            prompt_id,
            timeout=config.get("comfyui", {}).get("workflow_timeout", 600),
        )
        if not success:
            log.error(f"Scene {scene.number}: Video generation failed or timed out")
            continue

        # Retrieve output video
        output_file = comfyui.get_output(prefix, "mp4")
        if output_file and output_file.exists():
            shutil.move(str(output_file), str(video_file))
            log.info(f"Scene {scene.number}: Saved {video_file}")
            generated.append(video_file)
        else:
            log.error(f"Scene {scene.number}: Output video not found")

        # Clean up input copy
        try:
            input_path = Path(comfyui.COMFYUI_DIR) / "input" / input_name
            if input_path.exists():
                input_path.unlink()
        except Exception:
            pass

    log.info(f"Generated {len(generated)}/{len(story.scenes)} videos")
    return generated
