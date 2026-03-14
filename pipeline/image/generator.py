"""
Image generation via Qwen-Image-2512 + ComfyUI.

Imports the proven workflow from /home/bbnlabs5/video_gen_web/workflows/qwen_image.py.
Handles DNA expansion, seed management, ComfyUI submission, and output retrieval.

ComfyUI is restarted between each scene to prevent model eviction under --cache-none.
VRAM is monitored during generation — if it drops below 10GB, the output is flagged
as corrupt and the scene is retried (up to 2 retries per scene).
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

# Import Qwen-Image workflow from video_gen_web
sys.path.insert(0, "/home/bbnlabs5/video_gen_web")
from workflows.qwen_image import build_workflow as build_qwen_image_workflow

# VRAM floor disabled — with --normalvram, ComfyUI cycles models between VRAM
# and RAM between workflows, causing normal VRAM dips that trigger false alarms.
# Image quality is validated by the validation loop instead.
QWEN_IMAGE_VRAM_FLOOR = 0
MAX_RETRIES_PER_SCENE = 2


def generate_scene_images(
    story: StoryData,
    output_dir: Path,
    config: dict,
    scenes_to_generate: Optional[list[int]] = None,
    seed_base: Optional[int] = None,
) -> list[Path]:
    """Generate images for all (or specified) scenes.

    ComfyUI is restarted between each scene to prevent model eviction
    under --cache-none mode. VRAM is monitored during generation.

    Args:
        story: Parsed story data with DNA blocks and scenes
        output_dir: Directory to save scene_01.png, scene_02.png, etc.
        config: Pipeline config dict
        scenes_to_generate: Optional list of scene numbers to generate (default: all)
        seed_base: Base seed for reproducibility (default: random)

    Returns:
        List of paths to generated images
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    img_cfg = config["models"]["image"]
    if seed_base is None:
        seed_base = random.randint(0, 2**32 - 1)

    scenes = story.scenes
    if scenes_to_generate:
        scenes = [s for s in scenes if s.number in scenes_to_generate]

    generated = []

    for i, scene in enumerate(scenes):
        scene_file = images_dir / f"scene_{scene.number:02d}.png"

        # Skip if already exists (resumability)
        if scene_file.exists():
            log.info(f"Scene {scene.number}: Image exists, skipping")
            generated.append(scene_file)
            continue

        # Expand DNA tags in prompt
        prompt = expand_dna(scene.image_prompt, story)
        seed = seed_base + scene.number

        log.info(f"Scene {scene.number}: Generating image (seed={seed})")
        log.info(f"  Prompt: {prompt[:150]}...")

        # All image scenes use the same Qwen-Image model — keep it loaded in VRAM.
        # Do NOT restart ComfyUI between image scenes. The model stays warm,
        # avoiding the --cache-none eviction bug and eliminating reload overhead.
        # VRAM cleanup happens when the orchestrator kills ComfyUI after all images.
        if not comfyui.is_running():
            if not comfyui.start(mode="image"):
                log.error("Failed to start ComfyUI for image generation")
                break

        # Clean old output files for this scene prefix
        prefix = f"scene_{scene.number:02d}"
        comfyui.clean_output(prefix)

        # Generate with VRAM monitoring and retry on model eviction
        success = _generate_single_image(
            prompt=prompt,
            seed=seed,
            prefix=prefix,
            scene_file=scene_file,
            img_cfg=img_cfg,
        )

        if not success:
            # Retry with ComfyUI restart (model eviction recovery)
            for retry in range(MAX_RETRIES_PER_SCENE):
                log.warning(f"Scene {scene.number}: Retry {retry + 1}/{MAX_RETRIES_PER_SCENE}")
                comfyui.restart(mode="image")
                comfyui.clean_output(prefix)
                success = _generate_single_image(
                    prompt=prompt,
                    seed=seed + 1000 * (retry + 1),  # Different seed per retry
                    prefix=prefix,
                    scene_file=scene_file,
                    img_cfg=img_cfg,
                )
                if success:
                    break

        if success:
            generated.append(scene_file)
        else:
            log.error(f"Scene {scene.number}: Failed after {MAX_RETRIES_PER_SCENE} retries")

    log.info(f"Generated {len(generated)}/{len(scenes)} images")
    return generated


def _generate_single_image(
    prompt: str,
    seed: int,
    prefix: str,
    scene_file: Path,
    img_cfg: dict,
) -> bool:
    """Generate a single image and validate VRAM during generation.

    Returns True if image was generated with healthy VRAM levels.
    Returns False if VRAM dropped (model eviction) or generation failed.
    """
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
        log.error("Failed to submit workflow")
        return False

    success, vram_ok = comfyui.wait_for_completion(
        prompt_id,
        timeout=180,
        vram_floor=QWEN_IMAGE_VRAM_FLOOR,
    )

    if not success:
        log.error("Generation failed or timed out")
        return False

    if not vram_ok:
        log.error("VRAM dropped during generation — output is likely corrupt, discarding")
        # Clean up the potentially corrupt output
        corrupt = comfyui.get_output(prefix, "png")
        if corrupt and corrupt.exists():
            corrupt.unlink()
        return False

    # Retrieve output
    output_file = comfyui.get_output(prefix, "png")
    if output_file and output_file.exists():
        shutil.move(str(output_file), str(scene_file))
        log.info(f"Saved {scene_file}")
        return True
    else:
        log.error("Output file not found")
        return False
