"""
ComfyUI API wrapper for the video pipeline.

Manages ComfyUI process lifecycle (start/stop/restart), workflow submission,
and output retrieval. Supports two modes:
  - 'image': Qwen-Image-2512 (--gpu-only --cache-none)
  - 'video': LTX-2.3 22B (--normalvram --dont-upcast-attention --disable-pinned-memory)
"""

import glob
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

COMFYUI_DIR = Path("/home/bbnlabs5/video_gen_web/comfyui")
COMFYUI_PORT = 8199
COMFYUI_URL = f"http://localhost:{COMFYUI_PORT}"

# Common flags for all modes (Blackwell RTX 5090 + 16GB RAM)
COMMON_FLAGS = "--dont-upcast-attention --disable-pinned-memory --preview-method none"

# Mode-specific flags (ComfyUI v0.16.4)
MODE_FLAGS = {
    # --normalvram for BOTH modes. With 32GB VRAM + 31GB swap, ComfyUI's
    # standard memory management works. The 21GB GGUF model loads to VRAM
    # and stays cached between generations (6s reload vs 45s cold load).
    # DO NOT use --gpu-only --cache-none: causes mid-generation model eviction.
    "image": "--normalvram",
    "video": "--normalvram",
}


def is_running() -> bool:
    """Check if ComfyUI is responding."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start(mode: str = "video", timeout: int = 90) -> bool:
    """Start ComfyUI with mode-specific flags.

    Args:
        mode: 'image' for Qwen-Image-2512, 'video' for LTX-2.3
        timeout: Seconds to wait for ComfyUI to become ready
    """
    if is_running():
        log.info(f"ComfyUI already running on port {COMFYUI_PORT}")
        return True

    venv_path = COMFYUI_DIR / "venv"
    if not venv_path.exists():
        log.error(f"ComfyUI venv not found: {venv_path}")
        return False

    flags = MODE_FLAGS.get(mode, MODE_FLAGS["video"])
    log_file = Path("/home/bbnlabs5/video-pipeline/stories/comfyui.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"cd {COMFYUI_DIR} && source venv/bin/activate && "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"python main.py --listen 0.0.0.0 --port {COMFYUI_PORT} "
        f"{COMMON_FLAGS} {flags} >> {log_file} 2>&1"
    )

    log.info(f"Starting ComfyUI (mode={mode})...")
    subprocess.Popen(
        ["bash", "-c", cmd],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return _wait_ready(timeout)


def stop() -> bool:
    """Stop ComfyUI with 3-step graceful shutdown.

    Root cause of model eviction bug: pkill -9 (SIGKILL) kills the process
    instantly without allowing PyTorch to release CUDA context. The GPU driver
    reports 0 MB via nvidia-smi but CUDA internal state isn't fully cleaned up.
    The next ComfyUI process gets partial VRAM allocations, causing the 21GB
    GGUF model to load incompletely → garbage output.

    Fix: unload models via /free API first, then SIGTERM for graceful Python
    shutdown, then SIGKILL only as last resort.
    """
    log.info("Stopping ComfyUI...")

    # Step 1: Unload models via ComfyUI /free API (cleanest model lifecycle)
    if is_running():
        try:
            requests.post(
                f"{COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10,
            )
            log.info("Models unloaded via /free API")
            time.sleep(2)
        except Exception:
            pass  # Process might already be dying

    # Step 2: SIGTERM for graceful Python/PyTorch shutdown
    subprocess.run(["pkill", "-15", "-f", "python.*main.py.*--port"], capture_output=True)

    # Wait for graceful exit (up to 5s)
    for _ in range(5):
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py.*--port"],
            capture_output=True,
        )
        if result.returncode != 0:
            break  # Process exited
        time.sleep(1)

    # Step 3: SIGKILL only if still running (last resort)
    result = subprocess.run(
        ["pgrep", "-f", "python.*main.py.*--port"],
        capture_output=True,
    )
    if result.returncode == 0:
        log.info("Process still running after SIGTERM, sending SIGKILL...")
        subprocess.run(["pkill", "-9", "-f", "python.*main.py.*--port"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "comfyui"], capture_output=True)
        time.sleep(3)

    # Step 4: Wait for VRAM to fully clear
    for _ in range(15):
        vram = _get_vram_used()
        if vram is not None and vram < 500:
            log.info(f"GPU memory cleared ({vram} MB)")
            return True
        time.sleep(1)

    log.warning("GPU memory not fully cleared, continuing...")
    return True


def restart(mode: str = "video") -> bool:
    """Restart ComfyUI (stop then start). Used between scenes for VRAM cleanup."""
    stop()
    time.sleep(3)  # Extra wait for CUDA context to fully release
    return start(mode)


def free_memory() -> bool:
    """Unload all models via ComfyUI /free API (faster than restart)."""
    try:
        resp = requests.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        if resp.status_code == 200:
            log.info("Models unloaded, memory freed")
            return True
    except Exception as e:
        log.warning(f"/free error: {e}")
    return False


def submit_workflow(workflow: dict) -> Optional[str]:
    """Submit a workflow JSON to ComfyUI. Returns prompt_id or None."""
    try:
        resp = requests.post(
            f"{COMFYUI_URL}/prompt",
            json=workflow,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            log.error(f"Submit failed: HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        result = resp.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            log.error(f"No prompt_id in response: {result}")
            return None

        log.info(f"Submitted: {prompt_id[:8]}...")
        return prompt_id
    except Exception as e:
        log.error(f"Submit error: {e}")
        return None


def wait_for_completion(
    prompt_id: str,
    timeout: int = 600,
    vram_floor: int = 0,
) -> tuple[bool, bool]:
    """Wait for a workflow to finish.

    Args:
        prompt_id: ComfyUI prompt ID
        timeout: Max wait time in seconds
        vram_floor: Minimum expected VRAM during generation (MB).
                    If VRAM drops below this after the first 20s, the output
                    is flagged as potentially corrupt (model eviction).
                    Set to 0 to disable the check.

    Returns:
        (success, vram_ok) — success=True if workflow completed,
        vram_ok=True if VRAM stayed above floor (or floor=0).
    """
    start_time = time.time()
    last_log = start_time
    vram_ok = True
    vram_check_start = 20  # Don't check VRAM during initial model loading

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        if time.time() - last_log >= 15:
            vram = _get_vram_used()
            log.info(f"  {elapsed:.0f}s elapsed (VRAM: {vram} MB)")
            last_log = time.time()

            # VRAM sanity check: if VRAM drops below floor after model
            # should be loaded, the model was evicted mid-generation
            if vram_floor > 0 and elapsed > vram_check_start and vram is not None:
                if vram < vram_floor:
                    log.warning(
                        f"  VRAM dropped to {vram} MB (floor={vram_floor} MB) — "
                        f"model may have been evicted, output likely corrupt"
                    )
                    vram_ok = False

        try:
            resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    status = data[prompt_id].get("status", {})
                    if status.get("completed", False):
                        log.info(f"Completed in {elapsed:.1f}s")
                        return True, vram_ok
                    if status.get("status_str") == "error":
                        log.error(f"Workflow error: {data[prompt_id]}")
                        return False, vram_ok
        except requests.exceptions.ConnectionError:
            log.warning(f"Connection lost at {elapsed:.0f}s")
        except Exception as e:
            log.warning(f"Status check error: {e}")

        time.sleep(5)

    log.error(f"Timeout after {timeout}s")
    return False, vram_ok


def get_output(prefix: str, ext: str = "png") -> Optional[Path]:
    """Get the most recent output file matching a prefix and extension."""
    output_dir = COMFYUI_DIR / "output"
    pattern = str(output_dir / f"{prefix}*.{ext}")
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(matches[0]) if matches else None


def clean_output(prefix: str = "scene_"):
    """Delete old output files matching prefix to prevent stale file pickup.

    Should be called before each image/video generation phase.
    """
    output_dir = COMFYUI_DIR / "output"
    if not output_dir.exists():
        return

    removed = 0
    for ext in ("png", "mp4"):
        for f in output_dir.glob(f"{prefix}*.{ext}"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass

    if removed:
        log.info(f"Cleaned {removed} old output files matching '{prefix}*'")


def copy_to_input(source_path: Path, name: str) -> Path:
    """Copy a file to ComfyUI's input directory. Returns the destination path."""
    input_dir = COMFYUI_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / name
    shutil.copy(str(source_path), str(dest))
    return dest


def _wait_ready(timeout: int) -> bool:
    """Wait for ComfyUI to become ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 10 == 0:
            log.info(f"  Waiting... {elapsed}s")
        if is_running():
            log.info(f"ComfyUI ready after {elapsed}s")
            return True
        time.sleep(1)
    log.error(f"ComfyUI timeout after {timeout}s")
    return False


def _get_vram_used() -> Optional[int]:
    """Get current VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip().split()[0])
    except Exception:
        return None
