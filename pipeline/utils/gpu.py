"""
GPU and VRAM management utilities.
"""

import logging
import subprocess
import time
from typing import Optional

log = logging.getLogger(__name__)


def get_vram_usage() -> tuple[Optional[int], Optional[int]]:
    """Get VRAM (used_mb, total_mb). Returns (None, None) on error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(",")
        used = int(parts[0].strip().split()[0])
        total = int(parts[1].strip().split()[0])
        return used, total
    except Exception:
        return None, None


def wait_vram_free(threshold_mb: int = 1000, timeout: int = 30) -> bool:
    """Wait until VRAM usage drops below threshold."""
    for _ in range(timeout):
        used, _ = get_vram_usage()
        if used is not None and used < threshold_mb:
            return True
        time.sleep(1)
    return False


def log_vram():
    """Log current VRAM status."""
    used, total = get_vram_usage()
    if used is not None:
        log.info(f"VRAM: {used}/{total} MB")
