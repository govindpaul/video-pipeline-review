"""
Video trim + combine via ffmpeg.

Uses ffmpeg concat demuxer with 'inpoint' for per-scene trimming.
This avoids the stream-copy keyframe issue where -ss -c copy drops
all video frames (only keyframe is at t=0 in short clips).
"""

import logging
import subprocess
from pathlib import Path

from pipeline.story.parser import StoryData

log = logging.getLogger(__name__)


def combine_videos(
    videos: list[Path],
    story: StoryData,
    output_dir: Path,
) -> Path:
    """Trim and concatenate scene videos into combined.mp4.

    Uses concat demuxer 'inpoint' directive for trimming, then re-encodes
    to ensure consistent format and proper keyframe structure.

    Args:
        videos: List of scene video paths (ordered by scene number)
        story: Parsed story data (for trim notes)
        output_dir: Output directory for combined.mp4

    Returns:
        Path to combined.mp4
    """
    output_dir = Path(output_dir)
    combined_path = output_dir / "combined.mp4"

    if combined_path.exists():
        log.info("combined.mp4 already exists, skipping")
        return combined_path

    # Build trim map: scene_number -> trim_ms
    trim_map = {s.number: s.trim_ms for s in story.scenes}

    sorted_videos = sorted(videos, key=lambda v: v.name)

    # Create concat file with inpoint directives for trimming
    concat_file = output_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for video in sorted_videos:
            try:
                scene_num = int(video.stem.split("_")[1])
            except (IndexError, ValueError):
                scene_num = 0

            trim_ms = trim_map.get(scene_num, 500)
            trim_s = trim_ms / 1000.0

            f.write(f"file '{video.resolve()}'\n")
            if trim_ms > 0:
                f.write(f"inpoint {trim_s:.3f}\n")
                log.info(f"{video.name}: Trim {trim_ms}ms from start")
            else:
                log.info(f"{video.name}: No trim")

    # Concatenate with re-encoding (inpoint requires re-encode for accuracy)
    log.info(f"Concatenating {len(sorted_videos)} videos...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-crf", "19",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(combined_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        log.error(f"Concat failed: {result.stderr[:500]}")

        # Fallback: concat without trim (better than nothing)
        log.info("Fallback: concat without trim...")
        with open(concat_file, "w") as f:
            for video in sorted_videos:
                f.write(f"file '{video.resolve()}'\n")

        cmd_fallback = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(combined_path),
        ]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error(f"Fallback concat also failed: {result.stderr[:500]}")

    # Clean up
    try:
        concat_file.unlink()
    except Exception:
        pass

    if combined_path.exists():
        size_mb = combined_path.stat().st_size / (1024 * 1024)
        log.info(f"Combined: {combined_path} ({size_mb:.1f} MB)")
    else:
        log.error("combined.mp4 was not created")

    return combined_path
