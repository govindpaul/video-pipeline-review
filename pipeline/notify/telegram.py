"""
Telegram Bot API notifications.
"""

import logging
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)


def send_message(text: str, config: dict) -> bool:
    """Send a text message via Telegram Bot API."""
    tg = config.get("telegram", {})
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")

    if not token or not chat_id:
        log.warning("Telegram not configured")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=30,
        )
        if resp.status_code == 200:
            log.info("Telegram message sent")
            return True
        log.error(f"Telegram error: {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def send_video(video_path: Path | str, caption: str, config: dict) -> bool:
    """Send a video file via Telegram Bot API."""
    tg = config.get("telegram", {})
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")

    if not token or not chat_id:
        log.warning("Telegram not configured")
        return False

    video_path = Path(video_path)
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        return False

    # Telegram limit: 50MB for video
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        log.warning(f"Video too large for Telegram ({size_mb:.1f}MB > 50MB), sending message only")
        return send_message(f"{caption}\n\n(Video too large to send: {size_mb:.1f}MB)", config)

    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={"chat_id": chat_id, "caption": caption},
                files={"video": f},
                timeout=120,
            )
        if resp.status_code == 200:
            log.info("Telegram video sent")
            return True
        log.error(f"Telegram video error: {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        log.error(f"Telegram video error: {e}")
        return False


def send_photo(photo_path: Path | str, caption: str, config: dict) -> bool:
    """Send a photo via Telegram Bot API."""
    tg = config.get("telegram", {})
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")

    if not token or not chat_id:
        return False

    photo_path = Path(photo_path)
    if not photo_path.exists():
        return False

    try:
        with open(photo_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": f},
                timeout=60,
            )
        return resp.status_code == 200
    except Exception:
        return False
