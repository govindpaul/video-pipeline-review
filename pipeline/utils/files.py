"""
File utilities — Nextcloud upload.
"""

import logging
from pathlib import Path
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

log = logging.getLogger(__name__)


def upload_to_nextcloud(
    local_path: Path | str,
    config: dict,
    remote_folder: str = "AI_VIDEOS",
    remote_filename: Optional[str] = None,
) -> bool:
    """Upload a file to Nextcloud via WebDAV.

    Args:
        local_path: Path to local file
        config: Pipeline config dict (needs nextcloud.url, .user, .password)
        remote_folder: Nextcloud folder name
        remote_filename: Override filename (default: same as local)
    """
    local_path = Path(local_path)
    if not local_path.exists():
        log.error(f"File not found: {local_path}")
        return False

    nc = config.get("nextcloud", {})
    base_url = nc.get("url", "http://192.168.2.84/remote.php/dav/files/paul/AI_VIDEOS/")
    user = nc.get("user", "paul")
    password = nc.get("password", "")

    if remote_filename is None:
        remote_filename = local_path.name

    # Build WebDAV URL — base_url already ends with folder/
    webdav_url = f"{base_url.rstrip('/')}/{remote_filename}"

    log.info(f"Uploading {local_path.name} to Nextcloud/{remote_folder}...")

    try:
        with open(local_path, "rb") as f:
            resp = requests.put(
                webdav_url,
                data=f,
                auth=HTTPBasicAuth(user, password),
                timeout=300,
            )
        if resp.status_code in (200, 201, 204):
            log.info(f"Uploaded: {remote_filename}")
            return True
        else:
            log.error(f"Upload failed: HTTP {resp.status_code}")
            return False
    except requests.exceptions.Timeout:
        log.error(f"Upload timeout: {remote_filename}")
        return False
    except Exception as e:
        log.error(f"Upload error: {e}")
        return False
