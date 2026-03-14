"""
Ollama LLM wrapper for local model inference.

Handles chat (text) and vision (image+text) tasks.
Manages model loading/unloading for VRAM orchestration.
"""

import base64
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"


def health_check() -> bool:
    """Check if Ollama is running."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_running():
    """Start Ollama if not running."""
    if health_check():
        return
    log.info("Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(15):
        time.sleep(1)
        if health_check():
            log.info("Ollama ready")
            return
    raise RuntimeError("Failed to start Ollama")


def stop_all():
    """Stop all loaded Ollama models to free VRAM."""
    log.info("Stopping all Ollama models...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for model_info in data.get("models", []):
                name = model_info.get("name", "")
                if name:
                    requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=10,
                    )
                    log.info(f"Unloaded: {name}")
    except Exception as e:
        log.warning(f"Could not stop models cleanly: {e}")
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    format_json: bool = False,
    max_tokens: int = 4096,
) -> str:
    """Send a chat request to Ollama.

    Args:
        model: Model name (e.g. 'qwen3.5:35b-a3b')
        messages: List of {'role': 'system'|'user'|'assistant', 'content': str}
        temperature: Sampling temperature
        format_json: If True, request JSON output format
        max_tokens: Max tokens to generate

    Returns:
        Model response text
    """
    ensure_running()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if format_json:
        payload["format"] = "json"

    log.info(f"Chat: {model} ({len(messages)} messages)")
    start = time.time()

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    result = resp.json()
    content = result["message"]["content"]
    elapsed = time.time() - start

    eval_count = result.get("eval_count", 0)
    speed = eval_count / elapsed if elapsed > 0 else 0
    log.info(f"Chat done: {elapsed:.1f}s, {eval_count} tokens, {speed:.0f} tok/s")

    return _strip_thinking(content)


def vision(
    model: str,
    image_path: str | Path,
    prompt: str,
    temperature: float = 0.3,
    format_json: bool = False,
    max_tokens: int = 2048,
) -> str:
    """Send a vision request (image + text prompt) to Ollama.

    Args:
        model: Vision model name (e.g. 'qwen3-vl:30b-a3b')
        image_path: Path to image file
        prompt: Text prompt to accompany the image
        temperature: Sampling temperature
        format_json: If True, request JSON output format
        max_tokens: Max tokens to generate

    Returns:
        Model response text
    """
    ensure_running()

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if format_json:
        payload["format"] = "json"

    log.info(f"Vision: {model} + {image_path.name}")
    start = time.time()

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    result = resp.json()
    content = result["message"]["content"]
    elapsed = time.time() - start
    log.info(f"Vision done: {elapsed:.1f}s")

    return _strip_thinking(content)


def _strip_thinking(text: str) -> str:
    """Strip Qwen3.5 <think>...</think> blocks from model output."""
    import re
    # Remove <think>...</think> blocks (Qwen3.5 thinking mode)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()
