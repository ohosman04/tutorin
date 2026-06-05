import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

SYSTEM_PROMPT = "Summarize the user's statement in one sentence and ask one follow-up question."


def generate(prompt: str, timeout: float = 60.0, system: str | None = SYSTEM_PROMPT) -> dict:
    """Send a prompt to Ollama and return the parsed response dict with added latency_s."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    logger.info("Sending prompt to %s (model=%s)", url, OLLAMA_MODEL)

    t_start = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it running?"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Request to Ollama timed out after {timeout}s"
        ) from exc

    latency = time.perf_counter() - t_start

    if response.status_code == 404:
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' not found. Pull it with: ollama pull {OLLAMA_MODEL}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    data["latency_s"] = round(latency, 3)
    logger.info("Response received in %.2fs", latency)
    return data
