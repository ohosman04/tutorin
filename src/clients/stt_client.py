import logging
import os

import httpx

logger = logging.getLogger(__name__)

STT_URL = os.environ.get("STT_URL", "http://localhost:8000")


def transcribe_wav(wav_path: str, timeout: float = 30.0) -> dict:
    """Send a WAV file to the STT service and return the response dict."""
    url = f"{STT_URL}/transcribe"
    logger.info("Sending %s to %s", wav_path, url)

    with open(wav_path, "rb") as f:
        response = httpx.post(
            url,
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            timeout=timeout,
        )

    response.raise_for_status()
    return response.json()
