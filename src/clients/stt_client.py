import logging
import os

import httpx

logger = logging.getLogger(__name__)

STT_URL = os.environ.get("STT_URL", "http://localhost:8000")


def transcribe_wav(
    wav_path: str, timeout: float = 30.0, language: str | None = None
) -> dict:
    """Send a WAV file to the STT service and return the response dict.

    language: BCP-47 tag (e.g. 'en', 'es', 'fr'). Passed to the server as a
    form field; ignored silently if the server or model does not support it.
    """
    url = f"{STT_URL}/transcribe"
    logger.info("Sending %s to %s (language=%s)", wav_path, url, language or "default")

    data = {"language": language} if language else {}

    with open(wav_path, "rb") as f:
        response = httpx.post(
            url,
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data=data,
            timeout=timeout,
        )

    response.raise_for_status()
    return response.json()
