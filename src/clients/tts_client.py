import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.path.expanduser("~/piper/models/en_US-lessac-medium.onnx")
PIPER_MODEL = os.environ.get("PIPER_MODEL", _DEFAULT_MODEL)

_GRADE_SPOKEN = {
    "correct": "Correct.",
    "partially_correct": "Partially correct.",
    "incorrect": "Not quite.",
}


def build_spoken_feedback(grading: dict) -> str:
    """Build a concise spoken version of grading results."""
    parts = []

    prefix = _GRADE_SPOKEN.get(grading.get("grade", ""))
    if prefix:
        parts.append(prefix)

    feedback = grading.get("feedback", "").strip()
    if feedback:
        parts.append(feedback)

    missing = grading.get("missing_points", [])
    if missing:
        first_two = missing[:2]
        joined = " and ".join(first_two)
        parts.append(f"You were missing: {joined}.")

    return " ".join(parts)


def speak(text: str, model_path: str | None = None) -> None:
    """Synthesize text with Piper CLI and play via aplay. Raises RuntimeError on failure.

    model_path: path to .onnx Piper voice model. Defaults to PIPER_MODEL env var / built-in default.
    """
    if model_path is None:
        model_path = PIPER_MODEL
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        try:
            piper_proc = subprocess.run(
                ["piper", "--model", model_path, "--output-file", wav_path],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "piper not found — install with: pip install piper-tts  "
                "and: sudo apt install espeak-ng"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("piper synthesis timed out after 30s")

        if piper_proc.returncode != 0:
            stderr = piper_proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"piper exited {piper_proc.returncode}: {stderr}")

        logger.info("Piper synthesis complete — playing %s", wav_path)

        try:
            play_proc = subprocess.run(
                ["aplay", "-q", wav_path],
                capture_output=True,
                timeout=60,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "aplay not found — install with: sudo apt install alsa-utils"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("aplay timed out during playback")

        if play_proc.returncode != 0:
            stderr = play_proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"aplay exited {play_proc.returncode}: {stderr}")

        logger.info("TTS playback complete")

    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
