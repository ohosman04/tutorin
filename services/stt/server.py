import io
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------------------------------------------------------------------
# Model load — attempt GPU first, fall back to CPU
# Set WHISPER_MODEL env var to change the model (default: tiny.en).
# For multilingual decks use a non-.en model, e.g. WHISPER_MODEL=tiny
# ---------------------------------------------------------------------------
_device: str
_model: WhisperModel
_model_name: str = os.environ.get("WHISPER_MODEL", "tiny.en")
_is_english_only: bool = _model_name.endswith(".en")

t0 = time.perf_counter()
try:
    _model = WhisperModel(_model_name, device="cuda", compute_type="float16")
    _device = "cuda"
    logger.info("Model '%s' loaded on CUDA in %.2fs", _model_name, time.perf_counter() - t0)
except Exception as exc:
    logger.warning("CUDA init failed (%s), falling back to CPU", exc)
    t0 = time.perf_counter()
    _model = WhisperModel(_model_name, device="cpu", compute_type="int8")
    _device = "cpu"
    logger.info("Model '%s' loaded on CPU in %.2fs", _model_name, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")

    if language and _is_english_only and language != "en":
        logger.warning(
            "Language '%s' requested but model '%s' is English-only — "
            "set WHISPER_MODEL=tiny for multilingual support",
            language, _model_name,
        )

    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Pass language hint only for multilingual models; let english-only model
    # default to its built-in language.
    transcribe_kwargs = {}
    if language and not _is_english_only:
        transcribe_kwargs["language"] = language

    t_start = time.perf_counter()
    try:
        segments, info = _model.transcribe(audio_buffer, **transcribe_kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t_start
    logger.info("Transcribed %.2fs of audio in %.2fs on %s", info.duration, elapsed, _device)

    return {
        "transcript": text,
        "device": _device,
        "audio_duration_s": round(info.duration, 3),
        "transcription_time_s": round(elapsed, 3),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": _device}
