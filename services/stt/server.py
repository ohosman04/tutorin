import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------------------------------------------------------------------
# Model load — attempt GPU first, fall back to CPU
# ---------------------------------------------------------------------------
_device: str
_model: WhisperModel

t0 = time.perf_counter()
try:
    _model = WhisperModel("tiny.en", device="cuda", compute_type="float16")
    _device = "cuda"
    logger.info("Model loaded on CUDA in %.2fs", time.perf_counter() - t0)
except Exception as exc:
    logger.warning("CUDA init failed (%s), falling back to CPU", exc)
    t0 = time.perf_counter()
    _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    _device = "cpu"
    logger.info("Model loaded on CPU in %.2fs", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")

    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    t_start = time.perf_counter()
    try:
        segments, info = _model.transcribe(audio_buffer, language="en")
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
