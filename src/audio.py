import logging
import wave

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def record_audio(duration: float = 5.0, output_path: str = "recording.wav") -> str:
    """Record from the default microphone and save as a 16 kHz mono WAV file."""
    logger.info("Recording %.1f seconds at %d Hz to %s", duration, SAMPLE_RATE, output_path)

    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Microphone error: {exc}") from exc

    _save_wav(audio, output_path)
    logger.info("Saved to %s", output_path)
    return output_path


def _save_wav(audio: np.ndarray, path: str) -> None:
    with wave.open(path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    record_audio()
