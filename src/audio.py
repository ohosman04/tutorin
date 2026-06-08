import logging
import threading
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


def record_until_enter(output_path: str = "recording.wav", max_duration: float = 60.0) -> str:
    """Record from the default microphone until Enter is pressed or max_duration is reached."""
    chunks = []
    stop_event = threading.Event()

    def _audio_callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    def _wait_for_enter():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        stop_event.set()

    enter_thread = threading.Thread(target=_wait_for_enter, daemon=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=_audio_callback,
        ):
            print("  Recording... press Enter to stop", end="", flush=True)
            enter_thread.start()
            fired_by_timeout = not stop_event.wait(timeout=max_duration)
            if fired_by_timeout:
                print(f"\n  (max duration {max_duration}s reached)")
            else:
                print()  # newline after the prompt
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Microphone error: {exc}") from exc

    if not chunks:
        raise RuntimeError("No audio captured")

    audio = np.concatenate(chunks, axis=0)
    _save_wav(audio, output_path)
    logger.info("Saved %d samples (%.1fs) to %s", len(audio), len(audio) / SAMPLE_RATE, output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    record_audio()
