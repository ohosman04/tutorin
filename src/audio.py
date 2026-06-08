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


def record_until_silence(
    output_path: str = "recording.wav",
    max_duration: float = 60.0,
    silence_duration: float = 1.2,
    min_record_duration: float = 1.0,
    energy_threshold: float | None = None,
) -> dict:
    """
    Record until silence_duration seconds of quiet is detected (after min_record_duration),
    or until max_duration is reached.

    If energy_threshold is None, the first 0.5s of audio is used to calibrate ambient RMS,
    and threshold is set to max(ambient_rms * 2.5, 300.0). Calibration audio is kept.

    Returns:
        {
            "path": output_path,
            "duration_s": float,
            "stopped_reason": "silence" | "max_duration",
            "energy_threshold": float,
        }
    """
    CALIBRATION_SECS = 0.5
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()
    stopped_reason = "max_duration"

    # Mutable state shared with the callback — use a list as a simple container.
    state = {
        "threshold": energy_threshold,          # None until calibration finishes
        "calibrated": energy_threshold is not None,
        "calibration_samples": 0,
        "calibration_sum_sq": 0.0,
        "silence_samples": 0,
        "total_samples": 0,
    }
    silence_samples_needed = int(silence_duration * SAMPLE_RATE)
    min_samples_needed = int(min_record_duration * SAMPLE_RATE)
    calibration_samples_needed = int(CALIBRATION_SECS * SAMPLE_RATE)

    def _audio_callback(indata: np.ndarray, *_) -> None:
        chunk = indata.copy()
        chunks.append(chunk)

        samples = chunk.astype(np.float32).flatten()
        n = len(samples)
        state["total_samples"] += n

        # --- Calibration phase ---
        if not state["calibrated"]:
            state["calibration_sum_sq"] += float(np.sum(samples ** 2))
            state["calibration_samples"] += n
            if state["calibration_samples"] >= calibration_samples_needed:
                ambient_rms = (state["calibration_sum_sq"] / state["calibration_samples"]) ** 0.5
                state["threshold"] = max(ambient_rms * 2.5, 300.0)
                state["calibrated"] = True
                logger.debug(
                    "Calibrated: ambient_rms=%.1f threshold=%.1f",
                    ambient_rms, state["threshold"],
                )
            return  # don't check silence until calibrated

        # --- Silence detection phase ---
        rms = float(np.sqrt(np.mean(samples ** 2)))
        threshold = state["threshold"]

        if rms > threshold:
            state["silence_samples"] = 0  # speech detected — reset silence counter
        else:
            state["silence_samples"] += n
            if (
                state["total_samples"] >= min_samples_needed
                and state["silence_samples"] >= silence_samples_needed
            ):
                stop_event.set()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=_audio_callback,
        ):
            print("  Recording... (auto-stop on silence)", flush=True)
            fired_by_timeout = not stop_event.wait(timeout=max_duration)
            stopped_reason = "max_duration" if fired_by_timeout else "silence"
            if fired_by_timeout:
                print(f"  (max duration {max_duration}s reached)")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Microphone error: {exc}") from exc

    if not chunks:
        raise RuntimeError("No audio captured")

    audio = np.concatenate(chunks, axis=0)
    duration_s = len(audio) / SAMPLE_RATE
    _save_wav(audio, output_path)

    threshold_used = state["threshold"] or 0.0
    logger.info(
        "Recorded %.1fs — stopped: %s — threshold: %.1f",
        duration_s, stopped_reason, threshold_used,
    )
    return {
        "path": output_path,
        "duration_s": round(duration_s, 3),
        "stopped_reason": stopped_reason,
        "energy_threshold": round(threshold_used, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    record_audio()
