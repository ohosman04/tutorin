import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio import record_audio
from clients.stt_client import transcribe_wav
from clients.llm_client import generate, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RECORD_DURATION = float(os.environ.get("RECORD_DURATION", "5"))

TUTOR_SYSTEM = (
    "You are a concise oral tutoring assistant. "
    "Summarize what the user said in one sentence and ask one useful follow-up question."
)


def run_loop():
    print("=== Jetson Local Tutor ===")
    print(f"Model : {OLLAMA_MODEL}")
    print(f"Record: {RECORD_DURATION}s per turn")
    print("Press Enter to speak, q + Enter to quit.\n")

    while True:
        try:
            cmd = input("[ Press Enter to record / q to quit ] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd == "q":
            print("Goodbye.")
            break

        loop_start = time.perf_counter()

        # --- Record ---
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            record_audio(duration=RECORD_DURATION, output_path=wav_path)
        except RuntimeError as exc:
            print(f"[Audio error] {exc}")
            continue
        print(f"  Recorded in {time.perf_counter() - t0:.1f}s")

        # --- STT ---
        t0 = time.perf_counter()
        try:
            stt_result = transcribe_wav(wav_path)
        except Exception as exc:
            print(f"[STT error] {exc}")
            continue
        finally:
            os.unlink(wav_path)

        transcript = stt_result.get("transcript", "").strip()
        print(f"  STT in {time.perf_counter() - t0:.1f}s")
        print(f"\nYou said: {transcript or '(empty)'}\n")

        if not transcript:
            print("Nothing transcribed — try again.\n")
            continue

        # --- LLM ---
        prompt = f"The user said: \"{transcript}\""
        t0 = time.perf_counter()
        try:
            llm_result = generate(prompt, system=TUTOR_SYSTEM)
        except RuntimeError as exc:
            print(f"[LLM error] {exc}")
            continue
        print(f"  LLM in {time.perf_counter() - t0:.1f}s")

        print(f"\nTutor: {llm_result.get('response', '').strip()}\n")
        print(f"  Total loop: {time.perf_counter() - loop_start:.1f}s\n")


if __name__ == "__main__":
    run_loop()
