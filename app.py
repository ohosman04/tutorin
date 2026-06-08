import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio import record_audio
from clients.stt_client import transcribe_wav
from clients.llm_client import generate, OLLAMA_MODEL
from anki_parser import load_deck, inspect_deck_fields
from tutor.quiz_session import run_quiz

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FREEFORM_DURATION = float(os.environ.get("FREEFORM_DURATION", "5"))


def _freeform():
    """Diagnostic: freeform microphone -> STT -> Ollama conversation."""
    SYSTEM = (
        "You are a concise oral tutoring assistant. "
        "Summarize what the user said in one sentence and ask one useful follow-up question."
    )
    print("=== Freeform Mode (diagnostic) ===")
    print(f"Model : {OLLAMA_MODEL}  |  Record: {FREEFORM_DURATION}s\n")

    while True:
        try:
            cmd = input("[ Enter to record / q to quit ] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q":
            break

        loop_start = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        t0 = time.perf_counter()
        try:
            record_audio(duration=FREEFORM_DURATION, output_path=wav_path)
        except RuntimeError as exc:
            print(f"[Audio error] {exc}")
            continue
        logger.info("Recorded in %.1fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            stt_result = transcribe_wav(wav_path)
        except Exception as exc:
            print(f"[STT error] {exc}")
            continue
        finally:
            os.unlink(wav_path)
        transcript = stt_result.get("transcript", "").strip()
        logger.info("Transcribed in %.1fs", time.perf_counter() - t0)
        print(f"\nYou said: {transcript or '(empty)'}")

        if not transcript:
            continue

        t0 = time.perf_counter()
        try:
            llm_result = generate(f'The user said: "{transcript}"', system=SYSTEM)
        except RuntimeError as exc:
            print(f"[LLM error] {exc}")
            continue
        logger.info("LLM in %.1fs", time.perf_counter() - t0)

        print(f"Tutor: {llm_result.get('response', '').strip()}")
        logger.info("Total loop: %.1fs", time.perf_counter() - loop_start)


def _usage():
    print("Usage:")
    print("  python app.py quiz --deck path/to/deck.apkg [--duration SECONDS]")
    print("                     [--inspect-fields]")
    print("                     [--question-field NAME --answer-field NAME]")
    print("  python app.py freeform")
    sys.exit(1)


def main():
    args = sys.argv[1:]

    if not args:
        _usage()

    mode = args[0]

    if mode == "quiz":
        if "--deck" not in args:
            print("Error: quiz mode requires --deck path/to/deck.apkg")
            sys.exit(1)
        deck_path = args[args.index("--deck") + 1]

        # --inspect-fields: print field names and exit
        if "--inspect-fields" in args:
            try:
                info = inspect_deck_fields(deck_path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[Deck error] {exc}")
                sys.exit(1)
            print(f"Note types in {deck_path}:")
            for model_name, field_names in info.items():
                print(f"  {model_name}")
                for name in field_names:
                    print(f"    - {name}")
            sys.exit(0)

        # --record-mode
        record_mode = "auto"
        if "--record-mode" in args:
            try:
                record_mode = args[args.index("--record-mode") + 1]
            except IndexError:
                print("Error: --record-mode requires a value: auto | enter | fixed")
                sys.exit(1)
            if record_mode not in ("auto", "enter", "fixed"):
                print(f"Error: unknown --record-mode {record_mode!r}. Use: auto | enter | fixed")
                sys.exit(1)

        # --duration
        max_duration = 60.0
        if "--duration" in args:
            try:
                max_duration = float(args[args.index("--duration") + 1])
            except (IndexError, ValueError):
                print("Error: --duration requires a number (e.g. --duration 10)")
                sys.exit(1)

        # auto-mode tuning flags
        silence_duration = 1.2
        if "--silence-duration" in args:
            try:
                silence_duration = float(args[args.index("--silence-duration") + 1])
            except (IndexError, ValueError):
                print("Error: --silence-duration requires a number (e.g. --silence-duration 1.5)")
                sys.exit(1)

        min_record_duration = 1.0
        if "--min-record-duration" in args:
            try:
                min_record_duration = float(args[args.index("--min-record-duration") + 1])
            except (IndexError, ValueError):
                print("Error: --min-record-duration requires a number")
                sys.exit(1)

        energy_threshold = None
        if "--energy-threshold" in args:
            try:
                energy_threshold = float(args[args.index("--energy-threshold") + 1])
            except (IndexError, ValueError):
                print("Error: --energy-threshold requires a number (e.g. --energy-threshold 800)")
                sys.exit(1)

        # TTS flags
        speak_feedback = "--speak-feedback" in args
        speak_question = "--speak-question" in args

        # session flags
        from tutor.session_state import session_path_for
        session_file = None
        if "--resume" in args:
            session_file = session_path_for(os.path.abspath(deck_path))
        if "--reset-session" in args:
            sp = session_path_for(os.path.abspath(deck_path))
            if os.path.exists(sp):
                os.unlink(sp)
                print(f"Session reset: {sp}")
            else:
                print("No session file found to reset.")

        # field mapping
        question_field = None
        answer_field = None
        if "--question-field" in args:
            try:
                question_field = args[args.index("--question-field") + 1]
            except IndexError:
                print("Error: --question-field requires a field name")
                sys.exit(1)
        if "--answer-field" in args:
            try:
                answer_field = args[args.index("--answer-field") + 1]
            except IndexError:
                print("Error: --answer-field requires a field name")
                sys.exit(1)

        try:
            cards = load_deck(deck_path, question_field=question_field, answer_field=answer_field)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[Deck error] {exc}")
            sys.exit(1)
        if not cards:
            print("Deck loaded but contains no usable cards.")
            sys.exit(1)
        print(f"Loaded {len(cards)} cards from {deck_path}")
        run_quiz(
            cards,
            record_mode=record_mode,
            max_duration=max_duration,
            silence_duration=silence_duration,
            min_record_duration=min_record_duration,
            energy_threshold=energy_threshold,
            speak_feedback=speak_feedback,
            speak_question=speak_question,
            session_file=session_file,
        )

    elif mode == "freeform":
        _freeform()

    else:
        _usage()


if __name__ == "__main__":
    main()
