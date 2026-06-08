import logging
import os
import random
import tempfile
import time

from audio import record_until_enter
from clients.stt_client import transcribe_wav
from tutor.grader import grade_response

logger = logging.getLogger(__name__)

DEFAULT_MAX_DURATION = float(os.environ.get("RECORD_MAX_DURATION", "60"))

_GRADE_ICONS = {
    "correct": "✓",
    "partially_correct": "~",
    "incorrect": "✗",
}


def _prompt_enter_or_quit() -> bool:
    """Return True to continue, False to quit."""
    try:
        cmd = input("\n[ Press Enter to answer / q to quit ] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return cmd != "q"


def _record_and_transcribe(max_duration: float = DEFAULT_MAX_DURATION) -> str | None:
    """Record until Enter is pressed and return transcript, or None on failure."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    t0 = time.perf_counter()
    try:
        record_until_enter(output_path=wav_path, max_duration=max_duration)
        logger.info("Recorded in %.1fs", time.perf_counter() - t0)
    except RuntimeError as exc:
        print(f"  [Audio error] {exc}")
        os.unlink(wav_path)
        return None

    t0 = time.perf_counter()
    try:
        stt_result = transcribe_wav(wav_path)
    except Exception as exc:
        print(f"  [STT error] {exc}")
        return None
    finally:
        os.unlink(wav_path)

    logger.info("Transcribed in %.1fs", time.perf_counter() - t0)
    return stt_result.get("transcript", "").strip()


def _display_result(card: dict, transcript: str, grading: dict) -> None:
    icon = _GRADE_ICONS.get(grading["grade"], "?")
    print("\n" + "─" * 50)
    print(f"  Question : {card['question']}")
    print(f"  Expected : {card['answer']}")
    print(f"  You said : {transcript or '(nothing transcribed)'}")
    print(f"  Grade    : {icon}  {grading['grade']}")
    print(f"  Feedback : {grading['feedback']}")
    if grading["missing_points"]:
        for point in grading["missing_points"]:
            print(f"             • {point}")
    print(f"  Latency  : {grading['latency_s']}s grading")
    print("─" * 50)


def run_quiz(cards: list[dict], max_duration: float = DEFAULT_MAX_DURATION) -> None:
    deck = list(cards)
    random.shuffle(deck)
    total = len(deck)

    print(f"\n=== Quiz Mode — {total} cards ===")
    print(f"Max recording duration: {max_duration}s per answer (press Enter to stop early)\n")

    scores = {"correct": 0, "partially_correct": 0, "incorrect": 0}

    for i, card in enumerate(deck, 1):
        print(f"\nCard {i}/{total}")
        print(f"  Q: {card['question']}")

        if not _prompt_enter_or_quit():
            print("Quiz ended early.")
            break

        loop_start = time.perf_counter()

        transcript = _record_and_transcribe(max_duration=max_duration)
        if transcript is None:
            print("  Skipping card due to error.")
            continue

        if not transcript:
            print("  Nothing transcribed — skipping card.")
            continue

        t0 = time.perf_counter()
        try:
            grading = grade_response(card["question"], card["answer"], transcript)
        except RuntimeError as exc:
            print(f"  [Grader error] {exc}")
            continue
        logger.info("Graded in %.1fs", time.perf_counter() - t0)

        _display_result(card, transcript, grading)
        logger.info("Total card time: %.1fs", time.perf_counter() - loop_start)

        scores[grading["grade"]] = scores.get(grading["grade"], 0) + 1

    print(f"\n=== Session Summary ===")
    print(f"  Correct          : {scores['correct']}")
    print(f"  Partially correct: {scores['partially_correct']}")
    print(f"  Incorrect        : {scores['incorrect']}")
