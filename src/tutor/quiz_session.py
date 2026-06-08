import logging
import os
import random
import tempfile
import time

from audio import record_audio, record_until_enter, record_until_silence
from clients.stt_client import transcribe_wav
from clients.tts_client import build_spoken_feedback, speak
from tutor.grader import grade_response
from tutor.session_state import card_id, load_session, new_session, save_session

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


def _record_and_transcribe(
    record_mode: str = "auto",
    max_duration: float = DEFAULT_MAX_DURATION,
    silence_duration: float = 1.2,
    min_record_duration: float = 1.0,
    energy_threshold: float | None = None,
) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    t0 = time.perf_counter()
    try:
        if record_mode == "auto":
            result = record_until_silence(
                output_path=wav_path,
                max_duration=max_duration,
                silence_duration=silence_duration,
                min_record_duration=min_record_duration,
                energy_threshold=energy_threshold,
            )
            logger.info(
                "Recorded %.1fs — stopped: %s — threshold: %.1f",
                result["duration_s"], result["stopped_reason"], result["energy_threshold"],
            )
        elif record_mode == "enter":
            record_until_enter(output_path=wav_path, max_duration=max_duration)
            logger.info("Recorded in %.1fs", time.perf_counter() - t0)
        else:  # fixed
            record_audio(duration=max_duration, output_path=wav_path)
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


def _try_speak(text: str) -> None:
    """Speak text via TTS, printing a short error message on failure without raising."""
    try:
        speak(text)
    except RuntimeError as exc:
        print(f"  [TTS error] {exc}")
    except Exception as exc:
        print(f"  [TTS error] {exc}")


def _should_retry(transcript: str | None, grade: str) -> bool:
    return not transcript or grade in ("incorrect", "partially_correct")


def _answer_card(
    card: dict,
    label: str,
    record_mode: str,
    max_duration: float,
    silence_duration: float,
    min_record_duration: float,
    energy_threshold: float | None,
    speak_question: bool,
    speak_feedback: bool,
    scores: dict,
) -> tuple[str | None, str | None]:
    """
    Run the record→grade→display pipeline for one card.
    Returns (transcript, grade) — either may be None if the card was skipped.
    """
    print(f"  Q: {card['question']}")

    if speak_question:
        _try_speak(card["question"])

    if not _prompt_enter_or_quit():
        return "QUIT", None

    loop_start = time.perf_counter()

    transcript = _record_and_transcribe(
        record_mode=record_mode,
        max_duration=max_duration,
        silence_duration=silence_duration,
        min_record_duration=min_record_duration,
        energy_threshold=energy_threshold,
    )
    if transcript is None:
        print("  Skipping card due to error.")
        return None, None
    if not transcript:
        print("  Nothing transcribed — skipping card.")
        return "", None

    t0 = time.perf_counter()
    try:
        grading = grade_response(card["question"], card["answer"], transcript)
    except RuntimeError as exc:
        print(f"  [Grader error] {exc}")
        return transcript, None
    logger.info("Graded in %.1fs", time.perf_counter() - t0)

    _display_result(card, transcript, grading)

    if speak_feedback:
        _try_speak(build_spoken_feedback(grading))

    logger.info("Total card time: %.1fs", time.perf_counter() - loop_start)
    scores[grading["grade"]] = scores.get(grading["grade"], 0) + 1

    return transcript, grading["grade"]


def run_quiz(
    cards: list[dict],
    record_mode: str = "auto",
    max_duration: float = DEFAULT_MAX_DURATION,
    silence_duration: float = 1.2,
    min_record_duration: float = 1.0,
    energy_threshold: float | None = None,
    speak_feedback: bool = False,
    speak_question: bool = False,
    session_file: str | None = None,
) -> None:
    # ── record kwargs bundle (avoids repeating at every call site) ──
    rec_kw = dict(
        record_mode=record_mode,
        max_duration=max_duration,
        silence_duration=silence_duration,
        min_record_duration=min_record_duration,
        energy_threshold=energy_threshold,
        speak_question=speak_question,
        speak_feedback=speak_feedback,
    )

    # ── session setup ───────────────────────────────────────────────
    if session_file is not None:
        card_map = {card_id(c): c for c in cards}
        state = load_session(session_file)
        if state is None:
            state = new_session(cards)
            save_session(session_file, state)
            print(f"  New session — {len(state['card_order'])} cards  [{session_file}]")
        else:
            done = state["next_index"]
            total_deck = len(state["card_order"])
            msg = f"  Resuming — {done}/{total_deck} answered"
            if state["retry_queue"]:
                msg += f", {len(state['retry_queue'])} retries pending"
            print(msg)

        remaining_ids = state["card_order"][state["next_index"]:]
        main_cards = [card_map[cid] for cid in remaining_ids if cid in card_map]
        total_deck = len(state["card_order"])
    else:
        main_cards = list(cards)
        random.shuffle(main_cards)
        total_deck = len(main_cards)
        state = None

    # ── header ──────────────────────────────────────────────────────
    mode_hint = {
        "auto": f"auto-stop on silence (max {max_duration}s)",
        "enter": f"press Enter to stop (max {max_duration}s)",
        "fixed": f"fixed {max_duration}s",
    }.get(record_mode, record_mode)

    display_total = len(main_cards)
    if state and state["retry_queue"]:
        display_total += len(state["retry_queue"])

    print(f"\n=== Quiz Mode — {display_total} card(s) remaining ===")
    print(f"Recording: {mode_hint}\n")

    scores = {"correct": 0, "partially_correct": 0, "incorrect": 0}
    new_retry_ids: list[str] = []  # collected this run, folded into retry_queue after main pass

    # ── main pass ───────────────────────────────────────────────────
    start_num = (state["next_index"] if state else 0) + 1

    for i, card in enumerate(main_cards):
        print(f"\nCard {start_num + i}/{total_deck}")

        transcript, grade = _answer_card(card, "Card", scores=scores, **rec_kw)

        if transcript == "QUIT":
            # Save before exiting so we resume from this card next time
            if state:
                save_session(session_file, state)
            print("Quiz ended early.")
            _print_summary(scores)
            return

        # Card is consumed — advance index
        if state:
            state["next_index"] += 1
            # Collect retry candidates (save after folding below)
            if transcript is not None and _should_retry(transcript, grade or ""):
                cid = card_id(card)
                if cid not in new_retry_ids:
                    new_retry_ids.append(cid)
            save_session(session_file, state)

    # Fold this run's new retries into the persistent retry_queue
    if state and new_retry_ids:
        existing_set = set(state["retry_queue"])
        for cid in new_retry_ids:
            if cid not in existing_set:
                state["retry_queue"].append(cid)
                existing_set.add(cid)
        save_session(session_file, state)

    # ── retry pass ──────────────────────────────────────────────────
    if state and state["retry_queue"]:
        retry_ids = list(state["retry_queue"])
        retry_cards = [card_map[cid] for cid in retry_ids if cid in card_map]
        if retry_cards:
            print(f"\n  ── {len(retry_cards)} card(s) to retry ──")

        for j, card in enumerate(retry_cards, 1):
            cid = card_id(card)
            print(f"\nRetry {j}/{len(retry_cards)}")

            transcript, grade = _answer_card(card, "Retry", scores=scores, **rec_kw)

            if transcript == "QUIT":
                print("Quiz ended early.")
                _print_summary(scores)
                return

            # Answered correctly → clear from retry_queue
            if transcript and grade and not _should_retry(transcript, grade):
                if cid in state["retry_queue"]:
                    state["retry_queue"].remove(cid)
                    save_session(session_file, state)

    _print_summary(scores)


def _print_summary(scores: dict) -> None:
    print(f"\n=== Session Summary ===")
    print(f"  Correct          : {scores['correct']}")
    print(f"  Partially correct: {scores['partially_correct']}")
    print(f"  Incorrect        : {scores['incorrect']}")
