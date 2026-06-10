import logging

from clients.llm_client import generate

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an oral tutor generating a follow-up question for a student who just answered incorrectly.

Rules:
- Ask ONE short question only.
- Keep it concise and suitable for speech (one sentence).
- Target the student's specific misunderstanding.
- Do NOT reveal the answer.
- If the student was completely wrong, make the question easier and more direct than the original.
- Focus on the single most important missing concept.
- Return only the question text. No preamble, no explanation.\
"""


def generate_followup_question(
    question: str, expected: str, user_answer: str, feedback: str
) -> str:
    """Return a single follow-up question string, or '' on failure."""
    prompt = (
        f"Original question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Student's answer: {user_answer}\n"
        f"Grader feedback: {feedback}\n\n"
        "Generate one short follow-up question:"
    )
    try:
        result = generate(prompt, system=_SYSTEM)
        return result.get("response", "").strip()
    except RuntimeError as exc:
        logger.warning("Follow-up generation failed: %s", exc)
        return ""


def build_followup_response(grade: str, grading: dict) -> str:
    """Build a short spoken tutor response after grading the follow-up answer."""
    if grade == "correct":
        return "That's right."
    missing = grading.get("missing_points", [])
    first_sentence = grading.get("feedback", "").split(".")[0].strip()
    if grade == "partially_correct":
        return f"Almost. {first_sentence}." if first_sentence else "Almost."
    # incorrect
    if missing:
        return f"Not quite. {missing[0].capitalize()}."
    return f"Not quite. {first_sentence}." if first_sentence else "Not quite."
