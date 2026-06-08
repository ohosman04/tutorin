import json
import logging

from clients.llm_client import generate

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are grading an oral flashcard response. Your job is to evaluate whether the user demonstrated understanding, not whether they used the exact words.

Grading rubric — apply strictly:

correct:
- The user clearly expresses the main expected idea.
- Exact wording is not required, but the central concept must be present.
- Minor omissions are acceptable only if the core meaning is captured.

partially_correct:
- The user is on the right track but vague or incomplete.
- The user mentions only part of the expected answer.
- The user gives a related concept but misses a key detail.
- The answer would show partial understanding in an oral exam.

incorrect:
- The answer is unrelated to the expected answer.
- The answer contradicts the expected answer.
- The answer is generic filler with no specific content.
- The answer does not demonstrate understanding of the concept.

Grading rules:
- Do NOT mark vague answers as correct. Vague answers should be partially_correct at best.
- Prefer partially_correct over correct when key details are missing.
- Prefer partially_correct over incorrect when the answer is directionally related.
- Only mark incorrect when the answer is clearly wrong, unrelated, or empty.
- If the user said nothing meaningful, mark incorrect.

Feedback rules:
- Write 1-2 sentences of feedback.
- If correct: briefly confirm why the answer matches.
- If partially_correct: explain what was right and what specific detail was missing.
- If incorrect: explain why the answer does not demonstrate understanding.
- Never write feedback that simply says "correct" or "incorrect" without explanation.

Return only valid JSON. No markdown, no code fences, no explanation outside the JSON.\
"""

_SCHEMA_NOTE = """\
Return exactly this JSON shape:
{"grade": "correct|partially_correct|incorrect", "feedback": "<1-2 sentences>", "missing_points": ["<point>", ...], "confidence": <0.0-1.0>}\
"""

VALID_GRADES = {"correct", "partially_correct", "incorrect"}


def build_prompt(question: str, expected: str, transcript: str) -> str:
    return (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"User's spoken answer: {transcript}\n\n"
        f"{_SCHEMA_NOTE}"
    )


def _parse_response(raw: str) -> dict:
    """Extract and validate the JSON object from Ollama's response text."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back: find the first {...} block in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {raw!r}")
        data = json.loads(raw[start:end])

    grade = data.get("grade", "").strip().lower()
    if grade not in VALID_GRADES:
        logger.warning("Unexpected grade value %r — defaulting to incorrect", grade)
        grade = "incorrect"

    return {
        "grade": grade,
        "feedback": str(data.get("feedback", "")).strip(),
        "missing_points": [str(p) for p in data.get("missing_points", [])],
        "confidence": float(data.get("confidence", 0.0)),
    }


def grade_response(question: str, expected: str, transcript: str) -> dict:
    """
    Grade a spoken answer against the expected answer.
    Returns a validated dict with grade/feedback/missing_points/confidence
    plus latency_s from the LLM call.
    """
    prompt = build_prompt(question, expected, transcript)
    llm_result = generate(prompt, system=_SYSTEM)

    raw_text = llm_result.get("response", "")
    try:
        grading = _parse_response(raw_text)
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse grading response: %s | raw: %r", exc, raw_text)
        grading = {
            "grade": "incorrect",
            "feedback": "Grading failed — the model returned an invalid response. Please try again.",
            "missing_points": [],
            "confidence": 0.0,
        }

    grading["latency_s"] = llm_result.get("latency_s", 0.0)
    return grading
