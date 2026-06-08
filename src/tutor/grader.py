import json
import logging

from clients.llm_client import generate

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are grading an oral flashcard response. "
    "Compare the user's answer to the expected answer. "
    "Accept reasonable paraphrases and equivalent explanations. "
    "Return only valid JSON — no markdown, no explanation, no code fences."
)

_SCHEMA_NOTE = (
    'Return exactly this JSON shape:\n'
    '{"grade": "correct|partially_correct|incorrect", '
    '"feedback": "<one concise sentence>", '
    '"missing_points": ["<point>", ...], '
    '"confidence": <0.0-1.0>}'
)

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
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first {...} block in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {raw!r}")
        data = json.loads(raw[start:end])

    # Validate and normalise
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
            "feedback": "Grading failed — could not parse model response.",
            "missing_points": [],
            "confidence": 0.0,
        }

    grading["latency_s"] = llm_result.get("latency_s", 0.0)
    return grading
