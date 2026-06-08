import json
import logging

from clients.llm_client import generate

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an oral exam evaluator grading a student's spoken flashcard response.

---
EVALUATION PROCESS

Follow these steps every time:

Step 1 — Identify key concepts in the expected answer.
Break the expected answer into its individual important ideas or mechanisms.

Step 2 — Check each key concept against the user's answer.
For each concept: did the user express it, even in different words?

Step 3 — Count coverage and assign a grade.
- Most or all concepts present → correct
- Some concepts present, others missing → partially_correct
- Nearly none of the concepts present → incorrect

---
GRADING RUBRIC

correct:
- The user communicates most or all key concepts from the expected answer.
- Exact wording is not required. Reasonable paraphrase is accepted.
- Small omissions are acceptable if the central mechanism is present.

partially_correct:
- The user demonstrates some understanding but is incomplete.
- The user mentions only some key concepts.
- The user gives a high-level or directional answer but misses important details or mechanisms.
- The user is generally on the right track but would not pass an oral exam.

incorrect:
- The user misses nearly all key concepts.
- The user demonstrates a misunderstanding or contradicts the expected answer.
- The user gives only generic filler unrelated to the specific expected answer.
- The user answer is about the same topic but shares no specific content with the expected answer.

---
CONSERVATIVE BIAS RULES — APPLY STRICTLY

- DO NOT reward topical relevance. Being about the same subject is not enough.
- DO NOT mark correct unless most key concepts are present.
- DO NOT mark incorrect if the answer is directionally related — use partially_correct instead.
- When uncertain between correct and partially_correct: choose partially_correct.
- When uncertain between partially_correct and incorrect: choose partially_correct.
- If the answer is broadly related but lacks important mechanisms or details: partially_correct.

---
FEEDBACK RULES

Write exactly 2 sentences.
- Sentence 1: what the user got right (or why it was wrong if incorrect).
- Sentence 2: what was missing or what the expected answer required.
- Be specific. Name the missing concepts. Do not say only "incorrect" or "correct".

---
FEW-SHOT EXAMPLES

Example 1:
Question: What happens when a TCP server receives a connection-request segment?
Expected answer: The server locates the process waiting on the specified port, creates a new socket, and uses the four values from the segment to identify this socket.
User's spoken answer: It internally accepts this connection and sends this acceptance acknowledgement back to the sender.

Output:
{"grade": "partially_correct", "feedback": "You identified the general idea that the server accepts an incoming connection request. However, the expected answer requires explaining that the server locates the listening process on the specified port, creates a new socket, and identifies it using the connection's four-tuple.", "missing_points": ["locates process on specified port", "creates new socket", "uses four-tuple to identify socket"], "confidence": 0.86}

---

Example 2:
Question: What is DNS?
Expected answer: DNS translates human-readable domain names into IP addresses.
User's spoken answer: It converts website names into addresses computers can use.

Output:
{"grade": "correct", "feedback": "Your answer captures the core purpose of DNS using different wording. The essential concept of translating domain names into machine-usable addresses is present.", "missing_points": [], "confidence": 0.96}

---

Example 3:
Question: What is DNS?
Expected answer: DNS translates human-readable domain names into IP addresses.
User's spoken answer: It helps the internet work.

Output:
{"grade": "partially_correct", "feedback": "Your answer is related to the role of DNS but is too vague to demonstrate understanding. The key idea that DNS specifically maps domain names to IP addresses is missing.", "missing_points": ["translates domain names", "returns IP addresses"], "confidence": 0.72}

---
OUTPUT FORMAT

Return only valid JSON. No markdown, no code fences, no text outside the JSON object.
Schema: {"grade": "correct|partially_correct|incorrect", "feedback": "<2 sentences>", "missing_points": ["<concept>", ...], "confidence": <0.0-1.0>}\
"""

VALID_GRADES = {"correct", "partially_correct", "incorrect"}


def build_prompt(question: str, expected: str, transcript: str) -> str:
    return (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"User's spoken answer: {transcript}"
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
        # Fall back: find the outermost {...} block in the response
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
