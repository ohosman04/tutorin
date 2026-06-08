"""
Offline unit tests for grader.py prompt building and JSON parsing.
No Ollama or microphone required.

To test live grading against a running Ollama instance:
    python tests/test_grader_prompt.py --live
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tutor.grader import _SYSTEM, build_prompt, _parse_response, grade_response, VALID_GRADES


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------

def test_prompt_contains_question_expected_transcript():
    prompt = build_prompt(
        question="What is photosynthesis?",
        expected="The process by which plants convert sunlight into glucose.",
        transcript="Plants use sunlight to make food.",
    )
    assert "What is photosynthesis?" in prompt
    assert "convert sunlight into glucose" in prompt
    assert "Plants use sunlight to make food." in prompt
    print("PASS: build_prompt contains question, expected answer, and transcript")


def test_system_prompt_contains_rubric_grades():
    for grade in ("correct", "partially_correct", "incorrect"):
        assert grade in _SYSTEM, f"Rubric missing grade: {grade}"
    print("PASS: system prompt contains all three rubric grades")


def test_system_prompt_rejects_vague_as_correct():
    assert "vague" in _SYSTEM.lower(), "Prompt should warn against marking vague answers as correct"
    assert "partially_correct" in _SYSTEM
    print("PASS: system prompt instructs not to mark vague answers as correct")


def test_system_prompt_requires_explanation_when_incorrect():
    # The feedback rules section should instruct explanation for incorrect grades
    assert "incorrect" in _SYSTEM
    assert "explain" in _SYSTEM.lower(), "Prompt should require explanation in feedback"
    print("PASS: system prompt requires explanation for incorrect grades")


def test_system_prompt_prefers_partial_over_incorrect():
    assert "directionally" in _SYSTEM.lower() or "partially_correct over incorrect" in _SYSTEM
    print("PASS: system prompt prefers partially_correct over incorrect for related answers")


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    raw = json.dumps({
        "grade": "correct",
        "feedback": "The answer captures the core concept accurately.",
        "missing_points": [],
        "confidence": 0.9,
    })
    result = _parse_response(raw)
    assert result["grade"] == "correct"
    assert result["confidence"] == 0.9
    assert result["feedback"] != ""
    print("PASS: parse valid JSON")


def test_parse_partially_correct():
    raw = json.dumps({
        "grade": "partially_correct",
        "feedback": "You identified the right process but missed the glucose output.",
        "missing_points": ["glucose as the output of photosynthesis"],
        "confidence": 0.65,
    })
    result = _parse_response(raw)
    assert result["grade"] == "partially_correct"
    assert len(result["missing_points"]) == 1
    print("PASS: parse partially_correct with missing_points")


def test_parse_json_with_code_fence():
    raw = (
        '```json\n'
        '{"grade": "partially_correct", "feedback": "Close but missing detail.", '
        '"missing_points": ["key detail"], "confidence": 0.6}\n'
        '```'
    )
    result = _parse_response(raw)
    assert result["grade"] == "partially_correct"
    print("PASS: parse JSON wrapped in code fence")


def test_parse_json_embedded_in_text():
    raw = (
        'Here is my assessment: '
        '{"grade": "incorrect", "feedback": "The answer is unrelated to the concept.", '
        '"missing_points": [], "confidence": 0.1} Hope that helps.'
    )
    result = _parse_response(raw)
    assert result["grade"] == "incorrect"
    print("PASS: parse JSON embedded in surrounding text")


def test_invalid_grade_defaults_to_incorrect():
    raw = json.dumps({
        "grade": "perfect",
        "feedback": "Great!",
        "missing_points": [],
        "confidence": 1.0,
    })
    result = _parse_response(raw)
    assert result["grade"] == "incorrect"
    print("PASS: unrecognised grade value normalised to incorrect")


def test_malformed_json_raises_value_error():
    raw = "The answer is correct. No JSON here at all."
    raised = False
    try:
        _parse_response(raw)
    except (ValueError, Exception):
        raised = True
    assert raised, "Expected an exception for completely unparseable output"
    print("PASS: malformed response with no JSON raises an exception")


def test_all_valid_grades_accepted():
    for grade in VALID_GRADES:
        raw = json.dumps({
            "grade": grade,
            "feedback": "Some feedback.",
            "missing_points": [],
            "confidence": 0.5,
        })
        result = _parse_response(raw)
        assert result["grade"] == grade
    print("PASS: all three valid grades accepted by parser")


def run_offline_tests():
    test_prompt_contains_question_expected_transcript()
    test_system_prompt_contains_rubric_grades()
    test_system_prompt_rejects_vague_as_correct()
    test_system_prompt_requires_explanation_when_incorrect()
    test_system_prompt_prefers_partial_over_incorrect()
    test_parse_valid_json()
    test_parse_partially_correct()
    test_parse_json_with_code_fence()
    test_parse_json_embedded_in_text()
    test_invalid_grade_defaults_to_incorrect()
    test_malformed_json_raises_value_error()
    test_all_valid_grades_accepted()
    print("\nAll offline tests passed.")


# ---------------------------------------------------------------------------
# Live tests — require Ollama running
# ---------------------------------------------------------------------------

_LIVE_CASES = [
    {
        "label": "correct — clear paraphrase",
        "question": "What is the powerhouse of the cell?",
        "expected": "The mitochondria is the powerhouse of the cell.",
        "transcript": "The mitochondria produces energy for the cell in the form of ATP.",
        "expect_grade": "correct",
    },
    {
        "label": "partially_correct — vague but directional",
        "question": "What is the powerhouse of the cell?",
        "expected": "The mitochondria is the powerhouse of the cell.",
        "transcript": "It's some organelle that deals with energy.",
        "expect_grade": "partially_correct",
    },
    {
        "label": "incorrect — unrelated answer",
        "question": "What is the powerhouse of the cell?",
        "expected": "The mitochondria is the powerhouse of the cell.",
        "transcript": "The nucleus controls all cell activity.",
        "expect_grade": "incorrect",
    },
]


def run_live_tests():
    print("\n--- Live Grading Tests ---")
    failures = []
    for case in _LIVE_CASES:
        print(f"\n  [{case['label']}]")
        print(f"  Transcript: {case['transcript']}")
        result = grade_response(case["question"], case["expected"], case["transcript"])
        icon = "✓" if result["grade"] == case["expect_grade"] else "✗"
        print(f"  Grade     : {result['grade']}  (expected: {case['expect_grade']}) {icon}")
        print(f"  Feedback  : {result['feedback']}")
        print(f"  Missing   : {result['missing_points']}")
        print(f"  Latency   : {result['latency_s']}s")
        if result["grade"] != case["expect_grade"]:
            failures.append(case["label"])

    if failures:
        print(f"\nLive test mismatches: {failures}")
        print("(Small models may not always match expected grades — review feedback for quality.)")
    else:
        print("\nAll live tests matched expected grades.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live tests against Ollama")
    args = parser.parse_args()

    run_offline_tests()

    if args.live:
        run_live_tests()
