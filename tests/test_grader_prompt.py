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

from tutor.grader import build_prompt, _parse_response, grade_response, VALID_GRADES


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------

def test_build_prompt_contains_all_fields():
    prompt = build_prompt(
        question="What is photosynthesis?",
        expected="The process by which plants convert sunlight into glucose.",
        transcript="Plants use sunlight to make food.",
    )
    assert "What is photosynthesis?" in prompt
    assert "convert sunlight into glucose" in prompt
    assert "Plants use sunlight to make food." in prompt
    print("PASS: build_prompt contains all fields")


def test_parse_valid_json():
    raw = json.dumps({
        "grade": "correct",
        "feedback": "Good answer.",
        "missing_points": [],
        "confidence": 0.9,
    })
    result = _parse_response(raw)
    assert result["grade"] == "correct"
    assert result["confidence"] == 0.9
    print("PASS: parse valid JSON")


def test_parse_json_with_code_fence():
    raw = '```json\n{"grade": "partially_correct", "feedback": "Close.", "missing_points": ["detail"], "confidence": 0.6}\n```'
    result = _parse_response(raw)
    assert result["grade"] == "partially_correct"
    print("PASS: parse JSON wrapped in code fence")


def test_parse_json_embedded_in_text():
    raw = 'Here is my assessment: {"grade": "incorrect", "feedback": "Wrong.", "missing_points": [], "confidence": 0.1} Hope that helps.'
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
    print("PASS: invalid grade value defaults to incorrect")


def run_offline_tests():
    test_build_prompt_contains_all_fields()
    test_parse_valid_json()
    test_parse_json_with_code_fence()
    test_parse_json_embedded_in_text()
    test_invalid_grade_defaults_to_incorrect()
    print("\nAll offline tests passed.")


# ---------------------------------------------------------------------------
# Live test — requires Ollama running
# ---------------------------------------------------------------------------

def run_live_test():
    question = "What is the powerhouse of the cell?"
    expected = "The mitochondria is the powerhouse of the cell."
    transcript = "The mitochondria makes energy for the cell."

    print(f"\n--- Live Grading Test ---")
    print(f"Question  : {question}")
    print(f"Expected  : {expected}")
    print(f"Transcript: {transcript}\n")

    result = grade_response(question, expected, transcript)

    print(f"Grade     : {result['grade']}")
    print(f"Feedback  : {result['feedback']}")
    print(f"Missing   : {result['missing_points']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Latency   : {result['latency_s']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live test against Ollama")
    args = parser.parse_args()

    run_offline_tests()

    if args.live:
        run_live_test()
