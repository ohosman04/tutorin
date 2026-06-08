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
        expected="Plants convert sunlight into glucose via chlorophyll.",
        transcript="Plants use sunlight to make food.",
    )
    assert "What is photosynthesis?" in prompt
    assert "sunlight into glucose" in prompt
    assert "Plants use sunlight to make food." in prompt
    print("PASS: build_prompt contains question, expected answer, and transcript")


def test_system_prompt_contains_rubric_grades():
    for grade in ("correct", "partially_correct", "incorrect"):
        assert grade in _SYSTEM, f"Rubric missing grade: {grade}"
    print("PASS: system prompt contains all three rubric grades")


def test_system_prompt_has_three_step_evaluation():
    assert "Step 1" in _SYSTEM
    assert "Step 2" in _SYSTEM
    assert "Step 3" in _SYSTEM
    print("PASS: system prompt contains three-step evaluation process")


def test_system_prompt_rejects_topical_relevance_alone():
    lower = _SYSTEM.lower()
    assert "topical relevance" in lower or "same topic" in lower or "same subject" in lower, (
        "Prompt should warn against rewarding topical relevance alone"
    )
    print("PASS: system prompt explicitly rejects topical relevance as sufficient for correct")


def test_system_prompt_rejects_vague_as_correct():
    lower = _SYSTEM.lower()
    assert "vague" in lower
    assert "partially_correct" in _SYSTEM
    print("PASS: system prompt warns against marking vague answers as correct")


def test_system_prompt_requires_explanation_in_feedback():
    lower = _SYSTEM.lower()
    assert "explain" in lower or "name the missing" in lower
    print("PASS: system prompt requires explanation in feedback")


def test_system_prompt_has_conservative_bias():
    assert "uncertain" in _SYSTEM.lower(), "Prompt should state tie-breaking rules for uncertainty"
    assert "partially_correct over correct" in _SYSTEM or \
           "uncertain between correct and partially_correct" in _SYSTEM.lower(), (
        "Prompt should prefer partially_correct over correct when uncertain"
    )
    print("PASS: system prompt has conservative bias / tie-breaking rules")


def test_system_prompt_has_few_shot_examples():
    # All three canonical examples should be present
    assert "TCP" in _SYSTEM, "TCP few-shot example missing from system prompt"
    assert "DNS" in _SYSTEM, "DNS few-shot examples missing from system prompt"
    # Verify both DNS examples are present (one correct, one partially_correct)
    assert _SYSTEM.count("DNS") >= 2
    print("PASS: system prompt contains few-shot examples (TCP, DNS x2)")


def test_system_prompt_few_shot_tcp_is_partially_correct():
    # The TCP example in the system prompt should be labelled partially_correct
    tcp_idx = _SYSTEM.find("TCP")
    tcp_section = _SYSTEM[tcp_idx:tcp_idx + 500]
    assert "partially_correct" in tcp_section, (
        "TCP few-shot example should demonstrate partially_correct grade"
    )
    print("PASS: TCP few-shot example is labelled partially_correct")


def test_system_prompt_few_shot_dns_paraphrase_is_correct():
    # The DNS paraphrase example should be labelled correct
    # Find the example with "website names"
    idx = _SYSTEM.find("website names")
    assert idx != -1, "DNS paraphrase example ('website names') not found"
    section = _SYSTEM[idx:idx + 300]
    assert "correct" in section
    print("PASS: DNS paraphrase few-shot example is labelled correct")


def test_system_prompt_few_shot_dns_vague_is_partially_correct():
    # "It helps the internet work" should be labelled partially_correct
    idx = _SYSTEM.find("helps the internet work")
    assert idx != -1, "DNS vague example ('helps the internet work') not found"
    section = _SYSTEM[idx:idx + 300]
    assert "partially_correct" in section
    print("PASS: DNS vague few-shot example is labelled partially_correct")


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------

def test_parse_valid_correct():
    raw = json.dumps({
        "grade": "correct",
        "feedback": "Your answer captures the core concept. The key mechanism is clearly stated.",
        "missing_points": [],
        "confidence": 0.92,
    })
    result = _parse_response(raw)
    assert result["grade"] == "correct"
    assert result["confidence"] == 0.92
    assert result["feedback"] != ""
    print("PASS: parse valid correct JSON")


def test_parse_partially_correct_with_missing_points():
    raw = json.dumps({
        "grade": "partially_correct",
        "feedback": "You identified the right process but missed the glucose output. The expected answer also required naming the light-dependent reactions.",
        "missing_points": ["glucose as output", "light-dependent reactions"],
        "confidence": 0.65,
    })
    result = _parse_response(raw)
    assert result["grade"] == "partially_correct"
    assert len(result["missing_points"]) == 2
    print("PASS: parse partially_correct with multiple missing_points")


def test_parse_json_with_code_fence():
    raw = (
        '```json\n'
        '{"grade": "partially_correct", "feedback": "Close but missing detail. '
        'The key mechanism was not mentioned.", "missing_points": ["key mechanism"], "confidence": 0.6}\n'
        '```'
    )
    result = _parse_response(raw)
    assert result["grade"] == "partially_correct"
    print("PASS: parse JSON wrapped in code fence")


def test_parse_json_embedded_in_surrounding_text():
    raw = (
        'Here is my assessment: '
        '{"grade": "incorrect", "feedback": "The answer is unrelated. '
        'The expected answer described DNS mapping.", "missing_points": [], "confidence": 0.1}'
        ' Hope that helps.'
    )
    result = _parse_response(raw)
    assert result["grade"] == "incorrect"
    print("PASS: parse JSON embedded in surrounding text")


def test_invalid_grade_normalised_to_incorrect():
    raw = json.dumps({
        "grade": "excellent",
        "feedback": "Great answer!",
        "missing_points": [],
        "confidence": 1.0,
    })
    result = _parse_response(raw)
    assert result["grade"] == "incorrect"
    print("PASS: unrecognised grade value normalised to incorrect")


def test_malformed_json_raises():
    raw = "The answer is correct. Definitely correct. No JSON at all."
    raised = False
    try:
        _parse_response(raw)
    except Exception:
        raised = True
    assert raised, "Expected an exception for completely unparseable output"
    print("PASS: response with no JSON object raises exception")


def test_all_valid_grades_accepted():
    for grade in VALID_GRADES:
        raw = json.dumps({
            "grade": grade,
            "feedback": "Sentence one. Sentence two.",
            "missing_points": [],
            "confidence": 0.5,
        })
        result = _parse_response(raw)
        assert result["grade"] == grade
    print("PASS: all three valid grades accepted by parser")


def test_missing_points_defaults_to_empty_list():
    raw = json.dumps({
        "grade": "correct",
        "feedback": "Great answer. All concepts present.",
        "confidence": 0.9,
        # missing_points field omitted
    })
    result = _parse_response(raw)
    assert result["missing_points"] == []
    print("PASS: missing missing_points field defaults to empty list")


def run_offline_tests():
    test_prompt_contains_question_expected_transcript()
    test_system_prompt_contains_rubric_grades()
    test_system_prompt_has_three_step_evaluation()
    test_system_prompt_rejects_topical_relevance_alone()
    test_system_prompt_rejects_vague_as_correct()
    test_system_prompt_requires_explanation_in_feedback()
    test_system_prompt_has_conservative_bias()
    test_system_prompt_has_few_shot_examples()
    test_system_prompt_few_shot_tcp_is_partially_correct()
    test_system_prompt_few_shot_dns_paraphrase_is_correct()
    test_system_prompt_few_shot_dns_vague_is_partially_correct()
    test_parse_valid_correct()
    test_parse_partially_correct_with_missing_points()
    test_parse_json_with_code_fence()
    test_parse_json_embedded_in_surrounding_text()
    test_invalid_grade_normalised_to_incorrect()
    test_malformed_json_raises()
    test_all_valid_grades_accepted()
    test_missing_points_defaults_to_empty_list()
    print("\nAll offline tests passed.")


# ---------------------------------------------------------------------------
# Live tests — require Ollama running
# ---------------------------------------------------------------------------

_LIVE_CASES = [
    {
        "label": "partially_correct — TCP canonical case (the original failure)",
        "question": "What happens when a TCP server receives a connection-request segment?",
        "expected": "The server locates the process waiting on the specified port, creates a new socket, and uses the four values from the segment to identify this socket.",
        "transcript": "It internally accepts this connection and sends this acceptance acknowledgement back to the sender.",
        "expect_grade": "partially_correct",
    },
    {
        "label": "correct — DNS clear paraphrase",
        "question": "What is DNS?",
        "expected": "DNS translates human-readable domain names into IP addresses.",
        "transcript": "It converts website names into addresses computers can use.",
        "expect_grade": "correct",
    },
    {
        "label": "partially_correct — DNS vague/generic",
        "question": "What is DNS?",
        "expected": "DNS translates human-readable domain names into IP addresses.",
        "transcript": "It helps the internet work.",
        "expect_grade": "partially_correct",
    },
    {
        "label": "correct — mitochondria clear paraphrase",
        "question": "What is the powerhouse of the cell?",
        "expected": "The mitochondria is the powerhouse of the cell.",
        "transcript": "The mitochondria produces energy for the cell in the form of ATP.",
        "expect_grade": "correct",
    },
    {
        "label": "partially_correct — mitochondria vague",
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
    passed = 0
    failures = []
    for case in _LIVE_CASES:
        print(f"\n  [{case['label']}]")
        print(f"  Transcript: {case['transcript']}")
        result = grade_response(case["question"], case["expected"], case["transcript"])
        match = result["grade"] == case["expect_grade"]
        icon = "✓" if match else "✗"
        print(f"  Grade     : {result['grade']}  (expected: {case['expect_grade']}) {icon}")
        print(f"  Feedback  : {result['feedback']}")
        if result["missing_points"]:
            print(f"  Missing   : {result['missing_points']}")
        print(f"  Latency   : {result['latency_s']}s")
        if match:
            passed += 1
        else:
            failures.append(case["label"])

    print(f"\n{passed}/{len(_LIVE_CASES)} live tests matched expected grades.")
    if failures:
        print("Mismatches:")
        for f in failures:
            print(f"  - {f}")
        print("(Small models are not deterministic — review feedback quality even on mismatches.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live tests against Ollama")
    args = parser.parse_args()

    run_offline_tests()

    if args.live:
        run_live_tests()
