"""
Usage:
    python tests/test_llm_generate.py "I have been studying Python for three months."

Override defaults:
    OLLAMA_URL=http://192.168.1.100:11434 OLLAMA_MODEL=qwen2.5:1.5b \\
        python tests/test_llm_generate.py "your prompt here"
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clients.llm_client import OLLAMA_MODEL, OLLAMA_URL, generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Test Ollama LLM generation")
    parser.add_argument("prompt", help="Prompt text to send to the LLM")
    args = parser.parse_args()

    print(f"URL   : {OLLAMA_URL}")
    print(f"Model : {OLLAMA_MODEL}")
    print(f"Prompt: {args.prompt}")
    print()

    result = generate(args.prompt)

    print("--- Response ---")
    print(result.get("response", "").strip())
    print()
    print("--- Timing ---")
    print(f"Wall latency          : {result['latency_s']}s")

    # Ollama returns nanosecond counters when stream=false
    for key, label in [
        ("total_duration", "total_duration (ns)"),
        ("load_duration", "load_duration (ns)"),
        ("prompt_eval_duration", "prompt_eval_duration (ns)"),
        ("eval_duration", "eval_duration (ns)"),
        ("eval_count", "tokens generated"),
    ]:
        if key in result:
            print(f"{label:<30}: {result[key]}")


if __name__ == "__main__":
    main()
