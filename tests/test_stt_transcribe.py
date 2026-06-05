"""
Usage:
    python tests/test_stt_transcribe.py <path-to-wav>

Requires a running STT service. Override the default URL with:
    STT_URL=http://<host>:8000 python tests/test_stt_transcribe.py recording.wav
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clients.stt_client import transcribe_wav

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Test STT transcription service")
    parser.add_argument("wav_file", help="Path to WAV file to transcribe")
    args = parser.parse_args()

    if not os.path.exists(args.wav_file):
        print(f"ERROR: file not found: {args.wav_file}")
        sys.exit(1)

    print(f"Sending: {args.wav_file}")
    result = transcribe_wav(args.wav_file)

    print("\n--- Result ---")
    print(f"Transcript : {result['transcript']}")
    print(f"Device     : {result['device']}")
    print(f"Audio      : {result['audio_duration_s']}s")
    print(f"RTF        : {result['transcription_time_s']}s transcription time")


if __name__ == "__main__":
    main()
