# TutOrin

An AI-driven, voice-first tutoring agent that runs entirely at the edge on an NVIDIA Jetson. TutOrin loads flashcard decks, asks questions aloud, listens to spoken answers, grades them with a local LLM, and gives real-time feedback — no cloud required.

## The Problem

Students often struggle with complex topics and need personalized, interactive help — not just static flashcards. Traditional study tools lack the ability to evaluate *how* you explain something out loud, or to adapt when you get something wrong.

## The Solution

TutOrin is a conversational tutoring agent that turns any Anki deck into an oral exam. It uses speech-to-text, local reasoning, and text-to-speech to create a natural study loop you can run anywhere, anytime.

## How It Works

```
Agent picks card → Asks question → User answers (voice)
       ↑                                      ↓
User answers follow-up ← Follow-up question ← Agent grades response
```

1. **Pick a card** — TutOrin selects the next card from your Anki deck.
2. **Ask a question** — The question is displayed (and optionally spoken aloud).
3. **Listen** — You answer by voice; the agent records and transcribes your response.
4. **Grade** — A local LLM evaluates your answer against the expected response.
5. **Follow up** — On incorrect answers, the agent can ask a targeted follow-up question.
6. **Repeat** — Session progress is saved so you can resume where you left off.

## Architecture

TutOrin is designed to run fully on-device on an **NVIDIA Jetson Orin**, with each component playing a distinct role:

| Component | Role | Technology |
|-----------|------|------------|
| **Ear** (input) | Speech-to-text | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) |
| **Mouth** (output) | Text-to-speech | [Piper TTS](https://github.com/rhasspy/piper) |
| **Tutor** (reasoning) | Grading & feedback | [Qwen 2.5 1.5B](https://ollama.com/library/qwen2.5) via Ollama |
| **Agent** (orchestration) | Deck loading, session flow, conversation control | Python CLI (`app.py`) |

```
Speech → Reasoning → Feedback   (entirely at the edge)
```

## Features

- **Anki deck support** — Load any `.apkg` file and quiz yourself on its cards.
- **Voice-based answers** — Record with auto-stop on silence, press-Enter, or fixed duration.
- **LLM grading** — Paraphrase-aware evaluation with correct / partially correct / incorrect labels.
- **Spoken feedback** — Optional TTS for questions and grading feedback.
- **Adaptive follow-ups** — Targeted follow-up questions when you miss a concept.
- **Session persistence** — Resume a deck session across runs.
- **Multi-language** — Built-in support for English, Spanish, French, and German (STT, TTS, and grader).
- **Freeform mode** — Diagnostic conversation loop for testing the full pipeline.

## Prerequisites

- **Hardware:** NVIDIA Jetson Orin (or any machine with a microphone; GPU accelerates STT)
- **Python:** 3.10+
- **Ollama** with `qwen2.5:1.5b` pulled
- **Piper TTS** voice models (for spoken output)
- An Anki deck (`.apkg` file)

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd tutorin

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Jetson
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull qwen2.5:1.5b

# Start the STT service (in a separate terminal)
cd services/stt
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000

# Or build and run the STT Docker container on Jetson
docker build -t jetson-stt:latest ./services/stt
docker run --rm --runtime nvidia --gpus all -p 8000:8000 jetson-stt:latest
```

For Piper TTS, download voice models from the [Piper releases page](https://github.com/rhasspy/piper/releases) and place them in `~/piper/models/`.

## Usage

### Quiz mode (primary)

```bash
# Basic quiz with an Anki deck
python app.py quiz --deck path/to/deck.apkg

# Inspect field names in a deck before running
python app.py quiz --deck deck.apkg --inspect-fields

# Full experience: spoken questions, spoken feedback, follow-ups, resumable session
python app.py quiz --deck deck.apkg \
    --speak-question \
    --speak-feedback \
    --followups \
    --resume

# Spanish deck with language-aware STT, TTS, and grading
python app.py quiz --deck spanish.apkg --language es \
    --speak-question --speak-feedback
```

### Freeform mode (diagnostic)

```bash
python app.py freeform
```

Records a short utterance, transcribes it, and sends it to the LLM for a conversational response. Useful for verifying that audio, STT, and Ollama are all working.

### Run flags reference

See [`run_flags.txt`](run_flags.txt) for the complete list of CLI flags, including recording modes, auto-silence tuning, session management, and multi-language overrides.

## Project Structure

```
tutorin/
├── app.py                  # CLI entry point (quiz + freeform modes)
├── src/
│   ├── anki_parser.py      # .apkg deck loader
│   ├── audio.py            # Microphone recording utilities
│   ├── clients/
│   │   ├── stt_client.py   # Faster Whisper HTTP client
│   │   ├── tts_client.py   # Piper TTS wrapper
│   │   └── llm_client.py   # Ollama HTTP client
│   └── tutor/
│       ├── quiz_session.py # Main quiz loop
│       ├── grader.py       # LLM-based answer grading
│       ├── followup.py     # Adaptive follow-up generation
│       ├── session_state.py# Session persistence
│       └── lang_config.py  # Multi-language resolution
├── services/
│   └── stt/                # FastAPI STT microservice (Faster Whisper)
└── tests/                  # Unit and integration tests
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `STT_URL` | `http://localhost:8000` | STT service endpoint |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | LLM model name |
| `PIPER_MODEL` | `~/piper/models/en_US-lessac-medium.onnx` | Piper TTS voice model |
| `WHISPER_MODEL` | `tiny.en` | Whisper model for STT server |
| `RECORD_MAX_DURATION` | `60` | Max recording length (seconds) |

## Running Tests

```bash
# Offline grader prompt test
python tests/test_grader_prompt.py

# Live grading (requires Ollama)
python tests/test_grader_prompt.py --live

# STT with a WAV file
python tests/test_stt_transcribe.py recording.wav

# LLM generation
python tests/test_llm_generate.py "I have been studying Python for three months."

# Full test suite
pytest
```

## Goal

Make education more accessible and personalized — giving every student a patient, always-available tutor that meets them where they are, entirely on local hardware.

## License

No license has been specified yet.
