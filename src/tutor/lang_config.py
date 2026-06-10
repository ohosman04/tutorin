import os

# Known language configurations.
# piper_model paths use common Piper voice naming conventions — adjust to
# whichever voice files you have installed on the device.
LANGUAGE_CONFIGS = {
    "en": {
        "stt_language": "en",
        "piper_model": "~/piper/models/en_US-lessac-medium.onnx",
        "feedback_language": "English",
    },
    "es": {
        "stt_language": "es",
        "piper_model": "~/piper/models/es_ES-davefx-medium.onnx",
        "feedback_language": "Spanish",
    },
    "fr": {
        "stt_language": "fr",
        "piper_model": "~/piper/models/fr_FR-siwis-medium.onnx",
        "feedback_language": "French",
    },
    "de": {
        "stt_language": "de",
        "piper_model": "~/piper/models/de_DE-thorsten-medium.onnx",
        "feedback_language": "German",
    },
    "it": {
        "stt_language": "it",
        "piper_model": "~/piper/models/it_IT-riccardo-x_low.onnx",
        "feedback_language": "Italian",
    },
    "pt": {
        "stt_language": "pt",
        "piper_model": "~/piper/models/pt_PT-tugao-medium.onnx",
        "feedback_language": "Portuguese",
    },
    "zh": {
        "stt_language": "zh",
        "piper_model": "~/piper/models/zh_CN-huayan-medium.onnx",
        "feedback_language": "Chinese",
    },
    "ja": {
        "stt_language": "ja",
        "piper_model": "~/piper/models/ja_JP-kokoro-medium.onnx",
        "feedback_language": "Japanese",
    },
}


def resolve_lang_config(
    language: str | None = None,
    stt_language: str | None = None,
    piper_model: str | None = None,
) -> dict:
    """
    Resolve effective language config from CLI flags.

    Priority: explicit CLI flags > language-specific defaults > built-in defaults.

    Returns dict with keys:
      stt_language   — BCP-47 language tag for the STT service
      piper_model    — expanded absolute path to Piper .onnx model, or None (use env/default)
      feedback_language — full language name for LLM prompts (e.g. "Spanish")
    """
    base = LANGUAGE_CONFIGS.get(language, {}) if language else {}

    resolved_piper = piper_model or base.get("piper_model") or None

    return {
        "stt_language": stt_language or base.get("stt_language") or "en",
        "piper_model": os.path.expanduser(resolved_piper) if resolved_piper else None,
        "feedback_language": base.get("feedback_language") or "English",
    }
