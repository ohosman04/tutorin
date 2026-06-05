import os
import sys
import wave

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio import CHANNELS, SAMPLE_RATE, record_audio


TMP_WAV = "/tmp/test_recording.wav"


def test_record_creates_wav_file(tmp_path):
    out = str(tmp_path / "out.wav")
    record_audio(duration=1.0, output_path=out)
    assert os.path.exists(out), "WAV file was not created"


def test_wav_properties(tmp_path):
    out = str(tmp_path / "out.wav")
    record_audio(duration=1.0, output_path=out)

    with wave.open(out, "r") as wf:
        assert wf.getnchannels() == CHANNELS
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getnframes() > 0


def test_record_default_duration(tmp_path):
    out = str(tmp_path / "default.wav")
    record_audio(output_path=out)

    with wave.open(out, "r") as wf:
        duration = wf.getnframes() / wf.getframerate()
    assert abs(duration - 5.0) < 0.5


def test_missing_device_raises_runtime_error(monkeypatch):
    import sounddevice as sd

    def mock_rec(*args, **kwargs):
        raise sd.PortAudioError("No device")

    monkeypatch.setattr(sd, "rec", mock_rec)

    with pytest.raises(RuntimeError, match="Microphone error"):
        record_audio(duration=1.0)
