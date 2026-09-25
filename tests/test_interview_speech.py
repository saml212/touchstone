import shutil
import sys

import pytest

from touchstone.config import Settings
from touchstone.interview import speech
from touchstone.interview.speech import (
    BrowserTTS,
    FasterWhisperSTT,
    NoneSTT,
    OpenAISTT,
    SayTTS,
    Speech,
    SpeechError,
    validate_audio,
)

WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 40
OGG = b"OggS" + b"\x00" * 40
WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40
MP4 = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 40


@pytest.mark.parametrize("data,mime", [(WEBM, "audio/webm"), (OGG, "audio/ogg"),
                                       (WAV, "audio/wav"), (MP4, "audio/mp4")])
def test_validate_audio_accepts_known_containers(data, mime):
    assert validate_audio(data) == mime


def test_validate_audio_rejects_unknown_bytes():
    with pytest.raises(SpeechError):
        validate_audio(b"this is a text file, not audio")


def test_validate_audio_rejects_empty_and_oversized():
    with pytest.raises(SpeechError):
        validate_audio(b"")
    with pytest.raises(SpeechError):
        validate_audio(WEBM[:4] + b"\x00" * (speech.MAX_AUDIO_BYTES + 1))


def test_none_stt_raises_a_clear_error():
    with pytest.raises(SpeechError):
        NoneSTT().transcribe(WEBM, "audio/webm")


def _settings(stt="auto", tts="auto"):
    return Settings(stt=stt, tts=tts)


def test_auto_prefers_faster_whisper_when_importable(monkeypatch):
    monkeypatch.setattr(speech, "_whisper_importable", lambda: True)
    stt, res = speech.resolve_stt(_settings())
    assert isinstance(stt, FasterWhisperSTT)
    assert res.name == "faster-whisper"
    assert "150MB" in res.detail


def test_auto_falls_back_to_openai_when_key_present(monkeypatch):
    monkeypatch.setattr(speech, "_whisper_importable", lambda: False)
    monkeypatch.setattr(speech, "_openai_key", lambda s: "sk-test")
    stt, res = speech.resolve_stt(_settings())
    assert isinstance(stt, OpenAISTT)
    assert res.name == "openai"


def test_auto_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(speech, "_whisper_importable", lambda: False)
    monkeypatch.setattr(speech, "_openai_key", lambda s: None)
    stt, res = speech.resolve_stt(_settings())
    assert isinstance(stt, NoneSTT)
    assert res.name == "none"


def test_tts_auto_is_browser():
    tts, res = speech.resolve_tts(_settings())
    assert isinstance(tts, BrowserTTS)
    assert res.name == "browser"
    assert tts.synthesize("hi") is None


def test_explicit_faster_whisper_without_extra_degrades(monkeypatch):
    monkeypatch.setattr(speech, "_whisper_importable", lambda: False)
    stt, res = speech.resolve_stt(_settings(stt="faster-whisper"))
    assert isinstance(stt, NoneSTT)
    assert "whisper" in res.detail.lower()


def test_from_settings_and_status(monkeypatch):
    monkeypatch.setattr(speech, "_whisper_importable", lambda: False)
    monkeypatch.setattr(speech, "_openai_key", lambda s: None)
    sp = Speech.from_settings(_settings())
    assert isinstance(sp.stt, NoneSTT) and isinstance(sp.tts, BrowserTTS)
    kinds = {r.kind for r in speech.speech_status(_settings())}
    assert kinds == {"mode", "stt", "tts"}


def test_realtime_status_reports_mode_and_key(monkeypatch):
    import touchstone.config as config
    monkeypatch.setattr(config, "_openai_key", lambda s: None)
    auto_local = speech.realtime_status(_settings())  # default mode is now "auto"
    assert auto_local.name == "auto → local" and "no OPENAI_API_KEY" in auto_local.detail
    explicit_local = speech.realtime_status(Settings(speech_mode="local"))
    assert explicit_local.name == "local → local" and "no OPENAI_API_KEY" in explicit_local.detail
    monkeypatch.setattr(config, "_openai_key", lambda s: "sk-test")
    auto_rt = speech.realtime_status(_settings())  # auto + key -> realtime
    assert auto_rt.name == "auto → realtime" and "gpt-realtime" in auto_rt.detail
    rt = speech.realtime_status(Settings(speech_mode="realtime", realtime_model="gpt-realtime-2.1"))
    assert rt.name == "realtime → realtime" and "gpt-realtime-2.1" in rt.detail


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("say") is None,
    reason="requires macOS `say`",
)
def test_say_tts_synthesizes_real_audio():
    audio, mime = SayTTS().synthesize("hello")
    assert mime == "audio/aiff"
    assert audio.startswith(b"FORM")  # AIFF magic


def test_openai_stt_transcribes_with_a_stubbed_endpoint(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {"text": "  transcribed words  "}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(speech.httpx, "Client", lambda *a, **k: FakeClient())
    assert OpenAISTT("sk-test").transcribe(WEBM, "audio/webm") == "transcribed words"
