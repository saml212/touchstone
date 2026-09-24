"""Speech: STT + TTS providers, auto-detection, and audio-upload validation.

Text-only always works; speech is additive. STT turns an uploaded clip into a message; TTS turns an
agent reply into audio (or defers to the browser's speechSynthesis, the zero-dependency default).
Providers resolve from `[speech]` in the config, or auto-detect when a value is `"auto"`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from ..config import Settings
from ..llm._http import post_json
from ..llm.keychain import secret

MAX_AUDIO_BYTES = 25 * 1024 * 1024
_OPENAI_BASE = "https://api.openai.com/v1"


class SpeechError(RuntimeError):
    """A speech operation failed; the message is one safe sentence."""


@runtime_checkable
class STT(Protocol):
    def transcribe(self, audio: bytes, mime: str) -> str: ...


@runtime_checkable
class TTS(Protocol):
    def synthesize(self, text: str) -> tuple[bytes, str] | None: ...


# ---- audio validation ------------------------------------------------------

_MAGIC = [
    (b"\x1aE\xdf\xa3", "audio/webm"),  # EBML (webm / mkv)
    (b"OggS", "audio/ogg"),
    (b"RIFF", "audio/wav"),
]


def sniff_audio(data: bytes) -> str | None:
    """Return a mime type for a recognised audio container, else None."""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return None


def validate_audio(data: bytes) -> str:
    """Return the sniffed mime type, or raise SpeechError with one clear sentence."""
    if not data:
        raise SpeechError("The uploaded audio was empty.")
    if len(data) > MAX_AUDIO_BYTES:
        limit = MAX_AUDIO_BYTES // (1024 * 1024)
        raise SpeechError(f"The audio is larger than the {limit}MB limit.")
    mime = sniff_audio(data)
    if mime is None:
        raise SpeechError(
            "The upload is not a recognised audio file (expected webm, ogg, wav, or mp4)."
        )
    return mime


# ---- STT providers ---------------------------------------------------------


class NoneSTT:
    name = "none"

    def transcribe(self, audio: bytes, mime: str) -> str:
        raise SpeechError("Speech-to-text is disabled; type your message or set [speech] stt.")


@dataclass
class OpenAISTT:
    api_key: str
    model: str = "gpt-4o-mini-transcribe"
    timeout: float = 60.0
    name: str = "openai"

    def transcribe(self, audio: bytes, mime: str) -> str:
        ext = mime.split("/", 1)[-1]
        files = {"file": (f"audio.{ext}", audio, mime)}
        with httpx.Client() as client:
            try:
                resp = client.post(
                    f"{_OPENAI_BASE}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data={"model": self.model},
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise SpeechError("Could not reach the OpenAI transcription endpoint.") from exc
        if resp.status_code >= 400:
            raise SpeechError(f"Transcription failed with HTTP {resp.status_code}.")
        return (resp.json().get("text") or "").strip()


@dataclass
class FasterWhisperSTT:
    model_size: str = "base"
    name: str = "faster-whisper"
    _model: object = None

    def transcribe(self, audio: bytes, mime: str) -> str:
        from faster_whisper import WhisperModel  # lazy: heavy import, optional extra

        if self._model is None:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        with tempfile.NamedTemporaryFile(suffix="." + mime.split("/", 1)[-1]) as fh:
            fh.write(audio)
            fh.flush()
            segments, _ = self._model.transcribe(fh.name)
            return " ".join(seg.text.strip() for seg in segments).strip()


# ---- TTS providers ---------------------------------------------------------


class BrowserTTS:
    name = "browser"

    def synthesize(self, text: str) -> tuple[bytes, str] | None:
        return None  # the client speaks it with speechSynthesis


@dataclass
class SayTTS:
    name: str = "say"

    def synthesize(self, text: str) -> tuple[bytes, str] | None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speech.aiff"
            try:
                subprocess.run(["say", "-o", str(out), text], check=True, timeout=30,
                               capture_output=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise SpeechError("macOS `say` could not synthesize the audio.") from exc
            return out.read_bytes(), "audio/aiff"


@dataclass
class OpenAITTS:
    api_key: str
    model: str = "gpt-4o-mini-tts"
    voice: str = "marin"
    timeout: float = 60.0
    name: str = "openai"

    def synthesize(self, text: str) -> tuple[bytes, str] | None:
        with httpx.Client() as client:
            resp = post_json(
                client,
                f"{_OPENAI_BASE}/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                body={"model": self.model, "voice": self.voice, "input": text, "format": "mp3"},
                timeout=self.timeout,
                label="OpenAI speech request",
            )
        return resp.content, "audio/mpeg"


# ---- resolution ------------------------------------------------------------


@dataclass
class Resolution:
    kind: str  # "stt" | "tts"
    name: str
    detail: str


def _openai_key(settings: Settings) -> str | None:
    return secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai))


def _whisper_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def _auto_stt(settings: Settings) -> str:
    if _whisper_importable():
        return "faster-whisper"
    if _openai_key(settings):
        return "openai"
    return "none"


def _build_whisper_stt(reason: str) -> tuple[STT, Resolution]:
    if not _whisper_importable():
        return NoneSTT(), Resolution(
            "stt", "none", "faster-whisper not installed; `pip install touchstone[whisper]`"
        )
    detail = "model 'base' will download ~150MB on first use"
    return FasterWhisperSTT(), Resolution("stt", "faster-whisper", f"{reason}; {detail}")


def _build_openai_stt(settings: Settings, reason: str) -> tuple[STT, Resolution]:
    key = _openai_key(settings)
    if not key:
        return NoneSTT(), Resolution(
            "stt", "none", "openai selected but no OPENAI_API_KEY / keychain key"
        )
    return OpenAISTT(key), Resolution("stt", "openai", f"{reason}; gpt-4o-mini-transcribe")


def resolve_stt(settings: Settings) -> tuple[STT, Resolution]:
    choice = settings.stt
    if choice == "auto":
        choice = _auto_stt(settings)
        reason = f"auto -> {choice}"
    else:
        reason = "configured"

    if choice == "faster-whisper":
        return _build_whisper_stt(reason)
    if choice == "openai":
        return _build_openai_stt(settings, reason)
    detail = reason if choice == "none" else f"unknown stt {choice!r} -> none"
    return NoneSTT(), Resolution("stt", "none", detail)


def _build_say_tts(reason: str) -> tuple[TTS, Resolution]:
    if shutil.which("say") is None:
        return BrowserTTS(), Resolution(
            "tts", "browser", "say selected but not on PATH; using browser"
        )
    return SayTTS(), Resolution("tts", "say", f"{reason}; macOS `say`")


def _build_openai_tts(settings: Settings, reason: str) -> tuple[TTS, Resolution]:
    key = _openai_key(settings)
    if not key:
        return BrowserTTS(), Resolution(
            "tts", "browser", "openai selected but no key; using browser"
        )
    return OpenAITTS(key), Resolution("tts", "openai", f"{reason}; gpt-4o-mini-tts voice marin")


def resolve_tts(settings: Settings) -> tuple[TTS, Resolution]:
    choice = settings.tts
    reason = "auto -> browser" if choice == "auto" else "configured"
    if choice == "auto":
        choice = "browser"

    if choice == "say":
        return _build_say_tts(reason)
    if choice == "openai":
        return _build_openai_tts(settings, reason)
    detail = reason if choice == "browser" else f"unknown tts {choice!r} -> browser"
    return BrowserTTS(), Resolution("tts", "browser", detail)


@dataclass
class Speech:
    stt: STT
    tts: TTS
    stt_resolution: Resolution
    tts_resolution: Resolution

    @classmethod
    def from_settings(cls, settings: Settings) -> Speech:
        stt, stt_res = resolve_stt(settings)
        tts, tts_res = resolve_tts(settings)
        return cls(stt=stt, tts=tts, stt_resolution=stt_res, tts_resolution=tts_res)


def realtime_status(settings: Settings) -> Resolution:
    """For `doctor`: the speech mode and whether realtime is reachable (key present), no network."""
    has_key = _openai_key(settings) is not None
    if settings.speech_mode == "realtime":
        avail = settings.realtime_model if has_key else "no OPENAI_API_KEY — falls back to local"
        return Resolution("mode", "realtime", avail)
    avail = "available (key present)" if has_key else "needs OPENAI_API_KEY"
    return Resolution("mode", "local", f"realtime {avail}")


def speech_status(settings: Settings) -> list[Resolution]:
    """For `doctor`: the mode plus which STT/TTS resolved and why, without loading heavy models."""
    return [realtime_status(settings), resolve_stt(settings)[1], resolve_tts(settings)[1]]


__all__ = [
    "STT", "TTS", "Speech", "SpeechError", "Resolution",
    "NoneSTT", "OpenAISTT", "FasterWhisperSTT",
    "BrowserTTS", "SayTTS", "OpenAITTS",
    "validate_audio", "sniff_audio", "speech_status", "realtime_status",
    "resolve_stt", "resolve_tts", "MAX_AUDIO_BYTES",
]
