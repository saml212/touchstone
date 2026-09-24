"""Settings from touchstone.toml + TOUCHSTONE_* env vars. Env overrides file overrides defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = "./.touchstone/touchstone.db"


@dataclass
class Settings:
    db_path: str = DEFAULT_DB
    provider: str = "scripted"
    agent_provider: str = "codex-cli"
    keychain_prefix: str = "rockie-"
    keychain_openai: str = "openai-api-key"
    keychain_anthropic: str = "anthropic-api-key"
    stt: str = "none"
    tts: str = "browser"
    extra: dict = field(default_factory=dict)

    @property
    def db(self) -> Path:
        return Path(self.db_path).expanduser()

    def keychain_service(self, name: str) -> str:
        """Full service for a bare name: 'openai-api-key' -> 'rockie-openai-api-key'."""
        return f"{self.keychain_prefix}{name}"


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_settings(path: str | Path | None = None) -> Settings:
    toml_path = Path(path) if path else Path("touchstone.toml")
    data = _load_toml(toml_path)
    s = Settings()
    if "db_path" in data:
        s.db_path = data["db_path"]
    if "provider" in data:
        s.provider = data["provider"]
    if "agent_provider" in data:
        s.agent_provider = data["agent_provider"]
    if "keychain_prefix" in data:
        s.keychain_prefix = data["keychain_prefix"]
    keychain = data.get("keychain", {})
    s.keychain_openai = keychain.get("openai", s.keychain_openai)
    s.keychain_anthropic = keychain.get("anthropic", s.keychain_anthropic)
    speech = data.get("speech", {})
    s.stt = speech.get("stt", s.stt)
    s.tts = speech.get("tts", s.tts)
    s.extra = data

    if v := os.environ.get("TOUCHSTONE_DB"):
        s.db_path = v
    if v := os.environ.get("TOUCHSTONE_PROVIDER"):
        s.provider = v
    if v := os.environ.get("TOUCHSTONE_AGENT_PROVIDER"):
        s.agent_provider = v
    if v := os.environ.get("TOUCHSTONE_KEYCHAIN_PREFIX"):
        s.keychain_prefix = v
    if v := os.environ.get("TOUCHSTONE_STT"):
        s.stt = v
    if v := os.environ.get("TOUCHSTONE_TTS"):
        s.tts = v
    return s
