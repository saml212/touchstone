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
    keychain_prefix: str = "touchstone-"
    keychain_openai: str = "openai-api-key"
    keychain_anthropic: str = "anthropic-api-key"
    stt: str = "none"
    tts: str = "browser"
    extra: dict = field(default_factory=dict)

    @property
    def db(self) -> Path:
        return Path(self.db_path).expanduser()

    @property
    def root(self) -> Path:
        """Project root holding tasks/, checks.toml, benchmarks/ — beside `.touchstone/`."""
        db = self.db
        return db.parent.parent if db.parent.name == ".touchstone" else db.parent

    def keychain_service(self, name: str) -> str:
        """Full service for a bare name: 'openai-api-key' -> 'touchstone-openai-api-key'."""
        return f"{self.keychain_prefix}{name}"


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


_TOML_KEYS = ("db_path", "provider", "agent_provider", "keychain_prefix")
_ENV_KEYS = {
    "TOUCHSTONE_DB": "db_path",
    "TOUCHSTONE_PROVIDER": "provider",
    "TOUCHSTONE_AGENT_PROVIDER": "agent_provider",
    "TOUCHSTONE_KEYCHAIN_PREFIX": "keychain_prefix",
    "TOUCHSTONE_STT": "stt",
    "TOUCHSTONE_TTS": "tts",
}


def _apply_toml(s: Settings, data: dict) -> None:
    for key in _TOML_KEYS:
        if key in data:
            setattr(s, key, data[key])
    keychain = data.get("keychain", {})
    s.keychain_openai = keychain.get("openai", s.keychain_openai)
    s.keychain_anthropic = keychain.get("anthropic", s.keychain_anthropic)
    speech = data.get("speech", {})
    s.stt = speech.get("stt", s.stt)
    s.tts = speech.get("tts", s.tts)
    s.extra = data


def _apply_env(s: Settings) -> None:
    for env, attr in _ENV_KEYS.items():
        if v := os.environ.get(env):
            setattr(s, attr, v)


def load_settings(path: str | Path | None = None) -> Settings:
    toml_path = Path(path) if path else Path("touchstone.toml")
    s = Settings()
    _apply_toml(s, _load_toml(toml_path))
    _apply_env(s)
    return s
