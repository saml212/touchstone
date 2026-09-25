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
    agent_provider: str = "claude-cli"
    keychain_prefix: str = "touchstone-"
    keychain_openai: str = "openai-api-key"
    keychain_anthropic: str = "anthropic-api-key"
    stt: str = "none"
    tts: str = "browser"
    speech_mode: str = "local"  # local | realtime
    realtime_model: str = "gpt-realtime-2.1-mini"
    realtime_voice: str = "marin"
    # `harbor run` executes on this SSH host when the local machine has no Docker daemon; empty
    # means always run locally. `harbor_remote_root` is the directory on that host to sync into.
    harbor_host: str = ""
    harbor_remote_root: str = "~/touchstone-harbor"
    # Survey (read-only code-mapping agent): which CLI provider runs it, its model, the simulator
    # fidelity threshold, full names to scrub from recordings, and an optional interpreter that
    # replaces `uv run --project <repo>` when replaying tool calls (tests set it to skip uv).
    survey_provider: str = "claude-cli"
    survey_model: str = ""
    survey_fidelity_threshold: float = 0.8
    survey_names: list = field(default_factory=list)
    survey_python: str = ""
    survey_dataset_name: str = ""  # dataset.toml name; defaults to "<repo>/<repo>" when empty
    # Review room: the dataset directory (under the project root) the room reviews, and the Harbor
    # jobs directory it reads trials from (empty -> "<dataset>/jobs").
    review_dataset: str = "touchstone"
    review_jobs_dir: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def db(self) -> Path:
        return Path(self.db_path).expanduser()

    @property
    def root(self) -> Path:
        """Project root holding tasks/, checks.toml, benchmarks/ — beside `.touchstone/`."""
        db = self.db
        return db.parent.parent if db.parent.name == ".touchstone" else db.parent

    @property
    def review_dataset_dir(self) -> Path:
        """The dataset directory the review room reads (project root / review_dataset)."""
        return self.root / self.review_dataset

    @property
    def review_jobs(self) -> Path:
        """The Harbor jobs directory the review room reads trials from."""
        return Path(self.review_jobs_dir).expanduser() if self.review_jobs_dir \
            else self.review_dataset_dir / "jobs"

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
    "TOUCHSTONE_SPEECH_MODE": "speech_mode",
    "TOUCHSTONE_HARBOR_HOST": "harbor_host",
    "TOUCHSTONE_HARBOR_REMOTE_ROOT": "harbor_remote_root",
    "TOUCHSTONE_SURVEY_PROVIDER": "survey_provider",
    "TOUCHSTONE_SURVEY_MODEL": "survey_model",
    "TOUCHSTONE_SURVEY_PYTHON": "survey_python",
    "TOUCHSTONE_REVIEW_DATASET": "review_dataset",
    "TOUCHSTONE_REVIEW_JOBS_DIR": "review_jobs_dir",
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
    s.speech_mode = speech.get("mode", s.speech_mode)
    s.realtime_model = speech.get("realtime_model", s.realtime_model)
    s.realtime_voice = speech.get("realtime_voice", s.realtime_voice)
    harbor = data.get("harbor", {})
    s.harbor_host = harbor.get("host", s.harbor_host)
    s.harbor_remote_root = harbor.get("remote_root", s.harbor_remote_root)
    survey = data.get("survey", {})
    s.survey_provider = survey.get("provider", s.survey_provider)
    s.survey_model = survey.get("model", s.survey_model)
    s.survey_fidelity_threshold = survey.get("fidelity_threshold", s.survey_fidelity_threshold)
    s.survey_names = survey.get("names", s.survey_names)
    s.survey_python = survey.get("python", s.survey_python)
    s.survey_dataset_name = survey.get("dataset_name", s.survey_dataset_name)
    review = data.get("review", {})
    s.review_dataset = review.get("dataset", s.review_dataset)
    s.review_jobs_dir = review.get("jobs_dir", s.review_jobs_dir)
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
