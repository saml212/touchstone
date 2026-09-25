"""The survey provider: run a read-only coding agent over a repo and return its text answer.

Unlike the chat providers in `touchstone/llm` (which return a structured `Reply`), a survey run is
one prompt executed with file-reading tools inside the repo, answering with a JSON document. The
claude-cli and codex-cli implementations shell out to the logged-in CLI (the customer's
subscription pays); `ScriptedSurveyProvider` returns canned answers for tests and the zero-key path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Settings
from ..llm import _cli

_ALLOWED_TOOLS = "Read,Grep,Glob"
_SURVEY_TIMEOUT = 900.0


@runtime_checkable
class SurveyProvider(Protocol):
    name: str

    def run(self, prompt: str, cwd: Path) -> str:
        """Answer `prompt` after reading the repo at `cwd`; return the final message text."""
        ...


class ClaudeCLISurveyProvider:
    """`claude -p` with Read/Grep/Glob and cwd=repo (subscription, never an API key)."""

    def __init__(self, model: str | None = None, *, timeout: float = _SURVEY_TIMEOUT) -> None:
        _cli.require_binary("claude", "The `claude` CLI is not on PATH; install Claude Code.")
        self.model = model
        self.timeout = timeout
        self.name = f"claude-cli:{model}" if model else "claude-cli"

    def _cmd(self, prompt: str) -> list[str]:
        cmd = ["claude", "-p", prompt, "--allowedTools", _ALLOWED_TOOLS,
               "--permission-mode", "default", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def run(self, prompt: str, cwd: Path) -> str:
        env = _cli.scrubbed_env("ANTHROPIC_API_KEY")
        proc = _cli.run(self._cmd(prompt), label="claude survey", cwd=str(cwd), env=env,
                        timeout=self.timeout)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout
        return data.get("result", "") if isinstance(data, dict) else proc.stdout


class CodexCLISurveyProvider:
    """`codex exec -s read-only` with cwd=repo (subscription, no API key)."""

    def __init__(self, model: str | None = None, *, timeout: float = _SURVEY_TIMEOUT) -> None:
        _cli.require_binary("codex", "The `codex` CLI is not on PATH; install Codex.")
        self.model = model or "gpt-5.6-sol"
        self.timeout = timeout
        self.name = f"codex-cli:{self.model}"

    def run(self, prompt: str, cwd: Path) -> str:
        env = _cli.scrubbed_env("OPENAI_API_KEY")
        cmd = ["codex", "exec", "-m", self.model, "-s", "read-only",
               "--skip-git-repo-check", prompt]
        proc = _cli.run(cmd, label="codex survey", cwd=str(cwd), env=env, timeout=self.timeout)
        return proc.stdout or ""


class ScriptedSurveyProvider:
    """Return `responses` in order (last repeats). For tests and the zero-key demo path."""

    def __init__(self, responses: list[str], name: str = "scripted") -> None:
        self.responses = list(responses)
        self.name = name
        self.calls: list[str] = []

    def run(self, prompt: str, cwd: Path) -> str:
        self.calls.append(prompt)
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[i] if self.responses else ""


def survey_provider(settings: Settings, provider: str | None = None,
                    model: str | None = None) -> SurveyProvider:
    """Resolve the configured (or overridden) survey provider. `scripted` is test-only."""
    name = provider or settings.survey_provider
    chosen_model = model or settings.survey_model or None
    if name == "claude-cli":
        return ClaudeCLISurveyProvider(chosen_model)
    if name == "codex-cli":
        return CodexCLISurveyProvider(chosen_model)
    raise ValueError(f"unknown survey provider {name!r} (use claude-cli or codex-cli)")
