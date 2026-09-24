"""Resolve a provider spec string to a Provider, and describe provider availability for `doctor`.

Spec strings:
  scripted[:...]                          deterministic, no key
  reference                               replay each task's recorded reference (incumbent baseline)
  openai:<model>                          api.openai.com/v1, OPENAI_API_KEY | keychain
  openai-compatible:<base_url>:<model>    any /v1 endpoint (base_url may contain colons)
  anthropic:<model>                       Anthropic Messages API, ANTHROPIC_API_KEY | keychain
  claude-cli[:model]                      `claude -p` subprocess (subscription)
  codex-cli[:model]                       `codex exec` subprocess (subscription)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from ..config import Settings, load_settings
from ._http import ProviderError
from .base import Provider
from .keychain import secret
from .scripted import ScriptedProvider

OPENAI_BASE = "https://api.openai.com/v1"


def _unknown(spec: str) -> ValueError:
    return ValueError(f"unknown provider spec {spec!r}")


def _model(name: str, arg: str | None, spec: str) -> str:
    """The model after 'name:', or raise: 'unknown' with no colon, 'spec must be' with no model."""
    if arg is None:
        raise _unknown(spec)
    if not arg:
        raise ValueError(f"{name} spec must be '{name}:<model>'.")
    return arg


def _openai(arg: str | None, spec: str, settings: Settings) -> Provider:
    model = _model("openai", arg, spec)
    key = secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai))
    if not key:
        raise ProviderError(
            "No OpenAI API key found; set OPENAI_API_KEY or add it to the keychain."
        )
    from .openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(OPENAI_BASE, model, key)


def _openai_compatible(arg: str | None, spec: str, settings: Settings) -> Provider:
    form = "openai-compatible spec must be 'openai-compatible:<base_url>:<model>'."
    if arg is None:
        raise _unknown(spec)
    if ":" not in arg:
        raise ValueError(form)
    base_url, model = arg.rsplit(":", 1)
    if not model or not base_url:
        raise ValueError(form)
    key = secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai))
    from .openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(base_url, model, key)


def _anthropic(arg: str | None, spec: str, settings: Settings) -> Provider:
    model = _model("anthropic", arg, spec)
    key = secret("ANTHROPIC_API_KEY", settings.keychain_service(settings.keychain_anthropic))
    if not key:
        raise ProviderError(
            "No Anthropic API key found; set ANTHROPIC_API_KEY or add it to the keychain."
        )
    from .anthropic import AnthropicProvider
    return AnthropicProvider(model, key)


def _scripted(arg: str | None, spec: str, settings: Settings) -> Provider:
    return ScriptedProvider()


def _reference(arg: str | None, spec: str, settings: Settings) -> Provider:
    if arg is not None:  # "reference" takes no argument
        raise _unknown(spec)
    from .reference import ReferenceProvider
    return ReferenceProvider()


def _nop(arg: str | None, spec: str, settings: Settings) -> Provider:
    if arg is not None:  # "nop" takes no argument
        raise _unknown(spec)
    from .nop import NopProvider
    return NopProvider()


def _claude_cli(arg: str | None, spec: str, settings: Settings) -> Provider:
    from .claude_cli import ClaudeCLIProvider
    return ClaudeCLIProvider(arg or None)


def _codex_cli(arg: str | None, spec: str, settings: Settings) -> Provider:
    from .codex_cli import CodexCLIProvider
    return CodexCLIProvider(arg or None)


_BUILDERS = {
    "scripted": _scripted,
    "reference": _reference,
    "nop": _nop,
    "openai": _openai,
    "openai-compatible": _openai_compatible,
    "anthropic": _anthropic,
    "claude-cli": _claude_cli,
    "codex-cli": _codex_cli,
}


def provider_from_spec(spec: str, settings: Settings | None = None) -> Provider:
    settings = settings or load_settings()
    head, sep, rest = spec.partition(":")
    builder = _BUILDERS.get(head)
    if builder is None:
        raise _unknown(spec)
    return builder(rest if sep else None, spec, settings)


# ---- doctor ----------------------------------------------------------------


@dataclass
class ProviderStatus:
    name: str
    ok: bool
    detail: str


def provider_statuses(settings: Settings | None = None) -> list[ProviderStatus]:
    """Availability of each provider family, without any network call."""
    settings = settings or load_settings()
    statuses = [
        ProviderStatus("scripted", True, "always available"),
        ProviderStatus("reference", True, "always available (replays recorded references)"),
        ProviderStatus("nop", True, "always available (empty reply — the nop gate)"),
    ]

    openai_key = bool(secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai)))
    statuses.append(
        ProviderStatus(
            "openai", openai_key, "key found" if openai_key else "no OPENAI_API_KEY / keychain key"
        )
    )

    anthropic_key = bool(
        secret("ANTHROPIC_API_KEY", settings.keychain_service(settings.keychain_anthropic))
    )
    statuses.append(
        ProviderStatus(
            "anthropic",
            anthropic_key,
            "key found" if anthropic_key else "no ANTHROPIC_API_KEY / keychain key",
        )
    )

    claude = shutil.which("claude") is not None
    statuses.append(
        ProviderStatus("claude-cli", claude, "claude on PATH" if claude else "claude not on PATH")
    )

    codex = shutil.which("codex") is not None
    statuses.append(
        ProviderStatus("codex-cli", codex, "codex on PATH" if codex else "codex not on PATH")
    )

    return statuses
