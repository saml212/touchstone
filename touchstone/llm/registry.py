"""Resolve a provider spec string to a Provider, and describe provider availability for `doctor`.

Spec strings:
  scripted[:...]                          deterministic, no key
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


def provider_from_spec(spec: str, settings: Settings | None = None) -> Provider:
    settings = settings or load_settings()

    if spec == "scripted" or spec.startswith("scripted:"):
        return ScriptedProvider()

    if spec.startswith("openai:"):
        model = spec[len("openai:") :]
        key = secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai))
        if not key:
            raise ProviderError(
                "No OpenAI API key found; set OPENAI_API_KEY or add it to the keychain."
            )
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(OPENAI_BASE, model, key)

    if spec.startswith("openai-compatible:"):
        rest = spec[len("openai-compatible:") :]
        if ":" not in rest:
            raise ValueError(
                "openai-compatible spec must be 'openai-compatible:<base_url>:<model>'."
            )
        base_url, model = rest.rsplit(":", 1)
        key = secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai))
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(base_url, model, key)

    if spec.startswith("anthropic:"):
        model = spec[len("anthropic:") :]
        key = secret("ANTHROPIC_API_KEY", settings.keychain_service(settings.keychain_anthropic))
        if not key:
            raise ProviderError(
                "No Anthropic API key found; set ANTHROPIC_API_KEY or add it to the keychain."
            )
        from .anthropic import AnthropicProvider

        return AnthropicProvider(model, key)

    if spec == "claude-cli" or spec.startswith("claude-cli:"):
        from .claude_cli import ClaudeCLIProvider

        model = spec[len("claude-cli:") :] if ":" in spec else None
        return ClaudeCLIProvider(model or None)

    if spec == "codex-cli" or spec.startswith("codex-cli:"):
        from .codex_cli import CodexCLIProvider

        model = spec[len("codex-cli:") :] if ":" in spec else None
        return CodexCLIProvider(model or None)

    raise ValueError(f"unknown provider spec {spec!r}")


# ---- doctor ----------------------------------------------------------------


@dataclass
class ProviderStatus:
    name: str
    ok: bool
    detail: str


def provider_statuses(settings: Settings | None = None) -> list[ProviderStatus]:
    """Availability of each provider family, without any network call."""
    settings = settings or load_settings()
    statuses = [ProviderStatus("scripted", True, "always available")]

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
