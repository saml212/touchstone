"""Which environment variable holds a model provider's API key.

`provider_key_var("openai/gpt-4o-mini")` -> "OPENAI_API_KEY". Providers that need no key (nop,
claude-cli, codex-cli, reference, scripted) return None. Used by the packaged agent (to pass the key
into the sandbox) and by `run.py` (to forward it to the harbor host). No secret ever lives here.
"""

from __future__ import annotations

_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_key_var(model: str | None) -> str | None:
    """The API-key env var for `provider/model`'s provider, or None when no key is needed."""
    if not model:
        return None
    return _PROVIDER_KEY_ENV.get(model.split("/", 1)[0])
