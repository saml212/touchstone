"""Which environment variable holds a model provider's API key, and how to resolve its value.

`provider_key_var("openai/gpt-4o-mini")` -> "OPENAI_API_KEY". Providers that need no key (nop,
claude-cli, codex-cli, reference, scripted) return None. Used by the packaged agent (to pass the key
into the sandbox) and by `run.py` (to forward it to the harbor host). `resolve_key` looks a key up
the way bench does — env var first, then the macOS Keychain item `<prefix><provider>-api-key` — so
the packaged adapter check runs with the operator's real key instead of a placeholder. A resolved
secret is only ever returned; it is never logged, printed, or stored here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings

_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_KEY_VARS = frozenset(_PROVIDER_KEY_ENV.values())


def provider_key_var(model: str | None) -> str | None:
    """The API-key env var for `provider/model`'s provider, or None when no key is needed."""
    if not model:
        return None
    return _PROVIDER_KEY_ENV.get(model.split("/", 1)[0])


def provider_key_vars(names) -> set[str]:
    """The subset of env var names that are model-provider API keys (the rest are the customer's
    own service tokens, which a simulator ignores and a placeholder satisfies)."""
    return {n for n in names if n in _KEY_VARS}


def _keychain_suffix(key_var: str, settings: Settings) -> str | None:
    return {"OPENAI_API_KEY": settings.keychain_openai,
            "ANTHROPIC_API_KEY": settings.keychain_anthropic}.get(key_var)


def resolve_key(key_var: str, settings: Settings) -> str | None:
    """Resolve a provider key var to its value the way bench does: `$key_var`, then the Keychain
    item `<prefix><provider>-api-key`. None when unset. The value is only returned, never logged."""
    from ..llm.keychain import secret

    suffix = _keychain_suffix(key_var, settings)
    if suffix is None:
        return None
    return secret(key_var, settings.keychain_service(suffix))
