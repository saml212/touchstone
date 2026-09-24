"""Build a loop provider from a spec, degrading to None when no key/binary is available."""

from __future__ import annotations

from ..config import Settings


def provider_or_none(spec: str, settings: Settings):
    from ..llm import provider_from_spec

    try:
        return provider_from_spec(spec, settings)
    except Exception:  # no key/binary: the caller falls back (mechanical variants / skip demos)
        return None
