"""Resolve a provider spec string to a Provider. Only `scripted` exists in this stage."""

from __future__ import annotations

from .base import Provider
from .scripted import ScriptedProvider


def provider_from_spec(spec: str) -> Provider:
    if spec == "scripted" or spec.startswith("scripted:"):
        return ScriptedProvider()
    raise ValueError(f"unknown provider spec {spec!r}; only 'scripted' is available in this stage")
