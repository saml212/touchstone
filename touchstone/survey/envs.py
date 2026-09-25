"""Placeholder auth env vars, shared by replay, the packaged adapter check, and the packaged run.

A tool often constructs its HTTP/SDK client from an API key or token and raises when that env var is
unset. A simulator ignores auth, so every path that runs the customer's real code offline sets a
harmless placeholder for each auth-shaped env var the repo names — the same treatment in all three
places, so a tool that needs a dummy token to build its client works everywhere. Never overrides a
value already present in the environment (a real secret the operator set wins).
"""

from __future__ import annotations

from pathlib import Path

from .tool_reads import auth_env_names as _names_in_source

PLACEHOLDER = "touchstone-placeholder"
_SKIP_DIRS = {".venv", ".git", "__pycache__", "touchstone", ".touchstone", "node_modules"}


def auth_env_names(repo: str | Path) -> set[str]:
    """Auth-shaped env var names (KEY/TOKEN/SECRET/PASSWORD) read anywhere under `repo`."""
    names: set[str] = set()
    for path in Path(repo).rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            names |= _names_in_source(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return names


def placeholder_auth(names) -> dict[str, str]:
    """{name: placeholder} for each auth env var name — the value a simulator ignores."""
    return {name: PLACEHOLDER for name in names}
