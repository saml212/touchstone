"""Statically read which response keys a tool pulls off an HTTP response.

The recordings only show a tool's parsed output, never the raw response body, so a simulator has to
be told the wire shape. This walks the tool source and returns the nested key paths it reads off the
response (`data["current"]["temp_c"]` -> `current.temp_c`), which the simulator prompt lists as
required keys.
"""

from __future__ import annotations

import ast
import re

# Auth-shaped env var names (KEY/TOKEN/SECRET/PASSWORD). A tool often needs one to construct its
# client, and it fails when unset — but the value is irrelevant to a simulator (which ignores auth),
# so replay sets a harmless placeholder. Restricted to auth suffixes so a numeric/URL env is never
# clobbered. Matches direct reads and custom wrappers (`_require_env("FLIGHT_API_KEY")`) alike.
_AUTH_ENV = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD))\b")


def auth_env_names(source: str) -> set[str]:
    """Auth-shaped environment variable names named anywhere in `source`."""
    return set(_AUTH_ENV.findall(source))


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _subscript_path(node: ast.Subscript) -> list[str]:
    """The dot path of constant string keys read off a subscript chain (data['a']['b'] -> a.b)."""
    keys: list[str] = []
    while isinstance(node, ast.Subscript):
        key = _const_str(node.slice)
        if key is None:
            break
        keys.append(key)
        node = node.value
    return list(reversed(keys))


def response_key_paths(source: str) -> list[str]:
    """Nested key paths the tool source reads off an HTTP response (subscript chains), so the
    simulator knows the raw body shape the recordings never show. Longest paths only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            keys = _subscript_path(node)
            if keys:
                paths.add(".".join(keys))
    return sorted(p for p in paths if not any(o != p and o.startswith(p + ".") for o in paths))
