"""Find candidate system prompts and tool schemas in a code tree.

Line-based heuristics over source files: system-prompt strings, `tools=[...]` arrays, JSON schema
objects with a `"parameters"` key, and Anthropic `input_schema`. Deterministic and dependency-free;
skips vendored/build directories and files over 1MB so a scan of a real repo stays fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = frozenset({
    "node_modules", ".venv", "venv", ".git", "build", "dist", "target",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".next",
})
_EXTS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".yaml", ".yml", ".md",
})
_MAX_BYTES = 1_000_000
_MAX_BLOCK_LINES = 25

# Ordered by priority: the first pattern that matches a line labels that line.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("system_prompt", re.compile(r"\bSYSTEM_PROMPT\b")),
    ("system_prompt", re.compile(r"\bSYSTEM\s*[:=]")),
    ("system_prompt", re.compile(r"""["']?role["']?\s*[:=]\s*["']system["']""")),
    ("tools", re.compile(r"\btools\s*[:=]\s*\[")),
    ("input_schema", re.compile(r"\binput_schema\b")),
    ("parameters", re.compile(r'"parameters"\s*:')),
)


@dataclass(frozen=True)
class Snippet:
    path: str  # path as given on the command line, so it is copy-pasteable
    line: int  # 1-indexed
    kind: str  # system_prompt | tools | input_schema | parameters
    text: str


def scan_codebase(root: str | Path) -> list[Snippet]:
    """Return candidate prompt/tool snippets found under `root`, sorted by (path, line)."""
    root = Path(root)
    snippets: list[Snippet] = []
    for path in _walk(root):
        try:
            if path.stat().st_size > _MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root)) if path != root else str(path)
        snippets.extend(_scan_text(rel, text))
    snippets.sort(key=lambda s: (s.path, s.line))
    return snippets


def _walk(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _EXTS:
            yield path


def _scan_text(rel: str, text: str) -> list[Snippet]:
    lines = text.splitlines()
    out: list[Snippet] = []
    for idx, line in enumerate(lines):
        for kind, pattern in _PATTERNS:
            if pattern.search(line):
                out.append(Snippet(path=rel, line=idx + 1, kind=kind, text=_block(lines, idx)))
                break  # one label per line
    return out


def _block(lines: list[str], start: int) -> str:
    """Capture from the matched line through the close of any bracket it opens (bounded)."""
    first = lines[start]
    depth = _delta(first)
    out = [first]
    i = start + 1
    while depth > 0 and i < len(lines) and len(out) < _MAX_BLOCK_LINES:
        out.append(lines[i])
        depth += _delta(lines[i])
        i += 1
    return "\n".join(out).strip()


def _delta(line: str) -> int:
    return line.count("{") + line.count("[") - line.count("}") - line.count("]")
