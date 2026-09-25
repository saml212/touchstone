"""Verifier helpers, copied into a task's `tests/` and imported by its generated criteria.

Rewardkit runs the verifier with `uvx --from harbor-rewardkit rewardkit /tests`, so `rewardkit`
is importable there (not in this package's own env — the wrappers import it lazily). Three thin
wrappers over built-in criteria, one pure PII check, and a judge-TOML writer. Product-agnostic.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomli_w

# End-state first: a SQL query against a simulator's SQLite file must equal an expected value.


def sqlite_state(db_path: str, sql: str, expected, *, weight: float = 1.0):
    """Register: the SQL query on `db_path` returns `expected` (rewardkit's sqlite_query_equals)."""
    import rewardkit as rk

    return rk.sqlite_query_equals(db_path, sql, expected, weight=weight)


def tool_used(tool_name: str, *, min_count: int = 1, path: str | None = None, weight: float = 1.0):
    """Register: the agent's ATIF trajectory used `tool_name` at least `min_count` times."""
    import rewardkit as rk

    kwargs = {"min_count": min_count, "weight": weight}
    if path is not None:
        kwargs["path"] = path
    return rk.trajectory_tool_used(tool_name, **kwargs)


def tool_not_used(tool_name: str, *, path: str | None = None, weight: float = 1.0):
    """Register: the agent's ATIF trajectory never used `tool_name`."""
    import rewardkit as rk

    kwargs = {"weight": weight}
    if path is not None:
        kwargs["path"] = path
    return rk.trajectory_tool_not_used(tool_name, **kwargs)


_PII = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                 # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                    # US SSN
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),                   # card-like digit run
    re.compile(r"\b\+?\d{1,2}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),  # phone
)


def no_pii(text: str) -> bool:
    """True when `text` contains no email, SSN, card number, or phone number."""
    return not any(pattern.search(text or "") for pattern in _PII)


def write_judge(path: str | Path, *, model: str, files: list[str],
                criteria: list[dict]) -> Path:
    """Write a rewardkit judge TOML: a `[judge]` block plus one `[[criterion]]` per entry.

    Each criterion is a dict like {"description": "...", "type": "binary"} or
    {"description": "...", "type": "likert", "points": 5}.
    """
    path = Path(path)
    doc = {"judge": {"judge": model, "files": list(files)}, "criterion": list(criteria)}
    path.write_text(tomli_w.dumps(doc), encoding="utf-8")
    return path
