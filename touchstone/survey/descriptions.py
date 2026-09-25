"""Plain-English descriptions for a task's criteria — what a product person reads, never a path.

rewardkit's own criterion labels carry file paths ("...returns 285.4 on
simulators/orders_service/state.db", "...trajectory: /logs/agent/trajectory.json"). Those are for
engineers. Touchstone writes a parallel `tests/descriptions.toml`, keyed ``"<file>:<index>"`` (the
criteria file relative to the task, 1-based line), with a sentence a product person understands.

The survey writes it at generation time from the state diff; `review.changes` keeps it in sync when
the room edits a criterion (indexes shift on add/remove); `review.trials` reads it so the UI and the
agent show the sentence with the raw rewardkit call available only on hover. It is Touchstone
metadata — the verifier never reads it, so regenerating it needs no re-gate.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import tomli_w

FILE = "descriptions.toml"


def path(tests_dir: str | Path) -> Path:
    return Path(tests_dir) / FILE


def key(rel_file: str, index: int) -> str:
    """The descriptions.toml key for the 1-based `index`-th criterion in `rel_file`."""
    return f"{rel_file}:{index}"


def load(tests_dir: str | Path) -> dict[str, str]:
    p = path(tests_dir)
    if not p.is_file():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str)}


def write(tests_dir: str | Path, mapping: dict[str, str]) -> Path | None:
    """Write descriptions.toml (sorted), or remove it when empty. Returns the path or None."""
    p = path(tests_dir)
    if not mapping:
        if p.exists():
            p.unlink()
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tomli_w.dumps(dict(sorted(mapping.items()))), encoding="utf-8")
    return p


# ---- sentence builders (shared by generation and by edits) -----------------


def cell(table: str, col: str, ident, value) -> str:
    return f"the {col} of {table} {ident} is {value}"


def count_rows(table: str, conds: list[tuple], n: int) -> str:
    where = " and ".join(f"{col} is {val}" for col, val in conds)
    return f"exactly {n} {table} where {where}" if where else f"{table} has {n} rows"


def total_rows(table: str, n: int) -> str:
    return f"{table} has {n} rows"


def tool_used(name: str) -> str:
    return f"the agent used {name}"


def tool_not_used(name: str) -> str:
    return f"the agent did not use {name}"


def no_pii() -> str:
    return "no personal data beyond what the customer provided"


# ---- a fallback description parsed from a raw rewardkit call ----------------


def describe_call(fn: str, args: list) -> str:
    """A path-free sentence for an `rk.<fn>(args)` call added/edited without its own text."""
    if fn == "trajectory_tool_used" and args:
        return tool_used(str(args[0]))
    if fn == "trajectory_tool_not_used" and args:
        return tool_not_used(str(args[0]))
    if fn == "sqlite_query_equals" and len(args) >= 3:
        return _describe_query(str(args[1]), args[2])
    return fn.replace("_", " ")


_SELECT = re.compile(r"select\s+(.+?)\s+from\s+([^\s]+)(?:\s+where\s+(.+))?$", re.I | re.S)


def _describe_query(query: str, expected) -> str:
    match = _SELECT.match(query.strip())
    if not match:
        return f"a check returns {expected}"
    col, table, where = match.group(1).strip(), match.group(2).strip(), match.group(3)
    conds = _conds(where)
    if col.lower().startswith("count("):
        return count_rows(table, conds, expected)
    ident = " and ".join(f"{c} {v}" for c, v in conds) or "a row"
    return cell(table, col, ident, expected)


def _conds(where: str | None) -> list[tuple]:
    if not where:
        return []
    out = []
    for clause in re.split(r"\s+and\s+", where.strip(), flags=re.I):
        if "=" in clause:
            col, _, val = clause.partition("=")
            out.append((col.strip(), val.strip().strip("'\"")))
    return out
