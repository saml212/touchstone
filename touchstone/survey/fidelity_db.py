"""Dump a simulator's SQLite state and measure a db service's fidelity.

`dump_db` snapshots every table's rows (keyed by table name, with the primary-key column) so task
criteria can assert on end state. `measure_db` scores a db simulator: it seeds `state.db` from the
scrubbed source and replays each recorded call through the real tools against it, one fresh state.db
per episode (a mutating tool must not leak across episodes), grading only calls with a recorded
output. Shared replay/score/error helpers are imported lazily from `fidelity` to avoid a cycle.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import Settings
from .recordings import ToolEvent
from .scrub import Scrubber


def _table_names(conn) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    return [row[0] for row in conn.execute(q).fetchall()]


def _pk_col(conn, table: str) -> str:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row["pk"]:
            return row["name"]
    return "rowid"


def _dump_table(conn, table: str) -> dict:
    pk = _pk_col(conn, table)
    select = "*" if pk != "rowid" else "rowid AS rowid, *"
    rows = [dict(r) for r in conn.execute(f"SELECT {select} FROM {table}").fetchall()]
    return {"pk": pk, "rows": rows}


def dump_db(db_path: Path) -> dict:
    """Every table's rows keyed by name, with the primary-key column. Missing db -> empty dict."""
    path = Path(db_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return {table: _dump_table(conn, table) for table in _table_names(conn)}
    finally:
        conn.close()


def _scrub_calls(calls: list[ToolEvent], scrub: Scrubber) -> list[ToolEvent]:
    """state.db is seeded from scrubbed data, so the recorded calls are scrubbed the same way before
    replay + compare — one namespace, so an id in an argument resolves the document it seeded."""
    return [ToolEvent(tool=c.tool, arguments=scrub.scrub(c.arguments), output=scrub.scrub(c.output),
                      episode=c.episode, module=c.module) for c in calls]


def _by_episode(calls: list[ToolEvent]) -> list[list[ToolEvent]]:
    """Group calls into episodes, preserving order, so each episode replays against a fresh
    state.db (a mutating tool in one episode must not leak into the next)."""
    groups: dict[str, list[ToolEvent]] = {}
    for call in calls:
        groups.setdefault(call.episode, []).append(call)
    return list(groups.values())


def _replay_db(sim_dir: Path, repo: Path, calls: list[ToolEvent], ctx: dict,
               settings: Settings) -> list[dict]:
    from . import db_service, db_sim, fidelity

    got: list[dict] = []
    for episode in _by_episode(calls):
        db = db_sim.materialize(sim_dir)  # fresh state per episode
        base = db_service.value_for(str(db), url=ctx.get("db_url", False))
        got += fidelity._run_replay(repo, episode, base, ctx, settings)
    return got


def measure_db(sim_dir: Path, repo: Path, calls: list[ToolEvent], ctx: dict,
               settings: Settings, scrub: Scrubber, threshold: float) -> dict:
    """Fidelity for a db service. Only calls with a recorded output are graded (a proposed-but-
    unexecuted tool call carries no ground truth); an unsupported service is reported 0."""
    from . import db_sim, fidelity

    reason = db_sim.unsupported(sim_dir)
    if reason:
        result = fidelity._failed_result(calls, threshold, f"db service unsupported: {reason}")
        result["unsupported"] = reason
        return result
    graded = [c for c in _scrub_calls(calls, scrub) if c.output is not None]
    try:
        got_list = _replay_db(sim_dir, repo, graded, ctx, settings)
    except fidelity._SimError as exc:
        return fidelity._failed_result(graded, threshold, str(exc))
    return fidelity._score(graded, got_list, threshold, scrub)
