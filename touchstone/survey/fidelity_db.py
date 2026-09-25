"""Dump a simulator's SQLite state, so task criteria can assert on end state (rows before vs after).

Every table's rows keyed by table name, with the primary-key column; a missing database is an empty
dict. Read-only — the fidelity replay drives the state changes, this just snapshots them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


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
