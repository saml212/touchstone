"""SQLite persistence: the only place SQL lives. WAL + busy_timeout, typed dataclasses, no ORM.

Every function takes an open connection so callers (server + instrumented app) can each hold
their own connection safely under concurrent writes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

# The typed rows and their shared timestamp live in store_models (no SQL); re-exported here so
# callers keep using store.Episode / store.Span / store.now(), etc.
from .store_models import (  # noqa: F401
    Episode,
    Review,
    Room,
    RoomMessage,
    Span,
    now,
)


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _loads(text):
    return None if text is None else json.loads(text)


# v2 (2026-09): authored artifacts (tasks, checks, benchmarks) moved to files; the DB keeps only
# captured traces. A v1 DB opens fine — the old authored tables are dropped.
# v3 (2026-09): span `kind` 'llm' renamed to 'model'; `spans.tool_call_id` links a tool span to
# the model call that requested it. A v1/v2 DB migrates in place, preserving its spans.
# v4 (2026-09): a `difficulty` table (later removed).
# v5 (2026-09, v3 rework): Harbor is the run record — the `difficulty`, `runs`, and `results` tables
# are dropped; a `reviews` table records what a review room decided about a task's trial.
SCHEMA_VERSION = 5

# Tables the v1 schema created that no longer exist; dropped only on the v1 -> v2 migration.
_DROPPED = ("checks", "tasks", "benchmarks", "room_checks")
# Machine-run tables removed in v5 — the Harbor job directory is the run record now.
_DROPPED_V5 = ("difficulty", "runs", "results")

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, name TEXT, source TEXT, started_at TEXT, ended_at TEXT,
  outcome_score REAL, outcome_label TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, episode_id TEXT, parent_id TEXT, kind TEXT, name TEXT, model TEXT,
  started_at TEXT, ended_at TEXT, input TEXT, output TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, error TEXT, tool_call_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_episode ON spans(episode_id);
CREATE TABLE IF NOT EXISTS rooms (
  id TEXT PRIMARY KEY, task_id TEXT, topic TEXT, created_at TEXT, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS room_messages (
  id TEXT PRIMARY KEY, room_id TEXT, speaker TEXT, role TEXT, text TEXT, audio_path TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, task TEXT, trial TEXT, verdict TEXT, speaker TEXT, note TEXT, ts TEXT
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: a web request's connection is created, used, and closed across
    # different threadpool threads (in sequence, never concurrently), and each caller holds its own.
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_wal(conn)
    for pragma in ("synchronous=NORMAL", "busy_timeout=5000", "foreign_keys=ON"):
        conn.execute(f"PRAGMA {pragma}")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        _migrate(conn, version)
    return conn


def _migrate(conn: sqlite3.Connection, version: int) -> None:
    """Bring an older (or empty) database up to SCHEMA_VERSION, in one transaction."""
    with write(conn):
        _drop_retired_tables(conn, version)
        _create_tables(conn)
        if version >= 1:  # existing spans predate v3
            _migrate_spans_v3(conn)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _drop_retired_tables(conn: sqlite3.Connection, version: int) -> None:
    if version == 1:  # v1 -> v2: authored artifacts left the DB
        for table in _DROPPED:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    if 1 <= version < 5:  # v5: Harbor is the run record; drop the machine-run tables
        for table in _DROPPED_V5:
            conn.execute(f"DROP TABLE IF EXISTS {table}")


def _create_tables(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)


def _migrate_spans_v3(conn: sqlite3.Connection) -> None:
    """v3: rename span kind 'llm' -> 'model' and add the tool_call_id column if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(spans)")}
    if "tool_call_id" not in cols:
        conn.execute("ALTER TABLE spans ADD COLUMN tool_call_id TEXT")
    conn.execute("UPDATE spans SET kind='model' WHERE kind='llm'")


def _ensure_wal(conn: sqlite3.Connection) -> None:
    """journal_mode=WAL ignores busy_timeout: switch only when needed, retry the first-open race."""
    for attempt in range(5):
        if conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
            return
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


@contextmanager
def write(conn: sqlite3.Connection):
    """BEGIN IMMEDIATE takes the write lock up front so busy_timeout applies under contention."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# JSON-typed columns per table, so row<->dataclass conversion stays declarative.
_JSON_COLS = {
    "episodes": {"meta"},
    "spans": {"input", "output"},
}


def _row_to(cls, table: str, row: sqlite3.Row):
    if row is None:
        return None
    data = dict(row)
    for col in _JSON_COLS.get(table, ()):
        if col in data:
            data[col] = _loads(data[col])
    return cls(**data)


def _insert(conn: sqlite3.Connection, table: str, obj) -> None:
    data = asdict(obj)
    for col in _JSON_COLS.get(table, ()):
        data[col] = _dumps(data[col])
    cols = ", ".join(data)
    placeholders = ", ".join(f":{c}" for c in data)
    with write(conn):
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", data)


def _update(conn: sqlite3.Connection, table: str, id_col: str, id_val, **fields) -> None:
    json_cols = _JSON_COLS.get(table, ())
    fields = {k: (_dumps(v) if k in json_cols else v) for k, v in fields.items()}
    sets = ", ".join(f"{k}=:{k}" for k in fields)
    fields["_id"] = id_val
    with write(conn):
        conn.execute(f"UPDATE {table} SET {sets} WHERE {id_col}=:_id", fields)


# ---- episodes --------------------------------------------------------------


def insert_episode(conn, ep: Episode) -> Episode:
    _insert(conn, "episodes", ep)
    return ep


def get_episode(conn, id: str) -> Episode | None:
    row = conn.execute("SELECT * FROM episodes WHERE id=?", (id,)).fetchone()
    return _row_to(Episode, "episodes", row)


def list_episodes(conn, label: str | None = None) -> list[Episode]:
    if label is None:
        rows = conn.execute("SELECT * FROM episodes ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM episodes WHERE outcome_label=? ORDER BY id", (label,)
        ).fetchall()
    return [_row_to(Episode, "episodes", r) for r in rows]


def outcome(conn, episode_id: str, score: float | None, label: str | None) -> None:
    _update(
        conn,
        "episodes",
        "id",
        episode_id,
        outcome_score=score,
        outcome_label=label,
        ended_at=now(),
    )


def end_episode(conn, episode_id: str) -> None:
    _update(conn, "episodes", "id", episode_id, ended_at=now())


# ---- spans -----------------------------------------------------------------


def insert_span(conn, span: Span) -> Span:
    _insert(conn, "spans", span)
    return span


def get_span(conn, id: str) -> Span | None:
    row = conn.execute("SELECT * FROM spans WHERE id=?", (id,)).fetchone()
    return _row_to(Span, "spans", row)


def list_spans(conn, episode_id: str) -> list[Span]:
    rows = conn.execute(
        "SELECT * FROM spans WHERE episode_id=? ORDER BY id", (episode_id,)
    ).fetchall()
    return [_row_to(Span, "spans", r) for r in rows]


def update_span(conn, id: str, **fields) -> None:
    _update(conn, "spans", "id", id, **fields)


# ---- reviews ---------------------------------------------------------------


def insert_review(conn, review: Review) -> Review:
    _insert(conn, "reviews", review)
    return review


def list_reviews(conn, task: str | None = None) -> list[Review]:
    if task is None:
        rows = conn.execute("SELECT * FROM reviews ORDER BY id").fetchall()
    else:
        rows = conn.execute("SELECT * FROM reviews WHERE task=? ORDER BY id", (task,)).fetchall()
    return [_row_to(Review, "reviews", r) for r in rows]


# ---- rooms -----------------------------------------------------------------


def insert_room(conn, room: Room) -> Room:
    _insert(conn, "rooms", room)
    return room


def get_room(conn, id: str) -> Room | None:
    row = conn.execute("SELECT * FROM rooms WHERE id=?", (id,)).fetchone()
    return _row_to(Room, "rooms", row)


def list_rooms(conn) -> list[Room]:
    rows = conn.execute("SELECT * FROM rooms ORDER BY id").fetchall()
    return [_row_to(Room, "rooms", r) for r in rows]


def insert_room_message(conn, msg: RoomMessage) -> RoomMessage:
    _insert(conn, "room_messages", msg)
    return msg


def list_room_messages(conn, room_id: str) -> list[RoomMessage]:
    rows = conn.execute(
        "SELECT * FROM room_messages WHERE room_id=? ORDER BY id", (room_id,)
    ).fetchall()
    return [_row_to(RoomMessage, "room_messages", r) for r in rows]


def close_room(conn, room_id: str) -> None:
    _update(conn, "rooms", "id", room_id, closed_at=now())
