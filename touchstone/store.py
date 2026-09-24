"""SQLite persistence: the only place SQL lives. WAL + busy_timeout, typed dataclasses, no ORM.

Every function takes an open connection so callers (server + instrumented app) can each hold
their own connection safely under concurrent writes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .ids import new_id


def now() -> str:
    return datetime.now(UTC).isoformat()


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _loads(text):
    return None if text is None else json.loads(text)


# v2 (2026-09): authored artifacts (tasks, checks, benchmarks) moved to files; the DB keeps only
# captured traces and the machine run record. A v1 DB opens fine — the old authored tables and the
# stale run record are dropped, and `mine` rebuilds tasks/checks from the preserved episodes.
# v3 (2026-09): span `kind` 'llm' renamed to 'model'; `spans.tool_call_id` links a tool span to
# the model call that requested it. A v1/v2 DB migrates in place, preserving its spans.
# v4 (2026-09): `difficulty` table — empirical pass rate per (task, model_spec), the founder's
# "difficulty measured, not requested". Created via CREATE IF NOT EXISTS on any older DB.
SCHEMA_VERSION = 4

# Tables the v1 schema created that no longer exist; dropped only on the v1 -> v2 migration.
_DROPPED = ("checks", "tasks", "benchmarks", "room_checks", "runs", "results")

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
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, target TEXT, model_spec TEXT,
  started_at TEXT, finished_at TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT, task TEXT, passed INTEGER, reward REAL, check_results TEXT, output TEXT,
  latency_ms INTEGER, cost_usd REAL, error TEXT,
  PRIMARY KEY (run_id, task)
);
CREATE TABLE IF NOT EXISTS difficulty (
  task TEXT, model_spec TEXT, attempts INTEGER, passes INTEGER, pass_rate REAL, updated_at TEXT,
  PRIMARY KEY (task, model_spec)
);
CREATE TABLE IF NOT EXISTS rooms (
  id TEXT PRIMARY KEY, task_id TEXT, topic TEXT, created_at TEXT, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS room_messages (
  id TEXT PRIMARY KEY, room_id TEXT, speaker TEXT, role TEXT, text TEXT, audio_path TEXT, ts TEXT
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
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        with write(conn):
            if version == 1:  # v1 -> v2: authored artifacts left the DB; the run record is stale
                for table in _DROPPED:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            if version >= 1:  # existing spans predate v3
                _migrate_spans_v3(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


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


# ---- dataclasses -----------------------------------------------------------


@dataclass
class Episode:
    name: str
    source: str = "app"
    started_at: str = field(default_factory=now)
    ended_at: str | None = None
    outcome_score: float | None = None
    outcome_label: str | None = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)


@dataclass
class Span:
    episode_id: str
    kind: str  # 'model' | 'tool'
    name: str
    parent_id: str | None = None
    model: str | None = None
    started_at: str = field(default_factory=now)
    ended_at: str | None = None
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    tool_call_id: str | None = None
    id: str = field(default_factory=new_id)


@dataclass
class Run:
    target: str  # a benchmark name, a tasks/ path, or a glob
    model_spec: str
    started_at: str = field(default_factory=now)
    finished_at: str | None = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)


@dataclass
class Result:
    run_id: str
    task: str  # task directory name
    passed: int
    reward: float
    check_results: dict = field(default_factory=dict)  # keyed by check name
    output: dict = field(default_factory=dict)
    latency_ms: int | None = None
    cost_usd: float | None = None
    error: str | None = None


@dataclass
class Difficulty:
    task: str
    model_spec: str
    attempts: int = 0
    passes: int = 0
    pass_rate: float = 0.0
    updated_at: str = field(default_factory=now)


@dataclass
class Room:
    task_id: str | None
    topic: str
    created_at: str = field(default_factory=now)
    closed_at: str | None = None
    id: str = field(default_factory=new_id)


@dataclass
class RoomMessage:
    room_id: str
    speaker: str
    role: str
    text: str
    audio_path: str | None = None
    ts: str = field(default_factory=now)
    id: str = field(default_factory=new_id)


# JSON-typed columns per table, so row<->dataclass conversion stays declarative.
_JSON_COLS = {
    "episodes": {"meta"},
    "spans": {"input", "output"},
    "runs": {"meta"},
    "results": {"check_results", "output"},
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


# ---- runs ------------------------------------------------------------------


def insert_run(conn, run: Run) -> Run:
    _insert(conn, "runs", run)
    return run


def get_run(conn, id: str) -> Run | None:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (id,)).fetchone()
    return _row_to(Run, "runs", row)


def list_runs(conn, target: str | None = None) -> list[Run]:
    if target is None:
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE target=? ORDER BY id", (target,)
        ).fetchall()
    return [_row_to(Run, "runs", r) for r in rows]


def finish_run(conn, id: str) -> None:
    _update(conn, "runs", "id", id, finished_at=now())


# ---- results ---------------------------------------------------------------


def insert_result(conn, result: Result) -> Result:
    _insert(conn, "results", result)
    return result


def list_results(conn, run_id: str) -> list[Result]:
    rows = conn.execute(
        "SELECT * FROM results WHERE run_id=? ORDER BY task", (run_id,)
    ).fetchall()
    return [_row_to(Result, "results", r) for r in rows]


# ---- difficulty ------------------------------------------------------------


def upsert_difficulty(conn, task: str, model_spec: str, passed: bool) -> Difficulty:
    """Record one attempt for (task, model_spec) and return the updated running pass rate."""
    p = 1 if passed else 0
    params = {"task": task, "model": model_spec, "p": p, "now": now()}
    with write(conn):
        conn.execute(
            "INSERT INTO difficulty (task, model_spec, attempts, passes, pass_rate, updated_at) "
            "VALUES (:task, :model, 1, :p, :p, :now) "
            "ON CONFLICT(task, model_spec) DO UPDATE SET "
            "  attempts = attempts + 1, "
            "  passes = passes + :p, "
            "  pass_rate = (passes + :p) * 1.0 / (attempts + 1), "
            "  updated_at = :now",
            params,
        )
    return get_difficulty(conn, task, model_spec)


def get_difficulty(conn, task: str, model_spec: str) -> Difficulty | None:
    row = conn.execute(
        "SELECT * FROM difficulty WHERE task=? AND model_spec=?", (task, model_spec)
    ).fetchone()
    return _row_to(Difficulty, "difficulty", row)


def list_difficulty(conn, *, task: str | None = None,
                    model_spec: str | None = None) -> list[Difficulty]:
    clauses, args = [], []
    if task is not None:
        clauses.append("task=?")
        args.append(task)
    if model_spec is not None:
        clauses.append("model_spec=?")
        args.append(model_spec)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM difficulty{where} ORDER BY task, model_spec", args)
    return [_row_to(Difficulty, "difficulty", r) for r in rows.fetchall()]


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
