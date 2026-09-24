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


SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, name TEXT, source TEXT, started_at TEXT, ended_at TEXT,
  outcome_score REAL, outcome_label TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, episode_id TEXT, parent_id TEXT, kind TEXT, name TEXT, model TEXT,
  started_at TEXT, ended_at TEXT, input TEXT, output TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_episode ON spans(episode_id);
CREATE TABLE IF NOT EXISTS checks (
  id TEXT PRIMARY KEY, name TEXT, kind TEXT, params TEXT, applies_to TEXT, severity TEXT,
  source TEXT, rationale TEXT, enabled INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, name TEXT, episode_id TEXT, cut_span_id TEXT, context TEXT,
  reference TEXT, check_ids TEXT, kind TEXT, tags TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS benchmarks (
  id TEXT PRIMARY KEY, name TEXT, task_ids TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, benchmark_id TEXT, model_spec TEXT,
  started_at TEXT, finished_at TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT, task_id TEXT, passed INTEGER, check_results TEXT, output TEXT,
  latency_ms INTEGER, cost_usd REAL, error TEXT,
  PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS rooms (
  id TEXT PRIMARY KEY, task_id TEXT, topic TEXT, created_at TEXT, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS room_messages (
  id TEXT PRIMARY KEY, room_id TEXT, speaker TEXT, role TEXT, text TEXT, audio_path TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS room_checks (
  room_id TEXT, check_id TEXT, PRIMARY KEY (room_id, check_id)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    _ensure_wal(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        with write(conn):
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


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
    kind: str  # 'llm' | 'tool'
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
    id: str = field(default_factory=new_id)


@dataclass
class Check:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    applies_to: str = "final"
    severity: str = "hard"
    source: str = "manual"
    rationale: str = ""
    enabled: int = 0
    created_at: str = field(default_factory=now)
    id: str = field(default_factory=new_id)


@dataclass
class Task:
    name: str
    episode_id: str | None = None
    cut_span_id: str | None = None
    context: dict = field(default_factory=dict)
    reference: dict | None = None
    check_ids: list = field(default_factory=list)
    kind: str = "replay"
    tags: list = field(default_factory=list)
    created_at: str = field(default_factory=now)
    id: str = field(default_factory=new_id)


@dataclass
class Benchmark:
    name: str
    task_ids: list = field(default_factory=list)
    created_at: str = field(default_factory=now)
    id: str = field(default_factory=new_id)


@dataclass
class Run:
    benchmark_id: str
    model_spec: str
    started_at: str = field(default_factory=now)
    finished_at: str | None = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)


@dataclass
class Result:
    run_id: str
    task_id: str
    passed: int
    check_results: list = field(default_factory=list)
    output: dict = field(default_factory=dict)
    latency_ms: int | None = None
    cost_usd: float | None = None
    error: str | None = None


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
    "checks": {"params"},
    "tasks": {"context", "reference", "check_ids", "tags"},
    "benchmarks": {"task_ids"},
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


# ---- checks ----------------------------------------------------------------


def insert_check(conn, check: Check) -> Check:
    _insert(conn, "checks", check)
    return check


def get_check(conn, id: str) -> Check | None:
    row = conn.execute("SELECT * FROM checks WHERE id=?", (id,)).fetchone()
    return _row_to(Check, "checks", row)


def list_checks(conn, enabled: bool | None = None) -> list[Check]:
    if enabled is None:
        rows = conn.execute("SELECT * FROM checks ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM checks WHERE enabled=? ORDER BY id", (1 if enabled else 0,)
        ).fetchall()
    return [_row_to(Check, "checks", r) for r in rows]


def set_check_enabled(conn, id: str, enabled: bool) -> None:
    _update(conn, "checks", "id", id, enabled=1 if enabled else 0)


# ---- tasks -----------------------------------------------------------------


def insert_task(conn, task: Task) -> Task:
    _insert(conn, "tasks", task)
    return task


def get_task(conn, id: str) -> Task | None:
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (id,)).fetchone()
    return _row_to(Task, "tasks", row)


def list_tasks(conn, tag: str | None = None) -> list[Task]:
    rows = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    tasks = [_row_to(Task, "tasks", r) for r in rows]
    if tag is not None:
        tasks = [t for t in tasks if tag in (t.tags or [])]
    return tasks


def update_task(conn, id: str, **fields) -> None:
    _update(conn, "tasks", "id", id, **fields)


# ---- benchmarks ------------------------------------------------------------


def insert_benchmark(conn, bench: Benchmark) -> Benchmark:
    _insert(conn, "benchmarks", bench)
    return bench


def get_benchmark(conn, id_or_name: str) -> Benchmark | None:
    """Look up by id, then by name (newest wins), so CLI users can say `bench run demo`."""
    row = conn.execute("SELECT * FROM benchmarks WHERE id=?", (id_or_name,)).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM benchmarks WHERE name=? ORDER BY id DESC LIMIT 1", (id_or_name,)
        ).fetchone()
    return _row_to(Benchmark, "benchmarks", row)


def list_benchmarks(conn) -> list[Benchmark]:
    rows = conn.execute("SELECT * FROM benchmarks ORDER BY id").fetchall()
    return [_row_to(Benchmark, "benchmarks", r) for r in rows]


# ---- runs ------------------------------------------------------------------


def insert_run(conn, run: Run) -> Run:
    _insert(conn, "runs", run)
    return run


def get_run(conn, id: str) -> Run | None:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (id,)).fetchone()
    return _row_to(Run, "runs", row)


def list_runs(conn, benchmark_id: str | None = None) -> list[Run]:
    if benchmark_id is None:
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE benchmark_id=? ORDER BY id", (benchmark_id,)
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
        "SELECT * FROM results WHERE run_id=? ORDER BY task_id", (run_id,)
    ).fetchall()
    return [_row_to(Result, "results", r) for r in rows]


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


def link_room_check(conn, room_id: str, check_id: str) -> None:
    with write(conn):
        conn.execute(
            "INSERT OR IGNORE INTO room_checks (room_id, check_id) VALUES (?, ?)",
            (room_id, check_id),
        )
