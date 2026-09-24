"""Episode context + span recording. The instrumented app, the SDK patches, the litellm callback,
and the public helpers all funnel through `add_span` here.

A connection is thread-local and keyed to the configured db path, so concurrent writers (server +
app) each hold their own connection and rely on WAL + busy_timeout for safety.
"""

from __future__ import annotations

import inspect
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps

from .. import store

_db_path: str | None = None
_local = threading.local()
_current: ContextVar[EpisodeHandle | None] = ContextVar("touchstone_episode", default=None)
_untracked_lock = threading.Lock()
_untracked_ids: dict[str, str] = {}


def configure(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    _local.__dict__.pop("conn", None)


def is_configured() -> bool:
    return _db_path is not None


def get_conn():
    if _db_path is None:
        raise RuntimeError("touchstone.trace() has not been called")
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != _db_path:
        conn = store.connect(_db_path)
        _local.conn = conn
        _local.path = _db_path
    return conn


@dataclass
class EpisodeHandle:
    id: str

    def outcome(self, score: float | None, label: str | None) -> None:
        store.outcome(get_conn(), self.id, score, label)


@contextmanager
def episode(name: str, meta: dict | None = None, source: str = "app"):
    conn = get_conn()
    ep = store.insert_episode(conn, store.Episode(name=name, source=source, meta=meta or {}))
    handle = EpisodeHandle(ep.id)
    token = _current.set(handle)
    try:
        yield handle
    finally:
        store.end_episode(conn, ep.id)
        _current.reset(token)


def _find_untracked(conn, name: str) -> str | None:
    for e in store.list_episodes(conn):
        if e.name == name and e.source == "untracked":
            return e.id
    return None


def _untracked() -> EpisodeHandle:
    conn = get_conn()
    date = datetime.now(UTC).date().isoformat()
    name = f"untracked-{date}"
    with _untracked_lock:
        eid = _untracked_ids.get(date)
        if eid and store.get_episode(conn, eid):
            return EpisodeHandle(eid)
        found = _find_untracked(conn, name)
        if found is None:
            found = store.insert_episode(conn, store.Episode(name=name, source="untracked")).id
        _untracked_ids[date] = found
        return EpisodeHandle(found)


def current_episode() -> EpisodeHandle:
    return _current.get() or _untracked()


def add_span(
    kind: str,
    name: str,
    *,
    model: str | None = None,
    input: dict | None = None,
    output: dict | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    error: str | None = None,
    started_at: str | None = None,
    parent_id: str | None = None,
) -> store.Span:
    conn = get_conn()
    ep = current_episode()
    span = store.Span(
        episode_id=ep.id,
        kind=kind,
        name=name,
        model=model,
        parent_id=parent_id,
        input=input or {},
        output=output or {},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=error,
        started_at=started_at or store.now(),
    )
    span.ended_at = store.now()
    return store.insert_span(conn, span)


def jsonable(value):
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _bound_args(fn, args, kwargs) -> dict:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        return {k: jsonable(v) for k, v in bound.arguments.items()}
    except TypeError:
        return {
            "args": [jsonable(a) for a in args],
            "kwargs": {k: jsonable(v) for k, v in kwargs.items()},
        }


def tool(fn=None, *, name: str | None = None):
    """Decorator recording a tool span (name/arguments/result/error). Works sync and async."""

    def decorate(f):
        tname = name or f.__name__

        if inspect.iscoroutinefunction(f):
            @wraps(f)
            async def awrapper(*args, **kwargs):
                started = store.now()
                bound = _bound_args(f, args, kwargs)
                try:
                    result = await f(*args, **kwargs)
                except Exception as exc:
                    add_span("tool", tname, input={"name": tname, "arguments": bound},
                             error=repr(exc), started_at=started)
                    raise
                add_span("tool", tname, input={"name": tname, "arguments": bound},
                         output={"result": jsonable(result)}, started_at=started)
                return result

            return awrapper

        @wraps(f)
        def wrapper(*args, **kwargs):
            started = store.now()
            bound = _bound_args(f, args, kwargs)
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                add_span("tool", tname, input={"name": tname, "arguments": bound},
                         error=repr(exc), started_at=started)
                raise
            add_span("tool", tname, input={"name": tname, "arguments": bound},
                     output={"result": jsonable(result)}, started_at=started)
            return result

        return wrapper

    return decorate(fn) if fn else decorate
