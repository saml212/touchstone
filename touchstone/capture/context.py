"""Episode context + span recording. The instrumented app, the SDK patches, the litellm callback,
and the public helpers all funnel through `add_span` here.

A connection is thread-local and keyed to the configured db path, so concurrent writers (server +
app) each hold their own connection and rely on WAL + busy_timeout for safety.
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps

from .. import store

_log = logging.getLogger("touchstone")
_db_path: str | None = None
_local = threading.local()
_current: ContextVar[EpisodeHandle | None] = ContextVar("touchstone_episode", default=None)
_paused: ContextVar[bool] = ContextVar("touchstone_paused", default=False)
_untracked_lock = threading.Lock()
_untracked_ids: dict[str, str] = {}


def is_paused() -> bool:
    return _paused.get()


@contextmanager
def paused():
    """Suspend model/tool span recording for the duration of the block. Use it around a model call
    that is NOT the agent under test (a simulated user, a judge, an evaluator)."""
    token = _paused.set(True)
    try:
        yield
    finally:
        _paused.reset(token)


def configure(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    _local.__dict__.pop("conn", None)


def is_configured() -> bool:
    return _db_path is not None


def _safe(action: str, fn, default=None):
    """Run `fn`; if recording fails (e.g. a non-writable db), log one warning and return `default`.
    Capture must never raise into the user's application — the explicit helpers (`episode`, `tool`,
    `outcome`) funnel through here just as the SDK patches funnel through `spans._recorder`."""
    try:
        return fn()
    except Exception as exc:
        _log.warning("touchstone %s failed: %r", action, exc)
        return default


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
        if not self.id:  # detached handle from a failed episode start; nothing to record
            return
        _safe("outcome", lambda: store.outcome(get_conn(), self.id, score, label))


def _start_episode(name: str, meta: dict | None, source: str) -> EpisodeHandle:
    conn = get_conn()
    ep = store.insert_episode(conn, store.Episode(name=name, source=source, meta=meta or {}))
    return EpisodeHandle(ep.id)


@contextmanager
def episode(name: str, meta: dict | None = None, source: str = "app"):
    handle = _safe("episode start", lambda: _start_episode(name, meta, source))
    if handle is None:  # capture unavailable; the app must still run its `with` body
        yield EpisodeHandle("")
        return
    token = _current.set(handle)
    try:
        yield handle
    finally:
        _safe("episode end", lambda: store.end_episode(get_conn(), handle.id))
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
    cost_usd: float | None = None,
    error: str | None = None,
    started_at: str | None = None,
    parent_id: str | None = None,
    tool_call_id: str | None = None,
) -> store.Span | None:
    if _paused.get() and kind in ("model", "tool"):
        return None  # inside touchstone.capture.paused(): not the agent under test
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
        cost_usd=cost_usd,
        error=error,
        started_at=started_at or store.now(),
        tool_call_id=tool_call_id,
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


def _resolve_tool_call_id(tname: str) -> str | None:
    """The most recent model span's unresolved tool_call of this name, if any — so a dispatched
    tool span links back to the call that requested it without the caller passing the id."""
    try:
        conn = get_conn()
        spans = store.list_spans(conn, current_episode().id)
    except Exception:
        return None
    model_spans = [s for s in spans if s.kind == "model"]
    if not model_spans:
        return None
    used = {s.tool_call_id for s in spans if s.kind == "tool" and s.tool_call_id}
    calls = (model_spans[-1].output or {}).get("message", {}).get("tool_calls") or []
    for tc in calls:
        if tc.get("name") == tname and tc.get("id") and tc.get("id") not in used:
            return tc["id"]
    return None


def _record_tool(tname, bound, tool_call_id, started, *, result=None, error=None) -> None:
    def _do():
        call_id = tool_call_id if tool_call_id is not None else _resolve_tool_call_id(tname)
        output = None if error else {"result": jsonable(result)}
        add_span("tool", tname, input={"name": tname, "arguments": bound}, output=output,
                 error=error, started_at=started, tool_call_id=call_id)

    _safe("tool span", _do)  # a recording failure must never lose the tool's own result


def tool(fn=None, *, name: str | None = None):
    """Decorator recording a tool span (name/arguments/result/error/tool_call_id). Works sync and
    async. A caller may pass `tool_call_id=` to link the span to a specific model tool_call;
    otherwise it is auto-linked to the latest unresolved call of the same name."""

    def decorate(f):
        tname = name or f.__name__

        if inspect.iscoroutinefunction(f):
            @wraps(f)
            async def awrapper(*args, **kwargs):
                tcid = kwargs.pop("tool_call_id", None)
                started = store.now()
                bound = _bound_args(f, args, kwargs)
                try:
                    result = await f(*args, **kwargs)
                except Exception as exc:
                    _record_tool(tname, bound, tcid, started, error=repr(exc))
                    raise
                _record_tool(tname, bound, tcid, started, result=result)
                return result

            return awrapper

        @wraps(f)
        def wrapper(*args, **kwargs):
            tcid = kwargs.pop("tool_call_id", None)
            started = store.now()
            bound = _bound_args(f, args, kwargs)
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                _record_tool(tname, bound, tcid, started, error=repr(exc))
                raise
            _record_tool(tname, bound, tcid, started, result=result)
            return result

        return wrapper

    return decorate(fn) if fn else decorate
