"""Shared route dependencies and view helpers.

`get_conn` yields a per-request SQLite connection; `get_root` gives the project root that holds the
task/checks/benchmark files. The view helpers turn tasks and checks into the JSON the UI consumes.
"""

from __future__ import annotations

from fastapi import Request

from ... import store
from ...checks import Check
from ...tasks import Task


def get_conn(request: Request):
    conn = store.connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_root(request: Request):
    return request.app.state.settings.root


def check_view(c: Check) -> dict:
    return {
        "name": c.name, "kind": c.kind, "params": c.params, "applies_to": c.applies_to,
        "severity": c.severity, "source": c.source, "rule": c.rule, "because": c.because,
        "confidence": c.confidence,
    }


def task_view(t: Task) -> dict:
    return {
        "name": t.name, "kind": t.kind, "tags": t.tags or [], "episode_id": t.episode_id,
        "status": t.status, "check_count": len(t.checks),
    }


def task_detail(t: Task) -> dict:
    return {
        "name": t.name, "kind": t.kind, "tags": t.tags or [], "episode_id": t.episode_id,
        "cut_span_id": t.cut_span_id, "status": t.status, "status_reason": t.status_reason,
        "context": t.context, "reference": t.reference,
        "checks": [check_view(c) for c in t.checks],
    }


def paginate(items: list, limit: int | None, offset: int):
    return items[offset : offset + limit] if limit is not None else items[offset:]
