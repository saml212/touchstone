"""Shared route dependencies and view helpers.

`get_conn` yields a per-request SQLite connection from the app's configured db path. The view
helpers turn store dataclasses into the JSON the UI consumes; keeping them here stops each router
reinventing the shape.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Request

from ... import store


def get_conn(request: Request):
    conn = store.connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def check_view(c: store.Check) -> dict:
    return {
        "id": c.id, "name": c.name, "kind": c.kind, "params": c.params,
        "applies_to": c.applies_to, "severity": c.severity, "source": c.source,
        "rationale": c.rationale, "enabled": bool(c.enabled), "created_at": c.created_at,
    }


def task_view(conn, t: store.Task) -> dict:
    return {
        "id": t.id, "name": t.name, "kind": t.kind, "tags": t.tags or [],
        "episode_id": t.episode_id, "check_count": len(t.check_ids or []),
        "created_at": t.created_at,
    }


def task_detail(conn, t: store.Task) -> dict:
    checks = [check_view(c) for cid in (t.check_ids or []) if (c := store.get_check(conn, cid))]
    return {**asdict(t), "checks": checks}


def paginate(items: list, limit: int | None, offset: int):
    return items[offset : offset + limit] if limit is not None else items[offset:]
