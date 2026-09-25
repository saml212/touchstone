"""Shared route dependencies.

`get_conn` yields a per-request SQLite connection; `get_root` gives the project root that holds the
`touchstone/` dataset. `paginate` slices a list for the list endpoints.
"""

from __future__ import annotations

from fastapi import Request

from ... import store


def get_conn(request: Request):
    conn = store.connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_root(request: Request):
    return request.app.state.settings.root


def paginate(items: list, limit: int | None, offset: int):
    return items[offset : offset + limit] if limit is not None else items[offset:]
