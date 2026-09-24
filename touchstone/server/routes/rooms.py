"""GET /api/rooms — every interview room, open ones first, each with its task name."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import store
from ._deps import get_conn

router = APIRouter()


@router.get("/api/rooms")
def list_rooms(conn=Depends(get_conn)) -> dict:
    rooms = store.list_rooms(conn)
    rooms.sort(key=lambda r: (r.closed_at is not None, r.id))
    out = []
    for r in rooms:
        task = store.get_task(conn, r.task_id) if r.task_id else None
        out.append({
            "id": r.id, "task_id": r.task_id, "task_name": task.name if task else None,
            "topic": r.topic, "created_at": r.created_at, "closed_at": r.closed_at,
        })
    return {"rooms": out}
