"""Tasks: list (filter + paginate), detail with resolved checks, and attach/detach."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ... import store
from ._deps import get_conn, paginate, task_detail, task_view

router = APIRouter()


@router.get("/api/tasks")
def list_tasks(
    tag: str | None = None,
    limit: int | None = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
) -> dict:
    tasks = store.list_tasks(conn, tag)
    page = paginate(tasks, limit, offset)
    return {"total": len(tasks), "tasks": [task_view(conn, t) for t in page]}


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str, conn=Depends(get_conn)) -> dict:
    task = store.get_task(conn, task_id)
    if task is None:
        raise HTTPException(404, f"no task with id {task_id}")
    return task_detail(conn, task)


@router.post("/api/tasks/{task_id}/checks")
def attach_check(task_id: str, body: dict, conn=Depends(get_conn)) -> dict:
    check_id = (body.get("check_id") or "").strip()
    if not check_id:
        raise HTTPException(422, "check_id is required")
    try:
        task = store.set_task_check(conn, task_id, check_id, attach=True)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return task_detail(conn, task)


@router.delete("/api/tasks/{task_id}/checks/{check_id}")
def detach_check(task_id: str, check_id: str, conn=Depends(get_conn)) -> dict:
    try:
        task = store.set_task_check(conn, task_id, check_id, attach=False)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return task_detail(conn, task)
