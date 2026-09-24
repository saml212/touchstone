"""Tasks: list (filter + paginate), detail with its checks, and add/remove a check block."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ... import tasks as tasks_mod
from ...checks import Check
from ._deps import get_root, paginate, task_detail, task_view

router = APIRouter()


@router.get("/api/tasks")
def list_tasks(
    tag: str | None = None,
    limit: int | None = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    root=Depends(get_root),
) -> dict:
    tasks = tasks_mod.list_tasks(root, tag=tag)
    page = paginate(tasks, limit, offset)
    return {"total": len(tasks), "tasks": [task_view(t) for t in page]}


@router.get("/api/tasks/{name}")
def get_task(name: str, root=Depends(get_root)) -> dict:
    task = tasks_mod.get_task(root, name)
    if task is None:
        raise HTTPException(404, f"no task {name!r}")
    return task_detail(task)


@router.post("/api/tasks/{name}/checks")
def add_check(name: str, body: dict, root=Depends(get_root)) -> dict:
    if tasks_mod.get_task(root, name) is None:
        raise HTTPException(404, f"no task {name!r}")
    try:
        check = Check(kind=body.get("kind") or "", params=body.get("params") or {},
                      name=body.get("name") or body.get("kind") or "",
                      severity=body.get("severity", "hard"),
                      applies_to=body.get("applies_to", "final"),
                      rule=body.get("rule", ""), because=body.get("because", ""), source="manual")
        check.id = check.name
        check.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    tasks_mod.append_check(root, name, check)
    return task_detail(tasks_mod.get_task(root, name))


@router.delete("/api/tasks/{name}/checks/{check_name}")
def remove_check(name: str, check_name: str, root=Depends(get_root)) -> dict:
    if tasks_mod.get_task(root, name) is None:
        raise HTTPException(404, f"no task {name!r}")
    return task_detail(tasks_mod.remove_check(root, name, check_name))
