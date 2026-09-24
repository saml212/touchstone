"""`touchstone tasks` — list, show, attach, and detach replay tasks."""

from __future__ import annotations

import json

import typer

from .. import store
from ._common import _db, _fail, _fail_on

tasks_app = typer.Typer(help="Inspect and edit replay tasks.", no_args_is_help=True)


@tasks_app.command("list")
def tasks_list(tag: str = typer.Option(None, "--tag", help="Filter by tag.")) -> None:
    """List tasks with their tags and attached-check counts."""
    with _db() as conn:
        for t in store.list_tasks(conn, tag):
            tags = ",".join(t.tags or [])
            typer.echo(f"{t.id}  [{len(t.check_ids or [])} checks]  {t.name}  ({tags})")


@tasks_app.command("show")
def tasks_show(task_id: str) -> None:
    """Show one task: context turns, reference, and attached checks."""
    with _db() as conn:
        task = store.get_task(conn, task_id)
        if task is None:
            _fail(f"no task with id {task_id}")
        typer.echo(f"id:     {task.id}")
        typer.echo(f"name:   {task.name}")
        typer.echo(f"tags:   {', '.join(task.tags or [])}")
        typer.echo("context:")
        for m in (task.context or {}).get("messages", []):
            content = m.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            typer.echo(f"  {m.get('role', '?'):9} {text[:100]}")
        typer.echo(f"reference: {json.dumps(task.reference, ensure_ascii=False)}")
        typer.echo("checks:")
        for cid in task.check_ids or []:
            c = store.get_check(conn, cid)
            typer.echo(f"  {cid}  {c.kind if c else '(missing)'}")


@tasks_app.command("attach")
def tasks_attach(task_id: str, check_id: str) -> None:
    """Attach a check to a task."""
    _edit_task_checks(task_id, check_id, attach=True)


@tasks_app.command("detach")
def tasks_detach(task_id: str, check_id: str) -> None:
    """Detach a check from a task."""
    _edit_task_checks(task_id, check_id, attach=False)


def _edit_task_checks(task_id: str, check_id: str, attach: bool) -> None:
    with _db() as conn, _fail_on(ValueError):
        store.set_task_check(conn, task_id, check_id, attach)
    verb, prep = ("attached", "to") if attach else ("detached", "from")
    typer.echo(f"{verb} {check_id} {prep} {task_id}")
