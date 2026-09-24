"""`touchstone tasks` — list, show, and sync replay task directories."""

from __future__ import annotations

import json

import typer

from .. import tasks as tasks_mod
from ._common import _db, _fail, _root

tasks_app = typer.Typer(help="Inspect and rebuild replay tasks.", no_args_is_help=True)


@tasks_app.command("list")
def tasks_list(tag: str = typer.Option(None, "--tag", help="Filter by tag.")) -> None:
    """List tasks with their status, check count, and tags."""
    for t in tasks_mod.list_tasks(_root(), tag=tag):
        flag = "ok " if t.status == "active" else "rej"
        typer.echo(f"{t.name}  [{flag}] {len(t.checks)} checks  ({','.join(t.tags or [])})")


@tasks_app.command("show")
def tasks_show(name: str) -> None:
    """Show one task: context turns, reference, and its checks."""
    task = tasks_mod.get_task(_root(), name)
    if task is None:
        _fail(f"no task {name!r}")
    typer.echo(f"name:   {task.name}")
    typer.echo(f"status: {task.status}{f' — {task.reason}' if task.reason else ''}")
    typer.echo(f"tags:   {', '.join(task.tags or [])}")
    typer.echo("context:")
    for m in (task.context or {}).get("messages", []):
        content = m.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        typer.echo(f"  {m.get('role', '?'):9} {text[:100]}")
    typer.echo(f"reference: {json.dumps(task.reference, ensure_ascii=False)}")
    typer.echo("checks:")
    for c in task.checks:
        typer.echo(f"  {c.severity:4} {c.kind:16} {c.name} [{c.source}]")


@tasks_app.command("sync")
def tasks_sync() -> None:
    """Re-materialise every task directory from the captured episodes + current policies."""
    from ..mine import sync

    with _db() as conn:
        count = sync(conn, _root())
    typer.echo(f"synced {count} task(s)")
