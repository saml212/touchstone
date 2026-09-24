"""`touchstone tasks` — list, show, and sync replay task directories."""

from __future__ import annotations

import json

import typer

from .. import tasks as tasks_mod
from ._common import _db, _fail, _root

tasks_app = typer.Typer(help="Inspect and rebuild replay tasks.", no_args_is_help=True)

_QUEUES = ("active", "needs_checks", "needs_solution")
_QUEUE_HINT = {
    "active": "ready for benchmarks",
    "needs_checks": "interview to add a check that measures the work",
    "needs_solution": "ask a teacher (or a human) for a passing reply",
}


@tasks_app.command("list")
def tasks_list(tag: str = typer.Option(None, "--tag", help="Filter by tag.")) -> None:
    """List tasks grouped by work queue (active / needs_checks / needs_solution) with counts."""
    tasks = tasks_mod.list_tasks(_root(), tag=tag)
    by_status = {q: [t for t in tasks if t.status == q] for q in _QUEUES}
    for status in _QUEUES:
        group = by_status[status]
        typer.echo(f"{status}  ({len(group)}) — {_QUEUE_HINT[status]}")
        for t in group:
            typer.echo(f"  {t.name}  {len(t.checks)} checks  ({','.join(t.tags or [])})")


@tasks_app.command("show")
def tasks_show(name: str) -> None:
    """Show one task: context turns, reference, and its checks."""
    task = tasks_mod.get_task(_root(), name)
    if task is None:
        _fail(f"no task {name!r}")
    typer.echo(f"name:   {task.name}")
    reason = f" — {task.status_reason}" if task.status_reason else ""
    typer.echo(f"status: {task.status}{reason}")
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
