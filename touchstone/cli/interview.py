"""`touchstone interview` — open an interview room for a task and print its URL."""

from __future__ import annotations

import typer

from .. import tasks as tasks_mod
from . import app
from ._common import _db, _fail, _root


@app.command()
def interview(
    name: str,
    topic: str = typer.Option(None, "--topic", help="Room topic (defaults to the task name)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open an interview room for a task and print (and open) its URL."""
    from ..interview import rooms
    from ..interview.agent import Interviewer

    host, port = "127.0.0.1", 8765
    root = _root()
    task = tasks_mod.get_task(root, name)
    if task is None:
        _fail(f"no task {name!r}")
    with _db() as conn:
        room = rooms.open(conn, name, topic or f"review of {task.name}")
        rooms.post(conn, room.id, "agent", "assistant",
                   Interviewer(None, conn, room, root).open_statement())

    url = f"http://{host}:{port}/rooms/{room.id}"
    typer.echo(url)
    if not _server_up(host, port):
        typer.echo(f"Server not running — start it with `touchstone serve`, then open {url}.")
        return
    if not no_open:
        import webbrowser

        webbrowser.open(url)


def _server_up(host: str, port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://{host}:{port}/api/health", timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False
