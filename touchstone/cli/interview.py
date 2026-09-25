"""`touchstone interview` — open a review room and print its URL."""

from __future__ import annotations

import typer

from . import app
from ._common import _db, _root


@app.command()
def interview(
    topic: str = typer.Argument("quality review", help="What the room is reviewing."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open a review room and print (and open) its URL."""
    from ..interview import rooms
    from ..interview.agent import Interviewer

    host, port = "127.0.0.1", 8765
    root = _root()
    with _db() as conn:
        room = rooms.open(conn, task_id=None, topic=topic)
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
