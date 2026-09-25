"""`touchstone review` — open a review room over the dataset's finished tasks and trials."""

from __future__ import annotations

import os

import typer

from . import app
from ._common import _db


@app.command()
def review(
    dataset: str = typer.Option("touchstone", help="Dataset directory under the project root."),
    jobs_dir: str = typer.Option("", "--jobs-dir", help="Harbor jobs dir; default <dataset>/jobs."),
    topic: str = typer.Argument("review", help="What the room is reviewing."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open a review room and print (and open) its URL. Needs `touchstone serve` running."""
    os.environ["TOUCHSTONE_REVIEW_DATASET"] = dataset
    if jobs_dir:
        os.environ["TOUCHSTONE_REVIEW_JOBS_DIR"] = jobs_dir
    from ..config import load_settings
    from ..interview import rooms
    from ..review.agent import ReviewAgent

    settings = load_settings()
    host, port = "127.0.0.1", 8765
    with _db() as conn:
        room = rooms.open(conn, task_id=None, topic=topic)
        rooms.post(conn, room.id, "agent", "assistant",
                   ReviewAgent(None, conn, room, settings).open_statement())

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
