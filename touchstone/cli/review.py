"""`touchstone review` — one command: start the server if needed, open the room, talk in the
browser. The whole voice review experience lives behind this single command."""

from __future__ import annotations

import os

import typer

from . import app
from ._common import _build_server, _db, _require_tasks


@app.command()
def review(
    dataset: str = typer.Option("touchstone", help="Dataset directory under the project root."),
    jobs_dir: str = typer.Option("", "--jobs-dir", help="Harbor jobs dir; default <dataset>/jobs."),
    topic: str = typer.Argument("review", help="What the room is reviewing."),
    port: int = typer.Option(8765, "--port", help="Server port (started here if not running)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open a review room and talk to it in the browser. Starts the server itself if it isn't up."""
    _require_tasks(dataset)
    os.environ["TOUCHSTONE_REVIEW_DATASET"] = dataset
    if jobs_dir:
        os.environ["TOUCHSTONE_REVIEW_JOBS_DIR"] = jobs_dir
    from ..config import load_settings
    from ..interview import rooms
    from ..review.agent import ReviewAgent

    settings = load_settings()
    host = "127.0.0.1"
    with _db() as conn:
        room = rooms.open(conn, task_id=None, topic=topic)
        rooms.post(conn, room.id, "agent", "assistant",
                   ReviewAgent(None, conn, room, settings).open_statement())

    url = f"http://{host}:{port}/rooms/{room.id}"
    if _server_up(host, port):
        _announce(url, no_open)
        return
    server = _start_server(host, port)
    _announce(url, no_open)
    typer.echo("Talk in your browser. Ctrl-C here when you're done.")
    _wait_for_exit(server)


def _announce(url: str, no_open: bool) -> None:
    typer.echo(f"Review room: {url}")
    if not no_open:
        _open_browser(url)


def _open_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


def _start_server(host: str, port: int):
    """Start the Touchstone server in a daemon thread and wait for it to answer /api/health.
    Returns the uvicorn Server; set its `should_exit` to stop it."""
    import threading

    server = _build_server(host, port)
    threading.Thread(target=server.run, daemon=True).start()
    _await_health(host, port)
    return server


def _await_health(host: str, port: int, tries: int = 50) -> None:
    import time

    for _ in range(tries):  # ~10s at 0.2s/try
        if _server_up(host, port):
            return
        time.sleep(0.2)


def _wait_for_exit(server) -> None:
    """Block until the user interrupts, then ask the background server to stop and exit cleanly."""
    import time

    try:
        while not getattr(server, "should_exit", False):
            time.sleep(0.2)
    except KeyboardInterrupt:
        server.should_exit = True


def _server_up(host: str, port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://{host}:{port}/api/health", timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False
