"""Shared CLI helpers: db handles, and the one-sentence-then-exit-1 error pattern."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import typer

from .. import store
from ..config import load_settings


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _count_needs_review(review_root: Path) -> int:
    if not review_root.is_dir():
        return 0
    return sum(1 for d in review_root.iterdir() if (d / "gate.json").is_file())


def _first_gate_reason(review_root: Path) -> str:
    """The first parked task's reason from its gate.json, for the empty-tasks message."""
    if review_root.is_dir():
        for d in sorted(review_root.iterdir()):
            gate = d / "gate.json"
            if gate.is_file():
                info = json.loads(gate.read_text(encoding="utf-8"))
                reason = info.get("reason") or (
                    f"{info.get('failed_side')} failed "
                    f"(oracle={info.get('oracle')} nop={info.get('nop')})")
                return reason.splitlines()[0]
    return "see report.md"


def _survey_report_reason(root: Path) -> str:
    """Why tasks were parked: the gate-skip reason from report.md when the whole gate was skipped,
    else the first parked task's gate reason."""
    report = root / "report.md"
    if report.is_file():
        for line in report.read_text(encoding="utf-8").splitlines():
            if line.startswith("Skipped: "):
                return line[len("Skipped: "):].strip()
    return _first_gate_reason(root / "needs-review")


def _require_tasks(dataset: str) -> None:
    """One clear line + exit 1 when <dataset>/tasks holds no task — never Harbor's raw
    'Either datasets or tasks must be provided' traceback."""
    root = Path(dataset)
    tasks = root / "tasks"
    if tasks.is_dir() and any((d / "task.toml").is_file() for d in tasks.iterdir()):
        return
    n = _count_needs_review(root / "needs-review")
    _fail(f"No tasks in {tasks}. The survey parked {n} in {root / 'needs-review'} "
          f"(reason from the survey report: {_survey_report_reason(root)}). "
          "Fix the cause and run `touchstone survey --force`.")


def _debug() -> bool:
    """Tracebacks are opt-in: --debug (which sets this) or TOUCHSTONE_DEBUG=1."""
    return bool(os.environ.get("TOUCHSTONE_DEBUG"))


@contextmanager
def _daemon_guard():
    """Turn a missing Docker daemon into one clear line + exit 1, never a traceback (unless
    --debug). The single place the CLI renders the shared DockerDaemonError."""
    from ..harbor.run import DockerDaemonError

    try:
        yield
    except DockerDaemonError as exc:
        if _debug():
            raise
        _fail(str(exc))


@contextmanager
def _fail_on(exc_types, message: str = "{exc}"):
    """Turn an expected exception into one clear sentence on stderr and exit 1."""
    try:
        yield
    except exc_types as exc:
        _fail(message.format(exc=exc))


def _root():
    """The project root that holds the `touchstone/` dataset, beside `.touchstone/`."""
    return load_settings().root


def _open_db():
    try:
        return store.connect(load_settings().db_path)
    except (OSError, sqlite3.Error) as exc:
        _fail(f"cannot open database: {exc}")


@contextmanager
def _db():
    """Open the configured db, always closing it when the block exits."""
    conn = _open_db()
    try:
        yield conn
    finally:
        conn.close()


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _configure_logging() -> None:
    """Touchstone's own INFO lines (review tool calls, regrades) must reach the server log."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _build_server(host: str, port: int):
    """A uvicorn Server for the Touchstone app, shared by `serve` (foreground) and `review`
    (a daemon thread). Its `should_exit` flag is how the caller stops it."""
    import uvicorn

    from ..server import create_app

    _configure_logging()
    config = uvicorn.Config(create_app(load_settings()), host=host, port=port, log_level="info")
    return uvicorn.Server(config)
