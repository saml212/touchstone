"""Shared CLI helpers: db handles, and the one-sentence-then-exit-1 error pattern."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from contextlib import contextmanager

import typer

from .. import store
from ..config import load_settings


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


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
