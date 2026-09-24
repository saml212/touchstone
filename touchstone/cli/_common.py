"""Shared CLI helpers: db handles, and the one-sentence-then-exit-1 error pattern."""

from __future__ import annotations

import importlib.util
import sqlite3
from contextlib import contextmanager

import typer

from .. import store
from ..config import load_settings


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


@contextmanager
def _fail_on(exc_types, message: str = "{exc}"):
    """Turn an expected exception into one clear sentence on stderr and exit 1."""
    try:
        yield
    except exc_types as exc:
        _fail(message.format(exc=exc))


def _root():
    """The project root that holds tasks/, checks.toml and benchmarks/."""
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


def _agent_provider(spec: str):
    from ..llm import provider_from_spec

    try:
        return provider_from_spec(spec)
    except Exception as exc:  # no key / binary: mine on statistics alone
        typer.echo(f"note: provider {spec!r} unavailable ({exc}); mining stats only", err=True)
        return None


def _judge_provider():
    from ..llm import provider_from_spec

    try:
        return provider_from_spec(load_settings().provider)
    except Exception:  # a judge without a working provider degrades to 'skipped'
        return None
