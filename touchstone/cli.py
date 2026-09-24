"""Touchstone CLI. Stage 1: init, doctor, demo. Errors exit non-zero with one clear sentence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import typer

from . import store
from .config import DEFAULT_DB, load_settings

app = typer.Typer(
    help="Touchstone — prove a cheaper model is good enough before you switch.",
    no_args_is_help=True,
)

TOML_TEMPLATE = f"""# Touchstone config
db_path = "{DEFAULT_DB}"
provider = "scripted"
keychain_prefix = "touchstone"

[speech]
stt = "none"
tts = "browser"
"""


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.command()
def init(path: str = typer.Option("touchstone.toml", help="Config file to write.")) -> None:
    """Write touchstone.toml and create the .touchstone/ db directory."""
    cfg = Path(path)
    if cfg.exists():
        typer.echo(f"{cfg} already exists; leaving it untouched.")
    else:
        cfg.write_text(TOML_TEMPLATE, encoding="utf-8")
        typer.echo(f"wrote {cfg}")
    settings = load_settings(cfg)
    store.connect(settings.db_path).close()
    typer.echo(f"initialized db at {settings.db}")


@app.command()
def doctor() -> None:
    """Report environment: python, db, importable SDKs, available providers."""
    settings = load_settings()
    typer.echo(f"python:    {sys.version.split()[0]}")
    typer.echo(f"db path:   {settings.db}")
    typer.echo(f"db exists: {settings.db.exists()}")
    for name in ("openai", "anthropic", "litellm"):
        typer.echo(f"{name+':':10} {'importable' if _installed(name) else 'not installed'}")
    typer.echo("providers: scripted")


@app.command()
def demo(
    n: int = typer.Option(30, help="Number of episodes to generate."),
    db: str = typer.Option(None, help="Override db path."),
) -> None:
    """Run the built-in scripted support agent and capture real episodes."""
    import touchstone

    from .demo import run_demo

    settings = load_settings()
    db_path = db or settings.db_path
    touchstone.trace(db_path)
    run_demo(n=n)
    conn = store.connect(db_path)
    episodes = store.list_episodes(conn)
    resolved = len(store.list_episodes(conn, "resolved"))
    spans = sum(len(store.list_spans(conn, e.id)) for e in episodes)
    conn.close()
    typer.echo(
        f"captured {len(episodes)} episodes ({resolved} resolved), {spans} spans in {settings.db}"
    )


if __name__ == "__main__":
    app()
