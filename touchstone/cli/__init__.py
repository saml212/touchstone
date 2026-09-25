"""Touchstone CLI: init, doctor, demo, serve, interview, bench, jobs. Errors exit non-zero."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import typer

from .. import store
from ..config import DEFAULT_DB, load_settings
from ._common import _fail, _installed

app = typer.Typer(add_completion=True)

_LOOP = (
    ("capture", "import touchstone; touchstone.trace()  (or touchstone demo)"),
    ("bench", "touchstone bench -m <provider/model>"),
    ("review", "touchstone serve  /  touchstone interview"),
)
_EMPTY_SIGNALS = {"episodes": 0, "rooms": 0}


def _next_step() -> str:
    """This project's one next move — the same hint the Overview page shows."""
    from ..overview import next_step, project_signals

    settings = load_settings()
    if not settings.db.exists():  # a bare `touchstone` must not create a database
        return next_step(_EMPTY_SIGNALS)
    conn = store.connect(settings.db_path)
    try:
        return next_step(project_signals(conn))
    finally:
        conn.close()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Touchstone — prove a cheaper model is good enough before you switch."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("Touchstone — the loop:")
    for i, (label, cmd) in enumerate(_LOOP, 1):
        typer.echo(f"  {i}. {label:9} {cmd}")
    typer.echo(f"\nNext: {_next_step()}")

TOML_TEMPLATE = f"""# Touchstone config. Keys above a [table] header are top-level.
db_path = "{DEFAULT_DB}"
provider = "scripted"
agent_provider = "claude-cli"   # mines, interviews, teaches: claude-cli | codex-cli | openai:<m>
# keychain_prefix = "touchstone-"   # Keychain items <prefix>openai-api-key / anthropic-api-key
{{keychain_prefix}}
[speech]
mode = "local"                 # local | realtime (OpenAI Realtime, needs OPENAI_API_KEY)
stt = "none"
tts = "browser"

[survey]
provider = "claude-cli"        # read-only code-mapping agent: claude-cli | codex-cli
# model = ""                     # provider default when empty
fidelity_threshold = 0.8       # a simulator below this is flagged, never silently used
names = []                     # full names to scrub from recordings before they enter files
"""


@app.command()
def init(
    path: str = typer.Option("touchstone.toml", help="Config file to write."),
    keychain_prefix: str = typer.Option(
        None, "--keychain-prefix",
        help="Pin the macOS Keychain service prefix (default: touchstone-).",
    ),
) -> None:
    """Write touchstone.toml and create the .touchstone/ db directory."""
    cfg = Path(path)
    if cfg.exists():
        typer.echo(f"{cfg} already exists; leaving it untouched.")
    else:
        prefix_line = f'keychain_prefix = "{keychain_prefix}"\n' if keychain_prefix else ""
        cfg.write_text(TOML_TEMPLATE.format(keychain_prefix=prefix_line), encoding="utf-8")
        typer.echo(f"wrote {cfg}")
    settings = load_settings(cfg)
    try:
        store.connect(settings.db_path).close()
    except (OSError, sqlite3.Error) as exc:
        _fail(f"cannot open database: {exc}")
    typer.echo(f"initialized db at {settings.db}")


def _doctor_rows(settings) -> list[tuple[str, str, str]]:
    """One (component, status, detail) row per environment check. No installs, no network calls."""
    import shutil

    from ..interview.speech import speech_status
    from ..llm import provider_statuses

    rows: list[tuple[str, str, str]] = [
        ("python", "ok", sys.version.split()[0]),
        ("database", "ok",
         f"{settings.db} ({'exists' if settings.db.exists() else 'not created'})"),
    ]
    for name in ("openai", "anthropic", "litellm"):
        ok = _installed(name)
        rows.append((f"sdk: {name}", "ok" if ok else "missing",
                     "importable" if ok else "not installed"))
    for st in provider_statuses(settings):
        rows.append((f"provider: {st.name}", "ok" if st.ok else "missing", st.detail))
    for r in speech_status(settings):
        rows.append((f"speech: {r.kind}", "ok", f"{r.name} — {r.detail}"))
    for tool in ("harbor", "docker", "ffmpeg"):
        path = shutil.which(tool)
        rows.append((f"tool: {tool}", "ok" if path else "missing",
                     path or "not on PATH"))
    return rows


@app.command()
def doctor() -> None:
    """Report python, db, SDKs, providers, speech, and tools in one table."""
    rows = _doctor_rows(load_settings())
    widths = [max(len(r[i]) for r in [("component", "status", "detail"), *rows]) for i in range(3)]
    header = ("component", "status", "detail")
    typer.echo("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    typer.echo("  ".join("-" * widths[i] for i in range(3)))
    for comp, status, detail in rows:
        typer.echo(f"{comp.ljust(widths[0])}  {status.ljust(widths[1])}  {detail}")


@app.command()
def demo(
    n: int = typer.Option(30, help="Number of episodes to generate."),
    db: str = typer.Option(None, help="Override db path."),
) -> None:
    """Run the built-in scripted support agent and capture real episodes."""
    import touchstone

    from ..demo import run_demo

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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8765, help="Bind port."),
) -> None:
    """Run the local interview + review server."""
    import uvicorn

    from ..server import create_app

    typer.echo(f"Touchstone UI on http://{host}:{port}")
    uvicorn.run(create_app(load_settings()), host=host, port=port)


# Register the remaining top-level commands. Imported last so `app` and the shared helpers above
# already exist when each module binds onto them.
from . import bench, interview, survey  # noqa: E402,F401
