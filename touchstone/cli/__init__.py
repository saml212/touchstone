"""Touchstone CLI: init, doctor, demo, serve, review, bench, jobs. Errors exit non-zero."""

from __future__ import annotations

import os
import sqlite3
import sys
import tomllib
from pathlib import Path

import typer

from .. import store
from ..config import DEFAULT_DB, load_settings
from ._common import _fail, _installed

app = typer.Typer(add_completion=True)

_LOOP = (
    ("capture", "import touchstone; touchstone.trace()  (or touchstone demo)"),
    ("bench", "touchstone bench -m <provider/model>"),
    ("review", "touchstone serve  /  touchstone review"),
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


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        return _v("touchstone-bench")
    except PackageNotFoundError:
        return "0.0.0"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_package_version())
        raise typer.Exit()


# Distribution names touchstone has shipped under (the example locked the old `touchstone`).
_DIST_NAMES = ("touchstone-bench", "touchstone")


def _project_pin() -> str | None:
    """The touchstone version the cwd's uv.lock resolves as a dependency, or None when the cwd isn't
    a touchstone consumer (no lock, no touchstone dep, or the touchstone checkout itself)."""
    lock = Path.cwd() / "uv.lock"
    if not lock.is_file():
        return None
    try:
        data = tomllib.loads(lock.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    for pkg in data.get("package", []):
        source = pkg.get("source", {})
        if pkg.get("name") in _DIST_NAMES and "virtual" not in source and "editable" not in source:
            return pkg.get("version")
    return None


def _skew_note() -> None:
    """One line when the project pins a touchstone different from the CLI you're running — the skew
    that made the stranger's `uv run touchstone` resolve a version with no `survey` command."""
    pinned = _project_pin()
    running = _package_version()
    if pinned and pinned != running:
        typer.echo(f"Note: this project pins touchstone-bench {pinned}; you are running {running} "
                   "(uv sync --upgrade-package touchstone-bench).", err=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001 — consumed eagerly by the callback
        None, "--version", callback=_version_callback, is_eager=True,
        help="Print the version and exit."),
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks on error."),
) -> None:
    """Touchstone — turn your running agent into a benchmark, review it, train on it."""
    if debug:
        os.environ["TOUCHSTONE_DEBUG"] = "1"
    _skew_note()
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("Touchstone — the loop:")
    for i, (label, cmd) in enumerate(_LOOP, 1):
        typer.echo(f"  {i}. {label:9} {cmd}")
    typer.echo(f"\nNext: {_next_step()}")

TOML_TEMPLATE = f"""# Touchstone config. Keys above a [table] header are top-level.
db_path = "{DEFAULT_DB}"
provider = "scripted"
agent_provider = "claude-cli"   # surveys and reviews: claude-cli | codex-cli | openai:<m>
# keychain_prefix = "touchstone-"   # Keychain items <prefix>openai-api-key / anthropic-api-key
{{keychain_prefix}}
[speech]
mode = "auto"                  # auto | local | realtime (auto = realtime when a key resolves)
stt = "none"
tts = "browser"

[survey]
provider = "claude-cli"        # read-only code-mapping agent: claude-cli | codex-cli
# model = ""                     # provider default when empty
fidelity_threshold = 0.8       # a simulator below this is flagged, never silently used
names = []                     # full names to scrub from recordings before they enter files

[harbor]
# survey (its gate + baseline), bench, and train run in Docker. No daemon on this machine?
# Name a host you can ssh to and everything runs there instead:
# host = "my-build-box"
# remote_root = "/srv/touchstone"
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


def _harbor_host_row(settings) -> tuple[str, str, str]:
    """Reachability of the configured remote harbor host (an ssh probe, ConnectTimeout-bounded).
    When no host is configured, Touchstone runs Harbor against local Docker — reported as such."""
    import subprocess

    from ..harbor.run import _SSH_OPTS

    host = settings.harbor_host
    if not host:
        return ("harbor host", "ok", "not configured — using local docker")
    try:
        proc = subprocess.run(["ssh", *_SSH_OPTS, host, "true"], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return ("harbor host", "unreachable", f"{host}: {exc}")
    if proc.returncode == 0:
        return ("harbor host", "ok", f"{host} reachable")
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return ("harbor host", "unreachable", f"{host}: {detail[0] if detail else 'ssh failed'}")


def _doctor_rows(settings) -> list[tuple[str, str, str]]:
    """One (component, status, detail) row per environment check. No installs; the only network call
    is the harbor-host reachability probe, and only when a remote host is configured."""
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
    for tool in ("harbor", "ffmpeg"):
        path = shutil.which(tool)
        rows.append((f"tool: {tool}", "ok" if path else "missing",
                     path or "not on PATH"))
    rows.append(_docker_row())
    rows.append(_harbor_host_row(settings))
    if settings.harbor_host:
        rows.append(_harbor_host_docker_row(settings))
    return rows


def _docker_row() -> tuple[str, str, str]:
    """Probe the local Docker daemon, not just the binary — a false-green here sent the stranger
    into raw tracebacks (their daemon was down while doctor said docker was 'ok')."""
    from ..harbor.run import docker_daemon

    up, version = docker_daemon()
    if up:
        return ("tool: docker", "ok", version or "daemon running")
    return ("tool: docker", "missing",
            "not running — start Docker (Docker Desktop / colima start)")


def _harbor_host_docker_row(settings) -> tuple[str, str, str]:
    """The harbor host's Docker daemon, probed over ssh with the same `docker info`."""
    from ..harbor.run import remote_docker_daemon

    up, version = remote_docker_daemon(settings)
    if up:
        return ("harbor host docker", "ok", version or "daemon running")
    return ("harbor host docker", "missing",
            f"not running on {settings.harbor_host} — start Docker there")


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
def version() -> None:
    """Print the installed Touchstone version."""
    typer.echo(_package_version())


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
    """Run the local UI: overview, tasks, trials, review room, train."""
    from ._common import _build_server

    typer.echo(f"Touchstone UI on http://{host}:{port}")
    _build_server(host, port).run()


# Register the remaining top-level commands. Imported last so `app` and the shared helpers above
# already exist when each module binds onto them.
from . import bench, review, survey, train  # noqa: E402,F401
