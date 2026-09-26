"""`touchstone doctor` — report python, db, SDKs, providers, speech, and tools in one table.

Split out of the CLI package root so each file stays small. No installs; the only network call is
the harbor-host reachability probe, and only when a remote host is configured.
"""

from __future__ import annotations

import sys

import typer

import touchstone.cli as _cli  # read version helpers through the package so monkeypatch is seen

from ..config import load_settings
from . import app


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


def _version_row() -> tuple[str, str, str]:
    """touchstone's own version, flagged `skew` when the project pins a different one. doctor is the
    consistency command, so the version-skew note (elsewhere on stderr) is also a row in its table —
    a user piping `doctor` still sees the skew that made the stranger trust an old pinned doctor."""
    running = _cli._package_version()
    pinned = _cli._project_pin()
    if pinned and pinned != running:
        return ("touchstone", "skew", f"running {running}; project pins {pinned} "
                "(uv sync --upgrade-package touchstone-bench)")
    return ("touchstone", "ok", running)


def _doctor_rows(settings) -> list[tuple[str, str, str]]:
    """One (component, status, detail) row per environment check."""
    import shutil

    from ..interview.speech import speech_status
    from ..llm import provider_statuses

    rows: list[tuple[str, str, str]] = [
        _version_row(),
        ("python", "ok", sys.version.split()[0]),
        ("database", "ok",
         f"{settings.db} ({'exists' if settings.db.exists() else 'not created'})"),
    ]
    for name in ("openai", "anthropic", "litellm"):
        ok = _cli._installed(name)
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
