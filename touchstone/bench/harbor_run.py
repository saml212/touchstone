"""Run an exported Harbor task dir via the `harbor` CLI, or explain what is missing.

`harbor run` needs the `harbor` binary and a working Docker daemon. When either is absent we print
the exact command to run and where it failed, and return a non-zero code — we never install
anything.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _docker_ok() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker is not on PATH"
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=20, text=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"`docker info` could not run ({exc})"
    if proc.returncode != 0:
        return False, "`docker info` failed (is the Docker daemon running?)"
    return True, "docker ok"


def command(task_dir: str | Path, agent: str | None, extra: list[str] | None = None) -> list[str]:
    cmd = ["harbor", "run", "-p", str(task_dir)]
    if agent:
        cmd += ["-a", agent]
    cmd += list(extra or [])
    return cmd


def run_task(
    task_dir: str | Path,
    agent: str | None = None,
    extra: list[str] | None = None,
    *,
    echo=print,
) -> int:
    """Shell out to `harbor run`. Return its exit code, or 1 with guidance if a dep is missing."""
    cmd = command(task_dir, agent, extra)
    printable = " ".join(cmd)

    missing = []
    if shutil.which("harbor") is None:
        missing.append("the `harbor` CLI is not on PATH (install Harbor to run tasks)")
    ok, detail = _docker_ok()
    if not ok:
        missing.append(detail)

    if missing:
        echo("cannot run the Harbor task here:")
        for m in missing:
            echo(f"  - {m}")
        echo("run this where Docker and Harbor are available:")
        echo(f"  {printable}")
        return 1

    try:
        return subprocess.run(cmd).returncode
    except OSError as exc:
        echo(f"failed to launch harbor: {exc}")
        echo(f"  {printable}")
        return 1
