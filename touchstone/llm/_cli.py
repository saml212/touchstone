"""Subprocess plumbing shared by the CLI providers (claude-cli, codex-cli).

Runs a command in a throwaway temp dir with a scrubbed environment, enforces a timeout, and raises
`ProviderError` with a one-sentence message (never a secret) on timeout or non-zero exit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from ._http import ProviderError

# Markers of a transient CLI-side API failure (a Claude Code / Codex hiccup, not a bad request from
# us) that is worth one retry: the harness returns a non-zero exit with an error JSON result.
_TRANSIENT = ("is_error", "api_error", "api error", "overloaded", "rate limit", "429",
              "not found in available tools", "internal server error", "503", "500")


def require_binary(binary: str, hint: str) -> None:
    if shutil.which(binary) is None:
        raise ProviderError(hint)


def scrubbed_env(*drop: str) -> dict:
    env = dict(os.environ)
    for key in drop:
        env.pop(key, None)
    return env


@contextmanager
def temp_dir() -> Iterator[str]:
    with tempfile.TemporaryDirectory(prefix="touchstone-cli-") as path:
        yield path


def run(
    cmd: Sequence[str],
    *,
    label: str,
    cwd: str,
    env: dict,
    timeout: float,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            list(cmd),
            input=stdin_text if stdin_text is not None else "",
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"{label} timed out after {timeout:g}s.") from exc
    except OSError as exc:
        raise ProviderError(f"{label} could not be launched.") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        head = detail[0] if detail else "no output"
        raise ProviderError(f"{label} exited with code {proc.returncode}: {head}")
    return proc


def _is_transient(message: str) -> bool:
    m = message.lower()
    return "exited with code" in m and any(t in m for t in _TRANSIENT)


def run_retrying(cmd: Sequence[str], *, label: str, cwd: str, env: dict, timeout: float,
                 stdin_text: str | None = None, retries: int = 1, pause: float = 1.5,
                 sleep=time.sleep) -> subprocess.CompletedProcess:
    """Like `run`, but a non-zero exit that looks like a transient API error is retried once after a
    short pause. A timeout, a missing binary, or a plain bad request is raised immediately."""
    for attempt in range(retries + 1):
        try:
            return run(cmd, label=label, cwd=cwd, env=env, timeout=timeout, stdin_text=stdin_text)
        except ProviderError as exc:
            if attempt == retries or not _is_transient(str(exc)):
                raise
            sleep(pause)
    raise AssertionError("unreachable")  # pragma: no cover
