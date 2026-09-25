"""Measure how faithfully a simulator reproduces the recorded calls.

Start the simulator on a free port, seed it, replay every recorded call through the customer's real
tool functions (pointed at the simulator), and compare each returned value with the recorded one —
after masking volatile fields (ids, timestamps, tokens) so a fresh id is not a mismatch. The process
is always killed; a simulator that never becomes healthy scores 0.0 with its log tail, so the survey
continues and flags it.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from ..config import Settings
from .fidelity_db import dump_db  # noqa: F401 re-exported (task criteria import it from here)
from .fidelity_mask import _compare, _mask, _mask_field, masked_equal  # noqa: F401 re-exported
from .recordings import ToolEvent
from .scrub import Scrubber

_HEALTH_TIMEOUT = 20.0
_REPLAY_TIMEOUT = 180.0


class _SimError(RuntimeError):
    """The simulator could not be started, seeded, or replayed against."""


# ---- simulator process -----------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(log_path: Path, n: int = 1500) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return "(no simulator log)"


def _start_sim(sim_dir: Path, port: int, log_path: Path):
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "app.py", str(port)], cwd=str(sim_dir),
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log


def _health_ok(base: str) -> bool:
    try:
        return httpx.get(f"{base}/__health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _await_health(proc, base: str, log_path: Path) -> None:
    deadline = time.time() + _HEALTH_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise _SimError(f"simulator exited during startup:\n{_tail(log_path)}")
        if _health_ok(base):
            return
        time.sleep(0.1)
    raise _SimError(f"simulator never became healthy:\n{_tail(log_path)}")


def _reset(base: str) -> None:
    try:
        httpx.post(f"{base}/__reset", timeout=5.0)
    except httpx.HTTPError:
        pass  # some simulators seed at startup; a missing /__reset is not fatal


def _kill(proc, log) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if log is not None:
        log.close()


# ---- replay + score --------------------------------------------------------


def _replay_cmd(repo: Path, spec_path: str, settings: Settings) -> list[str]:
    if settings.survey_python:
        return [settings.survey_python, "-m", "touchstone.survey.replay", spec_path]
    return ["uv", "run", "--project", str(repo), "python", "-m",
            "touchstone.survey.replay", spec_path]


def _run_spec(repo: Path, spec: dict, settings: Settings) -> list[dict]:
    fd, spec_path = tempfile.mkstemp(suffix=".json", prefix="ts-replay-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, default=str)
    try:
        proc = subprocess.run(_replay_cmd(repo, spec_path, settings), cwd=str(repo),
                              capture_output=True, text=True, timeout=_REPLAY_TIMEOUT)
    finally:
        os.unlink(spec_path)
    if proc.returncode != 0:
        raise _SimError(f"replay failed:\n{(proc.stderr or proc.stdout)[-1500:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise _SimError(f"replay output was not JSON:\n{proc.stdout[-500:]}") from exc


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _external_host(host: str | None) -> bool:
    return bool(host) and host not in _LOOPBACK_HOSTS and not host.startswith("127.")


def _shim_host(host: str | None, base_url_env) -> bool:
    """Shim the host when it is the only redirect (no base_url_env) or a real external host — so a
    base_url_env the map got wrong (an API-key var) can't let replay reach the real service."""
    return bool(host) and (not base_url_env or _external_host(host))


def _replay_spec(calls: list[ToolEvent], base: str, ctx: dict) -> dict:
    """A replay spec that repoints the service at `base`: by env var when the map has one, and/or by
    rewriting the host with the net shim (simulators = {host: base}). `invoke` in ctx routes calls
    through the generated agent/invoke.py."""
    spec = {"tools": ctx["tools"],
            "calls": [{"tool": c.tool, "arguments": c.arguments} for c in calls]}
    if ctx.get("base_url_env"):
        spec["base_url_env"] = ctx["base_url_env"]
        spec["base_url"] = base
    if _shim_host(ctx.get("host"), ctx.get("base_url_env")):
        spec["simulators"] = {ctx["host"]: base}
    if ctx.get("invoke"):
        spec["invoke"] = ctx["invoke"]
    return spec


def _run_replay(repo: Path, calls: list[ToolEvent], base: str, ctx: dict,
                settings: Settings) -> list[dict]:
    return _run_spec(repo, _replay_spec(calls, base, ctx), settings)


def _failure(call: ToolEvent, got, scrub: Scrubber) -> dict:
    return {"tool": call.tool, "arguments": scrub.scrub(call.arguments),
            "expected": scrub.scrub(call.output), "got": scrub.scrub(got)}


def _score(calls: list[ToolEvent], got_list: list[dict], threshold: float,
           scrub: Scrubber) -> dict:
    failures: list[dict] = []
    masked: set = set()
    reproduced = 0
    for call, got in zip(calls, got_list, strict=False):
        ok, m = _compare(call.output, got.get("got"))
        masked |= m
        if ok:
            reproduced += 1
        elif len(failures) < 20:
            failures.append(_failure(call, got.get("got"), scrub))
    n = len(calls)
    return {"calls": n, "reproduced": reproduced,
            "score": round(reproduced / n, 4) if n else 1.0,
            "threshold": threshold, "masked_keys": sorted(masked), "failures": failures}


def _failed_result(calls: list[ToolEvent], threshold: float, detail: str) -> dict:
    return {"calls": len(calls), "reproduced": 0, "score": 0.0, "threshold": threshold,
            "masked_keys": [], "failures": [{"error": detail[-1500:]}]}


_NO_REDIRECT = (
    "service base URL is a constant the net shim cannot rewrite (no base_url_env and no http host "
    "to redirect), so fidelity was not measured — survey never calls the real service"
)


def _redirectable(ctx: dict) -> bool:
    """True when replay can reach the simulator instead of the real service: either an env var
    overrides the base URL, or the net shim can rewrite a constant http host."""
    return bool(ctx.get("base_url_env")) or bool(ctx.get("host") and ctx.get("kind") == "http")


# ---- state capture (for task criteria) -------------------------------------


def _start_mounts(mounts: list[dict]) -> list[dict]:
    started: list[dict] = []
    for mount in mounts:
        port = _free_port()
        log_path = mount["sim_dir"] / ".sim.log"
        proc, log = _start_sim(mount["sim_dir"], port, log_path)
        base = f"http://127.0.0.1:{port}"
        _await_health(proc, base, log_path)
        _reset(base)
        started.append({"proc": proc, "log": log, "base": base, "sim_dir": mount["sim_dir"],
                        "env": mount.get("env"), "host": mount.get("host")})
    return started


def _snapshots(started: list[dict]) -> dict:
    return {s["sim_dir"].name: dump_db(s["sim_dir"] / "state.db") for s in started}


def _base_urls(started: list[dict]) -> dict:
    return {s["env"]: s["base"] for s in started if s["env"]}


def _sim_hosts(started: list[dict]) -> dict:
    """host -> simulator base for the net-shim rewrite: services with no env var, plus a shield for
    any real external host so a wrong base_url_env can never reach the real service."""
    return {s["host"]: s["base"] for s in started if _shim_host(s.get("host"), s["env"])}


@contextlib.contextmanager
def simulators_running(mounts: list[dict]):
    """Start each simulator (`mounts` = [{"sim_dir", "env", "host"}]) and yield
    ({base_url_env: base}, {host: base}), always killing the processes.

    For a local check that needs the real services up (the packaged adapter check) rather than a
    fidelity score. Ports are free ports, so the base urls are yielded for the caller to pass on."""
    started = _start_mounts(mounts)
    try:
        yield _base_urls(started), _sim_hosts(started)
    finally:
        for s in started:
            _kill(s["proc"], s["log"])


def capture_state(mounts: list[dict], repo: Path, tools: dict, calls: list[ToolEvent],
                  settings: Settings, invoke: str | None = None) -> dict:
    """Start each simulator, seed it, dump state, replay `calls` (real tool functions), dump again.

    `mounts` is [{"sim_dir", "env"}]. Returns {"initial", "final", "replayed"} keyed by simulator
    name, or {"error": ...} if a simulator never became healthy or the replay failed."""
    started: list[dict] = []
    try:
        started = _start_mounts(mounts)
        initial = _snapshots(started)
        spec = {"base_urls": _base_urls(started), "simulators": _sim_hosts(started), "tools": tools,
                "calls": [{"tool": c.tool, "arguments": c.arguments} for c in calls]}
        if invoke:
            spec["invoke"] = invoke
        replayed = _run_spec(repo, spec, settings)
        return {"initial": initial, "final": _snapshots(started), "replayed": replayed}
    except _SimError as exc:
        return {"error": str(exc)}
    finally:
        for s in started:
            _kill(s["proc"], s["log"])


def measure_service(sim_dir: Path, repo: Path, calls: list[ToolEvent], ctx: dict,
                    settings: Settings, scrub: Scrubber, invoke: str | None = None) -> dict:
    """Fidelity of the simulator in `sim_dir` against `calls`. Always kills the process."""
    if invoke:
        ctx = {**ctx, "invoke": invoke}
    threshold = settings.survey_fidelity_threshold
    if calls and not _redirectable(ctx):
        # Nothing can repoint the tool at the simulator, so the real (maybe prod) base URL would be
        # hit. Never do that: flag the simulator below-threshold with an honest reason instead.
        return _failed_result(calls, threshold, _NO_REDIRECT)
    port = _free_port()
    log_path = sim_dir / ".sim.log"
    proc = log = None
    try:
        proc, log = _start_sim(sim_dir, port, log_path)
        base = f"http://127.0.0.1:{port}"
        _await_health(proc, base, log_path)
        _reset(base)
        got_list = _run_replay(repo, calls, base, ctx, settings)
        return _score(calls, got_list, threshold, scrub)
    except _SimError as exc:
        return _failed_result(calls, threshold, str(exc))
    finally:
        _kill(proc, log)
