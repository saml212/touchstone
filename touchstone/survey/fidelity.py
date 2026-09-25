"""Measure how faithfully a simulator reproduces the recorded calls.

Start the simulator on a free port, seed it, replay every recorded call for the service through the
customer's real tool functions (pointed at the simulator), and compare each returned value with the
recorded one — after masking volatile fields (ids, tickets, timestamps, tokens) so a fresh id does
not count as a mismatch. The simulator process is always killed. A simulator that never becomes
healthy scores 0.0 with the traceback tail, so the survey can continue and flag it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from ..config import Settings
from .recordings import ToolEvent
from .scrub import Scrubber

_VOLATILE_EXACT = {"id", "ticket", "timestamp", "ts", "token"}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_HEALTH_TIMEOUT = 20.0
_REPLAY_TIMEOUT = 180.0


class _SimError(RuntimeError):
    """The simulator could not be started, seeded, or replayed against."""


def _is_volatile(key: str) -> bool:
    k = key.lower()
    return (k in _VOLATILE_EXACT or k.endswith("_id")
            or k.startswith("created") or k.startswith("updated"))


def _mask(value, masked: set):
    if isinstance(value, dict):
        return {k: _mask_field(k, v, masked) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, masked) for v in value]
    if isinstance(value, str) and _ISO.search(value):
        masked.add("<iso-timestamp>")
        return "<ts>"
    return value


def _mask_field(key: str, value, masked: set):
    if _is_volatile(key):
        masked.add(key)
        return "<masked>"
    return _mask(value, masked)


def _compare(expected, got) -> tuple[bool, set]:
    masked: set = set()
    return _mask(expected, masked) == _mask(got, masked), masked


def masked_equal(expected, got) -> bool:
    """True when `got` matches `expected` after masking volatile fields (ids, timestamps, …)."""
    return _compare(expected, got)[0]


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


def _run_replay(repo: Path, calls: list[ToolEvent], base: str, ctx: dict,
                settings: Settings) -> list[dict]:
    spec = {"base_url_env": ctx["base_url_env"], "base_url": base, "tools": ctx["tools"],
            "calls": [{"tool": c.tool, "arguments": c.arguments} for c in calls]}
    return _run_spec(repo, spec, settings)


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


# ---- state capture (for task criteria) -------------------------------------


def _table_names(conn) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    return [row[0] for row in conn.execute(q).fetchall()]


def _pk_col(conn, table: str) -> str:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row["pk"]:
            return row["name"]
    return "rowid"


def _dump_table(conn, table: str) -> dict:
    pk = _pk_col(conn, table)
    select = "*" if pk != "rowid" else "rowid AS rowid, *"
    rows = [dict(r) for r in conn.execute(f"SELECT {select} FROM {table}").fetchall()]
    return {"pk": pk, "rows": rows}


def dump_db(db_path: Path) -> dict:
    """Every table's rows keyed by name, with the primary-key column. Missing db -> empty dict."""
    path = Path(db_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return {table: _dump_table(conn, table) for table in _table_names(conn)}
    finally:
        conn.close()


def _start_mounts(mounts: list[dict]) -> list[dict]:
    started: list[dict] = []
    for mount in mounts:
        port = _free_port()
        log_path = mount["sim_dir"] / ".sim.log"
        proc, log = _start_sim(mount["sim_dir"], port, log_path)
        base = f"http://127.0.0.1:{port}"
        _await_health(proc, base, log_path)
        _reset(base)
        started.append({"proc": proc, "log": log, "base": base,
                        "sim_dir": mount["sim_dir"], "env": mount.get("env")})
    return started


def _snapshots(started: list[dict]) -> dict:
    return {s["sim_dir"].name: dump_db(s["sim_dir"] / "state.db") for s in started}


@contextlib.contextmanager
def simulators_running(mounts: list[dict]):
    """Start each simulator (`mounts` = [{"sim_dir", "env"}]), yield {env: base_url}, always kill.

    For a local check that needs the real services up (the packaged adapter check) rather than a
    fidelity score. Ports are free ports, so the base urls are yielded for the caller to pass on."""
    started = _start_mounts(mounts)
    try:
        yield {s["env"]: s["base"] for s in started if s["env"]}
    finally:
        for s in started:
            _kill(s["proc"], s["log"])


def capture_state(mounts: list[dict], repo: Path, tools: dict, calls: list[ToolEvent],
                  settings: Settings) -> dict:
    """Start each simulator, seed it, dump state, replay `calls` (real tool functions), dump again.

    `mounts` is [{"sim_dir", "env"}]. Returns {"initial", "final", "replayed"} keyed by simulator
    name, or {"error": ...} if a simulator never became healthy or the replay failed."""
    started: list[dict] = []
    try:
        started = _start_mounts(mounts)
        base_urls = {s["env"]: s["base"] for s in started if s["env"]}
        initial = _snapshots(started)
        spec = {"base_urls": base_urls, "tools": tools,
                "calls": [{"tool": c.tool, "arguments": c.arguments} for c in calls]}
        replayed = _run_spec(repo, spec, settings)
        return {"initial": initial, "final": _snapshots(started), "replayed": replayed}
    except _SimError as exc:
        return {"error": str(exc)}
    finally:
        for s in started:
            _kill(s["proc"], s["log"])


def measure_service(sim_dir: Path, repo: Path, calls: list[ToolEvent], ctx: dict,
                    settings: Settings, scrub: Scrubber) -> dict:
    """Fidelity of the simulator in `sim_dir` against `calls`. Always kills the process."""
    threshold = settings.survey_fidelity_threshold
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
