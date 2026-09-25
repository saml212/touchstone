"""Package the customer's agent as a Harbor custom agent: `touchstone/agent/`.

Writes `agent.toml` (system prompt + model default read from the recorded model spans, the mode, and
the simulator manifest), `tools.py` (the replica dispatch: the recorded tool schemas plus a `call()`
that execs `replay --one` inside the sandbox), and — when the customer has runnable code — `run.sh`
+ `entry.py`, a survey-provider-generated `def run(user_message) -> str` that drives the real agent
loop for one message, importing the customer's own modules (never copying them).

An adapter check runs `entry.py` locally against the live simulator with `TOUCHSTONE_MODEL` unset:
it must complete, record at least one tool call, and exit 0. On failure it retries once with the
error, then falls back to the replica agent and flags it for `report.md`. Nothing here names a
customer or product; the generated files naturally carry the customer's own specifics.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

import tomli_w

from .. import store
from ..config import Settings
from . import fidelity
from .criteria import tools_map
from .provider import SurveyProvider
from .simulate import crossing_services
from .writes import atomic_write

MAX_STEPS = 8
_ENTRY_TIMEOUT = 180.0

RUN_SH = """\
#!/bin/bash
set -euo pipefail
# The repo root is /app; put it on the path so entry.py can import the customer's own modules. Run
# the sibling entry.py by location so the agent dir can live anywhere in the sandbox.
export PYTHONPATH="/app:${PYTHONPATH:-}"
python "$(dirname "$0")/entry.py"
"""

TOOLS_TEMPLATE = '''\
"""Replica dispatch: the recorded tool schemas and a call() that runs one recorded tool call through
the customer's real function inside the sandbox (`replay --one`), pointed at the simulators."""

import json
import shlex

TOOLS = {tools!r}
_IMPORTS = {imports!r}
_BASE_URLS = {base_urls!r}


async def call(name, arguments, environment):
    path = _IMPORTS.get(name)
    if path is None:
        return ""
    payload = json.dumps(arguments, ensure_ascii=False)
    cmd = ("python -m touchstone.survey.replay --one "
           + shlex.quote(path) + " " + shlex.quote(payload))
    result = await environment.exec(cmd, cwd="/app", env=dict(_BASE_URLS))
    return result.stdout
'''

ENTRY_PROMPT = '''You are packaging an existing AI agent as a test-harness entrypoint. Read the
repository (Read/Grep/Glob only — write nothing) and reply with the FULL contents of one Python
file, `agent/entry.py`, and nothing else (no prose, no markdown fences).

Map of this codebase:
{map}

entry.py must:
- `import touchstone` and call `touchstone.trace()` first, so the agent's model calls are captured.
- define `def run(user_message: str) -> str` that drives the customer's REAL agent loop for exactly
  one user message, the way production does, by importing the customer's own modules (never copy
  their code into entry.py). Build the model client exactly as the code does. Take the model id from
  the environment variable TOUCHSTONE_MODEL when it is set, otherwise the code's own default. Return
  the final assistant text.
- under `if __name__ == "__main__":` read the entire message from standard input, call run(), print
  the reply to stdout, and write {{"reply": <the reply>}} as JSON to the path in the environment
  variable TOUCHSTONE_OUTPUT, defaulting to "/app/output.json" when it is unset.

The repo root is already on sys.path, so `import <the customer's module>` works from entry.py.
Do not read argv. Do not hardcode a model id. Do not swallow errors — let them exit non-zero.
Return only the contents of entry.py.'''


# ---- reading the recorded spans --------------------------------------------


def _messages(span) -> list:
    return (span.input or {}).get("messages", [])


def _role_content(span, role: str) -> str:
    for msg in _messages(span):
        if msg.get("role") == role:
            return str(msg.get("content") or "")
    return ""


def _model_spans(conn):
    for ep in store.list_episodes(conn):
        for span in store.list_spans(conn, ep.id):
            if span.kind == "model":
                yield span


def _recorded_model_and_system(conn) -> tuple[str, str]:
    """The most-common recorded model id and the first recorded system prompt (full text)."""
    models: Counter = Counter()
    system = ""
    for span in _model_spans(conn):
        if span.model:
            models[span.model] += 1
        system = system or _role_content(span, "system")
    model = models.most_common(1)[0][0] if models else ""
    return model, system


def _first_user_turn(conn) -> str:
    for span in _model_spans(conn):
        turn = _role_content(span, "user")
        if turn:
            return turn
    return ""


# ---- writing agent.toml + tools.py -----------------------------------------


def _base_urls(env_result: dict) -> dict:
    ports, envs = env_result.get("ports", {}), env_result.get("base_url_envs", {})
    return {envs[n]: f"http://127.0.0.1:{ports[n]}"
            for n in env_result.get("services", []) if envs.get(n)}


def _sim_manifest(env_result: dict) -> list:
    ports, envs = env_result.get("ports", {}), env_result.get("base_url_envs", {})
    manifest = []
    for name in env_result.get("services", []):
        sim = {"name": name, "port": ports[name]}
        if envs.get(name):
            sim["base_url_env"] = envs[name]
        manifest.append(sim)
    return manifest


def _write_tools_py(agent_dir: Path, map_data: dict, env_result: dict) -> None:
    schemas = map_data.get("schemas") or {}
    tools = [schemas[t["name"]] for t in map_data.get("tools", []) if t.get("name") in schemas]
    atomic_write(agent_dir / "tools.py", TOOLS_TEMPLATE.format(
        tools=tools, imports=tools_map(map_data), base_urls=_base_urls(env_result)))


def _write_agent_toml(agent_dir: Path, system: str, model_id: str, mode: str,
                      map_data: dict, env_result: dict) -> None:
    agent = {"system": system, "max_steps": MAX_STEPS, "mode": mode, "model_default": model_id,
             "provider": map_data.get("model_call", {}).get("sdk", "")}
    sims = _sim_manifest(env_result)
    if sims:
        agent["simulator"] = sims
    atomic_write(agent_dir / "agent.toml", tomli_w.dumps({"agent": agent}))


# ---- packaged entry.py + adapter check -------------------------------------


def _map_digest(map_data: dict) -> str:
    keep = {k: map_data.get(k) for k in ("entrypoints", "system_prompts", "model_call", "tools")}
    return json.dumps(keep, indent=2)[:6000]


def _strip_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def _generate_entry(provider: SurveyProvider, repo: Path, prompt: str, error: str | None) -> str:
    full = prompt if error is None else (
        f"{prompt}\n\nYour previous entry.py failed: {error}\nReturn a corrected entry.py.")
    return _strip_fence(provider.run(full, repo))


def _entry_cmd(repo: Path, entry_path: Path, settings: Settings) -> list[str]:
    if settings.survey_python:
        return [settings.survey_python, str(entry_path)]
    return ["uv", "run", "--project", str(repo), "python", str(entry_path)]


def _run_entry(repo: Path, entry_path: Path, env_extra: dict, message: str, settings: Settings):
    env = {**os.environ, **env_extra}
    env.pop("TOUCHSTONE_MODEL", None)  # the adapter check runs on the code's own default model
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(_entry_cmd(repo, entry_path, settings), cwd=str(repo), input=message,
                          capture_output=True, text=True, timeout=_ENTRY_TIMEOUT, env=env)


def _tail(proc) -> str:
    return "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])


def _span_has_tool(span) -> bool:
    if span.kind == "tool":
        return True
    return bool((span.output or {}).get("message", {}).get("tool_calls"))


def _has_tool_call(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    conn = store.connect(db_path)
    try:
        return any(_span_has_tool(s) for ep in store.list_episodes(conn)
                   for s in store.list_spans(conn, ep.id))
    finally:
        conn.close()


def _sim_mounts(map_data: dict, env_result: dict, out: Path) -> list[dict]:
    envs = env_result.get("base_url_envs", {})
    return [{"sim_dir": out / "simulators" / s["name"], "env": envs.get(s["name"])}
            for s in crossing_services(map_data)]


def _run_and_check(agent_dir: Path, repo: Path, base_urls: dict, message: str,
                   settings: Settings) -> tuple[bool, str]:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "touchstone.db"
        env = {**base_urls, "TOUCHSTONE_DB": str(db),
               "TOUCHSTONE_OUTPUT": str(Path(tmp) / "out.json")}
        proc = _run_entry(repo, agent_dir / "entry.py", env, message, settings)
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {_tail(proc)}"
        if not _has_tool_call(db):
            return False, "entry.py recorded no tool call"
        return True, ""


def _adapter_check(agent_dir: Path, repo: Path, map_data: dict, env_result: dict, out: Path,
                   message: str, settings: Settings) -> tuple[bool, str]:
    try:
        with fidelity.simulators_running(_sim_mounts(map_data, env_result, out)) as base_urls:
            return _run_and_check(agent_dir, repo, base_urls, message, settings)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return False, f"adapter check error: {str(exc)[:400]}"


def _drop_packaged(agent_dir: Path) -> None:
    for name in ("entry.py", "run.sh"):
        path = agent_dir / name
        if path.exists():
            path.unlink()


def _try_packaged(agent_dir: Path, repo: Path, conn, map_data: dict, env_result: dict,
                  provider: SurveyProvider, out: Path, settings: Settings) -> dict:
    message = _first_user_turn(conn)
    if not map_data.get("entrypoints") or not message:
        return {"mode": "replica", "ok": False, "flag": "no runnable entrypoint or recorded turn"}
    prompt = ENTRY_PROMPT.format(map=_map_digest(map_data))
    error: str | None = None
    for _ in range(2):
        atomic_write(agent_dir / "entry.py", _generate_entry(provider, repo, prompt, error))
        atomic_write(agent_dir / "run.sh", RUN_SH)
        ok, reason = _adapter_check(agent_dir, repo, map_data, env_result, out, message, settings)
        if ok:
            return {"mode": "packaged", "ok": True, "flag": None}
        error = reason
    _drop_packaged(agent_dir)
    return {"mode": "replica", "ok": False, "flag": f"adapter check failed: {error}"}


def _existing(agent_dir: Path) -> dict:
    doc = tomllib.loads((agent_dir / "agent.toml").read_text(encoding="utf-8"))
    agent = doc.get("agent", {})
    mode = agent.get("mode", "replica")
    return {"mode": mode, "model_default": agent.get("model_default", ""),
            "provider": agent.get("provider", ""), "adapter_ok": mode == "packaged", "flag": None}


def build_package(repo: Path, conn, map_data: dict, env_result: dict, provider: SurveyProvider,
                  out: Path, settings: Settings, force: bool = False) -> dict:
    """Write touchstone/agent/. Returns {mode, model_default, provider, adapter_ok, flag}."""
    agent_dir = out / "agent"
    if (agent_dir / "agent.toml").exists() and not force:
        return _existing(agent_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    model_id, system = _recorded_model_and_system(conn)
    _write_tools_py(agent_dir, map_data, env_result)
    result = _try_packaged(agent_dir, repo, conn, map_data, env_result, provider, out, settings)
    _write_agent_toml(agent_dir, system, model_id, result["mode"], map_data, env_result)
    return {"mode": result["mode"], "model_default": model_id,
            "provider": map_data.get("model_call", {}).get("sdk", ""),
            "adapter_ok": result["ok"], "flag": result.get("flag")}
