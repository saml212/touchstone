"""Package the customer's agent as a Harbor custom agent: `touchstone/agent/`.

Writes `agent.toml` (system prompt + model default from the recorded model spans, the mode, and the
simulator manifest), `tools.py` (the replica dispatch: the recorded tool schemas plus a `call()`
that execs `replay --one` inside the sandbox), and — when the customer has runnable code — `run.sh`
+ `entry.py`, a survey-provider-generated `def run(user_message) -> str` that drives the real agent
loop for one message, importing the customer's own modules (never copying them).

An adapter check runs `entry.py` locally against the live simulator with `TOUCHSTONE_MODEL` unset:
it must complete, record at least one tool call, and exit 0. On failure it retries once with the
error, then falls back to the replica agent and flags it for `report.md`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from ..config import Settings
from .criteria import tools_map
from .envs import auth_env_names
from .package_entry import try_packaged
from .package_spans import recorded_model_and_system
from .provider import SurveyProvider
from .writes import atomic_write

MAX_STEPS = 8

TOOLS_TEMPLATE = '''\
"""Replica dispatch: the recorded tool schemas and a call() that runs one recorded tool call through
the customer's real function inside the sandbox (`replay --one`), pointed at the simulators."""

import json
import shlex

TOOLS = {tools!r}
_IMPORTS = {imports!r}
_BASE_URLS = {base_urls!r}
_SIMULATORS = {simulators!r}
_INVOKE = {invoke!r}  # container path to agent/invoke.py (call tools by name), or None


def _env():
    env = dict(_BASE_URLS)
    if _SIMULATORS:  # rewrite a constant hardcoded host to its simulator via the net shim
        env["TOUCHSTONE_SIMULATORS"] = json.dumps(_SIMULATORS)
    return env


def _cmd(name, path, payload):
    if _INVOKE:  # tools that need a constructed client / central dispatch go through invoke.py
        return ("python -m touchstone.survey.replay --invoke-one " + shlex.quote(_INVOKE)
                + " " + shlex.quote(name) + " " + shlex.quote(payload))
    return ("python -m touchstone.survey.replay --one "
            + shlex.quote(path) + " " + shlex.quote(payload))


async def call(name, arguments, environment):
    path = _IMPORTS.get(name)
    if path is None and not _INVOKE:
        return ""
    payload = json.dumps(arguments, ensure_ascii=False)
    result = await environment.exec(_cmd(name, path or "", payload), cwd="/app", env=_env())
    return result.stdout
'''

# ---- writing agent.toml + tools.py -----------------------------------------


def _service_base(name: str, env_result: dict) -> str:
    """The base URL a tool's env var is set to in the sandbox: a loopback port for an HTTP sim, or
    the container path of state.db (sqlite:///… when the code reads a URL) for a db sim."""
    from . import db_service

    if env_result.get("kinds", {}).get(name) == "db":
        return db_service.value_for(db_service.container_db_path(name),
                                    url=env_result.get("db_urls", {}).get(name, False))
    return f"http://127.0.0.1:{env_result['ports'][name]}"


def _base_urls(env_result: dict) -> dict:
    envs = env_result.get("base_url_envs", {})
    return {envs[n]: _service_base(n, env_result)
            for n in env_result.get("services", []) if envs.get(n)}


def _simulators_map(env_result: dict) -> dict:
    """host -> sim base URL per constant-base-URL service (net shim rewrites a hardcoded host)."""
    ports, envs, hosts = (env_result.get("ports", {}), env_result.get("base_url_envs", {}),
                          env_result.get("hosts", {}))
    return {hosts[n]: f"http://127.0.0.1:{ports[n]}"
            for n in env_result.get("services", []) if hosts.get(n) and not envs.get(n)}


def _sim_entry(name: str, env_result: dict) -> dict:
    envs, hosts, kinds = (env_result.get("base_url_envs", {}), env_result.get("hosts", {}),
                          env_result.get("kinds", {}))
    if kinds.get(name) == "db":  # no port and no host: setup() re-materializes state.db
        sim = {"name": name, "kind": "db", "base_url_env": envs[name],
               "db_url": env_result.get("db_urls", {}).get(name, False)}
        return sim
    sim = {"name": name, "port": env_result["ports"][name]}
    if envs.get(name):
        sim["base_url_env"] = envs[name]
    elif hosts.get(name):
        sim["host"] = hosts[name]
    return sim


def _sim_manifest(env_result: dict) -> list:
    return [_sim_entry(name, env_result) for name in env_result.get("services", [])]


def _write_tools_py(agent_dir: Path, map_data: dict, env_result: dict) -> None:
    from .environment import IMAGE_INVOKE

    schemas = map_data.get("schemas") or {}
    tools = [schemas[t["name"]] for t in map_data.get("tools", []) if t.get("name") in schemas]
    invoke = IMAGE_INVOKE if env_result.get("invoke") else None
    atomic_write(agent_dir / "tools.py", TOOLS_TEMPLATE.format(
        tools=tools, imports=tools_map(map_data), base_urls=_base_urls(env_result),
        simulators=_simulators_map(env_result), invoke=invoke))


def _write_agent_toml(agent_dir: Path, system: str, model_id: str, mode: str,
                      map_data: dict, env_result: dict, auth_env=()) -> None:
    agent = {"system": system, "max_steps": MAX_STEPS, "mode": mode, "model_default": model_id,
             "provider": map_data.get("model_call", {}).get("sdk", "")}
    sims = _sim_manifest(env_result)
    if sims:
        agent["simulator"] = sims
    if auth_env:  # the packaged sandbox run placeholder-fills these for the tool's client
        agent["auth_env"] = list(auth_env)
    atomic_write(agent_dir / "agent.toml", tomli_w.dumps({"agent": agent}))


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
    model_id, system = recorded_model_and_system(conn)
    _write_tools_py(agent_dir, map_data, env_result)
    result = try_packaged(agent_dir, repo, conn, map_data, env_result, provider, out, settings)
    _write_agent_toml(agent_dir, system, model_id, result["mode"], map_data, env_result,
                      auth_env=sorted(auth_env_names(repo)))
    return {"mode": result["mode"], "model_default": model_id,
            "provider": map_data.get("model_call", {}).get("sdk", ""),
            "adapter_ok": result["ok"], "flag": result.get("flag")}
