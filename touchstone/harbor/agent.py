"""TouchstoneAgent: the agent under test, packaged as a Harbor custom agent.

Run it with `harbor run --agent touchstone.harbor.agent:TouchstoneAgent --model <provider>/<model>`.
It runs an OpenAI-compatible tool-calling loop: the system prompt and step budget come from
`agent/agent.toml`, and the tool schemas (`TOOLS`) and dispatch (`call(name, arguments, env)`) come
from `agent/tools.py` in the dataset. Each model call and tool result is recorded, and the run
is written to `<logs_dir>/trajectory.json` in ATIF. This is the replica agent and the base that the
survey's packaged agent extends. Product-agnostic: it knows nothing about any one customer.

The dataset's `agent/` directory is found via TOUCHSTONE_AGENT_DIR (default: ./agent).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .. import store
from ..llm import provider_from_spec
from . import atif, conversation, keys

try:  # harbor lives in the run's own environment, not in touchstone's
    from harbor.agents.base import BaseAgent
    from harbor.agents.capabilities import AgentCapabilities
except ImportError:  # importable and unit-testable without harbor installed
    BaseAgent = object
    AgentCapabilities = None

DEFAULT_MAX_STEPS = 8
# Where the packaged customer app points its own capture inside the sandbox; _run_packaged downloads
# it back to the host afterwards. /logs/agent is the trial's agent dir.
PACKAGED_DB = "/logs/agent/touchstone.db"
PACKAGED_OUTPUT = "/app/output.json"
# The agent dir is uploaded here — NOT /app/agent, which would shadow a customer module named
# `agent` once /app is on the path. run.sh runs its sibling entry.py by location.
AGENT_SANDBOX = "/app/.touchstone_agent"


@dataclass
class AgentConfig:
    system: str
    max_steps: int
    tools: list[dict]
    call: object  # callable(name, arguments, environment) -> str | awaitable[str]
    mode: str = "replica"
    model_default: str = ""
    simulators: list = field(default_factory=list)  # [{name, port, base_url_env}]
    auth_env: list = field(default_factory=list)  # auth env vars to placeholder-fill for the client


def _agent_dir() -> Path:
    return Path(os.environ.get("TOUCHSTONE_AGENT_DIR", "agent"))


def _load_tools(agent_dir: Path):
    """Import `agent/tools.py`; return (TOOLS list, call callable). Missing file -> no tools."""
    path = agent_dir / "tools.py"
    if not path.is_file():
        return [], lambda name, arguments, environment: ""
    spec = importlib.util.spec_from_file_location("touchstone_task_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "TOOLS", [])), getattr(module, "call", None)


def load_config(agent_dir: Path | None = None) -> AgentConfig:
    agent_dir = agent_dir or _agent_dir()
    doc = tomllib.loads((agent_dir / "agent.toml").read_text(encoding="utf-8"))
    section = doc.get("agent", {})
    tools, call = _load_tools(agent_dir)
    return AgentConfig(system=section.get("system", ""),
                       max_steps=int(section.get("max_steps", DEFAULT_MAX_STEPS)),
                       tools=tools, call=call,
                       mode=section.get("mode", "replica"),
                       model_default=section.get("model_default", ""),
                       simulators=list(section.get("simulator", [])),
                       auth_env=list(section.get("auth_env", [])))


def _provider_spec(model_name: str) -> str:
    """Harbor's `provider/model` -> touchstone's `provider:model` (the first slash only)."""
    return model_name.replace("/", ":", 1) if model_name else model_name


class TouchstoneAgent(BaseAgent):
    """A Harbor custom agent that runs a recorded tool-calling loop against the sandbox."""

    if AgentCapabilities is not None:
        capabilities = AgentCapabilities(atif=True)

    def __init__(self, logs_dir, model_name: str | None = None, mode: str = "replica",
                 **kwargs) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        self.mode = mode  # "replica" (this loop) | "packaged" (customer's agent); via --ak mode=
        self._sim_env: dict[str, str] = {}  # base-url envs of the simulators setup() started
        self._sim_hosts: dict[str, str] = {}  # host -> sim base for constant-base-URL services
        if BaseAgent is not object:
            super().__init__(logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "touchstone"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        """Start the dataset's simulators inside the sandbox so both modes can reach them."""
        self._sim_env, self._sim_hosts = await _start_simulators(
            environment, load_config().simulators)

    async def run(self, instruction: str, environment, context) -> None:
        if self.mode == "packaged":
            usage, traj = await self._run_packaged(instruction, environment)
        else:
            usage, traj = await self._run_replica(instruction, environment)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        _apply_usage(context, usage)

    async def _run_replica(self, instruction: str, environment) -> tuple[dict, dict]:
        """Drive the recorded tool-calling loop and return (usage, ATIF trajectory)."""
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(Path(tmp) / "run.db")
            ep = store.insert_episode(conn, store.Episode(name=instruction[:120],
                                                          meta={"agent": self.name()}))
            config = load_config()
            provider = provider_from_spec(_provider_spec(self.model_name))
            messages = [{"role": "system", "content": config.system},
                        {"role": "user", "content": instruction}]
            _, usage, _ = await conversation.run_until_reply(
                provider, config, messages, environment, conn, ep.id, config.max_steps)
            await conversation.write_output(environment, conversation.final_text(messages))
            traj = atif.to_atif(conn, ep.id)
            conn.close()
        return usage, traj

    async def _run_packaged(self, instruction: str, environment) -> tuple[dict, dict]:
        """Upload the agent dir into the sandbox, run the customer's real entrypoint (run.sh) with
        the model as a setting, then download the trace db its own capture wrote and convert it."""
        await environment.upload_dir(_agent_dir(), AGENT_SANDBOX)
        env = self._packaged_env(load_config())
        cmd = f"printf '%s' {shlex.quote(instruction)} | bash {AGENT_SANDBOX}/run.sh"
        result = await environment.exec(cmd, cwd="/app", env=env)
        with tempfile.TemporaryDirectory() as tmp:
            local_db = Path(tmp) / "touchstone.db"
            await _download(environment, PACKAGED_DB, local_db)
            return _import_trajectory(local_db, result)

    def _packaged_env(self, config: AgentConfig) -> dict:
        from ..survey.envs import placeholder_auth

        env = dict(self._sim_env)
        if self._sim_hosts:  # constant-base-URL services: sitecustomize installs the net shim
            env["TOUCHSTONE_SIMULATORS"] = json.dumps(self._sim_hosts)
        # A tool that builds its client from a token needs a value: fill a placeholder for each auth
        # env the survey recorded, the same treatment replay and the adapter check give it.
        for name, placeholder in placeholder_auth(config.auth_env).items():
            env.setdefault(name, placeholder)
        env["TOUCHSTONE_MODEL"] = self._model_id(config)
        env["TOUCHSTONE_DB"] = PACKAGED_DB
        env["TOUCHSTONE_OUTPUT"] = PACKAGED_OUTPUT
        var = keys.provider_key_var(self.model_name)
        value = os.environ.get(var) if var else None
        if var and value:  # the real model-provider key overrides any placeholder set above
            env[var] = value
        return env

    def _model_id(self, config: AgentConfig) -> str:
        """The model id capture rewires to: the part after `provider/`, else the recorded one."""
        if self.model_name and "/" in self.model_name:
            return self.model_name.split("/", 1)[1]
        return self.model_name or config.model_default or ""

def _apply_usage(context, usage: dict) -> None:
    context.n_input_tokens = usage["tokens_in"] or None
    context.n_output_tokens = usage["tokens_out"] or None


async def _start_simulators(environment, simulators: list) -> tuple[dict, dict]:
    """Start each simulator on its fixed port. Return (base-url envs, host->base map): a service
    with a base_url_env is repointed by env var; a constant-host service by the net shim."""
    env: dict[str, str] = {}
    hosts: dict[str, str] = {}
    for sim in simulators:
        name, port = sim["name"], int(sim["port"])
        await environment.exec(f"bash /app/simulators/start.sh {name} {port}")
        base = f"http://127.0.0.1:{port}"
        if sim.get("base_url_env"):
            env[sim["base_url_env"]] = base
        elif sim.get("host"):
            hosts[sim["host"]] = base
    return env, hosts


def _tail(result) -> str:
    text = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return "\n".join(text.splitlines()[-30:])


def _last_episode(conn):
    for ep in reversed(store.list_episodes(conn)):
        if store.list_spans(conn, ep.id):
            return ep
    return None


def _usage_from(traj: dict) -> dict:
    metrics = traj.get("final_metrics") or {}
    return {"tokens_in": metrics.get("total_prompt_tokens") or 0,
            "tokens_out": metrics.get("total_completion_tokens") or 0}


async def _download(environment, remote: str, local: Path) -> None:
    """Best-effort download; a missing db is handled by _import_trajectory as a run failure."""
    try:
        await environment.download_file(remote, str(local))
    except Exception:  # noqa: BLE001 — treat any download failure as "no db", with the output tail
        pass


def _import_trajectory(db: Path, result) -> tuple[dict, dict]:
    """Convert the customer's own captured run (downloaded from the sandbox) to ATIF."""
    if not db.exists():
        raise RuntimeError(f"packaged run wrote no trace db\n{_tail(result)}")
    conn = store.connect(db)
    try:
        ep = _last_episode(conn)
        if ep is None:
            raise RuntimeError(f"packaged run produced no episode\n{_tail(result)}")
        traj = atif.to_atif(conn, ep.id)
    finally:
        conn.close()
    return _usage_from(traj), traj
