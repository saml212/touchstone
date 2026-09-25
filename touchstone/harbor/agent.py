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

import asyncio
import importlib.util
import inspect
import json
import os
import shlex
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .. import store
from ..llm import provider_from_spec
from . import atif, keys

try:  # harbor lives in the run's own environment, not in touchstone's
    from harbor.agents.base import BaseAgent
    from harbor.agents.capabilities import AgentCapabilities
except ImportError:  # importable and unit-testable without harbor installed
    BaseAgent = object
    AgentCapabilities = None

DEFAULT_MAX_STEPS = 8
# Inside the sandbox this is the host trial's agent dir (bind mount); the packaged customer app
# points its own capture here, and _run_packaged reads the db back from the host.
PACKAGED_DB = "/logs/agent/touchstone.db"

# Start one simulator in the background (nohup survives the exec), wait for health, reset it.
_SIM_START = """\
poke() {{ python - "$1" "${{2:-GET}}" <<'PY' 2>/dev/null
import sys, urllib.request as u
u.urlopen(u.Request(sys.argv[1], method=sys.argv[2]), timeout=2)
PY
}}
nohup python /app/simulators/{name}/app.py {port} >/tmp/ts-sim-{name}.log 2>&1 &
for _ in $(seq 1 100); do
  poke "http://127.0.0.1:{port}/__health" && break
  sleep 0.2
done
poke "http://127.0.0.1:{port}/__reset" POST || true
"""


@dataclass
class AgentConfig:
    system: str
    max_steps: int
    tools: list[dict]
    call: object  # callable(name, arguments, environment) -> str | awaitable[str]
    mode: str = "replica"
    model_default: str = ""
    simulators: list = field(default_factory=list)  # [{name, port, base_url_env}]


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
                       simulators=list(section.get("simulator", [])))


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
        if BaseAgent is not object:
            super().__init__(logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "touchstone"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        """Start the dataset's simulators inside the sandbox so both modes can reach them."""
        self._sim_env = await _start_simulators(environment, load_config().simulators)

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
            usage = await self._loop(provider, config, messages, environment, conn, ep.id)
            traj = atif.to_atif(conn, ep.id)
            conn.close()
        return usage, traj

    async def _run_packaged(self, instruction: str, environment) -> tuple[dict, dict]:
        """Run the customer's real entrypoint (`bash /app/agent/run.sh`) inside the sandbox with the
        model as a setting, then import the trajectory its own capture wrote to the trace db."""
        env = self._packaged_env(load_config())
        cmd = f"printf '%s' {shlex.quote(instruction)} | bash /app/agent/run.sh"
        result = await environment.exec(cmd, cwd="/app", env=env)
        return _import_trajectory(self.logs_dir, result)

    def _packaged_env(self, config: AgentConfig) -> dict:
        env = dict(self._sim_env)
        env["TOUCHSTONE_MODEL"] = self._model_id(config)
        env["TOUCHSTONE_DB"] = PACKAGED_DB
        var = keys.provider_key_var(self.model_name)
        value = os.environ.get(var) if var else None
        if var and value:
            env[var] = value
        return env

    def _model_id(self, config: AgentConfig) -> str:
        """The model id capture rewires to: the part after `provider/`, else the recorded one."""
        if self.model_name and "/" in self.model_name:
            return self.model_name.split("/", 1)[1]
        return self.model_name or config.model_default or ""

    async def _loop(self, provider, config, messages, environment, conn, episode_id) -> dict:
        totals = {"tokens_in": 0, "tokens_out": 0}
        for _ in range(config.max_steps):
            reply = await asyncio.to_thread(provider.chat, messages, config.tools or None)
            _add_usage(totals, reply.usage)
            _record_model_span(conn, episode_id, list(messages), config.tools, reply)
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": reply.tool_calls})
            if not reply.tool_calls:
                break
            for tool_call in reply.tool_calls:
                await self._run_tool(config, tool_call, environment, messages, conn, episode_id)
        return totals

    async def _run_tool(self, config, tool_call, environment, messages, conn, episode_id) -> None:
        name = tool_call.get("name") or "tool"
        try:
            arguments = json.loads(tool_call.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        result = await _dispatch(config.call, name, arguments, environment)
        result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        call_id = tool_call.get("id")
        store.insert_span(conn, store.Span(
            episode_id=episode_id, kind="tool", name=name,
            input={"name": name, "arguments": arguments},
            output={"result": result}, tool_call_id=call_id))
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})


async def _dispatch(call, name: str, arguments: dict, environment):
    """Call the task's tool dispatch off the event loop unless it is a coroutine function."""
    if call is None:
        return ""
    if inspect.iscoroutinefunction(call):
        return await call(name, arguments, environment)
    return await asyncio.to_thread(call, name, arguments, environment)


def _record_model_span(conn, episode_id: str, history: list[dict], tools, reply) -> None:
    output: dict = {"message": {"role": "assistant", "content": reply.content,
                                "tool_calls": reply.tool_calls}}
    if reply.usage:
        output["usage"] = reply.usage
    store.insert_span(conn, store.Span(
        episode_id=episode_id, kind="model", name="model",
        input={"messages": history, "tools": tools or [], "params": {}}, output=output,
        tokens_in=(reply.usage or {}).get("tokens_in"),
        tokens_out=(reply.usage or {}).get("tokens_out")))


def _add_usage(totals: dict, usage: dict | None) -> None:
    if usage:
        totals["tokens_in"] += usage.get("tokens_in") or 0
        totals["tokens_out"] += usage.get("tokens_out") or 0


def _apply_usage(context, usage: dict) -> None:
    context.n_input_tokens = usage["tokens_in"] or None
    context.n_output_tokens = usage["tokens_out"] or None


async def _start_simulators(environment, simulators: list) -> dict:
    """Start each simulator on its fixed port and return the base-url env each service reads."""
    env: dict[str, str] = {}
    for sim in simulators:
        name, port = sim["name"], int(sim["port"])
        await environment.exec(_SIM_START.format(name=name, port=port))
        if sim.get("base_url_env"):
            env[sim["base_url_env"]] = f"http://127.0.0.1:{port}"
    return env


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


def _import_trajectory(logs_dir: Path, result) -> tuple[dict, dict]:
    """Read the customer's own trace db from the trial's agent dir and convert its run to ATIF."""
    db = logs_dir / "touchstone.db"
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
