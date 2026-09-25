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
import inspect
import json
import os
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .. import store
from ..llm import provider_from_spec
from . import atif

try:  # harbor lives in the run's own environment, not in touchstone's
    from harbor.agents.base import BaseAgent
    from harbor.agents.capabilities import AgentCapabilities
except ImportError:  # importable and unit-testable without harbor installed
    BaseAgent = object
    AgentCapabilities = None

DEFAULT_MAX_STEPS = 8


@dataclass
class AgentConfig:
    system: str
    max_steps: int
    tools: list[dict]
    call: object  # callable(name, arguments, environment) -> str | awaitable[str]


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
                       tools=tools, call=call)


def _provider_spec(model_name: str) -> str:
    """Harbor's `provider/model` -> touchstone's `provider:model` (the first slash only)."""
    return model_name.replace("/", ":", 1) if model_name else model_name


class TouchstoneAgent(BaseAgent):
    """A Harbor custom agent that runs a recorded tool-calling loop against the sandbox."""

    if AgentCapabilities is not None:
        capabilities = AgentCapabilities(atif=True)

    def __init__(self, logs_dir, model_name: str | None = None, **kwargs) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        if BaseAgent is not object:
            super().__init__(logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "touchstone"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction: str, environment, context) -> None:
        config = load_config()
        provider = provider_from_spec(_provider_spec(self.model_name))
        messages = [{"role": "system", "content": config.system},
                    {"role": "user", "content": instruction}]
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(Path(tmp) / "run.db")
            ep = store.insert_episode(conn, store.Episode(name=instruction[:120],
                                                          meta={"agent": self.name()}))
            usage = await self._loop(provider, config, messages, environment, conn, ep.id)
            traj = atif.to_atif(conn, ep.id)
            conn.close()
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        _apply_usage(context, usage)

    async def _loop(self, provider, config, messages, environment, conn, episode_id) -> dict:
        totals = {"tokens_in": 0, "tokens_out": 0}
        for _ in range(config.max_steps):
            reply = provider.chat(messages, tools=config.tools or None)
            _add_usage(totals, reply.usage)
            _record_model_span(conn, episode_id, list(messages), reply)
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
        result = config.call(name, arguments, environment) if config.call else ""
        if inspect.isawaitable(result):
            result = await result
        result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        call_id = tool_call.get("id")
        store.insert_span(conn, store.Span(
            episode_id=episode_id, kind="tool", name=name, input={"name": name},
            output={"result": result}, tool_call_id=call_id))
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})


def _record_model_span(conn, episode_id: str, history: list[dict], reply) -> None:
    output: dict = {"message": {"role": "assistant", "content": reply.content,
                                "tool_calls": reply.tool_calls}}
    if reply.usage:
        output["usage"] = reply.usage
    store.insert_span(conn, store.Span(
        episode_id=episode_id, kind="model", name="model",
        input={"messages": history, "tools": [], "params": {}}, output=output,
        tokens_in=(reply.usage or {}).get("tokens_in"),
        tokens_out=(reply.usage or {}).get("tokens_out")))


def _add_usage(totals: dict, usage: dict | None) -> None:
    if usage:
        totals["tokens_in"] += usage.get("tokens_in") or 0
        totals["tokens_out"] += usage.get("tokens_out") or 0


def _apply_usage(context, usage: dict) -> None:
    context.n_input_tokens = usage["tokens_in"] or None
    context.n_output_tokens = usage["tokens_out"] or None
