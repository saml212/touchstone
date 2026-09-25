"""Run TouchstoneAgent as an ACP server, so Harbor's simulated user can drive it turn by turn.

A simulated-user trial launches the target agent as an ACP server (`acpx sessions ensure`) and the
user agent sends each message with `acpx prompt`. This module is that server: it runs inside the
sandbox (`python -m touchstone.harbor.acp_server`), starts the task's simulators, and on every user
turn drives the recorded tool-calling loop one turn further — the conversation, the step budget,
and the trajectory all accumulate across turns. After each turn it rewrites the trajectory
(`/logs/agent/trajectory.json`, every user turn included) and `/app/output.json`, which the verifier
grades.

The turn logic (`Session`) is plain and unit-testable; the thin ACP adapter (`TouchstoneACPAgent`)
imports the `acp` package lazily so the module imports without it.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .. import store
from ..llm import provider_from_spec
from . import atif, conversation
from .agent import _provider_spec, load_config

LOGS_DIR_DEFAULT = "/logs/agent"


class LocalEnv:
    """A Harbor-environment stand-in for a process already inside the sandbox: `exec` runs the
    command locally instead of shelling into another container. `base` env (the simulators' base
    URLs and any auth placeholders) is applied to every child so the tools reach the simulators."""

    def __init__(self, base: dict | None = None) -> None:
        self.base = dict(base or {})

    async def exec(self, command: str, cwd: str | None = None, env: dict | None = None):
        merged = {**os.environ, **self.base, **(env or {})}
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd, env=merged,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return SimpleNamespace(stdout=(out or b"").decode(), stderr=(err or b"").decode(),
                               return_code=proc.returncode)


def _sim_env(env: LocalEnv, simulators: list) -> dict:
    """Start each simulator locally and return the host->base map for constant-base-URL services;
    base_url_env services are pointed at their simulator directly in `env.base`."""
    hosts: dict[str, str] = {}
    for sim in simulators:
        name, port = sim["name"], int(sim["port"])
        os.system(f"bash /app/simulators/start.sh {name} {port}")  # noqa: S605 — fixed sim names
        base = f"http://127.0.0.1:{port}"
        if sim.get("base_url_env"):
            env.base[sim["base_url_env"]] = base
        elif sim.get("host"):
            hosts[sim["host"]] = base
    return hosts


class Session:
    """One simulated-user conversation: messages, step budget, and trajectory accumulate here."""

    def __init__(self, config, provider, env: LocalEnv, conn, episode_id: str,
                 logs_dir: Path) -> None:
        self.config = config
        self.provider = provider
        self.env = env
        self.conn = conn
        self.ep = episode_id
        self.logs_dir = logs_dir
        self.messages = [{"role": "system", "content": config.system}]
        self.remaining = config.max_steps

    async def turn(self, text: str) -> str:
        """Feed one user turn, run the loop until the assistant replies (within the remaining
        conversation-wide model-call budget), persist the artifacts, and return the reply text."""
        conversation.record_user(self.conn, self.ep, self.messages, text)
        reply, _usage, used = await conversation.run_until_reply(
            self.provider, self.config, self.messages, self.env, self.conn, self.ep,
            self.remaining)
        self.remaining = max(0, self.remaining - used)
        await self._persist()
        return reply

    async def _persist(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        traj = atif.to_atif(self.conn, self.ep)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        await conversation.write_output(self.env, conversation.final_text(self.messages))


def build_session(logs_dir: str | None = None) -> Session:
    """Assemble a Session from the sandbox env: agent config, provider, started simulators."""
    config = load_config()
    env = LocalEnv()
    hosts = _sim_env(env, config.simulators)
    if hosts:
        env.base["TOUCHSTONE_SIMULATORS"] = json.dumps(hosts)
    for name in config.auth_env:  # a tool that builds a client from a token needs a value
        os.environ.setdefault(name, "x")
    model_name = os.environ.get("TOUCHSTONE_MODEL_NAME") or config.model_default
    provider = provider_from_spec(_provider_spec(model_name))
    conn = store.connect(Path(os.environ.get("TOUCHSTONE_RUN_DB", "/tmp/touchstone-run.db")))
    ep = store.insert_episode(conn, store.Episode(name="simulated-user",
                                                  meta={"agent": "touchstone"}))
    return Session(config, provider, env, conn, ep.id,
                   Path(logs_dir or os.environ.get("TOUCHSTONE_LOGS_DIR", LOGS_DIR_DEFAULT)))


class TouchstoneACPAgent:
    """The thin ACP adapter: one Session, one text reply streamed per prompt."""

    def __init__(self) -> None:
        self.conn = None
        self.session: Session | None = None

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None,
                         **kwargs):
        from acp import PROTOCOL_VERSION
        from acp.schema import AgentCapabilities, InitializeResponse
        return InitializeResponse(protocol_version=PROTOCOL_VERSION,
                                  agent_capabilities=AgentCapabilities())

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        from acp.schema import NewSessionResponse
        # build_session starts the simulators (a bounded but blocking health poll) and builds the
        # provider; run it off the event loop so the ACP handshake stays responsive.
        self.session = await asyncio.to_thread(build_session)
        return NewSessionResponse(session_id="touchstone")

    async def prompt(self, session_id, prompt, **kwargs):
        from acp import text_block
        from acp.schema import AgentMessageChunk, PromptResponse
        text = " ".join(b.text for b in prompt if getattr(b, "type", None) == "text")
        reply = await self.session.turn(text)
        await self.conn.session_update(
            session_id=session_id,
            update=AgentMessageChunk(session_update="agent_message_chunk",
                                     content=text_block(reply or "")))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs):
        return None


def main() -> None:
    from acp import run_agent
    asyncio.run(run_agent(TouchstoneACPAgent()))


if __name__ == "__main__":
    main()
