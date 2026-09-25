"""The ACP server's Session: multi-turn accumulation, budget, and the graded artifacts.

Exercises Session.turn directly (no acpx, no `acp` package) with a scripted provider and a fake
in-sandbox environment, the way Harbor's simulated user would drive it one turn at a time.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from touchstone import store
from touchstone.harbor.acp_server import LocalEnv, Session
from touchstone.harbor.agent import load_config
from touchstone.llm import Reply

TOOLS_PY = '''\
TOOLS = [{"type": "function", "function": {"name": "book",
          "parameters": {"type": "object", "properties": {"who": {"type": "string"}}}}}]


async def call(name, arguments, environment):
    result = await environment.exec("echo " + arguments.get("who", ""))
    return result.stdout
'''


class RecordingEnv:
    """A fake in-sandbox env: records exec calls and returns canned stdout."""

    def __init__(self):
        self.calls = []

    async def exec(self, command, cwd=None, env=None):
        self.calls.append(command)
        return SimpleNamespace(stdout=f"ran:{command}", stderr="", return_code=0)


class Queued:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None, json=False, timeout=60):
        return self.replies.pop(0)


def _config(tmp_path, max_steps=8):
    d = tmp_path / "agent"
    d.mkdir()
    (d / "agent.toml").write_text(
        f'[agent]\nsystem = "you book flights."\nmax_steps = {max_steps}\n')
    (d / "tools.py").write_text(TOOLS_PY)
    return load_config(d)


def _session(tmp_path, provider, env, max_steps=8):
    conn = store.connect(tmp_path / "run.db")
    ep = store.insert_episode(conn, store.Episode(name="sim", meta={"agent": "touchstone"}))
    return Session(_config(tmp_path, max_steps), provider, env, conn, ep.id, tmp_path / "logs")


def test_session_accumulates_turns_and_writes_artifacts(tmp_path):
    provider = Queued([
        Reply(content="Sure — who is flying?", tool_calls=[], usage={"tokens_in": 3}),
        Reply(content="", tool_calls=[{"id": "c1", "name": "book",
              "arguments": '{"who": "Ben Cole"}'}], usage={"tokens_in": 4}),
        Reply(content="Booked for Ben Cole.", tool_calls=[], usage={"tokens_in": 2}),
    ])
    env = RecordingEnv()
    session = _session(tmp_path, provider, env)

    first = asyncio.run(session.turn("I want to book a flight"))
    assert "who is flying" in first
    second = asyncio.run(session.turn("Ben Cole, aisle seat"))
    assert "Booked" in second

    # every user turn shows up in the trajectory, interleaved with the agent steps
    traj = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    sources = [s["source"] for s in traj["steps"]]
    assert sources == ["system", "user", "agent", "user", "agent", "agent"]
    assert "echo Ben Cole" in env.calls  # the tool ran in the sandbox on the second turn
    # output.json was (re)written with the final answer for the verifier
    writes = [c for c in env.calls if "/app/output.json" in c]
    assert writes and "Booked for Ben Cole." in writes[-1]


def test_budget_counts_model_calls_across_the_whole_conversation(tmp_path):
    # a two-step first turn (tool then reply) leaves 1 model call for the rest of the conversation
    provider = Queued([
        Reply(content="", tool_calls=[{"id": "c1", "name": "book", "arguments": "{}"}]),
        Reply(content="done turn one", tool_calls=[]),
        Reply(content="turn two", tool_calls=[]),
    ])
    session = _session(tmp_path, provider, RecordingEnv(), max_steps=3)
    asyncio.run(session.turn("start"))
    assert session.remaining == 1  # 3 budget - 2 model calls used on turn one
    asyncio.run(session.turn("more"))
    assert session.remaining == 0


def test_local_env_runs_commands_and_merges_base(tmp_path):
    env = LocalEnv({"FLIGHT_API_BASE_URL": "http://127.0.0.1:8000"})
    result = asyncio.run(env.exec("echo $FLIGHT_API_BASE_URL"))
    assert result.return_code == 0
    assert "127.0.0.1:8000" in result.stdout
