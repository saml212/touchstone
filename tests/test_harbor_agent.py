"""TouchstoneAgent against a nop-style fake environment, a queued provider, and a real agent/ dir.

No harbor and no network: BaseAgent falls back to object, and the model is a queued fake.
"""

import asyncio
import json
from types import SimpleNamespace

from touchstone.harbor import agent as agent_mod
from touchstone.harbor.agent import TouchstoneAgent, load_config
from touchstone.llm import Reply

TOOLS_PY = '''\
TOOLS = [{"type": "function", "function": {"name": "lookup",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


async def call(name, arguments, environment):
    result = await environment.exec("echo " + arguments["q"])
    return result.stdout
'''


class FakeEnv:
    """A nop-style Harbor environment: records exec calls, returns canned stdout."""

    def __init__(self):
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append(command)
        return SimpleNamespace(stdout=f"ran:{command}", stderr="", return_code=0)


class QueuedProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None, json=False, timeout=60):
        self.seen.append(tools)
        return self.replies.pop(0)


def _agent_dir(tmp_path, *, max_steps=8, with_tools=True):
    d = tmp_path / "agent"
    d.mkdir()
    (d / "agent.toml").write_text(
        f'[agent]\nsystem = "You are the agent under test."\nmax_steps = {max_steps}\n')
    if with_tools:
        (d / "tools.py").write_text(TOOLS_PY)
    return d


def _run(tmp_path, monkeypatch, provider, *, logs=None):
    logs = logs or (tmp_path / "logs")
    logs.mkdir(exist_ok=True)
    monkeypatch.setenv("TOUCHSTONE_AGENT_DIR", str(tmp_path / "agent"))
    monkeypatch.setattr(agent_mod, "provider_from_spec", lambda spec: provider)
    env = FakeEnv()
    ctx = SimpleNamespace()
    ta = TouchstoneAgent(logs, model_name="openai/gpt-4o-mini")
    asyncio.run(ta.run("Where is order A1?", env, ctx))
    return env, ctx, logs


def test_load_config_reads_toml_and_tools(tmp_path):
    _agent_dir(tmp_path, max_steps=3)
    cfg = load_config(tmp_path / "agent")
    assert cfg.system == "You are the agent under test." and cfg.max_steps == 3
    assert cfg.tools[0]["function"]["name"] == "lookup" and callable(cfg.call)


def test_run_loops_tools_then_finishes_and_writes_atif(tmp_path, monkeypatch):
    _agent_dir(tmp_path)
    provider = QueuedProvider([
        Reply(content="", tool_calls=[{"id": "c1", "name": "lookup",
              "arguments": '{"q": "A1"}'}], usage={"tokens_in": 5, "tokens_out": 2}),
        Reply(content="It shipped.", tool_calls=[], usage={"tokens_in": 4, "tokens_out": 3}),
    ])
    env, ctx, logs = _run(tmp_path, monkeypatch, provider)

    assert "echo A1" in env.calls  # the tool ran inside the sandbox
    assert ctx.n_input_tokens == 9 and ctx.n_output_tokens == 5  # usage accumulated

    traj = json.loads((logs / "trajectory.json").read_text())
    assert traj["schema_version"].startswith("ATIF")
    sources = [s["source"] for s in traj["steps"]]
    assert sources == ["system", "user", "agent", "agent"]
    tool_step = traj["steps"][2]
    assert tool_step["tool_calls"][0]["function_name"] == "lookup"
    assert tool_step["observation"]["results"][0]["source_call_id"] == "c1"
    assert "ran:echo A1" in tool_step["observation"]["results"][0]["content"]
    assert traj["steps"][3]["message"] == "It shipped."


def test_run_passes_tool_schemas_to_the_model(tmp_path, monkeypatch):
    _agent_dir(tmp_path)
    provider = QueuedProvider([Reply(content="done", tool_calls=[])])
    _run(tmp_path, monkeypatch, provider)
    assert provider.seen[0][0]["function"]["name"] == "lookup"  # tools forwarded to chat


def test_run_with_no_tools_module_still_writes_a_trajectory(tmp_path, monkeypatch):
    _agent_dir(tmp_path, with_tools=False)
    provider = QueuedProvider([Reply(content="hi", tool_calls=[])])
    _, _, logs = _run(tmp_path, monkeypatch, provider)
    traj = json.loads((logs / "trajectory.json").read_text())
    assert [s["source"] for s in traj["steps"]] == ["system", "user", "agent"]
