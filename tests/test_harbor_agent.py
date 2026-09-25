"""TouchstoneAgent against a nop-style fake environment, a queued provider, and a real agent/ dir.

No harbor and no network: BaseAgent falls back to object, and the model is a queued fake.
"""

import asyncio
import json
from types import SimpleNamespace

from touchstone import store
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


def _plant_customer_db(db_path, *, with_spans=True):
    """Write a trace db as the customer's own capture would: one episode, one model span whose
    assistant message carries a tool call (ATIF surfaces it without a @touchstone.tool span)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = store.connect(db_path)
    ep = store.insert_episode(conn, store.Episode(name="support-B1", meta={"agent": "app"}))
    if with_spans:
        store.insert_span(conn, store.Span(
            episode_id=ep.id, kind="model", name="model", model="gpt-4o-mini",
            input={"messages": [{"role": "user", "content": "Where is B1?"}], "tools": [],
                   "params": {}},
            output={"message": {"role": "assistant", "content": "It shipped.",
                                "tool_calls": [{"id": "c1", "name": "order_status",
                                                "arguments": '{"order_id": "B1"}'}]}},
            tokens_in=11, tokens_out=4))
    conn.close()


def _packaged_agent(tmp_path, monkeypatch, *, simulators="", model="openai/gpt-4o-mini"):
    d = tmp_path / "agent"
    d.mkdir(exist_ok=True)
    (d / "agent.toml").write_text(
        '[agent]\nsystem = "s"\nmax_steps = 5\nmode = "packaged"\n'
        f'model_default = "gpt-4o-mini"\n{simulators}')
    monkeypatch.setenv("TOUCHSTONE_AGENT_DIR", str(d))
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return TouchstoneAgent(logs, model_name=model, mode="packaged"), logs


def test_packaged_runs_run_sh_and_imports_the_customer_trajectory(tmp_path, monkeypatch):
    ta, logs = _packaged_agent(tmp_path, monkeypatch)
    _plant_customer_db(logs / "touchstone.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class PlantingEnv(FakeEnv):
        async def exec(self, command, **kwargs):
            self.calls.append(command)
            self.env = kwargs.get("env")
            return SimpleNamespace(stdout="ran", stderr="", return_code=0)

    env = PlantingEnv()
    ctx = SimpleNamespace()
    asyncio.run(ta.run("Where is my order B1?", env, ctx))

    assert "bash /app/agent/run.sh" in env.calls[0]  # the customer's entrypoint ran
    assert env.env["TOUCHSTONE_MODEL"] == "gpt-4o-mini"  # model after provider/
    assert env.env["TOUCHSTONE_DB"] == "/logs/agent/touchstone.db"
    assert env.env["OPENAI_API_KEY"] == "sk-test"  # provider key forwarded into the sandbox
    traj = json.loads((logs / "trajectory.json").read_text())
    tool_steps = [s for s in traj["steps"] if s.get("tool_calls")]
    assert tool_steps and tool_steps[0]["tool_calls"][0]["function_name"] == "order_status"
    assert ctx.n_input_tokens == 11 and ctx.n_output_tokens == 4


def test_packaged_missing_episode_raises_with_output_tail(tmp_path, monkeypatch):
    import pytest

    ta, logs = _packaged_agent(tmp_path, monkeypatch)
    _plant_customer_db(logs / "touchstone.db", with_spans=False)  # episode, but no spans

    class ErrEnv(FakeEnv):
        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout="", stderr="Traceback: boom", return_code=1)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(ta.run("hi", ErrEnv(), SimpleNamespace()))


def test_setup_starts_simulators_and_records_base_urls(tmp_path, monkeypatch):
    ta, _ = _packaged_agent(tmp_path, monkeypatch, simulators=(
        '\n[[agent.simulator]]\nname = "orders_service"\nport = 8000\n'
        'base_url_env = "ORDERS_URL"\n'))
    env = FakeEnv()
    asyncio.run(ta.setup(env))
    assert any("simulators/orders_service/app.py 8000" in c for c in env.calls)
    assert ta._sim_env == {"ORDERS_URL": "http://127.0.0.1:8000"}


def test_run_with_no_tools_module_still_writes_a_trajectory(tmp_path, monkeypatch):
    _agent_dir(tmp_path, with_tools=False)
    provider = QueuedProvider([Reply(content="hi", tool_calls=[])])
    _, _, logs = _run(tmp_path, monkeypatch, provider)
    traj = json.loads((logs / "trajectory.json").read_text())
    assert [s["source"] for s in traj["steps"]] == ["system", "user", "agent"]
