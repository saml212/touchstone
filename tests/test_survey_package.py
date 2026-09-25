"""Package writer: agent.toml + tools.py content, and the adapter check picking packaged vs replica.

The adapter check runs a scripted entry.py through the real subprocess path (survey_python = this
interpreter) with no crossing services, so it stays hermetic — no uv, no model, no simulator.
"""

import sys
import tomllib

from touchstone import store
from touchstone.config import Settings
from touchstone.survey import package
from touchstone.survey.provider import ScriptedSurveyProvider

MAP = {
    "entrypoints": ["agent.py:main"],
    "model_call": {"file": "agent.py", "line": 1, "sdk": "openai", "model_setting": "default"},
    "system_prompts": [{"file": "agent.py", "line": 1, "text": "You are support."}],
    "tools": [{"name": "order_status", "import_path": "agent:order_status", "file": "a", "line": 1,
               "calls": ["orders_service"]}],
    "schemas": {"order_status": {"type": "function", "function": {"name": "order_status"}}},
    "services": [{"name": "orders_service", "kind": "http", "base_url_env": "ORDERS_URL",
                  "base_url_default": None, "calls": []}],
}
ENV_RESULT = {"services": ["orders_service"], "ports": {"orders_service": 8000},
              "base_url_envs": {"orders_service": "ORDERS_URL"}, "image_tag": "img:1"}

# An entry.py that records a model span with a tool call (adapter passes).
PASS_ENTRY = '''\
import os, sys
from touchstone import store
msg = sys.stdin.read()
conn = store.connect(os.environ["TOUCHSTONE_DB"])
ep = store.insert_episode(conn, store.Episode(name="e"))
store.insert_span(conn, store.Span(episode_id=ep.id, kind="model", name="model", model="m",
    input={"messages": [{"role": "user", "content": msg}]},
    output={"message": {"role": "assistant", "content": "ok",
                        "tool_calls": [{"id": "c1", "name": "order_status", "arguments": "{}"}]}}))
conn.close()
print("done")
'''

# Same, but no tool call (adapter fails -> replica).
FAIL_ENTRY = PASS_ENTRY.replace(
    '"tool_calls": [{"id": "c1", "name": "order_status", "arguments": "{}"}]', '"tool_calls": []')


def _conn(tmp_path):
    conn = store.connect(tmp_path / ".touchstone" / "touchstone.db")
    ep = store.insert_episode(conn, store.Episode(name="support-1", outcome_label="resolved"))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="model", model="gpt-4o-mini",
        input={"messages": [{"role": "system", "content": "You are a support agent."},
                            {"role": "user", "content": "Where is my order B1?"}]},
        output={"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "order_status", "arguments": "{}"}]}}))
    return conn


def _settings():
    s = Settings()
    s.survey_python = sys.executable  # run entry.py with this interpreter, no uv
    return s


def test_tools_py_and_agent_toml_content(tmp_path):
    out = tmp_path / "touchstone"
    agent_dir = out / "agent"
    agent_dir.mkdir(parents=True)
    package._write_tools_py(agent_dir, MAP, ENV_RESULT)
    package._write_agent_toml(agent_dir, "You are support.", "gpt-4o-mini", "replica",
                              MAP, ENV_RESULT)

    tools = (agent_dir / "tools.py").read_text()
    assert "'order_status': 'agent:order_status'" in tools  # import map
    assert "'ORDERS_URL': 'http://127.0.0.1:8000'" in tools  # baked base url
    assert "replay --one" in tools and "async def call" in tools

    doc = tomllib.loads((agent_dir / "agent.toml").read_text())
    assert doc["agent"]["provider"] == "openai"
    assert doc["agent"]["model_default"] == "gpt-4o-mini"
    assert doc["agent"]["simulator"] == [
        {"name": "orders_service", "port": 8000, "base_url_env": "ORDERS_URL"}]


def _no_service_map():
    m = dict(MAP)
    m["services"] = []
    return m


def test_build_package_packaged_when_adapter_passes(tmp_path):
    conn = _conn(tmp_path)
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([PASS_ENTRY])
    result = package.build_package(tmp_path, conn, _no_service_map(), {"services": [], "ports": {},
                                   "base_url_envs": {}}, provider, out, _settings())
    conn.close()
    assert result["mode"] == "packaged" and result["adapter_ok"] is True
    assert (out / "agent" / "entry.py").exists() and (out / "agent" / "run.sh").exists()
    assert (out / "agent" / "tools.py").exists()  # replica dispatch always written
    assert result["model_default"] == "gpt-4o-mini"


def test_build_package_replica_when_adapter_fails_twice(tmp_path):
    conn = _conn(tmp_path)
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([FAIL_ENTRY])
    result = package.build_package(tmp_path, conn, _no_service_map(), {"services": [], "ports": {},
                                   "base_url_envs": {}}, provider, out, _settings())
    conn.close()
    assert result["mode"] == "replica" and result["adapter_ok"] is False
    assert "adapter check failed" in result["flag"]
    assert not (out / "agent" / "entry.py").exists()  # dropped
    assert len(provider.calls) == 2  # generated, checked, retried, checked
    assert (out / "agent" / "tools.py").exists()


def test_build_package_replica_when_no_entrypoint(tmp_path):
    conn = _conn(tmp_path)
    out = tmp_path / "touchstone"
    no_entry = dict(_no_service_map())
    no_entry["entrypoints"] = []
    result = package.build_package(tmp_path, conn, no_entry, {"services": [], "ports": {},
                                   "base_url_envs": {}}, ScriptedSurveyProvider([PASS_ENTRY]),
                                   out, _settings())
    conn.close()
    assert result["mode"] == "replica" and "no runnable entrypoint" in result["flag"]


def test_build_package_idempotent(tmp_path):
    conn = _conn(tmp_path)
    out = tmp_path / "touchstone"
    empty_env = {"services": [], "ports": {}, "base_url_envs": {}}
    package.build_package(tmp_path, conn, _no_service_map(), empty_env,
                          ScriptedSurveyProvider([PASS_ENTRY]), out, _settings())
    again = package.build_package(tmp_path, conn, _no_service_map(), empty_env,
                                  ScriptedSurveyProvider([]), out, _settings())
    conn.close()
    assert again["mode"] == "packaged"  # read back from agent.toml, no provider call
