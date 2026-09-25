"""End-to-end survey with a scripted provider, a real trace DB, and a real customer tool module.

Exercises the full stage-3 pipeline (map -> fidelity -> group -> environment -> tasks) with the gate
skipped (the gate needs Docker/Harbor; it is covered in test_survey_gate)."""

import json
import sys

from tests.test_survey_simulate import CUSTOMER_TOOL, SIM_SRC
from touchstone import store
from touchstone.config import Settings
from touchstone.survey import survey as survey_mod
from touchstone.survey.provider import ScriptedSurveyProvider
from touchstone.survey.survey import run_survey

MAP = {
    "entrypoints": ["customer_tools.py:get_widget"],
    "system_prompts": [{"file": "customer_tools.py", "line": 1, "text": "n/a"}],
    "tools": [{"name": "get_widget", "import_path": "customer_tools:get_widget",
               "file": "customer_tools.py", "line": 7, "calls": ["widget"]}],
    "model_call": {"file": "customer_tools.py", "line": 1, "sdk": "openai", "model_setting": "arg"},
    "services": [{"name": "widget", "kind": "http", "base_url_env": "WIDGET_URL",
                  "base_url_default": "http://127.0.0.1:9",
                  "calls": [{"method": "GET", "path_template": "/widgets/{id}",
                             "from_tool": "get_widget"}]}],
    "schemas": {"get_widget": {"type": "object"}},
}
SIM = {"app.py": SIM_SRC, "seed.json": {"widgets": [{"id": "w1", "color": "red"}]},
       "README.md": "widget simulator"}
GROUPS = {"groups": [{"label": "Widget lookups", "slug": "widget-lookup", "episodes": ["ep1"]}]}
TEXT = {"instruction": "Tell me the colour of my widget.",
        "persona": "A shop owner checking a widget."}
# A canned packaged entrypoint: records a model span with a tool call so the adapter check passes.
ENTRY = '''\
import os, sys
from touchstone import store
msg = sys.stdin.read()
conn = store.connect(os.environ["TOUCHSTONE_DB"])
ep = store.insert_episode(conn, store.Episode(name="e"))
store.insert_span(conn, store.Span(episode_id=ep.id, kind="model", name="model", model="m",
    input={"messages": [{"role": "user", "content": msg}]},
    output={"message": {"role": "assistant", "content": "red",
                        "tool_calls": [{"id": "c1", "name": "get_widget", "arguments": "{}"}]}}))
conn.close()
print("red")
'''
# A canned agent/invoke.py: calls the customer tool by name (adapter check hits the live simulator).
INVOKE_PY = '''\
import customer_tools


def invoke(name, arguments):
    return getattr(customer_tools, name)(**arguments)
'''
PIPELINE = [json.dumps(MAP), json.dumps(SIM), INVOKE_PY, json.dumps(GROUPS), json.dumps(TEXT),
            ENTRY]


def _seed_db(repo, tool_name="get_widget"):
    db = repo / ".touchstone" / "touchstone.db"
    conn = store.connect(db)
    ep = store.insert_episode(conn, store.Episode(id="ep1", name="ep1", outcome_label="resolved",
                                                  outcome_score=1.0))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="gpt",
        input={"messages": [{"role": "user", "content": "widget w1?"}]},
        output={"message": {"content": "", "tool_calls": [
            {"id": "c1", "name": tool_name, "arguments": '{"widget_id": "w1"}'}]}}))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="gpt",
        input={"messages": [{"role": "tool", "tool_call_id": "c1",
                             "content": '{"id": "w1", "color": "red"}'}]},
        output={"message": {"content": "red", "tool_calls": []}}))
    conn.close()


def _repo(tmp_path, tool_name="get_widget"):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "customer_tools.py").write_text(CUSTOMER_TOOL, encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="c"\nversion="0.1"\ndependencies=["httpx>=0.27"]\n', encoding="utf-8")
    _seed_db(repo, tool_name)
    return repo


def _settings():
    return Settings(survey_python=sys.executable, survey_fidelity_threshold=0.8)


def _use(monkeypatch, provider):
    monkeypatch.setattr(survey_mod, "survey_provider", lambda *a, **k: provider)


def test_survey_end_to_end(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    provider = ScriptedSurveyProvider(PIPELINE)
    _use(monkeypatch, provider)
    line = run_survey(repo, settings=_settings(), skip_gate=True)
    assert line == ("Mapped 1 tool, 1 service. Simulator widget: fidelity 1.00 (1/1 calls) "
                    "Built 1 task from 1 conversation (gate skipped).")
    out = repo / "touchstone"
    assert (out / "map.json").exists()
    assert (out / "groups.json").exists()
    assert (out / "dataset.toml").exists()
    assert (out / "environment" / "Dockerfile").exists()
    task = out / "tasks" / "widget-lookup-1"
    assert (task / "task.toml").exists()
    assert (task / "instruction.md").exists()
    assert (task / "tests" / "correctness" / "trajectory.py").exists()
    report = (out / "report.md").read_text()
    assert "widget-lookup-1" in report
    assert "## Agent under test" in report
    assert (out / "agent" / "agent.toml").exists()
    assert (out / "agent" / "tools.py").exists()
    assert (out / "agent" / "entry.py").exists()  # adapter passed -> packaged


def test_survey_idempotent_then_force(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    provider = ScriptedSurveyProvider(PIPELINE * 2)
    _use(monkeypatch, provider)
    run_survey(repo, settings=_settings(), skip_gate=True)
    assert len(provider.calls) == 6  # map, sim, invoke.py, group, task text, entry.py
    run_survey(repo, settings=_settings(), skip_gate=True)
    assert len(provider.calls) == 6  # everything reused (agent read back from agent.toml)
    run_survey(repo, force=True, settings=_settings(), skip_gate=True)
    assert len(provider.calls) == 12  # force reruns every step


def test_survey_no_network_tools(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    no_net = dict(MAP)
    no_net["tools"] = [{"name": "summarize", "import_path": "customer_tools:x",
                        "file": "customer_tools.py", "line": 1, "calls": []}]
    no_net["services"] = []
    provider = ScriptedSurveyProvider([json.dumps(no_net)])
    _use(monkeypatch, provider)
    line = run_survey(repo, settings=_settings(), skip_gate=True)  # no trace DB, no services
    assert line == "Mapped 1 tool, 0 services."
    report = (repo / "touchstone" / "report.md").read_text()
    assert "No network-crossing services" in report
    assert len(provider.calls) == 1  # only the map step consulted the provider


def test_survey_flags_unmapped_tool(tmp_path, monkeypatch):
    # recordings reference 'ghost', which the map does not include -> episode skipped, tool flagged
    repo = _repo(tmp_path, tool_name="ghost")
    provider = ScriptedSurveyProvider([json.dumps(MAP), json.dumps(SIM), json.dumps(GROUPS)])
    _use(monkeypatch, provider)
    line = run_survey(repo, settings=_settings(), skip_gate=True)
    report = (repo / "touchstone" / "report.md").read_text()
    assert "ghost" in report
    assert "Built 0 tasks" in line
