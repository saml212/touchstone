"""Write Harbor tasks end to end: real customer tools, a real mutating simulator, state-diff
criteria, trajectory criteria, canary, scrubbing, idempotency, and the fidelity skip."""

import json
import sys

from touchstone import store
from touchstone.config import Settings
from touchstone.survey.provider import ScriptedSurveyProvider
from touchstone.survey.recordings import tool_events
from touchstone.survey.scrub import Scrubber
from touchstone.survey.tasks import write_tasks

TOOL_SRC = '''\
import os
import httpx

_c = httpx.Client(base_url=os.environ.get("WIDGET_URL", "http://127.0.0.1:9"), timeout=5)


def get_widget(widget_id: str) -> dict:
    return _c.get(f"/widgets/{widget_id}").json()


def paint(widget_id: str, color: str) -> dict:
    return _c.post(f"/widgets/{widget_id}/paint", json={"color": color}).json()


def add_note(widget_id: str, text: str) -> dict:
    return _c.post("/notes", json={"widget_id": widget_id, "text": text}).json()
'''

SIM_SRC = '''\
import json, sqlite3, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
HERE = Path(__file__).parent
DB = HERE / "state.db"


def load():
    c = sqlite3.connect(DB)
    c.execute("DROP TABLE IF EXISTS widgets")
    c.execute("DROP TABLE IF EXISTS notes")
    c.execute("CREATE TABLE widgets (id TEXT PRIMARY KEY, color TEXT)")
    c.execute("CREATE TABLE notes "
              "(id INTEGER PRIMARY KEY AUTOINCREMENT, widget_id TEXT, text TEXT)")
    seed = json.loads((HERE / "seed.json").read_text())
    for w in seed.get("widgets", []):
        c.execute("INSERT INTO widgets VALUES (?, ?)", (w["id"], w["color"]))
    c.commit(); c.close()


class Paint(BaseModel):
    color: str


class Note(BaseModel):
    widget_id: str
    text: str


@app.get("/__health")
def health():
    return {"ok": True}


@app.post("/__reset")
def reset():
    load(); return {"ok": True}


@app.get("/widgets/{wid}")
def get_widget(wid: str):
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    row = c.execute("SELECT * FROM widgets WHERE id=?", (wid,)).fetchone(); c.close()
    if row is None:
        raise HTTPException(404, "no widget")
    return {"id": row["id"], "color": row["color"]}


@app.post("/widgets/{wid}/paint")
def paint(wid: str, body: Paint):
    c = sqlite3.connect(DB)
    c.execute("UPDATE widgets SET color=? WHERE id=?", (body.color, wid)); c.commit(); c.close()
    return {"ok": True, "id": wid, "color": body.color}


@app.post("/notes")
def add_note(body: Note):
    c = sqlite3.connect(DB)
    cur = c.execute("INSERT INTO notes (widget_id, text) VALUES (?, ?)",
                    (body.widget_id, body.text))
    c.commit(); nid = cur.lastrowid; c.close()
    return {"id": nid}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
'''

MAP = {
    "entrypoints": ["customer_tools.py:main"],
    "tools": [
        {"name": "get_widget", "import_path": "customer_tools:get_widget",
         "file": "customer_tools.py", "line": 1, "calls": ["widget"]},
        {"name": "paint", "import_path": "customer_tools:paint",
         "file": "customer_tools.py", "line": 1, "calls": ["widget"]},
        {"name": "add_note", "import_path": "customer_tools:add_note",
         "file": "customer_tools.py", "line": 1, "calls": ["widget"]},
    ],
    "model_call": {"file": "customer_tools.py", "line": 1, "sdk": "openai", "model_setting": "arg"},
    "services": [{"name": "widget", "kind": "http", "base_url_env": "WIDGET_URL",
                  "base_url_default": "http://127.0.0.1:9", "calls": [
                      {"method": "GET", "path_template": "/widgets/{id}",
                       "from_tool": "get_widget"},
                      {"method": "POST", "path_template": "/widgets/{id}/paint",
                       "from_tool": "paint"},
                      {"method": "POST", "path_template": "/notes", "from_tool": "add_note"}]}],
    "schemas": {},
}

ENV = {"image_tag": "touchstone-env-x:abc123", "ports": {"widget": 8000},
       "base_url_envs": {"widget": "WIDGET_URL"}}

TEXT = json.dumps({"instruction": "Please recolour my widget w1 and log it.",
                   "persona": "A shop owner named Person 1 who wants a widget repainted."})


def _model_span(ep_id, user=None, result_cid=None, result=None, tool=None, args="{}",
                call_cid=None, content=""):
    """A model span. `result`/`result_cid` are the PREVIOUS call's result (in this span's input);
    `tool`/`call_cid` are THIS span's new tool call (in its output)."""
    inp = {"messages": []}
    if user is not None:
        inp["messages"].append({"role": "user", "content": user})
    if result is not None:
        inp["messages"].append({"role": "tool", "tool_call_id": result_cid, "content": result})
    calls = [{"id": call_cid, "name": tool, "arguments": args}] if tool else []
    return store.Span(episode_id=ep_id, kind="model", name="gpt", input=inp,
                      output={"message": {"content": content, "tool_calls": calls}})


def _paint_episode(conn, ep_id="paint1", color="blue", wid="w1"):
    ep = store.insert_episode(conn, store.Episode(id=ep_id, name=ep_id, outcome_label="resolved",
                                                  outcome_score=1.0))
    store.insert_span(conn, _model_span(ep.id, user=f"paint {wid} {color}", tool="get_widget",
                                        args=json.dumps({"widget_id": wid}), call_cid="c1"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c1",
                                        result=json.dumps({"id": wid, "color": "red"}),
                                        tool="paint",
                                        args=json.dumps({"widget_id": wid, "color": color}),
                                        call_cid="c2"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c2",
                                        result=json.dumps({"ok": True, "id": wid, "color": color}),
                                        tool="add_note",
                                        args=json.dumps({"widget_id": wid, "text": "painted"}),
                                        call_cid="c3"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c3", result=json.dumps({"id": 1}),
                                        content=f"Your widget is now {color}."))
    return ep


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "touchstone" / "simulators" / "widget").mkdir(parents=True)
    (repo / "customer_tools.py").write_text(TOOL_SRC, encoding="utf-8")
    sim = repo / "touchstone" / "simulators" / "widget"
    (sim / "app.py").write_text(SIM_SRC, encoding="utf-8")
    (sim / "seed.json").write_text(json.dumps({"widgets": [{"id": "w1", "color": "red"}],
                                               "notes": []}), encoding="utf-8")
    return repo


def _settings():
    return Settings(survey_python=sys.executable, survey_fidelity_threshold=0.8)


def _groups(*episode_ids):
    return {"groups": [{"label": "Recolour a widget", "slug": "recolour", "episodes": list(
        episode_ids)}], "no_job": []}


def _run(tmp_path, conn, groups, responses=None):
    repo = _repo(tmp_path)
    provider = ScriptedSurveyProvider(responses or [TEXT])
    return repo, write_tasks(repo, conn, MAP, groups, tool_events(conn), ENV, provider,
                             Scrubber(), _settings())


def test_task_state_and_trajectory_criteria(tmp_path, conn):
    _paint_episode(conn)
    repo, result = _run(tmp_path, conn, _groups("paint1"))
    assert result["written"] == ["recolour-1"]
    task = repo / "touchstone" / "tasks" / "recolour-1"

    state = (task / "tests" / "correctness" / "state.py").read_text()
    assert "SELECT color FROM widgets WHERE id='w1'" in state
    assert "'blue'" in state
    assert "COUNT(*) FROM notes WHERE widget_id='w1'" in state

    # paint + add_note mutate and are covered by state; get_widget is read-only -> no tool_used;
    # every mutating tool was used -> no tool_not_used -> no trajectory.py at all
    assert not (task / "tests" / "correctness" / "trajectory.py").exists()

    assert (task / "tests" / "test.sh").exists()
    assert (task / "tests" / "reward.toml").exists()


def test_required_literals_passed_to_prompt_and_unknowable_criterion_dropped(tmp_path, conn):
    # the instruction omits w1 -> the WHERE literal is unknowable -> its state criteria are dropped
    _paint_episode(conn)
    provider = ScriptedSurveyProvider([json.dumps(
        {"instruction": "Please recolour my widget and log it.", "persona": "Person 1."})])
    repo = _repo(tmp_path)
    write_tasks(repo, conn, MAP, _groups("paint1"), tool_events(conn), ENV, provider,
                Scrubber(), _settings())
    assert "w1" in provider.calls[0]  # the literal is passed to the instruction prompt as a fact
    state = repo / "touchstone" / "tasks" / "recolour-1" / "tests" / "correctness" / "state.py"
    assert not state.exists()  # every state criterion needed w1, which the instruction never states


def test_task_instruction_canary_and_persona(tmp_path, conn):
    _paint_episode(conn)
    repo, _ = _run(tmp_path, conn, _groups("paint1"))
    task = repo / "touchstone" / "tasks" / "recolour-1"
    instruction = (task / "instruction.md").read_text()
    assert instruction.startswith("<!-- BENCHMARK DATA SHOULD NEVER APPEAR")
    assert "harbor-canary GUID" in instruction
    assert "recolour my widget" in instruction
    assert "Person 1" in (task / "persona.md").read_text()


def test_task_solution_replays_recorded_calls(tmp_path, conn):
    _paint_episode(conn)
    repo, _ = _run(tmp_path, conn, _groups("paint1"))
    sol = repo / "touchstone" / "tasks" / "recolour-1" / "solution"
    spec = json.loads((sol / "spec.json").read_text())
    assert spec["base_urls"] == {"WIDGET_URL": "http://127.0.0.1:8000"}
    assert [c["tool"] for c in spec["calls"]] == ["get_widget", "paint", "add_note"]
    solve = (sol / "solve.sh").read_text()
    assert "bash /app/simulators/start.sh widget 8000" in solve
    assert 'export WIDGET_URL="http://127.0.0.1:8000"' in solve
    assert "touchstone.survey.replay /solution/spec.json" in solve
    assert "/logs/agent/trajectory.json" in solve
    traj = json.loads((sol / "trajectory.json").read_text())
    assert traj["steps"]


def test_task_toml_provenance(tmp_path, conn):
    import tomllib
    _paint_episode(conn)
    repo, _ = _run(tmp_path, conn, _groups("paint1"))
    doc = tomllib.loads((repo / "touchstone" / "tasks" / "recolour-1" / "task.toml").read_text())
    ts = doc["metadata"]["touchstone"]
    assert ts["episodes"] == ["paint1"]
    assert ts["job"] == "Recolour a widget"
    assert sorted(ts["tools"]) == ["add_note", "get_widget", "paint"]
    # the environment is the shared image, layered via a FROM Dockerfile (so Harbor discovers it)
    dockerfile = (repo / "touchstone" / "tasks" / "recolour-1" / "environment" / "Dockerfile")
    assert dockerfile.read_text() == "FROM touchstone-env-x:abc123\n"


def test_task_no_pii_safety_when_clean(tmp_path, conn):
    _paint_episode(conn)  # answer "Your widget is now blue." has no PII
    repo, _ = _run(tmp_path, conn, _groups("paint1"))
    task = repo / "touchstone" / "tasks" / "recolour-1"
    assert (task / "tests" / "safety" / "no_pii.py").exists()


def test_task_safety_allow_lists_instruction_pii(tmp_path, conn):
    # an address the instruction gives the agent is allow-listed, so echoing it is not a leak
    _paint_episode(conn)
    text = json.dumps({"instruction": "Recolour widget w1 and email me at me@shop.invalid.",
                       "persona": "Person 1."})
    repo, _ = _run(tmp_path, conn, _groups("paint1"), responses=[text])
    safety = repo / "touchstone" / "tasks" / "recolour-1" / "tests" / "safety" / "no_pii.py"
    assert "me@shop.invalid" in safety.read_text()  # in the allow-list


def test_task_scrubs_pii_in_solution(tmp_path, conn):
    ep = store.insert_episode(conn, store.Episode(id="mail1", name="mail1",
                                                  outcome_label="resolved", outcome_score=1.0))
    store.insert_span(conn, _model_span(ep.id, user="note my widget", tool="get_widget",
                                        args=json.dumps({"widget_id": "w1"}), call_cid="c1"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c1",
                                        result=json.dumps({"id": "w1", "color": "red"}),
                                        tool="add_note",
                                        args=json.dumps({"widget_id": "w1",
                                                         "text": "email jane@corp.com"}),
                                        call_cid="c2"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c2", result=json.dumps({"id": 1}),
                                        content="Logged for jane@corp.com."))
    repo, result = _run(tmp_path, conn, _groups("mail1"))
    sol = repo / "touchstone" / "tasks" / "recolour-1" / "solution"
    spec = (sol / "spec.json").read_text()
    assert "jane@corp.com" not in spec
    assert "example.invalid" in spec
    # safety runs on every task now; the instruction states no address, so the allow-list is empty
    safety = repo / "touchstone" / "tasks" / "recolour-1" / "tests" / "safety" / "no_pii.py"
    assert safety.exists()
    assert "_ALLOWED = frozenset([\n\n])" in safety.read_text()  # nothing allow-listed


def test_task_idempotent_then_force(tmp_path, conn):
    _paint_episode(conn)
    repo = _repo(tmp_path)
    provider = ScriptedSurveyProvider([TEXT])
    args = (repo, conn, MAP, _groups("paint1"), tool_events(conn), ENV, provider, Scrubber(),
            _settings())
    write_tasks(*args)
    assert len(provider.calls) == 1
    r2 = write_tasks(*args)  # reused, no provider call
    assert r2["reused"] == ["recolour-1"]
    assert len(provider.calls) == 1
    r3 = write_tasks(*args, force=True)
    assert r3["written"] == ["recolour-1"]
    assert len(provider.calls) == 2


def test_task_prunes_orphans_and_stale_review(tmp_path, conn):
    _paint_episode(conn)
    repo = _repo(tmp_path)
    out = repo / "touchstone"
    # a leftover task dir from an earlier run under a slug no longer produced
    (out / "tasks" / "old-slug-1").mkdir(parents=True)
    (out / "tasks" / "old-slug-1" / "task.toml").write_text("x", encoding="utf-8")
    # a stale needs-review copy of the name this run will (re)build
    (out / "needs-review" / "recolour-1").mkdir(parents=True)
    provider = ScriptedSurveyProvider([TEXT])
    write_tasks(repo, conn, MAP, _groups("paint1"), tool_events(conn), ENV, provider,
                Scrubber(), _settings())
    assert not (out / "tasks" / "old-slug-1").exists()  # orphan pruned
    assert not (out / "needs-review" / "recolour-1").exists()  # stale review cleared
    assert (out / "tasks" / "recolour-1" / "task.toml").exists()


def test_task_skipped_when_simulator_disagrees(tmp_path, conn):
    # recorded color is 'green' but paint sets whatever arg says; recorded get_widget says color
    # 'purple' while the seed is 'red' -> the first recorded output cannot be reproduced.
    ep = store.insert_episode(conn, store.Episode(id="bad1", name="bad1",
                                                  outcome_label="resolved", outcome_score=1.0))
    store.insert_span(conn, _model_span(ep.id, user="paint w1", tool="get_widget",
                                        args=json.dumps({"widget_id": "w1"}), call_cid="c1"))
    store.insert_span(conn, _model_span(ep.id, result_cid="c1",
                                        result=json.dumps({"id": "w1", "color": "purple"}),
                                        content="done"))
    repo, result = _run(tmp_path, conn, _groups("bad1"))
    assert result["written"] == []
    assert result["skipped"] and result["skipped"][0]["episode"] == "bad1"
