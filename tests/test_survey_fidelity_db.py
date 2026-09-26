"""Fidelity for a kind=="db" service: materialize state.db, replay the real tools against it (no
server), compare returned values. Covers a relational tool that reads the db directly and a
document-store tool driven through a generated invoke.py."""

import json
import sys

from touchstone.survey import db_sim
from touchstone.survey.fidelity import _replay_spec, measure_service
from touchstone.survey.recordings import ToolEvent
from touchstone.survey.scrub import Scrubber
from touchstone.survey.simulate import _replay_ctx


class _Settings:
    survey_fidelity_threshold = 0.8
    survey_python = sys.executable  # replay in-process, no `uv run`


WORLD = {"name": "world", "kind": "db", "base_url_env": None, "base_url_default": ":memory:"}


def test_replay_spec_carries_the_db_path():
    ctx = _replay_ctx(WORLD, [{"name": "get_user", "import_path": "db_tool:get_user"}])
    assert ctx["base_url_env"] == "TOUCHSTONE_DB_WORLD"
    spec = _replay_spec([], "/abs/state.db", ctx)
    assert spec["base_url_env"] == "TOUCHSTONE_DB_WORLD"
    assert spec["base_url"] == "/abs/state.db"
    assert "simulators" not in spec  # a db service has no host to net-shim


def test_relational_db_fidelity(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "db_tool.py").write_text(
        "import os, sqlite3\n"
        "def get_user(zip):\n"
        "    c = sqlite3.connect(os.environ['TOUCHSTONE_DB_WORLD'])\n"
        "    r = c.execute('SELECT name FROM users WHERE zip=?', (zip,)).fetchone()\n"
        "    return {'name': r[0]} if r else {'error': 'not found'}\n", encoding="utf-8")
    sim = tmp_path / "simulators" / "world"
    db_sim.write_sim(sim, {
        "schema.sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, zip TEXT);",
        "seed.json": {"users": [{"id": 1, "name": "Yusuf", "zip": "19122"}]}, "README.md": "x"})
    ctx = _replay_ctx(WORLD, [{"name": "get_user", "import_path": "db_tool:get_user"}])
    calls = [ToolEvent(tool="get_user", arguments={"zip": "19122"},
                       output={"name": "Yusuf"}, episode="e1")]
    result = measure_service(sim, repo, calls, ctx, _Settings(), Scrubber())
    assert result["score"] == 1.0, result


_INVOKE = """\
import os, sqlite3, json
def _load():
    c = sqlite3.connect(os.environ["TOUCHSTONE_DB_WORLD"])
    tables = [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    data = {t: {r[0]: json.loads(r[1]) for r in c.execute(f"SELECT id, doc FROM {t}")}
            for t in tables}
    return c, data
def invoke(name, arguments):
    import orders_tool
    c, data = _load()
    result = getattr(orders_tool, name)(data, **arguments)
    for t, docs in data.items():
        for i, doc in docs.items():
            c.execute(f"UPDATE {t} SET doc=? WHERE id=?",
                      (json.dumps(doc, ensure_ascii=False, sort_keys=True), i))
    c.commit(); c.close()
    return result
"""


def test_document_store_db_fidelity_through_invoke(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orders_tool.py").write_text(
        "import json\n"
        "def get_order(data, order_id):\n"
        "    return json.dumps(data['orders'][order_id])\n", encoding="utf-8")
    invoke_path = repo / "agent" / "invoke.py"
    invoke_path.parent.mkdir()
    invoke_path.write_text(_INVOKE, encoding="utf-8")
    sim = tmp_path / "simulators" / "world"
    db_sim.write_sim(sim, {"collections": {"orders": {"#W1": {"status": "pending", "total": 5}}},
                           "README.md": "x"})
    ctx = _replay_ctx(WORLD, [{"name": "get_order", "import_path": "orders_tool:get_order"}])
    calls = [ToolEvent(tool="get_order", arguments={"order_id": "#W1"},
                       output=json.dumps({"status": "pending", "total": 5}), episode="e1")]
    result = measure_service(sim, repo, calls, ctx, _Settings(), Scrubber(), str(invoke_path))
    assert result["score"] == 1.0, result
