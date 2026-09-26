"""db_sim materializes a kind=="db" simulator's state.db (relational + document-store shapes) and
db_service synthesizes the env var name/value the tools are pointed at."""

import json
import sqlite3

from touchstone.survey import db_service, db_sim
from touchstone.survey.scrub import Scrubber


def _rows(db, query):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def test_materialize_relational_schema_and_seed(tmp_path):
    files = {
        "schema.sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, zip TEXT);",
        "seed.json": {"users": [{"id": 1, "name": "Yusuf", "zip": "19122"},
                                {"id": 2, "name": "Ada", "zip": "10001"}]},
        "README.md": "users table",
    }
    db_sim.write_sim(tmp_path, files)
    db = db_sim.materialize(tmp_path)
    assert db is not None
    assert _rows(db, "SELECT name FROM users WHERE zip='19122'") == [("Yusuf",)]
    assert _rows(db, "SELECT COUNT(*) FROM users") == [(2,)]


def test_materialize_document_store_json_extract(tmp_path):
    files = {"collections": {"orders": {"#W1": {"status": "pending", "total": 5},
                                        "#W2": {"status": "delivered", "total": 9}}},
             "README.md": "orders documents"}
    db_sim.write_sim(tmp_path, files)
    assert db_sim.is_document_store(tmp_path)
    db = db_sim.materialize(tmp_path)
    got = _rows(db, "SELECT json_extract(doc,'$.status') FROM orders WHERE id='#W1'")
    assert got == [("pending",)]
    # doc column holds sorted-key JSON so a criterion reads deterministic text
    doc = _rows(db, "SELECT doc FROM orders WHERE id='#W2'")[0][0]
    assert json.loads(doc) == {"status": "delivered", "total": 9}


def test_unsupported_service(tmp_path):
    db_sim.write_sim(tmp_path, {"UNSUPPORTED": "raw psycopg with no URL env var"})
    assert db_sim.unsupported(tmp_path) == "raw psycopg with no URL env var"
    assert db_sim.materialize(tmp_path) is None
    assert not db_service.state_db(tmp_path).exists()


def test_parse_files_rejects_shapeless_answer(tmp_path):
    for shape in ('{"schema.sql": "CREATE TABLE t (id INTEGER)"}',
                  '{"collections": {}}', '{"UNSUPPORTED": "no sql"}'):
        assert isinstance(db_sim.parse_files(shape), dict)
    try:
        db_sim.parse_files('{"nonsense": 1}')
    except ValueError:
        return
    raise AssertionError("expected ValueError for a shapeless answer")


def test_env_name_and_value_path_vs_url():
    # synthesized when the map found no env var
    assert db_service.env_name({"name": "world", "kind": "db"}) == "TOUCHSTONE_DB_WORLD"
    # the mapped env var wins
    assert db_service.env_name({"name": "world", "base_url_env": "APP_DB"}) == "APP_DB"
    # a bare path when base_url_default is a file/:memory:
    file_svc = {"name": "world", "kind": "db", "base_url_default": ":memory:"}
    assert db_service.env_value("/x/state.db", file_svc) == "/x/state.db"
    # a sqlite URL when the code reads a URL
    url_svc = {"name": "world", "kind": "db", "base_url_default": "postgresql://h/db"}
    assert db_service.env_value("/x/state.db", url_svc) == "sqlite:////x/state.db"


class _JunkProvider:
    name = "junk"

    def run(self, prompt, cwd):
        return "sorry, I could not produce JSON"


class _S:
    survey_fidelity_threshold = 0.8
    survey_python = None


def test_generation_failure_is_unsupported_not_fatal(tmp_path):
    from touchstone.survey.db_sim import generate_db_simulator

    repo = tmp_path / "repo"
    repo.mkdir()
    service = {"name": "world", "kind": "db", "base_url_env": None, "base_url_default": ":memory:"}
    tools = [{"name": "t", "import_path": "m:t", "file": None}]
    result = generate_db_simulator(repo, _JunkProvider(), service, tools, [],
                                   tmp_path / "simulators", Scrubber(), _S())
    assert result["score"] == 0.0
    assert "unsupported" in result and result["unsupported"]  # flagged, survey continues
    assert db_sim.unsupported(tmp_path / "simulators" / "world")
