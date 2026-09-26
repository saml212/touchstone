"""Deterministic seeding: state.db is materialized from real repo data files (JSON dict / JSON list
/ CSV / SQLite) or the recorded reads — never model-written — with the copied contents scrubbed."""

import json
import sqlite3

from touchstone.survey import db_seed, db_sim
from touchstone.survey.recordings import ToolEvent
from touchstone.survey.scrub import Scrubber


def _rows(db, query):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def _repo_with(tmp_path, name, text):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / name).write_text(text, encoding="utf-8")
    return f"data/{name}"


def test_materialize_json_dict_of_docs(tmp_path):
    sim = tmp_path / "sim"
    rel = _repo_with(tmp_path, "orders.json",
                     json.dumps({"#W1": {"status": "pending"}, "#W2": {"status": "delivered"}}))
    db_seed.copy_data_files(sim, tmp_path, [rel], Scrubber())
    db = db_sim.materialize(sim)
    assert _rows(db, "SELECT json_extract(doc,'$.status') FROM orders WHERE id='#W1'") \
        == [("pending",)]


def test_materialize_json_list_keyed_by_id(tmp_path):
    sim = tmp_path / "sim"
    rel = _repo_with(tmp_path, "items.json",
                     json.dumps([{"id": "a1", "n": 1}, {"id": "a2", "n": 2}]))
    db_seed.copy_data_files(sim, tmp_path, [rel], Scrubber())
    db = db_sim.materialize(sim)
    assert _rows(db, "SELECT json_extract(doc,'$.n') FROM items WHERE id='a2'") == [(2,)]


def test_materialize_csv_relational(tmp_path):
    sim = tmp_path / "sim"
    rel = _repo_with(tmp_path, "users.csv", "id,zip\n1,19122\n2,10001\n")
    db_seed.copy_data_files(sim, tmp_path, [rel], Scrubber())
    db = db_sim.materialize(sim)
    assert _rows(db, "SELECT zip FROM users WHERE id='1'") == [("19122",)]


def test_materialize_sqlite_copied(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    conn = sqlite3.connect(str(src / "store.db"))
    conn.execute("CREATE TABLE t (id TEXT, email TEXT)")
    conn.execute("INSERT INTO t VALUES ('1', 'jane@x.com')")
    conn.commit()
    conn.close()
    sim = tmp_path / "sim"
    db_seed.copy_data_files(sim, tmp_path, ["data/store.db"], Scrubber())
    db = db_sim.materialize(sim)
    email = _rows(db, "SELECT email FROM t WHERE id='1'")[0][0]
    assert "jane@x.com" not in email and "@example.invalid" in email  # scrubbed in the copy


def test_scrub_runs_on_copied_files(tmp_path):
    sim = tmp_path / "sim"
    rel = _repo_with(tmp_path, "users.json",
                     json.dumps({"u1": {"email": "yusuf@shop.com"}}))
    db_seed.copy_data_files(sim, tmp_path, [rel], Scrubber())
    on_disk = (db_seed.source_dir(sim) / "users.json").read_text(encoding="utf-8")
    assert "yusuf@shop.com" not in on_disk  # no raw email ever enters source/


def test_seed_from_recorded_reads(tmp_path):
    sim = tmp_path / "sim"
    calls = [ToolEvent(tool="get_order", arguments={"order_id": "#W1"},
                       output={"id": "#W1", "status": "pending"}, episode="e")]
    assert db_seed.seed_recorded(sim, calls, Scrubber(), "orders")
    db = db_sim.materialize(sim)
    assert _rows(db, "SELECT json_extract(doc,'$.status') FROM orders WHERE id='#W1'") \
        == [("pending",)]


def test_data_files_for_derives_from_brace_default(tmp_path):
    _repo_with(tmp_path, "orders.json", "{}")
    _repo_with(tmp_path, "users.json", "{}")
    svc = {"name": "s", "kind": "db",
           "base_url_default": "data/{orders,users}.json"}
    assert db_seed.data_files_for(tmp_path, svc) == ["data/orders.json", "data/users.json"]


def test_data_files_for_prefers_explicit_and_skips_urls(tmp_path):
    rel = _repo_with(tmp_path, "store.json", "{}")
    assert db_seed.data_files_for(tmp_path, {"data_files": [rel]}) == [rel]
    assert db_seed.data_files_for(tmp_path, {"base_url_default": "postgresql://h/db"}) == []
