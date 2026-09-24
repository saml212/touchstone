import sqlite3
import threading
from pathlib import Path

from touchstone import store


def test_roundtrip_and_json_columns(conn):
    ep = store.insert_episode(conn, store.Episode(name="e1", meta={"k": "v", "n": 3}))
    got = store.get_episode(conn, ep.id)
    assert got.name == "e1" and got.meta == {"k": "v", "n": 3}

    span = store.insert_span(
        conn, store.Span(episode_id=ep.id, kind="model", name="m", input={"messages": [1, 2]})
    )
    assert store.get_span(conn, span.id).input == {"messages": [1, 2]}
    assert [s.id for s in store.list_spans(conn, ep.id)] == [span.id]


def test_outcome_update_and_label_filter(conn):
    a = store.insert_episode(conn, store.Episode(name="a"))
    store.insert_episode(conn, store.Episode(name="b"))
    store.outcome(conn, a.id, 1.0, "resolved")
    got = store.get_episode(conn, a.id)
    assert got.outcome_score == 1.0 and got.outcome_label == "resolved" and got.ended_at
    assert [e.id for e in store.list_episodes(conn, "resolved")] == [a.id]
    assert store.list_episodes(conn, "nope") == []


def test_runs_and_results_by_target_and_run(conn):
    r1 = store.insert_run(conn, store.Run(target="demo", model_spec="scripted"))
    store.insert_run(conn, store.Run(target="other", model_spec="scripted"))
    assert [r.id for r in store.list_runs(conn, "demo")] == [r1.id]
    assert len(store.list_runs(conn)) == 2

    store.insert_result(conn, store.Result(run_id=r1.id, task="t1", passed=1, reward=1.0,
                                           check_results={"c": {"passed": True}}))
    store.insert_result(conn, store.Result(run_id=r1.id, task="t2", passed=0, reward=0.0))
    store.insert_result(conn, store.Result(run_id="other", task="t3", passed=1, reward=1.0))
    res = store.list_results(conn, r1.id)
    assert len(res) == 2 and {x.task for x in res} == {"t1", "t2"}
    assert next(x for x in res if x.task == "t1").check_results == {"c": {"passed": True}}
    assert next(x for x in res if x.task == "t1").reward == 1.0


def test_unicode_and_one_megabyte_output(conn):
    ep = store.insert_episode(conn, store.Episode(name="ünïçōdé 🗿 日本語"))
    big = "x" * (1024 * 1024)
    span = store.insert_span(
        conn, store.Span(episode_id=ep.id, kind="model", name="big", output={"blob": big})
    )
    assert store.get_episode(conn, ep.id).name == "ünïçōdé 🗿 日本語"
    assert len(store.get_span(conn, span.id).output["blob"]) == 1024 * 1024


def test_concurrent_writers(db):
    writers, per = 8, 50
    errors = []

    def worker():
        c = None
        try:
            c = store.connect(db)
            for _ in range(per):
                store.insert_episode(c, store.Episode(name="w"))
        except Exception as e:
            errors.append(e)
        finally:
            if c:
                c.close()

    threads = [threading.Thread(target=worker) for _ in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    c = store.connect(db)
    try:
        assert len(store.list_episodes(c)) == writers * per
    finally:
        c.close()


def test_v1_db_migrates_cleanly(db):
    """A v1 DB opens fine: episodes/spans are preserved; the authored tables are dropped, and the
    run record is recreated with the new columns (target, reward, task)."""
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(db)
    raw.executescript(
        "CREATE TABLE episodes (id TEXT PRIMARY KEY, name TEXT, source TEXT, started_at TEXT,"
        " ended_at TEXT, outcome_score REAL, outcome_label TEXT, meta TEXT);"
        "CREATE TABLE spans (id TEXT PRIMARY KEY, episode_id TEXT, parent_id TEXT, kind TEXT,"
        " name TEXT, model TEXT, started_at TEXT, ended_at TEXT, input TEXT, output TEXT,"
        " tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, error TEXT);"
        "CREATE TABLE checks (id TEXT PRIMARY KEY);"
        "CREATE TABLE tasks (id TEXT PRIMARY KEY);"
        "CREATE TABLE benchmarks (id TEXT PRIMARY KEY);"
        "CREATE TABLE room_checks (room_id TEXT, check_id TEXT);"
        "CREATE TABLE runs (id TEXT PRIMARY KEY, benchmark_id TEXT, model_spec TEXT,"
        " started_at TEXT, finished_at TEXT, meta TEXT);"
        "CREATE TABLE results (run_id TEXT, task_id TEXT, passed INTEGER);"
        "PRAGMA user_version=1;"
    )
    raw.execute("INSERT INTO episodes (id, name) VALUES ('e1', 'kept')")
    raw.commit()
    raw.close()

    conn = store.connect(db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
        assert store.get_episode(conn, "e1").name == "kept"  # trace data preserved
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not ({"tasks", "checks", "benchmarks", "room_checks"} & tables)
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "target" in run_cols and "benchmark_id" not in run_cols
        result_cols = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
        assert {"task", "reward"} <= result_cols and "task_id" not in result_cols
    finally:
        conn.close()


def _v2_spans_table(path):
    """A minimal v2-schema DB: spans without tool_call_id, a span with the old 'llm' kind."""
    c = sqlite3.connect(str(path))
    c.executescript(
        "CREATE TABLE episodes (id TEXT PRIMARY KEY, name TEXT, source TEXT, started_at TEXT,"
        " ended_at TEXT, outcome_score REAL, outcome_label TEXT, meta TEXT);"
        "CREATE TABLE spans (id TEXT PRIMARY KEY, episode_id TEXT, parent_id TEXT, kind TEXT,"
        " name TEXT, model TEXT, started_at TEXT, ended_at TEXT, input TEXT, output TEXT,"
        " tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, error TEXT);"
    )
    c.execute("INSERT INTO episodes (id, name) VALUES ('e1', 'ep')")
    c.execute("INSERT INTO spans (id, episode_id, kind, name) VALUES ('s1', 'e1', 'llm', 'm')")
    c.execute("PRAGMA user_version=2")
    c.commit()
    c.close()


def test_v2_llm_spans_migrate_to_model(tmp_path):
    path = tmp_path / "old.db"
    _v2_spans_table(path)
    conn = store.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
        span = store.get_span(conn, "s1")
        assert span.kind == "model"  # llm -> model
        assert span.tool_call_id is None  # new column present, defaults null
        cols = {row[1] for row in conn.execute("PRAGMA table_info(spans)")}
        assert "tool_call_id" in cols
    finally:
        conn.close()
