import threading

from touchstone import store


def test_roundtrip_and_json_columns(conn):
    ep = store.insert_episode(conn, store.Episode(name="e1", meta={"k": "v", "n": 3}))
    got = store.get_episode(conn, ep.id)
    assert got.name == "e1" and got.meta == {"k": "v", "n": 3}

    span = store.insert_span(
        conn, store.Span(episode_id=ep.id, kind="llm", name="m", input={"messages": [1, 2]})
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


def test_tasks_by_tag_and_results_by_run(conn):
    store.insert_task(conn, store.Task(name="t1", tags=["failure", "x"]))
    store.insert_task(conn, store.Task(name="t2", tags=["x"]))
    assert [t.name for t in store.list_tasks(conn, "failure")] == ["t1"]
    assert len(store.list_tasks(conn)) == 2

    b = store.insert_benchmark(conn, store.Benchmark(name="b"))
    r = store.insert_run(conn, store.Run(benchmark_id=b.id, model_spec="scripted"))
    store.insert_result(conn, store.Result(run_id=r.id, task_id="t1", passed=1))
    store.insert_result(conn, store.Result(run_id=r.id, task_id="t2", passed=0))
    store.insert_result(conn, store.Result(run_id="other", task_id="t3", passed=1))
    res = store.list_results(conn, r.id)
    assert len(res) == 2 and {x.task_id for x in res} == {"t1", "t2"}


def test_check_enable_disable(conn):
    c = store.insert_check(conn, store.Check(name="c", kind="contains"))
    assert c.enabled == 0
    store.set_check_enabled(conn, c.id, True)
    assert store.get_check(conn, c.id).enabled == 1
    assert [x.id for x in store.list_checks(conn, enabled=True)] == [c.id]


def test_unicode_and_one_megabyte_output(conn):
    ep = store.insert_episode(conn, store.Episode(name="ünïçōdé 🗿 日本語"))
    big = "x" * (1024 * 1024)
    span = store.insert_span(
        conn, store.Span(episode_id=ep.id, kind="llm", name="big", output={"blob": big})
    )
    assert store.get_episode(conn, ep.id).name == "ünïçōdé 🗿 日本語"
    assert len(store.get_span(conn, span.id).output["blob"]) == 1024 * 1024


def test_concurrent_writers(db):
    writers, per = 8, 50

    def worker():
        c = store.connect(db)
        try:
            for _ in range(per):
                store.insert_episode(c, store.Episode(name="w"))
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    c = store.connect(db)
    try:
        assert len(store.list_episodes(c)) == writers * per
    finally:
        c.close()
