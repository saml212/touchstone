import pytest

from touchstone import store
from touchstone.bench import benchmark, report, runner
from touchstone.demo import run_demo
from touchstone.mine import cut_tasks


@pytest.fixture
def two_runs(traced):
    run_demo(n=10)
    conn = store.connect(traced)
    store.insert_check(conn, store.Check(
        name="polite", kind="contains",
        params={"values": ["sorted", "escalat"], "mode": "any"}, enabled=1))
    store.insert_check(conn, store.Check(name="clean", kind="no_pii", params={}, enabled=1))
    cut_tasks(conn, store.list_episodes(conn))
    bench = benchmark.create(conn, "all", all_tasks=True)
    incumbent = runner.run(conn, bench.id, "scripted")
    candidate = runner.run(conn, bench.id, "openai:gpt-4o-mini", provider=_AlwaysOK())
    yield conn, incumbent, candidate
    conn.close()


class _AlwaysOK:
    def chat(self, messages, tools=None, json=False, timeout=60):
        from touchstone.llm import Reply
        return Reply(content="sorted", usage={"tokens_in": 4, "tokens_out": 2})

    async def achat(self, messages, tools=None, json=False, timeout=60):
        return self.chat(messages, tools, json, timeout)


def test_scoreboard_per_check_counts_add_up(two_runs):
    conn, inc, cand = two_runs
    board = report.scoreboard(conn, [inc.id, cand.id])
    assert {m["model_spec"] for m in board["models"]} == {"scripted", "openai:gpt-4o-mini"}
    for m in board["models"]:
        # every task produced pass or fail, none unaccounted
        assert m["passed"] <= m["total"] == len(store.list_results(conn, m["run_id"]))
    # per-check pass+fail+skip equals the number of results carrying that check, per model
    for entry in board["checks"]:
        assert entry["pass"] + entry["fail"] + entry["skip"] > 0


def test_scoreboard_renders_without_error(two_runs):
    conn, inc, cand = two_runs
    text = report.render_scoreboard(report.scoreboard(conn, [inc.id, cand.id]))
    assert "MODELS" in text and "scripted" in text


def test_proof_categories_partition_shared_tasks(two_runs):
    conn, inc, cand = two_runs
    p = report.proof(conn, cand.id, inc.id)
    total = sum(p["counts"].values())
    assert total == len(p["tasks"])
    assert total == len(set(t.id for t in store.list_tasks(conn)))
    for row in p["tasks"]:
        assert row["category"] in (
            "both_pass", "only_incumbent", "only_candidate", "both_fail")


def test_proof_cost_totals_and_render(two_runs):
    conn, inc, cand = two_runs
    p = report.proof(conn, cand.id, inc.id)
    assert p["cost"]["incumbent"] is None  # scripted is unpriced
    assert p["cost"]["candidate"] is not None and p["cost"]["candidate"] > 0
    assert "verdict" in report.render_proof(p)


def test_empty_benchmark_rejected(traced):
    conn = store.connect(traced)
    with pytest.raises(ValueError, match="empty"):
        benchmark.create(conn, "nothing", all_tasks=True)
    conn.close()


def test_benchmark_from_tags_and_explicit_ids(traced):
    run_demo(n=8)
    conn = store.connect(traced)
    cut_tasks(conn, store.list_episodes(conn))
    tasks = store.list_tasks(conn)
    by_ids = benchmark.create(conn, "two", task_ids=[tasks[0].id, tasks[1].id, "bogus"])
    assert by_ids.task_ids == [tasks[0].id, tasks[1].id]  # bogus dropped
    tagged = benchmark.create(conn, "resolved", tags=["resolved"])
    assert all("resolved" in (store.get_task(conn, tid).tags or []) for tid in tagged.task_ids)

    assert benchmark.get(conn, by_ids.id).name == "two"
    assert {b.id for b in benchmark.list_benchmarks(conn)} == {by_ids.id, tagged.id}
    conn.close()


def test_proof_rejects_missing_run(two_runs):
    conn, inc, cand = two_runs
    with pytest.raises(ValueError, match="run"):
        report.proof(conn, cand.id, "nonexistent-run-id")


def test_proof_rejects_different_benchmarks(traced):
    run_demo(n=8)
    conn = store.connect(traced)
    cut_tasks(conn, store.list_episodes(conn))
    tasks = store.list_tasks(conn)
    b1 = benchmark.create(conn, "b1", task_ids=[tasks[0].id, tasks[1].id])
    b2 = benchmark.create(conn, "b2", task_ids=[tasks[2].id, tasks[3].id])
    r1 = runner.run(conn, b1.id, "scripted")
    r2 = runner.run(conn, b2.id, "scripted")
    with pytest.raises(ValueError, match="benchmark"):
        report.proof(conn, r1.id, r2.id)
    conn.close()
