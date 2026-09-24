import pytest

from touchstone import store, tasks
from touchstone.bench import benchmark, report, runner
from touchstone.llm import Reply


class _AlwaysOK:
    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content="sorted", usage={"tokens_in": 4, "tokens_out": 2})

    async def achat(self, messages, tools=None, json=False, timeout=60):
        return self.chat(messages, tools, json, timeout)


@pytest.fixture
def two_runs(project):
    conn, root = project(10)
    benchmark.create(root, "all", all_tasks=True)
    incumbent = runner.run(conn, root, "all", "scripted")
    candidate = runner.run(conn, root, "all", "openai:gpt-4o-mini", provider=_AlwaysOK())
    return conn, root, incumbent, candidate


def test_scoreboard_per_check_counts_add_up(two_runs):
    conn, root, inc, cand = two_runs
    board = report.scoreboard(conn, root, [inc.id, cand.id])
    assert {m["model_spec"] for m in board["models"]} == {"scripted", "openai:gpt-4o-mini"}
    for m in board["models"]:
        assert m["passed"] <= m["total"] == len(store.list_results(conn, m["run_id"]))
    for entry in board["checks"]:
        assert entry["pass"] + entry["fail"] + entry["skip"] > 0
        assert entry["name"] and entry["kind"]


def test_scoreboard_renders_without_error(two_runs):
    conn, root, inc, cand = two_runs
    text = report.render_scoreboard(report.scoreboard(conn, root, [inc.id, cand.id]))
    assert "MODELS" in text and "scripted" in text


def test_proof_categories_partition_shared_tasks(two_runs):
    conn, root, inc, cand = two_runs
    p = report.proof(conn, cand.id, inc.id)
    total = sum(p["counts"].values())
    assert total == len(p["tasks"])
    assert total == len(benchmark.resolve(root, "all"))
    for row in p["tasks"]:
        assert row["category"] in (
            "both_pass", "only_incumbent", "only_candidate", "both_fail")


def test_proof_cost_totals_and_render(two_runs):
    conn, root, inc, cand = two_runs
    p = report.proof(conn, cand.id, inc.id)
    assert p["cost"]["incumbent"] is None  # scripted is unpriced
    assert p["cost"]["candidate"] is not None and p["cost"]["candidate"] > 0
    assert "verdict" in report.render_proof(p)


def test_empty_benchmark_rejected(root):
    with pytest.raises(ValueError, match="empty"):
        benchmark.create(root, "nothing", all_tasks=True)


def test_benchmark_from_tags_explicit_names_and_listing(project):
    conn, root = project(8)
    active = [t.name for t in tasks.list_tasks(root, active_only=True)]
    benchmark.create(root, "two", task_names=[active[0], active[1], "bogus"])
    resolved = [d.name for d in benchmark.resolve(root, "two")]
    assert resolved == [active[0], active[1]]  # bogus dropped

    benchmark.create(root, "resolved", tags=["resolved"])
    for d in benchmark.resolve(root, "resolved"):
        assert "resolved" in (tasks.read_task(d).tags or [])

    assert set(benchmark.list_names(root)) == {"two", "resolved"}
    assert benchmark.view(root, "two")["task_count"] == 2


def test_glob_target_resolves_active_tasks(project):
    conn, root = project(8)
    dirs = benchmark.resolve(root, "tasks/*")
    assert dirs and all(tasks.read_task(d).status == "active" for d in dirs)


def test_proof_rejects_missing_run(two_runs):
    conn, root, inc, cand = two_runs
    with pytest.raises(ValueError, match="run"):
        report.proof(conn, cand.id, "nonexistent-run-id")


def test_proof_rejects_different_targets(project):
    conn, root = project(8)
    active = [t.name for t in tasks.list_tasks(root, active_only=True)]
    benchmark.create(root, "b1", task_names=[active[0]])
    benchmark.create(root, "b2", task_names=[active[1]])
    r1 = runner.run(conn, root, "b1", "scripted")
    r2 = runner.run(conn, root, "b2", "scripted")
    with pytest.raises(ValueError, match="target"):
        report.proof(conn, r1.id, r2.id)
