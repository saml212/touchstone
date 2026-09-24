import asyncio

import pytest

from touchstone import store
from touchstone.bench import benchmark, pricing, runner
from touchstone.demo import run_demo
from touchstone.llm import Reply
from touchstone.mine import cut_tasks


@pytest.fixture
def seeded(traced):
    """A demo db with mined tasks, one enabled positive check and one safety check."""
    run_demo(n=12)
    conn = store.connect(traced)
    store.insert_check(conn, store.Check(
        name="polite", kind="contains", params={"values": ["sorted", "escalat", "reply"],
                                                 "mode": "any"}, enabled=1))
    store.insert_check(conn, store.Check(name="clean", kind="no_pii", params={}, enabled=1))
    cut_tasks(conn, store.list_episodes(conn))
    bench = benchmark.create(conn, "all", all_tasks=True)
    yield conn, bench
    conn.close()


def test_scripted_run_is_deterministic_across_reruns(seeded):
    conn, bench = seeded
    a = runner.run(conn, bench.id, "scripted", concurrency=4)
    b = runner.run(conn, bench.id, "scripted", concurrency=4)
    view_a = sorted((runner.result_view(r) for r in store.list_results(conn, a.id)),
                    key=lambda v: v["task_id"])
    view_b = sorted((runner.result_view(r) for r in store.list_results(conn, b.id)),
                    key=lambda v: v["task_id"])
    assert view_a == view_b
    assert len(view_a) == len(bench.task_ids)


def test_concurrency_1_and_8_give_same_results(seeded):
    conn, bench = seeded
    r1 = runner.run(conn, bench.id, "scripted", concurrency=1)
    r8 = runner.run(conn, bench.id, "scripted", concurrency=8)
    v1 = sorted((runner.result_view(r) for r in store.list_results(conn, r1.id)),
                key=lambda v: v["task_id"])
    v8 = sorted((runner.result_view(r) for r in store.list_results(conn, r8.id)),
                key=lambda v: v["task_id"])
    assert v1 == v8


def test_run_marks_finished_and_stores_a_result_per_task(seeded):
    conn, bench = seeded
    run = runner.run(conn, bench.id, "scripted")
    assert store.get_run(conn, run.id).finished_at is not None
    assert len(store.list_results(conn, run.id)) == len(bench.task_ids)


class _EveryThird:
    """Raises on every 3rd call; otherwise a trivial reply. Exercises partial failure."""

    def __init__(self):
        self.calls = 0

    async def achat(self, messages, tools=None, json=False, timeout=60):
        self.calls += 1
        if self.calls % 3 == 0:
            raise RuntimeError("boom")
        return Reply(content="sorted", usage={"tokens_in": 3, "tokens_out": 1})


def test_partial_failure_leaves_errors_not_crashes(seeded):
    conn, bench = seeded
    run = runner.run(conn, bench.id, "flaky", provider=_EveryThird())
    results = store.list_results(conn, run.id)
    errored = [r for r in results if r.error]
    assert errored, "expected some tasks to error"
    assert all(r.passed == 0 for r in errored)
    assert len(results) == len(bench.task_ids)  # the run completed despite failures


class _Slow:
    async def achat(self, messages, tools=None, json=False, timeout=60):
        await asyncio.sleep(5)
        return Reply(content="late")


def test_timeout_becomes_an_error_result(seeded):
    conn, bench = seeded
    run = runner.run(conn, bench.id, "slow", provider=_Slow(), timeout=0.05)
    results = store.list_results(conn, run.id)
    assert results and all("timed out" in (r.error or "") for r in results)


class _SyncOnly:
    """No achat: the runner must fall back to a thread."""

    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content="sorted", usage={"tokens_in": 2, "tokens_out": 1})


def test_sync_only_provider_runs_in_a_thread(seeded):
    conn, bench = seeded
    run = runner.run(conn, bench.id, "sync", provider=_SyncOnly())
    results = store.list_results(conn, run.id)
    assert len(results) == len(bench.task_ids)
    assert all(r.error is None for r in results)


def test_cost_none_for_unknown_model_priced_for_known(seeded):
    conn, bench = seeded
    unknown = runner.run(conn, bench.id, "scripted")
    assert all(r.cost_usd is None for r in store.list_results(conn, unknown.id))

    known = runner.run(conn, bench.id, "openai:gpt-4o-mini", provider=_SyncOnly())
    costs = [r.cost_usd for r in store.list_results(conn, known.id)]
    assert all(c is not None and c > 0 for c in costs)


class _CannedJudge:
    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content='{"pass": true, "reason": "canned"}')

    async def achat(self, messages, tools=None, json=False, timeout=60):
        return self.chat(messages, tools, json, timeout)


def test_judge_provider_path(traced):
    run_demo(n=4)
    conn = store.connect(traced)
    judge = store.insert_check(conn, store.Check(
        name="j", kind="judge", params={"rubric": "is it helpful"}, severity="soft", enabled=1))
    cut_tasks(conn, store.list_episodes(conn))
    # attach the judge check to every task (soft, so it reports but never gates)
    for t in store.list_tasks(conn):
        ids = list(t.check_ids or [])
        if judge.id not in ids:
            ids.append(judge.id)
        store.update_task(conn, t.id, check_ids=ids)
    bench = benchmark.create(conn, "all", all_tasks=True)

    run = runner.run(conn, bench.id, "scripted", judge_provider=_CannedJudge())
    results = store.list_results(conn, run.id)
    judged = [cr for r in results for cr in (r.check_results or []) if cr["check_id"] == judge.id]
    assert judged and all(cr["passed"] is True for cr in judged)
    conn.close()


def test_pricing_unknown_and_longest_match():
    assert pricing.cost_usd("scripted", {"tokens_in": 5, "tokens_out": 5}) is None
    # gpt-4o-mini must win over gpt-4o
    assert pricing.price_for("openai:gpt-4o-mini") == pricing.PRICES["gpt-4o-mini"]
    assert pricing.cost_usd("openai:gpt-4o-mini", None) is None
