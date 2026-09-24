import asyncio

import httpx
import pytest

from touchstone import store, tasks
from touchstone.bench import benchmark, pricing, runner
from touchstone.checks import Check
from touchstone.llm import Reply
from touchstone.llm.openai_compat import OpenAICompatProvider


@pytest.fixture
def seeded(project):
    """A demo project (mined, enabled, synced) with a benchmark over every active task."""
    conn, root = project(12)
    benchmark.create(root, "all", all_tasks=True)
    ntasks = len(benchmark.resolve(root, "all"))
    return conn, root, ntasks


def test_scripted_run_is_deterministic_across_reruns(seeded):
    conn, root, ntasks = seeded
    a = runner.run(conn, root, "all", "scripted", concurrency=4)
    b = runner.run(conn, root, "all", "scripted", concurrency=4)
    view_a = sorted((runner.result_view(r) for r in store.list_results(conn, a.id)),
                    key=lambda v: v["task"])
    view_b = sorted((runner.result_view(r) for r in store.list_results(conn, b.id)),
                    key=lambda v: v["task"])
    assert view_a == view_b
    assert len(view_a) == ntasks


def test_concurrency_1_and_8_give_same_results(seeded):
    conn, root, _ = seeded
    r1 = runner.run(conn, root, "all", "scripted", concurrency=1)
    r8 = runner.run(conn, root, "all", "scripted", concurrency=8)
    v1 = sorted((runner.result_view(r) for r in store.list_results(conn, r1.id)),
                key=lambda v: v["task"])
    v8 = sorted((runner.result_view(r) for r in store.list_results(conn, r8.id)),
                key=lambda v: v["task"])
    assert v1 == v8


def test_run_marks_finished_stores_result_per_task_and_writes_files(seeded):
    conn, root, ntasks = seeded
    run = runner.run(conn, root, "all", "scripted")
    assert store.get_run(conn, run.id).finished_at is not None
    assert len(store.list_results(conn, run.id)) == ntasks
    from pathlib import Path
    run_dir = Path(root) / ".touchstone" / "runs" / run.id
    assert (run_dir / "run.json").exists() and (run_dir / "results.jsonl").exists()


def test_reference_run_scores_reward_one(seeded):
    conn, root, ntasks = seeded
    run = runner.run(conn, root, "all", "reference")
    results = store.list_results(conn, run.id)
    assert len(results) == ntasks
    assert all(r.reward == 1.0 and r.passed == 1 for r in results)


class _EveryThird:
    def __init__(self):
        self.calls = 0

    async def achat(self, messages, tools=None, json=False, timeout=60):
        self.calls += 1
        if self.calls % 3 == 0:
            raise RuntimeError("boom")
        return Reply(content="sorted", usage={"tokens_in": 3, "tokens_out": 1})


def test_partial_failure_leaves_errors_not_crashes(seeded):
    conn, root, ntasks = seeded
    run = runner.run(conn, root, "all", "flaky", provider=_EveryThird())
    results = store.list_results(conn, run.id)
    errored = [r for r in results if r.error]
    assert errored
    assert all(r.passed == 0 and r.reward == 0.0 for r in errored)
    assert len(results) == ntasks


class _Slow:
    async def achat(self, messages, tools=None, json=False, timeout=60):
        await asyncio.sleep(5)
        return Reply(content="late")


def test_timeout_becomes_an_error_result(seeded):
    conn, root, _ = seeded
    run = runner.run(conn, root, "all", "slow", provider=_Slow(), timeout=0.05)
    results = store.list_results(conn, run.id)
    assert results and all("timed out" in (r.error or "") for r in results)


class _SyncOnly:
    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content="sorted", usage={"tokens_in": 2, "tokens_out": 1})


def test_sync_only_provider_runs_in_a_thread(seeded):
    conn, root, ntasks = seeded
    run = runner.run(conn, root, "all", "sync", provider=_SyncOnly())
    results = store.list_results(conn, run.id)
    assert len(results) == ntasks and all(r.error is None for r in results)


def test_cost_none_for_unknown_model_priced_for_known(seeded):
    conn, root, _ = seeded
    unknown = runner.run(conn, root, "all", "scripted")
    assert all(r.cost_usd is None for r in store.list_results(conn, unknown.id))

    known = runner.run(conn, root, "all", "openai:gpt-4o-mini", provider=_SyncOnly())
    costs = [r.cost_usd for r in store.list_results(conn, known.id)]
    assert all(c is not None and c > 0 for c in costs)


class _CannedJudge:
    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content='{"pass": true, "reason": "canned"}')

    async def achat(self, messages, tools=None, json=False, timeout=60):
        return self.chat(messages, tools, json, timeout)


def test_judge_check_is_evaluated_but_does_not_gate(conn, root):
    task = tasks.Task(
        name="judged-01",
        reference={"content": "All sorted, thanks!", "tool_calls": []},
        context={"messages": [{"role": "user", "content": "help"}], "tools": []},
        checks=[
            Check(kind="contains", params={"values": ["sorted"]}, name="says sorted",
                  source="manual"),
            Check(kind="judge", params={"rubric": "is it helpful?"}, name="j", severity="soft",
                  source="manual"),
        ],
    )
    tasks.write_task(root, task)
    benchmark.create(root, "one", all_tasks=True)
    run = runner.run(conn, root, "one", "scripted", judge_provider=_CannedJudge())
    result = store.list_results(conn, run.id)[0]
    assert result.check_results["j"]["passed"] is True
    # the judge is sampled, so its per-check agreement is recorded (canned -> unanimous).
    assert result.check_results["j"]["agreement"] == 1.0


def test_replay_tool_context_to_openai_is_valid_wire_and_no_errors(conn, root):
    # The regression: a task whose context has assistant tool calls + tool results used to 400
    # against the OpenAI API because the stored context was not OpenAI-shaped.
    import json as _json

    context = {"messages": [
        {"role": "user", "content": "where is order A1?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "name": "order_status", "arguments": '{"order_id": "A1"}'}]},
        {"role": "tool", "tool_call_id": "c1", "name": "order_status",
         "content": '{"status": "ok"}'},
    ], "tools": [{"type": "function", "function": {"name": "order_status"}}]}
    task = tasks.Task(name="toolctx-01", context=context,
                      reference={"content": "sorted", "tool_calls": []},
                      checks=[Check(kind="contains", params={"values": ["sorted"]},
                                    name="says sorted", source="manual")])
    tasks.write_task(root, task)
    benchmark.create(root, "ctx", all_tasks=True)
    saw_tool_context = {"hit": False}

    def handler(req):
        for msg in _json.loads(req.content)["messages"]:
            if msg["role"] == "tool":
                saw_tool_context["hit"] = True
                assert msg.get("tool_call_id"), "tool message needs tool_call_id"
            for tc in msg.get("tool_calls") or []:
                assert tc["type"] == "function" and "arguments" in tc["function"]
        body = {"choices": [{"message": {"role": "assistant", "content": "sorted"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        return httpx.Response(200, json=body)

    provider = OpenAICompatProvider(
        "https://api.openai.com/v1", "gpt-4o-mini", "sk-test",
        transport=httpx.MockTransport(handler), backoff=0)
    run = runner.run(conn, root, "ctx", "openai:gpt-4o-mini", provider=provider)
    results = store.list_results(conn, run.id)
    assert results and all(r.error is None for r in results)
    assert saw_tool_context["hit"], "expected at least one task with tool context"


def test_pricing_unknown_and_longest_match():
    assert pricing.cost_usd("scripted", {"tokens_in": 5, "tokens_out": 5}) is None
    assert pricing.price_for("openai:gpt-4o-mini") == pricing.PRICES["gpt-4o-mini"]
    assert pricing.cost_usd("openai:gpt-4o-mini", None) is None


@pytest.mark.parametrize("bad", [0, -1])
def test_run_rejects_nonpositive_concurrency(seeded, bad):
    conn, root, _ = seeded
    with pytest.raises(ValueError, match="concurrency"):
        runner.start(conn, root, "all", "scripted", concurrency=bad)


def test_start_rejects_empty_target(seeded):
    conn, root, _ = seeded
    with pytest.raises(ValueError, match="no active tasks"):
        runner.start(conn, root, "nonexistent", "scripted")
