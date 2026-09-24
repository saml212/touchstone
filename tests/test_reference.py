"""The `reference` provider: replays each task's recorded reply as the incumbent baseline."""

from touchstone import store
from touchstone.bench import benchmark, runner
from touchstone.demo import run_demo
from touchstone.llm import ReferenceProvider, provider_from_spec
from touchstone.mine import cut_tasks


def test_reference_provider_returns_the_task_reference():
    prov = provider_from_spec("reference")
    assert isinstance(prov, ReferenceProvider)
    task = store.Task(name="t", reference={"content": "hello",
                                           "tool_calls": [{"name": "refund", "arguments": "{}"}]})
    reply = prov.chat([], task=task)
    assert reply.content == "hello"
    assert reply.tool_calls == [{"name": "refund", "arguments": "{}"}]


def test_reference_provider_empty_without_task():
    reply = provider_from_spec("reference").chat([{"role": "user", "content": "hi"}])
    assert reply.content == "" and reply.tool_calls == []


def test_reference_passes_every_non_failure_task(traced):
    """With reference-consistent checks (attached only when the reference passes), the reference
    provider passes every non-failure task by construction."""
    run_demo(n=16)
    conn = store.connect(traced)
    try:
        store.insert_check(conn, store.Check(
            name="polite", kind="contains",
            params={"values": ["sorted", "escalat"], "mode": "any"}, enabled=1))
        cut_tasks(conn, store.list_episodes(conn))
        bench = benchmark.create(conn, "all", all_tasks=True)

        run = runner.run(conn, bench.id, "reference")
        results = {r.task_id: r for r in store.list_results(conn, run.id)}

        non_failure = [t for t in store.list_tasks(conn) if "failure" not in (t.tags or [])]
        assert non_failure, "demo should produce non-failure tasks"
        for task in non_failure:
            assert results[task.id].passed == 1, f"reference should pass non-failure task {task.id}"
    finally:
        conn.close()


def test_reference_result_matches_evaluating_checks_against_the_reference(traced):
    """A safety check ('avoid this') always attaches, even when the recorded reply violates it —
    so the reference run mirrors evaluating each task's checks against its own reference."""
    from dataclasses import asdict

    from touchstone.checks import Check as DslCheck
    from touchstone.checks import Target, evaluate, passes

    run_demo(n=16)
    conn = store.connect(traced)
    try:
        store.insert_check(conn, store.Check(name="clean", kind="no_pii", params={}, enabled=1))
        cut_tasks(conn, store.list_episodes(conn))
        bench = benchmark.create(conn, "all", all_tasks=True)
        run = runner.run(conn, bench.id, "reference")
        results = {r.task_id: r for r in store.list_results(conn, run.id)}

        leaked = 0
        for task in store.list_tasks(conn):
            checks = [DslCheck.from_dict(asdict(store.get_check(conn, c)))
                      for c in (task.check_ids or [])]
            ref = task.reference or {}
            target = Target(output_text=ref.get("content") or "",
                            tool_calls=ref.get("tool_calls") or [], reference=ref)
            expected = passes(evaluate(checks, target), checks) if checks else True
            assert results[task.id].passed == (1 if expected else 0)
            if checks and not expected:
                leaked += 1
        assert leaked, "the demo should leave at least one PII-violating reference"
    finally:
        conn.close()
