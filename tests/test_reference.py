"""The `reference` provider: replays each task's recorded reply as the incumbent baseline."""

import pytest

from touchstone import store, tasks
from touchstone.bench import benchmark, runner
from touchstone.llm import NopProvider, ReferenceProvider, provider_from_spec


def test_reference_provider_returns_the_task_reference():
    prov = provider_from_spec("reference")
    assert isinstance(prov, ReferenceProvider)
    task = tasks.Task(name="t", reference={"content": "hello",
                                           "tool_calls": [{"name": "refund", "arguments": "{}"}]})
    reply = prov.chat([], task=task)
    assert reply.content == "hello"
    assert reply.tool_calls == [{"name": "refund", "arguments": "{}"}]


def test_reference_provider_empty_without_task():
    reply = provider_from_spec("reference").chat([{"role": "user", "content": "hi"}])
    assert reply.content == "" and reply.tool_calls == []


def test_nop_provider_is_the_empty_reply():
    prov = provider_from_spec("nop")
    assert isinstance(prov, NopProvider)
    reply = prov.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "" and reply.tool_calls == []


@pytest.mark.parametrize("spec", ["nop:x", "reference:x"])
def test_baseline_providers_take_no_argument(spec):
    with pytest.raises(ValueError):
        provider_from_spec(spec)


def test_reference_passes_every_active_task(project):
    """With the reference gate (a check attaches only when the reference passes it, and the oracle
    gate keeps only tasks the reference satisfies), reference passes every active task."""
    conn, root = project(16)
    benchmark.create(root, "all", all_tasks=True)
    run = runner.run(conn, root, "all", "reference")
    results = {r.task: r for r in store.list_results(conn, run.id)}

    active = [t for t in tasks.list_tasks(root, active_only=True)]
    assert active, "the demo should produce active tasks"
    for task in active:
        assert results[task.name].passed == 1, f"reference should pass active task {task.name}"
        assert results[task.name].reward == 1.0
