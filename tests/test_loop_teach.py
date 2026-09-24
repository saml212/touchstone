"""Teacher demonstrations: accept only verified replies; adopt oracles for needs_solution tasks."""

from touchstone import store, tasks
from touchstone.checks import Check
from touchstone.llm import Rule, ScriptedProvider
from touchstone.loop import teach


def _teacher():
    # a scripted teacher that always answers with the phrase the checks require
    return ScriptedProvider(rules=[Rule(substring="", content="here is your refund, all set")])


def _active_task(root, name):
    tasks.write_task(root, tasks.Task(
        name=name,
        context={"messages": [{"role": "user", "content": "refund please"}], "tools": []},
        reference={"content": "refund granted", "tool_calls": []},
        checks=[Check(kind="contains", params={"values": ["refund"]}, name="says refund",
                      source="policy")]))


def _needs_solution_task(root, name):
    # the recorded reply fails the hard check, so the task has no oracle yet
    tasks.write_task(root, tasks.Task(
        name=name, tags=["failure"],
        context={"messages": [{"role": "user", "content": "refund please"}], "tools": []},
        reference={"content": "escalated to a human", "tool_calls": []},
        checks=[Check(kind="contains", params={"values": ["refund"]}, name="says refund",
                      source="manual")]))
    assert tasks.get_task(root, name).status == "needs_solution"


def test_teach_stores_only_passing_demos(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t-ok")
    result = teach(conn, root, ["t-ok"], "scripted", teacher_provider=_teacher())
    assert result["accepted"] == ["t-ok"] and not result["failed"]
    demos = store.list_results(conn, result["run"])
    assert demos[0].task == "t-ok" and demos[0].passed == 1
    assert store.get_run(conn, result["run"]).model_spec == "teacher:scripted"


def test_teach_rejects_a_failing_demo(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t-ok")
    # a teacher that never says "refund" cannot pass the check
    dud = ScriptedProvider(rules=[Rule(substring="", content="I cannot help")])
    result = teach(conn, root, ["t-ok"], "scripted", teacher_provider=dud)
    assert result["failed"] == ["t-ok"] and not result["accepted"]
    assert store.list_results(conn, result["run"]) == []


def test_teach_adopts_reference_for_needs_solution(tmp_path, conn):
    root = str(tmp_path)
    _needs_solution_task(root, "t-fix")
    result = teach(conn, root, ["t-fix"], "scripted", teacher_provider=_teacher())
    assert result["accepted"] == ["t-fix"]
    fixed = tasks.get_task(root, "t-fix")
    assert fixed.status == "active"  # oracle now passes
    assert fixed.reference_from == "teacher:scripted"
    assert "refund" in fixed.reference["content"]


def test_teach_never_overwrites_an_active_reference(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t-ok")
    before = tasks.get_task(root, "t-ok").reference
    teach(conn, root, ["t-ok"], "scripted", teacher_provider=_teacher())
    after = tasks.get_task(root, "t-ok")
    assert after.reference == before and after.reference_from is None


def test_teach_without_provider_accepts_nothing(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t-ok")
    # a spec with no key/binary -> provider_or_none returns None -> everything is "failed"
    result = teach(conn, root, ["t-ok"], "anthropic:claude-sonnet-4-5")
    assert result["run"] is None and result["failed"] == ["t-ok"]
