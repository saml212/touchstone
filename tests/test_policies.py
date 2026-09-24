from touchstone import policies, tasks
from touchstone.checks import Check
from touchstone.policies import Policy


def _policy(name, kind, params, enabled=True, **kw):
    return Policy(check=Check(kind=kind, params=params, name=name, source="mined", **kw),
                  enabled=enabled)


def _task(**kw):
    defaults = dict(
        tags=["resolved"],
        context={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        reference={"content": "hello there",
                   "tool_calls": [{"name": "escalate", "arguments": "{}"}]},
    )
    defaults.update(kw)
    return tasks.Task(name="t", **defaults)


def test_read_write_roundtrip(tmp_path):
    pols = [_policy("calls escalate", "tool_called", {"name": "escalate"}, enabled=True),
            _policy("says hello", "contains", {"values": ["hello"]}, enabled=False,
                    severity="soft", because="common in good replies")]
    policies.write_policies(tmp_path, pols)
    got = policies.read_policies(tmp_path)
    assert [p.enabled for p in got] == [True, False]
    assert got[0].check.kind == "tool_called" and got[0].check.params == {"name": "escalate"}
    assert got[1].check.severity == "soft" and got[1].check.because == "common in good replies"


def test_read_missing_file_is_empty(tmp_path):
    assert policies.read_policies(tmp_path) == []


def test_materialize_applies_reference_gate(tmp_path):
    pols = [
        _policy("calls escalate", "tool_called", {"name": "escalate"}),  # ref calls it -> in
        _policy("calls refund", "tool_called", {"name": "refund"}),  # ref does not -> out
        _policy("no email", "no_pii", {"kinds": ["email"]}),  # safety -> always in
        _policy("disabled", "contains", {"values": ["hello"]}, enabled=False),  # off -> out
    ]
    checks = policies.materialize(_task(), pols)
    names = sorted(c.name for c in checks)
    assert names == ["calls escalate", "no email"]
    assert all(c.source == "policy" for c in checks)


def test_materialize_failure_task_gets_safety_only(tmp_path):
    pols = [
        _policy("calls escalate", "tool_called", {"name": "escalate"}),  # non-safety -> out
        _policy("no email", "no_pii", {"kinds": ["email"]}),  # safety -> in
    ]
    checks = policies.materialize(_task(tags=["failure"]), pols)
    assert [c.name for c in checks] == ["no email"]


def test_gate_check_safety_always_true():
    safety = Check(kind="not_contains", params={"values": ["x"]}, name="s")
    assert policies.gate_check(safety, {"content": "x present"}, [], failure=False) is True
