import sys
import tomllib
import types

from touchstone.harbor import rewardkit


def test_no_pii_flags_email_ssn_card_and_phone():
    assert rewardkit.no_pii("all good, order shipped") is True
    assert rewardkit.no_pii("") is True
    assert rewardkit.no_pii("reach me at a@b.com") is False
    assert rewardkit.no_pii("ssn 123-45-6789") is False
    assert rewardkit.no_pii("card 4111 1111 1111 1111") is False
    assert rewardkit.no_pii("call +1 415-555-1234") is False


def test_write_judge_emits_valid_rewardkit_toml(tmp_path):
    out = rewardkit.write_judge(
        tmp_path / "quality.toml", model="anthropic/claude-sonnet-5", files=["/app/reply.txt"],
        criteria=[{"description": "Is it polite?", "type": "binary"},
                  {"description": "How clear?", "type": "likert", "points": 5}])
    doc = tomllib.loads(out.read_text())
    assert doc["judge"]["judge"] == "anthropic/claude-sonnet-5"
    assert doc["judge"]["files"] == ["/app/reply.txt"]
    assert [c["type"] for c in doc["criterion"]] == ["binary", "likert"]


def _fake_rewardkit(monkeypatch):
    calls = []
    fake = types.ModuleType("rewardkit")
    fake.sqlite_query_equals = lambda *a, **k: calls.append(("sqlite", a, k))
    fake.trajectory_tool_used = lambda *a, **k: calls.append(("used", a, k))
    fake.trajectory_tool_not_used = lambda *a, **k: calls.append(("not_used", a, k))
    monkeypatch.setitem(sys.modules, "rewardkit", fake)
    return calls


def test_wrappers_forward_to_rewardkit(monkeypatch):
    calls = _fake_rewardkit(monkeypatch)
    rewardkit.sqlite_state("/app/orders.db", "SELECT count(*) FROM refunds", 1)
    rewardkit.tool_used("order_status", min_count=2)
    rewardkit.tool_not_used("escalate", path="/logs/agent/trajectory.json")
    assert calls[0] == ("sqlite", ("/app/orders.db", "SELECT count(*) FROM refunds", 1),
                        {"weight": 1.0})
    assert calls[1] == ("used", ("order_status",), {"min_count": 2, "weight": 1.0})
    assert calls[2] == ("not_used", ("escalate",),
                        {"weight": 1.0, "path": "/logs/agent/trajectory.json"})
