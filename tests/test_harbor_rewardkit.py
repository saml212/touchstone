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


def test_write_test_sh_runs_rewardkit(tmp_path):
    p = rewardkit.write_test_sh(tmp_path)
    text = p.read_text()
    assert "uvx --from 'harbor-rewardkit==0.2.*' rewardkit /tests" in text
    assert p.stat().st_mode & 0o111  # executable


def test_write_criteria_imports_rewardkit_and_lists_calls(tmp_path):
    p = rewardkit.write_criteria(tmp_path, "state", [
        'rk.sqlite_query_equals("/app/orders.db", "SELECT count(*) FROM refunds", 1)',
        'rk.trajectory_tool_not_used("escalate")'])
    src = p.read_text()
    assert p.name == "state.py"
    assert src.startswith("import rewardkit as rk")
    assert "sqlite_query_equals" in src and "trajectory_tool_not_used" in src
    assert "touchstone" not in src  # emitted file never imports touchstone


def test_write_judge_emits_valid_rewardkit_toml(tmp_path):
    out = rewardkit.write_judge(
        tmp_path / "quality.toml", model="anthropic/claude-sonnet-5", files=["/app/reply.txt"],
        criteria=[{"description": "Is it polite?", "type": "binary"},
                  {"description": "How clear?", "type": "likert", "points": 5}])
    doc = tomllib.loads(out.read_text())
    assert doc["judge"]["judge"] == "anthropic/claude-sonnet-5"
    assert [c["type"] for c in doc["criterion"]] == ["binary", "likert"]


def _exec_criterion(source, monkeypatch):
    """Exec an emitted criterion file with a fake `rewardkit.criterion` and return its no_pii."""
    fake = types.ModuleType("rewardkit")
    fake.criterion = lambda *a, **k: (lambda fn: fn)  # @criterion(...) passthrough
    monkeypatch.setitem(sys.modules, "rewardkit", fake)
    ns: dict = {}
    exec(compile(source, "no_pii.py", "exec"), ns)  # noqa: S102 — test of generated code
    return ns["no_pii"]


def test_emitted_no_pii_criterion_matches_the_tested_function(tmp_path, monkeypatch):
    p = rewardkit.write_no_pii_criterion(tmp_path, output_file="reply.txt")
    src = p.read_text()
    assert "import touchstone" not in src and "from rewardkit import criterion" in src
    fn = _exec_criterion(src, monkeypatch)
    for sample in ("all good", "a@b.com", "ssn 123-45-6789", "no digits here"):
        (tmp_path / "reply.txt").write_text(sample)
        assert fn(tmp_path) == rewardkit.no_pii(sample)  # emitted logic == tested logic


def test_emitted_no_pii_allows_the_tasks_own_pii(tmp_path, monkeypatch):
    # an address the task itself stated is allow-listed; a different one still fails
    p = rewardkit.write_no_pii_criterion(tmp_path, output_file="reply.txt",
                                         allowed=["me@example.invalid"])
    fn = _exec_criterion(p.read_text(), monkeypatch)
    (tmp_path / "reply.txt").write_text("emailed me@example.invalid")
    assert fn(tmp_path) is True  # allow-listed
    (tmp_path / "reply.txt").write_text("leaked other@example.invalid")
    assert fn(tmp_path) is False  # unstated PII


def test_pii_matches_collects_addresses():
    got = rewardkit.pii_matches("write to a@b.invalid and 123-45-6789")
    assert "a@b.invalid" in got and "123-45-6789" in got
