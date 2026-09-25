"""The baseline step is skipped when no task passed the gate (no valid harbor target then)."""

from touchstone.survey import survey as survey_mod


def _patch_gate(monkeypatch, gate_result):
    monkeypatch.setattr(survey_mod, "run_gate", lambda *a, **k: gate_result)


def test_baseline_skipped_when_zero_tasks_gated(monkeypatch):
    _patch_gate(monkeypatch, {"gated": [], "needs_review": ["t1"]})
    called = []
    monkeypatch.setattr(survey_mod, "run_baseline", lambda *a, **k: called.append(1))
    gate, baseline = survey_mod._gate_and_baseline(
        repo=".", env_result={"services": []}, settings=object(),
        force=False, skip_gate=False, skip_baseline=False)
    assert baseline is None and gate["gated"] == [] and called == []


def test_baseline_runs_when_a_task_is_gated(monkeypatch):
    _patch_gate(monkeypatch, {"gated": ["t1"], "needs_review": []})
    monkeypatch.setattr(survey_mod, "run_baseline", lambda *a, **k: {"passed": ["t1"]})
    gate, baseline = survey_mod._gate_and_baseline(
        repo=".", env_result={"services": []}, settings=object(),
        force=False, skip_gate=False, skip_baseline=False)
    assert baseline == {"passed": ["t1"]} and gate["gated"] == ["t1"]
