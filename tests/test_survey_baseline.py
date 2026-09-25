"""Baseline: run the agent under test over the gated tasks and summarize what it passes.

Harbor is mocked (jobs.pass_rates / run_mod.run) so this stays offline; test_survey_orchestrator and
the live verify cover the real run.
"""

import json

from touchstone.config import Settings
from touchstone.survey import baseline


def _agent_toml(out, mode="packaged", model="gpt-4o-mini", provider="openai"):
    (out / "agent").mkdir(parents=True, exist_ok=True)
    (out / "agent" / "agent.toml").write_text(
        f'[agent]\nmode = "{mode}"\nmodel_default = "{model}"\nprovider = "{provider}"\n')


def _mock_harbor(monkeypatch, rates, capture=None, rewards=None):
    def fake_run(path, agent, **kw):
        if capture is not None:
            capture.update(agent=agent, model=kw.get("model"), extra=kw.get("extra_args"),
                           jobs_dir=kw.get("jobs_dir"))
        return path / "jobs" / "j1"
    monkeypatch.setattr(baseline.run_mod, "run", fake_run)
    monkeypatch.setattr(baseline.jobs.Job, "read", staticmethod(lambda d: d))
    monkeypatch.setattr(baseline.jobs, "pass_rates", lambda job: rates)
    monkeypatch.setattr(baseline.jobs, "mean_rewards", lambda job: rewards or rates)


def test_run_baseline_writes_summary_and_passes_model_and_mode(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _agent_toml(out)
    seen = {}
    _mock_harbor(monkeypatch, {"t1": 1.0, "t2": 0.0, "t3": 1.0}, seen,
                 rewards={"t1": 1.0, "t2": 0.75, "t3": 1.0})
    data = baseline.run_baseline(tmp_path, {}, Settings())

    assert seen["agent"] == baseline.AGENT_PATH
    assert seen["model"] == "openai/gpt-4o-mini"  # provider/model
    assert seen["extra"] == ["--ak", "mode=packaged"]
    assert seen["jobs_dir"] == out / "jobs"  # under the dataset root, not cwd
    assert data["passed"] == ["t1", "t3"] and data["failed"] == ["t2"]
    assert data["rewards"] == {"t1": 1.0, "t2": 0.75, "t3": 1.0}  # partial credit kept
    assert data["model"] == "openai/gpt-4o-mini" and data["mode"] == "packaged"
    on_disk = json.loads((out / "baseline.json").read_text())
    assert on_disk["passed"] == ["t1", "t3"]


def test_run_baseline_idempotent(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _agent_toml(out)
    _mock_harbor(monkeypatch, {"t1": 1.0})
    baseline.run_baseline(tmp_path, {}, Settings())

    def boom(*a, **k):
        raise AssertionError("should not re-run when baseline.json exists")
    monkeypatch.setattr(baseline.run_mod, "run", boom)
    again = baseline.run_baseline(tmp_path, {}, Settings())
    assert again["passed"] == ["t1"]


def _task_dir(out, name):
    d = out / "tasks" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text("")


def test_run_baseline_reruns_when_a_new_task_is_absent_from_the_baseline(tmp_path, monkeypatch):
    # An earlier baseline ran t1; a later survey pass wrote t2. The stale baseline no longer covers
    # every current task, so it is NOT reused — the baseline re-runs over the full current set.
    out = tmp_path / "touchstone"
    _agent_toml(out)
    _task_dir(out, "t1")
    _mock_harbor(monkeypatch, {"t1": 1.0})
    baseline.run_baseline(tmp_path, {}, Settings())  # first baseline covers {t1}

    _task_dir(out, "t2")  # a new task appears after the baseline
    _mock_harbor(monkeypatch, {"t1": 1.0, "t2": 0.0})
    again = baseline.run_baseline(tmp_path, {}, Settings())
    assert set(again["pass_rates"]) == {"t1", "t2"} and again["failed"] == ["t2"]  # re-ran, merged


def test_run_baseline_reused_when_it_already_covers_every_task(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _agent_toml(out)
    _task_dir(out, "t1")
    _mock_harbor(monkeypatch, {"t1": 1.0})
    baseline.run_baseline(tmp_path, {}, Settings())

    def boom(*a, **k):
        raise AssertionError("should not re-run when the baseline covers every current task")
    monkeypatch.setattr(baseline.run_mod, "run", boom)
    again = baseline.run_baseline(tmp_path, {}, Settings())
    assert again["passed"] == ["t1"]


def test_run_baseline_skip_and_missing_agent(tmp_path, monkeypatch):
    assert baseline.run_baseline(tmp_path, {}, Settings(), skip=True) is None  # skip flag
    assert baseline.run_baseline(tmp_path, {}, Settings()) is None  # no agent.toml


def test_run_baseline_harbor_failure_is_captured(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _agent_toml(out, mode="replica")

    def boom(*a, **k):
        raise RuntimeError("harbor blew up")
    monkeypatch.setattr(baseline.run_mod, "run", boom)
    data = baseline.run_baseline(tmp_path, {}, Settings())
    assert "harbor blew up" in data["error"] and data["mode"] == "replica"
    assert not (out / "baseline.json").exists()


def test_model_ref_forms():
    assert baseline._model_ref({"model_default": "gpt-4o-mini", "provider": "openai"}) == \
        "openai/gpt-4o-mini"
    assert baseline._model_ref({"model_default": "openai/x", "provider": "openai"}) == "openai/x"
    assert baseline._model_ref({"model_default": "m", "provider": ""}) == "m"
