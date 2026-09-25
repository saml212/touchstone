"""Regrade command construction, the new-job diff, and the reward deltas."""

from __future__ import annotations

import json
from pathlib import Path

from touchstone.config import Settings
from touchstone.harbor import jobs
from touchstone.harbor import run as run_mod
from touchstone.review import regrade


def _trial(job: Path, task: str, reward: float) -> None:
    d = job / f"{task}__x"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task_name": task}))
    (d / "verifier").mkdir()
    (d / "verifier" / "reward.json").write_text(json.dumps({"reward": reward}))


def _job(root: Path, name: str, rewards: dict) -> Path:
    job = root / name
    job.mkdir(parents=True)
    (job / "config.json").write_text("{}")
    for task, reward in rewards.items():
        _trial(job, task, reward)
    return job


def test_diff_rewards_reports_only_moves_largest_first(tmp_path):
    old = jobs.Job.read(_job(tmp_path, "old", {"a": 0.5, "b": 1.0, "c": 0.75}))
    new = jobs.Job.read(_job(tmp_path, "new", {"a": 1.0, "b": 1.0, "c": 0.0}))
    deltas = regrade.diff_rewards(old, new)
    tasks = [d.task for d in deltas]
    assert "b" not in tasks  # unchanged
    assert tasks[0] == "c"   # biggest move (0.75 -> 0.0) first
    assert {d.task: (d.before, d.after) for d in deltas} == {
        "c": (0.75, 0.0), "a": (0.5, 1.0)}


def test_regrade_job_uses_injected_runner_and_diffs(tmp_path):
    jobs_dir = tmp_path / "jobs"
    _job(jobs_dir, "src", {"refund": 0.75})
    new_dir = _job(jobs_dir, "regraded", {"refund": 1.0})

    def fake_runner(job_dir, tasks_path, *, settings):
        assert job_dir == jobs_dir / "src"
        assert Path(tasks_path).name == "tasks"
        return new_dir

    out = regrade.regrade_job(tmp_path, jobs_dir, "src", Settings(), runner=fake_runner)
    assert out == {"job": "regraded", "deltas": [{"task": "refund", "before": 0.75, "after": 1.0}],
                   "failed": []}


def test_run_regrade_local_builds_command_and_returns_new_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    src = _job(jobs_dir, "src", {"refund": 0.5})
    tasks = tmp_path / "tasks"
    (tasks / "refund").mkdir(parents=True)
    (tasks / "refund" / "task.toml").write_text("")
    seen = {}

    def fake_call(cmd, env=None, stdin_data=None):
        seen["cmd"] = cmd
        _job(jobs_dir, "src-regraded", {"refund": 1.0})  # regrade writes a new job dir

    monkeypatch.setattr(run_mod, "_call", fake_call)
    new_dir = run_mod.regrade(src, tasks, settings=Settings())
    assert new_dir.name == "src-regraded"
    cmd = seen["cmd"]
    assert cmd[:3] == ["harbor", "job", "regrade"]
    assert str(src.resolve()) in cmd
    assert "-p" in cmd and str(tasks.resolve()) in cmd


def test_refused_trials_are_reported_as_failed_not_zero(tmp_path):
    import json

    from touchstone.review import regrade

    def job(name, trials):
        d = tmp_path / name
        for task, body in trials:
            t = d / f"{task}__x"
            (t / "verifier").mkdir(parents=True)
            (t / "result.json").write_text(json.dumps({"task_name": task, **body}))
            if "reward" in body:
                (t / "verifier" / "reward.txt").write_text(str(body["reward"]))
        return d

    job("old", [("a", {"reward": 1.0}), ("b", {"reward": 1.0})])
    new = job("new", [("a", {"reward": 1.0}),
                      ("b", {"exception_info": {"exception_type": "RegradeError",
                                                "exception_message": "/app/output.json missing"}})])
    result = regrade.regrade_job(tmp_path, tmp_path, "old", settings=None,
                                 runner=lambda *a, **k: new)
    assert result["deltas"] == []
    assert result["failed"] == [{"task": "b", "error": "RegradeError: /app/output.json missing"}]
