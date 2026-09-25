import json

from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()


def _trial(job_dir, name, task_name, reward, traj=None):
    d = job_dir / name
    (d / "verifier").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": task_name,
        "agent_info": {"name": "replica", "model_info": {"name": "openai/gpt-4o-mini"}},
    }))
    (d / "verifier" / "reward.txt").write_text(f"{reward}\n")
    if traj is not None:
        (d / "agent").mkdir()
        (d / "agent" / "trajectory.json").write_text(json.dumps(traj))


def _job(jobs_dir, name, config=None):
    d = jobs_dir / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or {}))
    return d


def test_train_end_to_end_writes_files_and_sentence(tmp_path):
    jobs = tmp_path / "touchstone" / "jobs"
    teacher = _job(jobs, "teacher",
                   {"agents": [{"name": "replica", "model_name": "openai/gpt-4o-mini"}]})
    _trial(teacher, "t1", "ds/refund", 1.0, traj={"schema_version": "ATIF-v1.8"})
    _trial(teacher, "t2", "ds/lookup", 1.0, traj={"schema_version": "ATIF-v1.8"})
    student = _job(jobs, "student", {"agents": [{"name": "replica"}]})
    _trial(student, "s1", "ds/refund", 0.0)  # distill
    _trial(student, "s2", "ds/lookup", 1.0)  # hold_out

    out = tmp_path / "train"
    result = runner.invoke(app, ["train", "--jobs-dir", str(jobs), "--teacher", str(teacher),
                                 "--student", str(student), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "distill: 2 trajectories from 2 tasks" in result.output
    assert "hold-out: 1" in result.output and str(out) in result.output
    assert (out / "distill.jsonl").exists() and (out / "rl_tasks.toml").exists()
    assert json.loads((out / "manifest.json").read_text())["counts"]["distill"] == 1


def test_train_defaults_student_to_newest_non_teacher(tmp_path):
    jobs = tmp_path / "touchstone" / "jobs"
    teacher = _job(jobs, "teacher")
    _trial(teacher, "t1", "ds/refund", 1.0, traj={"a": 1})
    student = _job(jobs, "student")
    _trial(student, "s1", "ds/refund", 0.0)
    import os
    os.utime(student, (student.stat().st_atime, teacher.stat().st_mtime + 10))

    out = tmp_path / "train"
    result = runner.invoke(app, ["train", "--jobs-dir", str(jobs),
                                 "--teacher", str(teacher), "--out", str(out)])
    assert result.exit_code == 0, result.output
    # student (newest, != teacher) had ds/refund fail; teacher passed -> distill route
    assert json.loads((out / "manifest.json").read_text())["counts"]["distill"] == 1


def test_train_fails_when_no_student_available(tmp_path):
    jobs = tmp_path / "touchstone" / "jobs"
    jobs.mkdir(parents=True)
    result = runner.invoke(app, ["train", "--jobs-dir", str(jobs), "--out", str(tmp_path / "o")])
    assert result.exit_code != 0
    assert "no student job dir" in result.output


def test_train_teacher_spec_runs_a_harbor_job(tmp_path, monkeypatch):
    import touchstone.harbor.run as run_mod

    jobs = tmp_path / "touchstone" / "jobs"
    teacher = _job(jobs, "teacher-run",
                   {"agents": [{"name": "replica", "model_name": "openai/gpt-4o-mini"}]})
    _trial(teacher, "t1", "ds/refund", 1.0, traj={"a": 1})
    student = _job(jobs, "student", {"agents": [{"name": "replica"}]})
    _trial(student, "s1", "ds/refund", 0.0)

    seen = {}

    def fake_run(path, agent, **kwargs):
        seen["path"], seen["agent"], seen["model"] = path, agent, kwargs.get("model")
        seen["extra_args"] = kwargs.get("extra_args")
        return teacher

    monkeypatch.setattr(run_mod, "run", fake_run)
    out = tmp_path / "train"
    result = runner.invoke(app, ["train", "--jobs-dir", str(jobs),
                                 "--teacher", "openai/gpt-4o-mini",
                                 "--student", str(student), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "touchstone.harbor.agent:TouchstoneAgent"
    assert seen["model"] == "openai/gpt-4o-mini"
    assert seen["extra_args"] == ["--ak", "mode=replica"]
    assert seen["path"] == jobs.parent  # dataset root = jobs_dir.parent
    assert json.loads((out / "manifest.json").read_text())["counts"]["distill"] == 1
