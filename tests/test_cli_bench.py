import json

from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()


def _trial(job_dir, name, task_name, reward):
    d = job_dir / name
    (d / "verifier").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task_name": task_name,
                                               "agent_info": {"name": "oracle"}}))
    (d / "verifier" / "reward.txt").write_text(f"{reward}\n")


def _job(jobs_dir, name):
    d = jobs_dir / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_jobs_reports_no_dirs_when_empty(tmp_path):
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(tmp_path / "jobs")])
    assert result.exit_code == 0 and "no job directories" in result.output


def test_jobs_lists_pass_rate_per_task(tmp_path):
    jobs = tmp_path / "jobs"
    j = _job(jobs, "2026-01-01__00-00-00")
    _trial(j, "t1__a", "ds/t1", 1.0)
    _trial(j, "t2__a", "ds/t2", 0.0)
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs)])
    assert result.exit_code == 0
    assert "ds/t1" in result.output and "100.0%" in result.output
    assert "ds/t2" in result.output and "0.0%" in result.output


def test_bench_runs_and_prints_scoreboard(tmp_path, monkeypatch):
    import touchstone.harbor.run as run_mod

    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "2026-01-01__00-00-00")
    _trial(job_dir, "t1__a", "ds/t1", 1.0)
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: job_dir)

    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini", "--dataset", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "ds/t1" in result.output and "100.0%" in result.output and "overall" in result.output


def test_bench_against_prints_comparison(tmp_path, monkeypatch):
    import touchstone.harbor.run as run_mod

    jobs = tmp_path / "jobs"
    new = _job(jobs, "new")
    _trial(new, "t1__a", "ds/t1", 1.0)
    _trial(new, "t2__a", "ds/t2", 1.0)
    old = _job(jobs, "old")
    _trial(old, "t1__a", "ds/t1", 1.0)
    _trial(old, "t2__a", "ds/t2", 0.0)
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: new)

    result = runner.invoke(app, ["bench", "-m", "m", "--against", str(old)])
    assert result.exit_code == 0, result.output
    assert "vs --against:" in result.output
    assert "pass in both: 1" in result.output and "only this run: 1" in result.output


def test_bench_rejects_unknown_agent(tmp_path):
    result = runner.invoke(app, ["bench", "-m", "m", "--agent", "wizard"])
    assert result.exit_code != 0
    assert "packaged" in result.output or "replica" in result.output


def test_bench_passes_mode_to_the_agent_kwargs(tmp_path, monkeypatch):
    import touchstone.harbor.run as run_mod

    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "j")
    _trial(job_dir, "t1__a", "ds/t1", 1.0)
    seen = {}

    def fake_run(path, agent, **kwargs):
        seen["agent"] = agent
        seen["extra_args"] = kwargs.get("extra_args")
        return job_dir

    monkeypatch.setattr(run_mod, "run", fake_run)
    result = runner.invoke(app, ["bench", "-m", "m", "--agent", "packaged"])
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "touchstone.harbor.agent:TouchstoneAgent"
    assert seen["extra_args"] == ["--ak", "mode=packaged"]
