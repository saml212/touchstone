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


def _errored(job_dir, name, task_name, message):
    """A trial that raised inside Harbor: exception_info, no verifier reward (else a false 0)."""
    d = job_dir / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"task_name": task_name, "agent_info": {"name": "TouchstoneAgent"},
         "exception_info": {"exception_type": "RuntimeError", "exception_message": message}}))


def _job(jobs_dir, name):
    d = jobs_dir / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_jobs_reports_no_dirs_when_empty(tmp_path, seed_task):
    seed_task(tmp_path)
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(tmp_path / "jobs")])
    assert result.exit_code == 0 and "no job directories" in result.output


def test_jobs_lists_pass_rate_per_task(tmp_path, seed_task):
    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    j = _job(jobs, "2026-01-01__00-00-00")
    _trial(j, "t1__a", "ds/t1", 1.0)
    _trial(j, "t2__a", "ds/t2", 0.0)
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs)])
    assert result.exit_code == 0
    assert "ds/t1" in result.output and "100.0%" in result.output
    assert "ds/t2" in result.output and "0.0%" in result.output


def test_jobs_reports_agent_did_not_run_when_no_trajectory(tmp_path, seed_task):
    # The live tau-bench failure: the verifier wrote a 0 but the agent never ran (no trajectory).
    # `jobs` must say so, not print a false 0.0%.
    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    j = _job(jobs, "2026-01-01__00-00-00")
    d = j / "t1__a"
    (d / "verifier").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task_name": "ds/t1",
                                               "agent_info": {"name": "touchstone"}}))
    (d / "verifier" / "reward.txt").write_text("0.0\n")
    (d / "artifacts").mkdir()
    (d / "artifacts" / "manifest.json").write_text(json.dumps(
        [{"source": "/logs/agent/trajectory.json", "destination": "agent/trajectory.json",
          "type": "file", "status": "failed"}]))
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs)])
    assert result.exit_code == 0
    assert "agent did not run (no trajectory)" in result.output
    assert "0.0%" not in result.output


def test_bench_runs_and_prints_scoreboard(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "2026-01-01__00-00-00")
    _trial(job_dir, "t1__a", "ds/t1", 1.0)
    seen = {}
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: seen.update(k) or job_dir)

    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini", "--dataset", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert seen["jobs_dir"] == f"{tmp_path}/jobs"  # job dirs default under the dataset root
    assert "ds/t1" in result.output and "100.0%" in result.output and "overall" in result.output


def test_bench_screams_and_exits_1_when_every_trial_errored(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "2026-01-01__00-00-00")
    for i in range(1, 8):
        _errored(job_dir, f"t{i}__a", f"ds/t{i}", "docker compose build RC=1\nlast build line")
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: job_dir)
    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini", "--dataset", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "7 of 7 trials errored — first: docker compose build RC=1" in result.output
    assert "0.0%" not in result.output


def test_bench_shows_an_errors_column_when_some_trials_errored(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "2026-01-01__00-00-00")
    _trial(job_dir, "t1a__a", "ds/t1", 1.0)
    _errored(job_dir, "t1b__a", "ds/t1", "RuntimeError: boom")
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: job_dir)
    result = runner.invoke(app, ["bench", "-m", "m", "--dataset", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "errors" in result.output and "ds/t1" in result.output


def test_bench_against_prints_comparison(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    new = _job(jobs, "new")
    _trial(new, "t1__a", "ds/t1", 1.0)
    _trial(new, "t2__a", "ds/t2", 1.0)
    old = _job(jobs, "old")
    _trial(old, "t1__a", "ds/t1", 1.0)
    _trial(old, "t2__a", "ds/t2", 0.0)
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: new)

    result = runner.invoke(app, ["bench", "-m", "m", "--against", str(old), "--dataset",
                                 str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "vs --against:" in result.output
    assert "pass in both: 1" in result.output and "only this run: 1" in result.output


def test_bench_prints_one_line_when_tasks_dir_is_empty(tmp_path, monkeypatch):
    # The survey parked everything; bench must say so, never Harbor's "Either datasets or tasks
    # must be provided" traceback.
    import touchstone.harbor.run as run_mod

    ds = tmp_path / "touchstone"
    (ds / "tasks").mkdir(parents=True)  # present but empty
    review = ds / "needs-review" / "issue-a-refund-1"
    review.mkdir(parents=True)
    (review / "gate.json").write_text(json.dumps({"failed_side": "oracle", "oracle": 0.5}))

    def _boom(*_a, **_k):
        raise AssertionError("Harbor must not be invoked when there are no tasks")

    monkeypatch.setattr(run_mod, "run", _boom)
    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini", "--dataset", str(ds)])
    assert result.exit_code == 1
    assert "No tasks in" in result.output and "parked 1 in" in result.output
    assert "touchstone survey --force" in result.output
    assert "Traceback" not in result.output


def test_bench_rejects_unknown_agent(tmp_path):
    result = runner.invoke(app, ["bench", "-m", "m", "--agent", "wizard"])
    assert result.exit_code != 0
    assert "packaged" in result.output or "replica" in result.output


def test_bench_passes_mode_to_the_agent_kwargs(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    seed_task(tmp_path)
    jobs = tmp_path / "jobs"
    job_dir = _job(jobs, "j")
    _trial(job_dir, "t1__a", "ds/t1", 1.0)
    seen = {}

    def fake_run(path, agent, **kwargs):
        seen["agent"] = agent
        seen["extra_args"] = kwargs.get("extra_args")
        return job_dir

    monkeypatch.setattr(run_mod, "run", fake_run)
    result = runner.invoke(app, ["bench", "-m", "m", "--agent", "packaged", "--dataset",
                                 str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "touchstone.harbor.agent:TouchstoneAgent"
    assert seen["extra_args"] == ["--ak", "mode=packaged"]


def test_bench_adds_simulated_user_flags_for_a_multi_turn_dataset(tmp_path, monkeypatch):
    import touchstone.harbor.run as run_mod

    ds = tmp_path / "touchstone"
    (ds / "tasks" / "chat").mkdir(parents=True)
    (ds / "tasks" / "chat" / "task.toml").write_text(
        "[metadata.touchstone]\nturns = 3\nmulti_turn = true\n")
    seen = {}
    monkeypatch.setattr(run_mod, "run",
                        lambda *a, **k: seen.update(k) or _job(tmp_path / "jobs", "j"))
    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini", "--dataset", str(ds)])
    assert result.exit_code == 0, result.output
    extra = seen["extra_args"]
    assert "--bridge" in extra and "acp" in extra and "--user-agent" in extra
    assert "openai/gpt-4o-mini" in extra  # user model defaults to the agent-under-test's model


def test_bench_prints_one_line_when_docker_daemon_is_down(tmp_path, monkeypatch, seed_task):
    # The stranger's blocker: bench dumped a raw traceback. It must print one sentence + exit 1.
    import touchstone.harbor.run as run_mod

    def _no_daemon(*_a, **_k):
        raise run_mod.DockerDaemonError(
            "Docker daemon not running on local. Start Docker (colima start / Docker Desktop) "
            "or set [harbor] host in touchstone.toml.")

    monkeypatch.delenv("TOUCHSTONE_DEBUG", raising=False)
    monkeypatch.setattr(run_mod, "run", _no_daemon)
    monkeypatch.chdir(tmp_path)
    seed_task(tmp_path / "touchstone")
    result = runner.invoke(app, ["bench", "-m", "openai/gpt-4o-mini"])
    assert result.exit_code == 1
    assert "Docker daemon not running on local" in result.output
    assert "Traceback" not in result.output and "run.py" not in result.output


def test_bench_debug_flag_lets_the_traceback_through(tmp_path, monkeypatch, seed_task):
    import touchstone.harbor.run as run_mod

    def _no_daemon(*_a, **_k):
        raise run_mod.DockerDaemonError("Docker daemon not running on local. ...")

    monkeypatch.delenv("TOUCHSTONE_DEBUG", raising=False)
    monkeypatch.setattr(run_mod, "run", _no_daemon)
    monkeypatch.chdir(tmp_path)
    seed_task(tmp_path / "touchstone")
    result = runner.invoke(app, ["--debug", "bench", "-m", "openai/gpt-4o-mini"])
    assert result.exit_code != 0
    assert isinstance(result.exception, run_mod.DockerDaemonError)
