from pathlib import Path

import pytest

from touchstone.config import Settings
from touchstone.harbor import run as run_mod


def _task(tmp_path, name="t1"):
    d = tmp_path / name
    d.mkdir()
    (d / "task.toml").write_text("")
    return d


def _dataset(tmp_path):
    root = tmp_path / "touchstone"
    (root / "tasks" / "t1").mkdir(parents=True)
    (root / "tasks" / "t1" / "task.toml").write_text("")
    return root


def test_run_path_resolves_task_dataset_and_taskdir(tmp_path):
    task = _task(tmp_path)
    assert run_mod._run_path(task) == task  # a single task
    ds = _dataset(tmp_path)
    assert run_mod._run_path(ds) == ds / "tasks"  # dataset root -> implicit tasks/


def _record_calls(monkeypatch, jobs_created="2026-01-01__00-00-00"):
    calls = []

    def fake_call(cmd):
        calls.append(cmd)
        # simulate harbor creating a job directory under the -o path
        if "run" in cmd and "-o" in cmd:
            jobs = Path(cmd[cmd.index("-o") + 1])
            (jobs / jobs_created).mkdir(parents=True, exist_ok=True)
        if cmd[0] == "rsync" and cmd[-1].endswith("jobs/"):  # remote job sync-back
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            (Path(cmd[-1]) / jobs_created).mkdir(exist_ok=True)
    monkeypatch.setattr(run_mod, "_call", fake_call)
    return calls


def test_run_local_builds_command_and_returns_job_dir(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    task = _task(tmp_path)
    job = run_mod.run(task, "oracle", jobs_dir=tmp_path / "jobs",
                      settings=Settings(harbor_host=""))
    cmd = calls[0]
    assert cmd[:2] == ["harbor", "run"]
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "oracle" and "-y" in cmd
    assert job.name == "2026-01-01__00-00-00"


def test_run_local_includes_model_when_given(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    run_mod.run(_task(tmp_path), "nop", model="openai/gpt-4o-mini",
                jobs_dir=tmp_path / "jobs", settings=Settings())
    cmd = calls[0]
    assert cmd[cmd.index("-m") + 1] == "openai/gpt-4o-mini"


def test_remote_when_no_docker_rsyncs_runs_over_ssh_and_syncs_back(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    ds = _dataset(tmp_path)
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    job = run_mod.run(ds, "oracle", jobs_dir=tmp_path / "jobs", settings=settings)

    kinds = [c[0] for c in calls]
    assert kinds == ["rsync", "ssh", "rsync"]  # push dataset, run, pull jobs
    push = calls[0]
    assert push[:4] == ["rsync", "-az", "--delete", "--exclude"] and push[4] == "jobs"
    dest = push[-1]  # mini:/remote/datasets/touchstone-<hex>/
    assert dest.startswith("mini:/remote/datasets/touchstone-") and dest.endswith("/")
    ssh = calls[1]
    assert ssh[0] == "ssh" and ssh[1] == "mini"
    assert "/remote/datasets/touchstone-" in ssh[2]
    assert "harbor run -p tasks -a oracle" in ssh[2]
    assert job.name == "2026-01-01__00-00-00"


def test_remote_custom_agent_also_ships_touchstone_and_uses_uvx(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    ds = _dataset(tmp_path)
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(ds, "touchstone.harbor.agent:TouchstoneAgent",
                jobs_dir=tmp_path / "jobs", settings=settings)
    # dataset push, touchstone repo push, ssh run, jobs pull
    assert [c[0] for c in calls] == ["rsync", "rsync", "ssh", "rsync"]
    assert calls[1][-1] == "mini:/remote/touchstone-src/"
    assert "uvx --from harbor --with /remote/touchstone-src harbor run" in calls[2][2]


def test_run_local_raises_if_no_job_created(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_call", lambda cmd: None)  # creates nothing
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    with pytest.raises(FileNotFoundError):
        run_mod.run(_task(tmp_path), "oracle", jobs_dir=tmp_path / "jobs", settings=Settings())
