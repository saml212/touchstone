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


class _Cmd(list):
    """The command list, with the env and stdin it was called with kept as attributes."""

    def __init__(self, cmd, env=None, stdin=None):
        super().__init__(cmd)
        self.env = env
        self.stdin = stdin


def _record_calls(monkeypatch, jobs_created="2026-01-01__00-00-00"):
    calls = []

    def fake_call(cmd, env=None, stdin_data=None):
        calls.append(_Cmd(cmd, env, stdin_data))
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
    assert push[0] == "rsync" and "--delete" in push and "jobs" in push
    dest = push[-1]  # mini:/remote/datasets/touchstone-<hex>/
    assert dest.startswith("mini:/remote/datasets/touchstone-") and dest.endswith("/")
    ssh = calls[1]
    assert ssh[0] == "ssh" and ssh[-2] == "mini"  # host is the penultimate arg, remote is last
    assert "/remote/datasets/touchstone-" in ssh[-1]
    assert "harbor run -p tasks -a oracle" in ssh[-1]
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
    assert "uvx --from harbor --with /remote/touchstone-src harbor run" in calls[2][-1]


AGENT = "touchstone.harbor.agent:TouchstoneAgent"


def test_local_custom_agent_forwards_key_via_env_not_argv(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    run_mod.run(_task(tmp_path), AGENT, model="openai/gpt-4o-mini",
                jobs_dir=tmp_path / "jobs", settings=Settings())
    cmd = calls[0]
    assert cmd.env["OPENAI_API_KEY"] == "sk-secret"  # in the child env
    assert "sk-secret" not in " ".join(cmd)  # never on argv


def test_remote_custom_agent_forwards_key_on_stdin_not_argv(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(_dataset(tmp_path), AGENT, model="openai/gpt-4o-mini",
                jobs_dir=tmp_path / "jobs", settings=settings)
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh.stdin == "sk-secret\n"  # fed on stdin
    assert 'read -r TS_KEY; export OPENAI_API_KEY="$TS_KEY";' in ssh[-1]
    assert "sk-secret" not in " ".join(ssh)  # never on argv


def test_builtin_agent_forwards_no_key(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    run_mod.run(_task(tmp_path), "oracle", model="openai/gpt-4o-mini",
                jobs_dir=tmp_path / "jobs", settings=Settings())
    assert calls[0].env is None  # oracle/nop need no key


def test_provider_key_none_when_no_key_needed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    assert run_mod._provider_key("claude-cli/sonnet", Settings()) is None  # subscription, no key
    assert run_mod._provider_key(None, Settings()) is None


def test_remote_ssh_and_rsync_bound_connect_timeout_so_unreachable_host_never_hangs(
        tmp_path, monkeypatch):
    # Attack (stage-7 harbor): the mini is unreachable. Every ssh/rsync to the host must carry a
    # bounded ConnectTimeout and BatchMode=yes so the command fails fast instead of hanging on the
    # TCP connect or a password/passphrase prompt.
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(_dataset(tmp_path), "oracle", jobs_dir=tmp_path / "jobs", settings=settings)
    for cmd in calls:
        joined = " ".join(cmd)
        assert "ConnectTimeout=10" in joined
        assert "BatchMode=yes" in joined
        if cmd[0] == "rsync":  # rsync tunnels ssh via -e, carrying the same options
            assert cmd[1] == "-e" and "ssh -o ConnectTimeout=10" in cmd[2]


def test_run_local_raises_if_no_job_created(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_call", lambda cmd, **kw: None)  # creates nothing
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    with pytest.raises(FileNotFoundError):
        run_mod.run(_task(tmp_path), "oracle", jobs_dir=tmp_path / "jobs", settings=Settings())
