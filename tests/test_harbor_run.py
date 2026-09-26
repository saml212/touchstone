from pathlib import Path

import pytest

from touchstone.config import Settings
from touchstone.harbor import remote
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


def test_dataset_is_multi_turn_reads_task_metadata(tmp_path):
    ds = tmp_path / "touchstone"
    (ds / "tasks" / "single").mkdir(parents=True)
    (ds / "tasks" / "single" / "task.toml").write_text(
        "[metadata.touchstone]\nmulti_turn = false\n")
    assert run_mod.dataset_is_multi_turn(ds) is False
    (ds / "tasks" / "chat").mkdir(parents=True)
    (ds / "tasks" / "chat" / "task.toml").write_text(
        "[metadata.touchstone]\nturns = 3\nmulti_turn = true\n")
    assert run_mod.dataset_is_multi_turn(ds) is True  # any multi-turn task flips the dataset
    assert run_mod.dataset_is_multi_turn(ds / "tasks" / "chat") is True  # a single task dir too
    assert run_mod.dataset_is_multi_turn(ds / "tasks" / "single") is False


def test_simulated_user_args_builds_the_bridge_flags():
    args = run_mod.simulated_user_args("claude-code", "openai/gpt-4o-mini")
    assert args == ["--user-agent", "claude-code", "--user-model", "openai/gpt-4o-mini",
                    "--bridge", "acp"]
    with_persona = run_mod.simulated_user_args("claude-code", "m", "tasks/t/persona.md")
    assert with_persona[-2:] == ["--user-persona-path", "tasks/t/persona.md"]


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
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
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
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
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
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(_dataset(tmp_path), AGENT, model="openai/gpt-4o-mini",
                jobs_dir=tmp_path / "jobs", settings=settings)
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh.stdin == "sk-secret\n"  # fed on stdin
    assert 'IFS= read -r _tsk; export OPENAI_API_KEY="$_tsk";' in ssh[-1]
    assert "sk-secret" not in " ".join(ssh)  # never on argv


def test_remote_forwards_both_agent_and_user_keys_on_stdin(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(_dataset(tmp_path), AGENT, model="openai/gpt-4.1-mini",
                user_model="anthropic/claude-haiku-4-5", jobs_dir=tmp_path / "jobs",
                settings=settings)
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh.stdin == "sk-openai\nsk-anthropic\n"  # one line per key, agent then user
    assert 'export OPENAI_API_KEY="$_tsk";' in ssh[-1]
    assert 'export ANTHROPIC_API_KEY="$_tsk";' in ssh[-1]
    assert "sk-openai" not in " ".join(ssh) and "sk-anthropic" not in " ".join(ssh)


def test_same_provider_agent_and_user_forward_one_key(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings = Settings(harbor_host="mini", harbor_remote_root="/remote")
    run_mod.run(_dataset(tmp_path), AGENT, model="openai/gpt-4.1-mini",
                user_model="openai/gpt-4.1-mini", jobs_dir=tmp_path / "jobs", settings=settings)
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh.stdin == "sk-openai\n"  # deduped to a single OPENAI_API_KEY line


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
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
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


def test_missing_rsync_or_ssh_binary_gives_a_clear_error(monkeypatch):
    # Attack (stage-7 harbor): rsync (or ssh) is not installed. A raw FileNotFoundError
    # ("[Errno 2] No such file or directory: 'rsync'") is opaque; the caller must get a clear,
    # actionable RuntimeError naming the tool, and (via _call) the gate records it in needs-review.
    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "rsync")

    monkeypatch.setattr(run_mod.subprocess, "run", missing)
    with pytest.raises(RuntimeError) as ei:
        run_mod._call(["rsync", "-az", "a", "b"])
    msg = str(ei.value)
    assert "rsync" in msg and "not installed" in msg and "PATH" in msg


# ---- shipping touchstone to the remote host from either layout (checkout vs wheel) --------------


def test_generated_pyproject_is_valid_toml_and_lists_the_package():
    import tomllib
    doc = tomllib.loads(remote.generated_pyproject())
    assert doc["project"]["name"] == "touchstone-bench"
    assert doc["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["touchstone"]
    assert isinstance(doc["project"]["dependencies"], list)
    assert doc["build-system"]["build-backend"] == "hatchling.build"


def test_stage_src_generates_pyproject_for_a_wheel_layout(tmp_path, monkeypatch):
    # A wheel install has the package under site-packages with no pyproject: _stage_src copies just
    # the package and writes a minimal pyproject so `uvx --with <dir>` resolves.
    import tomllib
    fake_site = tmp_path / "site-packages"
    (fake_site / "touchstone").mkdir(parents=True)
    (fake_site / "touchstone" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(remote, "_src_paths",
                        lambda: (fake_site, fake_site / "touchstone"))
    stage = tmp_path / "stage"
    stage.mkdir()
    out = remote._stage_src(stage)
    assert out == stage
    assert (stage / "touchstone" / "__init__.py").is_file()  # package copied
    doc = tomllib.loads((stage / "pyproject.toml").read_text())
    assert doc["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["touchstone"]


def test_stage_src_uses_the_checkout_root_when_it_has_a_pyproject(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "touchstone").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.setattr(remote, "_src_paths", lambda: (root, root / "touchstone"))
    assert remote._stage_src(tmp_path / "unused") == root  # synced as-is, no staging


def test_simulated_user_agent_follows_the_model_provider():
    from touchstone.harbor import run as run_mod

    assert run_mod.simulated_user_args("", "openai/gpt-4o-mini")[:2] == ["--user-agent", "codex"]
    assert run_mod.simulated_user_args("", "anthropic/claude-sonnet-4-5")[1] == "claude-code"
    assert run_mod.simulated_user_args("aider", "openai/x")[1] == "aider"


def test_require_target_local_when_local_docker_up(monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    assert run_mod._require_target(Settings(harbor_host="mini")) == "local"


def test_require_target_remote_when_only_host_docker_up(monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27.0"))
    assert run_mod._require_target(Settings(harbor_host="mini")) == "remote"


def test_require_target_raises_one_sentence_when_no_daemon(monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    with pytest.raises(run_mod.DockerDaemonError) as exc:
        run_mod._require_target(Settings())  # no local docker, no host
    msg = str(exc.value)
    assert msg == ("Docker daemon not running on local. Start Docker (colima start / Docker "
                   "Desktop) or set [harbor] host in touchstone.toml.")
    assert "\n" not in msg  # one line, no traceback


def test_require_target_names_the_host_when_its_daemon_is_down(monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (False, ""))
    with pytest.raises(run_mod.DockerDaemonError) as exc:
        run_mod._require_target(Settings(harbor_host="mini"))
    assert "on host mini" in str(exc.value)


def test_build_image_retries_once_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    calls = []

    def fake_exec(cmd, env=None, stdin_data=None):
        calls.append(cmd)
        return (0, "") if len(calls) == 2 else (1, "boom")

    monkeypatch.setattr(run_mod, "_exec", fake_exec)
    run_mod.build_image(tmp_path, "img:tag", Settings())
    assert len(calls) == 2  # failed once (transient), retried, then succeeded


def test_build_image_surfaces_last_8_lines_after_two_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    output = "\n".join(f"line{i}" for i in range(20))
    monkeypatch.setattr(run_mod, "_exec", lambda *a, **k: (1, output))
    with pytest.raises(RuntimeError) as exc:
        run_mod.build_image(tmp_path, "img:tag", Settings())
    msg = str(exc.value)
    assert "docker build failed on local" in msg
    assert "line19" in msg and "line12" in msg and "line11" not in msg  # last 8 lines only
