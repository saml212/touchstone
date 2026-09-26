from typer.testing import CliRunner

from touchstone import store
from touchstone.cli import app

runner = CliRunner()


def test_init_writes_config_and_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / "touchstone.toml").exists()
    assert (tmp_path / ".touchstone" / "touchstone.db").exists()


def test_init_omits_keychain_prefix_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    text = (tmp_path / "touchstone.toml").read_text()
    assert not any(line.startswith("keychain_prefix") for line in text.splitlines())


def test_init_writes_keychain_prefix_when_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--keychain-prefix", "rockie-"])
    text = (tmp_path / "touchstone.toml").read_text()
    assert 'keychain_prefix = "rockie-"' in text


def test_bare_command_prints_the_loop_and_next_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Touchstone — the loop:" in result.output
    # Numbered loop lines, then the project's next step.
    for i in range(1, 4):
        assert f"  {i}. " in result.output
    assert "Next: Capture traces first" in result.output  # empty project -> capture
    assert not (tmp_path / ".touchstone").exists()  # a bare run creates nothing


def test_doctor_reports_one_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for row in ("component", "python", "database", "provider: scripted", "provider: reference",
                "provider: claude-cli", "speech: mode", "speech: stt", "tool: harbor",
                "tool: docker", "tool: ffmpeg", "harbor host"):
        assert row in result.output, row
    assert "using local docker" in result.output  # no harbor host configured by default


def test_doctor_reports_harbor_host_reachability(monkeypatch):
    # Attack (stage-7 CLI): doctor must report whether the configured remote harbor host is
    # reachable, not just whether a local `harbor` binary is on PATH. The probe is a bounded ssh.
    import subprocess

    import touchstone.cli as cli_mod
    from touchstone.config import Settings

    seen = {}

    class _Proc:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    row = cli_mod._harbor_host_row(Settings(harbor_host="mini"))
    assert row == ("harbor host", "ok", "mini reachable")
    assert seen["cmd"][0] == "ssh" and seen["cmd"][-1] == "true"
    assert "ConnectTimeout=10" in seen["cmd"]  # the probe is connect-timeout bounded

    def fail_run(cmd, **kw):
        p = _Proc()
        p.returncode = 255
        p.stderr = "ssh: connect to host mini port 22: Connection refused\n"
        return p

    monkeypatch.setattr(subprocess, "run", fail_run)
    comp, status, detail = cli_mod._harbor_host_row(Settings(harbor_host="mini"))
    assert status == "unreachable" and "mini" in detail


def test_doctor_docker_row_probes_the_daemon_not_the_binary(monkeypatch):
    # The stranger's false-green: doctor said docker "ok" on binary presence while the daemon was
    # down. The row must reflect a real `docker info` probe.
    import touchstone.cli as cli_mod
    import touchstone.harbor.run as run_mod

    monkeypatch.setattr(run_mod, "docker_daemon", lambda: (True, "27.1.1"))
    assert cli_mod._docker_row() == ("tool: docker", "ok", "27.1.1")

    monkeypatch.setattr(run_mod, "docker_daemon", lambda: (False, ""))
    comp, status, detail = cli_mod._docker_row()
    assert comp == "tool: docker" and status == "missing"
    assert "not running" in detail and "colima start" in detail


def test_doctor_probes_harbor_host_docker_over_ssh(monkeypatch):
    import touchstone.cli as cli_mod
    import touchstone.harbor.run as run_mod
    from touchstone.config import Settings

    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (False, ""))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    # No host configured by default -> no host-docker row.
    assert "harbor host docker" not in result.output

    comp, status, detail = cli_mod._harbor_host_docker_row(Settings(harbor_host="mini"))
    assert comp == "harbor host docker" and status == "missing" and "mini" in detail


def test_version_flag_and_command_print_the_package_version(monkeypatch):
    import touchstone.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_package_version", lambda: "9.9.9")
    flag = runner.invoke(app, ["--version"])
    assert flag.exit_code == 0 and "9.9.9" in flag.output
    cmd = runner.invoke(app, ["version"])
    assert cmd.exit_code == 0 and "9.9.9" in cmd.output


def _write_lock(tmp_path, name, version, source='source = { git = "ssh://x/y" }'):
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n{source}\n', encoding="utf-8")


def test_version_skew_warns_when_project_pins_a_different_touchstone(tmp_path, monkeypatch):
    import touchstone.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_package_version", lambda: "0.1.2")
    _write_lock(tmp_path, "touchstone", "0.1.0")  # the stranger's exact skew (old dist name)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert ("Note: this project pins touchstone-bench 0.1.0; you are running 0.1.2 "
            "(uv sync --upgrade-package touchstone-bench).") in result.output


def test_version_skew_silent_when_versions_match_or_no_lock(tmp_path, monkeypatch):
    import touchstone.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_package_version", lambda: "0.1.2")
    assert "Note: this project pins" not in runner.invoke(app, []).output  # no uv.lock
    _write_lock(tmp_path, "touchstone-bench", "0.1.2")
    assert "Note: this project pins" not in runner.invoke(app, []).output  # same version


def test_version_skew_silent_in_the_touchstone_checkout_itself(tmp_path, monkeypatch):
    import touchstone.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_package_version", lambda: "0.1.3")
    # the checkout's own package entry is a virtual/editable root, not a dependency pin
    _write_lock(tmp_path, "touchstone-bench", "0.1.0", source='source = { virtual = "." }')
    assert "Note: this project pins" not in runner.invoke(app, []).output


def test_doctor_exits_zero_with_all_tools_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    import touchstone.cli as cli_mod

    monkeypatch.setattr(shutil, "which", lambda _name: None)  # no harbor/docker/ffmpeg on PATH
    monkeypatch.setattr(cli_mod, "_installed", lambda _name: False)  # no optional SDKs/backends
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "tool: harbor" in result.output and "missing" in result.output


def test_demo_generates_episodes(tmp_path):
    db = str(tmp_path / "d.db")
    result = runner.invoke(app, ["demo", "--n", "30", "--db", db])
    assert result.exit_code == 0, result.output
    conn = store.connect(db)
    try:
        eps = store.list_episodes(conn)
        resolved = store.list_episodes(conn, "resolved")
        assert len(eps) == 30
        assert 15 <= len(resolved) <= 28  # ~70% resolved
    finally:
        conn.close()


def test_demo_is_idempotent_second_run_adds_more_without_id_collision(tmp_path):
    db = str(tmp_path / "d.db")
    runner.invoke(app, ["demo", "--n", "30", "--db", db])
    runner.invoke(app, ["demo", "--n", "30", "--db", db])
    conn = store.connect(db)
    try:
        eps = store.list_episodes(conn)
        ids = [e.id for e in eps]
        assert len(eps) == 60
        assert len(set(ids)) == 60  # no id collisions across runs
    finally:
        conn.close()


def test_readonly_db_dir_fails_cleanly(tmp_path, monkeypatch):
    import os
    import stat

    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("TOUCHSTONE_DB", str(ro / "sub" / "t.db"))
    try:
        init = runner.invoke(app, ["init"], catch_exceptions=False)
        assert init.exit_code == 1  # cannot open the db under a read-only dir
    finally:
        os.chmod(ro, stat.S_IRWXU)


def _stub_review_server(monkeypatch, *, up):
    """Stub review's server plumbing so tests never bind a port or block on Ctrl-C.
    Returns (calls, opened): calls['port'] is the started port, opened['url'] the browser url."""
    import types

    from touchstone.cli import review as review_mod
    calls, opened = {"started": 0}, {}

    def fake_start(host, port):
        calls["started"] += 1
        calls["port"] = port
        return types.SimpleNamespace(should_exit=True)  # _wait_for_exit returns at once

    monkeypatch.setattr(review_mod, "_server_up", lambda h, p: up)
    monkeypatch.setattr(review_mod, "_start_server", fake_start)
    monkeypatch.setattr(review_mod, "_open_browser", lambda url: opened.__setitem__("url", url))
    return calls, opened


def test_review_opens_a_room_with_an_opening(tmp_path, monkeypatch, seed_task):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    seed_task(tmp_path / "touchstone")
    _stub_review_server(monkeypatch, up=True)  # server already up: no start, no block
    result = runner.invoke(app, ["review", "--no-open"])
    assert result.exit_code == 0
    assert "/rooms/" in result.stdout
    conn = store.connect(str(tmp_path / ".touchstone" / "touchstone.db"))
    try:
        rooms = store.list_rooms(conn)
        assert len(rooms) == 1
        msgs = store.list_room_messages(conn, rooms[0].id)
        assert msgs and msgs[0].role == "assistant"
    finally:
        conn.close()


def test_review_starts_server_when_down(tmp_path, monkeypatch, seed_task):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    seed_task(tmp_path / "touchstone")
    calls, opened = _stub_review_server(monkeypatch, up=False)
    result = runner.invoke(app, ["review", "--port", "8799"])
    assert result.exit_code == 0
    assert calls["started"] == 1 and calls["port"] == 8799  # started in-process on the given port
    assert opened["url"].startswith("http://127.0.0.1:8799/rooms/")  # browser opened on the room
    assert "Talk in your browser" in result.stdout


def test_review_stops_clean_when_no_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    (tmp_path / "touchstone" / "tasks").mkdir(parents=True)  # empty
    result = runner.invoke(app, ["review", "--no-open"])
    assert result.exit_code == 1
    assert "No tasks in" in result.output and "touchstone survey --force" in result.output


def test_review_reuses_running_server(tmp_path, monkeypatch, seed_task):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    seed_task(tmp_path / "touchstone")
    calls, opened = _stub_review_server(monkeypatch, up=True)
    result = runner.invoke(app, ["review", "--port", "8765"])
    assert result.exit_code == 0
    assert calls["started"] == 0  # already up -> the in-process starter is never called
    assert opened["url"].startswith("http://127.0.0.1:8765/rooms/")
