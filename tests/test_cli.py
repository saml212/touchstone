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
    assert "keychain_prefix" not in text


def test_init_writes_keychain_prefix_when_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--keychain-prefix", "rockie-"])
    text = (tmp_path / "touchstone.toml").read_text()
    assert 'keychain_prefix = "rockie-"' in text


def test_doctor_reports_one_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for row in ("component", "python", "database", "provider: scripted", "provider: reference",
                "provider: claude-cli", "speech: mode", "speech: stt", "tool: harbor", "tool: docker",
                "tool: ffmpeg", "train: art", "train: trl"):
        assert row in result.output, row


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


def _init(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])


def test_checks_add_list_show_eval_flow(tmp_path, monkeypatch):
    _init(runner, tmp_path, monkeypatch)
    add = runner.invoke(
        app,
        ["checks", "add", "--kind", "contains", "--params", '{"values": ["hi"]}', "--name", "g"],
    )
    assert add.exit_code == 0, add.output
    cid = add.output.strip().split()[-1]

    listed = runner.invoke(app, ["checks", "list"])
    assert cid in listed.output and "contains" in listed.output

    shown = runner.invoke(app, ["checks", "show", cid])
    assert "kind:" in shown.output and "contains" in shown.output

    ok = runner.invoke(app, ["checks", "eval", cid, "--text", "well hi there"])
    assert ok.exit_code == 0 and "PASS" in ok.output
    bad = runner.invoke(app, ["checks", "eval", cid, "--text", "nope"])
    assert "FAIL" in bad.output


def test_checks_enable_disable(tmp_path, monkeypatch):
    _init(runner, tmp_path, monkeypatch)
    add = runner.invoke(app, ["checks", "add", "--kind", "no_pii", "--params", "{}"])
    cid = add.output.strip().split()[-1]
    assert runner.invoke(app, ["checks", "disable", cid]).exit_code == 0
    assert "off" in runner.invoke(app, ["checks", "list"]).output
    assert runner.invoke(app, ["checks", "enable", cid]).exit_code == 0


def test_checks_add_bad_params_exits_nonzero(tmp_path, monkeypatch):
    _init(runner, tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["checks", "add", "--kind", "regex", "--params", '{"pattern": "["}']
    )
    assert result.exit_code == 1
    assert "invalid" in (result.output + str(result.stderr or ""))


def test_checks_eval_tool_calls(tmp_path, monkeypatch):
    _init(runner, tmp_path, monkeypatch)
    add = runner.invoke(
        app, ["checks", "add", "--kind", "tool_called", "--params", '{"name": "refund"}']
    )
    cid = add.output.strip().split()[-1]
    result = runner.invoke(
        app, ["checks", "eval", cid, "--tool-calls", '[{"name": "refund", "arguments": "{}"}]']
    )
    assert "PASS" in result.output


def test_checks_eval_unknown_id_nonzero(tmp_path, monkeypatch):
    _init(runner, tmp_path, monkeypatch)
    result = runner.invoke(app, ["checks", "eval", "nope", "--text", "x"])
    assert result.exit_code == 1


def test_readonly_db_dir_fails_cleanly(tmp_path, monkeypatch):
    import os
    import stat

    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("TOUCHSTONE_DB", str(ro / "sub" / "t.db"))
    try:
        result = runner.invoke(app, ["bench", "runs"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "\n" not in result.output.strip()
        init = runner.invoke(app, ["init"], catch_exceptions=False)
        assert init.exit_code == 1
    finally:
        os.chmod(ro, stat.S_IRWXU)
