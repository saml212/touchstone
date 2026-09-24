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


def test_doctor_reports_providers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "providers: scripted" in result.output
    assert "python:" in result.output


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
