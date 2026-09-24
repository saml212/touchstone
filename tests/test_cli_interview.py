from typer.testing import CliRunner

from touchstone import store
from touchstone.cli import app

runner = CliRunner()


def _seed(db):
    conn = store.connect(db)
    try:
        task = store.insert_task(conn, store.Task(name="t1"))
        return task.id
    finally:
        conn.close()


def test_doctor_reports_speech(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "speech:" in result.output
    assert "stt" in result.output and "tts" in result.output


def test_interview_prints_url_and_nudges_when_server_down(tmp_path, monkeypatch):
    db = str(tmp_path / "d.db")
    monkeypatch.setenv("TOUCHSTONE_DB", db)
    monkeypatch.setattr("touchstone.cli.interview._server_up", lambda host, port: False)
    task_id = _seed(db)
    result = runner.invoke(app, ["interview", task_id, "--no-open"])
    assert result.exit_code == 0, result.output
    assert "/rooms/" in result.output
    assert "touchstone serve" in result.output
    # the room and its opening message were persisted
    conn = store.connect(db)
    try:
        rooms = store.list_rooms(conn)
        assert len(rooms) == 1
        msgs = store.list_room_messages(conn, rooms[0].id)
        assert msgs and msgs[0].role == "assistant"
    finally:
        conn.close()


def test_interview_opens_browser_when_server_up(tmp_path, monkeypatch):
    db = str(tmp_path / "d.db")
    monkeypatch.setenv("TOUCHSTONE_DB", db)
    monkeypatch.setattr("touchstone.cli.interview._server_up", lambda host, port: True)
    opened = {}
    monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url))
    task_id = _seed(db)
    result = runner.invoke(app, ["interview", task_id])
    assert result.exit_code == 0, result.output
    assert opened["url"].endswith(result.output.strip().splitlines()[0].split("/rooms/")[-1])


def test_interview_unknown_task_fails(tmp_path, monkeypatch):
    db = str(tmp_path / "d.db")
    monkeypatch.setenv("TOUCHSTONE_DB", db)
    store.connect(db).close()
    result = runner.invoke(app, ["interview", "nope", "--no-open"])
    assert result.exit_code == 1
    assert "no task" in result.output
