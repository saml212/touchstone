from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()


def _demo_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    runner.invoke(app, ["demo", "--n", "30"])


def test_mine_no_llm_proposes_and_cuts(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mine", "--no-llm"])
    assert result.exit_code == 0, result.output
    assert "proposed" in result.output and "cut" in result.output
    assert "no_pii" in result.output  # stats table lists the PII proposal

    # Re-running does not duplicate mined checks (dedupe).
    listed_1 = runner.invoke(app, ["checks", "list"]).output
    runner.invoke(app, ["mine", "--no-llm"])
    listed_2 = runner.invoke(app, ["checks", "list"]).output
    assert listed_1.count("\n") == listed_2.count("\n")


def test_mine_with_code_scan(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    (tmp_path / "src.py").write_text('SYSTEM_PROMPT = "be nice"\n', encoding="utf-8")
    result = runner.invoke(app, ["mine", "--no-llm", "--code", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_checks_enable_all_mined(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    runner.invoke(app, ["mine", "--no-llm"])
    result = runner.invoke(app, ["checks", "enable", "--all-mined"])
    assert result.exit_code == 0
    assert "enabled" in result.output
    # Every mined check should now be on.
    listed = runner.invoke(app, ["checks", "list"]).output
    assert "[off]" not in listed


def test_checks_enable_requires_target(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["checks", "enable"])
    assert result.exit_code == 1


def test_tasks_list_show_and_sync(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    runner.invoke(app, ["mine", "--no-llm"])
    runner.invoke(app, ["checks", "enable", "--all-mined"])
    synced = runner.invoke(app, ["tasks", "sync"])
    assert synced.exit_code == 0 and "synced" in synced.output

    listed = runner.invoke(app, ["tasks", "list"])
    assert listed.exit_code == 0
    # grouped by queue: headers are flush-left, task rows are indented.
    task_lines = [ln for ln in listed.output.splitlines() if ln.startswith("  ")]
    assert task_lines, listed.output
    name = task_lines[0].split()[0]

    shown = runner.invoke(app, ["tasks", "show", name])
    assert shown.exit_code == 0
    assert "context:" in shown.output and "reference:" in shown.output and "checks:" in shown.output


def test_tasks_list_tag_filter(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    runner.invoke(app, ["mine", "--no-llm"])
    result = runner.invoke(app, ["tasks", "list", "--tag", "failure"])
    assert result.exit_code == 0
    for line in result.output.splitlines():
        if line.startswith("  "):  # a task row, not a queue header
            assert "failure" in line


def test_tasks_show_unknown_id_nonzero(tmp_path, monkeypatch):
    _demo_repo(tmp_path, monkeypatch)
    assert runner.invoke(app, ["tasks", "show", "nope"]).exit_code == 1
