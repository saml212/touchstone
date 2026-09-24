"""The sample CLI command end-to-end on the zero-key demo (scripted student + teacher)."""

from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()


def _ready_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    runner.invoke(app, ["demo", "--n", "12"])
    runner.invoke(app, ["mine", "--no-llm"])
    runner.invoke(app, ["checks", "enable", "--all-mined"])
    runner.invoke(app, ["tasks", "sync"])
    runner.invoke(app, ["bench", "create", "demo", "--all"])


def test_sample_cli_prints_frontier(tmp_path, monkeypatch):
    _ready_repo(tmp_path, monkeypatch)
    r = runner.invoke(app, ["sample", "demo", "--student", "scripted", "--variants", "1"])
    assert r.exit_code == 0, r.output
    assert "frontier:" in r.output and "learnability band" in r.output
