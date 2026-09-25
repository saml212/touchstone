from typer.testing import CliRunner

from touchstone.cli import app
from touchstone.survey import survey as survey_mod

runner = CliRunner()


def test_survey_command_echoes_summary(monkeypatch):
    seen = {}

    def fake(repo, force=False, provider=None, model=None):
        seen.update(repo=repo, force=force, provider=provider, model=model)
        return "Mapped 4 tools, 1 service."

    monkeypatch.setattr(survey_mod, "run_survey", fake)
    result = runner.invoke(app, ["survey", "/some/repo", "--force", "--provider", "codex-cli"])
    assert result.exit_code == 0
    assert "Mapped 4 tools, 1 service." in result.stdout
    assert seen == {"repo": "/some/repo", "force": True, "provider": "codex-cli", "model": None}


def test_survey_command_reports_failure(monkeypatch):
    def boom(repo, **kw):
        raise ValueError("no such repo")

    monkeypatch.setattr(survey_mod, "run_survey", boom)
    result = runner.invoke(app, ["survey", "/nope"])
    assert result.exit_code == 1
