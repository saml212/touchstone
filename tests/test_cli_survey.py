from typer.testing import CliRunner

from touchstone.cli import app
from touchstone.survey import survey as survey_mod

runner = CliRunner()


def test_survey_command_echoes_summary(monkeypatch):
    seen = {}

    def fake(repo, force=False, provider=None, model=None, skip_gate=False, skip_baseline=False,
             rebaseline=False):
        seen.update(repo=repo, force=force, provider=provider, model=model, skip_gate=skip_gate,
                    skip_baseline=skip_baseline)
        return "Mapped 4 tools, 1 service."

    monkeypatch.setattr(survey_mod, "run_survey", fake)
    result = runner.invoke(app, ["survey", "/some/repo", "--force", "--provider", "codex-cli",
                                 "--skip-gate", "--skip-baseline"])
    assert result.exit_code == 0
    assert "Mapped 4 tools, 1 service." in result.stdout
    assert seen == {"repo": "/some/repo", "force": True, "provider": "codex-cli", "model": None,
                    "skip_gate": True, "skip_baseline": True}


def test_survey_command_reports_failure(monkeypatch):
    def boom(repo, **kw):
        raise ValueError("no such repo")

    monkeypatch.setattr(survey_mod, "run_survey", boom)
    result = runner.invoke(app, ["survey", "/nope"])
    assert result.exit_code == 1


def test_survey_command_renders_docker_daemon_error_as_one_line(monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_DEBUG", raising=False)  # another test's --debug can leak in-proc
    from touchstone.harbor.run import DockerDaemonError

    def boom(repo, **kw):
        raise DockerDaemonError(
            "Docker daemon not running on local. Survey needs it for the gate and baseline. "
            "Start Docker, or set [harbor] host, then run again — or survey without Docker with "
            "--skip-gate --skip-baseline.")

    monkeypatch.setattr(survey_mod, "run_survey", boom)
    result = runner.invoke(app, ["survey", "/repo"])
    assert result.exit_code == 1
    assert "Survey needs it for the gate and baseline" in result.output
    assert "Traceback" not in result.output  # rendered as one clean line, not a stack dump
