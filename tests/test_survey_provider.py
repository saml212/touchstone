import pytest

from touchstone.config import Settings
from touchstone.survey.provider import ScriptedSurveyProvider, survey_provider


def test_scripted_returns_responses_in_order(tmp_path):
    p = ScriptedSurveyProvider(["one", "two"])
    assert p.run("a", tmp_path) == "one"
    assert p.run("b", tmp_path) == "two"
    assert p.run("c", tmp_path) == "two"  # last repeats
    assert p.calls == ["a", "b", "c"]


def test_survey_provider_rejects_unknown():
    with pytest.raises(ValueError):
        survey_provider(Settings(survey_provider="nope"))


def test_survey_provider_resolves_claude_or_skips_when_absent():
    import shutil
    s = Settings(survey_provider="claude-cli")
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")
    prov = survey_provider(s)
    assert prov.name.startswith("claude-cli")
