import json

from touchstone.survey.map import build_map
from touchstone.survey.provider import ScriptedSurveyProvider

VALID = {
    "entrypoints": ["agent.py:main"],
    "system_prompts": [{"file": "agent.py", "line": 17, "text": "You are a support agent."}],
    "tools": [
        {"name": "order_status", "import_path": "agent:order_status", "file": "agent.py",
         "line": 34, "calls": ["orders"]},
        {"name": "note", "import_path": "agent:note", "file": "agent.py", "line": 40, "calls": []},
    ],
    "model_call": {"file": "agent.py", "line": 71, "sdk": "openai", "model_setting": "model arg"},
    "services": [
        {"name": "orders", "kind": "http", "base_url_env": "ORDERS_URL",
         "base_url_default": "http://127.0.0.1:8710",
         "calls": [{"method": "GET", "path_template": "/orders/{id}", "from_tool": "order_status"}]}
    ],
    "schemas": {"order_status": {"type": "object"}},
}


def test_build_map_writes_and_caches(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    provider = ScriptedSurveyProvider([json.dumps(VALID)])
    data = build_map(repo, provider, out)
    assert data["tools"][0]["name"] == "order_status"
    assert (out / "map.json").exists()

    # cache: a second call must not consult the provider again
    again = build_map(repo, provider, out)
    assert again["services"][0]["name"] == "orders"
    assert len(provider.calls) == 1


def test_build_map_force_reruns(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    provider = ScriptedSurveyProvider([json.dumps(VALID), json.dumps(VALID)])
    build_map(repo, provider, out)
    build_map(repo, provider, out, force=True)
    assert len(provider.calls) == 2


def test_build_map_strips_markdown_fence(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    fenced = "Here is the map:\n```json\n" + json.dumps(VALID) + "\n```\n"
    provider = ScriptedSurveyProvider([fenced])
    data = build_map(repo, provider, out)
    assert data["model_call"]["sdk"] == "openai"


def test_build_map_retries_once_then_succeeds(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    invalid = json.dumps({"entrypoints": ["x"]})  # missing required keys
    provider = ScriptedSurveyProvider([invalid, json.dumps(VALID)])
    data = build_map(repo, provider, out)
    assert data["tools"]
    assert len(provider.calls) == 2
    assert "invalid" in provider.calls[1]  # retry prompt carried the error


def test_build_map_fails_loudly_when_never_valid(tmp_path):
    import pytest
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    provider = ScriptedSurveyProvider(["not json", "still not json"])
    with pytest.raises(ValueError):
        build_map(repo, provider, out)
