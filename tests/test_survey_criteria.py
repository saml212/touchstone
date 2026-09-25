"""Unit tests for the effect -> rewardkit criterion translation (state diff + trajectory)."""

from touchstone.survey.criteria import (
    _sql_literal,
    derive_criteria,
    episode_services,
    reproduced,
)
from touchstone.survey.recordings import ToolEvent

MAP = {
    "tools": [{"name": "get", "import_path": "t:get", "calls": ["svc"]},
              {"name": "paint", "import_path": "t:paint", "calls": ["svc"]},
              {"name": "wipe", "import_path": "t:wipe", "calls": ["svc"]}],
    "services": [{"name": "svc", "kind": "http", "base_url_env": "SVC", "calls": [
        {"method": "GET", "path_template": "/w/{id}", "from_tool": "get"},
        {"method": "POST", "path_template": "/w/{id}/paint", "from_tool": "paint"},
        {"method": "DELETE", "path_template": "/w/{id}", "from_tool": "wipe"}]}],
}


def _effect_change():
    return {
        "initial": {"svc": {"widgets": {"pk": "id", "rows": [{"id": "w1", "color": "red"}]},
                            "notes": {"pk": "id", "rows": []}}},
        "final": {"svc": {"widgets": {"pk": "id", "rows": [{"id": "w1", "color": "blue"}]},
                          "notes": {"pk": "id", "rows": [{"id": 1, "widget_id": "w1"}]}}},
    }


def test_sql_literal_quotes_and_numbers():
    assert _sql_literal("a'b") == "'a''b'"
    assert _sql_literal(3.5) == "3.5"
    assert _sql_literal(True) == "1"


def test_derive_state_criteria_changed_cell_and_added_row():
    calls = [ToolEvent("paint", {"widget_id": "w1", "color": "blue"}, {}, "e")]
    services = episode_services(MAP, calls)
    state, tool = derive_criteria(_effect_change(), services, MAP, calls)
    assert any("SELECT color FROM widgets WHERE id='w1'" in s and "'blue'" in s for s in state)
    assert any("COUNT(*) FROM notes WHERE widget_id='w1'" in s for s in state)


def test_derive_trajectory_used_and_avoid():
    calls = [ToolEvent("get", {"widget_id": "w1"}, {}, "e")]
    _, tool = derive_criteria({"initial": {}, "final": {}}, [], MAP, calls)
    assert "rk.trajectory_tool_used('get')" in tool
    # paint (POST) and wipe (DELETE) are mutating + unused -> must-not-use criteria
    assert "rk.trajectory_tool_not_used('paint')" in tool
    assert "rk.trajectory_tool_not_used('wipe')" in tool


def test_reproduced_true_and_false():
    calls = [ToolEvent("get", {}, {"id": "w1", "color": "red"}, "e")]
    ok = [{"tool": "get", "got": {"id": "w9", "color": "red"}}]  # id masked -> matches
    assert reproduced(calls, ok) is True
    bad = [{"tool": "get", "got": {"id": "w1", "color": "green"}}]
    assert reproduced(calls, bad) is False
    assert reproduced(calls, [{"tool": "get", "got": {"__error__": "boom"}}]) is False


def test_episode_services_only_touched():
    calls = [ToolEvent("get", {}, {}, "e")]
    assert [s["name"] for s in episode_services(MAP, calls)] == ["svc"]
    assert episode_services(MAP, []) == []
