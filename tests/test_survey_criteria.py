"""Unit tests for the effect -> rewardkit criterion translation (state diff + trajectory)."""

from touchstone.survey.criteria import (
    _sql_literal,
    derive_criteria,
    episode_services,
    reproduced,
)
from touchstone.survey.recordings import ToolEvent


def _calls(pairs):
    """derive_criteria returns (call, description) pairs; most tests check the call."""
    return [call for call, _ in pairs]


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
    state, tool, _ = derive_criteria(_effect_change(), services, MAP, calls)
    joined = "\n".join(_calls(state))
    assert "SELECT color FROM widgets WHERE id='w1'" in joined and "'blue'" in joined
    assert "COUNT(*) FROM notes WHERE widget_id='w1'" in joined
    # each criterion carries a plain-English, path-free description
    descs = [d for _, d in state]
    assert "the color of widgets w1 is blue" in descs
    assert all("state.db" not in d and "/logs/" not in d for d in descs)


def test_derive_returns_required_where_literals():
    # the changed-cell WHERE key (w1) and the added-row identifying value (email) must be reported
    effect = {
        "initial": {"svc": {"widgets": {"pk": "id", "rows": [{"id": "w1", "color": "red"}]},
                            "emails": {"pk": "id", "rows": []}}},
        "final": {"svc": {"widgets": {"pk": "id", "rows": [{"id": "w1", "color": "blue"}]},
                          "emails": {"pk": "id", "rows": [{"id": 1, "to_addr": "p1@ex.invalid"}]}}},
    }
    calls = [ToolEvent("paint", {"widget_id": "w1", "color": "blue", "to": "p1@ex.invalid"},
                       {}, "e")]
    _, _, literals = derive_criteria(effect, episode_services(MAP, calls), MAP, calls)
    assert "w1" in literals and "p1@ex.invalid" in literals


def test_added_row_without_identifying_token_emits_no_unscoped_count():
    # An added row whose columns are all free text (no value the agent supplied) has no column to
    # scope a WHERE. An unscoped `SELECT COUNT(*) FROM emails` would be a table-total coupled to the
    # seed — so no criterion is emitted (a scoped check already covers the real effect).
    effect = {
        "initial": {"svc": {"emails": {"pk": "id", "rows": [{"id": 7, "body": "old seed mail"}]}}},
        "final": {"svc": {"emails": {"pk": "id", "rows": [
            {"id": 7, "body": "old seed mail"}, {"id": 8, "body": "thanks for your order"}]}}},
    }
    calls = [ToolEvent("paint", {"widget_id": "w1", "color": "blue"}, {}, "e")]
    state, _, _ = derive_criteria(effect, episode_services(MAP, calls), MAP, calls)
    assert not any("COUNT(*) FROM emails" in line for line in _calls(state))  # no table-total


def test_read_only_tool_never_required():
    # get is read-only + no state change -> never a tool_used criterion; only avoids mutating
    calls = [ToolEvent("get", {"widget_id": "w1"}, {}, "e")]
    _, tool, _ = derive_criteria({"initial": {}, "final": {}}, [], MAP, calls)
    assert not any("trajectory_tool_used" in line for line in _calls(tool))
    assert "rk.trajectory_tool_not_used('paint')" in _calls(tool)
    assert "rk.trajectory_tool_not_used('wipe')" in _calls(tool)


def test_mutating_tool_covered_by_state_not_required():
    # paint mutates and its effect IS in the diff -> state covers it, no tool_used(paint)
    calls = [ToolEvent("paint", {"widget_id": "w1", "color": "blue"}, {}, "e")]
    state, tool, _ = derive_criteria(_effect_change(), episode_services(MAP, calls), MAP, calls)
    assert state  # a state criterion exists
    assert not any("trajectory_tool_used('paint')" in line for line in _calls(tool))
    assert "rk.trajectory_tool_not_used('wipe')" in _calls(tool)


def test_mutating_tool_without_state_effect_is_required():
    # a mutating tool whose effect leaves no diff -> require it in the trajectory (only evidence)
    calls = [ToolEvent("paint", {"widget_id": "w1"}, {}, "e")]
    _, tool, _ = derive_criteria({"initial": {}, "final": {}}, [], MAP, calls)
    assert "rk.trajectory_tool_used('paint')" in _calls(tool)


def test_identifying_where_rejects_free_text():
    # subject/body are free text (whitespace) -> excluded; to_addr is a token -> kept
    effect = {
        "initial": {"svc": {"emails": {"pk": "id", "rows": []}}},
        "final": {"svc": {"emails": {"pk": "id", "rows": [
            {"id": 1, "to_addr": "p1@example.invalid", "subject": "Refund for Order B1"}]}}},
    }
    calls = [ToolEvent("paint", {"to": "p1@example.invalid", "subject": "Refund for Order B1"},
                       {}, "e")]
    state, _, _ = derive_criteria(effect, episode_services(MAP, calls), MAP, calls)
    joined = "\n".join(_calls(state))
    assert "to_addr='p1@example.invalid'" in joined
    assert "subject=" not in joined


def test_avoid_excludes_non_tool_harness_helpers():
    # a from_tool that is not a real model tool (e.g. a seed helper) is never an avoid criterion
    m = {"tools": [{"name": "get", "calls": ["svc"]}],
         "services": [{"name": "svc", "calls": [
             {"method": "GET", "path_template": "/w/{id}", "from_tool": "get"},
             {"method": "POST", "path_template": "/seed", "from_tool": "seed_helper"}]}]}
    _, tool, _ = derive_criteria({"initial": {}, "final": {}}, [], m,
                              [ToolEvent("get", {}, {}, "e")])
    assert not any("seed_helper" in line for line in _calls(tool))


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


def test_plain_string_tool_result_is_handled():
    # a tool whose recorded result is a plain string (not JSON) must not crash the criteria path
    calls = [ToolEvent("get", {"widget_id": "w1"}, "all good", "e")]
    assert reproduced(calls, [{"tool": "get", "got": "all good"}]) is True
    assert reproduced(calls, [{"tool": "get", "got": "different"}]) is False
    # no state change -> no sqlite criteria; read-only get is never required (never an exception)
    state, tool, _ = derive_criteria({"initial": {}, "final": {}}, [], MAP, calls)
    assert state == []
    assert not any("trajectory_tool_used('get')" in line for line in _calls(tool))
