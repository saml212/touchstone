"""Attribution resolves a call whose tool name is shared by two crossing services, and drops a
service that ends up with zero attributed calls (tau-bench retail vs airline)."""

from touchstone.survey.attribute import (
    attribute,
    effective_crossing_services,
    service_calls,
)
from touchstone.survey.recordings import ToolEvent

MAP = {
    "tools": [
        {"name": "get_order", "import_path": "app.retail.get_order:f", "calls": ["retail"]},
        {"name": "get_user", "import_path": "app.retail.get_user:f", "calls": ["retail"]},
        {"name": "get_user", "import_path": "app.airline.get_user:f", "calls": ["airline"]},
        {"name": "book", "import_path": "app.airline.book:f", "calls": ["airline"]},
    ],
    "services": [
        {"name": "retail", "kind": "db", "base_url_env": None, "base_url_default": ":memory:",
         "calls": []},
        {"name": "airline", "kind": "db", "base_url_env": None, "base_url_default": ":memory:",
         "calls": []},
    ],
}


def _ev(tool, ep, module=None):
    return ToolEvent(tool=tool, arguments={}, output=None, episode=ep, module=module)


def test_module_attribution_wins():
    events = [_ev("get_user", "e1", module="app.airline.get_user")]
    assert [e.tool for e in service_calls(MAP, events, "airline")] == ["get_user"]
    assert service_calls(MAP, events, "retail") == []


def test_episode_cooccurrence_fallback_when_no_module():
    # A retail-only unique tool (get_order) in the episode pulls the shared get_user to retail.
    events = [_ev("get_order", "e1"), _ev("get_user", "e1")]
    assert {e.tool for e in service_calls(MAP, events, "retail")} == {"get_order", "get_user"}
    assert service_calls(MAP, events, "airline") == []


def test_zero_call_service_is_dropped():
    events = [_ev("get_order", "e1"), _ev("get_user", "e1")]
    names = [s["name"] for s in effective_crossing_services(MAP, events)]
    assert names == ["retail"]  # airline had no attributed call


def test_ambiguous_shared_call_is_unattributed():
    # No unique tool anywhere in the episode, no module: the shared call can't be placed.
    events = [_ev("get_user", "e9")]
    grouped = attribute(MAP, events)
    assert grouped["retail"] == [] and grouped["airline"] == []
