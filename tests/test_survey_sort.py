from touchstone.survey.recordings import ToolEvent
from touchstone.survey.sort import sort_tools

MAP = {
    "tools": [
        {"name": "order_status", "calls": ["orders"]},
        {"name": "refund", "calls": ["orders"]},
        {"name": "summarize", "calls": []},
    ]
}


def _ev(tool):
    return ToolEvent(tool=tool, arguments={}, output=None, episode="e")


def test_sort_splits_by_service_calls():
    result = sort_tools(MAP, [_ev("order_status"), _ev("refund"), _ev("summarize")])
    assert result["crosses_the_network"] == ["order_status", "refund"]
    assert result["runs_on_its_own"] == ["summarize"]
    assert result["unmapped"] == []
    assert result["unused"] == []


def test_sort_flags_unmapped_and_unused():
    # 'ghost' is recorded but not in the map; 'summarize' is mapped but never recorded.
    result = sort_tools(MAP, [_ev("order_status"), _ev("ghost")])
    assert result["unmapped"] == ["ghost"]
    # refund and summarize are in the map but never appear in these recordings
    assert result["unused"] == ["refund", "summarize"]
