"""Recordings reconstruct tool events from model spans and surface the tool function's `module`
(recorded on the tool span) so a shared-name call can be attributed to the module that ran."""

from dataclasses import dataclass

from touchstone.survey.recordings import _episode_events


@dataclass
class _Span:
    kind: str
    input: dict | None = None
    output: dict | None = None
    tool_call_id: str | None = None


def _model_span(call_id, name, args, result):
    return [
        _Span(kind="model", output={"message": {"tool_calls": [
            {"id": call_id, "name": name, "arguments": args}]}}),
        _Span(kind="model", input={"messages": [
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": result}]}),
    ]


def test_module_from_tool_span_is_surfaced():
    spans = _model_span("c1", "get_user_details", '{"user_id": "u1"}', '{"email": "a@b.com"}')
    spans.append(_Span(kind="tool", tool_call_id="c1",
                       input={"name": "get_user_details", "module": "app.retail.tools"}))
    events = _episode_events(spans)
    assert len(events) == 1
    assert events[0].tool == "get_user_details"
    assert events[0].module == "app.retail.tools"


def test_module_is_none_when_no_tool_span():
    spans = _model_span("c1", "get_order", '{"order_id": "#W1"}', '{"status": "pending"}')
    events = _episode_events(spans)
    assert events[0].module is None
