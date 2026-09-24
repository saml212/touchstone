import json

from touchstone.messages import canonical, to_anthropic, to_openai

OPENAI_WIRE = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "refund order 1"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_x", "type": "function",
         "function": {"name": "refund", "arguments": '{"amount": 5}'}},
    ]},
    {"role": "tool", "tool_call_id": "call_x", "name": "refund", "content": "done"},
    {"role": "assistant", "content": "All refunded."},
]

ANTHROPIC_WIRE = [
    {"role": "user", "content": "refund order 1"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "sure"},
        {"type": "tool_use", "id": "call_x", "name": "refund", "input": {"amount": 5}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_x", "content": "done"},
    ]},
]


def test_canonical_from_openai_wire():
    msgs = canonical(OPENAI_WIRE)
    assistant = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["tool_calls"] == [
        {"id": "call_x", "name": "refund", "arguments": '{"amount": 5}'}]
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_x" and tool["content"] == "done"


def test_canonical_from_anthropic_wire():
    msgs = canonical(ANTHROPIC_WIRE)
    assistant = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["content"] == "sure"
    assert assistant["tool_calls"] == [
        {"id": "call_x", "name": "refund", "arguments": '{"amount": 5}'}]
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_x" and tool["content"] == "done"


def test_already_canonical_is_unchanged():
    once = canonical(OPENAI_WIRE)
    assert canonical(once) == once


def test_idempotence_all_shapes():
    for wire in (OPENAI_WIRE, ANTHROPIC_WIRE):
        c = canonical(wire)
        assert canonical(c) == c


def test_to_openai_roundtrip_has_tool_call_id_and_function():
    wire = to_openai(canonical(OPENAI_WIRE))
    assistant = next(m for m in wire if m.get("tool_calls"))
    tc = assistant["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"] == {"name": "refund", "arguments": '{"amount": 5}'}
    tool = next(m for m in wire if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_x"


def test_to_anthropic_roundtrip_has_blocks():
    system, turns = to_anthropic(canonical(ANTHROPIC_WIRE))
    assert system == ""
    assistant = next(t for t in turns if t["role"] == "assistant")
    assert {"type": "tool_use", "id": "call_x", "name": "refund", "input": {"amount": 5}} in \
        assistant["content"]
    result_turn = turns[-1]
    assert result_turn["content"][0] == {
        "type": "tool_result", "tool_use_id": "call_x", "content": "done"}


def test_to_anthropic_extracts_system():
    system, turns = to_anthropic(canonical(OPENAI_WIRE))
    assert system == "be terse"
    assert all(t["role"] != "system" for t in turns)


def test_ids_generated_and_linked_by_name_order_without_tool_call_id():
    # Touchstone's own shape: no ids, tool messages carry only name.
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "order_status", "arguments": {"order_id": "A1"}},
            {"name": "refund", "arguments": {"order_id": "A1", "amount": 5}},
        ]},
        {"role": "tool", "name": "refund", "content": "{}"},
        {"role": "tool", "name": "order_status", "content": "{}"},
    ])
    calls = msgs[0]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_1", "call_2"]
    assert json.loads(calls[0]["arguments"]) == {"order_id": "A1"}
    tools = [m for m in msgs if m["role"] == "tool"]
    # linked by name, not position
    assert tools[0]["tool_call_id"] == "call_2"  # refund
    assert tools[1]["tool_call_id"] == "call_1"  # order_status


def test_link_by_order_when_no_name():
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [{"name": "a", "arguments": "{}"}]},
        {"role": "tool", "content": "r"},  # no name, no id
    ])
    assert msgs[1]["tool_call_id"] == "call_1"


def test_arguments_non_json_string_preserved_for_openai():
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "t", "arguments": "not json"}}]},
    ])
    assert msgs[0]["tool_calls"][0]["arguments"] == "not json"
    wire = to_openai(msgs)
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_arguments_non_json_string_becomes_input_for_anthropic():
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "t", "arguments": "not json"}]},
    ])
    _system, turns = to_anthropic(msgs)
    assert turns[0]["content"][0]["input"] == {"input": "not json"}


def test_content_parts_list_joined():
    msgs = canonical([
        {"role": "user", "content": [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]},
    ])
    assert msgs == [{"role": "user", "content": "line one\nline two"}]


def test_empty_and_none_are_safe():
    assert canonical([]) == []
    assert canonical(None) == []
    assert to_openai([]) == []
    assert to_anthropic([]) == ("", [])
