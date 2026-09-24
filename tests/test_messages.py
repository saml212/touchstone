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


# ---- v2: parts, reasoning, refusal, object acceptance ----------------------

def test_multimodal_parts_are_not_flattened():
    msgs = canonical([
        {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]},
    ])
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this"}
    assert content[1]["type"] == "image" and content[1]["image_url"] == {"url": "http://x/y.png"}
    assert canonical(msgs) == msgs  # idempotent


def test_text_of_marks_non_text_parts():
    from touchstone.messages import text_of
    msg = canonical([{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "input_image", "image_url": "http://x"},
        {"type": "input_file", "filename": "a.pdf"},
    ]}])[0]
    assert text_of(msg) == "look\n[image]\n[file]"
    assert text_of({"role": "user", "content": "plain"}) == "plain"


def test_assistant_reasoning_and_refusal_captured():
    msgs = canonical([{
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "let me think", "signature": "sig"},
            {"type": "redacted_thinking", "data": "xxx"},
            {"type": "text", "text": "the answer"},
        ],
        "refusal": None,
    }])
    m = msgs[0]
    assert m["content"] == "the answer"
    assert m["reasoning"] == [
        {"type": "thinking", "thinking": "let me think", "signature": "sig"},
        {"type": "redacted"},
    ]
    assert canonical(msgs) == msgs


def test_refusal_with_empty_content():
    m = canonical([{"role": "assistant", "content": None, "refusal": "I can't help"}])[0]
    assert m["content"] == "" and m["refusal"] == "I can't help"


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id, self.function = id, _FakeFunction(name, arguments)


class _FakeChatCompletionMessage:
    """No model_dump: forces the attribute-reading fallback path."""

    def __init__(self, content, tool_calls=None, refusal=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.refusal = refusal
        self.reasoning = None


def test_canonical_accepts_sdk_objects_without_model_dump():
    msg = _FakeChatCompletionMessage(
        content="", tool_calls=[_FakeToolCall("c1", "refund", '{"amt": 5}')])
    out = canonical([{"role": "user", "content": "hi"}, msg,
                     {"role": "tool", "tool_call_id": "c1", "content": "ok"}])
    assistant = next(m for m in out if m.get("tool_calls"))
    assert assistant["tool_calls"] == [{"id": "c1", "name": "refund", "arguments": '{"amt": 5}'}]
    assert next(m for m in out if m["role"] == "tool")["tool_call_id"] == "c1"


class _FakeModelDump:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_canonical_prefers_model_dump():
    obj = _FakeModelDump({"role": "assistant", "content": "hi", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
    out = canonical([obj])
    assert out[0]["content"] == "hi"
    assert out[0]["tool_calls"] == [{"id": "c2", "name": "t", "arguments": "{}"}]


def test_parallel_same_name_calls_link_by_id():
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "lookup", "arguments": '{"q": 1}'},
            {"id": "b", "name": "lookup", "arguments": '{"q": 2}'},
        ]},
        {"role": "tool", "tool_call_id": "b", "name": "lookup", "content": "second"},
        {"role": "tool", "tool_call_id": "a", "name": "lookup", "content": "first"},
    ])
    tools = [m for m in msgs if m["role"] == "tool"]
    assert tools[0]["tool_call_id"] == "b" and tools[0]["content"] == "second"
    assert tools[1]["tool_call_id"] == "a" and tools[1]["content"] == "first"


def test_reasoning_round_trips_to_anthropic_thinking():
    msgs = canonical([{"role": "assistant", "content": "done", "reasoning": [
        {"type": "thinking", "thinking": "hmm", "signature": "s"}]}])
    _system, turns = to_anthropic(msgs)
    blocks = turns[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "hmm", "signature": "s"}
    assert {"type": "text", "text": "done"} in blocks


def test_bytes_tool_result_is_decoded_not_crashed():
    # A tool that returns raw bytes must not crash canonical() (capture must never raise).
    msgs = canonical([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "fetch", "arguments": "{}"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "fetch",
         "content": b"raw \xe2\x98\x95 bytes"},
    ])
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["content"] == "raw ☕ bytes"


def test_bytes_in_anthropic_tool_result_block_is_decoded():
    msgs = canonical([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": b"ok"}]}])
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["content"] == "ok"


def test_unknown_responses_item_is_preserved_not_dropped():
    # A Responses API item of a type Touchstone doesn't model (web_search_call,
    # computer_call, ...) must survive verbatim in the trace, not be flattened into
    # an empty user turn that both loses the data and corrupts replay.
    item = {"type": "web_search_call", "id": "ws_1", "status": "completed",
            "action": {"type": "search", "query": "cats"}}
    msgs = canonical([{"role": "user", "content": "hi"}, dict(item),
                      {"role": "assistant", "content": "done"}])
    assert item in msgs
    assert not any(m.get("role") == "user" and m.get("content") == "" for m in msgs)
    assert canonical(msgs) == msgs  # idempotent
    to_openai(msgs)  # wire converters must not crash on the passthrough item
    to_anthropic(msgs)
