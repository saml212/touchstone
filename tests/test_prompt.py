import json

from touchstone.llm.prompt import extract_json, parse_cli_result, serialize_messages


def test_serialize_includes_system_tools_and_turns():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "refund my order"},
    ]
    tools = [{"type": "function", "function": {"name": "refund", "parameters": {}}}]
    out = serialize_messages(msgs, tools)
    assert "SYSTEM:" in out and "be terse" in out
    assert "USER: refund my order" in out
    assert "tool_calls" in out  # the tool-calling instruction
    assert "refund" in out


def test_serialize_tool_result_and_assistant_call():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "refund", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "name": "refund", "content": "ok"},
    ]
    out = serialize_messages(msgs)
    assert "called refund" in out
    assert "TOOL RESULT (refund): ok" in out


def test_extract_json_plain_object():
    assert json.loads(extract_json('{"a": 1}')) == {"a": 1}


def test_extract_json_with_leading_prose():
    text = 'Sure! Here you go:\n{"answer": 42, "nested": {"x": [1,2]}} cheers'
    assert json.loads(extract_json(text)) == {"answer": 42, "nested": {"x": [1, 2]}}


def test_extract_json_fenced_block():
    text = "```json\n{\"ok\": true}\n```"
    assert json.loads(extract_json(text)) == {"ok": True}


def test_extract_json_array():
    assert json.loads(extract_json("result: [1, 2, 3]")) == [1, 2, 3]


def test_extract_json_ignores_braces_in_strings():
    text = '{"note": "a } is fine", "n": 1}'
    assert json.loads(extract_json(text)) == {"note": "a } is fine", "n": 1}


def test_extract_json_none_when_absent():
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_extract_json_unbalanced_returns_none():
    assert extract_json('{"a": 1') is None


def test_parse_cli_result_tool_calls():
    text = 'I will call it:\n{"tool_calls": [{"name": "refund", "arguments": {"amount": 5}}]}'
    reply = parse_cli_result(text, want_json=False)
    assert reply.tool_calls[0]["name"] == "refund"
    assert json.loads(reply.tool_calls[0]["arguments"]) == {"amount": 5}


def test_parse_cli_result_json_mode_extracts():
    reply = parse_cli_result('here: {"answer": 1} done', want_json=True)
    assert json.loads(reply.content) == {"answer": 1}


def test_parse_cli_result_plain_text():
    reply = parse_cli_result("just some text", want_json=False)
    assert reply.content == "just some text"
    assert reply.tool_calls == []
