import json

import httpx
import pytest

from touchstone.llm import ProviderError
from touchstone.llm.anthropic import AnthropicProvider

KEY = "sk-ant-secret"


def make(handler, **kw):
    return AnthropicProvider(
        "claude-sonnet-4-5", KEY, transport=httpx.MockTransport(handler), backoff=0, **kw
    )


def _ok(blocks=None, usage=None):
    body = {
        "content": blocks or [{"type": "text", "text": "hi"}],
        "usage": usage or {"input_tokens": 4, "output_tokens": 3},
    }
    return httpx.Response(200, json=body)


def test_success_and_usage():
    r = make(lambda req: _ok([{"type": "text", "text": "hello"}])).chat(
        [{"role": "user", "content": "hi"}]
    )
    assert r.content == "hello"
    assert r.usage == {"tokens_in": 4, "tokens_out": 3}


def test_system_extracted_and_headers():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["headers"] = req.headers
        return _ok()

    make(handler).chat(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    assert seen["body"]["system"] == "be terse"
    assert seen["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert seen["headers"]["x-api-key"] == KEY
    assert seen["headers"]["anthropic-version"]
    assert "max_tokens" in seen["body"]


def test_tools_converted_to_input_schema():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok()

    params = {"type": "object", "properties": {"a": {"type": "number"}}}
    tools = [{"type": "function",
              "function": {"name": "refund", "description": "d", "parameters": params}}]
    make(handler).chat([{"role": "user", "content": "x"}], tools=tools)
    assert seen["body"]["tools"] == [
        {"name": "refund", "description": "d", "input_schema": params}
    ]


def test_tool_use_block_parsed():
    blocks = [{"type": "tool_use", "id": "u1", "name": "refund", "input": {"amount": 5}}]
    r = make(lambda req: _ok(blocks)).chat([{"role": "user", "content": "refund"}])
    assert r.tool_calls[0]["name"] == "refund"
    assert json.loads(r.tool_calls[0]["arguments"]) == {"amount": 5}


def test_tool_result_becomes_user_block():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok()

    msgs = [
        {"role": "user", "content": "refund order 1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "u1", "function": {"name": "refund", "arguments": '{"amount": 5}'}}]},
        {"role": "tool", "tool_call_id": "u1", "content": "done"},
    ]
    make(handler).chat(msgs)
    turns = seen["body"]["messages"]
    assert turns[1]["content"][0]["type"] == "tool_use"
    assert turns[2]["role"] == "user"
    assert turns[2]["content"][0] == {"type": "tool_result", "tool_use_id": "u1", "content": "done"}


def test_canonical_context_becomes_tool_use_and_tool_result_blocks():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok()

    canonical_context = [
        {"role": "user", "content": "refund order 1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "name": "refund", "arguments": '{"amount": 5}'}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "refund", "content": "done"},
    ]
    make(handler).chat(canonical_context)
    turns = seen["body"]["messages"]
    assistant = next(t for t in turns if t["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "refund", "input": {"amount": 5}}
    result = turns[-1]
    assert result["role"] == "user"
    assert result["content"][0] == {
        "type": "tool_result", "tool_use_id": "call_1", "content": "done"}


def test_json_mode_appends_instruction():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok()

    make(handler).chat([{"role": "user", "content": "hi"}], json=True)
    assert "JSON" in seen["body"]["system"]


def test_no_retry_on_401_key_hidden():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401)

    with pytest.raises(ProviderError) as exc:
        make(handler).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1
    assert KEY not in str(exc.value)


def test_retry_on_529_overloaded():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(529)  # anthropic "overloaded"
        return _ok()

    make(handler, retries=2).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2
