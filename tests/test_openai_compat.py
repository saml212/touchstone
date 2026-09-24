import json

import httpx
import pytest

from touchstone.llm import ProviderError
from touchstone.llm.openai_compat import OpenAICompatProvider

KEY = "sk-secret-do-not-leak"


def make(handler, **kw):
    return OpenAICompatProvider(
        "https://api.openai.com/v1", "gpt-4o-mini", KEY,
        transport=httpx.MockTransport(handler), backoff=0, **kw,
    )


def _ok(content="hi", tool_calls=None, usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    usage = usage or {"prompt_tokens": 3, "completion_tokens": 2}
    body = {"choices": [{"message": msg}], "usage": usage}
    return httpx.Response(200, json=body)


def test_success_parses_content_and_usage():
    p = make(lambda req: _ok("hello there"))
    r = p.chat([{"role": "user", "content": "hi"}])
    assert r.content == "hello there"
    assert r.usage == {"tokens_in": 3, "tokens_out": 2}


def test_tool_call_reply():
    tc = [{"id": "c1", "type": "function",
           "function": {"name": "refund", "arguments": '{"amount": 5}'}}]
    p = make(lambda req: _ok("", tool_calls=tc))
    r = p.chat([{"role": "user", "content": "refund"}])
    assert r.tool_calls == [{"id": "c1", "name": "refund", "arguments": '{"amount": 5}'}]


def test_json_mode_sets_response_format():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok('{"ok": true}')

    make(handler).chat([{"role": "user", "content": "hi"}], json=True)
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_canonical_context_becomes_valid_openai_wire():
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
    sent = seen["body"]["messages"]
    assistant = next(m for m in sent if m.get("tool_calls"))
    tc = assistant["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"] == {"name": "refund", "arguments": '{"amount": 5}'}
    tool = next(m for m in sent if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_1"


def test_tools_forwarded():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _ok()

    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    make(handler).chat([{"role": "user", "content": "x"}], tools=tools)
    assert seen["body"]["tools"] == tools


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return _ok("recovered")

    r = make(handler).chat([{"role": "user", "content": "hi"}])
    assert r.content == "recovered"
    assert calls["n"] == 2


def test_retries_exhausted_on_persistent_500():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(ProviderError, match="HTTP 503"):
        make(handler, retries=2).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3  # first try + 2 retries


def test_no_retry_on_401_and_key_not_leaked():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(ProviderError) as exc:
        make(handler).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # auth errors never retry
    assert KEY not in str(exc.value)


def test_timeout_retries_then_raises():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(ProviderError, match="timed out"):
        make(handler, retries=2).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3


def test_transport_error_message_has_no_key():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as exc:
        make(handler, retries=0).chat([{"role": "user", "content": "hi"}])
    assert KEY not in str(exc.value)


def test_malformed_json_response():
    def handler(req):
        return httpx.Response(200, content=b"not json")

    with pytest.raises(ProviderError, match="non-JSON"):
        make(handler).chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_async_success():
    p = make(lambda req: _ok("async-hi"))
    r = await p.achat([{"role": "user", "content": "hi"}])
    assert r.content == "async-hi"


def test_authorization_header_present():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("Authorization")
        return _ok()

    make(handler).chat([{"role": "user", "content": "hi"}])
    assert seen["auth"] == f"Bearer {KEY}"
