import asyncio

from _fakes import install_fake_openai

from touchstone import store
from touchstone.capture import context
from touchstone.capture.patch_openai import patch

RESPONSE = {
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "here you go"}]},
        {"type": "function_call", "call_id": "fc_1", "name": "lookup", "arguments": '{"q": 1}'},
        {"type": "function_call", "call_id": "fc_2", "name": "lookup", "arguments": '{"q": 2}'},
    ],
    "usage": {"input_tokens": 30, "output_tokens": 12,
              "input_tokens_details": {"cached_tokens": 10},
              "output_tokens_details": {"reasoning_tokens": 8}},
}

STREAM = [
    {"type": "response.output_text.delta", "delta": "Hel"},
    {"type": "response.output_text.delta", "delta": "lo"},
    {"type": "response.completed", "response": {
        "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2}}},
]

REFUSAL = {
    "status": "completed",
    "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I won't do that"}]}],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

INCOMPLETE = {
    "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
    "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "partial"}]}],
    "usage": {"input_tokens": 5, "output_tokens": 500},
}


def _responses(payload):
    class Responses:
        def create(self, **kwargs):
            if kwargs.get("stream"):
                return iter(STREAM)
            return payload

    class AsyncResponses:
        async def create(self, **kwargs):
            if kwargs.get("stream"):
                async def gen():
                    for ch in STREAM:
                        yield ch
                return gen()
            return payload

    return Responses, AsyncResponses


class _Chat:
    def create(self, **kwargs):
        return {}


def _install(monkeypatch, payload=RESPONSE):
    responses_cls, async_cls = _responses(payload)
    install_fake_openai(monkeypatch, _Chat, _Chat, responses_cls, async_cls)
    assert patch() is True
    from openai.resources import responses as r
    return r


def _last_span():
    conn = context.get_conn()
    eps = store.list_episodes(conn)
    return store.list_spans(conn, eps[-1].id)[-1]


def test_responses_message_reasoning_parallel_calls_and_usage(traced, monkeypatch):
    r = _install(monkeypatch)
    r.Responses().create(model="gpt-4o-mini", input="do it")

    span = _last_span()
    msg = span.output["message"]
    assert msg["content"] == "here you go"
    assert [c["id"] for c in msg["tool_calls"]] == ["fc_1", "fc_2"]  # parallel, distinct ids
    assert msg["reasoning"] == [{"type": "reasoning",
                                 "summary": [{"type": "summary_text", "text": "thinking..."}]}]
    assert span.output["stop_reason"] == "tool_calls"
    assert span.output["usage"]["cached_tokens"] == 10
    assert span.output["usage"]["reasoning_tokens"] == 8
    assert span.model == "gpt-4o-mini"


def test_responses_instructions_and_input_become_canonical_messages(traced, monkeypatch):
    r = _install(monkeypatch)
    r.Responses().create(model="gpt-4o-mini", instructions="be terse",
                         input=[{"type": "message", "role": "user",
                                 "content": [{"type": "input_text", "text": "hi"}]},
                                {"type": "function_call_output", "call_id": "x", "output": "42"}])
    stored = _last_span().input["messages"]
    assert stored[0] == {"role": "system", "content": "be terse"}
    assert stored[1] == {"role": "user", "content": "hi"}
    tool = next(m for m in stored if m["role"] == "tool")
    assert tool["tool_call_id"] == "x" and tool["content"] == "42"


def test_responses_streaming_prefers_completed_event(traced, monkeypatch):
    r = _install(monkeypatch)
    events = list(r.Responses().create(model="gpt-4o-mini", input="q", stream=True))
    assert len(events) == len(STREAM)
    span = _last_span()
    assert span.output["message"]["content"] == "Hello"
    assert span.tokens_in == 3 and span.tokens_out == 2


def test_responses_refusal(traced, monkeypatch):
    r = _install(monkeypatch, REFUSAL)
    r.Responses().create(model="gpt-4o-mini", input="bad")
    span = _last_span()
    assert span.output["message"]["refusal"] == "I won't do that"
    assert span.output["stop_reason"] == "refusal"


def test_responses_incomplete_maps_to_length(traced, monkeypatch):
    r = _install(monkeypatch, INCOMPLETE)
    r.Responses().create(model="gpt-4o-mini", input="long")
    assert _last_span().output["stop_reason"] == "length"


def test_responses_async(traced, monkeypatch):
    r = _install(monkeypatch)
    asyncio.run(r.AsyncResponses().create(model="gpt-4o-mini", input="do it"))
    assert _last_span().output["message"]["content"] == "here you go"
