"""The capture fidelity matrix from scratchpad/research-capture.md, as parametrized cases.

Each row drives a fake SDK (or a canonical payload for shapes no SDK call produces) and asserts the
stored span's messages, tool-call ids, stop_reason and usage, plus the invariant that canonicalizing
the stored messages is a fixed point: canonical(canonical(x)) == canonical(x).
"""

import asyncio

import pytest
from _fakes import install_fake_anthropic, install_fake_openai

from touchstone import store
from touchstone.capture import context
from touchstone.capture.patch_anthropic import patch as patch_anthropic
from touchstone.capture.patch_openai import patch as patch_openai
from touchstone.messages import canonical


def _last_span():
    conn = context.get_conn()
    return store.list_spans(conn, store.list_episodes(conn)[-1].id)[-1]


def _sync(cls):
    class Async:
        async def create(self, **kwargs):
            inner = cls().create(**kwargs)
            if kwargs.get("stream"):
                async def gen():
                    for ch in inner:
                        yield ch
                return gen()
            return inner
    return Async


# ---- drivers ----------------------------------------------------------------

def drive_openai(monkeypatch, payload, *, stream=False, is_async=False, msgs=None):
    class C:
        def create(self, **kwargs):
            return iter(payload) if kwargs.get("stream") else payload
    install_fake_openai(monkeypatch, C, _sync(C))
    patch_openai()
    from openai.resources.chat import completions as c
    kw = {"model": "gpt-4o-mini", "messages": msgs or [{"role": "user", "content": "hi"}]}
    if stream:
        kw["stream"] = True
    if is_async:
        async def run():
            r = await c.AsyncCompletions().create(**kw)
            if stream:
                return [x async for x in r]
        asyncio.run(run())
    else:
        r = c.Completions().create(**kw)
        if stream:
            list(r)
    return _last_span()


def drive_anthropic(monkeypatch, *, response=None, events=None, final=None):
    class M:
        def create(self, **kwargs):
            return iter(events) if kwargs.get("stream") else response

        def stream(self, **kwargs):
            return _AnthStream(events, final)
    install_fake_anthropic(monkeypatch, M, _sync(M))
    patch_anthropic()
    from anthropic.resources import messages as m
    q = [{"role": "user", "content": "q"}]
    if events is not None and final is not None:
        with m.Messages().stream(model="claude-x", messages=q) as s:
            list(s)
    elif events is not None:
        list(m.Messages().create(model="claude-x", messages=[{"role": "user", "content": "q"}],
                                 stream=True))
    else:
        m.Messages().create(model="claude-x", messages=[{"role": "user", "content": "hi"}])
    return _last_span()


class _AnthStream:
    def __init__(self, events, final):
        self._events, self._final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


def drive_responses(monkeypatch, payload, *, stream=False):
    class R:
        def create(self, **kwargs):
            return iter(payload) if kwargs.get("stream") else payload

    class Chat:
        def create(self, **kwargs):
            return {}
    install_fake_openai(monkeypatch, Chat, _sync(Chat), R, _sync(R))
    patch_openai()
    from openai.resources import responses as r
    if stream:
        list(r.Responses().create(model="gpt-4o-mini", input="q", stream=True))
    else:
        r.Responses().create(model="gpt-4o-mini", input="do it")
    return _last_span()


# ---- matrix -----------------------------------------------------------------

def _openai_msg(content="", tool_calls=None, refusal=None, finish="stop"):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if refusal is not None:
        msg["refusal"] = refusal
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def row_01(mp):  # plain OpenAI chat text
    span = drive_openai(mp, _openai_msg("hello"))
    assert span.output["message"]["content"] == "hello"
    assert span.output["stop_reason"] == "stop"
    assert span.tokens_in == 5 and span.tokens_out == 2
    return span


def row_02(mp):  # plain Anthropic text
    span = drive_anthropic(mp, response={
        "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1}})
    assert span.output["message"]["content"] == "hi"
    assert span.output["stop_reason"] == "stop"  # end_turn -> stop
    return span


def row_03(mp):  # Responses API
    span = drive_responses(mp, {
        "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "responded"}]}],
        "usage": {"input_tokens": 4, "output_tokens": 2}})
    assert span.output["message"]["content"] == "responded"
    assert span.tokens_in == 4 and span.tokens_out == 2
    return span


def row_04(mp):  # single tool call linked by id
    span = drive_openai(mp, _openai_msg(tool_calls=[
        {"id": "t1", "function": {"name": "refund", "arguments": "{}"}}], finish="tool_calls"))
    tc = span.output["message"]["tool_calls"][0]
    assert tc["id"] == "t1" and span.output["stop_reason"] == "tool_calls"
    return span


def row_05(mp):  # parallel tool calls, same name, distinct ids
    span = drive_openai(mp, _openai_msg(tool_calls=[
        {"id": "a", "function": {"name": "lookup", "arguments": '{"q":1}'}},
        {"id": "b", "function": {"name": "lookup", "arguments": '{"q":2}'}}], finish="tool_calls"))
    assert [c["id"] for c in span.output["message"]["tool_calls"]] == ["a", "b"]
    return span


def row_06(mp):  # streaming tool-call argument deltas, id from first delta
    stream = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c9", "function": {"name": "refund", "arguments": '{"amt":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "5}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]
    span = drive_openai(mp, stream, stream=True)
    tc = span.output["message"]["tool_calls"][0]
    assert tc["id"] == "c9" and tc["arguments"] == '{"amt":5}'
    assert span.output["stop_reason"] == "tool_calls"
    return span


def row_07(mp):  # streaming text deltas + end-of-stream usage
    stream = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
    ]
    span = drive_openai(mp, stream, stream=True)
    assert span.output["message"]["content"] == "Hello"
    assert span.tokens_in == 2 and span.tokens_out == 1
    return span


def row_08(mp):  # Anthropic thinking blocks
    span = drive_anthropic(mp, response={
        "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "answer"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
    assert span.output["message"]["reasoning"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "s"}]
    return span


def row_09(mp):  # OpenAI reasoning tokens (Responses)
    span = drive_responses(mp, {
        "status": "completed",
        "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "s"}]},
                   {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 5, "output_tokens": 9,
                  "output_tokens_details": {"reasoning_tokens": 7}}})
    assert span.output["usage"]["reasoning_tokens"] == 7
    assert span.output["message"]["reasoning"][0]["type"] == "reasoning"
    return span


def row_10(mp):  # refusal
    span = drive_openai(mp, _openai_msg(content=None, refusal="no"))
    assert span.output["message"]["refusal"] == "no"
    assert span.output["stop_reason"] == "refusal"
    return span


def row_11(mp):  # cached prompt tokens
    payload = _openai_msg("ok")
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 3}
    span = drive_openai(mp, payload)
    assert span.output["usage"]["cached_tokens"] == 3
    return span


def row_11b(mp):  # Anthropic cache read + creation
    span = drive_anthropic(mp, response={
        "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1,
                  "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2}})
    assert span.output["usage"]["cached_tokens"] == 4
    assert span.output["usage"]["cache_creation_tokens"] == 2
    return span


def row_13(mp):  # provider call error mid-flight is recorded, then re-raised
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("429 rate limited")
    install_fake_openai(mp, Boom, _sync(Boom))
    patch_openai()
    from openai.resources.chat import completions as c
    with pytest.raises(RuntimeError):
        c.Completions().create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    span = _last_span()
    assert "429 rate limited" in span.error
    return span


def row_14(mp):  # images in input are preserved as parts, not flattened
    span = drive_openai(mp, _openai_msg("desc"), msgs=[
        {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}}]}])
    stored = span.input["messages"][0]["content"]
    assert isinstance(stored, list) and stored[1]["type"] == "image"
    return span


def row_16(mp):  # structured output content preserved as a string
    span = drive_openai(mp, _openai_msg('{"a": 1}'))
    assert span.output["message"]["content"] == '{"a": 1}'
    return span


def row_19(mp):  # async has the identical span shape
    span = drive_openai(mp, _openai_msg("async-ok"), is_async=True)
    assert span.output["message"]["content"] == "async-ok"
    return span


def row_20(mp):  # Anthropic messages.stream() context manager
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "streamed"}},
        {"type": "message_delta", "usage": {"output_tokens": 1},
         "delta": {"stop_reason": "end_turn"}},
    ]
    final = {"content": [{"type": "text", "text": "streamed"}], "stop_reason": "end_turn",
             "usage": {"input_tokens": 2, "output_tokens": 1}}
    span = drive_anthropic(mp, events=events, final=final)
    assert span.output["message"]["content"] == "streamed"
    assert span.output["stop_reason"] == "stop"
    return span


def row_21(mp):  # function-call-only turn keeps empty content + tool_calls
    span = drive_openai(mp, _openai_msg(content="", tool_calls=[
        {"id": "x", "function": {"name": "t", "arguments": "{}"}}], finish="tool_calls"))
    msg = span.output["message"]
    assert msg["content"] == "" and msg["tool_calls"][0]["id"] == "x"
    return span


def row_24(mp):  # per-call cost computed at capture for a known model
    span = drive_openai(mp, _openai_msg("ok"))
    assert span.cost_usd == 5 / 1e6 * 0.15 + 2 / 1e6 * 0.60
    return span


def row_25(mp):  # malformed (non-JSON) tool arguments preserved verbatim
    span = drive_openai(mp, _openai_msg(tool_calls=[
        {"id": "m", "function": {"name": "t", "arguments": "order=A1, note=broke"}}],
        finish="tool_calls"))
    assert span.output["message"]["tool_calls"][0]["arguments"] == "order=A1, note=broke"
    return span


CAPTURE_ROWS = [
    row_01, row_02, row_03, row_04, row_05, row_06, row_07, row_08, row_09, row_10,
    row_11, row_11b, row_13, row_14, row_16, row_19, row_20, row_21, row_24, row_25,
]


@pytest.mark.parametrize("row", CAPTURE_ROWS, ids=[r.__name__ for r in CAPTURE_ROWS])
def test_capture_matrix(row, traced, monkeypatch):
    span = row(monkeypatch)
    # every stored message list is a canonical fixed point
    for messages in (span.input.get("messages", []), [span.output.get("message", {})]):
        assert canonical(messages) == messages


# ---- rows that no single SDK call produces: canonical fixed-point coverage --

IDEMPOTENT_SHAPES = {
    "audio_15": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "b64", "format": "wav"}}]}],
    "mcp_17": [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "mcp1", "name": "server.tool", "arguments": '{"x":1}'}]},
        {"role": "tool", "tool_call_id": "mcp1", "content": "done"}],
    "agents_handoff_18": [{"role": "assistant", "content": "handing off", "tool_calls": [
        {"id": "h1", "name": "transfer_to_billing", "arguments": "{}"}]}],
    "deterministic_23": [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "d1", "name": "route", "arguments": "{}"}]}],
    "responses_items": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call", "call_id": "f1", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "f1", "output": "ok"}],
    "refusal_shape": [{"role": "assistant", "content": "", "refusal": "no"}],
}


@pytest.mark.parametrize("shape", list(IDEMPOTENT_SHAPES), ids=list(IDEMPOTENT_SHAPES))
def test_canonical_is_a_fixed_point(shape):
    once = canonical(IDEMPOTENT_SHAPES[shape])
    assert canonical(once) == once
