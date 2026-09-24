import asyncio
import json

from _fakes import install_fake_anthropic

from touchstone import store
from touchstone.capture import context
from touchstone.capture.patch_anthropic import patch

RESPONSE = {
    "content": [
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "t1", "name": "order_status", "input": {"order_id": "A1"}},
    ],
    "usage": {"input_tokens": 9, "output_tokens": 4},
}

EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "tool_use", "id": "tu1", "name": "refund"}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "text_delta", "text": "Refunded "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "now"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "input_json_delta", "partial_json": '{"amt":'}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "input_json_delta", "partial_json": "5}"}},
    {"type": "message_delta", "usage": {"output_tokens": 6}},
]

FINAL = {
    "content": [
        {"type": "text", "text": "Refunded now"},
        {"type": "tool_use", "id": "tu1", "name": "refund", "input": {"amt": 5}},
    ],
    "usage": {"input_tokens": 11, "output_tokens": 6},
}


class FakeStreamMgr:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(EVENTS)

    def get_final_message(self):
        return FINAL


class Messages:
    def create(self, **kwargs):
        return iter(EVENTS) if kwargs.get("stream") else RESPONSE

    def stream(self, **kwargs):
        return FakeStreamMgr()


class AsyncMessages:
    async def create(self, **kwargs):
        return RESPONSE


def _last_span():
    conn = context.get_conn()
    eps = store.list_episodes(conn)
    return store.list_spans(conn, eps[-1].id)[-1]


def test_nonstream_extracts_text_tooluse_and_tokens(traced, monkeypatch):
    install_fake_anthropic(monkeypatch, Messages, AsyncMessages)
    assert patch() is True
    from anthropic.resources import messages as m

    m.Messages().create(model="claude-x", messages=[{"role": "user", "content": "hi"}])
    span = _last_span()
    msg = span.output["message"]
    assert msg["content"] == "hi"
    tc = msg["tool_calls"][0]
    assert tc["name"] == "order_status" and json.loads(tc["arguments"]) == {"order_id": "A1"}
    assert span.tokens_in == 9 and span.tokens_out == 4


def test_stream_create_accumulates(traced, monkeypatch):
    install_fake_anthropic(monkeypatch, Messages, AsyncMessages)
    patch()
    from anthropic.resources import messages as m

    events = list(m.Messages().create(
        model="claude-x", messages=[{"role": "user", "content": "q"}], stream=True))
    assert len(events) == len(EVENTS)
    msg = _last_span().output["message"]
    assert msg["content"] == "Refunded now"
    assert msg["tool_calls"][0]["arguments"] == '{"amt":5}'
    span = _last_span()
    assert span.tokens_in == 11 and span.tokens_out == 6


def test_stream_context_manager_records_on_exit(traced, monkeypatch):
    install_fake_anthropic(monkeypatch, Messages, AsyncMessages)
    patch()
    from anthropic.resources import messages as m

    collected = []
    with m.Messages().stream(model="claude-x", messages=[{"role": "user", "content": "q"}]) as s:
        for event in s:
            collected.append(event)
    assert len(collected) == len(EVENTS)
    msg = _last_span().output["message"]
    assert msg["content"] == "Refunded now"
    assert json.loads(msg["tool_calls"][0]["arguments"]) == {"amt": 5}


def test_async_create(traced, monkeypatch):
    install_fake_anthropic(monkeypatch, Messages, AsyncMessages)
    patch()
    from anthropic.resources import messages as m

    asyncio.run(m.AsyncMessages().create(model="c", messages=[{"role": "user", "content": "x"}]))
    assert _last_span().output["message"]["content"] == "hi"
