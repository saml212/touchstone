import asyncio

from _fakes import install_fake_openai

from touchstone import store
from touchstone.capture import context
from touchstone.capture.patch_openai import patch

NON_JSON_ARGS = 'order_id=A1, note="broke"'

RESPONSE = {
    "choices": [{
        "message": {
            "content": "here you go",
            "tool_calls": [
                {"id": "c1", "function": {"name": "order_status", "arguments": NON_JSON_ARGS}}
            ],
        }
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}

STREAM = [
    {"choices": [{"delta": {"content": "Hel"}}]},
    {"choices": [{"delta": {"content": "lo"}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c9", "function": {"name": "refund", "arguments": '{"amt":'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "5}"}}]}}]},
    {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
]


class Completions:
    def create(self, **kwargs):
        return iter(STREAM) if kwargs.get("stream") else RESPONSE


class AsyncCompletions:
    async def create(self, **kwargs):
        if kwargs.get("stream"):
            async def gen():
                for ch in STREAM:
                    yield ch
            return gen()
        return RESPONSE


def _last_span(traced):
    conn = context.get_conn()
    eps = store.list_episodes(conn)
    return store.list_spans(conn, eps[-1].id)[-1]


def test_nonstream_records_content_tokens_and_nonjson_tool_args(traced, monkeypatch):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    assert patch() is True
    from openai.resources.chat import completions as c

    resp = c.Completions().create(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    assert resp is RESPONSE  # patch is transparent to the caller

    span = _last_span(traced)
    msg = span.output["message"]
    assert msg["content"] == "here you go"
    assert msg["tool_calls"][0]["arguments"] == NON_JSON_ARGS
    assert span.tokens_in == 12 and span.tokens_out == 7
    assert span.model == "gpt-x"


def test_streaming_accumulates_into_one_span(traced, monkeypatch):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c

    stream = c.Completions().create(
        model="gpt-x", messages=[{"role": "user", "content": "q"}], stream=True
    )
    chunks = list(stream)
    assert len(chunks) == len(STREAM)

    span = _last_span(traced)
    assert span.output["message"]["content"] == "Hello"
    tc = span.output["message"]["tool_calls"][0]
    assert tc["name"] == "refund" and tc["arguments"] == '{"amt":5}' and tc["id"] == "c9"
    assert span.tokens_in == 3 and span.tokens_out == 4


def test_error_is_recorded_on_span(traced, monkeypatch):
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("upstream down")

    install_fake_openai(monkeypatch, Boom, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c

    try:
        c.Completions().create(model="m", messages=[{"role": "user", "content": "x"}])
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert "upstream down" in _last_span(traced).error


def test_stored_input_messages_are_canonical(traced, monkeypatch):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c

    wire_context = [
        {"role": "user", "content": "refund order 1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "refund", "arguments": '{"amount": 5}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
    c.Completions().create(model="gpt-x", messages=wire_context)

    stored = _last_span(traced).input["messages"]
    assistant = next(m for m in stored if m.get("tool_calls"))
    assert assistant["tool_calls"] == [
        {"id": "c1", "name": "refund", "arguments": '{"amount": 5}'}]
    assert "function" not in assistant["tool_calls"][0]
    tool = next(m for m in stored if m["role"] == "tool")
    assert tool["tool_call_id"] == "c1" and tool["content"] == "done"


def test_async_streaming(traced, monkeypatch):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c

    async def scenario():
        stream = await c.AsyncCompletions().create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True
        )
        return [ch async for ch in stream]

    chunks = asyncio.run(scenario())
    assert len(chunks) == len(STREAM)
    assert _last_span(traced).output["message"]["content"] == "Hello"


def test_stop_reason_cached_tokens_and_cost_recorded(traced, monkeypatch):
    resp = {
        "choices": [{"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                  "prompt_tokens_details": {"cached_tokens": 40},
                  "completion_tokens_details": {"reasoning_tokens": 5}},
    }

    class C:
        def create(self, **kwargs):
            return resp

    install_fake_openai(monkeypatch, C, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c
    c.Completions().create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    span = _last_span(traced)
    assert span.output["stop_reason"] == "stop"
    assert span.output["usage"]["cached_tokens"] == 40
    assert span.output["usage"]["reasoning_tokens"] == 5
    assert span.cost_usd == 100 / 1e6 * 0.15 + 20 / 1e6 * 0.60


def test_refusal_recorded_with_stop_reason(traced, monkeypatch):
    resp = {"choices": [{"message": {"content": None, "refusal": "I can't help with that"},
                         "finish_reason": "stop"}]}

    class C:
        def create(self, **kwargs):
            return resp

    install_fake_openai(monkeypatch, C, AsyncCompletions)
    patch()
    from openai.resources.chat import completions as c
    c.Completions().create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    span = _last_span(traced)
    assert span.output["message"]["refusal"] == "I can't help with that"
    assert span.output["stop_reason"] == "refusal"  # refusal overrides finish_reason


def test_capture_failure_never_breaks_the_call(traced, monkeypatch, caplog):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    patch()
    from touchstone.capture import spans

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(spans, "canonical", _boom)
    from openai.resources.chat import completions as c

    resp = c.Completions().create(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    assert resp is RESPONSE  # the user's app still gets its reply
    assert any("touchstone capture failed" in r.message for r in caplog.records)
