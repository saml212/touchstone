"""trace() registers the litellm CustomLogger so a litellm.completion() call records one model span,
and the logger dedupes against a call that also passed through a patched openai/anthropic SDK."""

import litellm

import touchstone
from touchstone import store
from touchstone.capture import context
from touchstone.capture.litellm import TouchstoneLogger

TOOLS = [{"type": "function",
          "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


def _model_spans(conn):
    return [s for ep in store.list_episodes(conn)
            for s in store.list_spans(conn, ep.id) if s.kind == "model"]


def test_trace_registers_logger_and_records_one_span(traced):
    info = touchstone.trace(traced)
    assert "litellm" in info["patched"]
    loggers = [cb for cb in litellm.callbacks if isinstance(cb, TouchstoneLogger)]
    assert len(loggers) == 1  # registered exactly once, even across repeated trace() calls

    # A real litellm ModelResponse from mock_response, recorded through the registered logger.
    # (litellm routes a *sync* completion's callbacks through the current event loop when one
    # exists, e.g. pytest-asyncio's; the live smoke covers the auto-fire path in a plain process.)
    resp = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
                              tools=TOOLS, mock_response="ok")
    kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}],
              "optional_params": {"tools": TOOLS}}
    loggers[0].log_pre_api_call("gpt-4o-mini", kwargs["messages"], kwargs)
    loggers[0].log_success_event(kwargs, resp, 0, 0)

    conn = context.get_conn()
    spans = _model_spans(conn)
    assert len(spans) == 1
    span = spans[0]
    assert span.input["tools"] == TOOLS
    assert span.input["messages"][0]["content"] == "hi"
    assert span.output["message"]["content"] == "ok"


def test_logger_dedupes_when_sdk_already_recorded(traced):
    logger = TouchstoneLogger()
    kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}],
              "optional_params": {"tools": TOOLS}}
    resp = {"choices": [{"message": {"content": "ok"}}]}

    logger.log_pre_api_call("gpt-4o-mini", kwargs["messages"], kwargs)
    context.note_sdk_span()  # a patched SDK recorded this call inside litellm
    logger.log_success_event(kwargs, resp, 0, 0)
    conn = context.get_conn()
    assert _model_spans(conn) == []  # logger skipped; the SDK span is the only record

    logger.log_pre_api_call("gpt-4o-mini", kwargs["messages"], kwargs)  # next call: no SDK span
    logger.log_success_event(kwargs, resp, 0, 0)
    assert len(_model_spans(conn)) == 1
