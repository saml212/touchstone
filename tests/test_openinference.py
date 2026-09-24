"""OpenInference ingest, exercised with hand-built ReadableSpan-like objects (no real OTel)."""

from touchstone import store
from touchstone.capture import context
from touchstone.capture.openinference import OpenInferenceSpanExporter


class _Ctx:
    def __init__(self, trace_id, span_id):
        self.trace_id, self.span_id = trace_id, span_id


class _Parent:
    def __init__(self, span_id):
        self.span_id = span_id


class _Span:
    def __init__(self, name, attributes, trace_id, span_id, parent=None):
        self.name = name
        self.attributes = attributes
        self.context = _Ctx(trace_id, span_id)
        self.parent = _Parent(parent) if parent else None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000


CHAIN = _Span("agent-run", {"openinference.span.kind": "CHAIN"}, trace_id=1, span_id=10)

LLM = _Span("chat", {
    "openinference.span.kind": "LLM",
    "llm.model_name": "gpt-4o-mini",
    "llm.input_messages.0.message.role": "user",
    "llm.input_messages.0.message.content": "refund order 1",
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.content": "",
    "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "refund",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"amount": 5}',
    "llm.token_count.prompt": 30,
    "llm.token_count.completion": 8,
    "llm.token_count.prompt_details.cache_read": 12,
}, trace_id=1, span_id=11, parent=10)

TOOL = _Span("refund", {
    "openinference.span.kind": "TOOL",
    "tool.name": "refund",
    "tool.parameters": '{"amount": 5}',
    "tool_call.id": "call_1",
    "output.value": "refunded",
}, trace_id=1, span_id=12, parent=11)


def test_ingest_llm_tool_under_one_episode(traced):
    conn = context.get_conn()
    OpenInferenceSpanExporter().export([CHAIN, LLM, TOOL])

    episodes = [e for e in store.list_episodes(conn) if e.source == "openinference"]
    assert len(episodes) == 1 and episodes[0].name == "agent-run"  # the root CHAIN names it
    spans = store.list_spans(conn, episodes[0].id)
    kinds = {s.kind for s in spans}
    assert kinds == {"model", "tool"}  # the CHAIN itself is not a span

    model = next(s for s in spans if s.kind == "model")
    assert model.model == "gpt-4o-mini"
    assert model.input["messages"][0] == {"role": "user", "content": "refund order 1"}
    reply = model.output["message"]
    assert reply["tool_calls"] == [{"id": "call_1", "name": "refund", "arguments": '{"amount": 5}'}]
    assert model.tokens_in == 30 and model.tokens_out == 8
    assert model.output["usage"]["cached_tokens"] == 12

    tool = next(s for s in spans if s.kind == "tool")
    assert tool.tool_call_id == "call_1"
    assert tool.output["result"] == "refunded"
    assert tool.parent_id == model.id  # parent/child preserved within the trace


def test_ingest_is_resilient_to_a_bad_span(traced):
    conn = context.get_conn()
    bad = _Span("x", None, trace_id=2, span_id=1)  # attributes None
    OpenInferenceSpanExporter().export([bad, LLM])  # must not raise
    assert any(e.source == "openinference" for e in store.list_episodes(conn))
