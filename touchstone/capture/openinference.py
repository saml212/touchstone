"""Optional ingest of OpenInference-attributed OpenTelemetry spans into the same tables.

A team already instrumented with OpenInference (Arize) auto-instrumentors can point Touchstone at
their traces with `touchstone.trace(otel=True)`: this registers an in-process OTel `SpanExporter`
that normalizes each span's flattened `openinference.*` attributes into canonical messages and
writes model / tool spans under one episode per trace. No collector process is involved. Requires
the `touchstone[otel]` extra (opentelemetry-sdk); the module imports without it for testing against
hand-built ReadableSpan-like objects.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .. import store
from ..messages import _args_str, canonical
from . import context


def _base_exporter():
    try:
        from opentelemetry.sdk.trace.export import SpanExporter

        return SpanExporter
    except Exception:
        return object


def _iso(ns) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).isoformat() if ns else store.now()


def _indexed(attrs: dict, prefix: str) -> dict[int, dict]:
    """Group `prefix.{i}.<tail>` attributes by index i."""
    out: dict[int, dict] = {}
    for key, val in attrs.items():
        if not key.startswith(prefix + "."):
            continue
        i, _, tail = key[len(prefix) + 1:].partition(".")
        if i.isdigit():
            out.setdefault(int(i), {})[tail] = val
    return out


def _tool_calls(fields: dict) -> list[dict]:
    by_j = _indexed(fields, "message.tool_calls")
    calls = []
    for j in sorted(by_j):
        f = by_j[j]
        calls.append({"id": f.get("tool_call.id"),
                      "name": f.get("tool_call.function.name"),
                      "arguments": _args_str(f.get("tool_call.function.arguments"))})
    return calls


def _message(fields: dict) -> dict:
    msg = {"role": fields.get("message.role", "user"),
           "content": fields.get("message.content", "") or ""}
    calls = _tool_calls(fields)
    if calls:
        msg["tool_calls"] = calls
    if fields.get("message.tool_call_id"):
        msg["tool_call_id"] = fields["message.tool_call_id"]
    return msg


def _messages(attrs: dict, prefix: str) -> list[dict]:
    by_i = _indexed(attrs, prefix)
    return canonical([_message(by_i[i]) for i in sorted(by_i)])


class OpenInferenceSpanExporter(_base_exporter()):
    """Writes OpenInference OTel spans into Touchstone's SQLite, one episode per trace."""

    def __init__(self):
        self._episodes: dict[int, str] = {}
        self._span_ids: dict[int, str] = {}

    def export(self, spans):
        for span in spans:
            try:
                self._ingest(span)
            except Exception:  # ingest must never break the app's tracer pipeline
                pass
        return _success()

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _episode_for(self, conn, span) -> str:
        trace_id = span.context.trace_id
        ep_id = self._episodes.get(trace_id)
        if ep_id is None:
            name = span.name if span.parent is None else "openinference"
            ep_id = store.insert_episode(
                conn, store.Episode(name=name, source="openinference")).id
            self._episodes[trace_id] = ep_id
        return ep_id

    def _parent_id(self, span) -> str | None:
        return self._span_ids.get(span.parent.span_id) if span.parent else None

    def _ingest(self, span) -> None:
        attrs = dict(span.attributes or {})
        kind = attrs.get("openinference.span.kind")
        conn = context.get_conn()
        ep_id = self._episode_for(conn, span)
        if kind == "LLM":
            record = self._llm(ep_id, span, attrs)
        elif kind == "TOOL":
            record = self._tool(ep_id, span, attrs)
        else:
            return  # AGENT / CHAIN / RETRIEVER only seed the episode
        self._span_ids[span.context.span_id] = store.insert_span(conn, record).id

    def _llm(self, ep_id, span, attrs) -> store.Span:
        outputs = _messages(attrs, "llm.output_messages")
        reply = outputs[-1] if outputs else {"role": "assistant", "content": ""}
        usage = {}
        if attrs.get("llm.token_count.prompt_details.cache_read") is not None:
            usage["cached_tokens"] = attrs["llm.token_count.prompt_details.cache_read"]
        output = {"message": reply}
        if usage:
            output["usage"] = usage
        return store.Span(
            episode_id=ep_id, kind="model", name=span.name, parent_id=self._parent_id(span),
            model=attrs.get("llm.model_name"),
            input={"messages": _messages(attrs, "llm.input_messages"), "tools": [], "params": {}},
            output=output,
            tokens_in=attrs.get("llm.token_count.prompt"),
            tokens_out=attrs.get("llm.token_count.completion"),
            started_at=_iso(span.start_time), ended_at=_iso(span.end_time))

    def _tool(self, ep_id, span, attrs) -> store.Span:
        name = attrs.get("tool.name") or span.name
        return store.Span(
            episode_id=ep_id, kind="tool", name=name, parent_id=self._parent_id(span),
            input={"name": name, "arguments": attrs.get("tool.parameters")},
            output={"result": attrs.get("output.value")},
            tool_call_id=attrs.get("tool_call.id"),
            started_at=_iso(span.start_time), ended_at=_iso(span.end_time))


def _success():
    try:
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS
    except Exception:
        return None


def register() -> bool:
    """Register the exporter on the global TracerProvider. False if opentelemetry-sdk is absent."""
    try:
        from opentelemetry import trace as ot
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except Exception:
        return False
    provider = ot.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        ot.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(OpenInferenceSpanExporter()))
    return True
