"""Patch anthropic messages (sync + async, non-streaming create, streaming create, and the
`messages.stream()` context manager) into one span each.

Content blocks and streaming events are read defensively; tool_use inputs are serialized to
argument strings; usage tokens and errors are recorded on the span. The span lifecycle lives in
`spans`; this module only maps anthropic's wire shapes.
"""

from __future__ import annotations

import functools

from . import spans
from .spans import as_str, get

DEFAULT_NAME = "anthropic.messages"


def _extract(message):
    parts, tool_calls = [], []
    for block in get(message, "content") or []:
        btype = get(block, "type")
        if btype == "text":
            parts.append(get(block, "text") or "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": get(block, "id"),
                "name": get(block, "name"),
                "arguments": as_str(get(block, "input")),
            })
    u = get(message, "usage")
    usage = None
    if u is not None:
        usage = {"tokens_in": get(u, "input_tokens"), "tokens_out": get(u, "output_tokens")}
    return "".join(parts), tool_calls, usage


def _state():
    return {"parts": [], "args": {}, "meta": {}, "tin": None, "tout": None}


def _accumulate(event, st):
    et = get(event, "type")
    if et == "message_start":
        u = get(get(event, "message"), "usage")
        if u:
            st["tin"] = get(u, "input_tokens")
    elif et == "content_block_start":
        idx = get(event, "index") or 0
        cb = get(event, "content_block")
        if get(cb, "type") == "tool_use":
            st["meta"][idx] = {"id": get(cb, "id"), "name": get(cb, "name")}
    elif et == "content_block_delta":
        idx = get(event, "index") or 0
        d = get(event, "delta")
        dt = get(d, "type")
        if dt == "text_delta":
            st["parts"].append(get(d, "text") or "")
        elif dt == "input_json_delta":
            st["args"][idx] = st["args"].get(idx, "") + (get(d, "partial_json") or "")
    elif et == "message_delta":
        u = get(event, "usage")
        if u and get(u, "output_tokens") is not None:
            st["tout"] = get(u, "output_tokens")


def _finish(st):
    content = "".join(st["parts"])
    calls = []
    for idx in sorted(set(st["args"]) | set(st["meta"])):
        meta = st["meta"].get(idx, {})
        calls.append(
            {"id": meta.get("id"), "name": meta.get("name"), "arguments": st["args"].get(idx, "")}
        )
    usage = None
    if st["tin"] is not None or st["tout"] is not None:
        usage = {"tokens_in": st["tin"], "tokens_out": st["tout"]}
    return content, calls, usage


_wrap_stream, _wrap_astream = spans.stream_wrappers(_state, _accumulate, _finish)


class _StreamProxy:
    """Wraps the object returned by `messages.stream(...)`; records the span on context exit,
    preferring `get_final_message()` and falling back to accumulated event deltas."""

    def __init__(self, mgr, model, messages, tools, params, started):
        self._mgr = mgr
        self._info = (model, messages, tools, params, started)
        self._st = _state()
        self._inner = None

    def __enter__(self):
        self._inner = self._mgr.__enter__()
        return self

    def __iter__(self):
        for event in self._inner:
            _accumulate(event, self._st)
            yield event

    def _record_final(self, error):
        model, messages, tools, params, started = self._info
        content, calls, usage = _finish(self._st)
        get_final = getattr(self._inner, "get_final_message", None)
        if get_final is not None:
            try:
                fc, ftc, fu = _extract(get_final())
                content = fc or content
                calls = ftc or calls
                usage = fu or usage
            except Exception:
                pass
        spans.record(DEFAULT_NAME, model, messages, tools, params,
                     content, calls, usage, error, started)

    def __exit__(self, exc_type, exc, tb):
        self._record_final(repr(exc) if exc else None)
        return self._mgr.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._inner if self._inner is not None else self._mgr, name)


def patch() -> bool:
    try:
        from anthropic.resources import messages as m
    except Exception:
        return False

    if not getattr(m.Messages.create, "_touchstone", False):
        m.Messages.create = spans.instrument_create(
            m.Messages.create, DEFAULT_NAME, _extract, _wrap_stream
        )
    if hasattr(m.Messages, "stream") and not getattr(m.Messages.stream, "_touchstone", False):
        sorig = m.Messages.stream

        @functools.wraps(sorig)
        def stream(self, *args, **kwargs):
            from .. import store
            model, messages, tools, _, params = spans.split(kwargs)
            mgr = sorig(self, *args, **kwargs)
            return _StreamProxy(mgr, model, messages, tools, params, store.now())

        stream._touchstone = True
        m.Messages.stream = stream
    if not getattr(m.AsyncMessages.create, "_touchstone", False):
        m.AsyncMessages.create = spans.instrument_acreate(
            m.AsyncMessages.create, DEFAULT_NAME, _extract, _wrap_astream
        )
    return True
