"""Patch anthropic messages (sync + async, non-streaming create, streaming create, and the
`messages.stream()` context manager) into one span each.

Content blocks and streaming events are read defensively; tool_use inputs are serialized to
argument strings; usage tokens and errors are recorded on the span. The span lifecycle lives in
`spans`; this module only maps anthropic's wire shapes.
"""

from __future__ import annotations

import functools

from . import spans
from .spans import as_str, get, model_result

DEFAULT_NAME = "anthropic.messages"


def _usage(u):
    if u is None:
        return None
    usage = {"tokens_in": get(u, "input_tokens"), "tokens_out": get(u, "output_tokens")}
    read = get(u, "cache_read_input_tokens")
    if read is not None:
        usage["cached_tokens"] = read
    creation = get(u, "cache_creation_input_tokens")
    if creation is not None:
        usage["cache_creation_tokens"] = creation
    return usage


def _reasoning_block(block) -> dict:
    if get(block, "type") == "redacted_thinking":
        return {"type": "redacted_thinking", "data": get(block, "data")}
    return {"type": "thinking", "thinking": get(block, "thinking") or "",
            "signature": get(block, "signature")}


def _extract(message):
    parts, tool_calls, reasoning = [], [], []
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
        elif btype in ("thinking", "redacted_thinking"):
            reasoning.append(_reasoning_block(block))
    return model_result("".join(parts), tool_calls, _usage(get(message, "usage")),
                        stop_reason=get(message, "stop_reason"), reasoning=reasoning or None)


def _state():
    return {"parts": [], "args": {}, "meta": {}, "reasoning": {},
            "tin": None, "tout": None, "cache_read": None, "cache_create": None, "stop": None}


def _on_message_start(event, st):
    u = get(get(event, "message"), "usage")
    if u:
        st["tin"] = get(u, "input_tokens")
        st["cache_read"] = get(u, "cache_read_input_tokens")
        st["cache_create"] = get(u, "cache_creation_input_tokens")


def _on_block_start(event, st):
    idx = get(event, "index") or 0
    cb = get(event, "content_block")
    ctype = get(cb, "type")
    if ctype == "tool_use":
        st["meta"][idx] = {"id": get(cb, "id"), "name": get(cb, "name")}
    elif ctype in ("thinking", "redacted_thinking"):
        st["reasoning"][idx] = _reasoning_block(cb)


def _on_block_delta(event, st):
    idx = get(event, "index") or 0
    d = get(event, "delta")
    dt = get(d, "type")
    if dt == "text_delta":
        st["parts"].append(get(d, "text") or "")
    elif dt == "input_json_delta":
        st["args"][idx] = st["args"].get(idx, "") + (get(d, "partial_json") or "")
    elif dt == "thinking_delta":
        block = st["reasoning"].setdefault(idx, {"type": "thinking", "thinking": ""})
        block["thinking"] = (block.get("thinking") or "") + (get(d, "thinking") or "")
    elif dt == "signature_delta":
        block = st["reasoning"].setdefault(idx, {"type": "thinking", "thinking": ""})
        block["signature"] = get(d, "signature")


def _on_message_delta(event, st):
    u = get(event, "usage")
    if u and get(u, "output_tokens") is not None:
        st["tout"] = get(u, "output_tokens")
    stop = get(get(event, "delta"), "stop_reason")
    if stop:
        st["stop"] = stop


_EVENT_HANDLERS = {
    "message_start": _on_message_start,
    "content_block_start": _on_block_start,
    "content_block_delta": _on_block_delta,
    "message_delta": _on_message_delta,
}


def _accumulate(event, st):
    handler = _EVENT_HANDLERS.get(get(event, "type"))
    if handler:
        handler(event, st)


def _finish(st):
    calls = []
    for idx in sorted(set(st["args"]) | set(st["meta"])):
        meta = st["meta"].get(idx, {})
        calls.append(
            {"id": meta.get("id"), "name": meta.get("name"), "arguments": st["args"].get(idx, "")}
        )
    usage = None
    if any(st[k] is not None for k in ("tin", "tout", "cache_read", "cache_create")):
        usage = {"tokens_in": st["tin"], "tokens_out": st["tout"]}
        if st["cache_read"] is not None:
            usage["cached_tokens"] = st["cache_read"]
        if st["cache_create"] is not None:
            usage["cache_creation_tokens"] = st["cache_create"]
    reasoning = [st["reasoning"][i] for i in sorted(st["reasoning"])] or None
    return model_result("".join(st["parts"]), calls, usage,
                        stop_reason=st["stop"], reasoning=reasoning)


def _prefer(final: dict, acc: dict) -> dict:
    """The final message's values where present, else what the stream accumulated."""
    out = {k: final.get(k) or acc.get(k) for k in
           ("content", "tool_calls", "usage", "stop_reason", "reasoning")}
    refusal = final.get("refusal")
    out["refusal"] = refusal if refusal is not None else acc.get("refusal")
    return out


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
        try:
            result = _finish(self._st)
            get_final = getattr(self._inner, "get_final_message", None)
            if get_final is not None:
                result = _prefer(_extract(get_final()), result)
            spans.record(DEFAULT_NAME, model, messages, tools, params, result, error, started)
        except Exception as exc:
            spans._log.warning("touchstone capture failed: %r", exc)

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
