"""Patch anthropic messages (sync + async, non-streaming create, streaming create, and the
`messages.stream()` context manager) into one span each.

Content blocks and streaming events are read defensively; tool_use inputs are serialized to
argument strings; usage tokens and errors are recorded on the span.
"""

from __future__ import annotations

import functools
import json

from ..messages import canonical
from . import context


def _get(obj, key):
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _extract(message):
    parts, tool_calls = [], []
    for block in _get(message, "content") or []:
        btype = _get(block, "type")
        if btype == "text":
            parts.append(_get(block, "text") or "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": _get(block, "id"),
                "name": _get(block, "name"),
                "arguments": _as_str(_get(block, "input")),
            })
    u = _get(message, "usage")
    usage = None
    if u is not None:
        usage = {"tokens_in": _get(u, "input_tokens"), "tokens_out": _get(u, "output_tokens")}
    return "".join(parts), tool_calls, usage


def _accumulate(event, st):
    et = _get(event, "type")
    if et == "message_start":
        u = _get(_get(event, "message"), "usage")
        if u:
            st["tin"] = _get(u, "input_tokens")
    elif et == "content_block_start":
        idx = _get(event, "index") or 0
        cb = _get(event, "content_block")
        if _get(cb, "type") == "tool_use":
            st["meta"][idx] = {"id": _get(cb, "id"), "name": _get(cb, "name")}
    elif et == "content_block_delta":
        idx = _get(event, "index") or 0
        d = _get(event, "delta")
        dt = _get(d, "type")
        if dt == "text_delta":
            st["parts"].append(_get(d, "text") or "")
        elif dt == "input_json_delta":
            st["args"][idx] = st["args"].get(idx, "") + (_get(d, "partial_json") or "")
    elif et == "message_delta":
        u = _get(event, "usage")
        if u and _get(u, "output_tokens") is not None:
            st["tout"] = _get(u, "output_tokens")


def _state():
    return {"parts": [], "args": {}, "meta": {}, "tin": None, "tout": None}


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


def _record(model, messages, tools, params, content, tool_calls, usage, error, started):
    tokens_in = usage.get("tokens_in") if usage else None
    tokens_out = usage.get("tokens_out") if usage else None
    reply = canonical([{"role": "assistant", "content": content, "tool_calls": tool_calls}])[0]
    context.add_span(
        "llm",
        model or "anthropic.messages",
        model=model,
        input={
            "messages": canonical(messages),
            "tools": tools or [],
            "params": {k: context.jsonable(v) for k, v in params.items()},
        },
        output={"message": reply},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=error,
        started_at=started,
    )


def _split(kwargs):
    model = kwargs.get("model")
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")
    stream = kwargs.get("stream", False)
    params = {k: v for k, v in kwargs.items() if k not in ("messages", "tools", "model")}
    return model, messages, tools, stream, params


def _wrap_stream(resp, model, messages, tools, params, started):
    st = _state()
    try:
        for event in resp:
            _accumulate(event, st)
            yield event
    except Exception as exc:
        content, calls, usage = _finish(st)
        _record(model, messages, tools, params, content, calls, usage, repr(exc), started)
        raise
    content, calls, usage = _finish(st)
    _record(model, messages, tools, params, content, calls, usage, None, started)


async def _wrap_astream(resp, model, messages, tools, params, started):
    st = _state()
    try:
        async for event in resp:
            _accumulate(event, st)
            yield event
    except Exception as exc:
        content, calls, usage = _finish(st)
        _record(model, messages, tools, params, content, calls, usage, repr(exc), started)
        raise
    content, calls, usage = _finish(st)
    _record(model, messages, tools, params, content, calls, usage, None, started)


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
        _record(model, messages, tools, params, content, calls, usage, error, started)

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
        orig = m.Messages.create

        @functools.wraps(orig)
        def create(self, *args, **kwargs):
            from .. import store
            model, messages, tools, stream, params = _split(kwargs)
            started = store.now()
            try:
                resp = orig(self, *args, **kwargs)
            except Exception as exc:
                _record(model, messages, tools, params, "", [], None, repr(exc), started)
                raise
            if stream:
                return _wrap_stream(resp, model, messages, tools, params, started)
            content, calls, usage = _extract(resp)
            _record(model, messages, tools, params, content, calls, usage, None, started)
            return resp

        create._touchstone = True
        m.Messages.create = create

    if hasattr(m.Messages, "stream") and not getattr(m.Messages.stream, "_touchstone", False):
        sorig = m.Messages.stream

        @functools.wraps(sorig)
        def stream(self, *args, **kwargs):
            from .. import store
            model, messages, tools, _, params = _split(kwargs)
            mgr = sorig(self, *args, **kwargs)
            return _StreamProxy(mgr, model, messages, tools, params, store.now())

        stream._touchstone = True
        m.Messages.stream = stream

    if not getattr(m.AsyncMessages.create, "_touchstone", False):
        aorig = m.AsyncMessages.create

        @functools.wraps(aorig)
        async def acreate(self, *args, **kwargs):
            from .. import store
            model, messages, tools, stream_flag, params = _split(kwargs)
            started = store.now()
            try:
                resp = await aorig(self, *args, **kwargs)
            except Exception as exc:
                _record(model, messages, tools, params, "", [], None, repr(exc), started)
                raise
            if stream_flag:
                return _wrap_astream(resp, model, messages, tools, params, started)
            content, calls, usage = _extract(resp)
            _record(model, messages, tools, params, content, calls, usage, None, started)
            return resp

        acreate._touchstone = True
        m.AsyncMessages.create = acreate

    return True
