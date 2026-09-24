"""Shared span lifecycle for the SDK patches.

The patches for openai and anthropic differ only in how they read their own wire shapes. Everything
else — reading defensively, splitting call kwargs, running one span per model call, accumulating a
stream, and recording output/usage/error — lives here. A patch supplies three shape-mapping
callbacks: `extract(resp) -> (content, tool_calls, usage)` for a whole response, plus a
`state`/`accumulate(chunk, state)`/`finish(state) -> (content, tool_calls, usage)` trio for streams.
"""

from __future__ import annotations

import functools
import json

from ..messages import canonical
from . import context


def get(obj, key):
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def split(kwargs):
    model = kwargs.get("model")
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")
    stream = kwargs.get("stream", False)
    params = {k: v for k, v in kwargs.items() if k not in ("messages", "tools", "model")}
    return model, messages, tools, stream, params


def record(default_name, model, messages, tools, params,
           content, tool_calls, usage, error, started):
    reply = canonical([{"role": "assistant", "content": content, "tool_calls": tool_calls}])[0]
    context.add_span(
        "llm",
        model or default_name,
        model=model,
        input={
            "messages": canonical(messages),
            "tools": tools or [],
            "params": {k: context.jsonable(v) for k, v in params.items()},
        },
        output={"message": reply},
        tokens_in=usage.get("tokens_in") if usage else None,
        tokens_out=usage.get("tokens_out") if usage else None,
        error=error,
        started_at=started,
    )


def _recorder(default_name, model, messages, tools, params, started):
    """A `(content, tool_calls, usage, error)` closure that writes this call's span."""
    def rec(content, tool_calls, usage, error):
        record(default_name, model, messages, tools, params,
               content, tool_calls, usage, error, started)
    return rec


def stream_wrappers(state_factory, accumulate, finish):
    """Build (sync, async) generator wrappers that accumulate a stream and record at its end."""
    def wrap(resp, rec):
        state = state_factory()
        try:
            for chunk in resp:
                accumulate(chunk, state)
                yield chunk
        except Exception as exc:
            rec(*finish(state), repr(exc))
            raise
        rec(*finish(state), None)

    async def awrap(resp, rec):
        state = state_factory()
        try:
            async for chunk in resp:
                accumulate(chunk, state)
                yield chunk
        except Exception as exc:
            rec(*finish(state), repr(exc))
            raise
        rec(*finish(state), None)

    return wrap, awrap


def instrument_create(orig, default_name, extract, wrap_stream):
    @functools.wraps(orig)
    def create(self, *args, **kwargs):
        from .. import store
        model, messages, tools, stream, params = split(kwargs)
        rec = _recorder(default_name, model, messages, tools, params, store.now())
        try:
            resp = orig(self, *args, **kwargs)
        except Exception as exc:
            rec("", [], None, repr(exc))
            raise
        if stream:
            return wrap_stream(resp, rec)
        rec(*extract(resp), None)
        return resp

    create._touchstone = True
    return create


def instrument_acreate(orig, default_name, extract, wrap_astream):
    @functools.wraps(orig)
    async def acreate(self, *args, **kwargs):
        from .. import store
        model, messages, tools, stream, params = split(kwargs)
        rec = _recorder(default_name, model, messages, tools, params, store.now())
        try:
            resp = await orig(self, *args, **kwargs)
        except Exception as exc:
            rec("", [], None, repr(exc))
            raise
        if stream:
            return wrap_astream(resp, rec)
        rec(*extract(resp), None)
        return resp

    acreate._touchstone = True
    return acreate
