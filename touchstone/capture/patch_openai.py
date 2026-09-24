"""Patch openai chat completions (sync + async, streaming + non-streaming) into one span each.

We read the response defensively (dict or object), accumulate stream chunks, keep tool-call
arguments as raw strings (they are often partial or non-JSON), and record errors on the span.
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


def _usage(resp):
    u = _get(resp, "usage")
    if u is None:
        return None
    return {"tokens_in": _get(u, "prompt_tokens"), "tokens_out": _get(u, "completion_tokens")}


def _extract(resp):
    choices = _get(resp, "choices") or []
    if not choices:
        return "", [], _usage(resp)
    msg = _get(choices[0], "message")
    tool_calls = []
    for tc in _get(msg, "tool_calls") or []:
        fn = _get(tc, "function")
        tool_calls.append({
            "id": _get(tc, "id"),
            "name": _get(fn, "name"),
            "arguments": _as_str(_get(fn, "arguments")),
        })
    return _get(msg, "content") or "", tool_calls, _usage(resp)


def _accumulate(chunk, parts, tool_args, tool_meta):
    choices = _get(chunk, "choices") or []
    if not choices:
        return
    delta = _get(choices[0], "delta")
    c = _get(delta, "content")
    if c:
        parts.append(c)
    for tc in _get(delta, "tool_calls") or []:
        idx = _get(tc, "index") or 0
        meta = tool_meta.setdefault(idx, {})
        if _get(tc, "id"):
            meta["id"] = _get(tc, "id")
        fn = _get(tc, "function")
        if _get(fn, "name"):
            meta["name"] = _get(fn, "name")
        a = _get(fn, "arguments")
        if a:
            tool_args[idx] = tool_args.get(idx, "") + a


def _stream_tool_calls(tool_args, tool_meta):
    calls = []
    for idx in sorted(set(tool_args) | set(tool_meta)):
        meta = tool_meta.get(idx, {})
        calls.append(
            {"id": meta.get("id"), "name": meta.get("name"), "arguments": tool_args.get(idx, "")}
        )
    return calls


def _record(model, messages, tools, params, content, tool_calls, usage, error, started):
    tokens_in = usage.get("tokens_in") if usage else None
    tokens_out = usage.get("tokens_out") if usage else None
    reply = canonical([{"role": "assistant", "content": content, "tool_calls": tool_calls}])[0]
    context.add_span(
        "llm",
        model or "openai.chat",
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
    parts, tool_args, tool_meta, usage = [], {}, {}, None
    try:
        for chunk in resp:
            _accumulate(chunk, parts, tool_args, tool_meta)
            u = _usage(chunk)
            if u and (u["tokens_in"] or u["tokens_out"]):
                usage = u
            yield chunk
    except Exception as exc:
        _record(model, messages, tools, params, "".join(parts),
                _stream_tool_calls(tool_args, tool_meta), usage, repr(exc), started)
        raise
    _record(model, messages, tools, params, "".join(parts),
            _stream_tool_calls(tool_args, tool_meta), usage, None, started)


async def _wrap_astream(resp, model, messages, tools, params, started):
    parts, tool_args, tool_meta, usage = [], {}, {}, None
    try:
        async for chunk in resp:
            _accumulate(chunk, parts, tool_args, tool_meta)
            u = _usage(chunk)
            if u and (u["tokens_in"] or u["tokens_out"]):
                usage = u
            yield chunk
    except Exception as exc:
        _record(model, messages, tools, params, "".join(parts),
                _stream_tool_calls(tool_args, tool_meta), usage, repr(exc), started)
        raise
    _record(model, messages, tools, params, "".join(parts),
            _stream_tool_calls(tool_args, tool_meta), usage, None, started)


def patch() -> bool:
    try:
        from openai.resources.chat import completions as c
    except Exception:
        return False

    if not getattr(c.Completions.create, "_touchstone", False):
        orig = c.Completions.create

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
            content, tool_calls, usage = _extract(resp)
            _record(model, messages, tools, params, content, tool_calls, usage, None, started)
            return resp

        create._touchstone = True
        c.Completions.create = create

    if not getattr(c.AsyncCompletions.create, "_touchstone", False):
        aorig = c.AsyncCompletions.create

        @functools.wraps(aorig)
        async def acreate(self, *args, **kwargs):
            from .. import store
            model, messages, tools, stream, params = _split(kwargs)
            started = store.now()
            try:
                resp = await aorig(self, *args, **kwargs)
            except Exception as exc:
                _record(model, messages, tools, params, "", [], None, repr(exc), started)
                raise
            if stream:
                return _wrap_astream(resp, model, messages, tools, params, started)
            content, tool_calls, usage = _extract(resp)
            _record(model, messages, tools, params, content, tool_calls, usage, None, started)
            return resp

        acreate._touchstone = True
        c.AsyncCompletions.create = acreate

    return True
