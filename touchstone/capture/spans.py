"""Shared span lifecycle for the SDK patches.

The patches for openai and anthropic differ only in how they read their own wire shapes. Everything
else — reading defensively, splitting call kwargs, rewriting the model when `TOUCHSTONE_MODEL` is
set (`override_model`), running one span per model call, accumulating a stream, normalizing the stop
reason, pricing the call, and recording output/usage/error — lives here. A patch supplies an
`extract(resp) -> result` for a whole response, plus a
`state`/`accumulate(chunk, state)`/`finish(state) -> result` trio for streams. A `result` is the
dict built by `model_result(...)`.

Capture must never raise into the user's application: every recording path is guarded, logs one
warning to the "touchstone" logger, and lets the SDK call proceed and return normally.
"""

from __future__ import annotations

import functools
import json
import logging
import os

from ..messages import canonical
from ..messages import get as get  # shared field accessor, re-exported for the SDK patches
from . import context

_log = logging.getLogger("touchstone")

# provider finish/stop reason -> normalized value
# (stop | tool_calls | length | content_filter | refusal | error)
_STOP = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop", "completed": "stop",
    "tool_calls": "tool_calls", "tool_use": "tool_calls", "function_call": "tool_calls",
    "length": "length", "max_tokens": "length", "max_output_tokens": "length",
    "content_filter": "content_filter", "refusal": "refusal", "error": "error",
}


def as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def normalize_stop(raw, refusal=None) -> str | None:
    if refusal:
        return "refusal"
    if raw is None:
        return None
    return _STOP.get(raw, str(raw))


def model_result(content, tool_calls, usage, *, stop_reason=None, reasoning=None, refusal=None):
    """The dict a patch hands back for one model call."""
    return {"content": content or "", "tool_calls": tool_calls or [], "usage": usage,
            "stop_reason": normalize_stop(stop_reason, refusal),
            "reasoning": reasoning or None, "refusal": refusal}


def override_model(kwargs: dict) -> None:
    """Run the real agent on a different model with one env var: if TOUCHSTONE_MODEL is set, rewrite
    `kwargs["model"]` to it before the SDK call goes through (same SDK, same everything else). A
    model passed positionally is left untouched (there is no keyword to rewrite) — see README."""
    forced = os.environ.get("TOUCHSTONE_MODEL")
    if forced and isinstance(kwargs.get("model"), str):
        kwargs["model"] = forced


def split(kwargs):
    model = kwargs.get("model")
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")
    stream = kwargs.get("stream", False)
    params = {k: v for k, v in kwargs.items() if k not in ("messages", "tools", "model")}
    return model, messages, tools, stream, params


def _reply_message(result: dict) -> dict:
    src = {"role": "assistant", "content": result.get("content") or "",
           "tool_calls": result.get("tool_calls") or []}
    if result.get("reasoning"):
        src["reasoning"] = result["reasoning"]
    if result.get("refusal") is not None:
        src["refusal"] = result["refusal"]
    return canonical([src])[0]


def _clean_usage(usage: dict | None) -> dict | None:
    if not usage:
        return None
    kept = {k: v for k, v in usage.items() if v is not None}
    return kept or None


def record(default_name, model, messages, tools, params, result, error, started):
    from .pricing import cost_usd

    usage = result.get("usage")
    output: dict = {"message": _reply_message(result)}
    if result.get("stop_reason"):
        output["stop_reason"] = result["stop_reason"]
    clean = _clean_usage(usage)
    if clean:
        output["usage"] = clean
    context.add_span(
        "model",
        model or default_name,
        model=model,
        input={
            "messages": canonical(messages),
            "tools": tools or [],
            "params": {k: context.jsonable(v) for k, v in params.items()},
        },
        output=output,
        tokens_in=usage.get("tokens_in") if usage else None,
        tokens_out=usage.get("tokens_out") if usage else None,
        cost_usd=cost_usd(model, usage) if model else None,
        error=error,
        started_at=started,
    )


def _recorder(default_name, model, messages, tools, params, started):
    """A guarded `(produce, error)` closure; `produce` yields the result dict lazily so extraction
    failures are caught here too and never reach the user's app."""
    def rec(produce, error):
        try:
            result = produce() if callable(produce) else produce
            record(default_name, model, messages, tools, params, result, error, started)
        except Exception as exc:  # capture must never break the caller's request
            _log.warning("touchstone capture failed: %r", exc)
    return rec


_ERROR = {"content": "", "tool_calls": [], "usage": None}


def _step(accumulate, chunk, state) -> None:
    try:
        accumulate(chunk, state)
    except Exception as exc:  # a bad chunk must not break the stream the user is reading
        _log.warning("touchstone stream capture failed: %r", exc)


def _sync_stream(resp, rec, state_factory, accumulate, finish):
    state = state_factory()
    try:
        for chunk in resp:
            _step(accumulate, chunk, state)
            yield chunk
    except Exception as exc:
        rec(lambda: finish(state), repr(exc))
        raise
    rec(lambda: finish(state), None)


async def _async_stream(resp, rec, state_factory, accumulate, finish):
    state = state_factory()
    try:
        async for chunk in resp:
            _step(accumulate, chunk, state)
            yield chunk
    except Exception as exc:
        rec(lambda: finish(state), repr(exc))
        raise
    rec(lambda: finish(state), None)


def stream_wrappers(state_factory, accumulate, finish):
    """Build (sync, async) generator wrappers that accumulate a stream and record at its end."""
    def wrap(resp, rec):
        return _sync_stream(resp, rec, state_factory, accumulate, finish)

    def awrap(resp, rec):
        return _async_stream(resp, rec, state_factory, accumulate, finish)

    return wrap, awrap


def instrument_create(orig, default_name, extract, wrap_stream, split_fn=split):
    @functools.wraps(orig)
    def create(self, *args, **kwargs):
        from .. import store
        override_model(kwargs)
        model, messages, tools, stream, params = split_fn(kwargs)
        rec = _recorder(default_name, model, messages, tools, params, store.now())
        try:
            resp = orig(self, *args, **kwargs)
        except Exception as exc:
            rec(_ERROR, repr(exc))
            raise
        if stream:
            return wrap_stream(resp, rec)
        rec(lambda: extract(resp), None)
        return resp

    create._touchstone = True
    return create


def instrument_acreate(orig, default_name, extract, wrap_astream, split_fn=split):
    @functools.wraps(orig)
    async def acreate(self, *args, **kwargs):
        from .. import store
        override_model(kwargs)
        model, messages, tools, stream, params = split_fn(kwargs)
        rec = _recorder(default_name, model, messages, tools, params, store.now())
        try:
            resp = await orig(self, *args, **kwargs)
        except Exception as exc:
            rec(_ERROR, repr(exc))
            raise
        if stream:
            return wrap_astream(resp, rec)
        rec(lambda: extract(resp), None)
        return resp

    acreate._touchstone = True
    return acreate
