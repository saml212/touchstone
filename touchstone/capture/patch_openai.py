"""Patch openai chat completions (sync + async, streaming + non-streaming) into one span each.

We read the response defensively (dict or object), accumulate stream chunks, keep tool-call
arguments as raw strings (they are often partial or non-JSON), and record errors on the span.
The span lifecycle lives in `spans`; this module only maps openai's wire shapes.
"""

from __future__ import annotations

from . import spans
from .spans import as_str, get, model_result

DEFAULT_NAME = "openai.chat"


def _usage(resp):
    u = get(resp, "usage")
    if u is None:
        return None
    usage = {"tokens_in": get(u, "prompt_tokens"), "tokens_out": get(u, "completion_tokens")}
    cached = get(get(u, "prompt_tokens_details"), "cached_tokens")
    if cached is not None:
        usage["cached_tokens"] = cached
    reasoning = get(get(u, "completion_tokens_details"), "reasoning_tokens")
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    return usage


def _extract(resp):
    choices = get(resp, "choices") or []
    if not choices:
        return model_result("", [], _usage(resp))
    choice = choices[0]
    msg = get(choice, "message")
    tool_calls = []
    for tc in get(msg, "tool_calls") or []:
        fn = get(tc, "function")
        tool_calls.append({
            "id": get(tc, "id"),
            "name": get(fn, "name"),
            "arguments": as_str(get(fn, "arguments")),
        })
    return model_result(get(msg, "content") or "", tool_calls, _usage(resp),
                        stop_reason=get(choice, "finish_reason"), refusal=get(msg, "refusal"))


def _state():
    return {"parts": [], "args": {}, "meta": {}, "usage": None, "stop": None, "refusal": []}


def _accumulate_tool_call(tc, st):
    idx = get(tc, "index") or 0
    meta = st["meta"].setdefault(idx, {})
    if get(tc, "id"):
        meta["id"] = get(tc, "id")
    fn = get(tc, "function")
    if get(fn, "name"):
        meta["name"] = get(fn, "name")
    a = get(fn, "arguments")
    if a:
        st["args"][idx] = st["args"].get(idx, "") + a


def _accumulate(chunk, st):
    u = _usage(chunk)
    if u and (u["tokens_in"] or u["tokens_out"]):
        st["usage"] = u
    choices = get(chunk, "choices") or []
    if not choices:
        return
    if get(choices[0], "finish_reason"):
        st["stop"] = get(choices[0], "finish_reason")
    delta = get(choices[0], "delta")
    c = get(delta, "content")
    if c:
        st["parts"].append(c)
    if get(delta, "refusal"):
        st["refusal"].append(get(delta, "refusal"))
    for tc in get(delta, "tool_calls") or []:
        _accumulate_tool_call(tc, st)


def _finish(st):
    calls = []
    for idx in sorted(set(st["args"]) | set(st["meta"])):
        meta = st["meta"].get(idx, {})
        calls.append(
            {"id": meta.get("id"), "name": meta.get("name"), "arguments": st["args"].get(idx, "")}
        )
    refusal = "".join(st["refusal"]) or None
    return model_result("".join(st["parts"]), calls, st["usage"],
                        stop_reason=st["stop"], refusal=refusal)


_wrap_stream, _wrap_astream = spans.stream_wrappers(_state, _accumulate, _finish)


def patch() -> bool:
    try:
        from openai.resources.chat import completions as c
    except Exception:
        return False

    if not getattr(c.Completions.create, "_touchstone", False):
        c.Completions.create = spans.instrument_create(
            c.Completions.create, DEFAULT_NAME, _extract, _wrap_stream
        )
    if not getattr(c.AsyncCompletions.create, "_touchstone", False):
        c.AsyncCompletions.create = spans.instrument_acreate(
            c.AsyncCompletions.create, DEFAULT_NAME, _extract, _wrap_astream
        )
    return True
