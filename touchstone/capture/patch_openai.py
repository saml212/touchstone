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


def _accumulate_usage(chunk, st):
    u = _usage(chunk)
    if u and (u["tokens_in"] or u["tokens_out"]):
        st["usage"] = u


def _accumulate_choice(choice, st):
    if get(choice, "finish_reason"):
        st["stop"] = get(choice, "finish_reason")
    delta = get(choice, "delta")
    c = get(delta, "content")
    if c:
        st["parts"].append(c)
    if get(delta, "refusal"):
        st["refusal"].append(get(delta, "refusal"))
    for tc in get(delta, "tool_calls") or []:
        _accumulate_tool_call(tc, st)


def _accumulate(chunk, st):
    _accumulate_usage(chunk, st)
    choices = get(chunk, "choices") or []
    if choices:
        _accumulate_choice(choices[0], st)


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


# ---- Responses API ---------------------------------------------------------

RESPONSES_NAME = "openai.responses"


def _split_responses(kwargs):
    """Responses uses `input` (str or items) + `instructions`, not `messages`."""
    model = kwargs.get("model")
    items = kwargs.get("input")
    messages = []
    if kwargs.get("instructions"):
        messages.append({"role": "system", "content": kwargs["instructions"]})
    if isinstance(items, str):
        messages.append({"role": "user", "content": items})
    elif isinstance(items, list):
        messages.extend(items)
    tools = kwargs.get("tools")
    skip = ("input", "tools", "model", "instructions")
    params = {k: v for k, v in kwargs.items() if k not in skip}
    return model, messages, tools, kwargs.get("stream", False), params


def _responses_usage(u):
    if u is None:
        return None
    usage = {"tokens_in": get(u, "input_tokens"), "tokens_out": get(u, "output_tokens")}
    cached = get(get(u, "input_tokens_details"), "cached_tokens")
    if cached is not None:
        usage["cached_tokens"] = cached
    reasoning = get(get(u, "output_tokens_details"), "reasoning_tokens")
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    return usage


def _responses_stop(resp, tool_calls):
    if get(resp, "status") == "incomplete":
        return get(get(resp, "incomplete_details"), "reason")
    return "tool_calls" if tool_calls else "stop"


def _message_item(item, texts, out):
    for block in get(item, "content") or []:
        bt = get(block, "type")
        if bt in ("output_text", "text"):
            texts.append(get(block, "text") or "")
        elif bt == "refusal":
            out["refusal"] = get(block, "refusal")


def _extract_responses(resp):
    texts, tool_calls, reasoning = [], [], []
    out = {"refusal": None}
    for item in get(resp, "output") or []:
        itype = get(item, "type")
        if itype == "message":
            _message_item(item, texts, out)
        elif itype == "function_call":
            tool_calls.append({"id": get(item, "call_id") or get(item, "id"),
                               "name": get(item, "name"),
                               "arguments": as_str(get(item, "arguments"))})
        elif itype == "reasoning":
            reasoning.append({"type": "reasoning", "summary": get(item, "summary"),
                              "content": get(item, "content")})
    content = "".join(texts) or (get(resp, "output_text") or "")
    return model_result(content, tool_calls, _responses_usage(get(resp, "usage")),
                        stop_reason=_responses_stop(resp, tool_calls),
                        reasoning=reasoning or None, refusal=out["refusal"])


def _state_r():
    return {"text": [], "refusal": [], "final": None}


_R_TERMINAL = ("response.completed", "response.incomplete", "response.failed")


def _accumulate_r(event, st):
    etype = get(event, "type") or ""
    if etype in _R_TERMINAL:
        st["final"] = get(event, "response")
    elif etype.endswith("output_text.delta"):
        st["text"].append(get(event, "delta") or "")
    elif etype.endswith("refusal.delta"):
        st["refusal"].append(get(event, "delta") or "")


def _finish_r(st):
    if st["final"] is not None:
        return _extract_responses(st["final"])
    return model_result("".join(st["text"]), [], None, refusal="".join(st["refusal"]) or None)


_wrap_stream_r, _wrap_astream_r = spans.stream_wrappers(_state_r, _accumulate_r, _finish_r)


def _patch_responses() -> None:
    try:
        from openai.resources import responses as r
    except Exception:
        return  # SDK too old for the Responses API
    if not getattr(r.Responses.create, "_touchstone", False):
        r.Responses.create = spans.instrument_create(
            r.Responses.create, RESPONSES_NAME, _extract_responses, _wrap_stream_r,
            split_fn=_split_responses)
    if not getattr(r.AsyncResponses.create, "_touchstone", False):
        r.AsyncResponses.create = spans.instrument_acreate(
            r.AsyncResponses.create, RESPONSES_NAME, _extract_responses, _wrap_astream_r,
            split_fn=_split_responses)


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
    _patch_responses()
    return True
