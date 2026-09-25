"""The one message shape the store holds, and how to read anything into it.

Entry point: `canonical(messages)` normalizes OpenAI chat, OpenAI Responses items, Anthropic wire
blocks, SDK objects (anything with `model_dump()`), and already-canonical messages into one shape,
idempotently (`canonical(canonical(x)) == canonical(x)`). A canonical message is a dict:

    {"role": "system"|"user"|"assistant"|"tool",
     "content": str | [ {"type": "text"|"image"|"audio"|"file", ...} ],  # str iff all-text
     "tool_calls"?: [{"id", "name", "arguments"(JSON str)}],  "reasoning"?: [...],  # assistant
     "refusal"?: str|None,  "tool_call_id"?: str,  "name"?: str}             # tool

Missing tool-call ids become deterministic `call_<n>`; a tool message with no `tool_call_id` links
to the preceding assistant call by id, then by name order. `text_of()` flattens a message to text
(non-text parts marked). Rendering canonical messages back to wire shape lives in `messages_wire`
(`to_openai` / `to_anthropic`), re-exported from here.
"""

from __future__ import annotations

import json


def _to_plain(obj):
    """A dict view of an SDK object (prefer `model_dump()`), else the object itself."""
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
        except Exception:
            result = None
        if isinstance(result, dict):
            return result
    return obj


def get(obj, key, default=None):
    """Read `key` from a dict or an attribute from an SDK object — the capture patches share it."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _args_str(value) -> str:
    """Arguments as a JSON string (OpenAI wants a string)."""
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def _args_obj(value) -> dict:
    """Arguments as a parsed object (Anthropic wants an object)."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {"input": value}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


# Provider content-part type -> canonical part type.
_PART_TYPE = {
    "text": "text", "input_text": "text", "output_text": "text",
    "image": "image", "image_url": "image", "input_image": "image",
    "audio": "audio", "input_audio": "audio",
    "file": "file", "input_file": "file", "document": "file",
}
_REASONING_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning", "redacted"})


def _norm_part(part) -> dict:
    part = _to_plain(part)
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}
    canon = _PART_TYPE.get(part.get("type"), part.get("type") or "text")
    if canon == "text":
        return {"type": "text", "text": part.get("text") or part.get("content") or ""}
    rest = {k: v for k, v in part.items() if k != "type"}
    rest["type"] = canon
    return rest


def _reasoning_item(block) -> dict:
    """An OpenAI Responses `reasoning` item -> canonical reasoning (redacted when encrypted)."""
    summary, content = get(block, "summary"), get(block, "content")
    if not summary and not content:
        return {"type": "redacted"}
    item = {"type": "reasoning"}
    if summary:
        item["summary"] = summary
    if content:
        item["content"] = content
    return item


def _thinking_item(block) -> dict:
    """An Anthropic `thinking` block -> canonical reasoning, keeping any signature."""
    item = {"type": "thinking", "thinking": get(block, "thinking") or get(block, "text") or ""}
    if get(block, "signature"):
        item["signature"] = get(block, "signature")
    return item


def _norm_reasoning_block(block) -> dict:
    block = _to_plain(block)
    btype = get(block, "type")
    if btype in ("redacted_thinking", "redacted"):
        return {"type": "redacted"}
    if btype == "reasoning":
        return _reasoning_item(block)
    return _thinking_item(block)


def _norm_reasoning(value) -> list[dict]:
    if not value:
        return []
    blocks = value if isinstance(value, list) else [value]
    return [_norm_reasoning_block(b) for b in blocks]


def _collapse(parts: list[dict]):
    """Text-only parts -> a joined string (never flattened when any part is non-text)."""
    if not parts:
        return ""
    if all(p.get("type") == "text" for p in parts):
        return "\n".join(p.get("text", "") for p in parts if p.get("text"))
    return parts


def _content_value(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", "replace")
    if isinstance(content, list):
        return _collapse([_norm_part(p) for p in content])
    return json.dumps(content, ensure_ascii=False, default=str)


def _norm_tool_call(tc: dict) -> dict:
    tc = _to_plain(tc)
    fn = _to_plain(get(tc, "function"))
    fn = fn if fn is not None else tc
    tcid = get(tc, "id") or get(tc, "call_id") or get(tc, "tool_call_id")
    return {"id": tcid, "name": get(fn, "name"), "arguments": _args_str(get(fn, "arguments"))}


def _tool_message(tool_call_id, name, content) -> dict:
    msg = {"role": "tool", "content": _content_value(content)}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    if name:
        msg["name"] = name
    return msg


def _split_block(block, parts, tool_calls, tool_results, reasoning) -> None:
    """Route one content block into the parts / tool-call / tool-result / reasoning bucket."""
    btype = get(block, "type") if not isinstance(block, str) else "text"
    if btype == "tool_use":
        tool_calls.append({"id": get(block, "id"), "name": get(block, "name"),
                           "arguments": _args_str(get(block, "input"))})
    elif btype == "tool_result":
        tool_results.append(_tool_message(get(block, "tool_use_id"), None, get(block, "content")))
    elif btype in _REASONING_TYPES:
        reasoning.append(_norm_reasoning_block(block))
    else:
        parts.append(_norm_part(block))


def _split_content(content) -> tuple[list, list, list, list]:
    """A content value -> (parts, tool_use calls, tool_result messages, reasoning blocks)."""
    parts: list[dict] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    reasoning: list[dict] = []
    if content is None:
        return parts, tool_calls, tool_results, reasoning
    if isinstance(content, str):
        parts.append({"type": "text", "text": content})
        return parts, tool_calls, tool_results, reasoning
    for raw in content if isinstance(content, list) else [content]:
        _split_block(_to_plain(raw), parts, tool_calls, tool_results, reasoning)
    return parts, tool_calls, tool_results, reasoning


_RESPONSES_ITEMS = frozenset({"function_call", "function_call_output", "reasoning"})


def _responses_item(itype: str, item) -> list[dict]:
    """A top-level OpenAI Responses input item (sibling of messages) -> canonical message(s)."""
    if itype == "function_call_output":
        return [_tool_message(get(item, "call_id"), None, get(item, "output"))]
    if itype == "reasoning":
        return [{"role": "assistant", "content": "", "reasoning": _norm_reasoning([item])}]
    return [{"role": "assistant", "content": "", "tool_calls": [_norm_tool_call(item)]}]


def _passthrough_item(msg, itype) -> list[dict]:
    """A role-less Responses item: a modelled type -> message(s); an unknown type (web_search_call,
    computer_call, ...) kept verbatim so the trace keeps it, never flattened to an empty turn."""
    if itype in _RESPONSES_ITEMS:
        return _responses_item(itype, msg)
    return [msg if isinstance(msg, dict) else dict(msg)]


def _from_role(role, msg) -> list[dict]:
    content = get(msg, "content")
    if role == "tool":
        return [_tool_message(get(msg, "tool_call_id"), get(msg, "name"), content)]
    parts, tool_calls, tool_results, reasoning = _split_content(content)
    for tc in get(msg, "tool_calls") or []:
        tool_calls.append(_norm_tool_call(tc))
    reasoning += _norm_reasoning(get(msg, "reasoning"))
    return _assemble(role, parts, tool_calls, tool_results, reasoning, get(msg, "refusal"))


def _canonical_one(msg) -> list[dict]:
    msg = _to_plain(msg)
    itype = get(msg, "type")
    if get(msg, "role") is None and itype is not None:
        return _passthrough_item(msg, itype)
    return _from_role(get(msg, "role") or "user", msg)


def _assemble(role, parts, tool_calls, tool_results, reasoning, refusal) -> list[dict]:
    out = list(tool_results)
    content = _collapse(parts)
    if content or tool_calls or reasoning or refusal is not None or not tool_results:
        primary = {"role": role, "content": content}
        if tool_calls:
            primary["tool_calls"] = tool_calls
        if reasoning:
            primary["reasoning"] = reasoning
        if refusal is not None:
            primary["refusal"] = refusal
        out.append(primary)
    return out


def _fill_call_ids(messages: list[dict]) -> None:
    n = 0
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            if not tc.get("id"):
                n += 1
                tc["id"] = f"call_{n}"


def _first_unused(pending: list[list], predicate) -> int | None:
    for i, entry in enumerate(pending):
        if not entry[2] and predicate(entry):
            return i
    return None


def _match_pending(pending: list[list], name) -> int | None:
    if name:
        idx = _first_unused(pending, lambda entry: entry[1] == name)
        if idx is not None:
            return idx
    return _first_unused(pending, lambda entry: True)


def _link_one_tool(msg: dict, pending: list[list]) -> None:
    """Give a tool message its `tool_call_id` from the preceding assistant call, and mark that
    call's pending slot consumed. An id the caller already supplied is kept, not overwritten."""
    existing = msg.get("tool_call_id")
    if existing:
        idx = _first_unused(pending, lambda entry, _id=existing: entry[0] == _id)
    else:
        idx = _match_pending(pending, msg.get("name"))
        if idx is not None:
            msg["tool_call_id"] = pending[idx][0]
    if idx is not None:
        pending[idx][2] = True


def _link_tool_messages(messages: list[dict]) -> None:
    pending: list[list] = []  # [id, name, consumed]
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pending = [[tc["id"], tc.get("name"), False] for tc in msg["tool_calls"]]
        elif msg.get("role") == "tool":
            _link_one_tool(msg, pending)


def canonical(messages: list[dict]) -> list[dict]:
    """Convert any supported message list into canonical messages (idempotent)."""
    out: list[dict] = []
    for msg in messages or []:
        out.extend(_canonical_one(msg))
    _fill_call_ids(out)
    _link_tool_messages(out)
    return out


def text_of(message: dict) -> str:
    """A flat string for a canonical message; non-text parts render as `[image]` / `[file]`."""
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered = [p.get("text", "") if p.get("type") == "text" else f"[{p.get('type')}]"
                    for p in content]
        return "\n".join(r for r in rendered if r)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


# Re-exported so callers keep importing them from `touchstone.messages` (see module docstring).
from .messages_wire import to_anthropic, to_openai  # noqa: E402, F401
