"""One canonical message shape, and conversions to/from provider wire shapes.

The store only ever holds canonical messages. A canonical message is a dict:

    {"role": "system" | "user" | "assistant" | "tool",
     "content": str | [ {"type": "text"|"image"|"audio"|"file", ...} ],
     "tool_calls"?: [{"id": str, "name": str, "arguments": str}],  # assistant only
     "reasoning"?: [ <thinking block> | {"type": "redacted"} ],     # assistant only
     "refusal"?: str | None,                                        # assistant only
     "tool_call_id"?: str,                                          # tool only
     "name"?: str}                                                  # tool only

`content` is a string when every part is text, otherwise the list of parts — image / audio /
file parts are never flattened away. `arguments` is always a JSON string. `canonical()` accepts
dicts and SDK objects (anything exposing `model_dump()` or the relevant attributes), in OpenAI
wire shape (tool calls nested under `function`, tool messages carrying `tool_call_id`,
content-parts lists), OpenAI Responses items (`function_call` / `reasoning`), Anthropic wire shape
(`text` / `tool_use` / `tool_result` / `thinking` / `redacted_thinking` blocks, system separate),
and already-canonical messages, and is idempotent: `canonical(canonical(x)) == canonical(x)`.
Missing tool-call ids become deterministic `call_<n>`; a tool message with no `tool_call_id` links
to the preceding assistant call by id first, then name order.

`to_openai()` / `to_anthropic()` convert canonical messages back to wire shape. `text_of()` gives a
flat string for callers (checks, prompts, instruction.md) that want text, marking non-text parts.
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


def _norm_reasoning_block(block) -> dict:
    block = _to_plain(block)
    btype = get(block, "type")
    if btype in ("redacted_thinking", "redacted"):
        return {"type": "redacted"}
    if btype == "reasoning":  # OpenAI Responses reasoning item
        summary, content = get(block, "summary"), get(block, "content")
        if not summary and not content:
            return {"type": "redacted"}  # encrypted-only
        item = {"type": "reasoning"}
        if summary:
            item["summary"] = summary
        if content:
            item["content"] = content
        return item
    item = {"type": "thinking", "thinking": get(block, "thinking") or get(block, "text") or ""}
    if get(block, "signature"):
        item["signature"] = get(block, "signature")
    return item


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
        block = _to_plain(raw)
        btype = get(block, "type") if not isinstance(block, str) else "text"
        if btype == "tool_use":
            tool_calls.append({"id": get(block, "id"), "name": get(block, "name"),
                               "arguments": _args_str(get(block, "input"))})
        elif btype == "tool_result":
            tool_results.append(
                _tool_message(get(block, "tool_use_id"), None, get(block, "content")))
        elif btype in _REASONING_TYPES:
            reasoning.append(_norm_reasoning_block(block))
        else:
            parts.append(_norm_part(block))
    return parts, tool_calls, tool_results, reasoning


_RESPONSES_ITEMS = frozenset({"function_call", "function_call_output", "reasoning"})


def _responses_item(itype: str, item) -> list[dict]:
    """A top-level OpenAI Responses input item (sibling of messages) -> canonical message(s)."""
    if itype == "function_call_output":
        return [_tool_message(get(item, "call_id"), None, get(item, "output"))]
    if itype == "reasoning":
        return [{"role": "assistant", "content": "", "reasoning": _norm_reasoning([item])}]
    return [{"role": "assistant", "content": "", "tool_calls": [_norm_tool_call(item)]}]


def _canonical_one(msg) -> list[dict]:
    msg = _to_plain(msg)
    itype = get(msg, "type")
    if get(msg, "role") is None and itype is not None:
        if itype in _RESPONSES_ITEMS:
            return _responses_item(itype, msg)
        # A Responses item of a type we don't model (web_search_call, computer_call, ...)
        # is preserved verbatim so the trace keeps it, rather than flattened to an empty turn.
        return [msg if isinstance(msg, dict) else dict(msg)]
    role = get(msg, "role") or "user"
    content = get(msg, "content")
    if role == "tool":
        return [_tool_message(get(msg, "tool_call_id"), get(msg, "name"), content)]

    parts, tool_calls, tool_results, reasoning = _split_content(content)
    for tc in get(msg, "tool_calls") or []:
        tool_calls.append(_norm_tool_call(tc))
    reasoning += _norm_reasoning(get(msg, "reasoning"))
    return _assemble(role, parts, tool_calls, tool_results, reasoning, get(msg, "refusal"))


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


def _link_tool_messages(messages: list[dict]) -> None:
    pending: list[list] = []  # [id, name, consumed]
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pending = [[tc["id"], tc.get("name"), False] for tc in msg["tool_calls"]]
        elif msg.get("role") == "tool":
            existing = msg.get("tool_call_id")
            if existing:  # keep an id the caller already supplied; just consume its pending slot
                idx = _first_unused(pending, lambda entry, _id=existing: entry[0] == _id)
            else:
                idx = _match_pending(pending, msg.get("name"))
                if idx is not None:
                    msg["tool_call_id"] = pending[idx][0]
            if idx is not None:
                pending[idx][2] = True


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


def context_text(context: dict | None) -> str:
    """Flattened user + system text of a task's context — the `context_text` an expr check reads."""
    messages = (context or {}).get("messages", [])
    parts = [text_of(m) for m in messages if m.get("role") in ("user", "system")]
    return "\n".join(p for p in parts if p)


# ---- back to wire ----------------------------------------------------------


def _openai_part(part: dict) -> dict:
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part.get("type") == "image":
        img = part.get("image_url") or part.get("source")
        return {"type": "image_url", "image_url": img if isinstance(img, dict | str) else part}
    return dict(part)


def _openai_content(content):
    return [_openai_part(p) for p in content] if isinstance(content, list) else content


def to_openai(messages: list[dict]) -> list[dict]:
    """Canonical messages -> OpenAI chat wire shape (reasoning is capture-only, dropped here)."""
    out = []
    for msg in canonical(messages):
        role = msg.get("role")
        if role is None:
            continue  # a preserved passthrough item has no chat-wire equivalent
        if role == "tool":
            wire = {"role": "tool", "content": _openai_content(msg.get("content", "")),
                    "tool_call_id": msg.get("tool_call_id", "")}
            if msg.get("name"):
                wire["name"] = msg["name"]
            out.append(wire)
            continue
        wire = {"role": role, "content": _openai_content(msg.get("content", ""))}
        if msg.get("tool_calls"):
            wire["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": _args_str(tc["arguments"])}}
                for tc in msg["tool_calls"]
            ]
        if msg.get("refusal") is not None:
            wire["refusal"] = msg["refusal"]
        out.append(wire)
    return out


def _anthropic_reasoning_blocks(reasoning: list[dict]) -> list[dict]:
    blocks = []
    for r in reasoning or []:
        if r.get("type") == "thinking":  # redacted data cannot be reconstructed; dropped
            block = {"type": "thinking", "thinking": r.get("thinking", "")}
            if r.get("signature"):
                block["signature"] = r["signature"]
            blocks.append(block)
    return blocks


def _anthropic_text_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for p in content or []:
        if p.get("type") == "text":
            if p.get("text"):
                blocks.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "image":
            blocks.append({"type": "image", "source": p.get("source") or p.get("image_url")})
        else:
            blocks.append({"type": "text", "text": json.dumps(p, ensure_ascii=False)})
    return blocks


def _anthropic_blocks(msg: dict) -> list[dict]:
    blocks = _anthropic_reasoning_blocks(msg.get("reasoning"))
    blocks += _anthropic_text_blocks(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                       "input": _args_obj(tc["arguments"])})
    return blocks


def _add_anthropic_message(msg: dict, systems: list[str], turns: list[dict]) -> None:
    role = msg.get("role")
    if role is None:
        return  # a preserved passthrough item has no chat-wire equivalent
    if role == "system":
        text = msg["content"] if isinstance(msg["content"], str) else text_of(msg)
        if text:
            systems.append(text)
    elif role == "tool":
        block = {"type": "tool_result",
                 "tool_use_id": msg.get("tool_call_id") or msg.get("name") or "",
                 "content": msg.get("content", "")}
        _append_turn(turns, "user", [block])
    else:
        blocks = _anthropic_blocks(msg)
        if blocks:
            _append_turn(turns, role, blocks)


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Canonical messages -> (system_text, Anthropic wire messages)."""
    systems: list[str] = []
    turns: list[dict] = []
    for msg in canonical(messages):
        _add_anthropic_message(msg, systems, turns)
    return "\n\n".join(systems), turns


def _append_turn(turns: list[dict], role: str, blocks: list[dict]) -> None:
    if turns and turns[-1]["role"] == role:  # coalesce adjacent same-role turns
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": blocks})
