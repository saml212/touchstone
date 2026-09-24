"""One canonical message shape, and conversions to/from provider wire shapes.

The store only ever holds canonical messages. A canonical message is a dict:

    {"role": "system" | "user" | "assistant" | "tool",
     "content": str,
     "tool_calls"?: [{"id": str, "name": str, "arguments": str}],  # assistant only
     "tool_call_id"?: str,                                          # tool only
     "name"?: str}                                                  # tool only

`arguments` is always a JSON string. `canonical()` accepts OpenAI wire shape (tool calls
nested under `function`, tool messages carrying `tool_call_id`, content-parts lists),
Anthropic wire shape (`text` / `tool_use` / `tool_result` content blocks, system passed
separately), and already-canonical messages, and is idempotent: `canonical(canonical(x))
== canonical(x)`. Missing tool-call ids become deterministic `call_<n>`; a tool message with
no `tool_call_id` is linked to the preceding assistant call by name order.

`to_openai()` and `to_anthropic()` convert canonical messages back to wire shape.
"""

from __future__ import annotations

import json


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


def _text_part(part) -> str:
    return (part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)


def _text(content) -> str:
    """Flatten content (str, content-parts list, or blocks) into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p for p in (_text_part(x) for x in content) if p)
    return json.dumps(content, ensure_ascii=False)


def _norm_tool_call(tc: dict) -> dict:
    fn = tc["function"] if isinstance(tc.get("function"), dict) else tc
    return {"id": tc.get("id"), "name": fn.get("name"), "arguments": _args_str(fn.get("arguments"))}


def _tool_message(tool_call_id, name, content) -> dict:
    msg = {"role": "tool", "content": _text(content)}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    if name:
        msg["name"] = name
    return msg


def _split_content_list(content: list) -> tuple[list[str], list[dict], list[dict]]:
    """A content-parts / block list -> (texts, Anthropic tool_use calls, tool_result messages)."""
    texts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            texts.append(str(part))
        elif part.get("type") == "tool_use":  # Anthropic assistant tool call
            tool_calls.append({"id": part.get("id"), "name": part.get("name"),
                               "arguments": _args_str(part.get("input"))})
        elif part.get("type") == "tool_result":  # Anthropic tool result (inside a user turn)
            tool_results.append(_tool_message(part.get("tool_use_id"), None, part.get("content")))
        else:
            texts.append(part.get("text") or part.get("content") or "")
    return texts, tool_calls, tool_results


def _canonical_one(msg: dict) -> list[dict]:
    role = msg.get("role", "user")
    content = msg.get("content")
    if role == "tool":
        return [_tool_message(msg.get("tool_call_id"), msg.get("name"), content)]

    if isinstance(content, list):
        texts, tool_calls, tool_results = _split_content_list(content)
    elif isinstance(content, str):
        texts, tool_calls, tool_results = [content], [], []
    elif content is not None:
        texts, tool_calls, tool_results = [json.dumps(content, ensure_ascii=False)], [], []
    else:
        texts, tool_calls, tool_results = [], [], []

    for tc in msg.get("tool_calls") or []:
        tool_calls.append(_norm_tool_call(tc))
    return _assemble(role, texts, tool_calls, tool_results)


def _assemble(role: str, texts: list[str], tool_calls: list[dict],
              tool_results: list[dict]) -> list[dict]:
    out = list(tool_results)
    joined = "\n".join(t for t in texts if t)
    if joined or tool_calls or not tool_results:
        primary = {"role": role, "content": joined}
        if tool_calls:
            primary["tool_calls"] = tool_calls
        out.append(primary)
    return out


def _fill_call_ids(messages: list[dict]) -> None:
    n = 0
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            if not tc.get("id"):
                n += 1
                tc["id"] = f"call_{n}"


def _link_one(msg: dict, pending: list[list]) -> None:
    idx = _match_pending(pending, msg.get("name"))
    if idx is not None:
        msg["tool_call_id"] = pending[idx][0]
        pending[idx][2] = True


def _link_tool_messages(messages: list[dict]) -> None:
    pending: list[list] = []  # [id, name, consumed]
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            pending = [[tc["id"], tc.get("name"), False] for tc in msg["tool_calls"]]
        elif msg["role"] == "tool" and not msg.get("tool_call_id"):
            _link_one(msg, pending)


def _assign_ids(messages: list[dict]) -> None:
    _fill_call_ids(messages)
    _link_tool_messages(messages)


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


def canonical(messages: list[dict]) -> list[dict]:
    """Convert any supported message list into canonical messages (idempotent)."""
    out: list[dict] = []
    for msg in messages or []:
        out.extend(_canonical_one(msg))
    _assign_ids(out)
    return out


def to_openai(messages: list[dict]) -> list[dict]:
    """Canonical messages -> OpenAI chat wire shape."""
    out = []
    for msg in canonical(messages):
        role = msg["role"]
        if role == "tool":
            wire = {"role": "tool", "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", "")}
            if msg.get("name"):
                wire["name"] = msg["name"]
            out.append(wire)
            continue
        wire = {"role": role, "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            wire["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": _args_str(tc["arguments"])}}
                for tc in msg["tool_calls"]
            ]
        out.append(wire)
    return out


def _anthropic_blocks(msg: dict) -> list[dict]:
    """The content + tool_use blocks for one non-system, non-tool canonical message."""
    blocks: list[dict] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                       "input": _args_obj(tc["arguments"])})
    return blocks


def _add_anthropic_message(msg: dict, systems: list[str], turns: list[dict]) -> None:
    role = msg["role"]
    if role == "system":
        if msg.get("content"):
            systems.append(msg["content"])
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
