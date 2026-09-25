"""Render canonical messages back to provider wire shapes.

`to_openai()` and `to_anthropic()` are the inverse of `canonical()` (in `messages.py`): they take
canonical messages and produce the request body a provider SDK expects. They are re-exported from
`touchstone.messages`, so callers import them from there; this module only holds the rendering.
Reasoning is capture-only and dropped by `to_openai`; Anthropic keeps reconstructable thinking.
"""

from __future__ import annotations

import json

from .messages import _args_obj, _args_str, canonical, text_of


def _openai_part(part: dict) -> dict:
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part.get("type") == "image":
        img = part.get("image_url") or part.get("source")
        return {"type": "image_url", "image_url": img if isinstance(img, dict | str) else part}
    return dict(part)


def _openai_content(content):
    return [_openai_part(p) for p in content] if isinstance(content, list) else content


def _openai_tool_message(msg: dict) -> dict:
    wire = {"role": "tool", "content": _openai_content(msg.get("content", "")),
            "tool_call_id": msg.get("tool_call_id", "")}
    if msg.get("name"):
        wire["name"] = msg["name"]
    return wire


def _openai_message(msg: dict) -> dict:
    wire = {"role": msg["role"], "content": _openai_content(msg.get("content", ""))}
    if msg.get("tool_calls"):
        wire["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": _args_str(tc["arguments"])}}
            for tc in msg["tool_calls"]
        ]
    if msg.get("refusal") is not None:
        wire["refusal"] = msg["refusal"]
    return wire


def to_openai(messages: list[dict]) -> list[dict]:
    """Canonical messages -> OpenAI chat wire shape (reasoning is capture-only, dropped here)."""
    out = []
    for msg in canonical(messages):
        role = msg.get("role")
        if role is None:
            continue  # a preserved passthrough item has no chat-wire equivalent
        out.append(_openai_tool_message(msg) if role == "tool" else _openai_message(msg))
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


def _anthropic_text_block(part: dict) -> dict | None:
    if part.get("type") == "text":
        return {"type": "text", "text": part["text"]} if part.get("text") else None
    if part.get("type") == "image":
        return {"type": "image", "source": part.get("source") or part.get("image_url")}
    return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}


def _anthropic_text_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for p in content or []:
        block = _anthropic_text_block(p)
        if block:
            blocks.append(block)
    return blocks


def _anthropic_blocks(msg: dict) -> list[dict]:
    blocks = _anthropic_reasoning_blocks(msg.get("reasoning"))
    blocks += _anthropic_text_blocks(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                       "input": _args_obj(tc["arguments"])})
    return blocks


def _anthropic_system_text(msg: dict) -> str:
    return msg["content"] if isinstance(msg["content"], str) else text_of(msg)


def _anthropic_tool_result(msg: dict) -> dict:
    return {"type": "tool_result",
            "tool_use_id": msg.get("tool_call_id") or msg.get("name") or "",
            "content": msg.get("content", "")}


def _append_turn(turns: list[dict], role: str, blocks: list[dict]) -> None:
    if turns and turns[-1]["role"] == role:  # coalesce adjacent same-role turns
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": blocks})


def _add_anthropic_message(msg: dict, systems: list[str], turns: list[dict]) -> None:
    role = msg.get("role")
    if role is None:
        return  # a preserved passthrough item has no chat-wire equivalent
    if role == "system":
        text = _anthropic_system_text(msg)
        if text:
            systems.append(text)
    elif role == "tool":
        _append_turn(turns, "user", [_anthropic_tool_result(msg)])
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
