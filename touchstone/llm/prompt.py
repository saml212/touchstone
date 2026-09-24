"""Shared helpers for the CLI providers (claude-cli, codex-cli).

They cannot take structured messages/tools, so `serialize_messages` flattens a chat into one prompt
and `extract_json` recovers a JSON value from free-form text. `parse_cli_result` turns a CLI's text
answer into a `Reply`, honouring the tool-call convention described in the serialized prompt.
"""

from __future__ import annotations

import json

from .base import Reply

_TOOL_INSTRUCTION = (
    "You may call a tool. To do so, reply with ONLY a JSON object of the form "
    '{"tool_calls": [{"name": "<tool>", "arguments": {<args>}}]}. '
    "Otherwise reply with your answer directly."
)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content-parts form
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or json.dumps(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def serialize_messages(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Flatten OpenAI-style messages (+ optional tools) into a single prompt string."""
    systems: list[str] = []
    turns: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            systems.append(_content_text(msg.get("content")))
            continue
        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            turns.append(f"TOOL RESULT ({name}): {_content_text(msg.get('content'))}")
            continue
        text = _content_text(msg.get("content"))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", tc)
            text += f"\n[called {fn.get('name')} with {fn.get('arguments')}]"
        turns.append(f"{role.upper()}: {text}".rstrip())

    blocks: list[str] = []
    if systems:
        blocks.append("SYSTEM:\n" + "\n".join(s for s in systems if s))
    if tools:
        blocks.append("TOOLS AVAILABLE (JSON schemas):\n" + json.dumps(tools, ensure_ascii=False))
        blocks.append(_TOOL_INSTRUCTION)
    blocks.extend(turns)
    return "\n\n".join(b for b in blocks if b)


def extract_json(text: str) -> str | None:
    """Return the first balanced JSON object/array substring in `text`, or None.

    Tolerates fenced ```json blocks and leading prose. Ignores braces inside strings.
    """
    if not text:
        return None
    candidate = _strip_fence(text)
    start = _first_open(candidate)
    if start is None:
        # Fence stripping may have removed the opener; retry on the raw text.
        candidate = text
        start = _first_open(candidate)
        if start is None:
            return None
    end = _match_bracket(candidate, start)
    if end is None:
        return None
    return candidate[start : end + 1]


def _strip_fence(text: str) -> str:
    if "```" not in text:
        return text
    after = text.split("```", 1)[1]
    if after[:4].lower() == "json":
        after = after[4:]
    return after.split("```", 1)[0] if "```" in after else after


def _first_open(text: str) -> int | None:
    positions = [p for p in (text.find("{"), text.find("[")) if p != -1]
    return min(positions) if positions else None


def _match_bracket(text: str, start: int) -> int | None:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def parse_cli_result(text: str, want_json: bool) -> Reply:
    """Build a Reply from a CLI's text answer, extracting tool calls / JSON when present."""
    text = text or ""
    obj_str = extract_json(text)
    obj = None
    if obj_str:
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        calls = []
        for tc in obj["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            args = tc.get("arguments", tc.get("args", {}))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({"id": tc.get("id"), "name": tc.get("name"), "arguments": args})
        return Reply(content=obj.get("content", "") or "", tool_calls=calls, raw=text)
    if want_json and obj_str is not None:
        return Reply(content=obj_str, raw=text)
    return Reply(content=text, raw=text)
