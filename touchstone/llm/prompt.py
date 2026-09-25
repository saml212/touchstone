"""Shared helpers for the CLI providers (claude-cli, codex-cli).

They cannot take structured messages/tools, so `serialize_messages` flattens a chat into one prompt
and `extract_json` recovers a JSON value from free-form text. `parse_cli_result` turns a CLI's text
answer into a `Reply`, honouring the tool-call convention described in the serialized prompt.
"""

from __future__ import annotations

import json

from .base import Reply

_TRANSCRIPT_NOTE = (
    "The user message is the conversation so far, one turn per line labelled USER, ASSISTANT, "
    "or TOOL RESULT. Continue it as the ASSISTANT: reply with your next turn only."
)

_TOOL_INSTRUCTION = (
    "You may call a tool. To do so, reply with ONLY a JSON object of the form "
    '{"tool_calls": [{"name": "<tool>", "arguments": {<args>}}]}. '
    "Otherwise reply with your answer directly."
)


def _part_text(part) -> str:
    if isinstance(part, dict):
        return part.get("text") or part.get("content") or json.dumps(part)
    return str(part)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content-parts form
        return "\n".join(_part_text(p) for p in content)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _chat_turn(msg: dict, role: str) -> str:
    text = _content_text(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        text += f"\n[called {fn.get('name')} with {fn.get('arguments')}]"
    return f"{role.upper()}: {text}".rstrip()


def _split(messages: list[dict], tools: list[dict] | None) -> tuple[str, str]:
    """(system text, conversation text). The system text carries every instruction — the system
    messages, the tool schemas, and the tool-call convention; the conversation is the plain
    dialogue, labelled by role, that the model continues as the assistant."""
    systems: list[str] = []
    turns: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            systems.append(_content_text(msg.get("content")))
        elif role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            turns.append(f"TOOL RESULT ({name}): {_content_text(msg.get('content'))}")
        else:
            turns.append(_chat_turn(msg, role))
    blocks = [s for s in systems if s]
    if tools:
        blocks.append("TOOLS AVAILABLE (JSON schemas):\n" + json.dumps(tools, ensure_ascii=False))
        blocks.append(_TOOL_INSTRUCTION)
    if turns:
        blocks.append(_TRANSCRIPT_NOTE)
    return "\n\n".join(blocks), "\n\n".join(turns)


def split_for_cli(messages: list[dict], tools: list[dict] | None = None) -> tuple[str, str]:
    """For a CLI that takes a real system prompt (claude -p --system-prompt): instructions and
    tool schemas go in the system prompt, the labelled conversation goes on stdin. Keeping the
    instructions out of the user turn is what stops the CLI's model from reading the transcript
    as an injected fake conversation."""
    return _split(messages, tools)


def serialize_messages(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Flatten OpenAI-style messages (+ optional tools) into a single prompt string, for a CLI
    with no system-prompt flag (codex exec)."""
    system, conversation = _split(messages, tools)
    blocks = [f"SYSTEM:\n{system}" if system else "", conversation]
    return "\n\n".join(b for b in blocks if b)


def extract_json(text: str) -> str | None:
    """Return the first balanced JSON object/array substring in `text`, or None.

    Tolerates fenced ```json blocks and leading prose. Ignores braces inside strings. Fence
    stripping is tried first, then the raw text — so JSON whose own string values contain ```
    (generated code, a README) is still recovered when the fenced slice would be truncated.
    """
    if not text:
        return None
    for candidate in (_strip_fence(text), text):
        start = _first_open(candidate)
        if start is None:
            continue
        end = _match_bracket(candidate, start)
        if end is not None:
            return candidate[start : end + 1]
    return None


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


def _scan_string(ch: str, escape: bool) -> tuple[bool, bool]:
    """Consume one char inside a JSON string; return (still_in_string, escape_next)."""
    if escape:
        return True, False
    if ch == "\\":
        return True, True
    return ch != '"', False


def _scan_char(ch: str, in_str: bool, escape: bool, depth: int, open_ch: str, close_ch: str):
    """Consume one char of the bracket scan; return (in_str, escape, depth)."""
    if in_str:
        in_str, escape = _scan_string(ch, escape)
    elif ch == '"':
        in_str = True
    elif ch == open_ch:
        depth += 1
    elif ch == close_ch:
        depth -= 1
    return in_str, escape, depth


def _match_bracket(text: str, start: int) -> int | None:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        in_str, escape, depth = _scan_char(text[i], in_str, escape, depth, open_ch, close_ch)
        if depth == 0:
            return i
    return None


def _tool_calls_from(items: list) -> list[dict]:
    calls = []
    for tc in items:
        if not isinstance(tc, dict):
            continue
        args = tc.get("arguments", tc.get("args", {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": tc.get("id"), "name": tc.get("name"), "arguments": args})
    return calls


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
        calls = _tool_calls_from(obj["tool_calls"])
        return Reply(content=obj.get("content", "") or "", tool_calls=calls, raw=text)
    if want_json and obj_str is not None:
        return Reply(content=obj_str, raw=text)
    return Reply(content=text, raw=text)
