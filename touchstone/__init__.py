"""Touchstone public API — the one-liners an instrumented app imports.

    import touchstone; touchstone.trace()

`trace()` wires capture to the SQLite db and patches whichever of openai/anthropic is importable.
It never raises if neither is installed.
"""

from __future__ import annotations

import json

from . import capture
from .capture import episode, tool
from .capture.patch_anthropic import patch as _patch_anthropic
from .capture.patch_openai import patch as _patch_openai
from .config import load_settings
from .messages import canonical

__all__ = ["trace", "episode", "outcome", "tool", "record_llm_call"]

_traced = False


def trace(db: str | None = None) -> dict:
    """Idempotent. Point capture at `db` (or configured/default) and patch installed SDKs."""
    global _traced
    settings = load_settings()
    db_path = db or settings.db_path
    capture.configure(db_path)
    patched = []
    if _patch_openai():
        patched.append("openai")
    if _patch_anthropic():
        patched.append("anthropic")
    _traced = True
    return {"db": db_path, "patched": patched}


def outcome(score: float | None, label: str | None) -> None:
    capture.current_episode().outcome(score, label)


def _normalize_calls(items) -> list[dict]:
    calls = []
    for tc in items:
        args = tc.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": tc.get("id"), "name": tc.get("name"), "arguments": args})
    return calls


def _normalize_reply(reply) -> tuple[str, list[dict], dict | None]:
    from .llm.base import Reply

    if isinstance(reply, Reply):
        reply = {"content": reply.content, "tool_calls": reply.tool_calls, "usage": reply.usage}
    if isinstance(reply, str):
        return reply, [], None
    if isinstance(reply, dict):
        calls = _normalize_calls(reply.get("tool_calls") or [])
        return reply.get("content", "") or "", calls, reply.get("usage")
    return str(reply), [], None


def record_llm_call(model, messages, reply, tools=None, usage=None) -> None:
    """Record an llm call made outside a patched SDK (e.g. via a Provider)."""
    content, tool_calls, reply_usage = _normalize_reply(reply)
    u = usage or reply_usage or {}
    convo = canonical(
        list(messages) + [{"role": "assistant", "content": content, "tool_calls": tool_calls}]
    )
    capture.add_span(
        "model",
        model or "model",
        model=model,
        input={"messages": convo[:-1], "tools": tools or [], "params": {}},
        output={"message": convo[-1]},
        tokens_in=u.get("tokens_in"),
        tokens_out=u.get("tokens_out"),
    )
