"""Touchstone public API — the one-liners an instrumented app imports.

    import touchstone; touchstone.trace()

`trace()` wires capture to the SQLite db and patches whichever of openai/anthropic is importable.
It never raises if neither is installed.
"""

from __future__ import annotations

import json
import logging

from . import capture
from .capture import episode, tool
from .capture.patch_anthropic import patch as _patch_anthropic
from .capture.patch_openai import patch as _patch_openai
from .config import load_settings
from .messages import canonical

__all__ = ["trace", "episode", "outcome", "tool", "record_llm_call"]

_log = logging.getLogger("touchstone")
_traced = False


def trace(db: str | None = None, otel: bool = False) -> dict:
    """Idempotent. Point capture at `db` (or configured/default) and patch installed SDKs.

    With `otel=True`, also register the OpenInference OTel span exporter (needs the
    `touchstone[otel]` extra) so existing OpenInference spans land in the store.
    """
    global _traced
    settings = load_settings()
    db_path = db or settings.db_path
    capture.configure(db_path)
    from .survey.netshim import install_from_env

    install_from_env()  # no-op in prod; rewrites a hardcoded host if TOUCHSTONE_SIMULATORS is set
    patched = []
    if _patch_openai():
        patched.append("openai")
    if _patch_anthropic():
        patched.append("anthropic")
    if otel:
        from .capture.openinference import register

        if register():
            patched.append("openinference")
    _traced = True
    return {"db": db_path, "patched": patched}


def outcome(score: float | None, label: str | None) -> None:
    try:  # resolving the current/untracked episode can touch the db; never raise into the app
        capture.current_episode().outcome(score, label)
    except Exception as exc:
        _log.warning("touchstone outcome failed: %r", exc)


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
    try:  # capture must never raise into the app (db unwritable, bad reply shape, …)
        _record_llm_call(model, messages, reply, tools, usage)
    except Exception as exc:
        _log.warning("touchstone record_llm_call failed: %r", exc)


def _record_llm_call(model, messages, reply, tools, usage) -> None:
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
