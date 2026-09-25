"""Read the recorded model spans for packaging: model id, system prompt, and the first user turn.

These pull the agent's defaults straight out of the recordings — the most-common model id and the
first system prompt seed `agent.toml`; the first user turn drives the packaged adapter check.
"""

from __future__ import annotations

from collections import Counter

from .. import store


def _messages(span) -> list:
    return (span.input or {}).get("messages", [])


def _role_content(span, role: str) -> str:
    for msg in _messages(span):
        if msg.get("role") == role:
            return str(msg.get("content") or "")
    return ""


def _model_spans(conn):
    for ep in store.list_episodes(conn):
        for span in store.list_spans(conn, ep.id):
            if span.kind == "model":
                yield span


def recorded_model_and_system(conn) -> tuple[str, str]:
    """The most-common recorded model id and the first recorded system prompt (full text)."""
    models: Counter = Counter()
    system = ""
    for span in _model_spans(conn):
        if span.model:
            models[span.model] += 1
        system = system or _role_content(span, "system")
    model = models.most_common(1)[0][0] if models else ""
    return model, system


def first_user_turn(conn) -> str:
    for span in _model_spans(conn):
        turn = _role_content(span, "user")
        if turn:
            return turn
    return ""
