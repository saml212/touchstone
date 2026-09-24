"""Cut episodes into replay tasks.

Each recorded assistant (LLM) turn can become a task: the messages leading up to it are the replay
context, the recorded assistant message is the reference, and enabled checks are attached. Failure
tasks (bad outcome) never get reference-derived checks, only safety ones. Idempotent on
`(episode_id, cut_span_id)`: re-running refreshes `check_ids` instead of duplicating tasks.
"""

from __future__ import annotations

from .. import store

# Checks whose meaning is "avoid this"; safe to attach even when the reference output was bad.
_SAFETY_KINDS = frozenset({"no_pii", "not_contains", "not_regex", "tool_not_called"})


def _is_good(ep: store.Episode) -> bool | None:
    if ep.outcome_score is None:
        return None
    return ep.outcome_score >= 0.5


def _cut_spans(spans: list[store.Span], every_turn: bool) -> list[store.Span]:
    llm = [s for s in spans if s.kind == "llm"]
    if not llm:
        return []
    return llm if every_turn else [llm[-1]]


def _reference(span: store.Span) -> dict:
    msg = (span.output or {}).get("message") or {}
    return {"content": msg.get("content", ""), "tool_calls": msg.get("tool_calls") or []}


def _applicable_checks(checks, is_failure: bool) -> list[str]:
    ids = []
    for c in checks:
        if is_failure and c.kind not in _SAFETY_KINDS:
            continue
        ids.append(c.id)
    return ids


def cut_tasks(conn, episodes: list[store.Episode], every_turn: bool = False) -> list[store.Task]:
    enabled = store.list_checks(conn, enabled=True)
    existing = {(t.episode_id, t.cut_span_id): t for t in store.list_tasks(conn)}

    tasks: list[store.Task] = []
    for ep in episodes:
        spans = store.list_spans(conn, ep.id)
        cut_spans = _cut_spans(spans, every_turn)
        tool_names = sorted({s.name for s in spans if s.kind == "tool"})
        good = _is_good(ep)
        is_failure = good is False

        tags = list(tool_names)
        if ep.outcome_label:
            tags.insert(0, ep.outcome_label)
        if is_failure:
            tags.append("failure")

        check_ids = _applicable_checks(enabled, is_failure)

        for i, span in enumerate(cut_spans):
            key = (ep.id, span.id)
            if key in existing:
                store.update_task(conn, existing[key].id, check_ids=check_ids)
                tasks.append(store.get_task(conn, existing[key].id))
                continue
            ctx = span.input or {}
            task = store.insert_task(conn, store.Task(
                name=f"{ep.name}#{i}",
                episode_id=ep.id,
                cut_span_id=span.id,
                context={"messages": ctx.get("messages", []), "tools": ctx.get("tools", [])},
                reference=_reference(span),
                check_ids=check_ids,
                kind="replay",
                tags=tags,
            ))
            tasks.append(task)
    return tasks
