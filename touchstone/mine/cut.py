"""Cut episodes into replay tasks.

Every recorded assistant (LLM) turn becomes a task: the messages leading up to it are the replay
context and the recorded assistant message is the reference. A check is attached to a task only when
the reference passes it, so a candidate is never asked to satisfy a check the recorded behaviour did
not (a tool call that already happened earlier in the context, a phrase from a different turn).
Safety checks ("avoid this") always attach; failure tasks (bad outcome) get only those, since their
reference is not to be trusted.
Idempotent on `(episode_id, cut_span_id)`: re-running refreshes `check_ids` instead of duplicating.
"""

from __future__ import annotations

from .. import store
from ..checks import Check, Target, evaluate

# Checks whose meaning is "avoid this"; safe to attach even when the reference output was bad.
_SAFETY_KINDS = frozenset({"no_pii", "not_contains", "not_regex", "tool_not_called"})


def _is_failure(ep: store.Episode) -> bool:
    return ep.outcome_score is not None and ep.outcome_score < 0.5


def _reference(span: store.Span) -> dict:
    msg = (span.output or {}).get("message") or {}
    return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []}


def _reference_target(reference: dict, messages: list[dict]) -> Target:
    return Target(
        output_text=reference["content"],
        tool_calls=reference["tool_calls"],
        messages=messages,
        reference=reference,
    )


def _attachable(
    checks: list[Check], reference: dict, messages: list[dict], failure: bool
) -> list[str]:
    if failure:
        return [c.id for c in checks if c.kind in _SAFETY_KINDS]
    target = _reference_target(reference, messages)
    return [
        c.id
        for c, r in zip(checks, evaluate(checks, target), strict=True)
        if c.kind in _SAFETY_KINDS or r.passed or (c.kind == "judge" and reference["content"])
    ]


def cut_tasks(conn, episodes: list[store.Episode]) -> list[store.Task]:
    enabled = [Check.from_dict(c.__dict__) for c in store.list_checks(conn, enabled=True)]
    existing = {(t.episode_id, t.cut_span_id): t for t in store.list_tasks(conn)}

    tasks: list[store.Task] = []
    for ep in episodes:
        spans = store.list_spans(conn, ep.id)
        tool_names = sorted({s.name for s in spans if s.kind == "tool"})
        failure = _is_failure(ep)
        tags = (
            ([ep.outcome_label] if ep.outcome_label else [])
            + tool_names
            + (["failure"] if failure else [])
        )

        for i, span in enumerate(s for s in spans if s.kind == "llm"):
            ctx = span.input or {}
            messages = ctx.get("messages", [])
            reference = _reference(span)
            check_ids = _attachable(enabled, reference, messages, failure)
            key = (ep.id, span.id)
            if key in existing:
                store.update_task(conn, existing[key].id, check_ids=check_ids)
                tasks.append(store.get_task(conn, existing[key].id))
                continue
            tasks.append(
                store.insert_task(
                    conn,
                    store.Task(
                        name=f"{ep.name}#{i}",
                        episode_id=ep.id,
                        cut_span_id=span.id,
                        context={"messages": messages, "tools": ctx.get("tools", [])},
                        reference=reference,
                        check_ids=check_ids,
                        kind="replay",
                        tags=tags,
                    ),
                )
            )
    return tasks
