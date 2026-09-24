"""Cut episodes into replay tasks.

Every recorded assistant (LLM) turn becomes a `Task`: the messages leading up to it are the replay
context, the recorded assistant message is the reference. Checks are attached separately by
`policies.materialize` (the reference gate); failure episodes carry safety checks only. This module
just builds the `Task` objects — writing them to directories is `write_task`'s job.
"""

from __future__ import annotations

from collections import Counter

from .. import store
from ..tasks import Task, _slug, task_name


def _is_failure(ep: store.Episode) -> bool:
    return ep.outcome_score is not None and ep.outcome_score < 0.5


def _reference(span: store.Span) -> dict:
    msg = (span.output or {}).get("message") or {}
    return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []}


def build_tasks(conn, episodes: list[store.Episode]) -> list[Task]:
    """One Task per recorded assistant turn across `episodes` (no checks, no I/O).

    Task names are `<episode-slug>-turn-<n>`; episodes that share a slug are disambiguated by a
    short span-id suffix so directory names stay unique yet readable.
    """
    slug_counts = Counter(_slug(ep.name) for ep in episodes)
    tasks: list[Task] = []
    for ep in episodes:
        spans = store.list_spans(conn, ep.id)
        tool_names = sorted({s.name for s in spans if s.kind == "tool"})
        failure = _is_failure(ep)
        tags = (
            ([ep.outcome_label] if ep.outcome_label else [])
            + tool_names
            + (["failure"] if failure else [])
        )
        disambiguate = slug_counts[_slug(ep.name)] > 1
        model_spans = [s for s in spans if s.kind == "model"]
        for turn, span in enumerate(model_spans, start=1):
            ctx = span.input or {}
            tasks.append(Task(
                name=task_name(ep, span, turn=turn, disambiguate=disambiguate),
                episode_id=ep.id,
                cut_span_id=span.id,
                kind="replay",
                tags=tags,
                description=f"Reply as the assistant in {ep.name}.",
                context={"messages": ctx.get("messages", []), "tools": ctx.get("tools", [])},
                reference=_reference(span),
            ))
    return tasks
