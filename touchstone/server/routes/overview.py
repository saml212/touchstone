"""GET /api/overview — the counts the landing page shows and drives its next-step hint from."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from ... import store
from ._deps import get_conn

router = APIRouter()


@router.get("/api/overview")
def overview(conn=Depends(get_conn)) -> dict:
    episodes = store.list_episodes(conn)
    outcomes = Counter(e.outcome_label or "unlabeled" for e in episodes)
    spans = sum(len(store.list_spans(conn, e.id)) for e in episodes)
    checks = store.list_checks(conn)
    rooms = store.list_rooms(conn)
    return {
        "episodes": {"total": len(episodes), "outcomes": dict(outcomes)},
        "spans": spans,
        "checks": {
            "total": len(checks),
            "enabled": sum(1 for c in checks if c.enabled),
            "by_source": dict(Counter(c.source for c in checks)),
        },
        "tasks": len(store.list_tasks(conn)),
        "benchmarks": len(store.list_benchmarks(conn)),
        "runs": len(store.list_runs(conn)),
        "rooms": {"total": len(rooms), "open": sum(1 for r in rooms if r.closed_at is None)},
    }
