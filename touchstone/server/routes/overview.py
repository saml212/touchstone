"""GET /api/overview — the counts the landing page shows and drives its next-step hint from."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from ... import policies as policies_mod
from ... import store
from ... import tasks as tasks_mod
from ...bench import benchmark
from ._deps import get_conn, get_root

router = APIRouter()

_TASK_QUEUES = ("active", "needs_checks", "needs_solution")


@router.get("/api/overview")
def overview(conn=Depends(get_conn), root=Depends(get_root)) -> dict:
    episodes = store.list_episodes(conn)
    outcomes = Counter(e.outcome_label or "unlabeled" for e in episodes)
    spans = sum(len(store.list_spans(conn, e.id)) for e in episodes)
    checks = policies_mod.read_policies(root)
    tasks = tasks_mod.list_tasks(root)
    by_status = {q: sum(1 for t in tasks if t.status == q) for q in _TASK_QUEUES}
    rooms = store.list_rooms(conn)
    return {
        "episodes": {"total": len(episodes), "outcomes": dict(outcomes)},
        "spans": spans,
        "checks": {
            "total": len(checks),
            "enabled": sum(1 for c in checks if c.enabled),
            "by_source": dict(Counter(c.check.source for c in checks)),
        },
        "tasks": {"total": len(tasks), "active": by_status["active"], "by_status": by_status},
        "benchmarks": len(benchmark.list_names(root)),
        "runs": len(store.list_runs(conn)),
        "rooms": {"total": len(rooms), "open": sum(1 for r in rooms if r.closed_at is None)},
    }
