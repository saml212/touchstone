"""GET /api/overview — the counts the landing page shows and its next-step hint."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from ... import overview as overview_mod
from ... import store
from ._deps import get_conn

router = APIRouter()


@router.get("/api/overview")
def overview(conn=Depends(get_conn)) -> dict:
    episodes = store.list_episodes(conn)
    outcomes = Counter(e.outcome_label or "unlabeled" for e in episodes)
    spans = sum(len(store.list_spans(conn, e.id)) for e in episodes)
    rooms = store.list_rooms(conn)
    return {
        "episodes": {"total": len(episodes), "outcomes": dict(outcomes)},
        "spans": spans,
        "rooms": {"total": len(rooms), "open": sum(1 for r in rooms if r.closed_at is None)},
        "next_step": overview_mod.next_step(overview_mod.project_signals(conn)),
    }
