"""Episode list (filter + paginate) and detail (episode plus its spans in recorded order)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from ... import store
from ._deps import get_conn, paginate

router = APIRouter()


@router.get("/api/episodes")
def list_episodes(
    label: str | None = None,
    limit: int | None = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
) -> dict:
    episodes = store.list_episodes(conn, label)
    page = paginate(episodes, limit, offset)
    return {"total": len(episodes), "episodes": [asdict(e) for e in page]}


@router.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str, conn=Depends(get_conn)) -> dict:
    episode = store.get_episode(conn, episode_id)
    if episode is None:
        raise HTTPException(404, f"no episode with id {episode_id}")
    spans = [asdict(s) for s in store.list_spans(conn, episode_id)]
    return {"episode": asdict(episode), "spans": spans}
