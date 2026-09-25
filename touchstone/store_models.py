"""The typed rows the trace store holds: plain dataclasses, no SQL.

`store.py` reads and writes these; `now()` is the shared UTC-ISO timestamp their defaults use. They
are re-exported from `touchstone.store`, so callers use `store.Episode`, `store.Span`, and friends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .ids import new_id


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Episode:
    name: str
    source: str = "app"
    started_at: str = field(default_factory=now)
    ended_at: str | None = None
    outcome_score: float | None = None
    outcome_label: str | None = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)


@dataclass
class Span:
    episode_id: str
    kind: str  # 'model' | 'tool'
    name: str
    parent_id: str | None = None
    model: str | None = None
    started_at: str = field(default_factory=now)
    ended_at: str | None = None
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    tool_call_id: str | None = None
    id: str = field(default_factory=new_id)


@dataclass
class Review:
    task: str  # the reviewed task directory name
    trial: str  # the trial (Harbor trial name) the room looked at
    verdict: str  # e.g. 'agree' | 'disagree'
    speaker: str
    note: str | None = None
    ts: str = field(default_factory=now)
    id: str = field(default_factory=new_id)


@dataclass
class Room:
    task_id: str | None
    topic: str
    created_at: str = field(default_factory=now)
    closed_at: str | None = None
    id: str = field(default_factory=new_id)


@dataclass
class RoomMessage:
    room_id: str
    speaker: str
    role: str
    text: str
    audio_path: str | None = None
    ts: str = field(default_factory=now)
    id: str = field(default_factory=new_id)
