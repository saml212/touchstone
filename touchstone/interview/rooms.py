"""Interview rooms: store-backed lifecycle plus in-process pub/sub for WebSocket fan-out.

Room rows and messages live in SQLite; the subscriber set lives only in the server process, so
events reach the WebSocket clients connected to *this* process. Each subscriber owns a bounded
asyncio queue: a slow or stalled client fills its own queue and starts dropping events, but never
blocks the broadcast to everyone else. A client that reconnects gets full state again over HTTP/WS,
so a dropped event is never fatal.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .. import store


def open(conn, task_id: str | None, topic: str) -> store.Room:
    return store.insert_room(conn, store.Room(task_id=task_id, topic=topic))


def post(conn, room_id, speaker, role, text, audio_path=None) -> store.RoomMessage:
    return store.insert_room_message(
        conn,
        store.RoomMessage(
            room_id=room_id, speaker=speaker, role=role, text=text, audio_path=audio_path
        ),
    )


def close(conn, room_id) -> None:
    store.close_room(conn, room_id)


@dataclass
class Event:
    type: str  # "message" | "draft" | "committed" | "closed"
    data: dict


class Hub:
    """Per-room fan-out of events to the WebSocket connections in this process."""

    def __init__(self, maxsize: int = 100) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._maxsize = maxsize

    def subscribe(self, room_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.setdefault(room_id, set()).add(queue)
        return queue

    def unsubscribe(self, room_id: str, queue: asyncio.Queue) -> None:
        subs = self._subs.get(room_id)
        if not subs:
            return
        subs.discard(queue)
        if not subs:
            self._subs.pop(room_id, None)

    def publish(self, room_id: str, event: Event) -> int:
        """Deliver to every live subscriber; drop for any whose queue is full. Returns delivered."""
        delivered = 0
        for queue in list(self._subs.get(room_id, ())):
            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                pass
        return delivered

    def subscriber_count(self, room_id: str) -> int:
        return len(self._subs.get(room_id, ()))
