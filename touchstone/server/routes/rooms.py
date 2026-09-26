"""Interview rooms: list, read, create, message, push-to-talk audio, close, and the WebSocket feed.

Every request gets its own SQLite connection (the `get_conn` dependency) and never shares it across
threads; the agent turn — the one slow, blocking step — runs in a threadpool with a fresh connection
of its own. Committed checks are written to task/checks files; the draft lives in room state. Room
events fan out through the in-process `Hub` so a slow WebSocket client never blocks the others.
"""

from __future__ import annotations

import asyncio
import getpass
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response

from ... import store
from ...config import Settings
from ...interview import rooms
from ...interview.realtime import Bridges
from ...interview.rooms import Event, Hub
from ...interview.speech import SpeechError, validate_audio
from ...llm._http import ProviderError
from ...review.agent import AgentTurn, ReviewAgent
from ._deps import get_conn, get_settings

router = APIRouter()


@router.get("/api/rooms")
def list_rooms(conn=Depends(get_conn)) -> dict:
    """Every interview room, open ones first, each with its task name."""
    all_rooms = store.list_rooms(conn)
    all_rooms.sort(key=lambda r: (r.closed_at is not None, r.id))
    out = [{
        "id": r.id, "task_id": r.task_id, "task_name": r.task_id,
        "topic": r.topic, "created_at": r.created_at, "closed_at": r.closed_at,
    } for r in all_rooms]
    return {"rooms": out}


@router.get("/api/rooms/{room_id}")
def get_room(room_id: str, conn=Depends(get_conn), settings=Depends(get_settings)) -> dict:
    state = _room_state(conn, settings, room_id)
    if state is None:
        raise HTTPException(404, f"no room with id {room_id}")
    return state


@router.post("/api/rooms")
def create_room(body: dict, conn=Depends(get_conn), settings=Depends(get_settings)) -> dict:
    topic = (body.get("topic") or "").strip() or "review"
    task_id = body.get("task_id") or body.get("task")
    room = rooms.open(conn, task_id=task_id, topic=topic)
    opening = ReviewAgent(None, conn, room, settings).open_statement()
    rooms.post(conn, room.id, "agent", "assistant", opening)
    return _room_state(conn, settings, room.id)


@router.post("/api/rooms/{room_id}/messages")
async def post_message(room_id: str, body: dict, request: Request) -> dict:
    text = (body.get("text") or "").strip()
    speaker = (body.get("speaker") or "").strip() or "guest"
    if not text:
        raise HTTPException(400, "message text is required")
    return await _ingest(request.app, room_id, speaker, text)


@router.post("/api/rooms/{room_id}/audio")
async def post_audio(
    room_id: str, request: Request, speaker: str = Form("guest"), file: UploadFile = File(...),
    conn=Depends(get_conn),
) -> dict:
    room = store.get_room(conn, room_id)
    if room is None:
        raise HTTPException(404, f"no room with id {room_id}")
    if room.closed_at is not None:
        raise HTTPException(409, "this room is closed")
    data = await file.read()
    try:
        mime = validate_audio(data)
        text = await asyncio.to_thread(request.app.state.speech.stt.transcribe, data, mime)
    except SpeechError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not text.strip():
        raise HTTPException(400, "Transcription produced no text.")
    result = await _ingest(request.app, room_id, speaker.strip() or "guest", text.strip())
    result["transcript"] = text.strip()
    return result


@router.get("/api/rooms/{room_id}/audio/{message_id}")
def get_audio(room_id: str, message_id: str, request: Request, conn=Depends(get_conn)):
    msg = next(
        (m for m in store.list_room_messages(conn, room_id) if m.id == message_id), None
    )
    if msg is None:
        raise HTTPException(404, "no such message")
    if msg.audio_path and Path(msg.audio_path).exists():
        return FileResponse(msg.audio_path)
    if msg.role != "assistant":
        raise HTTPException(404, "no audio for this message")
    spoken = request.app.state.speech.tts.synthesize(msg.text)
    if spoken is None:
        return Response(status_code=204)  # browser speaks it via speechSynthesis
    audio, media_type = spoken
    return Response(content=audio, media_type=media_type)


@router.post("/api/rooms/{room_id}/close")
async def close_room(room_id: str, request: Request, conn=Depends(get_conn),
                     settings=Depends(get_settings)) -> dict:
    if store.get_room(conn, room_id) is None:
        raise HTTPException(404, f"no room with id {room_id}")
    rooms.close(conn, room_id)
    await request.app.state.bridges.close(room_id)
    request.app.state.hub.publish(room_id, Event("closed", {}))
    return _room_state(conn, settings, room_id)


@router.websocket("/ws/rooms/{room_id}")
async def room_feed(websocket: WebSocket, room_id: str) -> None:
    await websocket.accept()
    settings: Settings = websocket.app.state.settings
    conn = store.connect(settings.db_path)
    try:
        state = _room_state(conn, settings, room_id)
    finally:
        conn.close()
    if state is None:
        await websocket.close(code=4004)
        return
    await websocket.send_json({"type": "state", "data": state})

    hub: Hub = websocket.app.state.hub
    bridges: Bridges = websocket.app.state.bridges
    realtime = settings.voice_mode() == "realtime"
    if realtime:
        bridges.client_here(room_id)
    queue = hub.subscribe(room_id)
    pump = asyncio.ensure_future(_pump(websocket, queue))
    try:
        await _recv_loop(websocket, bridges, room_id, realtime)
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()
        hub.unsubscribe(room_id, queue)
        if realtime:
            bridges.client_gone(room_id)


async def _recv_loop(websocket: WebSocket, bridges: Bridges, room_id: str, realtime: bool) -> None:
    """Watch for the client's close; in realtime mode route its audio/ptt payloads to the bridge."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if realtime and message.get("text"):
            await _route_client(bridges, room_id, message["text"])


async def _route_client(bridges: Bridges, room_id: str, text: str) -> None:
    """Route a browser WS payload (push-to-talk audio / ptt state) to the room's realtime bridge."""
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return
    kind = msg.get("type")
    if kind not in ("audio", "ptt"):
        return
    bridge = await bridges.get(room_id)
    if bridge is None:
        return
    if kind == "audio" and msg.get("b64"):
        await bridge.append_audio(msg["b64"])
    elif kind == "ptt":
        await bridge.ptt(msg.get("state", ""), msg.get("speaker", "guest"))


# ---- shared request handling ----------------------------------------------


async def ingest_turn(app, room_id: str, speaker: str, text: str) -> dict:
    """Public entry for a participant turn (text, local speech, or realtime voice): all three go
    through the same ReviewAgent path so voice is never a second, weaker agent."""
    return await _ingest(app, room_id, speaker, text)


async def _ingest(app, room_id: str, speaker: str, text: str) -> dict:
    """Post a participant message, run the agent in a thread, and broadcast every event."""
    settings: Settings = app.state.settings
    hub: Hub = app.state.hub

    conn = store.connect(settings.db_path)
    try:
        room = store.get_room(conn, room_id)
        if room is None:
            raise HTTPException(404, f"no room with id {room_id}")
        if room.closed_at is not None:
            raise HTTPException(409, "this room is closed")
        user_msg = rooms.post(conn, room_id, speaker, "user", text)
        history = _history(conn, room_id)
        user_view = _msg_view(user_msg)
    finally:
        conn.close()
    hub.publish(room_id, Event("message", user_view))

    loop = asyncio.get_running_loop()

    def on_status(state: str | None) -> None:
        # Called from the agent thread while a regrade runs; hop to the loop to touch the hub.
        loop.call_soon_threadsafe(hub.publish, room_id, Event("status", {"state": state}))

    step = await asyncio.to_thread(
        _agent_step, settings, app.state.provider_factory, room_id, history, on_status
    )

    hub.publish(room_id, Event("message", step["agent"]))
    hub.publish(room_id, Event("draft", {"change": step["draft"]}))
    if step["turn"]["commit"]:
        hub.publish(room_id, Event("committed", {"applied": step["turn"]["commit"]}))
    if step["closed"]:
        hub.publish(room_id, Event("closed", {}))
    # A full-state event closes every turn so the page re-renders the left panel, trust, and chips
    # authoritatively — a dropped or reordered delta can never leave the DOM stale.
    conn = store.connect(settings.db_path)
    try:
        state = _room_state(conn, settings, room_id)
    finally:
        conn.close()
    hub.publish(room_id, Event("state", state))
    return {"user": user_view, **step}


def _agent_step(settings: Settings, provider_factory, room_id: str, history: list[dict],
                on_status=None) -> dict:
    conn = store.connect(settings.db_path)
    try:
        room = store.get_room(conn, room_id)
        agent = ReviewAgent(provider_factory(), conn, room, settings, on_status=on_status)
        turn = _safe_respond(agent, conn, room_id, history)
        agent_msg = rooms.post(conn, room_id, "agent", "assistant", turn.say)
        closed = store.get_room(conn, room_id).closed_at is not None
        return {
            "turn": turn.to_dict(),
            "agent": _msg_view(agent_msg),
            "draft": agent.draft(),
            "committed": agent.committed(),
            "review": agent.review_state(),
            "closed": closed,
        }
    finally:
        conn.close()


_PROVIDER_DOWN = "I hit a problem generating a reply — say that again and I'll continue."


def _safe_respond(agent, conn, room_id: str, history: list[dict]) -> AgentTurn:
    """Run the agent turn; a provider failure becomes a visible reply (with the error in the room
    log), never a 500 or a silent room. Anything the tools committed first (a review, a regrade)
    stays committed."""
    try:
        return agent.respond(history)
    except ProviderError as exc:
        rooms.post(conn, room_id, "system", "note",
                   f"(provider error: {str(exc).splitlines()[0]})")
        return AgentTurn(say=_PROVIDER_DOWN)


async def _pump(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        event: Event = await queue.get()
        await websocket.send_json({"type": event.type, "data": event.data})


# ---- views -----------------------------------------------------------------


def _current_user() -> str:
    """The local OS user, so the page can name the speaker without a join card ('you' fallback)."""
    try:
        return getpass.getuser() or "you"
    except Exception:
        return "you"


def _msg_view(m: store.RoomMessage) -> dict:
    return {"id": m.id, "speaker": m.speaker, "role": m.role, "text": m.text,
            "audio_path": m.audio_path, "ts": m.ts,
            "has_audio": bool(m.audio_path) or m.role == "assistant"}


def _history(conn, room_id: str) -> list[dict]:
    """Conversation messages (the draft snapshots are room state, not conversation)."""
    return [_msg_view(m) for m in store.list_room_messages(conn, room_id) if m.role != "draft"]


def _room_state(conn, settings: Settings, room_id: str) -> dict | None:
    room = store.get_room(conn, room_id)
    if room is None:
        return None
    agent = ReviewAgent(None, conn, room, settings)
    return {
        "room": asdict(room),
        "mode": settings.voice_mode(),
        "you": _current_user(),
        "messages": _history(conn, room_id),
        "draft": agent.draft(),
        "committed": agent.committed(),
        "review": agent.review_state(),
    }
