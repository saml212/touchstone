"""FastAPI app for the interview: rooms, messages, push-to-talk audio, and a WebSocket feed.

Every request gets its own SQLite connection (a dependency) and never shares it across threads;
the agent turn — the one slow, blocking step — runs in a threadpool with a fresh connection of its
own. Room events fan out through the in-process `Hub` so a slow WebSocket client never blocks the
others, and a client that (re)connects always receives full room state first.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import store
from ..config import Settings, load_settings
from ..interview import rooms
from ..interview.agent import Interviewer
from ..interview.rooms import Event, Hub
from ..interview.speech import Speech, SpeechError, validate_audio
from .routes import ROUTERS

STATIC = Path(__file__).parent / "static"


def _default_provider_factory(settings: Settings):
    def make():
        from ..llm import provider_from_spec

        try:
            return provider_from_spec(settings.agent_provider, settings)
        except Exception:  # no key/binary: the interviewer degrades to plain questions
            return None

    return make


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Touchstone interview")
    app.state.settings = settings
    app.state.hub = Hub()
    app.state.speech = Speech.from_settings(settings)
    app.state.provider_factory = _default_provider_factory(settings)

    def get_conn():
        conn = store.connect(settings.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # -- reads ---------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/rooms/{room_id}")
    def get_room(room_id: str, conn=Depends(get_conn)) -> dict:
        state = _room_state(conn, room_id)
        if state is None:
            raise HTTPException(404, f"no room with id {room_id}")
        return state

    # -- writes --------------------------------------------------------------

    @app.post("/api/rooms")
    def create_room(body: dict, conn=Depends(get_conn)) -> dict:
        topic = (body.get("topic") or "").strip() or "quality review"
        room = rooms.open(conn, task_id=body.get("task_id"), topic=topic)
        opening = Interviewer(None, conn, room).open_statement()
        rooms.post(conn, room.id, "agent", "assistant", opening)
        return _room_state(conn, room.id)

    @app.post("/api/rooms/{room_id}/messages")
    async def post_message(room_id: str, body: dict) -> dict:
        text = (body.get("text") or "").strip()
        speaker = (body.get("speaker") or "").strip() or "guest"
        if not text:
            raise HTTPException(400, "message text is required")
        return await _ingest(app, room_id, speaker, text)

    @app.post("/api/rooms/{room_id}/audio")
    async def post_audio(
        room_id: str, speaker: str = Form("guest"), file: UploadFile = File(...)
    ) -> dict:
        data = await file.read()
        try:
            mime = validate_audio(data)
            text = await asyncio.to_thread(app.state.speech.stt.transcribe, data, mime)
        except SpeechError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not text.strip():
            raise HTTPException(400, "Transcription produced no text.")
        result = await _ingest(app, room_id, speaker.strip() or "guest", text.strip())
        result["transcript"] = text.strip()
        return result

    @app.get("/api/rooms/{room_id}/audio/{message_id}")
    def get_audio(room_id: str, message_id: str, conn=Depends(get_conn)):
        msg = next(
            (m for m in store.list_room_messages(conn, room_id) if m.id == message_id), None
        )
        if msg is None:
            raise HTTPException(404, "no such message")
        if msg.audio_path and Path(msg.audio_path).exists():
            return FileResponse(msg.audio_path)
        if msg.role != "assistant":
            raise HTTPException(404, "no audio for this message")
        spoken = app.state.speech.tts.synthesize(msg.text)
        if spoken is None:
            return Response(status_code=204)  # browser speaks it via speechSynthesis
        audio, media_type = spoken
        return Response(content=audio, media_type=media_type)

    @app.post("/api/rooms/{room_id}/close")
    def close_room(room_id: str, conn=Depends(get_conn)) -> dict:
        if store.get_room(conn, room_id) is None:
            raise HTTPException(404, f"no room with id {room_id}")
        rooms.close(conn, room_id)
        app.state.hub.publish(room_id, Event("closed", {}))
        return _room_state(conn, room_id)

    # -- websocket -----------------------------------------------------------

    @app.websocket("/ws/rooms/{room_id}")
    async def room_feed(websocket: WebSocket, room_id: str) -> None:
        await websocket.accept()
        conn = store.connect(settings.db_path)
        try:
            state = _room_state(conn, room_id)
        finally:
            conn.close()
        if state is None:
            await websocket.close(code=4004)
            return
        await websocket.send_json({"type": "state", "data": state})

        hub: Hub = app.state.hub
        queue = hub.subscribe(room_id)
        pump = asyncio.ensure_future(_pump(websocket, queue))
        try:
            while True:
                message = await websocket.receive()  # client payloads ignored; watch for close
                if message["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            pump.cancel()
            hub.unsubscribe(room_id, queue)

    # -- static UI -----------------------------------------------------------

    _NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=_NO_STORE)

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers=_NO_STORE)

    @app.get("/app.css")
    def app_css() -> FileResponse:
        return FileResponse(STATIC / "app.css", media_type="text/css", headers=_NO_STORE)

    @app.get("/rooms/{room_id}")
    def room_page(room_id: str) -> FileResponse:
        return FileResponse(STATIC / "room.html")

    for router in ROUTERS:
        app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


# ---- shared request handling ----------------------------------------------


async def _ingest(app: FastAPI, room_id: str, speaker: str, text: str) -> dict:
    """Post a participant message, run the agent in a thread, and broadcast every event."""
    settings: Settings = app.state.settings
    hub: Hub = app.state.hub

    conn = store.connect(settings.db_path)
    try:
        if store.get_room(conn, room_id) is None:
            raise HTTPException(404, f"no room with id {room_id}")
        user_msg = rooms.post(conn, room_id, speaker, "user", text)
        history = [_msg_view(m) for m in store.list_room_messages(conn, room_id)]
        user_view = _msg_view(user_msg)
    finally:
        conn.close()
    hub.publish(room_id, Event("message", user_view))

    step = await asyncio.to_thread(
        _agent_step, settings, app.state.provider_factory, room_id, history
    )

    hub.publish(room_id, Event("message", step["agent"]))
    hub.publish(room_id, Event("draft", {"checks": step["draft"]}))
    if step["turn"]["commit"]:
        hub.publish(room_id, Event("committed", {"checks": step["turn"]["commit"]}))
    if step["closed"]:
        hub.publish(room_id, Event("closed", {}))
    return {"user": user_view, **step}


def _agent_step(settings: Settings, provider_factory, room_id: str, history: list[dict]) -> dict:
    conn = store.connect(settings.db_path)
    try:
        room = store.get_room(conn, room_id)
        agent = Interviewer(provider_factory(), conn, room)
        turn = agent.respond(history)
        agent_msg = rooms.post(conn, room_id, "agent", "assistant", turn.say)
        closed = store.get_room(conn, room_id).closed_at is not None
        return {
            "turn": turn.to_dict(),
            "agent": _msg_view(agent_msg),
            "draft": _draft_view(conn, room_id),
            "committed": _committed_view(conn, room_id),
            "closed": closed,
        }
    finally:
        conn.close()


async def _pump(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        event: Event = await queue.get()
        await websocket.send_json({"type": event.type, "data": event.data})


# ---- views -----------------------------------------------------------------


def _msg_view(m: store.RoomMessage) -> dict:
    return {"id": m.id, "speaker": m.speaker, "role": m.role, "text": m.text,
            "audio_path": m.audio_path, "ts": m.ts,
            "has_audio": bool(m.audio_path) or m.role == "assistant"}


def _check_view(c: store.Check) -> dict:
    return {"id": c.id, "name": c.name, "kind": c.kind, "params": c.params,
            "applies_to": c.applies_to, "severity": c.severity, "rationale": c.rationale,
            "enabled": bool(c.enabled)}


def _room_checks(conn, room_id: str, enabled: bool) -> list[dict]:
    out = []
    for cid in store.list_room_check_ids(conn, room_id):
        c = store.get_check(conn, cid)
        if c and bool(c.enabled) == enabled:
            out.append(_check_view(c))
    return out


def _draft_view(conn, room_id: str) -> list[dict]:
    return _room_checks(conn, room_id, enabled=False)


def _committed_view(conn, room_id: str) -> list[dict]:
    return _room_checks(conn, room_id, enabled=True)


def _room_state(conn, room_id: str) -> dict | None:
    room = store.get_room(conn, room_id)
    if room is None:
        return None
    return {
        "room": asdict(room),
        "messages": [_msg_view(m) for m in store.list_room_messages(conn, room_id)],
        "draft": _draft_view(conn, room_id),
        "committed": _committed_view(conn, room_id),
    }


__all__ = ["create_app"]
