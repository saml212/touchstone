"""Realtime voice bridge: one OpenAI Realtime session per review room, as an STT+TTS front-end.

The server holds the key and opens a single WebSocket to OpenAI per open room. Browsers stream
push-to-talk PCM16 over the room WebSocket; the bridge appends it to the session. The Realtime model
only transcribes (`create_response` is off) — it never answers on its own. Each completed user
transcription is fed to the SAME `ReviewAgent` the text and local-speech paths use, via the injected
`respond` callback; that runs the review tools (record/propose/apply-change → regrade), posts the
user + agent messages, and publishes room state. The bridge then asks the session to voice the
ReviewAgent's reply. So the review agent is always the source of truth; voice is just the surface.

Any OpenAI error posts one room message and falls back to local mode for that room — never a 500.
A dropped socket reconnects once; a second drop falls back too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable

import websockets

from .. import store
from ..config import Settings
from ..llm.keychain import secret
from . import rooms
from .rooms import Event, Hub

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# respond(room_id, speaker, text) -> the ReviewAgent's reply text (or None). Runs the same agent
# turn the text path runs (tools + hub events); injected by the server so voice == text.
Respond = Callable[[str, str, str], Awaitable[str | None]]

_REALTIME_GUIDANCE = (
    "You are the voice of Touchstone's review agent. You transcribe what product people say and "
    "read back replies you are given, verbatim and naturally. Do not answer on your own or invent "
    "review verdicts — the review agent decides. Here is what the room is reviewing:\n"
)

# No realtime tool surface: the review tools run inside ReviewAgent on the text path, not here.
TOOLS: list[dict] = []


def realtime_available(settings: Settings) -> bool:
    """True when realtime mode is on and an OpenAI key resolves (no network)."""
    if settings.speech_mode != "realtime":
        return False
    return secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai)) is not None


class RealtimeBridge:
    """Owns one OpenAI Realtime WebSocket for a room. All methods are safe after `close`."""

    def __init__(self, settings: Settings, room_id: str, hub: Hub, *,
                 url: str | None = None, server_vad: bool = True,
                 respond: Respond | None = None) -> None:
        self.settings = settings
        self.room_id = room_id
        self.hub = hub
        self.server_vad = server_vad
        self._respond = respond  # runs the ReviewAgent turn for a transcribed user message
        self._url = f"{url or OPENAI_REALTIME_URL}?model={settings.realtime_model}"
        self._ws: websockets.ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()  # set once the session is configured (or the bridge is done)
        self._ptt_speaker = "guest"
        self._ptt_holder: str | None = None  # who currently holds the shared mic (first-wins)
        self.closed = False
        self.failed = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self.closed = True
        self._ready.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _run(self) -> None:
        for _ in range(2):  # initial connection + one reconnect on a dropped socket
            if self.closed or self.failed:
                return
            try:
                await self._open()
                await self._consume()
                return
            except (OSError, websockets.WebSocketException):
                continue
        if not self.closed and not self.failed:
            await self._fail("The voice connection dropped.")

    async def _open(self) -> None:
        self._ws = await websockets.connect(self._url, additional_headers=self._headers())
        await self._ws.send(json.dumps(self._session_update()))
        self._ready.set()

    def _headers(self) -> dict:
        service = self.settings.keychain_service(self.settings.keychain_openai)
        key = secret("OPENAI_API_KEY", service)
        return {"Authorization": f"Bearer {key}"} if key else {}

    # -- inbound from browsers ----------------------------------------------

    async def append_audio(self, b64: str) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def ptt(self, state: str, speaker: str) -> None:
        """One shared session, so push-to-talk is a floor: the first holder wins and a second
        holder is told to wait. A release only counts from whoever holds the floor."""
        speaker = speaker or "guest"
        if state == "down":
            if self._ptt_holder is not None and self._ptt_holder != speaker:
                self._post("Interviewer", "assistant",
                           f"{self._ptt_holder} has the mic — {speaker}, hold on, you're next.")
                return
            self._ptt_holder = speaker
            self._ptt_speaker = speaker
            return
        if speaker != self._ptt_holder:  # a release from someone who never held the floor
            return
        self._ptt_holder = None
        if not self.server_vad:
            await self._send({"type": "input_audio_buffer.commit"})
            await self._send({"type": "response.create"})

    async def _send(self, payload: dict) -> None:
        if not self._ready.is_set():
            await self._ready.wait()  # first audio can arrive before the socket is configured
        if self._ws is None or self.closed:
            return
        with contextlib.suppress(websockets.WebSocketException, OSError):
            await self._ws.send(json.dumps(payload))

    # -- outbound from OpenAI ------------------------------------------------

    async def _consume(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            await self._handle(event)
            if self.closed or self.failed:
                return

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "response.output_audio.delta":
            self.hub.publish(self.room_id, Event("audio", {"b64": event.get("delta", "")}))
        elif kind == "conversation.item.input_audio_transcription.completed":
            await self._on_user_text(self._ptt_speaker, event.get("transcript", ""))
        elif kind == "error":
            await self._fail("The voice service returned an error.")

    async def _on_user_text(self, speaker: str, text: str) -> None:
        """A finished user transcription is a user turn: run it through the ReviewAgent (posts the
        user + agent messages, runs the review tools, publishes state), then voice the reply."""
        if not text.strip() or self._respond is None:
            return
        reply = await self._respond(self.room_id, speaker, text.strip())
        if reply:
            await self._speak(reply)

    async def _speak(self, text: str) -> None:
        """Have the Realtime session read the ReviewAgent's reply aloud (audio only; the reply text
        is already posted to the room by `respond`)."""
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "input_text", "text": text}]}})
        await self._send({"type": "response.create", "response": {
            "instructions": "Read the review agent's reply above aloud, verbatim.",
            "output_modalities": ["audio"]}})

    # -- helpers -------------------------------------------------------------

    def _post(self, speaker: str, role: str, text: str) -> None:
        if not text.strip():
            return
        conn = store.connect(self.settings.db_path)
        try:
            msg = rooms.post(conn, self.room_id, speaker, role, text)
        finally:
            conn.close()
        self.hub.publish(self.room_id, Event("message", {
            "id": msg.id, "speaker": msg.speaker, "role": msg.role, "text": msg.text,
            "audio_path": None, "ts": msg.ts, "has_audio": False}))

    async def _fail(self, message: str) -> None:
        self.failed = True
        self._ready.set()
        self._post("Interviewer", "assistant",
                   f"{message} Switching this room to local voice mode.")
        self.hub.publish(self.room_id, Event("fallback", {"reason": message}))
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()

    def _instructions(self) -> str:
        from ..review.agent import ReviewAgent  # lazy: review depends on interview, not the reverse

        conn = store.connect(self.settings.db_path)
        try:
            room = store.get_room(conn, self.room_id)
            summary = ReviewAgent(None, conn, room, self.settings).open_statement()
        finally:
            conn.close()
        return _REALTIME_GUIDANCE + summary

    def _session_update(self) -> dict:
        pcm = {"type": "audio/pcm", "rate": 24000}
        # create_response off: server VAD segments + transcribes the user, but the model never
        # auto-answers — the ReviewAgent produces every reply and we voice it via `_speak`.
        vad = {"type": "server_vad", "create_response": False} if self.server_vad else None
        audio_in = {"format": pcm, "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": vad}
        return {"type": "session.update", "session": {
            "type": "realtime",
            "instructions": self._instructions(),
            "output_modalities": ["audio"],
            "audio": {"input": audio_in,
                      "output": {"format": pcm, "voice": self.settings.realtime_voice}},
            "tools": TOOLS,
            "tool_choice": "auto"}}


class Bridges:
    """Per-room `RealtimeBridge` registry: lazy create on first audio, idle-close when empty."""

    def __init__(self, settings: Settings, hub: Hub, *, idle_seconds: float = 60.0,
                 sleep=asyncio.sleep, respond: Respond | None = None) -> None:
        self.settings = settings
        self.hub = hub
        self.idle_seconds = idle_seconds
        self._sleep = sleep
        self._respond = respond  # the ReviewAgent turn, injected by the server
        self._bridges: dict[str, RealtimeBridge] = {}
        self._idle: dict[str, asyncio.Task] = {}
        self._failed: set[str] = set()

    async def get(self, room_id: str) -> RealtimeBridge | None:
        """The room's bridge, created and started on first use; None once the room fell back."""
        if room_id in self._failed:
            return None
        bridge = self._bridges.get(room_id)
        if bridge is not None and bridge.failed:
            await self.close(room_id)
            return None
        if bridge is None:
            bridge = RealtimeBridge(self.settings, room_id, self.hub, respond=self._respond)
            self._bridges[room_id] = bridge
            await bridge.start()
        self._cancel_idle(room_id)
        return bridge

    def client_here(self, room_id: str) -> None:
        """A client (re)joined; cancel any pending idle-close for its bridge."""
        self._cancel_idle(room_id)

    def client_gone(self, room_id: str) -> None:
        """A client left; close the bridge after `idle_seconds` if no client returns."""
        if room_id not in self._bridges or self.hub.subscriber_count(room_id) > 0:
            return
        self._cancel_idle(room_id)
        self._idle[room_id] = asyncio.create_task(self._close_after_idle(room_id))

    async def _close_after_idle(self, room_id: str) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await self._sleep(self.idle_seconds)
            if self.hub.subscriber_count(room_id) == 0:
                await self.close(room_id)

    def _cancel_idle(self, room_id: str) -> None:
        task = self._idle.pop(room_id, None)
        if task is not None:
            task.cancel()

    async def close(self, room_id: str) -> None:
        self._cancel_idle(room_id)
        bridge = self._bridges.pop(room_id, None)
        if bridge is not None:
            if bridge.failed:
                self._failed.add(room_id)
            await bridge.close()

    async def close_all(self) -> None:
        for room_id in list(self._bridges):
            await self.close(room_id)


__all__ = ["RealtimeBridge", "Bridges", "TOOLS", "OPENAI_REALTIME_URL", "realtime_available"]
