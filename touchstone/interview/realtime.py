"""Realtime voice bridge: one OpenAI Realtime speech-to-speech session per interview room.

The server holds the key and opens a single WebSocket to OpenAI per open room. Browsers stream
push-to-talk PCM16 over the room WebSocket; the bridge appends it to the session, dispatches the
model's tool calls (`draft_check`/`commit_check`/`show_task`/`next_task`) through the same
`Interviewer` the text mode uses, and fans the agent's audio + transcripts back out over the room
`Hub`. The transcript and the committed checks land in the store, so text stays the source of truth.

Any OpenAI error posts one room message and falls back to local mode for that room — never a 500.
A dropped socket reconnects once; a second drop falls back too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import websockets

from .. import store
from ..config import Settings
from ..llm.keychain import secret
from . import rooms
from .agent import Interviewer
from .rooms import Event, Hub

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

_REALTIME_GUIDANCE = (
    "You are Touchstone's voice interviewer. Turn stakeholders' opinions about an agent's "
    "behaviour into concrete checks. Ask one concrete question at a time, address people by name. "
    "When "
    "someone states a rule, call draft_check (prefer a programmatic kind; use judge only when "
    "nothing else fits), then read the check back in plain words using the read_back the tool "
    "returns, and only call commit_check after someone says yes. Use show_task to summarize the "
    "task and next_task to move on. Here is the task under review:\n"
)

TOOLS = [
    {"type": "function", "name": "draft_check",
     "description": "Draft a check (not committed) from a stakeholder rule and show it in the "
                    "room. Prefer a programmatic kind; use judge only if nothing else fits. "
                    "Always draft, then read back, before committing.",
     "parameters": {"type": "object", "required": ["kind", "params"], "properties": {
         "kind": {"type": "string"},
         "params": {"type": "object"},
         "name": {"type": "string"},
         "rule": {"type": "string"},
         "severity": {"type": "string", "enum": ["hard", "soft"]},
         "applies_to": {"type": "string", "enum": ["final", "any_turn", "tool_calls"]},
         "rationale": {"type": "string"}}}},
    {"type": "function", "name": "commit_check",
     "description": "Commit a check the room has agreed to, AFTER reading it back and hearing yes. "
                    "Pass the same fields you drafted.",
     "parameters": {"type": "object", "required": ["kind", "params"], "properties": {
         "kind": {"type": "string"},
         "params": {"type": "object"},
         "name": {"type": "string"},
         "rule": {"type": "string"},
         "severity": {"type": "string", "enum": ["hard", "soft"]},
         "applies_to": {"type": "string", "enum": ["final", "any_turn", "tool_calls"]}}}},
    {"type": "function", "name": "show_task",
     "description": "Fetch the current task summary and its checks to narrate aloud.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "next_task",
     "description": "Open a room on the next task in the same work queue and return its URL.",
     "parameters": {"type": "object", "properties": {}}},
]


def realtime_available(settings: Settings) -> bool:
    """True when realtime mode is on and an OpenAI key resolves (no network)."""
    if settings.speech_mode != "realtime":
        return False
    return secret("OPENAI_API_KEY", settings.keychain_service(settings.keychain_openai)) is not None


class RealtimeBridge:
    """Owns one OpenAI Realtime WebSocket for a room. All methods are safe after `close`."""

    def __init__(self, settings: Settings, room_id: str, hub: Hub, *,
                 url: str | None = None, server_vad: bool = True) -> None:
        self.settings = settings
        self.room_id = room_id
        self.hub = hub
        self.server_vad = server_vad
        self._url = f"{url or OPENAI_REALTIME_URL}?model={settings.realtime_model}"
        self._ws: websockets.ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._ptt_speaker = "guest"
        self.closed = False
        self.failed = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self.closed = True
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

    def _headers(self) -> dict:
        service = self.settings.keychain_service(self.settings.keychain_openai)
        key = secret("OPENAI_API_KEY", service)
        headers = {"OpenAI-Beta": "realtime=v1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # -- inbound from browsers ----------------------------------------------

    async def append_audio(self, b64: str) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def ptt(self, state: str, speaker: str) -> None:
        self._ptt_speaker = speaker or self._ptt_speaker
        if state == "up" and not self.server_vad:
            await self._send({"type": "input_audio_buffer.commit"})
            await self._send({"type": "response.create"})

    async def _send(self, payload: dict) -> None:
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
        if kind == "response.audio.delta":
            self.hub.publish(self.room_id, Event("audio", {"b64": event.get("delta", "")}))
        elif kind == "response.audio_transcript.done":
            self._post("Interviewer", "assistant", event.get("transcript", ""))
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._post(self._ptt_speaker, "user", event.get("transcript", ""))
        elif kind == "response.function_call_arguments.done":
            await self._dispatch_tool(event)
        elif kind == "error":
            await self._fail("The voice service returned an error.")

    async def _dispatch_tool(self, event: dict) -> None:
        try:
            args = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        result = self._run_action(event.get("name", ""), args if isinstance(args, dict) else {})
        await self._send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": event.get("call_id", ""),
            "output": json.dumps(result, ensure_ascii=False)}})
        await self._send({"type": "response.create"})
        self._broadcast_action(event.get("name", ""), result)

    def _run_action(self, name: str, args: dict) -> dict:
        conn = store.connect(self.settings.db_path)
        try:
            room = store.get_room(conn, self.room_id)
            if room is None:
                return {"error": "room is gone"}
            agent = Interviewer(None, conn, room, self.settings.root)
            action = {"draft_check": lambda: agent.draft_check(args),
                      "commit_check": lambda: agent.commit_check(args),
                      "show_task": agent.show_task,
                      "next_task": agent.next_task}.get(name)
            return action() if action else {"error": f"unknown tool {name}"}
        finally:
            conn.close()

    def _broadcast_action(self, name: str, result: dict) -> None:
        if name == "draft_check" and result.get("check"):
            self.hub.publish(self.room_id, Event("draft", {"checks": [result["check"]]}))
        elif name == "commit_check" and result:
            self.hub.publish(self.room_id, Event("committed", {"checks": [result]}))

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
        self._post("Interviewer", "assistant",
                   f"{message} Switching this room to local voice mode.")
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()

    def _instructions(self) -> str:
        conn = store.connect(self.settings.db_path)
        try:
            room = store.get_room(conn, self.room_id)
            summary = Interviewer(None, conn, room, self.settings.root).open_statement()
        finally:
            conn.close()
        return _REALTIME_GUIDANCE + summary

    def _session_update(self) -> dict:
        vad = {"type": "server_vad", "interrupt_response": True} if self.server_vad else None
        return {"type": "session.update", "session": {
            "instructions": self._instructions(),
            "voice": self.settings.realtime_voice,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            "turn_detection": vad,
            "tools": TOOLS,
            "tool_choice": "auto"}}


class Bridges:
    """Per-room `RealtimeBridge` registry: lazy create on first audio, idle-close when empty."""

    def __init__(self, settings: Settings, hub: Hub, *, idle_seconds: float = 60.0,
                 sleep=asyncio.sleep) -> None:
        self.settings = settings
        self.hub = hub
        self.idle_seconds = idle_seconds
        self._sleep = sleep
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
            bridge = RealtimeBridge(self.settings, room_id, self.hub)
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
