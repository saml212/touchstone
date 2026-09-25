"""RealtimeBridge against a scripted fake OpenAI Realtime server on localhost — no network.

The fake accepts the session.update, then pushes the events a real session would (function calls,
audio deltas, transcripts, or an error), and records what the bridge sends back. The bridge's run
loop ends when the fake closes the socket, so each test awaits `bridge._task` to know it is done.
"""

from __future__ import annotations

import asyncio
import json

import websockets

from touchstone import store
from touchstone.config import Settings
from touchstone.interview import rooms
from touchstone.interview.realtime import Bridges, RealtimeBridge
from touchstone.interview.rooms import Hub


class FakeRealtime:
    """A scripted OpenAI Realtime endpoint. `script` is pushed after session.update; sent client
    frames land in `received`. `drop_first` makes the first connection abort (to test reconnect)."""

    def __init__(self, script, *, drop_first=False):
        self.script = script
        self.drop_first = drop_first
        self.received: list[dict] = []
        self.connections = 0
        self._server = None
        self.url = ""

    async def __aenter__(self):
        self._server = await websockets.serve(self._handle, "localhost", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://localhost:{port}/v1/realtime"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, ws):
        self.connections += 1
        self.received.append(json.loads(await ws.recv()))  # session.update
        await ws.send(json.dumps({"type": "session.updated"}))
        if self.drop_first and self.connections == 1:
            raise RuntimeError("simulated drop")  # closes 1011 -> client reconnects
        for event in self.script:
            await ws.send(json.dumps(event))
        await self._drain(ws)

    async def _drain(self, ws):
        try:
            while True:
                self.received.append(json.loads(await asyncio.wait_for(ws.recv(), 0.5)))
        except (TimeoutError, websockets.WebSocketException):
            pass


def _project(tmp_path):
    db = str(tmp_path / ".touchstone" / "touchstone.db")
    settings = Settings(db_path=db, speech_mode="realtime")
    conn = store.connect(db)
    room = rooms.open(conn, task_id=None, topic="orders")
    conn.close()
    return settings, room.id


async def _run_bridge(settings, room_id, hub, fake, **kw):
    bridge = RealtimeBridge(settings, room_id, hub, url=fake.url, **kw)
    await bridge.start()
    await asyncio.wait_for(bridge._task, timeout=5)
    return bridge


def _kinds(fake):
    return [r.get("type") for r in fake.received]


async def test_session_update_carries_pcm16_and_the_topic(tmp_path):
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([]) as fake:
        await _run_bridge(settings, room_id, Hub(), fake)
    session = fake.received[0]["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["tools"] == []  # no realtime tool surface in v3 stage 1
    assert "orders" in session["instructions"]  # the room topic is in the prompt


async def test_user_transcription_is_routed_to_the_review_agent_and_the_reply_is_voiced(tmp_path):
    # Attack (stage-7 review): the realtime voice path must drive the SAME ReviewAgent as text, not
    # the old interview agent. A completed user transcription is fed to the injected `respond`
    # (the ReviewAgent turn) attributed to the ptt speaker; its reply is then spoken by the session.
    settings, room_id = _project(tmp_path)
    seen = []

    async def respond(rid, speaker, text):
        seen.append((rid, speaker, text))
        return "The verifier scored that trial 1.0."

    script = [{"type": "conversation.item.input_audio_transcription.completed",
               "transcript": "did that trial pass?"}]
    async with FakeRealtime(script) as fake:
        bridge = RealtimeBridge(settings, room_id, Hub(), url=fake.url, respond=respond)
        await bridge.start()
        await bridge.ptt("down", "sam")  # attribute the user turn to sam
        await asyncio.wait_for(bridge._task, timeout=5)
    assert seen == [(room_id, "sam", "did that trial pass?")]  # routed to the ReviewAgent turn
    # the reply is sent to the session to be voiced (create item + response.create), verbatim
    item = next(r for r in fake.received if r.get("type") == "conversation.item.create")
    assert item["item"]["content"][0]["text"] == "The verifier scored that trial 1.0."
    assert any(r.get("type") == "response.create" for r in fake.received)


async def test_session_disables_model_autoresponse_so_reviewagent_is_the_only_voice(tmp_path):
    # create_response off: the Realtime model transcribes but never answers on its own — every
    # reply comes from the ReviewAgent (via respond), so voice cannot contradict the text room.
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([]) as fake:
        await _run_bridge(settings, room_id, Hub(), fake)
    td = fake.received[0]["session"]["audio"]["input"]["turn_detection"]
    assert td == {"type": "server_vad", "create_response": False}


async def test_audio_deltas_broadcast_to_two_clients(tmp_path):
    settings, room_id = _project(tmp_path)
    hub = Hub()
    a, b = hub.subscribe(room_id), hub.subscribe(room_id)
    async with FakeRealtime([{"type": "response.output_audio.delta", "delta": "AAAA"}]) as fake:
        await _run_bridge(settings, room_id, hub, fake)
    ea, eb = a.get_nowait(), b.get_nowait()
    assert ea.type == "audio" and ea.data["b64"] == "AAAA"
    assert eb.type == "audio" and eb.data["b64"] == "AAAA"


async def test_openai_error_falls_back_to_local_with_a_room_message(tmp_path):
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([{"type": "error", "error": {"message": "boom"}}]) as fake:
        bridge = await _run_bridge(settings, room_id, Hub(), fake)
    assert bridge.failed is True
    conn = store.connect(settings.db_path)
    texts = [m.text for m in store.list_room_messages(conn, room_id)]
    conn.close()
    assert any("local voice mode" in t for t in texts)


async def test_fallback_publishes_an_event_so_the_room_switches_to_local(tmp_path):
    # The browser needs to know the session fell back, so the bridge broadcasts one
    # `fallback` event (the room UI flips to local voice on it) alongside the message.
    settings, room_id = _project(tmp_path)
    hub = Hub()
    sub = hub.subscribe(room_id)
    async with FakeRealtime([{"type": "error", "error": {"message": "boom"}}]) as fake:
        await _run_bridge(settings, room_id, hub, fake)
    events = []
    while not sub.empty():
        events.append(sub.get_nowait())
    assert any(e.type == "fallback" for e in events)


async def test_reconnects_once_after_a_dropped_socket(tmp_path):
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([], drop_first=True) as fake:
        await _run_bridge(settings, room_id, Hub(), fake)
    assert fake.connections == 2  # reconnected once


async def test_ptt_release_commits_the_buffer_when_vad_is_off(tmp_path):
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([]) as fake:
        bridge = RealtimeBridge(settings, room_id, Hub(), url=fake.url, server_vad=False)
        await bridge.start()
        await bridge.append_audio("AAAA")
        await bridge.ptt("down", "sam")
        await bridge.ptt("up", "sam")
        await asyncio.sleep(0.1)
        await bridge.close()
    kinds = _kinds(fake)
    assert "input_audio_buffer.append" in kinds
    assert "input_audio_buffer.commit" in kinds
    assert "response.create" in kinds


async def test_ptt_release_does_not_commit_under_server_vad(tmp_path):
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([]) as fake:
        bridge = RealtimeBridge(settings, room_id, Hub(), url=fake.url, server_vad=True)
        await bridge.start()
        await bridge.ptt("up", "sam")
        await asyncio.sleep(0.1)
        await bridge.close()
    assert "input_audio_buffer.commit" not in _kinds(fake)


# -- Bridges registry: lazy create + idle close -----------------------------


class _Clock:
    """A patched sleep that only returns once `release` is called."""

    def __init__(self):
        self.slept = None
        self._gate = asyncio.Event()

    async def sleep(self, seconds):
        self.slept = seconds
        await self._gate.wait()

    def release(self):
        self._gate.set()


async def test_bridges_idle_close_waits_then_closes(tmp_path, monkeypatch):
    settings, room_id = _project(tmp_path)
    hub = Hub()
    clock = _Clock()
    bridges = Bridges(settings, hub, idle_seconds=60.0, sleep=clock.sleep)

    # Point the bridge at a fake so start() connects to something local.
    async with FakeRealtime([]) as fake:
        monkeypatch.setattr("touchstone.interview.realtime.OPENAI_REALTIME_URL", fake.url)
        bridge = await bridges.get(room_id)
        assert bridge is not None and room_id in bridges._bridges

        hub.subscribe(room_id)  # a client is present...
        bridges.client_gone(room_id)  # ...count > 0, so no idle timer starts
        assert room_id not in bridges._idle

        hub._subs[room_id].clear()  # now no clients
        bridges.client_gone(room_id)
        await asyncio.sleep(0)  # let the idle task reach the (patched) sleep
        assert clock.slept == 60.0  # the idle timer is waiting on the patched clock
        clock.release()
        await asyncio.sleep(0.05)
        assert room_id not in bridges._bridges  # closed after idle
    await bridges.close_all()


async def test_bridges_client_here_cancels_a_pending_idle_close(tmp_path, monkeypatch):
    settings, room_id = _project(tmp_path)
    hub = Hub()
    clock = _Clock()
    bridges = Bridges(settings, hub, idle_seconds=60.0, sleep=clock.sleep)
    async with FakeRealtime([]) as fake:
        monkeypatch.setattr("touchstone.interview.realtime.OPENAI_REALTIME_URL", fake.url)
        await bridges.get(room_id)
        bridges.client_gone(room_id)
        assert room_id in bridges._idle
        bridges.client_here(room_id)  # a client returned
        assert room_id not in bridges._idle
        assert room_id in bridges._bridges  # bridge survives
    await bridges.close_all()


def test_realtime_available_needs_mode_and_key(monkeypatch):
    from touchstone.interview import realtime
    monkeypatch.setattr(realtime, "secret", lambda *a: "sk-test")
    assert realtime.realtime_available(Settings(speech_mode="realtime")) is True
    assert realtime.realtime_available(Settings(speech_mode="local")) is False
    monkeypatch.setattr(realtime, "secret", lambda *a: None)
    assert realtime.realtime_available(Settings(speech_mode="realtime")) is False


async def test_second_ptt_holder_is_told_to_wait_first_holder_keeps_floor(tmp_path):
    # Two stakeholders share one session. If a second presses push-to-talk while the
    # first is holding, the first keeps the floor and the second gets a room message —
    # otherwise their mic audio mixes into one buffer under a single attribution.
    settings, room_id = _project(tmp_path)
    async with FakeRealtime([]) as fake:
        bridge = RealtimeBridge(settings, room_id, Hub(), url=fake.url, server_vad=False)
        await bridge.start()
        await bridge.ptt("down", "sam")   # sam takes the floor
        await bridge.ptt("down", "alex")  # collision while sam holds
        # alex releasing must NOT commit the buffer — alex never held the floor
        await bridge.ptt("up", "alex")
        await asyncio.sleep(0.1)
        await bridge.close()
    conn = store.connect(settings.db_path)
    texts = [m.text for m in store.list_room_messages(conn, room_id)]
    conn.close()
    assert any("sam" in t and "alex" in t for t in texts)
    assert "input_audio_buffer.commit" not in _kinds(fake)
