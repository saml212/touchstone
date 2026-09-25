import httpx
import pytest
from fastapi.testclient import TestClient

from touchstone.config import Settings
from touchstone.server import create_app


def _seed_task(db):
    """Room creation takes an opaque task_id string (v3 stores it, no task file lookup)."""
    return "t1"


def _app(db, agent_provider="scripted"):
    return create_app(Settings(db_path=db, agent_provider=agent_provider))


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_health(db):
    app = _app(db)
    async with _client(app) as c:
        assert (await c.get("/api/health")).json() == {"status": "ok"}


async def test_create_room_posts_an_opening_statement(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        r = await c.post("/api/rooms", json={"task_id": task_id, "topic": "refunds"})
        assert r.status_code == 200
        state = r.json()
        assert state["room"]["topic"] == "refunds"
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "assistant"


async def test_two_speakers_post_and_persist_in_order(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        await c.post(f"/api/rooms/{room_id}/messages", json={"speaker": "alice", "text": "hi"})
        await c.post(f"/api/rooms/{room_id}/messages", json={"speaker": "bob", "text": "hey"})
        msgs = (await c.get(f"/api/rooms/{room_id}")).json()["messages"]
        users = [m for m in msgs if m["role"] == "user"]
        assert [m["speaker"] for m in users] == ["alice", "bob"]
        assert [m["text"] for m in users] == ["hi", "hey"]


async def test_empty_message_rejected(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        r = await c.post(f"/api/rooms/{room_id}/messages", json={"speaker": "x", "text": "  "})
        assert r.status_code == 400


async def test_audio_rejects_bad_magic_bytes(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        files = {"file": ("x.webm", b"not audio at all", "audio/webm")}
        r = await c.post(f"/api/rooms/{room_id}/audio", data={"speaker": "sam"}, files=files)
        assert r.status_code == 400


async def test_audio_with_stt_none_returns_clear_error(db):
    task_id = _seed_task(db)
    app = _app(db)  # default stt = none
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        webm = b"\x1aE\xdf\xa3" + b"\x00" * 40
        files = {"file": ("clip.webm", webm, "audio/webm")}
        r = await c.post(f"/api/rooms/{room_id}/audio", data={"speaker": "sam"}, files=files)
        assert r.status_code == 400
        assert "disabled" in r.json()["detail"].lower()


async def test_audio_with_fake_stt_posts_transcript(db):
    task_id = _seed_task(db)
    app = _app(db)

    class FakeSTT:
        def transcribe(self, audio, mime):
            return "spoken words"

    app.state.speech.stt = FakeSTT()
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        webm = b"\x1aE\xdf\xa3" + b"\x00" * 40
        files = {"file": ("clip.webm", webm, "audio/webm")}
        r = await c.post(f"/api/rooms/{room_id}/audio", data={"speaker": "sam"}, files=files)
        assert r.status_code == 200
        assert r.json()["transcript"] == "spoken words"
        assert r.json()["user"]["text"] == "spoken words"


async def test_agent_audio_defaults_to_browser_tts(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        state = (await c.post("/api/rooms", json={"task_id": task_id})).json()
        room_id = state["room"]["id"]
        agent_msg_id = state["messages"][0]["id"]
        r = await c.get(f"/api/rooms/{room_id}/audio/{agent_msg_id}")
        assert r.status_code == 204


async def test_close_room(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        r = await c.post(f"/api/rooms/{room_id}/close")
        assert r.status_code == 200 and r.json()["room"]["closed_at"] is not None


def test_websocket_sends_state_then_events_in_order(db):
    task_id = _seed_task(db)
    app = _app(db)
    client = TestClient(app)
    room_id = client.post("/api/rooms", json={"task_id": task_id}).json()["room"]["id"]
    with client.websocket_connect(f"/ws/rooms/{room_id}") as ws:
        assert ws.receive_json()["type"] == "state"
        client.post(f"/api/rooms/{room_id}/messages", json={"speaker": "sam", "text": "hello"})
        types = [ws.receive_json()["type"] for _ in range(3)]
        assert types[0] == "message"  # the user's message
        assert types[1] == "message"  # the agent's reply
        assert types[2] == "draft"


def test_two_turns_each_push_full_state(db):
    # A dropped or reordered delta must never leave the page stale: every turn ends with a full
    # "state" event carrying the messages and the review block, so the room re-renders correctly.
    app = _app(db)
    client = TestClient(app)
    room_id = client.post("/api/rooms", json={"task_id": _seed_task(db)}).json()["room"]["id"]
    with client.websocket_connect(f"/ws/rooms/{room_id}") as ws:
        assert ws.receive_json()["type"] == "state"  # sent on connect
        states = []
        for turn in range(2):
            client.post(f"/api/rooms/{room_id}/messages", json={"speaker": "sam", "text": "hi"})
            state = None
            for _ in range(8):  # drain this turn's deltas until its closing state event
                ev = ws.receive_json()
                if ev["type"] == "state":
                    state = ev["data"]
                    break
            assert state is not None, f"turn {turn} produced no state event"
            assert "messages" in state and "review" in state
            states.append(len(state["messages"]))
    assert states[1] > states[0]  # the second turn's state reflects the new messages


def test_websocket_on_missing_room_closes(db):
    from starlette.websockets import WebSocketDisconnect

    app = _app(db)
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/rooms/does-not-exist") as ws:
            ws.receive_json()


async def test_post_to_closed_room_is_rejected(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        await c.post(f"/api/rooms/{room_id}/close")
        r = await c.post(f"/api/rooms/{room_id}/messages", json={"speaker": "s", "text": "hi"})
        assert r.status_code == 409
        # audio ingest path is guarded too
        ra = await c.post(f"/api/rooms/{room_id}/audio",
                          files={"file": ("c.webm", b"x", "audio/webm")})
        assert ra.status_code == 409


async def test_done_twice_second_is_rejected(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        done = {"speaker": "s", "text": "/done"}
        first = await c.post(f"/api/rooms/{room_id}/messages", json=done)
        assert first.status_code == 200
        second = await c.post(f"/api/rooms/{room_id}/messages", json=done)
        assert second.status_code == 409


async def test_room_state_reports_speech_mode(db):
    task_id = _seed_task(db)
    async with _client(_app(db)) as c:  # default local
        state = (await c.post("/api/rooms", json={"task_id": task_id})).json()
        assert state["mode"] == "local"
    rt = create_app(Settings(db_path=db, agent_provider="scripted", speech_mode="realtime"))
    async with _client(rt) as c:
        state = (await c.post("/api/rooms", json={"task_id": task_id})).json()
        assert state["mode"] == "realtime"


class _FakeBridge:
    def __init__(self):
        self.audio = []
        self.ptts = []

    async def append_audio(self, b64):
        self.audio.append(b64)

    async def ptt(self, state, speaker):
        self.ptts.append((state, speaker))


class _FakeBridges:
    def __init__(self):
        self.bridge = _FakeBridge()
        self.here = self.gone = 0

    def client_here(self, room_id):
        self.here += 1

    def client_gone(self, room_id):
        self.gone += 1

    async def get(self, room_id):
        return self.bridge

    async def close(self, room_id):
        pass

    async def close_all(self):
        pass


def test_realtime_ws_routes_audio_and_ptt_to_the_bridge(db):
    task_id = _seed_task(db)
    app = create_app(Settings(db_path=db, agent_provider="scripted", speech_mode="realtime"))
    fake = _FakeBridges()
    app.state.bridges = fake
    client = TestClient(app)
    room_id = client.post("/api/rooms", json={"task_id": task_id}).json()["room"]["id"]
    with client.websocket_connect(f"/ws/rooms/{room_id}") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "audio", "b64": "AAAA", "speaker": "sam"})
        ws.send_json({"type": "ptt", "state": "down", "speaker": "sam"})
        ws.send_json({"type": "message-noise"})  # ignored, not audio/ptt
    assert fake.bridge.audio == ["AAAA"]
    assert ("down", "sam") in fake.bridge.ptts
    assert fake.here == 1 and fake.gone == 1


def test_local_ws_ignores_audio_frames(db):
    task_id = _seed_task(db)
    app = _app(db)  # local mode
    fake = _FakeBridges()
    app.state.bridges = fake
    client = TestClient(app)
    room_id = client.post("/api/rooms", json={"task_id": task_id}).json()["room"]["id"]
    with client.websocket_connect(f"/ws/rooms/{room_id}") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "audio", "b64": "AAAA", "speaker": "sam"})
    assert fake.bridge.audio == []  # local mode never touches the realtime bridge
