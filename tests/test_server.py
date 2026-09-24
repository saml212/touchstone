import json

import httpx
import pytest
from fastapi.testclient import TestClient

from touchstone import store
from touchstone.config import Settings
from touchstone.llm import Rule, ScriptedProvider
from touchstone.server import create_app

DRAFT_JSON = json.dumps(
    {
        "say": "Should it always mention the refund policy?",
        "draft": [{"kind": "contains", "params": {"values": ["refund"], "mode": "any"},
                   "name": "mentions refund", "severity": "hard"}],
        "commit": [],
    }
)


def _seed_task(db):
    conn = store.connect(db)
    try:
        ep = store.insert_episode(conn, store.Episode(name="ep1", outcome_label="ok"))
        task = store.insert_task(
            conn,
            store.Task(name="t1", episode_id=ep.id,
                       context={"messages": [{"role": "user", "content": "help"}], "tools": []},
                       reference={"content": "sure", "tool_calls": []}),
        )
        return task.id
    finally:
        conn.close()


def _app(db, agent_provider="scripted"):
    return create_app(Settings(db_path=db, agent_provider=agent_provider))


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_health_and_task_lookup(db):
    task_id = _seed_task(db)
    app = _app(db)
    async with _client(app) as c:
        assert (await c.get("/api/health")).json() == {"status": "ok"}
        assert (await c.get("/api/tasks/nope")).status_code == 404
        got = await c.get(f"/api/tasks/{task_id}")
        assert got.status_code == 200 and got.json()["name"] == "t1"


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


async def test_draft_then_confirm_commits_over_http(db):
    task_id = _seed_task(db)
    app = _app(db)
    app.state.provider_factory = lambda: ScriptedProvider(
        rules=[Rule(substring="refund policy", content=DRAFT_JSON)]
    )
    async with _client(app) as c:
        room_id = (await c.post("/api/rooms", json={"task_id": task_id})).json()["room"]["id"]
        r1 = await c.post(f"/api/rooms/{room_id}/messages",
                          json={"speaker": "sam", "text": "must state the refund policy"})
        assert len(r1.json()["draft"]) == 1
        r2 = await c.post(f"/api/rooms/{room_id}/messages",
                          json={"speaker": "sam", "text": "yes"})
        assert len(r2.json()["turn"]["commit"]) == 1
        committed = (await c.get(f"/api/rooms/{room_id}")).json()["committed"]
        assert committed and committed[0]["kind"] == "contains"


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


def test_websocket_on_missing_room_closes(db):
    from starlette.websockets import WebSocketDisconnect

    app = _app(db)
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/rooms/does-not-exist") as ws:
            ws.receive_json()
