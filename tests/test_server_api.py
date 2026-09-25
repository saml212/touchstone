"""JSON API + static shell over a seeded trace database."""

from fastapi.testclient import TestClient

from touchstone import store
from touchstone.config import Settings
from touchstone.server import create_app


def _seed(db):
    conn = store.connect(db)
    try:
        ep1 = store.insert_episode(conn, store.Episode(name="resolve refund", outcome_label="ok"))
        store.insert_span(conn, store.Span(
            episode_id=ep1.id, kind="model", name="chat",
            input={"messages": [{"role": "system", "content": "you are support"},
                                {"role": "user", "content": "I want a refund"}], "tools": []},
            output={"message": {"role": "assistant", "content": "our refund policy allows it",
                                "tool_calls": []}},
        ))
        ep2 = store.insert_episode(conn, store.Episode(name="angry user", outcome_label="fail"))
        store.insert_span(conn, store.Span(episode_id=ep2.id, kind="model", name="chat",
            input={"messages": [{"role": "user", "content": "help"}]},
            output={"message": {"role": "assistant", "content": "no"}}))
    finally:
        conn.close()
    return {"ep1": ep1.id, "ep2": ep2.id}


def _client(db):
    return TestClient(create_app(
        Settings(db_path=db, provider="scripted", agent_provider="scripted")))


def test_overview_counts_match_the_traces(db):
    _seed(db)
    c = _client(db)
    o = c.get("/api/overview").json()
    assert o["episodes"]["total"] == 2
    assert o["episodes"]["outcomes"] == {"ok": 1, "fail": 1}
    assert o["spans"] == 2
    assert o["rooms"] == {"total": 0, "open": 0}
    assert "bench" in o["next_step"]


def test_episode_list_detail_and_pagination(db):
    ids = _seed(db)
    c = _client(db)
    listed = c.get("/api/episodes").json()
    assert listed["total"] == 2 and len(listed["episodes"]) == 2
    assert c.get("/api/episodes?label=fail").json()["total"] == 1
    assert c.get("/api/episodes?offset=999").json()["episodes"] == []
    assert c.get("/api/episodes?offset=-1").status_code == 422
    assert len(c.get("/api/episodes?limit=1").json()["episodes"]) == 1

    detail = c.get(f"/api/episodes/{ids['ep1']}").json()
    assert detail["episode"]["name"] == "resolve refund"
    assert len(detail["spans"]) == 1
    assert detail["spans"][0]["input"]["messages"][0]["role"] == "system"
    assert c.get("/api/episodes/nope").status_code == 404


def test_rooms_list_open_first(db):
    _seed(db)
    c = _client(db)
    r_open = c.post("/api/rooms", json={"task_id": "t1", "topic": "a"}).json()["room"]["id"]
    r_closed = c.post("/api/rooms", json={"task_id": "t2", "topic": "b"}).json()["room"]["id"]
    c.post(f"/api/rooms/{r_closed}/close")
    rooms = c.get("/api/rooms").json()["rooms"]
    assert rooms[0]["id"] == r_open and rooms[0]["closed_at"] is None
    assert rooms[0]["task_name"] == "t1"


def test_static_shell_and_app_js(db):
    c = _client(db)
    root = c.get("/")
    assert root.status_code == 200 and "Touchstone" in root.text and 'id="nav"' in root.text
    js = c.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_no_api_route_500s_on_an_empty_project(db):
    # `touchstone init` then `serve` with zero episodes: every read route must answer
    # (200 with empty data, or a clean 404/422), never a 500.
    store.connect(db).close()  # schema only, no episodes
    c = TestClient(create_app(Settings(db_path=db, provider="scripted",
                                       agent_provider="scripted")),
                   raise_server_exceptions=False)
    gets = ["/api/overview", "/api/episodes", "/api/episodes/nope",
            "/api/rooms", "/api/rooms/nope"]
    for path in gets:
        assert c.get(path).status_code < 500, path
    assert c.post("/api/rooms", json={"topic": "x"}).status_code < 500
