"""JSON API + static shell: routes exercised on a seeded db, plus API/CLI scoreboard parity."""

import time

from fastapi.testclient import TestClient

from touchstone import store
from touchstone.bench import scoreboard
from touchstone.config import Settings
from touchstone.server import create_app


def _seed(db):
    conn = store.connect(db)
    try:
        ep1 = store.insert_episode(conn, store.Episode(name="resolve refund", outcome_label="ok"))
        store.insert_span(conn, store.Span(
            episode_id=ep1.id, kind="llm", name="chat",
            input={"messages": [{"role": "system", "content": "you are support"},
                                {"role": "user", "content": "I want a refund"}], "tools": []},
            output={"message": {"role": "assistant", "content": "our refund policy allows it",
                                "tool_calls": []}},
        ))
        ep2 = store.insert_episode(conn, store.Episode(name="angry user", outcome_label="fail"))
        store.insert_span(conn, store.Span(episode_id=ep2.id, kind="llm", name="chat",
            input={"messages": [{"role": "user", "content": "help"}]},
            output={"message": {"role": "assistant", "content": "no"}}))
        check = store.insert_check(conn, store.Check(
            name="mentions refund", kind="contains",
            params={"values": ["refund"], "mode": "any"}, source="manual", enabled=1))
        t1 = store.insert_task(conn, store.Task(
            name="t1", episode_id=ep1.id, tags=["support"],
            context={"messages": [{"role": "user", "content": "I want a refund"}], "tools": []},
            reference={"content": "refund ok", "tool_calls": []}, check_ids=[check.id]))
        t2 = store.insert_task(conn, store.Task(
            name="t2", episode_id=ep2.id, tags=["failure"],
            context={"messages": [{"role": "user", "content": "more refund"}], "tools": []},
            reference={"content": "refund ok", "tool_calls": []}, check_ids=[check.id]))
        bench = store.insert_benchmark(conn, store.Benchmark(name="demo", task_ids=[t1.id, t2.id]))
        return {"ep1": ep1.id, "ep2": ep2.id, "check": check.id,
                "t1": t1.id, "t2": t2.id, "bench": bench.id}
    finally:
        conn.close()


def _client(db):
    settings = Settings(db_path=db, provider="scripted", agent_provider="scripted")
    return TestClient(create_app(settings))


def test_overview_counts_match_store(db):
    ids = _seed(db)
    c = _client(db)
    o = c.get("/api/overview").json()
    assert o["episodes"]["total"] == 2
    assert o["episodes"]["outcomes"] == {"ok": 1, "fail": 1}
    assert o["spans"] == 2
    assert o["checks"] == {"total": 1, "enabled": 1, "by_source": {"manual": 1}}
    assert o["tasks"] == 2 and o["benchmarks"] == 1 and o["runs"] == 0
    assert o["rooms"] == {"total": 0, "open": 0}
    assert ids  # seeded


def test_episode_list_detail_and_pagination(db):
    ids = _seed(db)
    c = _client(db)
    listed = c.get("/api/episodes").json()
    assert listed["total"] == 2 and len(listed["episodes"]) == 2

    filtered = c.get("/api/episodes?label=fail").json()
    assert filtered["total"] == 1

    assert c.get("/api/episodes?offset=999").json()["episodes"] == []
    assert c.get("/api/episodes?offset=-1").status_code == 422
    assert c.get("/api/episodes?limit=1").json()["episodes"].__len__() == 1

    detail = c.get(f"/api/episodes/{ids['ep1']}").json()
    assert detail["episode"]["name"] == "resolve refund"
    assert len(detail["spans"]) == 1
    assert detail["spans"][0]["input"]["messages"][0]["role"] == "system"
    assert c.get("/api/episodes/nope").status_code == 404


def test_check_create_invalid_params_returns_422_with_message(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/checks", json={"kind": "contains", "params": {}})
    assert r.status_code == 422
    assert "values" in r.json()["detail"]


def test_check_create_patch_and_kinds(db):
    _seed(db)
    c = _client(db)
    assert "contains" in c.get("/api/checks/kinds").json()["kinds"]
    made = c.post("/api/checks", json={
        "kind": "max_length", "params": {"max": 10}, "name": "brief",
        "rationale": "keep it short"}).json()
    assert made["source"] == "manual" and made["enabled"] is True

    off = c.patch(f"/api/checks/{made['id']}", json={"enabled": False}).json()
    assert off["enabled"] is False
    bad = c.patch(f"/api/checks/{made['id']}", json={"params": {"max": "nope"}})
    assert bad.status_code == 422
    assert c.patch("/api/checks/nope", json={"enabled": True}).status_code == 404


def test_check_eval(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post(f"/api/checks/{ids['check']}/eval", json={"text": "we offer a refund"})
    assert r.json()["passed"] is True
    r2 = c.post(f"/api/checks/{ids['check']}/eval", json={"text": "no"})
    assert r2.json()["passed"] is False
    assert c.post("/api/checks/nope/eval", json={"text": "x"}).status_code == 404


def test_task_attach_detach_idempotent(db):
    ids = _seed(db)
    c = _client(db)
    other = c.post("/api/checks", json={"kind": "min_length", "params": {"min": 1}}).json()
    first = c.post(f"/api/tasks/{ids['t1']}/checks", json={"check_id": other["id"]}).json()
    n = len(first["check_ids"])
    again = c.post(f"/api/tasks/{ids['t1']}/checks", json={"check_id": other["id"]}).json()
    assert len(again["check_ids"]) == n  # attach is idempotent

    c.request("DELETE", f"/api/tasks/{ids['t1']}/checks/{other['id']}")
    after = c.request("DELETE", f"/api/tasks/{ids['t1']}/checks/{other['id']}").json()
    assert other["id"] not in after["check_ids"]  # detach is idempotent
    assert c.post("/api/tasks/nope/checks", json={"check_id": other["id"]}).status_code == 404
    assert c.post(f"/api/tasks/{ids['t1']}/checks", json={"check_id": "nope"}).status_code == 404


def test_task_list_and_detail(db):
    ids = _seed(db)
    c = _client(db)
    listed = c.get("/api/tasks").json()
    assert listed["total"] == 2
    assert c.get("/api/tasks?tag=failure").json()["total"] == 1
    detail = c.get(f"/api/tasks/{ids['t1']}").json()
    assert detail["checks"][0]["kind"] == "contains"
    assert c.get("/api/tasks/nope").status_code == 404


def test_mine_no_llm_and_scripted_provider(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/mine", json={"no_llm": True}).json()
    assert r["tasks_cut"] >= 1 and r["llm"] == 0 and "proposals" in r
    r2 = c.post("/api/mine", json={"provider": "scripted", "no_llm": False})
    assert r2.status_code == 200  # scripted proposals are dropped, never crash


def test_benchmark_create_and_run_report_proof(db):
    _seed(db)
    c = _client(db)
    made = c.post("/api/benchmarks", json={"name": "all", "all": True}).json()
    assert len(made["task_ids"]) == 2
    assert c.post("/api/benchmarks", json={"name": "empty", "tags": ["nomatch"]}).status_code == 422

    rid = _run_and_wait(c, made["id"])
    rid2 = _run_and_wait(c, made["id"])

    report = c.get(f"/api/report?run={rid}&run={rid2}").json()
    assert {m["model_spec"] for m in report["models"]} == {"scripted"}
    assert len(report["models"]) == 2 and report["checks"] and report["tags"]

    proof = c.get(f"/api/proof?candidate={rid}&incumbent={rid2}").json()
    assert proof["counts"]["both_pass"] + proof["counts"]["both_fail"] == 2  # deterministic rerun
    assert c.get(f"/api/proof?candidate={rid}&incumbent=nope").status_code == 404
    assert c.get("/api/runs/nope").status_code == 404


def test_api_and_cli_scoreboard_identical(db):
    _seed(db)
    c = _client(db)
    bench = c.post("/api/benchmarks", json={"name": "all", "all": True}).json()
    rid = _run_and_wait(c, bench["id"])

    api_board = c.get(f"/api/report?run={rid}").json()
    conn = store.connect(db)
    try:
        cli_board = scoreboard(conn, [rid])
    finally:
        conn.close()
    assert api_board == cli_board


def test_rooms_list_open_first(db):
    ids = _seed(db)
    c = _client(db)
    r_open = c.post("/api/rooms", json={"task_id": ids["t1"], "topic": "a"}).json()["room"]["id"]
    r_closed = c.post("/api/rooms", json={"task_id": ids["t2"], "topic": "b"}).json()["room"]["id"]
    c.post(f"/api/rooms/{r_closed}/close")
    rooms = c.get("/api/rooms").json()["rooms"]
    assert rooms[0]["id"] == r_open and rooms[0]["closed_at"] is None
    assert rooms[0]["task_name"] == "t1"


def test_export_harbor(db, tmp_path):
    _seed(db)
    c = _client(db)
    bench = c.post("/api/benchmarks", json={"name": "all", "all": True}).json()
    out = str(tmp_path / "harbor")
    r = c.post("/api/export/harbor", json={"benchmark": bench["id"], "out": out}).json()
    assert r["count"] == 2 and len(r["dirs"]) == 2
    assert c.post("/api/export/harbor", json={"benchmark": "nope", "out": out}).status_code == 422


def test_static_shell_and_app_js(db):
    c = _client(db)
    root = c.get("/")
    assert root.status_code == 200 and "Touchstone" in root.text and 'id="nav"' in root.text
    js = c.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def _run_and_wait(c, benchmark_id, spec="scripted"):
    run = c.post("/api/runs", json={"benchmark": benchmark_id, "model_spec": spec}).json()
    for _ in range(100):
        cur = c.get(f"/api/runs/{run['id']}").json()
        if cur["finished_at"]:
            return run["id"]
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_train_prepare_and_download(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/train/prepare", json={"benchmark": ids["bench"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["counts"]["rl_tasks"] == 2
    assert set(body["files"]) >= {"sft", "preference", "rl_tasks", "manifest"}

    dl = c.get(f"/api/train/download?benchmark={ids['bench']}&file=rl_tasks.jsonl")
    assert dl.status_code == 200 and dl.text.strip()
    # a filename not on the allowlist is refused (no path traversal)
    assert c.get(f"/api/train/download?benchmark={ids['bench']}&file=../secret").status_code == 404


def test_train_download_before_prepare_is_404(db):
    ids = _seed(db)
    c = _client(db)
    assert c.get(f"/api/train/download?benchmark={ids['bench']}&file=sft.jsonl").status_code == 404


def test_train_submit_null_writes_plan(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/train/submit", json={"benchmark": ids["bench"], "backend": "null"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "planned"
    plan = c.get(f"/api/train/download?benchmark={ids['bench']}&file=train_plan.md")
    assert plan.status_code == 200 and "Training plan" in plan.text


def test_train_submit_art_reports_infra_required(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/train/submit", json={"benchmark": ids["bench"], "backend": "art"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infra_required"
    assert "art_train.py" in body["detail"]


def test_train_submit_unknown_backend_422(db):
    ids = _seed(db)
    c = _client(db)
    assert c.post("/api/train/submit",
                  json={"benchmark": ids["bench"], "backend": "nope"}).status_code == 422
