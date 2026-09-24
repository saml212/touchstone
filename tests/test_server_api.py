"""JSON API + static shell over a seeded file-based project, plus scoreboard parity."""

import time

from fastapi.testclient import TestClient

from touchstone import policies, store, tasks
from touchstone.bench import benchmark, scoreboard
from touchstone.checks import Check
from touchstone.config import Settings
from touchstone.policies import Policy
from touchstone.server import create_app


def _seed(db):
    root = Settings(db_path=db).root
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

    check = Check(kind="contains", params={"values": ["refund"], "mode": "any"},
                  name="refund", source="manual")
    check.id = check.name
    policies.write_policies(root, [Policy(check=check, enabled=True)])
    for name, ep, tags in [("t1", ep1, ["support"]), ("t2", ep2, ["failure"])]:
        tasks.write_task(root, tasks.Task(
            name=name, episode_id=ep.id, tags=tags,
            context={"messages": [{"role": "user", "content": "I want a refund"}], "tools": []},
            reference={"content": "refund ok", "tool_calls": []}, checks=[check]))
    benchmark.create(root, "demo", task_names=["t1", "t2"])
    return {"ep1": ep1.id, "ep2": ep2.id, "check": "refund",
            "t1": "t1", "t2": "t2", "bench": "demo"}


def _client(db):
    return TestClient(create_app(
        Settings(db_path=db, provider="scripted", agent_provider="scripted")))


def test_overview_counts_match_files(db):
    ids = _seed(db)
    c = _client(db)
    o = c.get("/api/overview").json()
    assert o["episodes"]["total"] == 2
    assert o["episodes"]["outcomes"] == {"ok": 1, "fail": 1}
    assert o["spans"] == 2
    assert o["checks"] == {"total": 1, "enabled": 1, "by_source": {"manual": 1}}
    assert o["tasks"]["total"] == 2 and o["benchmarks"] == 1 and o["runs"] == 0
    assert set(o["tasks"]["by_status"]) == {"active", "needs_checks", "needs_solution"}
    assert o["rooms"] == {"total": 0, "open": 0}
    assert ids


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


def test_check_create_invalid_params_returns_422_with_message(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/checks", json={"kind": "contains", "params": {}, "name": "bad"})
    assert r.status_code == 422
    assert "values" in r.json()["detail"]


def test_check_create_patch_and_kinds(db):
    _seed(db)
    c = _client(db)
    assert "contains" in c.get("/api/checks/kinds").json()["kinds"]
    made = c.post("/api/checks", json={
        "kind": "max_length", "params": {"max": 10}, "name": "brief",
        "because": "keep it short"}).json()
    assert made["source"] == "manual" and made["enabled"] is True

    off = c.patch("/api/checks/brief", json={"enabled": False}).json()
    assert off["enabled"] is False
    bad = c.patch("/api/checks/brief", json={"params": {"max": "nope"}})
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


def test_task_add_and_remove_check(db):
    ids = _seed(db)
    c = _client(db)
    first = c.post(f"/api/tasks/{ids['t1']}/checks",
                   json={"kind": "min_length", "params": {"min": 1}, "name": "nonempty"}).json()
    n = len(first["checks"])
    again = c.post(f"/api/tasks/{ids['t1']}/checks",
                   json={"kind": "min_length", "params": {"min": 1}, "name": "nonempty"}).json()
    assert len(again["checks"]) == n  # add is idempotent (dedupe)

    after = c.request("DELETE", f"/api/tasks/{ids['t1']}/checks/nonempty").json()
    assert "nonempty" not in {ch["name"] for ch in after["checks"]}
    assert c.post("/api/tasks/nope/checks", json={"kind": "min_length",
                                                  "params": {"min": 1}}).status_code == 404


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
    assert r2.status_code == 200


def test_benchmark_create_and_run_report_proof(db):
    _seed(db)
    c = _client(db)
    made = c.post("/api/benchmarks", json={"name": "all", "all": True}).json()
    assert made["task_count"] == 2
    assert c.post("/api/benchmarks",
                  json={"name": "empty", "tags": ["nomatch"]}).status_code == 422

    rid = _run_and_wait(c, "all")
    rid2 = _run_and_wait(c, "all")

    report = c.get(f"/api/report?run={rid}&run={rid2}").json()
    assert {m["model_spec"] for m in report["models"]} == {"scripted"}
    assert len(report["models"]) == 2 and report["checks"] and report["tags"]

    proof = c.get(f"/api/proof?candidate={rid}&incumbent={rid2}").json()
    assert proof["counts"]["both_pass"] + proof["counts"]["both_fail"] == 2
    assert c.get(f"/api/proof?candidate={rid}&incumbent=nope").status_code == 404
    assert c.get("/api/runs/nope").status_code == 404


def test_api_and_cli_scoreboard_identical(db):
    _seed(db)
    c = _client(db)
    c.post("/api/benchmarks", json={"name": "all", "all": True})
    rid = _run_and_wait(c, "all")

    api_board = c.get(f"/api/report?run={rid}").json()
    conn = store.connect(db)
    try:
        cli_board = scoreboard(conn, Settings(db_path=db).root, [rid])
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


def test_create_room_accepts_task_key_and_summarizes(db):
    # The room create route must resolve the task whether the body carries `task_id`
    # or the shorter `task`; a dropped key silently degrades the opening to the
    # taskless "Let's define what good looks like" fallback instead of a summary.
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/rooms", json={"task": ids["t1"], "topic": "needs a positive check"}).json()
    assert r["room"]["task_id"] == ids["t1"]
    opening = r["messages"][0]["text"]
    assert "Let's define what good looks like" not in opening
    assert "t1" in opening and "refund" in opening


def test_export_harbor_points_at_tasks(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/export/harbor", json={}).json()
    assert "harbor run -p" in r["command"] and r["tasks_path"].endswith("tasks")


def test_static_shell_and_app_js(db):
    c = _client(db)
    root = c.get("/")
    assert root.status_code == 200 and "Touchstone" in root.text and 'id="nav"' in root.text
    js = c.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def _run_and_wait(c, target, spec="scripted"):
    run = c.post("/api/runs", json={"target": target, "model_spec": spec}).json()
    for _ in range(100):
        cur = c.get(f"/api/runs/{run['id']}").json()
        if cur["finished_at"]:
            return run["id"]
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_train_prepare_and_download(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/train/prepare", json={"target": ids["bench"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["counts"]["rl_tasks"] == 2
    assert set(body["files"]) >= {"sft", "preference", "rl_tasks", "manifest"}

    dl = c.get(f"/api/train/download?benchmark={ids['bench']}&file=rl_tasks.jsonl")
    assert dl.status_code == 200 and dl.text.strip()
    assert c.get(
        f"/api/train/download?benchmark={ids['bench']}&file=../secret").status_code == 404


def test_train_download_before_prepare_is_404(db):
    ids = _seed(db)
    c = _client(db)
    assert c.get(
        f"/api/train/download?benchmark={ids['bench']}&file=sft.jsonl").status_code == 404


def test_train_submit_null_writes_plan(db):
    ids = _seed(db)
    c = _client(db)
    r = c.post("/api/train/submit", json={"target": ids["bench"], "backend": "null"})
    assert r.status_code == 200 and r.json()["status"] == "planned"
    plan = c.get(f"/api/train/download?benchmark={ids['bench']}&file=train_plan.md")
    assert plan.status_code == 200 and "Training plan" in plan.text


def test_train_submit_art_reports_infra_required(db):
    ids = _seed(db)
    c = _client(db)
    body = c.post("/api/train/submit", json={"target": ids["bench"], "backend": "art"}).json()
    assert body["status"] == "infra_required"
    assert "art_train.py" in body["detail"]


def test_train_submit_unknown_backend_422(db):
    ids = _seed(db)
    c = _client(db)
    assert c.post("/api/train/submit",
                  json={"target": ids["bench"], "backend": "nope"}).status_code == 422


def test_check_eval_malformed_tool_calls_is_422_not_500(db):
    ids = _seed(db)
    c = _client(db)
    for bad in (5, "astring", [1, 2, 3], {"name": "x"}):
        r = c.post(f"/api/checks/{ids['check']}/eval", json={"text": "x", "tool_calls": bad})
        assert r.status_code == 422, (bad, r.status_code)


def test_create_run_rejects_nonpositive_concurrency(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/runs", json={"target": "demo", "model_spec": "scripted", "concurrency": 0})
    assert r.status_code == 422
    assert "concurrency" in r.json()["detail"]


def test_sample_endpoint_returns_frontier(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/sample", json={"target": "demo", "student": "scripted", "variants": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == "scripted"
    assert "frontier" in body and set(body["frontier_split"]) == {"learnability", "only_incumbent"}
    # validation errors are one clear sentence
    assert c.post("/api/sample", json={"target": "demo"}).status_code == 422
    assert c.post("/api/sample", json={"target": "nope", "student": "scripted"}).status_code == 404


def test_distill_endpoint_packages_frontier(db):
    _seed(db)
    c = _client(db)
    c.post("/api/sample", json={"target": "demo", "student": "scripted", "variants": 1})
    r = c.post("/api/distill", json={"target": "demo", "student": "scripted", "backend": "null"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "frontier" in body and "plan" in body and body["backend"] == "null"
    assert c.get("/api/loop/demo").json()["student"] == "scripted"


def test_teach_task_endpoint(db):
    _seed(db)
    c = _client(db)
    r = c.post("/api/tasks/t1/teach", json={})
    assert r.status_code == 200 and "status" in r.json() and "teacher" in r.json()
    assert c.post("/api/tasks/nope/teach", json={}).status_code == 404
