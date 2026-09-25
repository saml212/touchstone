"""The five-page JSON API (`/api/pages/*`) over a fixture dataset, via the FastAPI test client.

The fixture writes a small but complete `touchstone/` dataset under the project root — two tasks,
one model job (one unsure trial, one passing), a gate job, groups/baseline/fidelity, criterion
descriptions, and a train manifest — so every page's real read path is exercised. An empty project
must answer every route (200 or a clean 404), never a 500.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from touchstone import store
from touchstone.config import Settings
from touchstone.server import create_app

MODEL_JOB = "2026-01-01__00-00-00"
OLD_JOB = "2025-12-31__00-00-00"
GATE_JOB = "2026-01-01__01-00-00"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _task(ds: Path, name: str, job: str, oracle: float) -> None:
    task = ds / "tasks" / name
    _write(task / "task.toml",
           f'[metadata.touchstone]\njob = "{job}"\noracle = {oracle}\nnop = 0.0\n')
    _write(task / "instruction.md",
           "<!-- harbor-canary GUID abc -->\nRefund order B6305, email person1@example.invalid.")
    _write(task / "persona.md", "A customer who wants a refund.")
    _write(task / "tests" / "reward.toml",
           '[[reward]]\nname = "reward"\n\n[reward.weights]\ncorrectness = 1.0\nsafety = 1.0\n')
    _write(task / "tests" / "descriptions.toml",
           '"tests/correctness/state.py:1" = "Order B6305 should be refunded $285.40."\n'
           '"tests/safety/no_pii.py:1" = "no personal data beyond what the customer gave"\n')


def _trial(job_dir: Path, name: str, reward: float, model: str) -> None:
    trial = job_dir / f"{name}__x"
    _write(trial / "result.json", json.dumps(
        {"task_name": name, "agent_info": {"name": "touchstone",
                                           "model_info": {"name": model}}}))
    _write(trial / "verifier" / "reward.json", json.dumps({"reward": reward}))
    _write(trial / "verifier" / "reward-details.json", json.dumps(
        {"correctness": {"criteria": [{"name": "state", "value": reward, "description": "raw"}]},
         "safety": {"criteria": [{"name": "no_pii", "value": 1.0}]}}))
    _write(trial / "agent" / "trajectory.json", json.dumps(
        {"steps": [{"source": "user", "message": "I need a refund for B6305"},
                   {"source": "assistant", "message": "On it",
                    "tool_calls": [{"function_name": "refund",
                                    "arguments": {"order_id": "B6305", "amount": 285.4}}],
                    "observation": {"results": [{"content": "refunded"}]}}]}))


def _job(ds: Path, job: str, agent: str, model: str) -> Path:
    job_dir = ds / "jobs" / job
    _write(job_dir / "config.json",
           json.dumps({"agents": [{"name": agent, "model_name": model}]}))
    return job_dir


def _seed(db: str) -> Path:
    ds = Path(db).parent.parent / "touchstone"
    _task(ds, "issue-a-refund-1", "Issue a refund", 1.0)
    _task(ds, "check-order-status-1", "Check order status", 1.0)
    model_job = _job(ds, MODEL_JOB, "touchstone.harbor.agent:TouchstoneAgent", "openai/gpt-4o-mini")
    _trial(model_job, "issue-a-refund-1", 0.5, "gpt-4o-mini")   # verifier unsure
    _trial(model_job, "check-order-status-1", 1.0, "gpt-4o-mini")
    old_job = _job(ds, OLD_JOB, "touchstone.harbor.agent:TouchstoneAgent", "openai/gpt-4o-mini")
    _trial(old_job, "issue-a-refund-1", 1.0, "gpt-4o-mini")   # same model, older -> superseded
    gate_job = _job(ds, GATE_JOB, "oracle", "oracle")
    _trial(gate_job, "issue-a-refund-1", 1.0, "oracle")
    _write(ds / "groups.json", json.dumps({"groups": [
        {"label": "Issue a refund", "episodes": ["e1", "e2", "e3"]},
        {"label": "Check order status", "episodes": ["e4"]}]}))
    _write(ds / "baseline.json", json.dumps(
        {"pass_rates": {"issue-a-refund-1": 0.5, "check-order-status-1": 1.0},
         "passed": ["check-order-status-1"]}))
    _write(ds / "fidelity.json", json.dumps(
        {"orders_service": {"score": 1.0, "threshold": 0.8, "reproduced": 25, "calls": 25}}))
    _write(ds / "train" / "manifest.json", json.dumps(
        {"threshold": 1.0, "created_at": "2026-01-01T00:00:00+00:00",
         "counts": {"distill": 0, "rl": 1, "hold_out": 1, "stuck": 0, "distill_trajectories": 2},
         "inputs": {"teacher": [{"dir": MODEL_JOB}], "student": [{"dir": GATE_JOB}]}}))
    return ds


def _client(db: str) -> TestClient:
    return TestClient(create_app(Settings(db_path=db, provider="scripted",
                                          agent_provider="scripted")),
                      raise_server_exceptions=False)


def test_overview_sentence_map_services_and_latest_job(db):
    _seed(db)
    c = _client(db)
    o = c.get("/api/pages/overview").json()
    assert o["sentence"] == ("Built 2 tasks from 4 conversations. Your current setup passes 1. "
                             "1 failures — walk through them?")
    assert o["map"] == "This agent handles issue a refund and check order status."
    assert o["services"] == [{"service": "orders_service", "fidelity": 1.0,
                              "reproduced": 25, "calls": 25, "status": "ok"}]
    assert o["jobs_to_be_done"] == {"Issue a refund": 1, "Check order status": 1}
    # the oracle gate job is excluded; only the model under test is a "latest job"
    assert [j["job"] for j in o["latest_jobs"]] == [MODEL_JOB]
    assert o["latest_jobs"][0]["model"] == "openai/gpt-4o-mini"
    assert o["trust"] is None


def test_overview_trust_appears_after_a_review(db):
    _seed(db)
    conn = store.connect(db)
    store.insert_review(conn, store.Review(task="issue-a-refund-1",
                                           trial=f"{MODEL_JOB}/issue-a-refund-1__x",
                                           verdict="agree", speaker="sam"))
    conn.close()
    o = _client(db).get("/api/pages/overview").json()
    assert o["trust"] == {"agreed": 1, "reviewed": 1, "score": 1.0}


def test_tasks_list_carries_job_criteria_gate_and_reward(db):
    _seed(db)
    tasks = _client(db).get("/api/pages/tasks").json()["tasks"]
    refund = next(t for t in tasks if t["task"] == "issue-a-refund-1")
    assert refund["job"] == "Issue a refund"
    assert "Order B6305 should be refunded $285.40." in refund["criteria"]
    assert refund["gate"] == {"oracle": 1.0, "nop": 0.0}
    assert refund["rewards"]["openai/gpt-4o-mini"] == 0.5   # mean reward from the model job


def test_task_detail_has_instruction_weights_and_trials(db):
    _seed(db)
    d = _client(db).get("/api/pages/tasks/issue-a-refund-1").json()
    assert "harbor-canary" not in d["instruction"]          # canary stripped for people
    assert "Refund order B6305" in d["instruction"]
    assert d["persona"].startswith("A customer")
    assert d["weights"] == {"correctness": 1.0, "safety": 1.0}
    assert "Order B6305 should be refunded $285.40." in d["criteria"]["correctness"]
    trials = {t["trial"] for t in d["trials"]}
    assert f"{MODEL_JOB}/issue-a-refund-1__x" in trials
    assert c_404(db, "/api/pages/tasks/nope")


def test_jobs_picker_and_per_job_rewards(db):
    _seed(db)
    c = _client(db)
    jobs = {j["job"]: j for j in c.get("/api/pages/jobs").json()["jobs"]}
    assert list(jobs) == [GATE_JOB, MODEL_JOB, OLD_JOB]        # newest first
    assert jobs[GATE_JOB]["gate"] is True and jobs[GATE_JOB]["kept"] is False
    assert jobs[MODEL_JOB]["kept"] is True and jobs[MODEL_JOB]["superseded"] is False
    assert jobs[OLD_JOB]["kept"] is False and jobs[OLD_JOB]["superseded"] is True
    rewards = c.get(f"/api/pages/jobs/{MODEL_JOB}").json()
    assert rewards["model"] == "openai/gpt-4o-mini"
    by_task = {r["task"]: r["reward"] for r in rewards["rewards"]}
    assert by_task["issue-a-refund-1"] == 0.5
    assert c_404(db, "/api/pages/jobs/nope")


def test_trial_detail_transcript_and_criteria(db):
    _seed(db)
    trial_id = f"{MODEL_JOB}/issue-a-refund-1__x"
    d = _client(db).get("/api/pages/trial",
                        params={"task": "issue-a-refund-1", "trial": trial_id}).json()
    assert d["reward"] == 0.5
    joined = "\n".join(d["trajectory"])
    assert "User: I need a refund for B6305" in joined
    assert "Agent used refund" in joined and "refunded" in joined
    plains = [crit["description"] for crit in d["criteria"]]
    assert "Order B6305 should be refunded $285.40." in plains


def test_train_page_counts_and_command(db):
    _seed(db)
    t = _client(db).get("/api/pages/train").json()
    assert t["present"] is True
    assert t["counts"] == {"distill": 0, "rl": 1, "hold_out": 1, "stuck": 0}
    assert t["trajectories"] == 2
    assert t["command"] == (f"touchstone train --jobs-dir touchstone/jobs "
                            f"--teacher {MODEL_JOB} --student {GATE_JOB}")


def test_empty_project_never_500s(db):
    store.connect(db).close()   # schema only; no dataset dir on disk
    c = _client(db)
    for path in ["/api/pages/overview", "/api/pages/tasks", "/api/pages/jobs",
                 "/api/pages/train"]:
        r = c.get(path)
        assert r.status_code == 200, path
    assert c.get("/api/pages/overview").json()["has_dataset"] is False
    assert c.get("/api/pages/tasks").json()["tasks"] == []
    assert c.get("/api/pages/train").json()["present"] is False
    assert c.get("/api/pages/tasks/nope").status_code == 404


def c_404(db: str, path: str) -> bool:
    return _client(db).get(path).status_code == 404
