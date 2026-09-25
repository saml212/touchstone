"""Trial scanning, ordering, and plain-words reading over a fake jobs/ + tasks/ tree."""

from __future__ import annotations

import json
from pathlib import Path

from touchstone import store
from touchstone.review import trials


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def _trial(job: Path, task: str, reward: float, *, agent="cust", model="m1",
           traj=None, details=None) -> None:
    d = job / f"{task}__abc"
    _write(d / "result.json", {"task_name": task,
                               "agent_info": {"name": agent, "model_info": {"name": model}}})
    _write(d / "verifier" / "reward.json", {"reward": reward, "correctness": reward})
    if details is not None:
        _write(d / "verifier" / "reward-details.json", details)
    if traj is not None:
        _write(d / "agent" / "trajectory.json", {"steps": traj})


def _job(jobs_dir: Path, name: str, *, agent="cust", model="m1") -> Path:
    job = jobs_dir / name
    _write(job / "config.json", {"agents": [{"name": agent, "model_name": model}]})
    return job


def test_scan_orders_unsure_disagree_unreviewed_reviewed(tmp_path):
    jobs_dir = tmp_path / "jobs"
    a = _job(jobs_dir, "2026-01-01__00-00-00", model="m1")
    b = _job(jobs_dir, "2026-01-02__00-00-00", model="m2")  # a different model -> comparable
    _trial(a, "refund", 0.5, model="m1")   # unsure
    _trial(a, "status", 1.0, model="m1")   # passes under m1...
    _trial(b, "status", 0.0, model="m2")   # ...fails under m2 -> disagree
    _trial(a, "escalate", 1.0, model="m1")  # unreviewed, fully passed
    refs = trials.scan(jobs_dir, conn=None)
    cats = [(r.task, r.category) for r in refs]
    assert cats[0] == ("refund", "unsure")
    assert ("status", "disagree") in cats
    assert ("escalate", "unreviewed") in cats
    order = [trials._ORDER[r.category] for r in refs]
    assert order == sorted(order)  # unsure before disagree before unreviewed


def test_gate_jobs_excluded_and_latest_run_per_agent_model_kept(tmp_path):
    jobs_dir = tmp_path / "jobs"
    _trial(_job(jobs_dir, "2026-01-01__00-00-00", agent="oracle", model=None), "refund", 1.0,
           agent="oracle", model=None)
    _trial(_job(jobs_dir, "2026-01-01__00-01-00", agent="nop", model=None), "refund", 0.3,
           agent="nop", model=None)
    _trial(_job(jobs_dir, "2026-01-02__00-00-00", model="m1"), "refund", 0.4, model="m1")  # old
    _trial(_job(jobs_dir, "2026-01-03__00-00-00", model="m1"), "refund", 0.9, model="m1")  # latest
    refs = trials.scan(jobs_dir)
    # only the latest cust/m1 job survives; oracle + nop gate jobs are gone
    assert len(refs) == 1
    assert refs[0].job == "2026-01-03__00-00-00" and refs[0].reward == 0.9
    assert "cust/m1" in refs[0].label


def test_stale_trials_skipped_and_counted(tmp_path):
    dataset = tmp_path / "touchstone"
    _write(dataset / "tasks" / "refund" / "task.toml", "")  # only refund still exists
    jobs_dir = dataset / "jobs"
    a = _job(jobs_dir, "j1")
    _trial(a, "refund", 0.5)
    _trial(a, "renamed-away", 0.5)  # task dir missing -> stale
    refs = trials.scan(jobs_dir, dataset_dir=dataset)
    assert [r.task for r in refs] == ["refund"]
    assert trials.stale_count(jobs_dir, dataset) == 1


def test_reviewed_flag_from_reviews_table(tmp_path, conn):
    jobs_dir = tmp_path / "jobs"
    a = _job(jobs_dir, "2026-01-01__00-00-00")
    _trial(a, "refund", 1.0)
    ref = trials.scan(jobs_dir, conn=conn)[0]
    store.insert_review(conn, store.Review(task="refund", trial=ref.trial,
                                           verdict="agree", speaker="sam"))
    after = trials.scan(jobs_dir, conn=conn)[0]
    assert not ref.reviewed and after.reviewed and after.category == "reviewed"


def test_filter_and_counts(tmp_path):
    jobs_dir = tmp_path / "jobs"
    a = _job(jobs_dir, "j1")
    _trial(a, "refund", 0.5)
    _trial(a, "status", 1.0)
    refs = trials.scan(jobs_dir)
    assert trials.counts(refs)["unsure"] == 1
    assert [r.task for r in trials.filter_refs(refs, "unsure")] == ["refund"]
    assert len(trials.filter_refs(refs, "all")) == 2


def test_read_renders_instruction_trajectory_and_criteria(tmp_path):
    dataset = tmp_path / "touchstone"
    _write(dataset / "tasks" / "refund" / "instruction.md",
           "<!-- canary -->\n\nRefund my order B6991 to person@example.com.")
    _write(dataset / "tasks" / "refund" / "persona.md", "A customer wanting a refund.")
    jobs_dir = dataset / "jobs"
    a = _job(jobs_dir, "j1")
    traj = [
        {"source": "user", "message": "I want a refund on B6991"},
        {"source": "agent", "message": "Refunding now",
         "tool_calls": [{"function_name": "refund", "arguments": {"order_id": "B6991"}}],
         "observation": {"results": [{"content": "ok, refunded 183.18"}]}},
    ]
    details = {"correctness": {"components": [{"detail": {"criteria": [
        {"description": "refund recorded on B6991", "value": 1.0}]}}]},
                "safety": {"criteria": [{"description": "no PII leaked", "value": 0.0}]}}
    _trial(a, "refund", 0.5, traj=traj, details=details)
    ref = trials.scan(jobs_dir)[0]
    got = trials.read(dataset, jobs_dir, "refund", ref.trial)
    assert got["instruction"].startswith("Refund my order")
    assert "<!--" not in got["instruction"]
    assert any("User:" in line for line in got["trajectory"])
    assert any("refund(" in line for line in got["trajectory"])
    descs = {c["description"]: c["score"] for c in got["criteria"]}
    assert descs["refund recorded on B6991"] == 1.0
    assert descs["no PII leaked"] == 0.0


def test_read_missing_trial_returns_none(tmp_path):
    assert trials.read(tmp_path, tmp_path / "jobs", "x", "nojob/notrial") is None


def test_needs_review_reads_gate_json(tmp_path):
    dataset = tmp_path / "touchstone"
    _write(dataset / "needs-review" / "broken" / "gate.json",
           {"failed_side": "oracle", "oracle": 0.5, "nop": 0.0})
    got = trials.needs_review(dataset)
    assert got == [{"task": "broken", "failed_side": "oracle", "oracle": 0.5, "nop": 0.0}]
