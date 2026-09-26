import json

from touchstone.harbor.jobs import Job
from touchstone.train.datasets import distill, distill_excluded, rl_tasks, route


def _trial(job_dir, name, task_name, *, reward, traj=None, model="openai/gpt-4o-mini"):
    d = job_dir / name
    (d / "verifier").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": task_name,
        "agent_info": {"name": "replica", "model_info": {"name": model}},
    }))
    (d / "verifier" / "reward.txt").write_text(f"{reward}\n")
    if traj is not None:
        (d / "agent").mkdir()
        (d / "agent" / "trajectory.json").write_text(json.dumps(traj))
    return d


def _job(tmp_path, name, config=None):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config or {}))
    return d


def test_distill_keeps_only_passing_trials_with_trajectory(tmp_path):
    jd = _job(tmp_path, "teacher")
    _trial(jd, "t1__a", "ds/refund", reward=1.0, traj={"schema_version": "ATIF-v1.8"})
    _trial(jd, "t2__a", "ds/lookup", reward=0.0, traj={"x": 1})
    records = distill([Job.read(jd)])
    assert len(records) == 1
    r = records[0]
    assert r["task"] == "ds/refund" and r["trial"] == "t1__a"
    assert r["reward"] == 1.0 and r["model"] == "openai/gpt-4o-mini"
    assert r["trajectory"] == {"schema_version": "ATIF-v1.8"}


def test_distill_threshold_edge_099_vs_100(tmp_path):
    jd = _job(tmp_path, "teacher")
    _trial(jd, "t1__a", "ds/refund", reward=0.99, traj={"a": 1})
    job = Job.read(jd)
    assert distill([job], threshold=1.0) == []
    assert len(distill([job], threshold=0.99)) == 1


def test_distill_missing_trajectory_counted_not_crashed(tmp_path):
    jd = _job(tmp_path, "teacher")
    _trial(jd, "t1__a", "ds/refund", reward=1.0, traj=None)
    _trial(jd, "t2__a", "ds/lookup", reward=1.0, traj={"a": 1})
    job = Job.read(jd)
    assert len(distill([job])) == 1
    assert distill_excluded([job], {"ds/refund", "ds/lookup"}) == 1


def test_rl_band_excludes_zero_and_one(tmp_path):
    jd = _job(tmp_path, "student")
    # ds/refund: one pass one fail -> 0.5 (in band); ds/all: pass -> 1.0; ds/none: fail -> 0.0
    _trial(jd, "r1", "ds/refund", reward=1.0)
    _trial(jd, "r2", "ds/refund", reward=0.0)
    _trial(jd, "a1", "ds/all", reward=1.0)
    _trial(jd, "n1", "ds/none", reward=0.0)
    records = rl_tasks([Job.read(jd)])
    assert [r["task"] for r in records] == ["ds/refund"]
    r = records[0]
    assert r["pass_rate"] == 0.5 and r["attempts"] == 2 and r["path"] == "tasks/refund"


def test_route_teacher_rl_and_stuck(tmp_path):
    teacher = _job(tmp_path, "teacher")
    _trial(teacher, "t1", "ds/distillable", reward=1.0, traj={"a": 1})
    # teacher fails ds/stuck (no success there)
    _trial(teacher, "t2", "ds/stuck", reward=0.0)
    student = _job(tmp_path, "student")
    _trial(student, "s1", "ds/distillable", reward=0.0)          # rate 0 + teacher pass -> distill
    _trial(student, "s2a", "ds/rl", reward=1.0)                  # rate 0.5 -> rl
    _trial(student, "s2b", "ds/rl", reward=0.0)
    _trial(student, "s4", "ds/stuck", reward=0.0)              # rate 0 + no teacher pass -> stuck
    routing = route([Job.read(teacher)], [Job.read(student)])
    assert routing == {"ds/distillable": "distill", "ds/rl": "rl", "ds/stuck": "stuck"}


def test_route_splits_passing_tasks_80_20_never_all_holdout(tmp_path):
    # The stranger's bug: 5 tasks passed at 1.0 and all went to hold_out, distill was empty.
    student = _job(tmp_path, "student")
    for i in range(5):
        _trial(student, f"p{i}", f"ds/pass{i}", reward=1.0)  # all pass, no teacher
    routing = route([], [Job.read(student)])
    dests = list(routing.values())
    assert dests.count("distill") == 4 and dests.count("hold_out") == 1  # ~80/20, never 0/5


def test_route_single_pass_still_distills(tmp_path):
    student = _job(tmp_path, "student")
    _trial(student, "p0", "ds/only", reward=1.0)
    assert route([], [Job.read(student)]) == {"ds/only": "distill"}  # N=1 -> at least one distills
