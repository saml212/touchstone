import json

from touchstone.harbor.jobs import (
    NO_TRAJECTORY,
    Job,
    compare,
    pass_rates,
    ran_tasks,
    task_outcomes,
)


def _trial(job_dir, name, task_name, *, reward=None, rewards=None, exc=None, traj=False,
           agent="oracle", model=None, manifest=None):
    d = job_dir / name
    (d / "verifier").mkdir(parents=True)
    result = {"task_name": task_name, "agent_info": {"name": agent,
              "model_info": {"name": model} if model else None}}
    if exc:
        result["exception_info"] = {"exception_type": exc}
    (d / "result.json").write_text(json.dumps(result))
    if rewards is not None:
        (d / "verifier" / "reward.json").write_text(json.dumps(rewards))
    elif reward is not None:
        (d / "verifier" / "reward.txt").write_text(f"{reward}\n")
    if traj:
        (d / "agent").mkdir()
        (d / "agent" / "trajectory.json").write_text("{}")
    if manifest is not None:
        (d / "artifacts").mkdir()
        (d / "artifacts" / "manifest.json").write_text(json.dumps(manifest))
    return d


# A Harbor artifacts manifest whose agent-trajectory collection failed (the agent never ran).
_FAILED_TRAJ = [{"source": "/logs/agent/trajectory.json", "destination": "agent/trajectory.json",
                 "type": "file", "status": "failed"},
                {"source": "/app/output.json", "destination": "output.json",
                 "type": "file", "status": "failed"}]
_OK_TRAJ = [{"source": "/logs/agent/trajectory.json", "destination": "agent/trajectory.json",
             "type": "file", "status": "ok"}]


def _job(tmp_path, name="job"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"tasks": []}))
    return d


def test_read_trial_fields_reward_txt_and_trajectory(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__aaa", "ds/t1", reward=1.0, traj=True, model="anthropic/claude")
    job = Job.read(jd)
    assert len(job.trials) == 1
    t = job.trials[0]
    assert t.task_name == "ds/t1" and t.agent == "oracle" and t.model == "anthropic/claude"
    assert t.reward == 1.0 and t.passed is True
    assert t.trajectory_path is not None and t.trajectory_path.name == "trajectory.json"


def test_reward_json_multidim_scalar_is_mean(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__aaa", "ds/t1", rewards={"correctness": 1.0, "safety": 0.0})
    t = Job.read(jd).trials[0]
    assert t.rewards == {"correctness": 1.0, "safety": 0.0}
    assert t.reward == 0.5 and t.passed is False


def test_missing_reward_records_exception_and_no_pass(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__aaa", "ds/t1", exc="RewardFileNotFoundError")
    t = Job.read(jd).trials[0]
    assert t.reward is None and t.passed is False and t.exception == "RewardFileNotFoundError"


def test_pass_rates_average_per_task(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__a", "ds/t1", reward=1.0)
    _trial(jd, "t1__b", "ds/t1", reward=0.0)
    _trial(jd, "t2__a", "ds/t2", reward=1.0)
    assert pass_rates(Job.read(jd)) == {"ds/t1": 0.5, "ds/t2": 1.0}


def test_no_trajectory_manifest_failed_is_an_error_not_a_zero(tmp_path):
    jd = _job(tmp_path)
    # A real agent-under-test trial: the verifier wrote a 0, but Harbor's manifest says the
    # trajectory was never collected. That 0 is false — it must read as an error, not a reward.
    _trial(jd, "t1__aaa", "ds/t1", reward=0.0, agent="touchstone", model="m",
           manifest=_FAILED_TRAJ)
    t = Job.read(jd).trials[0]
    assert t.reward is None and t.passed is False and t.error == NO_TRAJECTORY


def test_ok_manifest_and_trajectory_keep_the_reward(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__aaa", "ds/t1", reward=0.0, agent="touchstone", model="m", traj=True,
           manifest=_OK_TRAJ)
    t = Job.read(jd).trials[0]
    assert t.reward == 0.0 and t.error is None


def test_gate_agent_missing_trajectory_keeps_its_reward(tmp_path):
    # oracle/nop never write a trajectory; a failed manifest entry must not zero out gate grading.
    jd = _job(tmp_path)
    _trial(jd, "t1__aaa", "ds/t1", reward=1.0, agent="oracle", manifest=_FAILED_TRAJ)
    t = Job.read(jd).trials[0]
    assert t.reward == 1.0 and t.error is None


def test_task_outcomes_shows_error_when_agent_never_ran(tmp_path):
    jd = _job(tmp_path)
    _trial(jd, "t1__a", "ds/t1", reward=1.0, agent="touchstone", model="m", traj=True,
           manifest=_OK_TRAJ)
    _trial(jd, "t2__a", "ds/t2", reward=0.0, agent="touchstone", model="m", manifest=_FAILED_TRAJ)
    job = Job.read(jd)
    outcomes = task_outcomes(job)
    assert outcomes["ds/t1"] == 1.0 and outcomes["ds/t2"] == NO_TRAJECTORY
    assert ran_tasks(job) == {"ds/t1"}


def test_compare_splits_tasks_both_only_neither(tmp_path):
    a = _job(tmp_path, "a")
    _trial(a, "t1__a", "ds/t1", reward=1.0)
    _trial(a, "t2__a", "ds/t2", reward=1.0)
    _trial(a, "t3__a", "ds/t3", reward=0.0)
    b = _job(tmp_path, "b")
    _trial(b, "t1__b", "ds/t1", reward=1.0)
    _trial(b, "t2__b", "ds/t2", reward=0.0)
    _trial(b, "t3__b", "ds/t3", reward=0.0)
    out = compare(Job.read(a), Job.read(b))
    assert out == {"both": ["ds/t1"], "only_a": ["ds/t2"], "only_b": [], "neither": ["ds/t3"]}
