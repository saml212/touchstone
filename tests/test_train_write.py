import json
import tomllib

from touchstone.harbor.jobs import Job
from touchstone.train.write import write_datasets


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


def _job(tmp_path, name, config):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


def _teacher(tmp_path):
    cfg = {"agents": [{"name": "touchstone.harbor.agent:TouchstoneAgent",
                       "model_name": "openai/gpt-4o-mini"}]}
    jd = _job(tmp_path, "teacher", cfg)
    _trial(jd, "t1", "ds/refund", reward=1.0, traj={"schema_version": "ATIF-v1.8"})
    _trial(jd, "t2", "ds/lookup", reward=1.0, traj={"schema_version": "ATIF-v1.8"})
    _trial(jd, "t3", "ds/stuck", reward=0.0)
    return jd


def _student(tmp_path):
    cfg = {"agents": [{"name": "touchstone.harbor.agent:TouchstoneAgent",
                       "model_name": "openai/gpt-4o-mini", "kwargs": {"mode": "replica"}}]}
    jd = _job(tmp_path, "student", cfg)
    _trial(jd, "s1", "ds/refund", reward=0.0)   # distill (teacher passes)
    _trial(jd, "s2", "ds/lookup", reward=1.0)   # hold_out
    _trial(jd, "s3", "ds/stuck", reward=0.0)    # stuck (teacher fails too)
    return jd


def test_write_produces_three_files_and_manifest(tmp_path):
    out = tmp_path / "train"
    written = write_datasets(out, [Job.read(_teacher(tmp_path))], [Job.read(_student(tmp_path))])

    distill_lines = (out / "distill.jsonl").read_text().splitlines()
    assert len(distill_lines) == 2
    first = json.loads(distill_lines[0])
    assert set(first) == {"task", "trial", "reward", "model", "trajectory"}

    rl = tomllib.loads((out / "rl_tasks.toml").read_text())
    assert rl.get("task", []) == []  # single-trial student -> no learnability band

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"] == {
        "distill": 1, "rl": 0, "hold_out": 1, "stuck": 1,
        "distill_trajectories": 2, "distill_missing_trajectory": 0,
    }
    assert manifest["inputs"]["teacher"][0]["model"] == "openai/gpt-4o-mini"
    assert manifest["threshold"] == 1.0 and manifest["touchstone_version"]
    assert written.manifest == manifest


def test_sentence_reports_routes_and_out(tmp_path):
    out = tmp_path / "train"
    written = write_datasets(out, [Job.read(_teacher(tmp_path))], [Job.read(_student(tmp_path))])
    s = written.sentence()
    assert "distill: 2 trajectories from 2 tasks" in s
    assert "rl: 0 tasks" in s and "hold-out: 1" in s and "stuck: 1" in s
    assert s.endswith(f"{out}/")


def test_rl_toml_carries_band_tasks(tmp_path):
    cfg = {"agents": [{"name": "replica", "model_name": "m"}]}
    jd = _job(tmp_path, "student", cfg)
    _trial(jd, "a", "ds/refund", reward=1.0)
    _trial(jd, "b", "ds/refund", reward=0.0)
    out = tmp_path / "train"
    written = write_datasets(out, [], [Job.read(jd)])
    rl = tomllib.loads((out / "rl_tasks.toml").read_text())["task"]
    assert rl == [{"name": "ds/refund", "path": "tasks/refund",
                   "pass_rate": 0.5, "attempts": 2}]
    assert "rl: 1 tasks (pass 0.50)" in written.sentence()
