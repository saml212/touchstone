import asyncio
import json

from touchstone.train.plugin import TrainPlugin


def _trial(job_dir, name, task_name, reward, traj=None):
    d = job_dir / name
    (d / "verifier").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": task_name,
        "agent_info": {"name": "replica", "model_info": {"name": "openai/gpt-4o-mini"}},
    }))
    (d / "verifier" / "reward.txt").write_text(f"{reward}\n")
    if traj is not None:
        (d / "agent").mkdir()
        (d / "agent" / "trajectory.json").write_text(json.dumps(traj))


def _job(tmp_path):
    jobs = tmp_path / "touchstone" / "jobs"
    d = jobs / "job"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(
        {"agents": [{"name": "replica", "model_name": "openai/gpt-4o-mini"}]}))
    _trial(d, "t1", "ds/refund", 1.0, traj={"schema_version": "ATIF-v1.8"})
    _trial(d, "t2", "ds/stuck", 0.0)
    return d


class _FakeJob:
    def __init__(self, job_dir):
        self.job_dir = job_dir


def test_plugin_writes_datasets_under_dataset_train_on_job_end(tmp_path):
    job_dir = _job(tmp_path)
    plugin = TrainPlugin()
    asyncio.run(plugin.on_job_start(_FakeJob(job_dir)))
    asyncio.run(plugin.on_job_end(object()))

    train = tmp_path / "touchstone" / "train"
    assert (train / "distill.jsonl").exists()
    manifest = json.loads((train / "manifest.json").read_text())
    # the job is both teacher and student: ds/refund distills (self-pass), ds/stuck is stuck
    assert manifest["counts"]["distill_trajectories"] == 1
    assert manifest["counts"]["hold_out"] == 1  # ds/refund passed -> hold_out for the student view
    assert manifest["counts"]["stuck"] == 1


def test_plugin_out_and_threshold_kwargs(tmp_path):
    job_dir = _job(tmp_path)
    out = tmp_path / "custom"
    written = TrainPlugin(out=str(out), threshold=0.5).write(job_dir)
    assert written.out == out
    assert (out / "rl_tasks.toml").exists()


def test_plugin_no_op_without_job_start(tmp_path):
    # on_job_end before on_job_start must not raise
    asyncio.run(TrainPlugin().on_job_end(object()))
