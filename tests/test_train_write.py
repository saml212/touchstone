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
        # lookup is the only passing student task (N=1) -> it distills, none held out; the
        # student's own passing lookup trial has no trajectory, so it is excluded (teacher's kept).
        "distill": 2, "rl": 0, "hold_out": 0, "stuck": 1,
        "distill_trajectories": 2, "distill_missing_trajectory": 1,
    }
    assert manifest["inputs"]["teacher"][0]["model"] == "openai/gpt-4o-mini"
    assert manifest["threshold"] == 1.0 and manifest["touchstone_version"]
    assert written.manifest == manifest


def test_sentence_reports_routes_and_out(tmp_path):
    out = tmp_path / "train"
    written = write_datasets(out, [Job.read(_teacher(tmp_path))], [Job.read(_student(tmp_path))])
    s = written.sentence()
    assert "distill: 2 trajectories from 2 tasks" in s
    assert "excluded 1: no trajectory artifact" in s  # the artifact-less passing trial is named
    assert "rl: 0 tasks" in s and "hold-out: 0" in s and "stuck: 1" in s
    assert s.endswith(f"{out}/")


def test_empty_rl_says_why_all_at_100(tmp_path):
    # The stranger got an empty rl_tasks.toml with no explanation: six trials, all at 100%, so
    # nothing lands in the 0<pass<1 learnability band. Both the sentence and the manifest say so.
    cfg = {"agents": [{"name": "replica", "model_name": "m"}]}
    jd = _job(tmp_path, "student", cfg)
    for i in range(6):
        _trial(jd, f"p{i}", f"ds/pass{i}", reward=1.0, traj={"schema_version": "ATIF-v1.8"})
    out = tmp_path / "train"
    written = write_datasets(out, [], [Job.read(jd)])
    assert not written.rl
    assert "rl: 0 tasks (none between 0% and 100% pass — all 6 at 100%)" in written.sentence()
    assert written.manifest["rl_note"] == "none between 0% and 100% pass — all 6 at 100%"
    # a run with a band task carries no note
    jd2 = _job(tmp_path, "banded", cfg)
    _trial(jd2, "a", "ds/x", reward=1.0)
    _trial(jd2, "b", "ds/x", reward=0.0)
    written2 = write_datasets(tmp_path / "train2", [], [Job.read(jd2)])
    assert written2.rl and written2.manifest["rl_note"] is None


def test_five_passing_trials_distill_at_least_four(tmp_path):
    # The stranger's bug head-on: 5 tasks passed at 1.0, distill was empty. Now >= 4 distill.
    cfg = {"agents": [{"name": "replica", "model_name": "m"}]}
    jd = _job(tmp_path, "baseline", cfg)
    for i in range(5):
        _trial(jd, f"p{i}", f"ds/pass{i}", reward=1.0, traj={"schema_version": "ATIF-v1.8"})
    out = tmp_path / "train"
    written = write_datasets(out, [], [Job.read(jd)])
    assert len(written.distill) >= 4
    assert len((out / "distill.jsonl").read_text().splitlines()) >= 4
    assert written.manifest["counts"]["hold_out"] == 1  # ~20% held out, not all five


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


def test_manifest_and_datasets_are_byte_identical_except_created_at(tmp_path):
    # Attack (stage-7 train): manifest reproducibility — the same inputs must produce byte-identical
    # distill.jsonl, rl_tasks.toml, and manifest.json (bar the created_at timestamp), so a re-run in
    # CI is a clean diff, not churn from set/dict ordering.
    teacher, student = [Job.read(_teacher(tmp_path))], [Job.read(_student(tmp_path))]
    a, b = tmp_path / "a", tmp_path / "b"
    write_datasets(a, teacher, student)
    write_datasets(b, teacher, student)

    assert (a / "distill.jsonl").read_bytes() == (b / "distill.jsonl").read_bytes()
    assert (a / "rl_tasks.toml").read_bytes() == (b / "rl_tasks.toml").read_bytes()

    def _drop_created(path):
        doc = json.loads(path.read_text())
        doc.pop("created_at")
        return doc

    assert _drop_created(a / "manifest.json") == _drop_created(b / "manifest.json")
    # everything but created_at must match byte-for-byte too: only that one line differs
    la = (a / "manifest.json").read_text().splitlines()
    lb = (b / "manifest.json").read_text().splitlines()
    differing = [x for x, y in zip(la, lb, strict=True) if x != y]
    assert all("created_at" in d for d in differing)
