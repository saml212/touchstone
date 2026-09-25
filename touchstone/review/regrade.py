"""Regrade a job after a criterion change and report which rewards moved.

`harbor job regrade` reruns only the verifier against the recorded outputs (no agent, no key), so a
criterion change is graded in seconds. This reads the source job's rewards, runs the regrade through
`touchstone.harbor.run` (local or remote, same sync rules as `run`), reads the new job, and diffs
per-task mean reward so the room can read out the trial that changed and any others that moved with
it. Trials Harbor refused to regrade are listed under `failed`, never counted as 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..harbor import jobs
from ..harbor import run as run_mod


@dataclass
class Delta:
    task: str
    before: float | None
    after: float | None

    def to_dict(self) -> dict:
        return {"task": self.task, "before": self.before, "after": self.after}


def diff_rewards(old: jobs.Job, new: jobs.Job) -> list[Delta]:
    """Per-task mean-reward changes between the source and the regraded job, largest move first."""
    before, after = jobs.mean_rewards(old), jobs.mean_rewards(new)
    deltas = []
    for task in sorted(set(before) | set(after)):
        b, a = before.get(task), after.get(task)
        if not _close(b, a):
            deltas.append(Delta(task=task, before=b, after=a))
    deltas.sort(key=lambda d: -abs((d.after or 0) - (d.before or 0)))
    return deltas


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9


def regrade_job(dataset_dir: str | Path, jobs_dir: str | Path, job_name: str,
                settings, runner=run_mod.regrade) -> dict:
    """Regrade one job against the current tasks/; return the new job name and the reward deltas.

    `runner` is the regrade call, injected so tests can supply a fixture job pair.
    """
    jobs_dir, dataset_dir = Path(jobs_dir), Path(dataset_dir)
    old = jobs.Job.read(jobs_dir / job_name)
    new_dir = runner(jobs_dir / job_name, dataset_dir / "tasks", settings=settings)
    new = jobs.Job.read(new_dir)
    graded, failed = _split_failed(new)
    skip = {f["task"] for f in failed}
    old_graded = jobs.Job(dir=old.dir, config=old.config,
                          trials=[t for t in old.trials if t.task_name not in skip])
    deltas = diff_rewards(old_graded, graded) if graded.trials else []
    return {"job": Path(new_dir).name, "deltas": [d.to_dict() for d in deltas], "failed": failed}


def _split_failed(job: jobs.Job) -> tuple[jobs.Job, list[dict]]:
    """Trials Harbor refused to regrade (a RegradeError, a missing artifact) are not 0% — they are
    reported apart, with Harbor's reason, so the room says "could not regrade" instead of "dropped"."""
    ok = [t for t in job.trials if not t.exception]
    failed = [{"task": t.task_name, "error": f"{t.exception}: {(t.error or '')[:200]}"}
              for t in job.trials if t.exception]
    return jobs.Job(dir=job.dir, config=job.config, trials=ok), failed
