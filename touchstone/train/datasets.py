"""Pure functions over Harbor `Job` values — the training-data theory of `docs/DESIGN.md`.

- `distill`: a teacher's passing trials become SFT trajectories (the student copies them).
- `rl_tasks`: tasks in the student's learnability band (0 < pass rate < 1); the verifier is the
  reward.
- `route`: per task, where it goes — distill, rl, hold_out, or stuck.

Nothing here knows any customer or domain. Trajectories are read from
`<trial>/agent/trajectory.json` in ATIF; a passing trial without one is skipped, not fatal
(`distill_skipped` counts them).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..harbor.jobs import Job, Trial

PASS = 1.0


def _load_trajectory(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _passing(jobs: list[Job], threshold: float) -> list[Trial]:
    return [t for job in jobs for t in job.trials
            if t.reward is not None and t.reward >= threshold]


def distill(jobs: list[Job], *, threshold: float = 1.0) -> list[dict]:
    """One record per trial whose reward >= threshold, with its ATIF trajectory loaded.

    Trials whose `agent/trajectory.json` is missing or unreadable are skipped (see
    `distill_skipped`); the record carries the task, trial dir name, reward, model, and trajectory.
    """
    records = []
    for trial in _passing(jobs, threshold):
        if trial.trajectory_path is None:
            continue
        trajectory = _load_trajectory(trial.trajectory_path)
        if trajectory is None:
            continue
        records.append({
            "task": trial.task_name,
            "trial": trial.trajectory_path.parent.parent.name,
            "reward": trial.reward,
            "model": trial.model,
            "trajectory": trajectory,
        })
    return records


def distill_skipped(jobs: list[Job], *, threshold: float = 1.0) -> int:
    """Passing trials that would distill but have no readable trajectory — counted, not dropped."""
    skipped = 0
    for trial in _passing(jobs, threshold):
        if trial.trajectory_path is None or _load_trajectory(trial.trajectory_path) is None:
            skipped += 1
    return skipped


def _task_dir(trial: Trial) -> str:
    """The task dir relative to the dataset root: `tasks/<name>` (jobs/ sits beside tasks/)."""
    return f"tasks/{trial.task_name.split('/')[-1]}"


def _pass_rate(trials: list[Trial]) -> float:
    passed = sum(1 for t in trials if t.reward is not None and t.reward >= PASS)
    return passed / len(trials)


def _tasks(jobs: list[Job]) -> dict[str, list[Trial]]:
    """Every task name -> its trials across the given jobs, sorted by task name."""
    tasks: dict[str, list[Trial]] = {}
    for job in jobs:
        for trial in job.trials:
            tasks.setdefault(trial.task_name, []).append(trial)
    return dict(sorted(tasks.items()))


def rl_tasks(jobs: list[Job]) -> list[dict]:
    """Student tasks in the learnability band: keep those with 0 < pass rate < 1 across the jobs."""
    records = []
    for name, trials in _tasks(jobs).items():
        rate = _pass_rate(trials)
        if 0 < rate < 1:
            records.append({
                "task": name,
                "path": _task_dir(trials[0]),
                "pass_rate": rate,
                "attempts": len(trials),
            })
    return records


def _teacher_passes(teacher_jobs: list[Job], threshold: float) -> set[str]:
    return {t.task_name for t in _passing(teacher_jobs, threshold)}


def route(teacher_jobs: list[Job], student_jobs: list[Job], *,
          threshold: float = 1.0) -> dict[str, str]:
    """Per student task, its destination: distill, rl, hold_out, or stuck. Sorted by task name."""
    teacher = _teacher_passes(teacher_jobs, threshold)
    routing: dict[str, str] = {}
    for name, trials in _tasks(student_jobs).items():
        rate = _pass_rate(trials)
        if rate >= 1:
            routing[name] = "hold_out"
        elif rate > 0:
            routing[name] = "rl"
        elif name in teacher:
            routing[name] = "distill"
        else:
            routing[name] = "stuck"
    return routing
