"""Pure functions over Harbor `Job` values — the training-data theory of `docs/DESIGN.md`.

- `distill`: passing trials become SFT trajectories (a student copies them). Which tasks distill is
  decided by `route`; `write_datasets` filters these records to the distill-routed tasks.
- `rl_tasks`: tasks in the student's learnability band (0 < pass rate < 1); the verifier is the
  reward.
- `route`: per task, where it goes — distill, rl, hold_out, or stuck. Passing tasks (rate 1) split
  ~80/20 distill/hold_out so a small all-passing run never sends every trial to hold_out.

Nothing here knows any customer or domain. Trajectories are read from
`<trial>/agent/trajectory.json` in ATIF; a passing trial routed to distill without one is skipped,
not fatal (`distill_excluded` counts them).
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
    `distill_excluded`); the record carries the task, trial dir name, reward, model, and trajectory.
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


def distill_excluded(jobs: list[Job], distill_tasks: set[str], *, threshold: float = 1.0) -> int:
    """Passing trials of distill-routed tasks with no readable trajectory — counted, not fatal."""
    excluded = 0
    for trial in _passing(jobs, threshold):
        if trial.task_name in distill_tasks and (
                trial.trajectory_path is None or _load_trajectory(trial.trajectory_path) is None):
            excluded += 1
    return excluded


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


def rl_note(student_jobs: list[Job]) -> str | None:
    """Why rl_tasks is empty, or None when the learnability band has tasks. RL keeps tasks whose
    pass rate is strictly between 0 and 1; a single-attempt run scores only 0 or 1, so it never
    fills the band — the note says so instead of leaving an unexplained empty rl_tasks.toml."""
    rates = [_pass_rate(trials) for trials in _tasks(student_jobs).values()]
    if not rates or any(0 < rate < 1 for rate in rates):
        return None
    n = len(rates)
    passing = sum(1 for rate in rates if rate >= PASS)
    if passing == n:
        return f"none between 0% and 100% pass — all {n} at 100%"
    if passing == 0:
        return f"none between 0% and 100% pass — all {n} at 0%"
    return f"none between 0% and 100% pass — {passing} at 100%, {n - passing} at 0%"


def _teacher_passes(teacher_jobs: list[Job], threshold: float) -> set[str]:
    return {t.task_name for t in _passing(teacher_jobs, threshold)}


def _split_passing(passing: list[str]) -> tuple[set[str], set[str]]:
    """~80/20 (distill, hold_out) over the passing tasks, deterministic by sorted name. At least one
    distills whenever there is a passing task, so a tiny all-passing run never holds out all."""
    ordered = sorted(passing)
    n_hold = len(ordered) // 5  # ~20%, and 0 for n < 5 so a single pass still distills
    hold = set(ordered[len(ordered) - n_hold:]) if n_hold else set()
    return set(ordered) - hold, hold


def _destination(rate: float, name: str, teacher: set[str], to_distill: set[str]) -> str:
    if rate >= PASS:
        return "distill" if name in to_distill else "hold_out"
    if rate > 0:
        return "rl"
    return "distill" if name in teacher else "stuck"


def route(teacher_jobs: list[Job], student_jobs: list[Job], *,
          threshold: float = 1.0) -> dict[str, str]:
    """Per student task, its destination: distill, rl, hold_out, or stuck. Sorted by task name.

    Passing tasks (rate 1) split ~80/20 distill/hold_out — never all hold_out — so passing trials
    are the SFT signal on a small run with no teacher, not silently discarded."""
    teacher = _teacher_passes(teacher_jobs, threshold)
    tasks = _tasks(student_jobs)
    passing = [name for name, trials in tasks.items() if _pass_rate(trials) >= PASS]
    to_distill, _ = _split_passing(passing)
    return {name: _destination(_pass_rate(trials), name, teacher, to_distill)
            for name, trials in tasks.items()}
