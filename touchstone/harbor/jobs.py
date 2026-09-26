"""Read a Harbor job directory. Harbor owns the run record; Touchstone only reads it.

A job directory holds one `result.json`/`config.json` and one directory per trial. Each trial
directory has its own `result.json`, a `verifier/reward.{txt,json}` the verifier wrote, and an
`agent/trajectory.json`. This module turns that into `Job` and `Trial` values and computes pass
rates and a per-task comparison between two jobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

NO_TRAJECTORY = "agent did not run (no trajectory)"
_GATE_AGENTS = ("oracle", "nop")
_TRAJECTORY_SUFFIX = "agent/trajectory.json"


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_gate_agent(agent: str | None) -> bool:
    """Harbor's oracle/nop gate agents never write an agent trajectory; only the agent under test
    is expected to, so the no-trajectory rule below is scoped away from them."""
    return bool(agent) and any(g in agent.lower() for g in _GATE_AGENTS)


def _manifest_entries(trial_dir: Path) -> list:
    """The artifacts manifest as its list of entries; [] when absent or malformed."""
    try:
        data = json.loads((trial_dir / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _agent_ran(trial_dir: Path, trajectory_exists: bool, agent: str | None) -> bool:
    """False when the agent under test produced no trajectory. Decided from Harbor's own artifacts
    manifest (the record of what it collected): a `failed` trajectory entry — or the file missing
    once a manifest exists — means the agent never ran. Gate agents and manifest-less unit fixtures
    are trusted as-is, so gate grading and existing tests are unaffected."""
    if _is_gate_agent(agent):
        return True
    manifest = _manifest_entries(trial_dir)
    if not manifest:
        return True
    for entry in manifest:
        if str(entry.get("source", "")).endswith(_TRAJECTORY_SUFFIX) \
                and entry.get("status") == "failed":
            return False
    return trajectory_exists


def _scalar(rewards: dict) -> float | None:
    """The single headline reward: the `reward` key, else the mean of the dimensions."""
    if not rewards:
        return None
    if "reward" in rewards:
        return float(rewards["reward"])
    values = [float(v) for v in rewards.values()]
    return sum(values) / len(values)


def _read_rewards(trial_dir: Path) -> dict:
    """The reward dict the verifier wrote: reward.json (a dict), else reward.txt (one float)."""
    as_json = _load_json(trial_dir / "verifier" / "reward.json")
    if as_json:
        return {k: float(v) for k, v in as_json.items()}
    txt = trial_dir / "verifier" / "reward.txt"
    try:
        return {"reward": float(txt.read_text().strip())}
    except (OSError, ValueError):
        return {}


@dataclass
class Trial:
    task_name: str
    agent: str | None
    model: str | None
    reward: float | None
    rewards: dict
    trajectory_path: Path | None
    exception: str | None
    error: str | None = None  # the exception message, when the trial raised

    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward >= 1.0

    @classmethod
    def read(cls, trial_dir: Path) -> Trial:
        result = _load_json(trial_dir / "result.json")
        agent_info = result.get("agent_info") or {}
        model_info = agent_info.get("model_info") or {}
        rewards = _read_rewards(trial_dir)
        if not rewards:
            verifier = result.get("verifier_result") or {}
            rewards = {k: float(v) for k, v in (verifier.get("rewards") or {}).items()}
        exc = result.get("exception_info") or {}
        traj = trial_dir / "agent" / "trajectory.json"
        reward, error = _scalar(rewards), exc.get("exception_message")
        if not _agent_ran(trial_dir, traj.is_file(), agent_info.get("name")):
            # Harbor's manifest says the trajectory was never collected: the agent did not run. An
            # empty reward here is a false 0, not a real score — record it as an error instead.
            reward, rewards, error = None, {}, error or NO_TRAJECTORY
        return cls(
            task_name=result.get("task_name") or trial_dir.name,
            agent=agent_info.get("name"),
            model=model_info.get("name") or (result.get("config", {}).get("agent") or {})
                                              .get("model_name"),
            reward=reward,
            rewards=rewards,
            trajectory_path=traj if traj.is_file() else None,
            exception=exc.get("exception_type"),
            error=error,
        )


@dataclass
class Job:
    dir: Path
    config: dict = field(default_factory=dict)
    trials: list[Trial] = field(default_factory=list)

    @classmethod
    def read(cls, job_dir: str | Path) -> Job:
        job_dir = Path(job_dir)
        trials = [Trial.read(d) for d in sorted(job_dir.iterdir())
                  if d.is_dir() and (d / "result.json").is_file()]
        return cls(dir=job_dir, config=_load_json(job_dir / "config.json"), trials=trials)


def pass_rates(job: Job) -> dict[str, float]:
    """Share of each task's trials that passed (reward >= 1.0), by task name, sorted."""
    tasks: dict[str, list[Trial]] = {}
    for trial in job.trials:
        tasks.setdefault(trial.task_name, []).append(trial)
    return {name: sum(t.passed for t in ts) / len(ts) for name, ts in sorted(tasks.items())}


def ran_tasks(job: Job) -> set[str]:
    """Task names with at least one trial that produced a reward — where the agent actually ran."""
    return {t.task_name for t in job.trials if t.reward is not None}


def _task_error(trials: list[Trial]) -> str:
    """The error to print for a task whose every trial errored: the no-trajectory sentence when
    that is why, else the first exception/message, else a generic one."""
    if any(t.error == NO_TRAJECTORY for t in trials):
        return NO_TRAJECTORY
    return next((t.error or t.exception for t in trials if t.error or t.exception), "errored")


def task_outcomes(job: Job) -> dict[str, float | str]:
    """Per task, the pass rate — or an error string when every trial of it errored (no reward), so
    a run the agent never completed reads as an error, never a false 0%. Sorted by task name."""
    tasks: dict[str, list[Trial]] = {}
    for trial in job.trials:
        tasks.setdefault(trial.task_name, []).append(trial)
    out: dict[str, float | str] = {}
    for name, ts in sorted(tasks.items()):
        out[name] = _task_error(ts) if all(t.reward is None for t in ts) \
            else sum(t.passed for t in ts) / len(ts)
    return out


def mean_rewards(job: Job) -> dict[str, float]:
    """Mean scalar reward per task (a partial-credit view), by task name, sorted."""
    tasks: dict[str, list[float]] = {}
    for trial in job.trials:
        tasks.setdefault(trial.task_name, []).append(
            trial.reward if trial.reward is not None else 0.0)
    return {name: sum(rs) / len(rs) for name, rs in sorted(tasks.items())}


def _passed_tasks(job: Job) -> set[str]:
    """Tasks a job passed outright — every trial of the task scored 1.0."""
    return {name for name, rate in pass_rates(job).items() if rate >= 1.0}


def compare(job_a: Job, job_b: Job) -> dict[str, list[str]]:
    """Per task (union of both jobs): passed in both, only A, only B, or neither. Sorted lists."""
    a, b = _passed_tasks(job_a), _passed_tasks(job_b)
    tasks = set(pass_rates(job_a)) | set(pass_rates(job_b))
    out: dict[str, list[str]] = {"both": [], "only_a": [], "only_b": [], "neither": []}
    for name in sorted(tasks):
        in_a, in_b = name in a, name in b
        key = "both" if in_a and in_b else "only_a" if in_a else "only_b" if in_b else "neither"
        out[key].append(name)
    return out
