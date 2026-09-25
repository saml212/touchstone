"""Read the Harbor job directories and decide which trial the room walks through next.

A trial is addressed as ``<job>/<trial_dir>`` so a regrade knows its job. `scan` reads every job
under the jobs directory into `TrialRef`s and orders them the way the design asks: verifier unsure
(reward strictly between 0 and 1) first, then models disagree (a task that passed in one job and
failed in another), then never reviewed, then already reviewed. `read` returns one trial in plain
words — the instruction, the trajectory as a readable transcript, and each criterion's description
and score — for the agent to present. Needs-review tasks (gate failures) are read separately from
``needs-review/<task>/gate.json``; they have no trials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..harbor import jobs

# Priority of the four buckets the design walks in order; lower sorts first.
_ORDER = {"unsure": 0, "disagree": 1, "unreviewed": 2, "reviewed": 3}


@dataclass
class TrialRef:
    task: str
    trial: str  # "<job>/<trial_dir>"
    job: str
    reward: float | None
    rewards: dict
    category: str
    reviewed: bool


def _job_dirs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(d for d in jobs_dir.iterdir()
                  if d.is_dir() and (d / "config.json").is_file())


def _trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(d for d in job_dir.iterdir()
                  if d.is_dir() and (d / "result.json").is_file())


def _disagreeing_tasks(job_dirs: list[Path]) -> set[str]:
    """Tasks whose pass/fail outcome is not the same across every job that ran them."""
    seen: dict[str, set[bool]] = {}
    for job_dir in job_dirs:
        for trial_dir in _trial_dirs(job_dir):
            trial = jobs.Trial.read(trial_dir)
            seen.setdefault(trial.task_name, set()).add(trial.passed)
    return {task for task, outcomes in seen.items() if len(outcomes) > 1}


def _unsure(reward: float | None) -> bool:
    return reward is not None and 0.0 < reward < 1.0


def _category(reward: float | None, task: str, disagree: set[str], reviewed: bool) -> str:
    if _unsure(reward):
        return "unsure"
    if task in disagree:
        return "disagree"
    return "reviewed" if reviewed else "unreviewed"


def _reviewed_keys(conn) -> set[tuple[str, str]]:
    from .. import store

    return {(r.task, r.trial) for r in store.list_reviews(conn)}


def scan(jobs_dir: Path, conn=None) -> list[TrialRef]:
    """Every trial across every job, ordered unsure -> disagree -> unreviewed -> reviewed."""
    job_dirs = _job_dirs(jobs_dir)
    disagree = _disagreeing_tasks(job_dirs)
    reviewed_keys = _reviewed_keys(conn) if conn is not None else set()
    refs: list[TrialRef] = []
    for job_dir in job_dirs:
        for trial_dir in _trial_dirs(job_dir):
            trial = jobs.Trial.read(trial_dir)
            trial_id = f"{job_dir.name}/{trial_dir.name}"
            reviewed = (trial.task_name, trial_id) in reviewed_keys
            category = _category(trial.reward, trial.task_name, disagree, reviewed)
            refs.append(TrialRef(task=trial.task_name, trial=trial_id, job=job_dir.name,
                                 reward=trial.reward, rewards=trial.rewards,
                                 category=category, reviewed=reviewed))
    refs.sort(key=lambda r: (_ORDER[r.category], r.task, r.trial))
    return refs


def filter_refs(refs: list[TrialRef], which: str | None) -> list[TrialRef]:
    """Keep the refs matching a filter chip; None/"all" keeps every ref."""
    if not which or which == "all":
        return refs
    return [r for r in refs if r.category == which]


def counts(refs: list[TrialRef]) -> dict[str, int]:
    out = {name: 0 for name in _ORDER}
    for ref in refs:
        out[ref.category] += 1
    return out


# ---- reading one trial in plain words --------------------------------------


def _trial_dir(jobs_dir: Path, trial_id: str) -> Path | None:
    job, _, trial = trial_id.partition("/")
    path = jobs_dir / job / trial
    return path if (path / "result.json").is_file() else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _instruction(dataset_dir: Path, task: str) -> str:
    """The instruction with the canary comment stripped (the room speaks in plain words)."""
    text = _read_text(dataset_dir / "tasks" / task / "instruction.md")
    if text.startswith("<!--"):
        text = text.split("-->", 1)[-1].strip()
    return text


def _agent_line(step: dict, message: str) -> str | None:
    parts = [f"Agent: {message}"] if message else []
    for call in step.get("tool_calls") or []:
        args = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
        parts.append(f"Agent used {call.get('function_name', 'tool')}({args})")
    for result in (step.get("observation") or {}).get("results", []):
        parts.append(f"Result: {_clip(str(result.get('content', '')))}")
    return "\n".join(parts) or None


def _step_line(step: dict) -> str | None:
    """One trajectory step as a sentence: who spoke / what the agent did / what came back."""
    source = step.get("source")
    message = _message_text(step.get("message"))
    if source == "system":
        return None
    if source == "user":
        return f"User: {message}" if message else None
    return _agent_line(step, message)


def _message_text(message) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        return " ".join(p.get("text", "") for p in message if p.get("type") == "text").strip()
    return ""


def _clip(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _trajectory(trial_dir: Path) -> list[str]:
    try:
        traj = json.loads((trial_dir / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [line for step in traj.get("steps", []) if (line := _step_line(step))]


def _flatten_criteria(details: dict) -> list[dict]:
    """reward-details.json -> a flat [{dimension, description, score}] list, in file order."""
    out: list[dict] = []
    for dimension, block in details.items():
        if not isinstance(block, dict):
            continue
        for component in block.get("components", [block]):
            detail = component.get("detail", component)
            for crit in detail.get("criteria", []):
                out.append({"dimension": dimension,
                            "description": crit.get("description", crit.get("name", "")),
                            "score": crit.get("value")})
    return out


def read(dataset_dir: Path, jobs_dir: Path, task: str, trial_id: str) -> dict | None:
    """Instruction, persona, trajectory transcript, and criteria+scores for one trial."""
    trial_dir = _trial_dir(jobs_dir, trial_id)
    if trial_dir is None:
        return None
    trial = jobs.Trial.read(trial_dir)
    details = _load_json(trial_dir / "verifier" / "reward-details.json")
    return {
        "task": task, "trial": trial_id, "reward": trial.reward, "rewards": trial.rewards,
        "instruction": _instruction(dataset_dir, task),
        "persona": _read_text(dataset_dir / "tasks" / task / "persona.md"),
        "trajectory": _trajectory(trial_dir),
        "criteria": _flatten_criteria(details),
    }


# ---- needs-review tasks (gate failures) ------------------------------------


def needs_review(dataset_dir: Path) -> list[dict]:
    """Gate failures under needs-review/<task>/gate.json, each with its failed side + reason."""
    root = dataset_dir / "needs-review"
    if not root.is_dir():
        return []
    out = []
    for task_dir in sorted(root.iterdir()):
        gate = task_dir / "gate.json"
        if not gate.is_file():
            continue
        try:
            info = json.loads(gate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {}
        out.append({"task": task_dir.name, **info})
    return out
