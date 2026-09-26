"""Read the Harbor job directories and decide which trial the room walks through next.

A trial is addressed as ``<job>/<trial_dir>`` so a regrade knows its job. `scan` orders the
review-worthy jobs the way the design asks: verifier unsure (reward strictly 0..1), then models
disagree, then never reviewed, then reviewed. Oracle/nop gate jobs are never review material, and
only the latest job per (agent, model) is surfaced (older runs stay addressable by job id via
`read`). `read` returns one trial in plain words; needs-review gate failures are read separately
from ``needs-review/<task>/gate.json`` and have no trials.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..harbor import jobs
from ..survey import descriptions

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
    label: str = ""  # "<agent>/<model> · <job>" — which run this trial came from


# Harbor's gate agents; their jobs prove a task is solvable/non-trivial, never review material.
_GATE_AGENTS = ("oracle", "nop")


def _job_dirs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(d for d in jobs_dir.iterdir()
                  if d.is_dir() and (d / "config.json").is_file())


def _agent_model(job_dir: Path) -> tuple[str | None, str | None]:
    """The (agent, model) a job ran, from its first trial's result (fallback: config.json)."""
    for trial_dir in _trial_dirs(job_dir):
        result = _load_json(trial_dir / "result.json")
        info = result.get("agent_info") or {}
        cfg = (result.get("config") or {}).get("agent") or {}
        model = (info.get("model_info") or {}).get("name") or cfg.get("model_name")
        return info.get("name") or cfg.get("name"), model
    agents = _load_json(job_dir / "config.json").get("agents") or []
    return (agents[0].get("name"), agents[0].get("model_name")) if agents else (None, None)


def _is_gate(agent: str | None) -> bool:
    return bool(agent) and any(g in agent.lower() for g in _GATE_AGENTS)


def _label(agent: str | None, model: str | None, job: str) -> str:
    return f"{(agent or 'agent').split(':')[-1]}/{model or '?'} · {job}"


def _review_jobs(jobs_dir: Path) -> list[tuple[Path, str]]:
    """Review-worthy jobs: gate jobs dropped, only the latest job per (agent, model) kept (older
    jobs stay addressable by explicit job id via `read`). Returns (job_dir, label)."""
    latest: dict[tuple, Path] = {}
    for job_dir in _job_dirs(jobs_dir):  # ascending by timestamped name -> last write wins
        agent, model = _agent_model(job_dir)
        if _is_gate(agent) or not _has_rewards(job_dir):
            continue
        latest[(agent, model)] = job_dir
    return [(jd, _label(*_agent_model(jd), jd.name)) for jd in latest.values()]


def _has_rewards(job_dir: Path) -> bool:
    """A run that produced no reward at all (every trial raised) is broken, not review material."""
    return any(t.reward is not None for t in jobs.Job.read(job_dir).trials)


def latest_non_gate_job(jobs_dir: Path) -> Path | None:
    """The newest non-gate job dir, or None — the run the opening reports on (errors and all)."""
    non_gate = [jd for jd in _job_dirs(jobs_dir) if not _is_gate(_agent_model(jd)[0])]
    return non_gate[-1] if non_gate else None  # _job_dirs is ascending -> the last is newest


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


def _task_exists(dataset_dir: Path | None, task: str) -> bool:
    """A trial is stale when its task was regenerated/renamed away; never present those."""
    return dataset_dir is None or (dataset_dir / "tasks" / task / "task.toml").is_file()


def scan(jobs_dir: Path, conn=None, dataset_dir: Path | None = None) -> list[TrialRef]:
    """Review trials across the latest non-gate job per (agent, model), ordered unsure -> disagree
    -> unreviewed -> reviewed. Oracle/nop gate jobs, superseded runs, and trials whose task dir no
    longer exists (stale) are excluded."""
    review_jobs = _review_jobs(jobs_dir)
    disagree = _disagreeing_tasks([jd for jd, _ in review_jobs])
    reviewed_keys = _reviewed_keys(conn) if conn is not None else set()
    refs: list[TrialRef] = []
    for job_dir, label in review_jobs:
        for trial_dir in _trial_dirs(job_dir):
            trial = jobs.Trial.read(trial_dir)
            if not _task_exists(dataset_dir, trial.task_name):
                continue
            trial_id = f"{job_dir.name}/{trial_dir.name}"
            reviewed = (trial.task_name, trial_id) in reviewed_keys
            category = _category(trial.reward, trial.task_name, disagree, reviewed)
            refs.append(TrialRef(task=trial.task_name, trial=trial_id, job=job_dir.name,
                                 reward=trial.reward, rewards=trial.rewards,
                                 category=category, reviewed=reviewed, label=label))
    refs.sort(key=lambda r: (_ORDER[r.category], r.task, r.trial))
    return refs


def stale_count(jobs_dir: Path, dataset_dir: Path) -> int:
    """Trials in the review jobs whose task dir no longer exists (shown in the chips tooltip)."""
    n = 0
    for job_dir, _ in _review_jobs(jobs_dir):
        for trial_dir in _trial_dirs(job_dir):
            if not _task_exists(dataset_dir, jobs.Trial.read(trial_dir).task_name):
                n += 1
    return n


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


_CANARY = re.compile(r"<!--.*?(?:BENCHMARK DATA|harbor-canary).*?-->", re.S | re.I)


def _strip_canary(text: str) -> str:
    """Remove Harbor's canary comment from anything shown to a person; kept in instruction.md."""
    return _CANARY.sub("", text).strip()


def _instruction(dataset_dir: Path, task: str) -> str:
    """The instruction with the canary comment stripped (the room speaks in plain words)."""
    return _strip_canary(_read_text(dataset_dir / "tasks" / task / "instruction.md"))


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
        message = _strip_canary(message)
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


def _crit_view(crit: dict, dimension: str, rel: str, index: int, descs: dict) -> dict:
    """One criterion as a person reads it: the plain description, with the raw call on hover."""
    raw = crit.get("description", crit.get("name", ""))
    plain = descs.get(descriptions.key(rel, index)) or raw
    return {"dimension": dimension, "description": plain, "raw": raw, "score": crit.get("value")}


def _component_file(dimension: str, component: dict, crit: dict) -> str:
    """The criteria file a reward-details criterion came from, to look up its description."""
    name = component.get("name") or crit.get("name", "")
    return f"tests/{dimension}/{name}.py"


def _flatten_criteria(details: dict, descs: dict) -> list[dict]:
    """reward-details.json -> a flat [{dimension, description, raw, score}] list, in file order,
    each with the product-readable description from descriptions.toml when one exists."""
    out: list[dict] = []
    for dimension, block in details.items():
        if not isinstance(block, dict):
            continue
        for component in block.get("components", [block]):
            detail = component.get("detail", component)
            crits = detail.get("criteria", [])
            for i, crit in enumerate(crits, 1):
                rel = _component_file(dimension, component, crit)
                out.append(_crit_view(crit, dimension, rel, i, descs))
    return out


def read(dataset_dir: Path, jobs_dir: Path, task: str, trial_id: str) -> dict | None:
    """Instruction, persona, trajectory transcript, and criteria+scores for one trial."""
    trial_dir = _trial_dir(jobs_dir, trial_id)
    if trial_dir is None:
        return None
    trial = jobs.Trial.read(trial_dir)
    details = _load_json(trial_dir / "verifier" / "reward-details.json")
    descs = descriptions.load(dataset_dir / "tasks" / task / "tests")
    return {
        "task": task, "trial": trial_id, "reward": trial.reward, "rewards": trial.rewards,
        "instruction": _instruction(dataset_dir, task),
        "persona": _read_text(dataset_dir / "tasks" / task / "persona.md"),
        "trajectory": _trajectory(trial_dir),
        "criteria": _flatten_criteria(details, descs),
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
