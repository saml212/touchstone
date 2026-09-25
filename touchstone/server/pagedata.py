"""Low-level reads of the dataset directory and the Harbor job dirs, shared by the page builders.

`server/pages.py` assembles each UI page from these: safe JSON/text file reads, the task and job
directory listings, and one job's (agent, model). Pure file-reads, no cache; a missing or malformed
file reads as empty, never an exception.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..harbor import jobs as harbor_jobs

# Harbor's gate agents: their per-task oracle/nop probes are not a model under test.
_GATE = ("oracle", "nop")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _short(agent: str | None) -> str:
    return (agent or "agent").split(":")[-1]


def _is_gate(agent: str | None) -> bool:
    return bool(agent) and any(g in agent.lower() for g in _GATE)


def _task_dirs(dataset_dir: Path) -> list[Path]:
    tasks = dataset_dir / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(d for d in tasks.glob("*") if (d / "task.toml").is_file())


def _all_job_dirs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(d for d in jobs_dir.iterdir()
                  if d.is_dir() and (d / "config.json").is_file())


def _trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(d for d in job_dir.iterdir()
                  if d.is_dir() and (d / "result.json").is_file())


def _job_meta(job_dir: Path) -> dict:
    """The (agent, model) a job ran, from config.json, falling back to its first trial."""
    agents = _load_json(job_dir / "config.json").get("agents") or []
    if agents:
        return {"agent": agents[0].get("name"), "model": agents[0].get("model_name")}
    trials = _trial_dirs(job_dir)
    first = harbor_jobs.Trial.read(trials[0]) if trials else None
    return {"agent": first.agent if first else None, "model": first.model if first else None}
