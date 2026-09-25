"""Plain facts about a survey dataset, read from its files at request time.

Job labels, baseline pass counts, a task's job metadata, the tasks that share a job, and the
criteria a review room may edit — shared by the review agent (`review.agent`) and the UI pages
(`server.pages`). Pure file-reads; a missing or malformed file reads as empty, never an exception.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from . import changes


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _touchstone_meta(task_dir: Path) -> dict:
    return _read_toml(task_dir / "task.toml").get("metadata", {}).get("touchstone", {})


def _job_labels(dataset_dir: Path) -> list[str]:
    groups = dataset_dir / "groups.json"
    if groups.is_file():
        try:
            data = json.loads(groups.read_text(encoding="utf-8"))
            return [g["label"] for g in data.get("groups", []) if g.get("label")]
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return _labels_from_tasks(dataset_dir)


def _labels_from_tasks(dataset_dir: Path) -> list[str]:
    labels: list[str] = []
    for task_dir in sorted((dataset_dir / "tasks").glob("*")):
        job = _touchstone_meta(task_dir).get("job")
        if job and job not in labels:
            labels.append(job)
    return labels


def _baseline_counts(dataset_dir: Path) -> dict | None:
    baseline = dataset_dir / "baseline.json"
    if not baseline.is_file():
        return None
    try:
        data = json.loads(baseline.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    rates = data.get("pass_rates") or {}
    return {"tasks": len(rates), "passed": len(data.get("passed", []))}


def _task_count(dataset_dir: Path) -> int:
    tasks = dataset_dir / "tasks"
    return sum(1 for d in tasks.glob("*") if (d / "task.toml").is_file()) if tasks.is_dir() else 0


def _join(items: list[str]) -> str:
    items = [i.lower() for i in items]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _shared_tasks(dataset_dir: Path, task: str) -> list[str]:
    """Every task with the same job-to-be-done label as `task` (for an 'always' change)."""
    job = _touchstone_meta(dataset_dir / "tasks" / task).get("job")
    if not job:
        return [task]
    out = [d.name for d in sorted((dataset_dir / "tasks").glob("*"))
           if _touchstone_meta(d).get("job") == job]
    return out or [task]


def _editable(task_dir: Path) -> list[dict]:
    """The criteria the room can change, keyed by file + 1-based index (or reward dimension)."""
    out: list[dict] = []
    tests = task_dir / "tests"
    for py in sorted(tests.rglob("*.py")):
        try:
            calls = changes.parse_criteria(py)
        except changes.ChangeError:
            continue
        rel = py.relative_to(task_dir).as_posix()
        out += [{"handle": f"{rel}:{i}", "file": rel, "index": i, "criterion": src}
                for i, src in enumerate(calls, 1)]
    reward = tests / "reward.toml"
    if reward.is_file():
        weights = _read_toml(reward).get("reward", [{}])[0].get("weights", {})
        out += [{"handle": f"tests/reward.toml:{dim}", "file": "tests/reward.toml",
                 "dimension": dim, "weight": w} for dim, w in weights.items()]
    return out
