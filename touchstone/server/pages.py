"""Assemble the JSON each of the five UI pages shows, reading the dataset and the Harbor job dirs
at request time. Pure file-reads — no cache, so nothing goes stale; the only database touch is the
trust figure, which reads the `reviews` table.

Nothing here names a customer or a product: every label comes from the dataset (job-to-be-done from
`[metadata.touchstone].job`, group labels, `descriptions.toml`). The dataset-fact helpers are shared
with the review agent (`review.agent`, `review.trials`); the survey descriptions loader gives the
plain-English criterion sentences.
"""

from __future__ import annotations

from pathlib import Path

from ..harbor import jobs as harbor_jobs
from ..review import trials as trials_mod
from ..review.agent import (
    _baseline_counts,
    _job_labels,
    _join,
    _read_toml,
    _task_count,
    _touchstone_meta,
)
from ..survey import descriptions
from .pagedata import (
    _all_job_dirs,
    _is_gate,
    _job_meta,
    _load_json,
    _read_text,
    _short,
    _task_dirs,
    _trial_dirs,
)

# ---- Overview --------------------------------------------------------------


def _conversations(dataset_dir: Path) -> int:
    """How many recorded conversations the survey grouped into jobs-to-be-done (groups.json)."""
    data = _load_json(dataset_dir / "groups.json")
    return sum(len(g.get("episodes", [])) for g in data.get("groups", []))


def _sentence(dataset_dir: Path) -> str:
    """The first-five-minutes sentence: built N tasks from M conversations, passes K, F failures."""
    built = f"Built {_task_count(dataset_dir)} tasks"
    convos = _conversations(dataset_dir)
    if convos:
        built += f" from {convos} conversations"
    counts = _baseline_counts(dataset_dir)
    if not counts:
        return built + "."
    failures = counts["tasks"] - counts["passed"]
    tail = f" Your current setup passes {counts['passed']}."
    if failures:
        return f"{built}.{tail} {failures} failures — walk through them?"
    return f"{built}.{tail} Everything passes — spot-check a few?"


def _map_paragraph(dataset_dir: Path) -> str:
    does = _join(_job_labels(dataset_dir))
    return f"This agent handles {does}." if does else ""


def _service_ok(info: dict) -> bool:
    score, threshold = info.get("score"), info.get("threshold")
    return score is not None and threshold is not None and score >= threshold


def _services(dataset_dir: Path) -> list[dict]:
    """Each simulated service with its fidelity score and status (from fidelity.json)."""
    out = []
    for name, info in sorted(_load_json(dataset_dir / "fidelity.json").items()):
        if not isinstance(info, dict):
            continue
        out.append({"service": name, "fidelity": info.get("score"),
                    "reproduced": info.get("reproduced"), "calls": info.get("calls"),
                    "status": "ok" if _service_ok(info) else "flagged"})
    return out


def _tasks_by_job(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task_dir in _task_dirs(dataset_dir):
        job = _touchstone_meta(task_dir).get("job") or "—"
        counts[job] = counts.get(job, 0) + 1
    return counts


def _job_summary(job_dir: Path) -> dict:
    job = harbor_jobs.Job.read(job_dir)
    rates = harbor_jobs.pass_rates(job)
    meta = _job_meta(job_dir)
    return {"job": job_dir.name, "agent": _short(meta["agent"]), "model": meta["model"],
            "tasks": len(rates), "passed": sum(1 for r in rates.values() if r >= 1.0),
            "gate": _is_gate(meta["agent"])}


def _latest_jobs(jobs_dir: Path) -> list[dict]:
    """The latest non-gate job per (agent, model), newest first — the models under test."""
    latest: dict[tuple, dict] = {}
    for job_dir in _all_job_dirs(jobs_dir):  # ascending timestamped name -> last write wins
        summary = _job_summary(job_dir)
        if summary["gate"]:
            continue
        latest[(summary["agent"], summary["model"])] = summary
    return sorted(latest.values(), key=lambda s: s["job"], reverse=True)


def _trust(conn) -> dict | None:
    """Share of reviewed trials where the human agreed with the verifier, or None if none yet."""
    if conn is None:
        return None
    from .. import store

    seen: dict[tuple, str] = {}
    for review in store.list_reviews(conn):
        seen[(review.task, review.trial)] = review.verdict
    if not seen:
        return None
    agreed = sum(1 for v in seen.values() if v == "agree")
    return {"agreed": agreed, "reviewed": len(seen), "score": agreed / len(seen)}


def overview(dataset_dir: Path, jobs_dir: Path, conn=None) -> dict:
    return {
        "sentence": _sentence(dataset_dir),
        "map": _map_paragraph(dataset_dir),
        "services": _services(dataset_dir),
        "jobs_to_be_done": _tasks_by_job(dataset_dir),
        "latest_jobs": _latest_jobs(jobs_dir),
        "trust": _trust(conn),
        "has_dataset": (dataset_dir / "tasks").is_dir(),
    }


# ---- Tasks -----------------------------------------------------------------


def _gate(task_dir: Path) -> dict:
    meta = _touchstone_meta(task_dir)
    return {"oracle": meta.get("oracle"), "nop": meta.get("nop")}


def _criteria_labels(task_dir: Path) -> list[str]:
    descs = descriptions.load(task_dir / "tests")
    return [descs[k] for k in sorted(descs)]


def _latest_nongate_jobs(jobs_dir: Path) -> dict[str, Path]:
    """model-label -> its latest non-gate job dir (latest wins by ascending name)."""
    latest: dict[str, Path] = {}
    for job_dir in _all_job_dirs(jobs_dir):
        meta = _job_meta(job_dir)
        if _is_gate(meta["agent"]):
            continue
        latest[meta["model"] or _short(meta["agent"])] = job_dir
    return latest


def _task_rewards(jobs_dir: Path) -> dict[str, dict[str, float]]:
    """task name -> {model-label: mean reward} across the latest non-gate job per model."""
    out: dict[str, dict[str, float]] = {}
    for label, job_dir in _latest_nongate_jobs(jobs_dir).items():
        for name, reward in harbor_jobs.mean_rewards(harbor_jobs.Job.read(job_dir)).items():
            out.setdefault(name, {})[label] = reward
    return out


def tasks(dataset_dir: Path, jobs_dir: Path) -> list[dict]:
    rewards = _task_rewards(jobs_dir)
    out = []
    for task_dir in _task_dirs(dataset_dir):
        out.append({
            "task": task_dir.name,
            "job": _touchstone_meta(task_dir).get("job"),
            "criteria": _criteria_labels(task_dir),
            "gate": _gate(task_dir),
            "rewards": rewards.get(task_dir.name, {}),
        })
    return out


# ---- Task detail -----------------------------------------------------------


def _weights(task_dir: Path) -> dict:
    """The reward dimension weights from tests/reward.toml ([[reward]] -> [reward.weights])."""
    entries = _read_toml(task_dir / "tests" / "reward.toml").get("reward")
    if isinstance(entries, list) and entries and isinstance(entries[0].get("weights"), dict):
        return entries[0]["weights"]
    return {}


def _dimension_of(key: str) -> str:
    """The dimension a descriptions.toml key belongs to: tests/<dimension>/<file>:index."""
    parts = key.split("/")
    return parts[1] if len(parts) >= 3 else "other"


def _criteria_by_dimension(task_dir: Path) -> dict[str, list[str]]:
    descs = descriptions.load(task_dir / "tests")
    out: dict[str, list[str]] = {}
    for key in sorted(descs):
        out.setdefault(_dimension_of(key), []).append(descs[key])
    return out


def _task_trials(jobs_dir: Path, task: str) -> list[dict]:
    out = []
    for job_dir in _all_job_dirs(jobs_dir):
        for trial_dir in _trial_dirs(job_dir):
            trial = harbor_jobs.Trial.read(trial_dir)
            if trial.task_name == task:
                out.append({"job": job_dir.name, "model": trial.model, "reward": trial.reward,
                            "trial": f"{job_dir.name}/{trial_dir.name}"})
    return out


def task_detail(dataset_dir: Path, jobs_dir: Path, name: str) -> dict | None:
    task_dir = dataset_dir / "tasks" / name
    if not (task_dir / "task.toml").is_file():
        return None
    return {
        "task": name,
        "job": _touchstone_meta(task_dir).get("job"),
        "instruction": trials_mod._instruction(dataset_dir, name),
        "persona": _read_text(task_dir / "persona.md"),
        "weights": _weights(task_dir),
        "criteria": _criteria_by_dimension(task_dir),
        "gate": _gate(task_dir),
        "trials": _task_trials(jobs_dir, name),
    }


# ---- Trials (job picker + one trial) ---------------------------------------


def _mark_superseded(summaries: list[dict]) -> None:
    """Flag each row `kept` (a model run that is the latest for its agent+model — what the review
    room walks) or `superseded` (an older non-gate run); gate rows are neither."""
    latest: dict[tuple, str] = {}
    for s in summaries:  # ascending timestamped name -> last wins
        if not s["gate"]:
            latest[(s["agent"], s["model"])] = s["job"]
    for s in summaries:
        s["kept"] = not s["gate"] and latest.get((s["agent"], s["model"])) == s["job"]
        s["superseded"] = not s["gate"] and not s["kept"]


def jobs(jobs_dir: Path) -> list[dict]:
    """Every job dir as a picker row, newest first, each marked kept / superseded / gate so the UI
    can default to the runs the review room keeps and reveal the rest behind a toggle."""
    summaries = [_job_summary(d) for d in _all_job_dirs(jobs_dir)]
    _mark_superseded(summaries)
    summaries.sort(key=lambda r: r["job"], reverse=True)
    return summaries


def job_rewards(jobs_dir: Path, job: str) -> dict | None:
    job_dir = jobs_dir / job
    if not (job_dir / "config.json").is_file():
        return None
    rows = []
    for trial_dir in _trial_dirs(job_dir):
        trial = harbor_jobs.Trial.read(trial_dir)
        rows.append({"task": trial.task_name, "reward": trial.reward, "model": trial.model,
                     "trial": f"{job}/{trial_dir.name}"})
    rows.sort(key=lambda r: r["task"])
    meta = _job_meta(job_dir)
    return {"job": job, "agent": _short(meta["agent"]), "model": meta["model"], "rewards": rows}


def trial(dataset_dir: Path, jobs_dir: Path, task: str, trial_id: str) -> dict | None:
    return trials_mod.read(dataset_dir, jobs_dir, task, trial_id)


# ---- Train -----------------------------------------------------------------


def _train_command(dataset_dir: Path, manifest: dict) -> str:
    inputs = manifest.get("inputs", {})
    teacher = ",".join(t["dir"] for t in inputs.get("teacher", []) if t.get("dir"))
    student = ",".join(s["dir"] for s in inputs.get("student", []) if s.get("dir"))
    parts = ["touchstone train", f"--jobs-dir {dataset_dir.name}/jobs"]
    if teacher:
        parts.append(f"--teacher {teacher}")
    if student:
        parts.append(f"--student {student}")
    return " ".join(parts)


def train(dataset_dir: Path) -> dict:
    manifest = _load_json(dataset_dir / "train" / "manifest.json")
    if not manifest:
        return {"present": False}
    counts = manifest.get("counts", {})
    return {
        "present": True,
        "counts": {k: counts.get(k, 0) for k in ("distill", "rl", "hold_out", "stuck")},
        "trajectories": counts.get("distill_trajectories", 0),
        "threshold": manifest.get("threshold"),
        "created_at": manifest.get("created_at"),
        "inputs": manifest.get("inputs", {}),
        "command": _train_command(dataset_dir, manifest),
    }
