"""Write the three training-data files from teacher/student jobs, atomically.

    <out>/distill.jsonl    one JSON trajectory record per line (the teacher's passing trials)
    <out>/rl_tasks.toml    the student's band tasks ([[task]] name/path/pass_rate/attempts)
    <out>/manifest.json    inputs (job dirs + config agent/model), counts per route, thresholds,
                           touchstone version, created_at

`write_datasets` is what both the CLI and the Harbor plugin call, so `harbor run --plugin` and
`touchstone train` produce identical files.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomli_w

from ..harbor.jobs import Job
from ..survey.writes import atomic_write
from .datasets import distill, distill_excluded, rl_note, rl_tasks, route


def _all_jobs(teacher_jobs: list[Job], student_jobs: list[Job]) -> list[Job]:
    """Teacher then student, de-duplicated by dir — the plugin passes one job as both."""
    seen: set[str] = set()
    out = []
    for job in [*teacher_jobs, *student_jobs]:
        key = str(job.dir)
        if key not in seen:
            seen.add(key)
            out.append(job)
    return out


def _touchstone_version() -> str:
    try:
        return version("touchstone-bench")
    except PackageNotFoundError:
        return "0.0.0"


def _agent_model(job: Job) -> dict:
    """The agent name and model a job ran, from its config.json `agents[0]`."""
    agents = job.config.get("agents") or [{}]
    first = agents[0] if agents else {}
    return {"dir": job.dir.name, "agent": first.get("name"), "model": first.get("model_name")}


def _write_distill(out: Path, records: list[dict]) -> None:
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    atomic_write(out / "distill.jsonl", lines)


def _write_rl(out: Path, records: list[dict]) -> None:
    tasks = [{"name": r["task"], "path": r["path"], "pass_rate": r["pass_rate"],
              "attempts": r["attempts"]} for r in records]
    atomic_write(out / "rl_tasks.toml", tomli_w.dumps({"task": tasks}))


def _manifest(teacher_jobs: list[Job], student_jobs: list[Job], routing: dict[str, str],
              distilled: list[dict], excluded: int, threshold: float, note: str | None) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "touchstone_version": _touchstone_version(),
        "threshold": threshold,
        "rl_note": note,
        "inputs": {
            "teacher": [_agent_model(j) for j in teacher_jobs],
            "student": [_agent_model(j) for j in student_jobs],
        },
        "counts": {
            **{route_name: 0 for route_name in ("distill", "rl", "hold_out", "stuck")},
            **Counter(routing.values()),
            "distill_trajectories": len(distilled),
            "distill_missing_trajectory": excluded,
        },
    }


def _fmt(rate: float) -> str:
    return f"{rate:.2f}"


@dataclass
class Written:
    """What a write produced — the paths' manifest plus the records, so callers can summarize."""

    out: Path
    manifest: dict
    distill: list[dict]
    rl: list[dict]
    excluded: int = 0
    rl_note: str | None = None

    def sentence(self) -> str:
        """The one-line result the CLI prints."""
        tasks = len({r["task"] for r in self.distill})
        counts = self.manifest["counts"]
        if self.rl:
            rates = [r["pass_rate"] for r in self.rl]
            lo, hi = _fmt(min(rates)), _fmt(max(rates))
            band = f" (pass {lo})" if lo == hi else f" (pass {lo}–{hi})"
        else:
            band = f" ({self.rl_note})" if self.rl_note else ""
        excl = f" (excluded {self.excluded}: no trajectory artifact)" if self.excluded else ""
        return (f"distill: {len(self.distill)} trajectories from {tasks} tasks{excl} · "
                f"rl: {len(self.rl)} tasks{band} · hold-out: {counts['hold_out']} · "
                f"stuck: {counts['stuck']} → {self.out}/")


def write_datasets(out: str | Path, teacher_jobs: list[Job], student_jobs: list[Job], *,
                   threshold: float = 1.0) -> Written:
    """Write distill.jsonl, rl_tasks.toml, manifest.json under `out`; return what was written."""
    out = Path(out)
    jobs = _all_jobs(teacher_jobs, student_jobs)
    rl = rl_tasks(student_jobs)
    routing = route(teacher_jobs, student_jobs, threshold=threshold)
    distill_tasks = {name for name, dest in routing.items() if dest == "distill"}
    distilled = [r for r in distill(jobs, threshold=threshold) if r["task"] in distill_tasks]
    excluded = distill_excluded(jobs, distill_tasks, threshold=threshold)
    note = rl_note(student_jobs) if not rl else None
    _write_distill(out, distilled)
    _write_rl(out, rl)
    manifest = _manifest(teacher_jobs, student_jobs, routing, distilled, excluded, threshold, note)
    atomic_write(out / "manifest.json",
                 json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return Written(out=out, manifest=manifest, distill=distilled, rl=rl, excluded=excluded,
                   rl_note=note)
