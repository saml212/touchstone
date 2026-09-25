"""Gate every task with Harbor's oracle and nop agents; a task counts only when it is solvable and
not trivially passable.

Runs `harbor run -a oracle` and `-a nop` once each over `touchstone/tasks` (all tasks per job) and
keeps a task iff oracle == 1.0 and nop < 1.0. A task that fails either side MOVES to
`touchstone/needs-review/<name>/` with a `gate.json` naming the failing side and the rewards, so
`harbor run -p touchstone/tasks` only ever runs gated tasks. Passing tasks record their result in
`task.toml` [metadata.touchstone].

Idempotent: when every task is already gated, no Harbor run happens (unless force). A Harbor build
or run failure moves the pending tasks to needs-review with the exception type; the survey exits 0.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

from ..harbor import jobs
from ..harbor import run as run_mod
from .writes import atomic_write, atomic_write_json


def _task_dirs(out: Path) -> list[Path]:
    tasks = out / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(d for d in tasks.iterdir() if (d / "task.toml").is_file())


def _read_toml(task_dir: Path) -> dict:
    return tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))


def _is_gated(doc: dict) -> bool:
    return "oracle" in doc.get("metadata", {}).get("touchstone", {})


def _rates(job_dir: Path) -> dict:
    return jobs.pass_rates(jobs.Job.read(job_dir))


def _rate_for(rates: dict, name: str) -> float | None:
    if name in rates:
        return rates[name]
    for key, value in rates.items():
        if key.endswith(f"/{name}"):
            return value
    return None


def _record_gate(task_dir: Path, oracle: float, nop: float) -> None:
    doc = _read_toml(task_dir)
    ts = doc.setdefault("metadata", {}).setdefault("touchstone", {})
    ts.update({"oracle": oracle, "nop": nop, "gated_at": datetime.now(UTC).isoformat()})
    atomic_write(task_dir / "task.toml", tomli_w.dumps(doc))


def _move_needs_review(out: Path, name: str, info: dict) -> None:
    dest = out / "needs-review" / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out / "tasks" / name), str(dest))
    atomic_write_json(dest / "gate.json", info)


def _passed(oracle: float | None, nop: float | None) -> bool:
    return oracle == 1.0 and nop is not None and nop < 1.0


def _failed_side(oracle: float | None, nop: float | None) -> str:
    return "oracle" if oracle != 1.0 else "nop"


def _apply_gate(pending: list[str], out: Path, oracle: dict, nop: dict) -> dict:
    gated, needs_review = [], []
    for name in pending:
        o, n = _rate_for(oracle, name), _rate_for(nop, name)
        if _passed(o, n):
            _record_gate(out / "tasks" / name, o, n)
            gated.append(name)
        else:
            info = {"failed_side": _failed_side(o, n), "oracle": o, "nop": n}
            _move_needs_review(out, name, info)
            needs_review.append({"name": name, **info})
    return {"gated": sorted(gated), "needs_review": needs_review, "skipped_gate": None}


def _fail_all(pending: list[str], out: Path, exc: Exception) -> dict:
    reason = str(exc)[:2000]
    needs_review = []
    for name in pending:
        info = {"failed_side": "harbor", "reason": reason}
        _move_needs_review(out, name, info)
        needs_review.append({"name": name, **info})
    return {"gated": [], "needs_review": needs_review, "skipped_gate": None}


def _read_gate(review_dir: Path) -> dict:
    return json.loads((review_dir / "gate.json").read_text(encoding="utf-8"))


def _existing_summary(out: Path, dirs: list[Path]) -> dict:
    review_root = out / "needs-review"
    reviews = []
    if review_root.is_dir():
        for d in sorted(review_root.iterdir()):
            if (d / "gate.json").is_file():
                reviews.append({"name": d.name, **_read_gate(d)})
    return {"gated": [d.name for d in dirs], "needs_review": reviews, "skipped_gate": None}


def run_gate(repo: Path, env_result: dict, settings, force: bool = False) -> dict:
    """Gate every ungated task with oracle + nop. Returns gated / needs_review names."""
    out = repo / "touchstone"
    dirs = _task_dirs(out)
    if not env_result.get("deps_ok", True):
        return {"gated": [], "needs_review": [],
                "skipped_gate": f"dependencies unresolved ({env_result.get('deps_reason')})"}
    pending = [d.name for d in dirs] if force else [d.name for d in dirs if not _is_gated(
        _read_toml(d))]
    if not pending:
        return _existing_summary(out, [d for d in dirs if _is_gated(_read_toml(d))])
    try:
        run_mod.build_image(out / "environment", env_result["image_tag"], settings)
        jobs_dir = out / "jobs"
        oracle = _rates(run_mod.run(out, "oracle", jobs_dir=jobs_dir, settings=settings))
        nop = _rates(run_mod.run(out, "nop", jobs_dir=jobs_dir, settings=settings))
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        return _fail_all(pending, out, exc)
    return _apply_gate(pending, out, oracle, nop)
