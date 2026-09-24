"""Difficulty measured from run results, the failure frontier, and the loop's stopping criteria.

Difficulty is measured, not requested: `record_run` upserts each result into the store's difficulty
table and caches the latest pass rate into the task's `task.toml` (the DB is the source; the file
value is a display cache). The `frontier` is where a student can still improve — every active task
with `0 < pass_rate < 1` (the learnability band) plus the ones the incumbent passes and the student
fails (`pass_rate == 0`), which is the proof table's "only incumbent" set. `check_stop` ends the
loop when any criterion holds: an empty frontier, a pass-rate delta below threshold vs the previous
Sample, the teacher failing the gate on every frontier task, or a per-round cost cap.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import store
from .. import tasks as tasks_mod


def record_run(conn, root: str | Path, run: store.Run) -> None:
    """Fold a completed run's results into the difficulty table + each task's file cache."""
    for result in store.list_results(conn, run.id):
        diff = store.upsert_difficulty(conn, result.task, run.model_spec, bool(result.passed))
        _cache_pass_rate(root, result.task, run.model_spec, diff.pass_rate)


def _cache_pass_rate(root: str | Path, task_name: str, model_spec: str, pass_rate: float) -> None:
    task = tasks_mod.get_task(root, task_name)
    if task is None:
        return
    task.difficulty[model_spec] = round(pass_rate, 4)
    tasks_mod.write_task(root, task)


def _active_names(root: str | Path) -> set[str]:
    return {t.name for t in tasks_mod.list_tasks(root, active_only=True)}


def frontier_split(conn, root: str | Path, model_spec: str) -> dict[str, list[str]]:
    """The frontier split into the learnability band (0 < pr < 1) and only-incumbent (pr == 0)."""
    active = _active_names(root)
    learnability, only_incumbent = [], []
    for d in store.list_difficulty(conn, model_spec=model_spec):
        if d.task not in active:
            continue
        if d.pass_rate <= 0.0:
            only_incumbent.append(d.task)
        elif d.pass_rate < 1.0:
            learnability.append(d.task)
    return {"learnability": sorted(learnability), "only_incumbent": sorted(only_incumbent)}


def frontier(conn, root: str | Path, model_spec: str) -> list[str]:
    """Active task names the student does not yet fully pass — the tasks worth teaching."""
    split = frontier_split(conn, root, model_spec)
    return sorted(set(split["learnability"]) | set(split["only_incumbent"]))


def mean_pass_rate(conn, root: str | Path, model_spec: str) -> float:
    """Mean pass rate across the active tasks this student has been sampled on (0.0 when none)."""
    active = _active_names(root)
    rates = [d.pass_rate for d in store.list_difficulty(conn, model_spec=model_spec)
             if d.task in active]
    return sum(rates) / len(rates) if rates else 0.0


def check_stop(
    conn,
    root: str | Path,
    model_spec: str,
    *,
    prev_pass_rate: float | None = None,
    delta_threshold: float = 0.01,
    round_cost: float = 0.0,
    cost_cap: float | None = None,
    teacher_all_failed: bool = False,
) -> tuple[bool, str]:
    """(stop, reason) — the loop halts when any stopping criterion holds."""
    if not frontier(conn, root, model_spec):
        return True, "frontier is empty: no task with 0 < pass_rate < 1 remains"
    if teacher_all_failed:
        return True, "the teacher failed the gate on every frontier task — escalate to a human"
    if cost_cap is not None and round_cost >= cost_cap:
        return True, f"cost cap reached (${round_cost:.4f} >= ${cost_cap:.4f})"
    if prev_pass_rate is not None:
        delta = mean_pass_rate(conn, root, model_spec) - prev_pass_rate
        if delta < delta_threshold:
            return True, f"pass-rate delta {delta:+.3f} below threshold {delta_threshold}"
    return False, ""


# ---- loop state (last Sample/Distill timestamps + frontier size, per benchmark) -------------


def _loop_dir(root: str | Path) -> Path:
    return Path(root) / ".touchstone" / "loop"


def loop_state_path(root: str | Path, benchmark: str) -> Path:
    return _loop_dir(root) / f"{benchmark}.json"


def read_loop_state(root: str | Path, benchmark: str) -> dict:
    path = loop_state_path(root, benchmark)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_loop_state(root: str | Path, benchmark: str, **fields) -> dict:
    state = read_loop_state(root, benchmark)
    state.update(fields)
    path = loop_state_path(root, benchmark)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state
