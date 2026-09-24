"""The project's next-step hint — one function the Overview page and the bare `touchstone` share.

The hint reads the work queues and the Sample/Distill loop state and names the single next move:
capture, mine, interview, teach, run, sample, or distill.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import store
from . import tasks as tasks_mod


def loop_signals(root: str | Path) -> tuple[bool, int]:
    """(any Sample recorded, largest frontier) from the per-benchmark loop-state files."""
    loop_dir = Path(root) / ".touchstone" / "loop"
    sampled, frontier = False, 0
    for path in sorted(loop_dir.glob("*.json")) if loop_dir.exists() else []:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sampled = sampled or bool(state.get("last_sample"))
        frontier = max(frontier, int(state.get("frontier_size") or 0))
    return sampled, frontier


def project_signals(conn, root: str | Path) -> dict:
    """The counts `next_step` reads: episodes, task queues, runs, and the loop state."""
    tasks = tasks_mod.list_tasks(root)
    sampled, frontier = loop_signals(root)
    return {
        "episodes": len(store.list_episodes(conn)),
        "tasks": len(tasks),
        "active": sum(1 for t in tasks if t.status == "active"),
        "needs_checks": sum(1 for t in tasks if t.status == "needs_checks"),
        "needs_solution": sum(1 for t in tasks if t.status == "needs_solution"),
        "runs": len(store.list_runs(conn)),
        "sampled": sampled,
        "frontier": frontier,
    }


def _n(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _queue_hint(needs_checks: int, needs_solution: int) -> str:
    if needs_checks >= needs_solution:
        return f"{_n(needs_checks, 'task')} need a positive rule — open an interview."
    return f"{_n(needs_solution, 'task')} need a solution — Ask teacher."


def _early_hint(s: dict) -> str | None:
    """Before any benchmark: capture, mine, then clear the task queues."""
    if not s["episodes"]:
        return "Capture traces first — run touchstone demo, or add touchstone.trace() to your app."
    if not s["tasks"]:
        return "Mine your episodes into replay tasks — run touchstone mine."
    if s["needs_checks"] or s["needs_solution"]:
        return _queue_hint(s["needs_checks"], s["needs_solution"])
    return None


def _loop_hint(s: dict) -> str:
    """Once tasks are active: run, sample, distill, then rest."""
    if s["active"] and not s["runs"]:
        return f"{_n(s['active'], 'active task')} are ready — Run a model against them."
    if s["runs"] and not s["sampled"]:
        return "Runs are in — Sample a candidate to find where it falls short."
    if s["frontier"] > 0:
        return f"{_n(s['frontier'], 'task')} on the frontier — Distill them into training data."
    return "The frontier is closed — you're set. Keep sampling as you change models."


def next_step(s: dict) -> str:
    """The one line the UI and CLI show. `s` is a `project_signals` dict."""
    return _early_hint(s) or _loop_hint(s)
