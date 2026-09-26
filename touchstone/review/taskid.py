"""Resolve which task a change acts on — the reviewer never asks the person for a task id.

A change defaults to the trial currently open in the room. A task name in the person's words is
matched leniently against the dataset: an exact task dir wins; a prefix without the ``-N`` suffix
(``check-order-status`` for the open ``check-order-status-1``) resolves to the open trial's task;
otherwise the closest task that starts with that prefix. Only when nothing matches and no trial is
open does it raise — and then it says to open a trial, never "tell me the task id".
"""

from __future__ import annotations

from pathlib import Path

from .changes import ChangeError


def _task_names(dataset_dir: Path) -> set[str]:
    tasks = dataset_dir / "tasks"
    if not tasks.is_dir():
        return set()
    return {d.name for d in tasks.iterdir() if (d / "task.toml").is_file()}


def _closest(named: str, names: set[str], open_task: str) -> str | None:
    """The task the person's prefix points at: the open trial's task if it matches, else the first
    task named ``<prefix>-N`` (the open task is preferred so a stray prefix stays on the trial)."""
    matches = sorted(n for n in names if n == named or n.startswith(named + "-"))
    if not matches:
        return None
    return open_task if open_task in matches else matches[0]


def _match_named(named: str, names: set[str], open_task: str) -> str | None:
    """The task a non-empty person-supplied name resolves to: exact, then a prefix of the open
    trial's task, then the closest ``<prefix>-N`` task — or None when nothing matches."""
    if named in names:
        return named
    if open_task and open_task.startswith(named):
        return open_task
    return _closest(named, names, open_task)


def resolve_task(dataset_dir: str | Path, named: str, open_task: str) -> str:
    """The task a change acts on, matched leniently and defaulting to the open trial."""
    dataset_dir = Path(dataset_dir)
    named, open_task = (named or "").strip(), (open_task or "").strip()
    matched = _match_named(named, _task_names(dataset_dir), open_task) if named else None
    if matched is not None:
        return matched
    if open_task:
        return open_task
    raise ChangeError("no trial is open — read a trial first, then change it.")
