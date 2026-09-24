"""A benchmark is `benchmarks/<name>.toml`: an explicit `tasks = [...]` list, or a `glob` + `tags`.

Resolving a benchmark yields the active task directories it names. A target passed to the runner is
either a benchmark name, a `tasks/` path, or a glob — `resolve` handles all three.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from .. import tasks as tasks_mod


def benchmarks_dir(root: str | Path) -> Path:
    return Path(root) / "benchmarks"


def spec_path(root: str | Path, name: str) -> Path:
    return benchmarks_dir(root) / f"{name}.toml"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _active(task_dir: Path) -> bool:
    return (task_dir / "task.toml").exists() and tasks_mod.read_task(task_dir).status == "active"


def _by_glob(root: Path, pattern: str, tags: list[str] | None) -> list[Path]:
    wanted = set(tags or [])
    out = []
    for task_dir in sorted(root.glob(pattern)):
        if not _active(task_dir):
            continue
        if wanted and not wanted & set(tasks_mod.read_task(task_dir).tags or []):
            continue
        out.append(task_dir)
    return out


def resolve(root: str | Path, target: str) -> list[Path]:
    """Task dirs for a benchmark name, a `tasks/` path, or a glob — active tasks only."""
    root = Path(root)
    spec = spec_path(root, target)
    if spec.exists():
        doc = tomllib.loads(spec.read_text(encoding="utf-8"))
        if "tasks" in doc:
            return [root / t for t in doc["tasks"] if _active(root / t)]
        return _by_glob(root, doc.get("glob", "tasks/*"), doc.get("tags"))
    return _by_glob(root, target, None)


def create(
    root: str | Path,
    name: str,
    *,
    task_names: list[str] | None = None,
    tags: list[str] | None = None,
    all_tasks: bool = False,
) -> Path:
    """Write `benchmarks/<name>.toml` from explicit task names, tag filters, or every task."""
    if task_names:
        doc: dict = {"tasks": [f"tasks/{n}" for n in _dedupe(task_names)]}
    elif tags:
        doc = {"glob": "tasks/*", "tags": list(tags)}
    elif all_tasks:
        doc = {"glob": "tasks/*"}
    else:
        raise ValueError("give task names, tags, or all_tasks")

    path = spec_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(doc), encoding="utf-8")
    if not resolve(root, name):
        path.unlink()
        raise ValueError("benchmark would be empty; no matching active tasks")
    return path


def list_names(root: str | Path) -> list[str]:
    base = benchmarks_dir(root)
    return sorted(p.stem for p in base.glob("*.toml")) if base.exists() else []


def view(root: str | Path, name: str) -> dict:
    return {"name": name, "task_count": len(resolve(root, name)),
            "tasks": [d.name for d in resolve(root, name)]}


def harbor_tasks_path(root: str | Path) -> Path:
    """Tasks are already Harbor tasks — this is the path `harbor run -p` takes, no export step."""
    return Path(root) / "tasks"
