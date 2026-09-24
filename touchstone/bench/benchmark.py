"""A benchmark is a named, frozen list of task ids.

`create` resolves its members from explicit ids, tag filters, or every task, drops ids that don't
exist, and stores the ordered de-duplicated result.
"""

from __future__ import annotations

from .. import store


def _resolve_task_ids(
    conn, task_ids: list[str] | None, tags: list[str] | None, all_tasks: bool
) -> list[str]:
    tasks = store.list_tasks(conn)
    if task_ids:
        existing = {t.id for t in tasks}
        return [tid for tid in _dedupe(task_ids) if tid in existing]
    if tags:
        wanted = set(tags)
        return [t.id for t in tasks if wanted & set(t.tags or [])]
    if all_tasks:
        return [t.id for t in tasks]
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def create(
    conn,
    name: str,
    task_ids: list[str] | None = None,
    *,
    tags: list[str] | None = None,
    all_tasks: bool = False,
) -> store.Benchmark:
    """Freeze tasks into a named benchmark, from explicit ids, tag filters, or all_tasks."""
    resolved = _resolve_task_ids(conn, task_ids, tags, all_tasks)
    if not resolved:
        raise ValueError("benchmark would be empty; no matching tasks")
    return store.insert_benchmark(conn, store.Benchmark(name=name, task_ids=resolved))


def get(conn, benchmark_id: str) -> store.Benchmark | None:
    return store.get_benchmark(conn, benchmark_id)


def list_benchmarks(conn) -> list[store.Benchmark]:
    return store.list_benchmarks(conn)
