"""Orchestrate mining: statistics (`mine/stats.py`), one LLM pass (`mine/llm.py`), then cut tasks.

`dedupe` drops proposals identical to an existing check; `mine` runs the statistical proposals
always, the LLM pass unless suppressed, inserts the fresh ones as `mined` policies, and
re-materialises every task directory. `sync` re-materialises without proposing anything new.
"""

from __future__ import annotations

import json

from .. import store
from ..policies import materialize, read_policies, write_policies
from ..tasks import Task, migrate_names, write_task
from . import cut as cut_mod
from .codebase import Snippet
from .llm import mine_llm
from .stats import Proposal, mine_stats


def _norm(kind: str, params: dict) -> str:
    return kind + ":" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)


def dedupe(existing_checks, proposals: list[Proposal]) -> list[Proposal]:
    seen = {_norm(c.kind, c.params or {}) for c in existing_checks}
    out: list[Proposal] = []
    for p in proposals:
        key = _norm(p.kind, p.params)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _materialize(root: str, conn, episodes: list[store.Episode]) -> list[Task]:
    """Rebuild every task directory from `episodes` and the currently enabled policies."""
    enabled = [p for p in read_policies(root) if p.enabled]
    built = cut_mod.build_tasks(conn, episodes)
    migrate_names(root, {(t.episode_id, t.cut_span_id): t.name for t in built})
    for task in built:
        task.checks = materialize(task, enabled)
        write_task(root, task)
    return built


def sync(conn, root: str) -> int:
    """Re-materialise every task from the captured episodes + current policies."""
    return len(_materialize(root, conn, store.list_episodes(conn)))


def mine(
    conn,
    root: str,
    *,
    provider=None,
    code_snippets: list[Snippet] | None = None,
    no_llm: bool = False,
    limit: int | None = None,
) -> dict:
    episodes = store.list_episodes(conn)
    if limit is not None:
        episodes = episodes[:limit]

    stats = mine_stats(conn, episodes)
    llm = [] if no_llm else mine_llm(conn, provider, episodes, code_snippets or [])
    existing = read_policies(root)
    fresh = dedupe([p.check for p in existing], stats + llm)
    write_policies(root, existing + [p.policy() for p in fresh])

    tasks = _materialize(root, conn, episodes)
    return {
        "proposals": fresh,
        "stats": len(stats),
        "llm": len(llm),
        "inserted": len(fresh),
        "tasks_cut": len(tasks),
    }
