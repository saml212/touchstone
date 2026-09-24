"""Scoreboard and proof reports over stored runs, as data + plain-text tables (no rich dependency).

`scoreboard` aggregates one or more runs into pass rates per model, per check, and per tag. `proof`
diffs a candidate run against an incumbent run task-by-task into four categories plus cost totals.
`render_*` turn either into aligned text tables; the dicts themselves are the JSON form.
"""

from __future__ import annotations

from .. import store

_CATEGORIES = ("both_pass", "only_incumbent", "only_candidate", "both_fail")


def _pct(passed: int, total: int) -> float:
    return round(100.0 * passed / total, 1) if total else 0.0


def _check_meta(conn, check_id: str) -> tuple[str, str]:
    row = store.get_check(conn, check_id)
    if row is None:
        return "(deleted)", "?"
    return row.name or row.kind, row.kind


def _model_summary(run: store.Run, results: list[store.Result]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    return {
        "run_id": run.id,
        "model_spec": run.model_spec,
        "total": total,
        "passed": passed,
        "errors": errors,
        "pass_rate": _pct(passed, total),
        "cost_usd": round(sum(costs), 6) if costs else None,
    }


def _tally_checks(conn, results: list[store.Result], per_check: dict) -> None:
    for r in results:
        for cr in r.check_results or []:
            cid = cr.get("check_id") or ""
            entry = per_check.setdefault(cid, {"pass": 0, "fail": 0, "skip": 0})
            passed = cr.get("passed")
            entry["pass" if passed is True else "skip" if passed is None else "fail"] += 1
    for cid, entry in per_check.items():
        name, kind = _check_meta(conn, cid)
        entry.setdefault("check_id", cid)
        entry["name"], entry["kind"] = name, kind


def _tally_tags(conn, results: list[store.Result], per_tag: dict) -> None:
    for r in results:
        task = store.get_task(conn, r.task_id)
        for tag in (task.tags or []) if task else []:
            entry = per_tag.setdefault(tag, {"tag": tag, "total": 0, "passed": 0})
            entry["total"] += 1
            entry["passed"] += 1 if r.passed else 0
    for entry in per_tag.values():
        entry["pass_rate"] = _pct(entry["passed"], entry["total"])


def scoreboard(conn, run_ids: list[str]) -> dict:
    """Aggregate the given runs: a per-model row, and per-check / per-tag counts keyed by model."""
    models, checks, tags = [], [], []
    for run_id in run_ids:
        run = store.get_run(conn, run_id)
        if run is None:
            continue
        results = store.list_results(conn, run_id)
        models.append(_model_summary(run, results))

        per_check: dict = {}
        _tally_checks(conn, results, per_check)
        for entry in per_check.values():
            checks.append({"model_spec": run.model_spec, **entry})

        per_tag: dict = {}
        _tally_tags(conn, results, per_tag)
        for entry in per_tag.values():
            tags.append({"model_spec": run.model_spec, **entry})

    return {"models": models, "checks": checks, "tags": tags}


def proof(conn, candidate_run: str, incumbent_run: str) -> dict:
    """Task-by-task diff of candidate vs incumbent into four categories, plus cost totals."""
    cand_run = store.get_run(conn, candidate_run)
    inc_run = store.get_run(conn, incumbent_run)
    for label, rid, run in (("candidate", candidate_run, cand_run),
                            ("incumbent", incumbent_run, inc_run)):
        if run is None:
            raise ValueError(f"no {label} run with id {rid!r}")
    if cand_run.benchmark_id != inc_run.benchmark_id:
        raise ValueError(
            "cannot compare runs from different benchmarks "
            f"({cand_run.benchmark_id} vs {inc_run.benchmark_id})"
        )
    cand = {r.task_id: r for r in store.list_results(conn, candidate_run)}
    inc = {r.task_id: r for r in store.list_results(conn, incumbent_run)}
    counts = dict.fromkeys(_CATEGORIES, 0)
    rows = []
    for task_id in sorted(set(cand) & set(inc)):
        c_pass = bool(cand[task_id].passed)
        i_pass = bool(inc[task_id].passed)
        category = (
            "both_pass" if c_pass and i_pass
            else "only_candidate" if c_pass
            else "only_incumbent" if i_pass
            else "both_fail"
        )
        counts[category] += 1
        task = store.get_task(conn, task_id)
        rows.append({
            "task_id": task_id,
            "name": task.name if task else "(deleted)",
            "candidate_passed": c_pass,
            "incumbent_passed": i_pass,
            "category": category,
        })
    return {
        "candidate_run": candidate_run,
        "incumbent_run": incumbent_run,
        "counts": counts,
        "tasks": rows,
        "cost": {
            "candidate": _total_cost(cand.values()),
            "incumbent": _total_cost(inc.values()),
        },
    }


def _total_cost(results) -> float | None:
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    return round(sum(costs), 6) if costs else None


# ---- renderers -------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * w for w in widths)]
    out.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(out)


def _cost_str(cost: float | None) -> str:
    return "-" if cost is None else f"${cost:.4f}"


def render_scoreboard(board: dict) -> str:
    blocks = ["MODELS", _table(
        ["model", "tasks", "pass", "rate%", "errors", "cost"],
        [[m["model_spec"], str(m["total"]), str(m["passed"]), str(m["pass_rate"]),
          str(m["errors"]), _cost_str(m["cost_usd"])] for m in board["models"]],
    )]
    if board["checks"]:
        blocks += ["\nCHECKS", _table(
            ["model", "check", "kind", "pass", "fail", "skip"],
            [[c["model_spec"], c["name"], c["kind"], str(c["pass"]), str(c["fail"]),
              str(c["skip"])] for c in board["checks"]],
        )]
    if board["tags"]:
        blocks += ["\nTAGS", _table(
            ["model", "tag", "total", "passed", "rate%"],
            [[t["model_spec"], t["tag"], str(t["total"]), str(t["passed"]),
              str(t["pass_rate"])] for t in board["tags"]],
        )]
    return "\n".join(blocks)


def render_proof(report: dict) -> str:
    verdict = {"both_pass": "= both", "only_incumbent": "- lost", "only_candidate": "+ gained",
               "both_fail": "x both fail"}
    table = _table(
        ["task", "incumbent", "candidate", "verdict"],
        [[r["name"], "pass" if r["incumbent_passed"] else "fail",
          "pass" if r["candidate_passed"] else "fail", verdict[r["category"]]]
         for r in report["tasks"]],
    )
    c = report["counts"]
    summary = (
        f"both pass: {c['both_pass']}   only incumbent: {c['only_incumbent']}   "
        f"only candidate: {c['only_candidate']}   both fail: {c['both_fail']}"
    )
    cost = (
        f"cost  incumbent: {_cost_str(report['cost']['incumbent'])}   "
        f"candidate: {_cost_str(report['cost']['candidate'])}"
    )
    return "\n".join([table, "", summary, cost])
