"""Benchmarks, runs, reports, proofs, and Harbor export.

A run is created synchronously (so the id comes back at once) and executed on a background thread
with its own connection; progress is read back by counting stored results against the run's task
count. Report and proof call the same `scoreboard` / `proof` functions the CLI prints.
"""

from __future__ import annotations

import threading
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ... import store
from ...bench import benchmark, proof, runner, scoreboard
from ...bench import export as export_harbor
from ._deps import get_conn

router = APIRouter()


@router.get("/api/benchmarks")
def list_benchmarks(conn=Depends(get_conn)) -> dict:
    return {"benchmarks": [asdict(b) for b in store.list_benchmarks(conn)]}


@router.post("/api/benchmarks")
def create_benchmark(body: dict, conn=Depends(get_conn)) -> dict:
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "name is required")
    try:
        bench = benchmark.create(
            conn, name, task_ids=body.get("task_ids"),
            tags=body.get("tags"), all_tasks=bool(body.get("all")),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return asdict(bench)


@router.get("/api/runs")
def list_runs(conn=Depends(get_conn)) -> dict:
    return {"runs": [_run_view(conn, r) for r in store.list_runs(conn)]}


@router.post("/api/runs")
def create_run(body: dict, request: Request, conn=Depends(get_conn)) -> dict:
    benchmark_id = (body.get("benchmark") or "").strip()
    model_spec = (body.get("model_spec") or "").strip()
    if not benchmark_id or not model_spec:
        raise HTTPException(422, "benchmark and model_spec are required")
    settings = request.app.state.settings
    from ...llm import provider_from_spec

    try:
        provider = provider_from_spec(model_spec, settings)
    except Exception as exc:
        raise HTTPException(422, f"cannot use model {model_spec!r}: {exc}") from exc
    judge = body.get("judge")
    judge_provider = None
    if judge:
        try:
            judge_provider = provider_from_spec(judge, settings)
        except Exception as exc:
            raise HTTPException(422, f"cannot use judge {judge!r}: {exc}") from exc
    try:
        run = runner.start(conn, benchmark_id, model_spec,
                           concurrency=int(body.get("concurrency", 4)))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    _execute_in_background(settings.db_path, run.id, provider, judge_provider)
    return _run_view(conn, run)


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, conn=Depends(get_conn)) -> dict:
    run = store.get_run(conn, run_id)
    if run is None:
        raise HTTPException(404, f"no run with id {run_id}")
    return _run_view(conn, run)


@router.get("/api/report")
def report(run: list[str] = Query(default=[]), conn=Depends(get_conn)) -> dict:
    return scoreboard(conn, run)


@router.get("/api/proof")
def get_proof(candidate: str, incumbent: str, conn=Depends(get_conn)) -> dict:
    for run_id in (candidate, incumbent):
        if store.get_run(conn, run_id) is None:
            raise HTTPException(404, f"no run with id {run_id}")
    return proof(conn, candidate, incumbent)


@router.post("/api/export/harbor")
def export_to_harbor(body: dict, conn=Depends(get_conn)) -> dict:
    benchmark_id = (body.get("benchmark") or "").strip()
    out = (body.get("out") or "./harbor-tasks").strip()
    if not benchmark_id:
        raise HTTPException(422, "benchmark is required")
    try:
        dirs = export_harbor(conn, benchmark_id, out)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"out": out, "count": len(dirs), "dirs": [str(d) for d in dirs]}


# ---- helpers ---------------------------------------------------------------


def _run_view(conn, run: store.Run) -> dict:
    done = len(store.list_results(conn, run.id))
    total = run.meta.get("task_count", 0)
    return {
        "id": run.id, "benchmark_id": run.benchmark_id, "model_spec": run.model_spec,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "done": done, "total": total,
    }


def _execute_in_background(db_path, run_id, provider, judge_provider) -> None:
    def work() -> None:
        conn = store.connect(db_path)
        try:
            run = store.get_run(conn, run_id)
            runner.execute(conn, run, judge_provider=judge_provider, provider=provider)
        finally:
            conn.close()

    threading.Thread(target=work, daemon=True).start()
