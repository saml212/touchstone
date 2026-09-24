"""Benchmarks, runs, reports, proofs, and the Harbor-tasks pointer.

A run is created synchronously (so the id comes back at once) and executed on a background thread
with its own connection; progress is read back by counting stored results against the run's task
count. Report and proof call the same `scoreboard` / `proof` functions the CLI prints.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ... import store
from ...bench import benchmark, harbor_tasks_path, proof, runner, scoreboard
from ._deps import get_conn, get_root

router = APIRouter()


@router.get("/api/benchmarks")
def list_benchmarks(root=Depends(get_root)) -> dict:
    return {"benchmarks": [benchmark.view(root, name) for name in benchmark.list_names(root)]}


@router.post("/api/benchmarks")
def create_benchmark(body: dict, root=Depends(get_root)) -> dict:
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "name is required")
    try:
        benchmark.create(root, name, task_names=body.get("task_names"),
                         tags=body.get("tags"), all_tasks=bool(body.get("all")))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return benchmark.view(root, name)


@router.get("/api/runs")
def list_runs(conn=Depends(get_conn)) -> dict:
    return {"runs": [_run_view(conn, r) for r in store.list_runs(conn)]}


@router.post("/api/runs")
def create_run(body: dict, request: Request, conn=Depends(get_conn),
               root=Depends(get_root)) -> dict:
    target = (body.get("target") or body.get("benchmark") or "").strip()
    model_spec = (body.get("model_spec") or "").strip()
    if not target or not model_spec:
        raise HTTPException(422, "target and model_spec are required")
    settings = request.app.state.settings
    from ...llm import provider_from_spec

    try:
        provider = provider_from_spec(model_spec, settings)
    except Exception as exc:
        raise HTTPException(422, f"cannot use model {model_spec!r}: {exc}") from exc
    judge_provider = None
    if body.get("judge"):
        try:
            judge_provider = provider_from_spec(body["judge"], settings)
        except Exception as exc:
            raise HTTPException(422, f"cannot use judge {body['judge']!r}: {exc}") from exc
    try:
        run = runner.start(conn, root, target, model_spec,
                           concurrency=int(body.get("concurrency", 4)))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    _execute_in_background(settings.db_path, settings.root, run.id, provider, judge_provider)
    return _run_view(conn, run)


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, conn=Depends(get_conn)) -> dict:
    run = store.get_run(conn, run_id)
    if run is None:
        raise HTTPException(404, f"no run with id {run_id}")
    return _run_view(conn, run)


@router.get("/api/report")
def report(run: list[str] = Query(default=[]), conn=Depends(get_conn),
           root=Depends(get_root)) -> dict:
    return scoreboard(conn, root, run)


@router.get("/api/proof")
def get_proof(candidate: str, incumbent: str, conn=Depends(get_conn)) -> dict:
    for run_id in (candidate, incumbent):
        if store.get_run(conn, run_id) is None:
            raise HTTPException(404, f"no run with id {run_id}")
    return proof(conn, candidate, incumbent)


@router.post("/api/export/harbor")
def export_to_harbor(body: dict, root=Depends(get_root)) -> dict:
    path = harbor_tasks_path(root)
    return {"tasks_path": str(path),
            "command": f"harbor run -p {path}",
            "note": "Your tasks are already Harbor tasks; run harbor directly on the tasks dir."}


# ---- helpers ---------------------------------------------------------------


def _run_view(conn, run: store.Run) -> dict:
    done = len(store.list_results(conn, run.id))
    return {
        "id": run.id, "target": run.target, "model_spec": run.model_spec,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "done": done, "total": run.meta.get("task_count", 0),
    }


def _execute_in_background(db_path, root, run_id, provider, judge_provider) -> None:
    def work() -> None:
        conn = store.connect(db_path)
        try:
            run = store.get_run(conn, run_id)
            runner.execute(conn, root, run, judge_provider=judge_provider, provider=provider)
        finally:
            conn.close()

    threading.Thread(target=work, daemon=True).start()
