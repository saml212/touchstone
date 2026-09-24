"""Train datasets: prepare, download the jsonl files, and submit to a backend.

Prepare and submit call the same `touchstone.train` functions the CLI does. Downloads are served
from the prepared output dir under `.touchstone/train/<benchmark>/`, restricted to the known dataset
filenames so a path can't escape that dir.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ... import store
from ...train import InfraRequired, TrainConfig, default_out_dir, prepare, trainer_for
from ._deps import get_conn

router = APIRouter()

_DOWNLOADS = {"sft.jsonl", "preference.jsonl", "rl_tasks.jsonl", "manifest.json", "train_plan.md"}


def _out_dir(request: Request, conn, benchmark_id: str) -> tuple[store.Benchmark, object]:
    bench = store.get_benchmark(conn, benchmark_id)
    if bench is None:
        raise HTTPException(404, f"no benchmark {benchmark_id!r}")
    db_path = request.app.state.settings.db_path
    return bench, default_out_dir(db_path, bench)


def _bundle_view(bundle) -> dict:
    return {
        "benchmark_id": bundle.benchmark_id,
        "benchmark_name": bundle.benchmark_name,
        "out_dir": str(bundle.out_dir),
        "counts": bundle.counts,
        "files": {k: v.name for k, v in bundle.paths.items()},
    }


@router.post("/api/train/prepare")
def train_prepare(body: dict, request: Request, conn=Depends(get_conn)) -> dict:
    benchmark_id = (body.get("benchmark") or "").strip()
    if not benchmark_id:
        raise HTTPException(422, "benchmark is required")
    bench, out_dir = _out_dir(request, conn, benchmark_id)
    return _bundle_view(prepare(conn, bench.id, out_dir))


@router.post("/api/train/submit")
def train_submit(body: dict, request: Request, conn=Depends(get_conn)) -> dict:
    benchmark_id = (body.get("benchmark") or "").strip()
    backend = (body.get("backend") or "null").strip()
    if not benchmark_id:
        raise HTTPException(422, "benchmark is required")
    try:
        trainer = trainer_for(backend)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    bench, out_dir = _out_dir(request, conn, benchmark_id)
    config = TrainConfig(base_model=body["base_model"]) if body.get("base_model") else TrainConfig()
    bundle = trainer.prepare(conn, bench.id, out_dir)
    try:
        handle = trainer.submit(bundle, config)
    except InfraRequired as exc:
        # Expected for art/trl: the config was written, the run needs GPUs we don't have here.
        return {"backend": backend, "status": "infra_required", "detail": str(exc),
                "out_dir": str(bundle.out_dir), "counts": bundle.counts}
    return {"backend": handle.backend, "status": handle.status, "detail": handle.detail,
            "artifacts": handle.artifacts, "out_dir": str(bundle.out_dir), "counts": bundle.counts}


@router.get("/api/train/download")
def train_download(
    request: Request, benchmark: str, file: str = Query(...), conn=Depends(get_conn),
) -> FileResponse:
    if file not in _DOWNLOADS:
        raise HTTPException(404, f"no downloadable file {file!r}")
    _bench, out_dir = _out_dir(request, conn, benchmark)
    path = out_dir / file
    if not path.exists():
        raise HTTPException(404, f"{file} not prepared yet — run prepare first")
    return FileResponse(path, filename=file, media_type="application/octet-stream")
