"""Train datasets: prepare, download the jsonl files, and submit to a backend.

Prepare and submit call the same `touchstone.train` functions the CLI does. Downloads are served
from the prepared output dir under `.touchstone/train/<benchmark>/`, restricted to the known dataset
filenames so a path can't escape that dir.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ...bench import benchmark as benchmark_mod
from ...train import InfraRequired, TrainConfig, default_out_dir, prepare, trainer_for
from ._deps import get_conn, get_root

router = APIRouter()

_DOWNLOADS = {"sft.jsonl", "preference.jsonl", "rl_tasks.jsonl", "manifest.json", "train_plan.md"}


def _require_target(request: Request, root, target: str):
    if not benchmark_mod.resolve(root, target):
        raise HTTPException(404, f"no benchmark or tasks for {target!r}")
    return default_out_dir(request.app.state.settings.db_path, target)


def _bundle_view(bundle) -> dict:
    return {
        "target": bundle.benchmark_id,
        "out_dir": str(bundle.out_dir),
        "counts": bundle.counts,
        "files": {k: v.name for k, v in bundle.paths.items()},
    }


@router.post("/api/train/prepare")
def train_prepare(body: dict, request: Request, conn=Depends(get_conn),
                  root=Depends(get_root)) -> dict:
    target = (body.get("target") or body.get("benchmark") or "").strip()
    if not target:
        raise HTTPException(422, "target is required")
    out_dir = _require_target(request, root, target)
    return _bundle_view(prepare(conn, root, target, out_dir))


@router.post("/api/train/submit")
def train_submit(body: dict, request: Request, conn=Depends(get_conn),
                 root=Depends(get_root)) -> dict:
    target = (body.get("target") or body.get("benchmark") or "").strip()
    backend = (body.get("backend") or "null").strip()
    if not target:
        raise HTTPException(422, "target is required")
    try:
        trainer = trainer_for(backend)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    out_dir = _require_target(request, root, target)
    config = TrainConfig(base_model=body["base_model"]) if body.get("base_model") else TrainConfig()
    bundle = trainer.prepare(conn, root, target, out_dir)
    try:
        handle = trainer.submit(bundle, config)
    except InfraRequired as exc:
        return {"backend": backend, "status": "infra_required", "detail": str(exc),
                "out_dir": str(bundle.out_dir), "counts": bundle.counts}
    return {"backend": handle.backend, "status": handle.status, "detail": handle.detail,
            "artifacts": handle.artifacts, "out_dir": str(bundle.out_dir), "counts": bundle.counts}


@router.get("/api/train/download")
def train_download(
    request: Request, benchmark: str, file: str = Query(...), root=Depends(get_root),
) -> FileResponse:
    if file not in _DOWNLOADS:
        raise HTTPException(404, f"no downloadable file {file!r}")
    out_dir = _require_target(request, root, benchmark)
    path = out_dir / file
    if not path.exists():
        raise HTTPException(404, f"{file} not prepared yet — run prepare first")
    return FileResponse(path, filename=file, media_type="application/octet-stream")
