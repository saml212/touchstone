"""The Sample / Distill loop over HTTP — the two buttons, calling the same functions the CLI does.

Sample runs the student and generates teacher variants; it can be slow with a real model, so like
a benchmark run it is a synchronous call the UI shows a spinner for (the `scripted` provider is
instant). Both routes validate the model specs up front and return a one-clear-sentence 422.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ...bench import benchmark as benchmark_mod
from ._deps import get_conn, get_root

router = APIRouter()


def _require_target(root, target: str) -> None:
    if not benchmark_mod.resolve(root, target):
        raise HTTPException(404, f"no benchmark or active tasks for {target!r}")


def _judge(settings, spec):
    from ...llm import provider_from_spec

    if not spec:
        return None
    try:
        return provider_from_spec(spec, settings)
    except Exception as exc:
        raise HTTPException(422, f"cannot use judge {spec!r}: {exc}") from exc


@router.post("/api/sample")
def post_sample(body: dict, request: Request, conn=Depends(get_conn),
                root=Depends(get_root)) -> dict:
    from ...loop import sample as run_sample

    target = (body.get("target") or body.get("benchmark") or "").strip()
    student = (body.get("student") or "").strip()
    if not target or not student:
        raise HTTPException(422, "target and student are required")
    _require_target(root, target)
    settings = request.app.state.settings
    try:
        return run_sample(conn, root, target, student,
                          teacher_spec=body.get("teacher") or None,
                          variants=int(body.get("variants", 1)),
                          settings=settings,
                          judge_provider=_judge(settings, body.get("judge")),
                          concurrency=int(body.get("concurrency", 4)))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"sample failed: {exc}") from exc


@router.post("/api/distill")
def post_distill(body: dict, request: Request, conn=Depends(get_conn),
                 root=Depends(get_root)) -> dict:
    from ...loop import distill as run_distill

    target = (body.get("target") or body.get("benchmark") or "").strip()
    student = (body.get("student") or "").strip()
    if not target or not student:
        raise HTTPException(422, "target and student are required")
    _require_target(root, target)
    settings = request.app.state.settings
    try:
        return run_distill(conn, root, target, student,
                           teacher_spec=body.get("teacher") or None,
                           backend=(body.get("backend") or "null").strip(),
                           base_model=body.get("base_model") or None,
                           settings=settings,
                           judge_provider=_judge(settings, body.get("judge")))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"distill failed: {exc}") from exc


@router.get("/api/loop/{benchmark}")
def get_loop_state(benchmark: str, root=Depends(get_root)) -> dict:
    from ...loop import read_loop_state

    return read_loop_state(root, benchmark)


@router.post("/api/tasks/{name}/teach")
def teach_task(name: str, request: Request, conn=Depends(get_conn),
               root=Depends(get_root)) -> dict:
    """Ask the teacher for a verified reply to one task (adopts it as the oracle if needed)."""
    from ... import tasks as tasks_mod
    from ...loop import teach

    if tasks_mod.get_task(root, name) is None:
        raise HTTPException(404, f"no task {name!r}")
    result = teach(conn, root, [name], settings=request.app.state.settings)
    task = tasks_mod.get_task(root, name)
    return {"accepted": name in result["accepted"], "status": task.status,
            "teacher": result["teacher"]}
