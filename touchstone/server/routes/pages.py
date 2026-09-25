"""GET /api/pages/* — the JSON each of the five UI pages reads, assembled from the dataset files
and Harbor job dirs at request time (`server/pages.py`). Empty dataset -> empty data, never a 500.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import pages
from ._deps import get_conn, get_settings

router = APIRouter(prefix="/api/pages")


@router.get("/overview")
def overview(conn=Depends(get_conn), settings=Depends(get_settings)) -> dict:
    return pages.overview(settings.review_dataset_dir, settings.review_jobs, conn)


@router.get("/tasks")
def tasks(settings=Depends(get_settings)) -> dict:
    return {"tasks": pages.tasks(settings.review_dataset_dir, settings.review_jobs)}


@router.get("/tasks/{name}")
def task_detail(name: str, settings=Depends(get_settings)) -> dict:
    detail = pages.task_detail(settings.review_dataset_dir, settings.review_jobs, name)
    if detail is None:
        raise HTTPException(404, f"no task {name}")
    return detail


@router.get("/jobs")
def jobs(settings=Depends(get_settings)) -> dict:
    return {"jobs": pages.jobs(settings.review_jobs)}


@router.get("/jobs/{job}")
def job_rewards(job: str, settings=Depends(get_settings)) -> dict:
    detail = pages.job_rewards(settings.review_jobs, job)
    if detail is None:
        raise HTTPException(404, f"no job {job}")
    return detail


@router.get("/trial")
def trial(task: str = Query(...), trial_id: str = Query(..., alias="trial"),
          settings=Depends(get_settings)) -> dict:
    detail = pages.trial(settings.review_dataset_dir, settings.review_jobs, task, trial_id)
    if detail is None:
        raise HTTPException(404, "no such trial")
    return detail


@router.get("/train")
def train(settings=Depends(get_settings)) -> dict:
    return pages.train(settings.review_dataset_dir)
