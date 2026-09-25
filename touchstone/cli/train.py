"""`touchstone train` turns finished Harbor jobs into distillation + RL datasets.

Reads job dirs under `<dataset>/jobs` (default), routes each student task (distill / rl / hold-out /
stuck), and writes `distill.jsonl`, `rl_tasks.toml`, `manifest.json`. `--teacher provider/model`
runs the teacher job first (replica agent); a `--teacher <job dir>` is used as-is. `--student`
defaults to the newest job dir that is not the teacher. Stops at the GPU line — no training here.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import app
from ._common import _fail

AGENT_PATH = "touchstone.harbor.agent:TouchstoneAgent"


def _newest_dir(jobs_dir: Path, exclude: Path | None) -> Path | None:
    if not jobs_dir.is_dir():
        return None
    dirs = [d for d in jobs_dir.iterdir()
            if d.is_dir() and (exclude is None or d.resolve() != exclude.resolve())]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def _resolve_teacher(teacher: str, jobs_dir: Path, n_concurrent: int) -> Path:
    """A teacher job dir used as-is, or a `provider/model` spec run first as the replica agent."""
    if Path(teacher).is_dir():
        return Path(teacher)
    from ..harbor import run as run_mod

    typer.echo(f"running teacher job for {teacher} ...")
    return run_mod.run(jobs_dir.parent, AGENT_PATH, model=teacher, jobs_dir=str(jobs_dir),
                       n_concurrent=n_concurrent, extra_args=["--ak", "mode=replica"])


@app.command()
def train(
    jobs_dir: str = typer.Option("touchstone/jobs", "--jobs-dir",
                                 help="Directory holding job dirs (default <dataset>/jobs)."),
    teacher: str = typer.Option(None, "--teacher",
                                help="Teacher job dir, or provider/model to run first."),
    student: str = typer.Option(None, "--student",
                                help="Student job dir (default: newest that is not the teacher)."),
    out: str = typer.Option("touchstone/train", "--out", help="Where to write the datasets."),
    threshold: float = typer.Option(1.0, "--threshold", help="Min reward a trial distills at."),
    n_concurrent: int = typer.Option(4, "-n", "--n-concurrent",
                                     help="Concurrent trials when running a teacher spec."),
) -> None:
    """Read Harbor jobs and write distillation + RL datasets a training stack consumes."""
    from ..harbor.jobs import Job
    from ..train.write import write_datasets

    jobs_path = Path(jobs_dir)
    teacher_dir = _resolve_teacher(teacher, jobs_path, n_concurrent) if teacher else None
    student_dir = Path(student) if student else _newest_dir(jobs_path, teacher_dir)
    if student_dir is None:
        _fail(f"no student job dir: pass --student or add jobs under {jobs_path}")

    teacher_jobs = [Job.read(teacher_dir)] if teacher_dir else []
    written = write_datasets(out, teacher_jobs, [Job.read(student_dir)], threshold=threshold)
    typer.echo(written.sentence())
