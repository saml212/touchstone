"""`touchstone train` turns finished Harbor jobs into distillation + RL datasets.

Reads job dirs under `<dataset>/jobs` (default), routes each student task (distill / rl / hold-out /
stuck), and writes `distill.jsonl`, `rl_tasks.toml`, `manifest.json` under `<dataset>/train`.

`--teacher` and `--student` each accept a job dir, a comma-separated list of job dirs (pooled — the
routing reads pass rates across all of them), or a `provider/model` spec. A spec runs a Harbor job
first with the replica agent and `--attempts` per task, so a re-run model's pass rates are real
fractions (the 0 < rate < 1 band the routing turns on). `--student` defaults to the newest job the
review treats as material (gate jobs and errored-only jobs excluded, the same filter as review),
never the nop/oracle gate job. Stops at the GPU line — no training here.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import app
from ._common import _daemon_guard, _fail, _require_tasks

AGENT_PATH = "touchstone.harbor.agent:TouchstoneAgent"


def _newest_reviewable_dir(jobs_dir: Path, exclude: set[Path]) -> Path | None:
    """The most recent job the review would treat as material, reusing the review's own filter:
    oracle/nop gate jobs dropped (a gate job is no training signal), errored-only jobs dropped,
    latest per (agent, model). So train never silently picks the nop gate job as its student."""
    from ..review.trials import reviewable_jobs

    dirs = [d for d in reviewable_jobs(jobs_dir) if d.resolve() not in exclude]
    return dirs[-1] if dirs else None  # reviewable_jobs is ascending by name -> last is newest


def _run_spec(spec: str, jobs_dir: Path, attempts: int, n_concurrent: int) -> Path:
    """Run a `provider/model` spec as the replica agent, `attempts` trials per task."""
    from ..harbor import run as run_mod

    typer.echo(f"running job for {spec} ({attempts} attempts/task) ...")
    return run_mod.run(jobs_dir.parent, AGENT_PATH, model=spec, jobs_dir=str(jobs_dir),
                       n_concurrent=n_concurrent,
                       extra_args=["--ak", "mode=replica", "--n-attempts", str(attempts)])


def _resolve(value: str, jobs_dir: Path, attempts: int, n_concurrent: int) -> list[Path]:
    """A job dir, a comma-separated list of job dirs, or a `provider/model` spec run first."""
    if "," in value:
        return [Path(p.strip()) for p in value.split(",") if p.strip()]
    if Path(value).is_dir():
        return [Path(value)]
    return [_run_spec(value, jobs_dir, attempts, n_concurrent)]


def _student_dirs(student: str, jobs_dir: Path, teacher_dirs: list[Path], attempts: int,
                  n_concurrent: int) -> list[Path]:
    """The student job dirs: what `--student` names, else the newest job that is not the teacher."""
    if student:
        return _resolve(student, jobs_dir, attempts, n_concurrent)
    newest = _newest_reviewable_dir(jobs_dir, {d.resolve() for d in teacher_dirs})
    if newest is None:
        _fail("no reviewable job (non-gate, with rewards) — run touchstone bench first")
    return [newest]


@app.command()
def train(
    jobs_dir: str = typer.Option("touchstone/jobs", "--jobs-dir",
                                 help="Directory holding job dirs (default <dataset>/jobs)."),
    teacher: str = typer.Option(None, "--teacher",
                                help="Teacher job dir(s) (comma-separated) or provider/model."),
    student: str = typer.Option(None, "--student",
                                help="Student job dir(s) or provider/model (default: newest job)."),
    attempts: int = typer.Option(3, "-k", "--attempts",
                                 help="Trials per task when running a provider/model spec."),
    out: str = typer.Option(None, "--out", help="Where to write (default <dataset>/train)."),
    threshold: float = typer.Option(1.0, "--threshold", help="Min reward a trial distills at."),
    n_concurrent: int = typer.Option(4, "-n", "--n-concurrent", help="Concurrent trials."),
) -> None:
    """Read Harbor jobs and write distillation + RL datasets a training stack consumes."""
    from ..harbor.jobs import Job, passing_trials
    from ..train.write import write_datasets

    jobs_path = Path(jobs_dir)
    _require_tasks(str(jobs_path.parent))
    with _daemon_guard():  # a teacher/student spec runs Harbor, which needs a Docker daemon
        teacher_dirs = _resolve(teacher, jobs_path, attempts, n_concurrent) if teacher else []
        student_dirs = _student_dirs(student, jobs_path, teacher_dirs, attempts, n_concurrent)

    student_jobs = [Job.read(d) for d in student_dirs]
    passing = sum(passing_trials(j) for j in student_jobs)
    typer.echo(f"training from job {student_dirs[-1].name} ({passing} passing trials)")
    out_path = Path(out) if out else jobs_path.parent / "train"
    written = write_datasets(out_path, [Job.read(d) for d in teacher_dirs],
                             student_jobs, threshold=threshold)
    typer.echo(written.sentence())
