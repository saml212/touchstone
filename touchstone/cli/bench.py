"""`touchstone bench` runs a model over the dataset and prints a scoreboard; `touchstone jobs`
lists past job directories with their pass rate per task."""

from __future__ import annotations

from pathlib import Path

import typer

from . import app
from ._common import _daemon_guard, _require_tasks

AGENT_PATH = "touchstone.harbor.agent:TouchstoneAgent"
_MODES = ("packaged", "replica")


def _print_rates(rates: dict[str, float | str], errors: dict[str, int] | None = None) -> None:
    if not rates:
        typer.echo("(no trials)")
        return
    width = max(len(t) for t in rates)
    if errors:
        typer.echo(f"  {'task'.ljust(width)}  rate    errors")
    for task, rate in rates.items():
        cell = f"{rate * 100:5.1f}%" if isinstance(rate, (int, float)) else rate
        suffix = f"   {errors.get(task, 0)}" if errors else ""
        typer.echo(f"  {task.ljust(width)}  {cell}{suffix}")
    numeric = [v for v in rates.values() if isinstance(v, (int, float))]
    if numeric:
        typer.echo(f"  {'overall'.ljust(width)}  {sum(numeric) / len(numeric) * 100:5.1f}%")


def _print_compare(comparison: dict[str, list[str]]) -> None:
    labels = {"both": "pass in both", "only_a": "only this run",
              "only_b": "only the other", "neither": "fail in both"}
    typer.echo("\nvs --against:")
    for key, label in labels.items():
        typer.echo(f"  {label}: {len(comparison[key])}")


@app.command()
def bench(
    model: str = typer.Option(..., "-m", "--model", help="Model as provider/model."),
    agent: str = typer.Option("replica", "--agent", help="packaged | replica."),
    dataset: str = typer.Option("touchstone", "--dataset", help="Dataset root or a task dir."),
    against: str = typer.Option(None, "--against", help="A past job dir to compare against."),
    jobs_dir: str = typer.Option(None, "--jobs-dir",
                                 help="Where Harbor writes job dirs (default <dataset>/jobs)."),
    n_concurrent: int = typer.Option(4, "-n", "--n-concurrent", help="Concurrent trials."),
) -> None:
    """Run a model over the dataset with Harbor and print the pass rate per task."""
    from ..config import load_settings
    from ..harbor import jobs as jobs_mod
    from ..harbor import run as run_mod

    if agent not in _MODES:
        raise typer.BadParameter("agent must be 'packaged' or 'replica'")
    _require_tasks(dataset)
    settings = load_settings()
    extra = ["--ak", f"mode={agent}"]
    user_model = None
    if run_mod.dataset_is_multi_turn(dataset):  # conversational tasks need Harbor's simulated user
        user_model = settings.survey_user_model or model
        extra += run_mod.simulated_user_args(settings.survey_user_agent, user_model)
    with _daemon_guard():
        job_dir = run_mod.run(dataset, AGENT_PATH, model=model, user_model=user_model,
                              jobs_dir=jobs_dir or f"{dataset}/jobs",
                              n_concurrent=n_concurrent, extra_args=extra, settings=settings)
    job = jobs_mod.Job.read(job_dir)
    typer.echo(f"\n{job_dir}")
    _report(job, jobs_mod)
    if against:
        _print_compare(jobs_mod.compare(job, jobs_mod.Job.read(against)))


def _report(job, jobs_mod) -> None:
    """Print the scoreboard — or, when every trial errored, scream the reason and exit 1 so a run
    that produced nothing can never read as a clean 0%."""
    if jobs_mod.all_errored(job):
        n = len(job.trials)
        typer.echo(f"{n} of {n} trials errored — first: {jobs_mod.first_error(job)}", err=True)
        raise typer.Exit(1)
    errors = jobs_mod.task_error_counts(job)
    _print_rates(jobs_mod.task_outcomes(job), errors or None)


@app.command()
def jobs(
    jobs_dir: str = typer.Option("touchstone/jobs", "--jobs-dir",
                                 help="Directory holding job dirs (default <dataset>/jobs)."),
) -> None:
    """List job directories, each with its pass rate per task."""
    from ..harbor import jobs as jobs_mod

    _require_tasks(str(Path(jobs_dir).parent))
    root = Path(jobs_dir)
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if not dirs:
        typer.echo(f"no job directories under {root}")
        return
    for d in dirs:
        typer.echo(f"\n{d.name}")
        job = jobs_mod.Job.read(d)
        _print_rates(jobs_mod.task_outcomes(job), jobs_mod.task_error_counts(job) or None)
