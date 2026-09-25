"""`touchstone bench` runs a model over the dataset and prints a scoreboard; `touchstone jobs`
lists past job directories with their pass rate per task."""

from __future__ import annotations

from pathlib import Path

import typer

from . import app

_AGENTS = {"packaged": "touchstone.harbor.agent:TouchstoneAgent",
           "replica": "touchstone.harbor.agent:TouchstoneAgent"}


def _print_rates(rates: dict[str, float]) -> None:
    if not rates:
        typer.echo("(no trials)")
        return
    width = max(len(t) for t in rates)
    for task, rate in rates.items():
        typer.echo(f"  {task.ljust(width)}  {rate * 100:5.1f}%")
    mean = sum(rates.values()) / len(rates)
    typer.echo(f"  {'overall'.ljust(width)}  {mean * 100:5.1f}%")


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
    jobs_dir: str = typer.Option("jobs", "--jobs-dir", help="Where Harbor writes job dirs."),
    n_concurrent: int = typer.Option(4, "-n", "--n-concurrent", help="Concurrent trials."),
) -> None:
    """Run a model over the dataset with Harbor and print the pass rate per task."""
    from ..harbor import jobs as jobs_mod
    from ..harbor import run as run_mod

    if agent not in _AGENTS:
        raise typer.BadParameter("agent must be 'packaged' or 'replica'")
    job_dir = run_mod.run(dataset, _AGENTS[agent], model=model, jobs_dir=jobs_dir,
                          n_concurrent=n_concurrent)
    job = jobs_mod.Job.read(job_dir)
    typer.echo(f"\n{job_dir}")
    _print_rates(jobs_mod.pass_rates(job))
    if against:
        _print_compare(jobs_mod.compare(job, jobs_mod.Job.read(against)))


@app.command()
def jobs(
    jobs_dir: str = typer.Option("jobs", "--jobs-dir", help="Directory holding job dirs."),
) -> None:
    """List job directories, each with its pass rate per task."""
    from ..harbor import jobs as jobs_mod

    root = Path(jobs_dir)
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if not dirs:
        typer.echo(f"no job directories under {root}")
        return
    for d in dirs:
        typer.echo(f"\n{d.name}")
        _print_rates(jobs_mod.pass_rates(jobs_mod.Job.read(d)))
