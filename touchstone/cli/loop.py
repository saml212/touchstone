"""`touchstone sample` / `touchstone distill` — the two buttons of the teacher-student loop."""

from __future__ import annotations

import typer

from . import app
from ._common import _agent_provider, _db, _fail_on, _root


@app.command()
def sample(
    target: str,
    student: str = typer.Option(..., "--student", help="Candidate model spec to evaluate."),
    teacher: str = typer.Option(None, "--teacher", help="Teacher spec (default: agent_provider)."),
    variants: int = typer.Option(1, "--variants", help="Variants per failing task."),
    concurrency: int = typer.Option(4, "--concurrency", help="Parallel replays."),
    timeout: float = typer.Option(60.0, "--timeout", help="Per-task timeout in seconds."),
    judge: str = typer.Option(None, "--judge", help="Provider spec for judge checks."),
) -> None:
    """Run the student, generate teacher variants of its failures, and print the frontier."""
    from ..loop import sample as run_sample

    judge_provider = _agent_provider(judge) if judge else None
    with _db() as conn, _fail_on(Exception, "sample failed: {exc}"):
        result = run_sample(conn, _root(), target, student, teacher_spec=teacher,
                            variants=variants, concurrency=concurrency, timeout=timeout,
                            judge_provider=judge_provider)
    _print_frontier(result)


def _print_frontier(result: dict) -> None:
    split = result["frontier_split"]
    typer.echo(f"student {result['student']} on {result['benchmark']} "
               f"(teacher {result['teacher']})")
    typer.echo(f"  ran {result['student_run']}; "
               f"variants created: {len(result['variants_created'])}")
    typer.echo(f"  frontier: {len(result['frontier'])} task(s) — "
               f"{len(split['learnability'])} in the learnability band, "
               f"{len(split['only_incumbent'])} only-incumbent")
    for name in result["frontier"]:
        typer.echo(f"    {name}")


@app.command()
def distill(
    target: str,
    student: str = typer.Option(..., "--student", help="Candidate model whose frontier to close."),
    teacher: str = typer.Option(None, "--teacher", help="Teacher spec (default: agent_provider)."),
    backend: str = typer.Option("null", "--backend", help="null | art | trl."),
    base_model: str = typer.Option(None, "--base-model", help="Base model for the backend config."),
    out: str = typer.Option(None, "--out", help="Output dir (default .touchstone/train/<target>)."),
) -> None:
    """Package the student's frontier into training data with teacher demos, then run a backend."""
    from ..loop import distill as run_distill

    with _db() as conn, _fail_on(Exception, "distill failed: {exc}"):
        result = run_distill(conn, _root(), target, student, teacher_spec=teacher,
                             backend=backend, base_model=base_model, out_dir=out)
    _print_distill(result)


def _print_distill(result: dict) -> None:
    typer.echo(f"distilled {result['student']} on {result['benchmark']} "
               f"(teacher {result['teacher']})")
    typer.echo(f"  frontier: {len(result['frontier'])} task(s); "
               f"teacher demos: {len(result['demos'])}")
    if result.get("escalated"):
        typer.echo(f"  escalated to interview rooms: {', '.join(result['escalated'])}")
    if result.get("out_dir"):
        typer.echo(f"  datasets in {result['out_dir']} ({result['status']})")
    typer.echo(f"  {result['plan']}")
