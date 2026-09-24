"""`touchstone bench` and `touchstone export` — benchmarks, runs, reports, and exports."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from .. import store
from ._common import _agent_provider, _db, _fail_on

bench_app = typer.Typer(help="Assemble benchmarks, run models, and report results.",
                        no_args_is_help=True)


@bench_app.command("create")
def bench_create(
    name: str,
    tag: Annotated[list[str], typer.Option("--tag", help="Task tag (repeatable).")] = None,
    all_tasks: bool = typer.Option(False, "--all", help="Include every task."),
    task_id: Annotated[list[str], typer.Option("--task", help="Task id (repeatable).")] = None,
) -> None:
    """Freeze a named benchmark from explicit task ids, tag filters, or --all."""
    from ..bench import benchmark

    with _db() as conn, _fail_on(ValueError):
        bench = benchmark.create(conn, name, task_ids=task_id or None,
                                 tags=tag or None, all_tasks=all_tasks)
    typer.echo(f"created benchmark {bench.id} ({len(bench.task_ids)} tasks)")


@bench_app.command("run")
def bench_run(
    benchmark_id: str,
    model: Annotated[list[str], typer.Option("-m", "--model", help="Model spec (repeatable).")],
    concurrency: int = typer.Option(4, "--concurrency", help="Parallel replays per model."),
    judge: str = typer.Option(None, "--judge", help="Provider spec for judge checks."),
    timeout: float = typer.Option(60.0, "--timeout", help="Per-task timeout in seconds."),
) -> None:
    """Replay a benchmark against each model spec in turn, then print the scoreboard."""
    from ..bench import render_scoreboard, runner, scoreboard

    judge_provider = _agent_provider(judge) if judge else None
    with _db() as conn:
        run_ids = []
        for spec in model:
            with _fail_on(Exception, f"run failed for {spec!r}: {{exc}}"):
                run = runner.run(conn, benchmark_id, spec, concurrency=concurrency,
                                 timeout=timeout, judge_provider=judge_provider)
            run_ids.append(run.id)
            typer.echo(f"ran {spec} -> {run.id}")
        typer.echo("")
        typer.echo(render_scoreboard(scoreboard(conn, run_ids)))


@bench_app.command("report")
def bench_report(
    run_id: Annotated[list[str], typer.Argument(help="Run ids (default: all runs).")] = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of tables."),
) -> None:
    """Show a scoreboard over the given runs (or all runs)."""
    from ..bench import render_scoreboard, scoreboard

    with _db() as conn:
        ids = list(run_id) if run_id else [r.id for r in store.list_runs(conn)]
        board = scoreboard(conn, ids)
    typer.echo(json.dumps(board, ensure_ascii=False, indent=2) if as_json
               else render_scoreboard(board))


@bench_app.command("proof")
def bench_proof(
    candidate_run: str,
    incumbent_run: str,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Diff a candidate run against an incumbent run task-by-task."""
    from ..bench import proof, render_proof

    with _db() as conn, _fail_on(ValueError):
        report = proof(conn, candidate_run, incumbent_run)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2) if as_json
               else render_proof(report))


@bench_app.command("runs")
def bench_runs() -> None:
    """List runs with their model, benchmark, and finished state."""
    with _db() as conn:
        for r in store.list_runs(conn):
            state = "done" if r.finished_at else "unfinished"
            typer.echo(f"{r.id}  {state:10} {r.model_spec:28} bench={r.benchmark_id}")


@bench_app.command("harbor-run")
def bench_harbor_run(
    task_dir: str,
    agent: str = typer.Option(None, "--agent", "-a", help="Harbor agent name."),
) -> None:
    """Run an exported Harbor task dir via `harbor run` (or explain what's missing)."""
    from ..bench import harbor_run_task

    code = harbor_run_task(task_dir, agent=agent, echo=typer.echo)
    raise typer.Exit(code)


export_app = typer.Typer(help="Export tasks to Harbor or an episode to ATIF.",
                         no_args_is_help=True)


@export_app.command("harbor")
def export_harbor(
    benchmark_id: str,
    out: str = typer.Option("./harbor-tasks", "--out", help="Output directory."),
) -> None:
    """Write one Harbor task directory per task in the benchmark."""
    from ..bench import export as export_harbor_tasks

    with _db() as conn, _fail_on(ValueError):
        dirs = export_harbor_tasks(conn, benchmark_id, out)
    typer.echo(f"exported {len(dirs)} task(s) to {out}")


@export_app.command("atif")
def export_atif_cmd(
    episode_id: str,
    out: str = typer.Option(None, "--out", help="Output file (default <episode>.atif.json)."),
) -> None:
    """Export one captured episode to a Harbor ATIF trajectory JSON."""
    from ..capture import export_atif

    path = out or f"{episode_id}.atif.json"
    with _db() as conn, _fail_on(ValueError):
        export_atif(conn, episode_id, path)
    typer.echo(f"wrote {path}")
