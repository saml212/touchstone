"""`touchstone survey <repo>` — map the code, simulate its services, write the dataset skeleton."""

from __future__ import annotations

import typer

from . import app
from ._common import _fail, _fail_on


@app.command()
def survey(
    repo: str = typer.Argument(..., help="Path to the repository to survey (read-only)."),
    force: bool = typer.Option(False, "--force", help="Rebuild outputs instead of reusing them."),
    provider: str = typer.Option(None, "--provider", help="Override [survey] provider."),
    model: str = typer.Option(None, "--model", help="Override [survey] model."),
    skip_gate: bool = typer.Option(False, "--skip-gate",
                                   help="Write tasks without running the oracle/nop gate."),
    rebaseline: bool = typer.Option(False, "--rebaseline",
                                    help="Re-run only the baseline (new model or upgrade)."),
    skip_baseline: bool = typer.Option(False, "--skip-baseline",
                                       help="Skip running the agent under test for the baseline."),
) -> None:
    """Read a repo's recordings and code and build its touchstone/ survey outputs."""
    from ..survey.survey import run_survey

    try:
        line = run_survey(repo, force=force, provider=provider or None, model=model or None,
                          skip_gate=skip_gate, skip_baseline=skip_baseline,
                         rebaseline=rebaseline)
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"survey failed: {exc}")
    typer.echo(line)


@app.command("survey-bench")
def survey_bench(
    config: str = typer.Argument("benchmarks/survey/targets.toml",
                                 help="TOML listing target repos to score."),
    out: str = typer.Option(".", "--out", help="Root under which benchmarks/survey/ is written."),
) -> None:
    """Score the survey agent across the targets in <config> (the survey's own scoreboard)."""
    from ..survey.benchmark import survey_bench as _run

    with _fail_on((FileNotFoundError, ValueError, KeyError, OSError), "survey-bench failed: {exc}"):
        doc = _run(config, out)
    typer.echo(f"scored {len(doc['targets'])} target(s); wrote benchmarks/survey/")
