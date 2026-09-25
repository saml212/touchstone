"""`touchstone survey <repo>` — map the code, simulate its services, write the dataset skeleton."""

from __future__ import annotations

import typer

from . import app
from ._common import _fail


@app.command()
def survey(
    repo: str = typer.Argument(..., help="Path to the repository to survey (read-only)."),
    force: bool = typer.Option(False, "--force", help="Rebuild outputs instead of reusing them."),
    provider: str = typer.Option(None, "--provider", help="Override [survey] provider."),
    model: str = typer.Option(None, "--model", help="Override [survey] model."),
    skip_gate: bool = typer.Option(False, "--skip-gate",
                                   help="Write tasks without running the oracle/nop gate."),
) -> None:
    """Read a repo's recordings and code and build its touchstone/ survey outputs."""
    from ..survey.survey import run_survey

    try:
        line = run_survey(repo, force=force, provider=provider or None, model=model or None,
                          skip_gate=skip_gate)
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"survey failed: {exc}")
    typer.echo(line)
