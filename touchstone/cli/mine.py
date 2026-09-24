"""`touchstone mine` — propose checks from captured episodes and cut replay tasks."""

from __future__ import annotations

import typer

from . import app
from ._common import _agent_provider, _db


@app.command()
def mine(
    code: str = typer.Option(None, "--code", help="Path to a code tree to scan for prompts/tools."),
    provider: str = typer.Option(None, "--provider", help="Agent provider spec for LLM proposals."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM pass; statistics only."),
    limit: int = typer.Option(None, "--limit", help="Only mine the first N episodes."),
) -> None:
    """Propose checks from captured episodes and cut replay tasks."""
    from ..config import load_settings
    from ..mine import mine as run_mine
    from ..mine import scan_codebase

    settings = load_settings()
    snippets = scan_codebase(code) if code else []
    prov = None
    if not no_llm:
        prov = _agent_provider(provider or settings.agent_provider)
    with _db() as conn:
        summary = run_mine(
            conn, provider=prov, code_snippets=snippets,
            no_llm=no_llm, limit=limit,
        )

    proposals = summary["proposals"]
    if proposals:
        typer.echo(f"{'KIND':16} {'SEV':4} {'SUP':>4}  RATIONALE")
        for p in proposals:
            head = (p.rationale or "")[:60]
            typer.echo(f"{p.kind:16} {p.severity:4} {p.support_count:>4}  {head}")
    typer.echo(
        f"proposed {summary['inserted']} check(s) "
        f"({summary['stats']} stats, {summary['llm']} llm), cut {summary['tasks_cut']} task(s)"
    )
