"""`touchstone train` — build training datasets and emit runnable backend configs."""

from __future__ import annotations

from pathlib import Path

import typer

from ..config import load_settings
from ._common import _db, _fail_on, _root

train_app = typer.Typer(help="Build training datasets and emit runnable backend configs.",
                        no_args_is_help=True)


def _out_dir(target: str, out: str | None) -> Path:
    from ..train import default_out_dir

    return Path(out) if out else default_out_dir(load_settings().db_path, target)


@train_app.command("prepare")
def train_prepare(
    target: str,
    out: str = typer.Option(None, "--out", help="Output dir (default .touchstone/train/<target>)."),
) -> None:
    """Write sft.jsonl, preference.jsonl, rl_tasks.jsonl and manifest.json for a benchmark."""
    from ..train import prepare

    with _db() as conn, _fail_on((OSError, ValueError), "could not prepare datasets: {exc}"):
        bundle = prepare(conn, _root(), target, _out_dir(target, out))
    c = bundle.counts
    typer.echo(f"wrote datasets to {bundle.out_dir}")
    typer.echo(f"  sft.jsonl        {c['sft']} rows")
    typer.echo(f"  preference.jsonl {c['preference']} rows")
    typer.echo(f"  rl_tasks.jsonl   {c['rl_tasks']} rows")


@train_app.command("submit")
def train_submit(
    target: str,
    backend: str = typer.Option("null", "--backend", help="null | art | trl."),
    out: str = typer.Option(None, "--out", help="Output dir (default .touchstone/train/<target>)."),
    base_model: str = typer.Option(None, "--base-model", help="Base model for the backend config."),
) -> None:
    """Prepare datasets then run a backend: null writes a plan; art/trl write a GPU config."""
    from ..train import InfraRequired, TrainConfig, trainer_for

    with _fail_on(ValueError):
        trainer = trainer_for(backend)
    config = TrainConfig(base_model=base_model) if base_model else TrainConfig()

    with _db() as conn, _fail_on((OSError, ValueError), "could not prepare datasets: {exc}"):
        bundle = trainer.prepare(conn, _root(), target, _out_dir(target, out))
        try:
            handle = trainer.submit(bundle, config)
        except InfraRequired as exc:
            typer.echo(f"prepared datasets in {bundle.out_dir}")
            typer.echo(str(exc))
            return
    typer.echo(f"prepared datasets in {bundle.out_dir}")
    typer.echo(f"{handle.status}: {handle.detail}")
