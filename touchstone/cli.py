"""Touchstone CLI: init, doctor, demo, checks. Errors exit non-zero with one clear sentence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import typer

from . import store
from .config import DEFAULT_DB, load_settings

app = typer.Typer(
    help="Touchstone — prove a cheaper model is good enough before you switch.",
    no_args_is_help=True,
)

TOML_TEMPLATE = f"""# Touchstone config
db_path = "{DEFAULT_DB}"
provider = "scripted"
keychain_prefix = "rockie-"

[speech]
stt = "none"
tts = "browser"
"""


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.command()
def init(path: str = typer.Option("touchstone.toml", help="Config file to write.")) -> None:
    """Write touchstone.toml and create the .touchstone/ db directory."""
    cfg = Path(path)
    if cfg.exists():
        typer.echo(f"{cfg} already exists; leaving it untouched.")
    else:
        cfg.write_text(TOML_TEMPLATE, encoding="utf-8")
        typer.echo(f"wrote {cfg}")
    settings = load_settings(cfg)
    store.connect(settings.db_path).close()
    typer.echo(f"initialized db at {settings.db}")


@app.command()
def doctor() -> None:
    """Report environment: python, db, importable SDKs, provider availability (no network calls)."""
    from .llm import provider_statuses

    settings = load_settings()
    typer.echo(f"python:    {sys.version.split()[0]}")
    typer.echo(f"db path:   {settings.db}")
    typer.echo(f"db exists: {settings.db.exists()}")
    for name in ("openai", "anthropic", "litellm"):
        typer.echo(f"{name+':':10} {'importable' if _installed(name) else 'not installed'}")
    typer.echo("providers:")
    for st in provider_statuses(settings):
        mark = "ok" if st.ok else "missing"
        typer.echo(f"  {st.name:16} {mark:8} {st.detail}")


@app.command()
def demo(
    n: int = typer.Option(30, help="Number of episodes to generate."),
    db: str = typer.Option(None, help="Override db path."),
) -> None:
    """Run the built-in scripted support agent and capture real episodes."""
    import touchstone

    from .demo import run_demo

    settings = load_settings()
    db_path = db or settings.db_path
    touchstone.trace(db_path)
    run_demo(n=n)
    conn = store.connect(db_path)
    episodes = store.list_episodes(conn)
    resolved = len(store.list_episodes(conn, "resolved"))
    spans = sum(len(store.list_spans(conn, e.id)) for e in episodes)
    conn.close()
    typer.echo(
        f"captured {len(episodes)} episodes ({resolved} resolved), {spans} spans in {settings.db}"
    )


checks_app = typer.Typer(help="Manage and test checks.", no_args_is_help=True)
app.add_typer(checks_app, name="checks")


def _open_db():
    return store.connect(load_settings().db_path)


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


@checks_app.command("list")
def checks_list(
    enabled: bool = typer.Option(None, help="Filter to enabled (--enabled) or disabled checks."),
) -> None:
    """List checks with their id, enabled flag, severity, and kind."""
    conn = _open_db()
    try:
        for c in store.list_checks(conn, enabled):
            flag = "on " if c.enabled else "off"
            typer.echo(f"{c.id}  [{flag}] {c.severity:4} {c.kind:16} {c.name}")
    finally:
        conn.close()


@checks_app.command("add")
def checks_add(
    kind: str = typer.Option(..., help="Check kind, e.g. contains, tool_called, no_pii."),
    params: str = typer.Option("{}", help="Params as a JSON object."),
    name: str = typer.Option("", help="Human-readable name."),
    severity: str = typer.Option("hard", help="hard | soft."),
    applies_to: str = typer.Option("final", help="final | any_turn | tool_calls."),
    enabled: bool = typer.Option(True, help="Enable the check immediately."),
) -> None:
    """Add a manual check after validating its params."""
    from .checks import Check as DslCheck

    try:
        parsed = json.loads(params)
    except json.JSONDecodeError as exc:
        _fail(f"--params is not valid JSON: {exc}")
    try:
        DslCheck(kind=kind, params=parsed, severity=severity, applies_to=applies_to).validate()
    except ValueError as exc:
        _fail(f"invalid check: {exc}")
    conn = _open_db()
    try:
        check = store.insert_check(
            conn,
            store.Check(
                name=name or kind,
                kind=kind,
                params=parsed,
                applies_to=applies_to,
                severity=severity,
                source="manual",
                enabled=1 if enabled else 0,
            ),
        )
    finally:
        conn.close()
    typer.echo(f"added check {check.id}")


@checks_app.command("enable")
def checks_enable(check_id: str) -> None:
    """Enable a check."""
    _set_enabled(check_id, True)


@checks_app.command("disable")
def checks_disable(check_id: str) -> None:
    """Disable a check."""
    _set_enabled(check_id, False)


def _set_enabled(check_id: str, enabled: bool) -> None:
    conn = _open_db()
    try:
        if store.get_check(conn, check_id) is None:
            _fail(f"no check with id {check_id}")
        store.set_check_enabled(conn, check_id, enabled)
    finally:
        conn.close()
    typer.echo(f"{'enabled' if enabled else 'disabled'} {check_id}")


@checks_app.command("show")
def checks_show(check_id: str) -> None:
    """Show one check's full definition."""
    conn = _open_db()
    try:
        check = store.get_check(conn, check_id)
    finally:
        conn.close()
    if check is None:
        _fail(f"no check with id {check_id}")
    typer.echo(f"id:         {check.id}")
    typer.echo(f"name:       {check.name}")
    typer.echo(f"kind:       {check.kind}")
    typer.echo(f"severity:   {check.severity}")
    typer.echo(f"applies_to: {check.applies_to}")
    typer.echo(f"source:     {check.source}")
    typer.echo(f"enabled:    {bool(check.enabled)}")
    typer.echo(f"params:     {json.dumps(check.params, ensure_ascii=False)}")


@checks_app.command("eval")
def checks_eval(
    check_id: str,
    text: str = typer.Option("", "--text", help="Output text to evaluate the check against."),
    tool_calls: str = typer.Option(
        "", "--tool-calls", help="Optional tool calls as a JSON list of {name, arguments}."
    ),
) -> None:
    """Evaluate one stored check against a supplied output for quick manual testing."""
    from .checks import Check as DslCheck
    from .checks import Target, evaluate

    conn = _open_db()
    try:
        row = store.get_check(conn, check_id)
    finally:
        conn.close()
    if row is None:
        _fail(f"no check with id {check_id}")
    calls = []
    if tool_calls:
        try:
            calls = json.loads(tool_calls)
        except json.JSONDecodeError as exc:
            _fail(f"--tool-calls is not valid JSON: {exc}")
    target = Target(output_text=text, tool_calls=calls)
    provider = _judge_provider() if row.kind == "judge" else None
    check = DslCheck.from_dict(
        {"kind": row.kind, "params": row.params, "id": row.id, "name": row.name,
         "applies_to": row.applies_to, "severity": row.severity}
    )
    result = evaluate([check], target, judge_provider=provider)[0]
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[result.passed]
    typer.echo(f"{verdict}  {result.evidence}")


def _judge_provider():
    from .llm import provider_from_spec

    try:
        return provider_from_spec(load_settings().provider)
    except Exception:  # a judge without a working provider degrades to 'skipped'
        return None


if __name__ == "__main__":
    app()
