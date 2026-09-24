"""`touchstone checks` — list, add, enable, disable, show, and eval checks."""

from __future__ import annotations

import json

import typer

from .. import store
from ._common import _db, _fail, _fail_on, _judge_provider

checks_app = typer.Typer(help="Manage and test checks.", no_args_is_help=True)


@checks_app.command("list")
def checks_list(
    enabled: bool = typer.Option(None, help="Filter to enabled (--enabled) or disabled checks."),
) -> None:
    """List checks with their id, enabled flag, severity, and kind."""
    with _db() as conn:
        for c in store.list_checks(conn, enabled):
            flag = "on " if c.enabled else "off"
            typer.echo(f"{c.id}  [{flag}] {c.severity:4} {c.kind:16} {c.name}")


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
    from ..checks import Check as DslCheck

    with _fail_on(json.JSONDecodeError, "--params is not valid JSON: {exc}"):
        parsed = json.loads(params)
    with _fail_on(ValueError, "invalid check: {exc}"):
        DslCheck(kind=kind, params=parsed, severity=severity, applies_to=applies_to).validate()
    with _db() as conn:
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
    typer.echo(f"added check {check.id}")


@checks_app.command("enable")
def checks_enable(
    check_id: str = typer.Argument(None, help="Check id to enable."),
    all_mined: bool = typer.Option(False, "--all-mined", help="Enable every mined check."),
) -> None:
    """Enable a check by id, or every mined check with --all-mined."""
    if all_mined:
        with _db() as conn:
            mined = [c for c in store.list_checks(conn) if c.source == "mined" and not c.enabled]
            for c in mined:
                store.set_check_enabled(conn, c.id, True)
        typer.echo(f"enabled {len(mined)} mined check(s)")
        return
    if not check_id:
        _fail("give a check id or --all-mined")
    _set_enabled(check_id, True)


@checks_app.command("disable")
def checks_disable(check_id: str) -> None:
    """Disable a check."""
    _set_enabled(check_id, False)


def _set_enabled(check_id: str, enabled: bool) -> None:
    with _db() as conn:
        if store.get_check(conn, check_id) is None:
            _fail(f"no check with id {check_id}")
        store.set_check_enabled(conn, check_id, enabled)
    typer.echo(f"{'enabled' if enabled else 'disabled'} {check_id}")


@checks_app.command("show")
def checks_show(check_id: str) -> None:
    """Show one check's full definition."""
    with _db() as conn:
        check = store.get_check(conn, check_id)
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
    from ..checks import Check as DslCheck
    from ..checks import Target, coerce_tool_calls, evaluate

    with _db() as conn:
        row = store.get_check(conn, check_id)
    if row is None:
        _fail(f"no check with id {check_id}")
    calls = []
    if tool_calls:
        try:
            calls = coerce_tool_calls(json.loads(tool_calls))
        except json.JSONDecodeError as exc:
            _fail(f"--tool-calls is not valid JSON: {exc}")
        except ValueError as exc:
            _fail(f"--tool-calls {exc}")
    target = Target(output_text=text, tool_calls=calls)
    provider = _judge_provider() if row.kind == "judge" else None
    check = DslCheck.from_dict(
        {"kind": row.kind, "params": row.params, "id": row.id, "name": row.name,
         "applies_to": row.applies_to, "severity": row.severity}
    )
    result = evaluate([check], target, judge_provider=provider)[0]
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[result.passed]
    typer.echo(f"{verdict}  {result.evidence}")
