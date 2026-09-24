"""`touchstone checks` — manage the policies in checks.toml, and test a check against an output."""

from __future__ import annotations

import json

import typer

from .. import policies as policies_mod
from ..policies import Policy
from ._common import _fail, _fail_on, _judge_provider, _root

checks_app = typer.Typer(help="Manage policy checks (checks.toml) and test them.",
                         no_args_is_help=True)


@checks_app.command("list")
def checks_list(
    enabled: bool = typer.Option(None, help="Filter to enabled (--enabled) or disabled checks."),
) -> None:
    """List policy checks with their enabled flag, severity, kind, and source."""
    for p in policies_mod.read_policies(_root()):
        if enabled is not None and p.enabled != enabled:
            continue
        flag = "on " if p.enabled else "off"
        c = p.check
        typer.echo(f"[{flag}] {c.severity:4} {c.kind:16} {c.name} [{c.source}]")


@checks_app.command("add")
def checks_add(
    kind: str = typer.Option(..., help="Check kind, e.g. contains, tool_called, no_pii."),
    params: str = typer.Option("{}", help="Params as a JSON object."),
    name: str = typer.Option("", help="Human-readable name (its identity)."),
    severity: str = typer.Option("hard", help="hard | soft."),
    applies_to: str = typer.Option("final", help="final | any_turn | tool_calls."),
    enabled: bool = typer.Option(True, help="Enable the policy immediately."),
) -> None:
    """Add a manual policy to checks.toml after validating its params."""
    from ..checks import Check

    with _fail_on(json.JSONDecodeError, "--params is not valid JSON: {exc}"):
        parsed = json.loads(params)
    check = Check(kind=kind, params=parsed, name=name or kind, severity=severity,
                  applies_to=applies_to, source="manual")
    check.id = check.name
    with _fail_on(ValueError, "invalid check: {exc}"):
        check.validate()
    root = _root()
    if policies_mod.get_policy(root, check.name) is not None:
        _fail(f"a check named {check.name!r} already exists")
    policies_mod.add_policy(root, Policy(check=check, enabled=enabled))
    typer.echo(f"added check {check.name}")


@checks_app.command("enable")
def checks_enable(
    name: str = typer.Argument(None, help="Check name to enable."),
    all_mined: bool = typer.Option(False, "--all-mined", help="Enable every mined check."),
) -> None:
    """Enable a check by name, or every mined check with --all-mined."""
    root = _root()
    if all_mined:
        typer.echo(f"enabled {policies_mod.enable_all_mined(root)} mined check(s)")
        return
    if not name:
        _fail("give a check name or --all-mined")
    if not policies_mod.set_enabled(root, name, True):
        _fail(f"no check named {name!r}")
    typer.echo(f"enabled {name!r}")


@checks_app.command("disable")
def checks_disable(name: str) -> None:
    """Disable a policy check by name."""
    if not policies_mod.set_enabled(_root(), name, False):
        _fail(f"no check named {name!r}")
    typer.echo(f"disabled {name!r}")


@checks_app.command("show")
def checks_show(name: str) -> None:
    """Show one policy check's full definition."""
    p = policies_mod.get_policy(_root(), name)
    if p is None:
        _fail(f"no check named {name!r}")
    c = p.check
    typer.echo(f"name:       {c.name}")
    typer.echo(f"kind:       {c.kind}")
    typer.echo(f"severity:   {c.severity}")
    typer.echo(f"applies_to: {c.applies_to}")
    typer.echo(f"source:     {c.source}")
    typer.echo(f"enabled:    {p.enabled}")
    typer.echo(f"because:    {c.because}")
    typer.echo(f"params:     {json.dumps(c.params, ensure_ascii=False)}")


@checks_app.command("eval")
def checks_eval(
    name: str,
    text: str = typer.Option("", "--text", help="Output text to evaluate the check against."),
    tool_calls: str = typer.Option(
        "", "--tool-calls", help="Optional tool calls as a JSON list of {name, arguments}."
    ),
) -> None:
    """Evaluate one policy check against a supplied output for quick manual testing."""
    from ..checks import Target, coerce_tool_calls, evaluate

    p = policies_mod.get_policy(_root(), name)
    if p is None:
        _fail(f"no check named {name!r}")
    calls = []
    if tool_calls:
        try:
            calls = coerce_tool_calls(json.loads(tool_calls))
        except json.JSONDecodeError as exc:
            _fail(f"--tool-calls is not valid JSON: {exc}")
        except ValueError as exc:
            _fail(f"--tool-calls {exc}")
    target = Target(output_text=text, tool_calls=calls)
    provider = _judge_provider() if p.check.kind == "judge" else None
    result = evaluate([p.check], target, judge_provider=provider)[0]
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[result.passed]
    typer.echo(f"{verdict}  {result.evidence}")
