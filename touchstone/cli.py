"""Touchstone CLI: init, doctor, demo, checks. Errors exit non-zero with one clear sentence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Annotated

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
{{keychain_prefix}}
[speech]
stt = "none"
tts = "browser"
"""


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.command()
def init(
    path: str = typer.Option("touchstone.toml", help="Config file to write."),
    keychain_prefix: str = typer.Option(
        None, "--keychain-prefix",
        help="Pin the macOS Keychain service prefix (default: touchstone-).",
    ),
) -> None:
    """Write touchstone.toml and create the .touchstone/ db directory."""
    cfg = Path(path)
    if cfg.exists():
        typer.echo(f"{cfg} already exists; leaving it untouched.")
    else:
        prefix_line = f'keychain_prefix = "{keychain_prefix}"\n' if keychain_prefix else ""
        cfg.write_text(TOML_TEMPLATE.format(keychain_prefix=prefix_line), encoding="utf-8")
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

    from .interview.speech import speech_status

    typer.echo("speech:")
    for r in speech_status(settings):
        typer.echo(f"  {r.kind:4} {r.name:16} {r.detail}")


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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8765, help="Bind port."),
) -> None:
    """Run the local interview + review server."""
    import uvicorn

    from .server import create_app

    typer.echo(f"Touchstone UI on http://{host}:{port}")
    uvicorn.run(create_app(load_settings()), host=host, port=port)


@app.command()
def interview(
    task_id: str,
    topic: str = typer.Option(None, "--topic", help="Room topic (defaults to the task name)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open an interview room for a task and print (and open) its URL."""
    from .interview import rooms
    from .interview.agent import Interviewer

    host, port = "127.0.0.1", 8765
    conn = _open_db()
    try:
        task = store.get_task(conn, task_id)
        if task is None:
            _fail(f"no task with id {task_id}")
        room = rooms.open(conn, task_id, topic or f"review of {task.name}")
        rooms.post(conn, room.id, "agent", "assistant",
                   Interviewer(None, conn, room).open_statement())
    finally:
        conn.close()

    url = f"http://{host}:{port}/rooms/{room.id}"
    typer.echo(url)
    if not _server_up(host, port):
        typer.echo(f"Server not running — start it with `touchstone serve`, then open {url}.")
        return
    if not no_open:
        import webbrowser

        webbrowser.open(url)


def _server_up(host: str, port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://{host}:{port}/api/health", timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


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
def checks_enable(
    check_id: str = typer.Argument(None, help="Check id to enable."),
    all_mined: bool = typer.Option(False, "--all-mined", help="Enable every mined check."),
) -> None:
    """Enable a check by id, or every mined check with --all-mined."""
    if all_mined:
        conn = _open_db()
        try:
            mined = [c for c in store.list_checks(conn) if c.source == "mined" and not c.enabled]
            for c in mined:
                store.set_check_enabled(conn, c.id, True)
        finally:
            conn.close()
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


@app.command()
def mine(
    code: str = typer.Option(None, "--code", help="Path to a code tree to scan for prompts/tools."),
    provider: str = typer.Option(None, "--provider", help="Agent provider spec for LLM proposals."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM pass; statistics only."),
    limit: int = typer.Option(None, "--limit", help="Only mine the first N episodes."),
) -> None:
    """Propose checks from captured episodes and cut replay tasks."""
    from .mine import mine as run_mine
    from .mine import scan_codebase

    settings = load_settings()
    snippets = scan_codebase(code) if code else []
    prov = None
    if not no_llm:
        prov = _agent_provider(provider or settings.agent_provider)
    conn = _open_db()
    try:
        summary = run_mine(
            conn, provider=prov, code_snippets=snippets,
            no_llm=no_llm, limit=limit,
        )
    finally:
        conn.close()

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


def _agent_provider(spec: str):
    from .llm import provider_from_spec

    try:
        return provider_from_spec(spec)
    except Exception as exc:  # no key / binary: mine on statistics alone
        typer.echo(f"note: provider {spec!r} unavailable ({exc}); mining stats only", err=True)
        return None


tasks_app = typer.Typer(help="Inspect and edit replay tasks.", no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("list")
def tasks_list(tag: str = typer.Option(None, "--tag", help="Filter by tag.")) -> None:
    """List tasks with their tags and attached-check counts."""
    conn = _open_db()
    try:
        for t in store.list_tasks(conn, tag):
            tags = ",".join(t.tags or [])
            typer.echo(f"{t.id}  [{len(t.check_ids or [])} checks]  {t.name}  ({tags})")
    finally:
        conn.close()


@tasks_app.command("show")
def tasks_show(task_id: str) -> None:
    """Show one task: context turns, reference, and attached checks."""
    conn = _open_db()
    try:
        task = store.get_task(conn, task_id)
        if task is None:
            _fail(f"no task with id {task_id}")
        typer.echo(f"id:     {task.id}")
        typer.echo(f"name:   {task.name}")
        typer.echo(f"tags:   {', '.join(task.tags or [])}")
        typer.echo("context:")
        for m in (task.context or {}).get("messages", []):
            content = m.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            typer.echo(f"  {m.get('role', '?'):9} {text[:100]}")
        typer.echo(f"reference: {json.dumps(task.reference, ensure_ascii=False)}")
        typer.echo("checks:")
        for cid in task.check_ids or []:
            c = store.get_check(conn, cid)
            typer.echo(f"  {cid}  {c.kind if c else '(missing)'}")
    finally:
        conn.close()


@tasks_app.command("attach")
def tasks_attach(task_id: str, check_id: str) -> None:
    """Attach a check to a task."""
    _edit_task_checks(task_id, check_id, attach=True)


@tasks_app.command("detach")
def tasks_detach(task_id: str, check_id: str) -> None:
    """Detach a check from a task."""
    _edit_task_checks(task_id, check_id, attach=False)


def _edit_task_checks(task_id: str, check_id: str, attach: bool) -> None:
    conn = _open_db()
    try:
        store.set_task_check(conn, task_id, check_id, attach)
    except ValueError as exc:
        _fail(str(exc))
    finally:
        conn.close()
    verb, prep = ("attached", "to") if attach else ("detached", "from")
    typer.echo(f"{verb} {check_id} {prep} {task_id}")


# ---- bench -----------------------------------------------------------------

bench_app = typer.Typer(help="Assemble benchmarks, run models, and report results.",
                        no_args_is_help=True)
app.add_typer(bench_app, name="bench")


@bench_app.command("create")
def bench_create(
    name: str,
    tag: Annotated[list[str], typer.Option("--tag", help="Task tag (repeatable).")] = None,
    all_tasks: bool = typer.Option(False, "--all", help="Include every task."),
    task_id: Annotated[list[str], typer.Option("--task", help="Task id (repeatable).")] = None,
) -> None:
    """Freeze a named benchmark from explicit task ids, tag filters, or --all."""
    from .bench import benchmark

    conn = _open_db()
    try:
        bench = benchmark.create(conn, name, task_ids=task_id or None,
                                 tags=tag or None, all_tasks=all_tasks)
    except ValueError as exc:
        _fail(str(exc))
    finally:
        conn.close()
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
    from .bench import render_scoreboard, runner, scoreboard

    judge_provider = _agent_provider(judge) if judge else None
    conn = _open_db()
    try:
        run_ids = []
        for spec in model:
            try:
                run = runner.run(conn, benchmark_id, spec, concurrency=concurrency,
                                 timeout=timeout, judge_provider=judge_provider)
            except Exception as exc:
                _fail(f"run failed for {spec!r}: {exc}")
            run_ids.append(run.id)
            typer.echo(f"ran {spec} -> {run.id}")
        typer.echo("")
        typer.echo(render_scoreboard(scoreboard(conn, run_ids)))
    finally:
        conn.close()


@bench_app.command("report")
def bench_report(
    run_id: Annotated[list[str], typer.Argument(help="Run ids (default: all runs).")] = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of tables."),
) -> None:
    """Show a scoreboard over the given runs (or all runs)."""
    from .bench import render_scoreboard, scoreboard

    conn = _open_db()
    try:
        ids = list(run_id) if run_id else [r.id for r in store.list_runs(conn)]
        board = scoreboard(conn, ids)
    finally:
        conn.close()
    typer.echo(json.dumps(board, ensure_ascii=False, indent=2) if as_json
               else render_scoreboard(board))


@bench_app.command("proof")
def bench_proof(
    candidate_run: str,
    incumbent_run: str,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Diff a candidate run against an incumbent run task-by-task."""
    from .bench import proof, render_proof

    conn = _open_db()
    try:
        report = proof(conn, candidate_run, incumbent_run)
    finally:
        conn.close()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2) if as_json
               else render_proof(report))


@bench_app.command("runs")
def bench_runs() -> None:
    """List runs with their model, benchmark, and finished state."""
    conn = _open_db()
    try:
        for r in store.list_runs(conn):
            state = "done" if r.finished_at else "unfinished"
            typer.echo(f"{r.id}  {state:10} {r.model_spec:28} bench={r.benchmark_id}")
    finally:
        conn.close()


@bench_app.command("harbor-run")
def bench_harbor_run(
    task_dir: str,
    agent: str = typer.Option(None, "--agent", "-a", help="Harbor agent name."),
) -> None:
    """Run an exported Harbor task dir via `harbor run` (or explain what's missing)."""
    from .bench import harbor_run_task

    code = harbor_run_task(task_dir, agent=agent, echo=typer.echo)
    raise typer.Exit(code)


# ---- train -----------------------------------------------------------------

train_app = typer.Typer(help="Build training datasets and emit runnable backend configs.",
                        no_args_is_help=True)
app.add_typer(train_app, name="train")


def _resolve_out(conn, benchmark_id: str, out: str | None):
    from .train import default_out_dir

    bench = store.get_benchmark(conn, benchmark_id)
    if bench is None:
        _fail(f"no benchmark {benchmark_id!r}")
    return bench, (Path(out) if out else default_out_dir(load_settings().db_path, bench))


@train_app.command("prepare")
def train_prepare(
    benchmark_id: str,
    out: str = typer.Option(None, "--out", help="Output dir (default .touchstone/train/<bench>)."),
) -> None:
    """Write sft.jsonl, preference.jsonl, rl_tasks.jsonl and manifest.json for a benchmark."""
    from .train import prepare

    conn = _open_db()
    try:
        bench, out_dir = _resolve_out(conn, benchmark_id, out)
        bundle = prepare(conn, bench.id, out_dir)
    finally:
        conn.close()
    c = bundle.counts
    typer.echo(f"wrote datasets to {bundle.out_dir}")
    typer.echo(f"  sft.jsonl        {c['sft']} rows")
    typer.echo(f"  preference.jsonl {c['preference']} rows")
    typer.echo(f"  rl_tasks.jsonl   {c['rl_tasks']} rows")


@train_app.command("submit")
def train_submit(
    benchmark_id: str,
    backend: str = typer.Option("null", "--backend", help="null | art | trl."),
    out: str = typer.Option(None, "--out", help="Output dir (default .touchstone/train/<bench>)."),
    base_model: str = typer.Option(None, "--base-model", help="Base model for the backend config."),
) -> None:
    """Prepare datasets then run a backend: null writes a plan; art/trl write a GPU config."""
    from .train import InfraRequired, TrainConfig, trainer_for

    try:
        trainer = trainer_for(backend)
    except ValueError as exc:
        _fail(str(exc))
    config = TrainConfig(base_model=base_model) if base_model else TrainConfig()

    conn = _open_db()
    try:
        bench, out_dir = _resolve_out(conn, benchmark_id, out)
        bundle = trainer.prepare(conn, bench.id, out_dir)
        try:
            handle = trainer.submit(bundle, config)
        except InfraRequired as exc:
            typer.echo(f"prepared datasets in {bundle.out_dir}")
            typer.echo(str(exc))
            return
    finally:
        conn.close()
    typer.echo(f"prepared datasets in {bundle.out_dir}")
    typer.echo(f"{handle.status}: {handle.detail}")


# ---- export ----------------------------------------------------------------

export_app = typer.Typer(help="Export tasks to Harbor or an episode to ATIF.",
                         no_args_is_help=True)
app.add_typer(export_app, name="export")


@export_app.command("harbor")
def export_harbor(
    benchmark_id: str,
    out: str = typer.Option("./harbor-tasks", "--out", help="Output directory."),
) -> None:
    """Write one Harbor task directory per task in the benchmark."""
    from .bench import export as export_harbor_tasks

    conn = _open_db()
    try:
        dirs = export_harbor_tasks(conn, benchmark_id, out)
    except ValueError as exc:
        _fail(str(exc))
    finally:
        conn.close()
    typer.echo(f"exported {len(dirs)} task(s) to {out}")


@export_app.command("atif")
def export_atif_cmd(
    episode_id: str,
    out: str = typer.Option(None, "--out", help="Output file (default <episode>.atif.json)."),
) -> None:
    """Export one captured episode to a Harbor ATIF trajectory JSON."""
    from .capture import export_atif

    path = out or f"{episode_id}.atif.json"
    conn = _open_db()
    try:
        export_atif(conn, episode_id, path)
    except ValueError as exc:
        _fail(str(exc))
    finally:
        conn.close()
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
