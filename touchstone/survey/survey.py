"""Orchestrate the survey: map -> sort -> simulate -> group -> environment -> tasks -> gate ->
baseline -> dataset.toml -> report.

Idempotent and diff-friendly: existing outputs are reused unless `force`, every file is written
atomically, and nothing outside `touchstone/` in the target repo is touched (the trace DB is only
read, never created). One progress line per step goes to stderr; the return value is the one-line
summary. `baseline` is a stage-3B seam (a no-op returning None here); the environment step precedes
tasks because tasks embed the environment's image tag and simulator ports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import store
from ..config import Settings, load_settings
from ..harbor.dataset import Dataset
from .baseline import run_baseline
from .environment import build_environment
from .gate import run_gate
from .group import group_episodes
from .map import build_map
from .package import build_package
from .provider import survey_provider
from .recordings import tool_events
from .report import render_report, summary
from .scrub import Scrubber
from .simulate import crossing_services, generate_simulator, service_tools
from .sort import sort_tools
from .tasks import write_tasks
from .writes import atomic_write, atomic_write_json


def _log(message: str) -> None:
    print(f"survey: {message}", file=sys.stderr, flush=True)


def _open_db(repo: Path):
    db = repo / ".touchstone" / "touchstone.db"
    return store.connect(db) if db.exists() else None


def _simulate_all(repo, provider, map_data, events, out, scrub, settings, force) -> dict:
    fidelity_path = out / "fidelity.json"
    if fidelity_path.exists() and not force:
        return json.loads(fidelity_path.read_text(encoding="utf-8"))
    sim_root = out / "simulators"
    results: dict = {}
    for service in crossing_services(map_data):
        tools = service_tools(map_data, service)
        _log(f"simulate: {service['name']} ({len(tools)} tool(s))")
        results[service["name"]] = generate_simulator(
            repo, provider, service, tools, events, sim_root, scrub, settings, force)
    return results


def _map_and_fidelity(repo, prov, out, events, scrub, settings, force):
    _log(f"map: reading {repo.name}")
    map_data = build_map(repo, prov, out, force)
    _log("sort: classifying tools by network boundary")
    map_data["sort"] = sort_tools(map_data, events)
    atomic_write_json(out / "map.json", map_data)
    fidelity_data = _simulate_all(repo, prov, map_data, events, out, scrub, settings, force)
    atomic_write_json(out / "fidelity.json", fidelity_data)
    return map_data, fidelity_data


def _build_tasks(repo, conn, map_data, out, events, scrub, prov, settings, force):
    """group -> environment -> tasks. Returns (groups, env_result, task_result)."""
    if conn is None:
        return None, None, None
    _log("group: clustering episodes by job-to-be-done")
    groups = group_episodes(conn, events, prov, repo, out, scrub, force)
    _log("environment: snapshotting the repo into an image")
    env_result = build_environment(repo, map_data, out, force)
    _log("tasks: writing Harbor tasks")
    tasks = write_tasks(repo, conn, map_data, groups, events, env_result, prov, scrub,
                        settings, force)
    return groups, env_result, tasks


def _package_agent(repo, conn, map_data, env_result, prov, out, settings, force):
    """Write touchstone/agent/ (packaged or replica). None with no trace db or environment."""
    if conn is None or env_result is None:
        return None
    _log("package: building the agent under test")
    return build_package(repo, conn, map_data, env_result, prov, out, settings, force)


def _gate_and_baseline(repo, env_result, settings, force, skip_gate):
    if env_result is None or skip_gate:
        return None
    _log("gate: running oracle and nop over the tasks")
    gate = run_gate(repo, env_result, settings, force)
    run_baseline(repo, env_result, settings, force)
    return gate


def _stats(conn, tasks, gate, skip_gate) -> dict:
    built = set((tasks or {}).get("written", []) + (tasks or {}).get("reused", []))
    return {"tasks": len(built),
            "conversations": len(store.list_episodes(conn)) if conn else 0,
            "gated": len((gate or {}).get("gated", [])),
            "needs_review": len((gate or {}).get("needs_review", [])),
            "gate_skipped": skip_gate or (gate is None and bool(built))}


def _dataset_name(repo: Path, settings: Settings) -> str:
    return settings.survey_dataset_name or f"{repo.name}/{repo.name}"


def run_survey(repo: str | Path, force: bool = False, provider: str | None = None,
               model: str | None = None, settings: Settings | None = None,
               skip_gate: bool = False) -> str:
    settings = settings or load_settings()
    repo = Path(repo).expanduser().resolve()
    out = repo / "touchstone"
    out.mkdir(parents=True, exist_ok=True)
    prov = survey_provider(settings, provider, model)
    scrub = Scrubber(settings.survey_names)
    conn = _open_db(repo)
    try:
        events = tool_events(conn) if conn else []
        map_data, fidelity_data = _map_and_fidelity(repo, prov, out, events, scrub, settings, force)
        groups, env_result, tasks = _build_tasks(
            repo, conn, map_data, out, events, scrub, prov, settings, force)
        package = _package_agent(repo, conn, map_data, env_result, prov, out, settings, force)
        gate = _gate_and_baseline(repo, env_result, settings, force, skip_gate)
        Dataset(name=_dataset_name(repo, settings)).write(out)
        _log("report: writing report.md")
        atomic_write(out / "report.md",
                     render_report(map_data, fidelity_data, tasks, gate, groups, package))
        return summary(map_data, fidelity_data, _stats(conn, tasks, gate, skip_gate))
    finally:
        if conn is not None:
            conn.close()
