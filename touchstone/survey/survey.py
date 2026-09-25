"""Orchestrate the survey: map -> sort -> simulate -> fidelity -> report.

Idempotent and diff-friendly: existing outputs are reused unless `force`, every file is written
atomically, and nothing outside `touchstone/` in the target repo is touched (the trace DB is only
read, never created). One progress line per step goes to stderr; the return value is the one-line
summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import store
from ..config import Settings, load_settings
from .map import build_map
from .provider import survey_provider
from .recordings import tool_events
from .report import render_report, summary
from .scrub import Scrubber
from .simulate import crossing_services, generate_simulator, service_tools
from .sort import sort_tools
from .writes import atomic_write, atomic_write_json


def _log(message: str) -> None:
    print(f"survey: {message}", file=sys.stderr, flush=True)


def _read_events(repo: Path):
    db = repo / ".touchstone" / "touchstone.db"
    if not db.exists():  # do not create anything outside touchstone/
        return []
    conn = store.connect(db)
    try:
        return tool_events(conn)
    finally:
        conn.close()


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


def run_survey(repo: str | Path, force: bool = False, provider: str | None = None,
               model: str | None = None, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    repo = Path(repo).expanduser().resolve()
    out = repo / "touchstone"
    out.mkdir(parents=True, exist_ok=True)
    prov = survey_provider(settings, provider, model)

    events = _read_events(repo)
    _log(f"map: reading {repo.name}")
    map_data = build_map(repo, prov, out, force)
    _log("sort: classifying tools by network boundary")
    map_data["sort"] = sort_tools(map_data, events)
    atomic_write_json(out / "map.json", map_data)

    scrub = Scrubber(settings.survey_names)
    fidelity_data = _simulate_all(repo, prov, map_data, events, out, scrub, settings, force)
    atomic_write_json(out / "fidelity.json", fidelity_data)

    _log("report: writing report.md")
    atomic_write(out / "report.md", render_report(map_data, fidelity_data))
    return summary(map_data, fidelity_data)
