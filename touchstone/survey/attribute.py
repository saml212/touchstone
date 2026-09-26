"""Attribute each recorded call to the crossing service it actually hit.

Two services can share every tool name (tau-bench's retail and airline both expose
`get_user_details`, `think`, `calculate`). Filtering recorded calls by tool NAME alone hands a call
to both services, so one gets calls it never served and neither's simulator is scored honestly. This
module attributes a call to one service using, in order:

1. the tool span's `module` (recorded by capture) — the service whose tool `import_path` module ran;
2. a uniquely-owned tool name — a name only one crossing service exposes belongs to that service;
3. episode co-occurrence — a shared-name call is attributed to the single service whose
   uniquely-owned tools appear in the same episode (the entrypoint that ran).

A shared-name call with none of these signals is left unattributed. A crossing service with zero
attributed calls is dropped (`effective_crossing_services`). Pure functions over map + events.
"""

from __future__ import annotations

from collections import defaultdict

from .recordings import ToolEvent
from .simulate import crossing_services


def _names_by_service(map_data: dict) -> dict[str, set[str]]:
    tools = map_data.get("tools", [])
    return {s["name"]: {t["name"] for t in tools if s["name"] in (t.get("calls") or [])}
            for s in crossing_services(map_data)}


def _owner_of_name(names_by_service: dict[str, set[str]]) -> dict[str, str | None]:
    """Tool name -> the one crossing service that exposes it, or None when more than one does."""
    owner: dict[str, str | None] = {}
    for svc, names in names_by_service.items():
        for name in names:
            owner[name] = svc if name not in owner else None
    return owner


def _module_service(map_data: dict) -> dict[str, str]:
    """import_path module -> the crossing service that module's tool calls."""
    crossing = {s["name"] for s in crossing_services(map_data)}
    out: dict[str, str] = {}
    for tool in map_data.get("tools", []):
        module = (tool.get("import_path") or "").split(":")[0]
        svcs = [c for c in (tool.get("calls") or []) if c in crossing]
        if module and svcs:
            out.setdefault(module, svcs[0])
    return out


def _episode_owners(events: list[ToolEvent], owner: dict[str, str | None]) -> dict[str, set[str]]:
    by_ep: dict[str, set[str]] = defaultdict(set)
    for event in events:
        svc = owner.get(event.tool)
        if svc:
            by_ep[event.episode].add(svc)
    return by_ep


def _attribute_one(event: ToolEvent, module_service: dict[str, str],
                   owner: dict[str, str | None], episode_owners: dict[str, set[str]]) -> str | None:
    if event.module and event.module in module_service:
        return module_service[event.module]
    direct = owner.get(event.tool)
    if direct:
        return direct
    owners = episode_owners.get(event.episode) or set()
    return next(iter(owners)) if len(owners) == 1 else None


def attribute(map_data: dict, events: list[ToolEvent]) -> dict[str, list[ToolEvent]]:
    """Recorded calls grouped by the crossing service each hit (unattributed calls are omitted)."""
    names_by_service = _names_by_service(map_data)
    owner = _owner_of_name(names_by_service)
    module_service = _module_service(map_data)
    episode_owners = _episode_owners(events, owner)
    out: dict[str, list[ToolEvent]] = {svc: [] for svc in names_by_service}
    for event in events:
        if event.tool not in owner:  # not a crossing-service tool (runs on its own)
            continue
        svc = _attribute_one(event, module_service, owner, episode_owners)
        if svc in out:
            out[svc].append(event)
    return out


def service_calls(map_data: dict, events: list[ToolEvent], name: str) -> list[ToolEvent]:
    return attribute(map_data, events).get(name, [])


def resolve_service_calls(service: dict, tools: list[dict], events: list[ToolEvent],
                          calls: list[ToolEvent] | None) -> list[ToolEvent]:
    """The recorded calls a service served: the already-attributed `calls` when the caller resolved
    shared names, else `events` filtered by this service's tool names (fine when no name clash)."""
    if calls is not None:
        return calls
    names = {t["name"] for t in tools}
    return [e for e in events if e.tool in names]


def effective_crossing_services(map_data: dict, events: list[ToolEvent]) -> list[dict]:
    """Crossing services with at least one attributed call — the rest are dropped as noise."""
    attributed = attribute(map_data, events)
    return [s for s in crossing_services(map_data) if attributed.get(s["name"])]
