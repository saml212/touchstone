"""Turn a replayed episode into rewardkit criterion lines — the verifier's substance.

Order matches the design's "end-state first": the episode is replayed against the simulators (real
tool functions), the simulator databases are diffed before and after, and each change becomes a
`sqlite_query_equals` line (changed cells, then added rows keyed by an input-matching column). Then
come trajectory checks — `trajectory_tool_used` for every tool the episode used and
`trajectory_tool_not_used` for the mutating tools it avoided. The no-PII safety check and any judge
live in the task writer, which owns the tests/ layout.

`capture_effect` and `reproduced` also serve the per-episode fidelity gate: an episode the simulator
cannot reproduce is skipped, never made into a task.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import fidelity
from .recordings import ToolEvent
from .simulate import crossing_services

_VOLATILE = re.compile(r"(^id$|^rowid$|_id$|^created|^updated|^ts$|^timestamp$|token)", re.I)
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
_MAX_AVOID = 3
_MAX_IDENT_LEN = 64


# ---- service + tool mapping ------------------------------------------------


def episode_services(map_data: dict, calls: list[ToolEvent]) -> list[dict]:
    used_tools = {c.tool for c in calls}
    services = []
    for service in crossing_services(map_data):
        tools = [t for t in map_data.get("tools", []) if service["name"] in (t.get("calls") or [])]
        if any(t["name"] in used_tools for t in tools):
            services.append(service)
    return services


def tools_map(map_data: dict) -> dict:
    return {t["name"]: t["import_path"] for t in map_data.get("tools", []) if t.get("import_path")}


def _mutating_tools(map_data: dict) -> set:
    mutating = set()
    for service in map_data.get("services", []):
        for call in service.get("calls", []):
            if (call.get("method") or "").upper() in _MUTATING:
                mutating.add(call.get("from_tool"))
    return {t for t in mutating if t}


def _arg_values(calls: list[ToolEvent]) -> set:
    values: set = set()
    for call in calls:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        values.update(v for v in args.values() if isinstance(v, str | int | float))
    return values


# ---- state diff -> sqlite criteria -----------------------------------------


def _sql_literal(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _rows_by_pk(dump: dict) -> dict:
    pk = dump.get("pk", "rowid")
    return {row.get(pk): row for row in dump.get("rows", [])}


def _cell_lines(table: str, pk: str, key, old: dict, row: dict, db_rel: str) -> list[str]:
    out = []
    for col, val in row.items():
        if col != pk and not _VOLATILE.search(col) and old.get(col) != val:
            query = f"SELECT {col} FROM {table} WHERE {pk}={_sql_literal(key)}"
            out.append(f"rk.sqlite_query_equals({db_rel!r}, {query!r}, {val!r})")
    return out


def _changed_cells(table: str, before: dict, after: dict, db_rel: str) -> list[str]:
    pk = after.get("pk", "rowid")
    before_rows, after_rows = _rows_by_pk(before), _rows_by_pk(after)
    lines = []
    for key, row in after_rows.items():
        old = before_rows.get(key)
        if old is not None:
            lines += _cell_lines(table, pk, key, old, row, db_rel)
    return lines


def _short_value(val) -> bool:
    return isinstance(val, int | float) or (isinstance(val, str) and len(val) <= _MAX_IDENT_LEN)


def _identifying_where(row: dict, pk: str, arg_values: set) -> str:
    parts = []
    for col, val in row.items():
        if col != pk and val in arg_values and _short_value(val):
            parts.append(f"{col}={_sql_literal(val)}")
    return " AND ".join(parts)


def _added_rows(table: str, before: dict, after: dict, db_rel: str, arg_values: set) -> list[str]:
    pk = after.get("pk", "rowid")
    before_keys = set(_rows_by_pk(before))
    added = [row for key, row in _rows_by_pk(after).items() if key not in before_keys]
    if not added:
        return []
    where = _identifying_where(added[0], pk, arg_values)
    if where:
        query = f"SELECT COUNT(*) FROM {table} WHERE {where}"
        return [f"rk.sqlite_query_equals({db_rel!r}, {query!r}, {len(added)})"]
    total = len(after.get("rows", []))
    return [f"rk.sqlite_query_equals({db_rel!r}, {f'SELECT COUNT(*) FROM {table}'!r}, {total})"]


def _state_criteria(effect: dict, svc: str, arg_values: set) -> list[str]:
    db_rel = f"simulators/{svc}/state.db"
    initial = effect["initial"].get(svc, {})
    final = effect["final"].get(svc, {})
    lines = []
    for table, after in final.items():
        before = initial.get(table, {})
        lines += _changed_cells(table, before, after, db_rel)
        lines += _added_rows(table, before, after, db_rel, arg_values)
    return lines


# ---- trajectory criteria ---------------------------------------------------


def _avoid_tools(map_data: dict, calls: list[ToolEvent]) -> list[str]:
    used = {c.tool for c in calls}
    return sorted(_mutating_tools(map_data) - used)[:_MAX_AVOID]


def _tool_criteria(calls: list[ToolEvent], avoid: list[str]) -> list[str]:
    used = sorted({c.tool for c in calls if c.tool})
    lines = [f"rk.trajectory_tool_used({name!r})" for name in used]
    lines += [f"rk.trajectory_tool_not_used({name!r})" for name in avoid]
    return lines


# ---- capture + reproduce ---------------------------------------------------


def _mounts(services: list[dict], out: Path, base_url_envs: dict) -> list[dict]:
    return [{"sim_dir": out / "simulators" / s["name"], "env": base_url_envs.get(s["name"])}
            for s in services]


def capture_effect(services, out, base_url_envs, repo, tools, calls, settings) -> dict:
    """Replay `calls` against the simulators and return {"initial", "final", "replayed"}."""
    if not services:
        return {"initial": {}, "final": {}, "replayed": []}
    mounts = _mounts(services, out, base_url_envs)
    return fidelity.capture_state(mounts, repo, tools, calls, settings)


def reproduced(calls: list[ToolEvent], replayed: list[dict]) -> bool:
    """The simulator reproduced this episode iff every replayed output matches the recording."""
    if len(replayed) < len(calls):
        return False
    for call, got in zip(calls, replayed, strict=False):
        value = got.get("got")
        if isinstance(value, dict) and "__error__" in value:
            return False
        if not fidelity.masked_equal(call.output, value):
            return False
    return True


def derive_criteria(effect: dict, services: list[dict], map_data: dict,
                    calls: list[ToolEvent]) -> tuple[list[str], list[str]]:
    """Return (state criteria lines, trajectory criteria lines) for the replayed episode."""
    arg_values = _arg_values(calls)
    state: list[str] = []
    for service in services:
        state += _state_criteria(effect, service["name"], arg_values)
    tool = _tool_criteria(calls, _avoid_tools(map_data, calls))
    return state, tool
