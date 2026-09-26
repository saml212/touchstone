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

import json
import re
from pathlib import Path

from . import descriptions, fidelity, recordings
from .minted_ids import scalar_leaves
from .recordings import ToolEvent
from .simulate import crossing_services

_VOLATILE = re.compile(r"(^id$|^rowid$|_id$|^created|^updated|^ts$|^timestamp$|token)", re.I)
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
_MAX_AVOID = 3
_MAX_IDENT_LEN = 32


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
    tool_names = {t["name"] for t in map_data.get("tools", [])}
    mutating = set()
    for service in map_data.get("services", []):
        for call in service.get("calls", []):
            if (call.get("method") or "").upper() in _MUTATING:
                mutating.add(call.get("from_tool"))
    return {t for t in mutating if t in tool_names}  # real model tools only, not harness helpers


def _arg_values(calls: list[ToolEvent]) -> set:
    values: set = set()
    for call in calls:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        values.update(v for v in args.values() if isinstance(v, str | int | float))
    return values


def _referenced_values(calls: list[ToolEvent]) -> set:
    """Every scalar value the episode's tool ARGUMENTS or RESULTS carried — the only values a state
    criterion may key its WHERE on. A row the agent never named (a simulator's internal counter, an
    id-sequence table) is not part of the task's observable effect, so a criterion keyed on it is
    coupled to the seed rather than to what the agent did."""
    vals: set = set()
    for call in calls:
        vals |= scalar_leaves(call.arguments)
        vals |= scalar_leaves(call.output)
    return vals


# SQLite's own AUTOINCREMENT bookkeeping table — never a graded effect.
_INTERNAL_TABLES = {"sqlite_sequence"}


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


def _cell_lines(table: str, pk: str, key, old: dict, row: dict, db_rel: str,
                literals: set) -> list[tuple[str, str]]:
    out = []
    for col, val in row.items():
        if col != pk and not _VOLATILE.search(col) and old.get(col) != val:
            query = f"SELECT {col} FROM {table} WHERE {pk}={_sql_literal(key)}"
            call = f"rk.sqlite_query_equals({db_rel!r}, {query!r}, {val!r})"
            out.append((call, descriptions.cell(table, col, key, val)))
    if out:
        literals.add(str(key))  # the WHERE key the agent must target
    return out


def _changed_cells(table: str, before: dict, after: dict, db_rel: str,
                   literals: set, referenced: set) -> list[tuple[str, str]]:
    pk = after.get("pk", "rowid")
    before_rows, after_rows = _rows_by_pk(before), _rows_by_pk(after)
    is_doc = _is_doc_rows(after_rows)
    lines = []
    for key, row in after_rows.items():
        old = before_rows.get(key)
        # Only grade a changed row the agent actually named (its key appears in a tool arg/result).
        # A counter/sequence row the simulator bumps on its own (WHERE name='email') is not.
        if old is not None and key in referenced:
            lines += (_doc_cell_lines(table, pk, key, old, row, db_rel, literals) if is_doc
                      else _cell_lines(table, pk, key, old, row, db_rel, literals))
    return lines


# ---- document-store tables (id TEXT PRIMARY KEY, doc TEXT) ------------------


def _is_doc_rows(rows_by_pk: dict) -> bool:
    """A document-store table: every row is exactly (id, doc). Its criteria read `doc` fields with
    json_extract instead of grading the whole JSON blob."""
    rows = list(rows_by_pk.values())
    return bool(rows) and all("doc" in r and set(r) <= {"id", "doc"} for r in rows)


def _doc_of(row: dict):
    raw = row.get("doc")
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except (ValueError, TypeError):
        return None


def _flatten_doc(doc, prefix: str = "$") -> dict:
    """Scalar leaves of a JSON document keyed by their json_extract path ($.a.b, $.items[0].id)."""
    if isinstance(doc, dict):
        out: dict = {}
        for k, v in doc.items():
            out.update(_flatten_doc(v, f"{prefix}.{k}"))
        return out
    if isinstance(doc, list):
        out = {}
        for i, v in enumerate(doc):
            out.update(_flatten_doc(v, f"{prefix}[{i}]"))
        return out
    return {prefix: doc}


def _field_name(path: str) -> str:
    return path[2:] if path.startswith("$.") else path


def _doc_cell_lines(table: str, pk: str, key, old_row: dict, new_row: dict, db_rel: str,
                    literals: set) -> list[tuple[str, str]]:
    old = _flatten_doc(_doc_of(old_row) or {})
    new = _flatten_doc(_doc_of(new_row) or {})
    out = []
    for path, val in new.items():
        field = _field_name(path)
        if old.get(path) != val and not _VOLATILE.search(field) and _token_value(val):
            query = f"SELECT json_extract(doc,'{path}') FROM {table} WHERE {pk}={_sql_literal(key)}"
            out.append((f"rk.sqlite_query_equals({db_rel!r}, {query!r}, {val!r})",
                        descriptions.cell(table, field, key, val)))
    if out:
        literals.add(str(key))
    return out


def _token_value(val) -> bool:
    """A stable identifier: a number, or a short whitespace-free string. Free text (subjects,
    bodies, reasons) is rejected so another correct model's wording never fails a criterion."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int | float):
        return True
    return isinstance(val, str) and len(val) <= _MAX_IDENT_LEN and not any(c.isspace() for c in val)


def _identifying_pairs(row: dict, pk: str, arg_values: set) -> list[tuple]:
    return [(col, val) for col, val in row.items()
            if col != pk and val in arg_values and _token_value(val)]


def _added_rows(table: str, before: dict, after: dict, db_rel: str, arg_values: set,
                literals: set) -> list[tuple[str, str]]:
    pk = after.get("pk", "rowid")
    before_keys = set(_rows_by_pk(before))
    added = [row for key, row in _rows_by_pk(after).items() if key not in before_keys]
    if not added:
        return []
    pairs = _identifying_pairs(added[0], pk, arg_values)
    if not pairs:
        # No identifying column to scope the WHERE. An unscoped `SELECT COUNT(*) FROM <table>` would
        # be a table-total: coupled to the seed (seed rows + this episode's) and duplicating the
        # scoped check the same effect already produced (e.g. "exactly one email to <addr>"). Drop
        # it rather than emit a brittle total.
        return []
    literals.update(str(val) for _, val in pairs)  # values the agent must supply
    where = " AND ".join(f"{col}={_sql_literal(val)}" for col, val in pairs)
    query = f"SELECT COUNT(*) FROM {table} WHERE {where}"
    call = f"rk.sqlite_query_equals({db_rel!r}, {query!r}, {len(added)})"
    return [(call, descriptions.count_rows(table, pairs, len(added)))]


def _state_criteria(effect: dict, svc: str, arg_values: set, literals: set,
                    referenced: set) -> list[tuple[str, str]]:
    db_rel = f"simulators/{svc}/state.db"
    initial = effect["initial"].get(svc, {})
    final = effect["final"].get(svc, {})
    lines = []
    for table, after in final.items():
        if table in _INTERNAL_TABLES:  # never grade SQLite's AUTOINCREMENT bookkeeping
            continue
        before = initial.get(table, {})
        lines += _changed_cells(table, before, after, db_rel, literals, referenced)
        lines += _added_rows(table, before, after, db_rel, arg_values, literals)
    return lines


# ---- trajectory criteria ---------------------------------------------------


def _avoid_tools(map_data: dict, calls: list[ToolEvent]) -> list[str]:
    used = {c.tool for c in calls}
    return sorted(_mutating_tools(map_data) - used)[:_MAX_AVOID]


def _used(t: str) -> tuple[str, str]:
    return f"rk.trajectory_tool_used({t!r})", descriptions.tool_used(t)


def _not_used(t: str) -> tuple[str, str]:
    return f"rk.trajectory_tool_not_used({t!r})", descriptions.tool_not_used(t)


def _tool_criteria(map_data: dict, calls: list[ToolEvent],
                   has_state: bool) -> list[tuple[str, str]]:
    """Trajectory criteria that don't over-fit: require a mutating tool the episode used only when
    its effect is NOT already captured by a state criterion; forbid the mutating tools it avoided;
    never require a read-only tool. Fall back to requiring the used tools only when nothing else
    would verify the task at all."""
    used_mutating = sorted({c.tool for c in calls if c.tool in _mutating_tools(map_data)})
    avoid = _avoid_tools(map_data, calls)
    lines = [] if has_state else [_used(t) for t in used_mutating]
    lines += [_not_used(t) for t in avoid]
    if not lines and not has_state:
        lines = [_used(t) for t in sorted({c.tool for c in calls if c.tool})]
    return lines


# ---- capture + reproduce ---------------------------------------------------


def _mounts(services: list[dict], out: Path, base_url_envs: dict) -> list[dict]:
    from . import db_service
    from .simulate import service_host

    return [{"sim_dir": out / "simulators" / s["name"], "env": base_url_envs.get(s["name"]),
             "host": service_host(s), "kind": s.get("kind"),
             "db_url": db_service.is_url(s)} for s in services]


def capture_effect(services, out, base_url_envs, repo, tools, calls, settings,
                   invoke=None) -> dict:
    """Replay `calls` against the simulators and return {"initial", "final", "replayed"}."""
    if not services:
        return {"initial": {}, "final": {}, "replayed": []}
    mounts = _mounts(services, out, base_url_envs)
    return fidelity.capture_state(mounts, repo, tools, calls, settings, invoke)


def reproduced(calls: list[ToolEvent], replayed: list[dict]) -> bool:
    """The simulator reproduced this episode iff every replayed output matches the recording."""
    if len(replayed) < len(calls):
        return False
    for call, got in zip(calls, replayed, strict=False):
        raw = got.get("got")
        if isinstance(raw, dict) and "__error__" in raw:
            return False
        # Decode a JSON-string result the same way recorded outputs are stored, so a db tool that
        # returns a JSON string still matches its decoded recording.
        if not fidelity.masked_equal(call.output, recordings.decode_output(raw)):
            return False
    return True


def derive_criteria(effect: dict, services: list[dict], map_data: dict, calls: list[ToolEvent]
                    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Return (state, trajectory, required literals) for the replayed episode; state and trajectory
    are (rewardkit call, plain-English description) pairs. Every literal is a value used in a state
    criterion's WHERE that the agent must know — so the task writer can make it knowable (state it
    in the instruction) or drop the criterion."""
    arg_values = _arg_values(calls)
    referenced = _referenced_values(calls)
    literals: set = set()
    state: list[tuple[str, str]] = []
    for service in services:
        state += _state_criteria(effect, service["name"], arg_values, literals, referenced)
    tool = _tool_criteria(map_data, calls, bool(state))
    return state, tool, sorted(literals)
