"""Render `report.md` from the map and fidelity results.

Each section is a function returning markdown, so stage 3 can append its own (tasks, gates,
baseline) without touching these. Nothing here names a product — it reports whatever was mapped.
"""

from __future__ import annotations

from .simulate import crossing_services


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_"
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def map_section(map_data: dict) -> str:
    entrypoints = ", ".join(map_data.get("entrypoints", [])) or "_none_"
    mc = map_data.get("model_call", {})
    call_site = (f"{mc.get('file')}:{mc.get('line')} — {mc.get('sdk')} "
                 f"(model: {mc.get('model_setting')})")
    rows = [[t.get("name"), t.get("import_path"), ", ".join(t.get("calls") or []) or "—"]
            for t in map_data.get("tools", [])]
    table = _table(["tool", "import path", "calls"], rows)
    return (f"## Map\n\n- Entrypoints: {entrypoints}\n- Model call site: {call_site}\n\n"
            f"### Tools\n\n{table}")


def services_section(map_data: dict) -> str:
    rows = []
    for s in map_data.get("services", []):
        routes = ", ".join(f"{c.get('method')} {c.get('path_template')}"
                           for c in s.get("calls", [])) or "—"
        rows.append([s.get("name"), s.get("kind"), s.get("base_url_env") or "—", routes])
    return "## Services\n\n" + _table(["service", "kind", "base_url env", "routes"], rows)


def _sim_status(result: dict) -> str:
    return "ok" if result.get("score", 0) >= result.get("threshold", 0) else "below threshold"


def simulators_section(fidelity: dict) -> str:
    if not fidelity:
        return "## Simulators\n\nNo network-crossing services; no simulators needed."
    rows = [[name, f"{r.get('score', 0):.2f}",
             f"{r.get('reproduced', 0)}/{r.get('calls', 0)}", _sim_status(r)]
            for name, r in sorted(fidelity.items())]
    return "## Simulators\n\n" + _table(["service", "fidelity", "reproduced", "status"], rows)


def _below_threshold(fidelity: dict) -> list[str]:
    return [name for name, r in fidelity.items()
            if r.get("score", 0) < r.get("threshold", 0)]


def flags_section(map_data: dict, fidelity: dict) -> str:
    sort = map_data.get("sort", {})
    groups = [
        ("Simulators below fidelity threshold", _below_threshold(fidelity)),
        ("Tools in recordings but not in the map", sort.get("unmapped", [])),
        ("Tools in the map never seen in recordings", sort.get("unused", [])),
    ]
    lines = ["## Flags", ""]
    any_flag = False
    for label, items in groups:
        if items:
            any_flag = True
            lines.append(f"- {label}: {', '.join(items)}")
    if not any_flag:
        lines.append("None.")
    return "\n".join(lines)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def tasks_section(tasks: dict | None, gate: dict | None) -> str:
    if not tasks:
        return "## Tasks\n\n_none_"
    built = sorted(set(tasks.get("written", []) + tasks.get("reused", [])))
    rows = [[name, "gated" if _is_gated(name, gate) else "written"] for name in built]
    body = _table(["task", "status"], rows)
    skipped = tasks.get("skipped", [])
    if skipped:
        lines = "\n".join(f"- {s['episode']}: {s['reason']}" for s in skipped)
        body += f"\n\n### Skipped episodes (simulator could not reproduce)\n\n{lines}"
    return "## Tasks\n\n" + body


def _is_gated(name: str, gate: dict | None) -> bool:
    return bool(gate) and name in gate.get("gated", [])


def needs_review_section(gate: dict | None) -> str:
    if not gate:
        return "## Needs review\n\n_gate not run_"
    reviews = gate.get("needs_review", [])
    if not reviews:
        return "## Needs review\n\nNone."
    rows = [[r["name"], r.get("failed_side", "?"),
             r.get("reason") or f"oracle={r.get('oracle')} nop={r.get('nop')}"] for r in reviews]
    return "## Needs review\n\n" + _table(["task", "failed", "detail"], rows)


def no_job_section(groups: dict | None) -> str:
    no_job = (groups or {}).get("no_job", [])
    if not no_job:
        return "## Unclustered conversations\n\nNone."
    return "## Unclustered conversations\n\n" + ", ".join(no_job)


def render_report(map_data: dict, fidelity: dict, tasks: dict | None = None,
                  gate: dict | None = None, groups: dict | None = None) -> str:
    sections = [
        "# Survey report",
        map_section(map_data),
        services_section(map_data),
        simulators_section(fidelity),
        tasks_section(tasks, gate),
        needs_review_section(gate),
        no_job_section(groups),
        flags_section(map_data, fidelity),
    ]
    return "\n\n".join(sections) + "\n"


def _sim_summary(fidelity: dict) -> str:
    sims = [f"Simulator {name}: fidelity {r.get('score', 0):.2f} "
            f"({r.get('reproduced', 0)}/{r.get('calls', 0)} calls)"
            for name, r in sorted(fidelity.items())]
    return (" " + "; ".join(sims)) if sims else ""


def _task_clause(stats: dict) -> str:
    tasks, convos = stats.get("tasks", 0), stats.get("conversations", 0)
    if not convos:
        return ""
    head = f" Built {_plural(tasks, 'task')} from {_plural(convos, 'conversation')}"
    if stats.get("gate_skipped"):
        return head + " (gate skipped)."
    return head + f" ({stats.get('gated', 0)} gated, {stats.get('needs_review', 0)} needs review)."


def summary(map_data: dict, fidelity: dict, stats: dict | None = None) -> str:
    tools = len(map_data.get("tools", []))
    services = len(crossing_services(map_data))
    head = f"Mapped {_plural(tools, 'tool')}, {_plural(services, 'service')}."
    return head + _sim_summary(fidelity) + (_task_clause(stats) if stats else "")
