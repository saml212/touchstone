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


def render_report(map_data: dict, fidelity: dict) -> str:
    sections = [
        "# Survey report",
        map_section(map_data),
        services_section(map_data),
        simulators_section(fidelity),
        flags_section(map_data, fidelity),
    ]
    return "\n\n".join(sections) + "\n"


def summary(map_data: dict, fidelity: dict) -> str:
    tools = len(map_data.get("tools", []))
    services = len(crossing_services(map_data))
    head = f"Mapped {tools} tool{'' if tools == 1 else 's'}, "
    head += f"{services} service{'' if services == 1 else 's'}."
    sims = [f"Simulator {name}: fidelity {r.get('score', 0):.2f} "
            f"({r.get('reproduced', 0)}/{r.get('calls', 0)} calls)"
            for name, r in sorted(fidelity.items())]
    return head + (" " + "; ".join(sims) if sims else "")
