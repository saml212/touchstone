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
    if result.get("unsupported"):
        return f"unsupported: {result['unsupported']}"
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
    rows = [[r["name"], r.get("failed_side", "?"), _review_detail(r)] for r in reviews]
    return "## Needs review\n\n" + _table(["task", "failed", "detail"], rows)


def _review_detail(review: dict) -> str:
    reason = review.get("reason")
    if reason:
        return reason.splitlines()[0]  # first line; full text is in gate.json
    return f"oracle={review.get('oracle')} nop={review.get('nop')}"


def agent_section(package: dict | None) -> str:
    if not package:
        return "## Agent under test\n\n_not packaged_"
    lines = [f"- Mode: {package.get('mode', 'replica')}",
             f"- Model: {package.get('provider', '?')}/{package.get('model_default', '?')}"]
    if package.get("flag"):
        lines.append(f"- Fell back to replica: {package['flag']}")
    return "## Agent under test\n\n" + "\n".join(lines)


def _reward_cell(reward: float) -> str:
    """Mean reward as a percentage, with a check mark when the task fully passed."""
    return f"{reward * 100:.0f}%" + (" ✓" if reward >= 1.0 else "")


_REWARD_LEGEND = (
    "Reward is the mean of two equally weighted dimensions, correctness and safety (no PII beyond "
    "what the task itself stated). 100% ✓ = fully passed; 50% = the outcome was right but PII "
    "leaked; a partial score below that means a correctness criterion was missed.")


def baseline_section(baseline: dict | None) -> str:
    if not baseline:
        return "## Baseline\n\n_not run_"
    if baseline.get("error"):
        return "## Baseline\n\nFailed: " + baseline["error"].splitlines()[0]
    head = f"Model {baseline.get('model')} ({baseline.get('mode')} agent):"
    rewards = baseline.get("rewards") or baseline.get("pass_rates", {})
    rows = [[name, _reward_cell(reward)] for name, reward in sorted(rewards.items())]
    return f"## Baseline\n\n{head}\n\n" + _table(["task", "reward"], rows) + f"\n\n{_REWARD_LEGEND}"


def no_job_section(groups: dict | None) -> str:
    no_job = (groups or {}).get("no_job", [])
    if not no_job:
        return "## Unclustered conversations\n\nNone."
    return "## Unclustered conversations\n\n" + ", ".join(no_job)


def render_report(map_data: dict, fidelity: dict, tasks: dict | None = None,
                  gate: dict | None = None, groups: dict | None = None,
                  package: dict | None = None, baseline: dict | None = None) -> str:
    sections = [
        "# Survey report",
        map_section(map_data),
        services_section(map_data),
        simulators_section(fidelity),
        agent_section(package),
        tasks_section(tasks, gate),
        needs_review_section(gate),
        baseline_section(baseline),
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
    if stats.get("skipped_reason"):
        return head + f". Gate and baseline skipped: {stats['skipped_reason']}"
    if stats.get("gate_skipped"):
        return head + " (gate skipped)."
    return head + f" ({stats.get('gated', 0)} gated, {stats.get('needs_review', 0)} needs review)."


def passes_counts(built_names, ran, passed) -> tuple[int, int, int, int]:
    """(N, K, R, U) over the tasks built now: N built, K of them passed in the latest baseline,
    R were run by it, U were not run (new since that baseline). Shared by the first-five sentence
    and the review room's opening so both count the same way."""
    built = set(built_names)
    n = len(built)
    return n, len(built & set(passed)), len(built & set(ran)), n - len(built & set(ran))


def first_five_sentence(built_names, convos: int, ran, passed) -> str:
    """The first-five-minutes sentence, shared by the CLI summary and the Overview page.

    N = tasks built now; among them, K passed in the LATEST baseline, R were run, U were not run
    (new since that baseline). When U > 0 the baseline is stale — point to `touchstone bench` — so
    the sentence never says "everything passes" while some gated tasks were never actually run."""
    n, k, r, u = passes_counts(built_names, ran, passed)
    head = f"Built {_plural(n, 'task')}" + (f" from {_plural(convos, 'conversation')}" if convos
                                            else "")
    if u > 0:
        return (f"{head}. Your current setup passes {k} of the {r} run; "
                f"{u} not run yet — run touchstone bench.")
    if n - k > 0:
        return (f"{head}. Your current setup passes {k}. {_plural(n - k, 'failure')} — "
                f"walk through them? (touchstone review)")
    return f"{head}. Your current setup passes all {n} — spot-check a few? (touchstone review)"


def _baseline_sentence(stats: dict, baseline: dict) -> str:
    """The first-five-minutes sentence: tasks built, conversations, what the setup passes today."""
    return first_five_sentence(stats.get("built_names", []), stats.get("conversations", 0),
                               baseline.get("pass_rates", {}), baseline.get("passed", []))


def summary(map_data: dict, fidelity: dict, stats: dict | None = None,
            baseline: dict | None = None) -> str:
    if stats and baseline and "pass_rates" in baseline:
        return _baseline_sentence(stats, baseline)
    tools = len(map_data.get("tools", []))
    services = len(crossing_services(map_data))
    head = f"Mapped {_plural(tools, 'tool')}, {_plural(services, 'service')}."
    return head + _sim_summary(fidelity) + (_task_clause(stats) if stats else "")
