"""The survey agent's own scoreboard: how well the survey copied a set of target repos.

`survey_bench(targets.toml)` reads each already-surveyed target's `touchstone/` outputs and scores
four things per target: simulator fidelity per service, tasks written vs gated (oracle==1, nop<1),
the baseline pass rate of the packaged/replica agent on the recorded model, and agreement — the
share of gated tasks where the recorded episode's real outcome matches the baseline verdict (tasks
are built from recorded successes, so a task the baseline also passes agrees with what happened).

It reads outputs only; recording and `touchstone survey <repo>` run per target beforehand (the
`run` field in targets.toml documents how each target's conversations were recorded). Results are
written to `benchmarks/survey/<date>.json` plus a markdown table. Nothing here names a target.
"""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from .. import store

_RESOLVED_LABELS = {"resolved", "success", "ok", "done"}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _fidelity(out: Path) -> dict:
    data = _load_json(out / "fidelity.json") or {}
    return {name: round(float(r.get("score", 0.0)), 4) for name, r in data.items()}


def _task_meta(task_dir: Path) -> dict:
    doc = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    return doc.get("metadata", {}).get("touchstone", {})


def _is_gated(meta: dict) -> bool:
    return meta.get("oracle") == 1.0 and (meta.get("nop") is not None and meta["nop"] < 1.0)


def _task_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(d for d in parent.iterdir() if (d / "task.toml").is_file())


def _gated_tasks(out: Path) -> list[Path]:
    return [d for d in _task_dirs(out / "tasks") if _is_gated(_task_meta(d))]


def _counts(out: Path) -> dict:
    tasks, review = _task_dirs(out / "tasks"), [d for d in (out / "needs-review").iterdir()
                                                if (out / "needs-review").is_dir()
                                                and (d / "task.toml").is_file()]
    return {"written": len(tasks) + len(review), "gated": len(_gated_tasks(out))}


def _rate_for(rates: dict, name: str) -> float | None:
    if name in rates:
        return rates[name]
    for key, value in rates.items():
        if key.endswith(f"/{name}"):
            return value
    return None


def _baseline_rates(out: Path) -> dict | None:
    data = _load_json(out / "baseline.json")
    return data["pass_rates"] if data and "pass_rates" in data else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _recorded_passed(episode) -> bool:
    label = (episode.outcome_label or "").lower()
    return (episode.outcome_score or 0) >= 1.0 or label in _RESOLVED_LABELS


def _outcomes(repo: Path) -> dict:
    db = repo / ".touchstone" / "touchstone.db"
    if not db.is_file():
        return {}
    conn = store.connect(str(db))
    try:
        return {ep.id: _recorded_passed(ep) for ep in store.list_episodes(conn)}
    finally:
        conn.close()


def _task_recorded_passed(meta: dict, outcomes: dict) -> bool | None:
    labelled = [outcomes[e] for e in meta.get("episodes", []) if e in outcomes]
    return all(labelled) if labelled else None


def _agreement(out: Path, rates: dict | None, outcomes: dict) -> float | None:
    if rates is None:
        return None
    agree = []
    for task_dir in _gated_tasks(out):
        recorded = _task_recorded_passed(_task_meta(task_dir), outcomes)
        rate = _rate_for(rates, task_dir.name)
        if recorded is not None and rate is not None:
            agree.append(1.0 if recorded == (rate >= 1.0) else 0.0)
    return _mean(agree)


def score_target(name: str, repo: str | Path, run: str = "") -> dict:
    """Score one already-surveyed target from its touchstone/ outputs (+ recorded outcomes)."""
    repo = Path(repo).expanduser()
    out = repo / "touchstone"
    fidelity = _fidelity(out)
    rates = _baseline_rates(out)
    return {"name": name, "repo": str(repo), "run": run, "services": fidelity,
            "fidelity_min": min(fidelity.values()) if fidelity else None,
            **_counts(out),
            "baseline_pass": _mean(list(rates.values())) if rates else None,
            "agreement": _agreement(out, rates, _outcomes(repo))}


def _targets(config: Path) -> list[dict]:
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    return data.get("target", [])


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}" if isinstance(value, float) else str(value)


def _services_cell(row: dict) -> str:
    if not row["services"]:
        return "—"
    return ", ".join(f"{n} {s:.0%}" for n, s in sorted(row["services"].items()))


def render_markdown(results: list[dict], generated_at: str) -> str:
    head = ("# Survey benchmark\n\n"
            f"Generated {generated_at}. Each target was recorded, then surveyed; "
            "this scores the outputs.\n\n"
            "| target | services (fidelity) | tasks written/gated | baseline pass | agreement |\n"
            "|---|---|---|---|---|\n")
    rows = [f"| {r['name']} | {_services_cell(r)} | {r['written']}/{r['gated']} | "
            f"{_fmt(r['baseline_pass'])} | {_fmt(r['agreement'])} |" for r in results]
    return head + "\n".join(rows) + "\n"


def survey_bench(config: str | Path, out_root: str | Path = ".") -> dict:
    """Score every target in `config`, write benchmarks/survey/<date>.{json,md}, return the doc."""
    results = [score_target(t["name"], t["repo"], t.get("run", "")) for t in _targets(Path(config))]
    generated_at = datetime.now(UTC).isoformat()
    doc = {"generated_at": generated_at, "targets": results}
    bench_dir = Path(out_root) / "benchmarks" / "survey"
    bench_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    (bench_dir / f"{stamp}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (bench_dir / f"{stamp}.md").write_text(render_markdown(results, generated_at), encoding="utf-8")
    return doc
