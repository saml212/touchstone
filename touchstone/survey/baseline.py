"""Baseline: run the packaged (or replica) agent under its current model and report what it passes.

After the gate, run the agent under test over the gated tasks with the recorded model, one trial per
task, and write `touchstone/baseline.json` {job_dir, model, mode, pass_rates, passed, failed}. This
is the "your current setup passes N" clause of the first-five-minutes sentence. Idempotent: an
existing baseline.json is reused unless force; a Harbor failure is captured, not raised.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from ..harbor import jobs
from ..harbor import run as run_mod
from .writes import atomic_write_json

AGENT_PATH = "touchstone.harbor.agent:TouchstoneAgent"


def _agent_cfg(out: Path) -> dict | None:
    path = out / "agent" / "agent.toml"
    if not path.is_file():
        return None
    agent = tomllib.loads(path.read_text(encoding="utf-8")).get("agent", {})
    return {"mode": agent.get("mode", "replica"), "model_default": agent.get("model_default", ""),
            "provider": agent.get("provider", "")}


def _current_tasks(out: Path) -> set:
    """The task ids currently in the dataset (tasks/*/task.toml) — the set a fresh baseline runs."""
    tasks = out / "tasks"
    return {d.name for d in tasks.glob("*") if (d / "task.toml").is_file()} if tasks.is_dir() \
        else set()


def _model_ref(cfg: dict) -> str:
    model = cfg["model_default"]
    if "/" in model or not cfg["provider"]:
        return model
    return f"{cfg['provider']}/{model}"


def _summarize(job_dir: Path, model: str, mode: str) -> dict:
    job = jobs.Job.read(job_dir)
    rates = jobs.pass_rates(job)
    passed = sorted(n for n, r in rates.items() if r >= 1.0)
    failed = sorted(n for n, r in rates.items() if r < 1.0)
    return {"job_dir": str(job_dir), "model": model, "mode": mode, "pass_rates": rates,
            "rewards": jobs.mean_rewards(job), "passed": passed, "failed": failed}


def _reusable(existing: Path, out: Path, force: bool) -> dict | None:
    """The existing baseline when it already ran every current gated task; else None (re-run). A
    baseline that predates newly added tasks (its set != the current gated set) is not reused."""
    if force or not existing.exists():
        return None
    data = json.loads(existing.read_text(encoding="utf-8"))
    return data if not (_current_tasks(out) - set(data.get("pass_rates", {}))) else None


def _baseline_extra(cfg: dict, model: str, out: Path, settings) -> list[str]:
    extra = ["--ak", f"mode={cfg['mode']}"]
    if run_mod.dataset_is_multi_turn(out):  # conversational tasks need Harbor's simulated user
        extra += run_mod.simulated_user_args(settings.survey_user_agent,
                                             settings.survey_user_model or model)
    return extra


def run_baseline(repo, env_result: dict, settings, force: bool = False,
                 skip: bool = False) -> dict | None:
    """Run the agent under test over the gated tasks and write baseline.json. None when skipped or
    when there is no packaged agent/model; {"error": ...} when the Harbor run fails."""
    out = Path(repo) / "touchstone"
    cfg = _agent_cfg(out)
    if skip or cfg is None or not cfg["model_default"]:
        return None
    existing = out / "baseline.json"
    reused = _reusable(existing, out, force)
    if reused is not None:
        return reused
    model = _model_ref(cfg)
    try:
        job_dir = run_mod.run(out, AGENT_PATH, model=model, jobs_dir=out / "jobs",
                              extra_args=_baseline_extra(cfg, model, out, settings),
                              settings=settings)
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        return {"error": str(exc)[:2000], "model": model, "mode": cfg["mode"]}
    data = _summarize(job_dir, model, cfg["mode"])
    atomic_write_json(existing, data)
    return data
