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


def _model_ref(cfg: dict) -> str:
    model = cfg["model_default"]
    if "/" in model or not cfg["provider"]:
        return model
    return f"{cfg['provider']}/{model}"


def _summarize(job_dir: Path, model: str, mode: str) -> dict:
    rates = jobs.pass_rates(jobs.Job.read(job_dir))
    passed = sorted(n for n, r in rates.items() if r >= 1.0)
    failed = sorted(n for n, r in rates.items() if r < 1.0)
    return {"job_dir": str(job_dir), "model": model, "mode": mode, "pass_rates": rates,
            "passed": passed, "failed": failed}


def run_baseline(repo, env_result: dict, settings, force: bool = False,
                 skip: bool = False) -> dict | None:
    """Run the agent under test over the gated tasks and write baseline.json. None when skipped or
    when there is no packaged agent/model; {"error": ...} when the Harbor run fails."""
    out = Path(repo) / "touchstone"
    cfg = _agent_cfg(out)
    if skip or cfg is None or not cfg["model_default"]:
        return None
    existing = out / "baseline.json"
    if existing.exists() and not force:
        return json.loads(existing.read_text(encoding="utf-8"))
    model = _model_ref(cfg)
    try:
        job_dir = run_mod.run(out, AGENT_PATH, model=model, jobs_dir=out / "jobs",
                              extra_args=["--ak", f"mode={cfg['mode']}"], settings=settings)
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        return {"error": str(exc)[:2000], "model": model, "mode": cfg["mode"]}
    data = _summarize(job_dir, model, cfg["mode"])
    atomic_write_json(existing, data)
    return data
