"""Baseline: run the packaged agent under its current model and report what it already passes.

Stage 3B fills this in (the packaged agent + a Harbor run of it over the gated tasks, producing the
"your current setup passes N" clause of the first-five-minutes sentence). For stage 3A it is a
deliberate no-op seam so the orchestrator's shape is final: it is called in order and returns None.
"""

from __future__ import annotations


def run_baseline(repo, env_result: dict, settings, force: bool = False) -> None:
    """No-op until stage 3B packages the agent and runs it. Returns None (no baseline yet)."""
    return None
