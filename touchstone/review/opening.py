"""The review room's opening statement — what jobs the agent handles, what the latest run passed
or errored, and what to walk through next. Split out of `review.agent` so the state machine stays
small and the counting stays unit-testable on its own.
"""

from __future__ import annotations

from pathlib import Path

from ..survey.report import passes_counts
from .facts import (
    _baseline_sets,
    _gated_tasks,
    _job_labels,
    _join,
    errored_sentence,
    latest_run_errors,
    latest_run_sets,
)


def statement(dataset_dir: Path, jobs_dir: Path, topic: str, counts: dict) -> str:
    """The room's opening line. `counts` is the review-trial bucket count for the offer."""
    jobs_done = _job_labels(dataset_dir)
    built = _gated_tasks(dataset_dir)
    errs = latest_run_errors(jobs_dir)
    if not jobs_done and not built and not errs:
        return (f"Let's review “{topic}”. I couldn't find a benchmark here "
                "yet — run `touchstone survey` first, then reopen this room.")
    does = _join(jobs_done) or "several jobs"
    if errs and errs["errored"]:  # a failed run is never "everything passes"
        return f"Your agent handles {does}; {errored_sentence(errs)}"
    return f"Your agent handles {does}; {_pass_tail(dataset_dir, jobs_dir, built)} {_offer(counts)}"


def _pass_tail(dataset_dir: Path, jobs_dir: Path, built: list[str]) -> str:
    """"N tasks, passes K of the R run [(U not run)]" — run/passed counted from the latest rewarded
    job (the run the room is about to review), so six scored trials read as "0 of the 6 run", never
    "0 of the 0 run (6 not run)" off a stale baseline. Falls back to the baseline file, then the
    plain task count, when no rewarded job exists."""
    sets = latest_run_sets(jobs_dir) or _baseline_sets(dataset_dir)
    if sets is None:
        return f"{len(built)} tasks."
    n, k, r, u = passes_counts(built, sets["ran"], sets["passed"])
    if u > 0:
        return f"{n} tasks, your current setup passes {k} of the {r} run ({u} not run yet)."
    return f"{n} tasks, your current setup passes {k} of the {r} run."


def _offer(counts: dict) -> str:
    """The opening's offer, chosen from what is actually there to review."""
    if counts["unsure"]:
        return "Want to walk through the trials the verifier was unsure about?"
    if counts["disagree"]:
        return "Want to look at the tasks where the models disagree?"
    if counts["unreviewed"]:
        return "Want to walk through the ones nobody has reviewed yet?"
    return "everything passes — want to spot-check a few?"
