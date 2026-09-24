"""Distill — package exactly the student's failures into training data, then hand off to a backend.

Distill draws the frontier for the student, has the teacher produce a verified demonstration for
each frontier task (stored under the benchmark so it pairs against the student's failing reply), and
runs `train.datasets.prepare` restricted to that frontier: SFT from the verified references,
preference pairs (teacher-chosen over student-rejected), and RL tasks (the frontier tasks with their
checks). The backend (Null by default) writes its plan/config. When the teacher fails the gate on
every frontier task the loop escalates — interview rooms open on those tasks and their URLs are
printed. The plan file records the loop step so a reader knows to Sample again after training.
"""

from __future__ import annotations

from pathlib import Path

from ..bench import benchmark as benchmark_mod
from ..config import Settings, load_settings
from ..train import TrainConfig, default_out_dir, trainer_for
from ..train.datasets import prepare
from ._providers import provider_or_none
from .frontier import frontier, write_loop_state
from .teach import escalate, teach


def _plan_line(front: list[str], demos: int) -> str:
    return (f"Sample found {len(front)} frontier task(s); Distill packaged {demos} teacher "
            "demo(s); run Sample again after training to prove the gap closed.")


def _record_loop_step(out_dir: Path, line: str) -> None:
    """Append the loop step to the plan file so a reader knows the next move (Sample again)."""
    plan = out_dir / "train_plan.md"
    text = f"\n## Loop step\n\n{line}\n"
    if plan.exists():
        plan.write_text(plan.read_text(encoding="utf-8") + text, encoding="utf-8")
    else:
        (out_dir / "loop_step.md").write_text(text, encoding="utf-8")


def distill(
    conn,
    root: str,
    benchmark_name: str,
    student_spec: str,
    *,
    teacher_spec: str | None = None,
    backend: str = "null",
    out_dir: str | Path | None = None,
    base_model: str | None = None,
    settings: Settings | None = None,
    teacher_provider=None,
    judge_provider=None,
) -> dict:
    """Package the frontier into training data and submit it to `backend`."""
    settings = settings or load_settings()
    teacher_spec = teacher_spec or settings.agent_provider
    if teacher_provider is None:
        teacher_provider = provider_or_none(teacher_spec, settings)

    # Distill packages this benchmark's frontier: the tasks it resolves that the student fails.
    # (A glob benchmark already includes the Sample variants; an explicit one does not.)
    bench_names = {d.name for d in benchmark_mod.resolve(root, benchmark_name)}
    front = [t for t in frontier(conn, root, student_spec) if t in bench_names]
    if not front:
        write_loop_state(root, benchmark_name, student=student_spec, last_distill=_now(),
                         frontier_size=0)
        return {"benchmark": benchmark_name, "student": student_spec, "frontier": [],
                "demos": [], "escalated": [], "out_dir": None, "counts": {},
                "backend": backend, "status": "empty",
                "plan": "frontier is empty — nothing to distill; the student matches the incumbent"}

    demos = teach(conn, root, front, teacher_spec, target=benchmark_name, settings=settings,
                  teacher_provider=teacher_provider, judge_provider=judge_provider)
    escalated = escalate(conn, root, demos["failed"]) if not demos["accepted"] else []

    out = Path(out_dir) if out_dir else default_out_dir(settings.db_path, benchmark_name)
    bundle = prepare(conn, root, benchmark_name, out, only=set(front))
    config = TrainConfig(base_model=base_model) if base_model else TrainConfig()
    handle = trainer_for(backend).submit(bundle, config)

    line = _plan_line(front, len(demos["accepted"]))
    _record_loop_step(bundle.out_dir, line)
    write_loop_state(root, benchmark_name, student=student_spec, last_distill=_now(),
                     frontier_size=len(front))
    return {
        "benchmark": benchmark_name,
        "student": student_spec,
        "teacher": demos["teacher"],
        "frontier": front,
        "demos": demos["accepted"],
        "escalated": escalated,
        "out_dir": str(bundle.out_dir),
        "counts": bundle.counts,
        "backend": handle.backend,
        "status": handle.status,
        "plan": line,
    }


def _now() -> str:
    from .. import store

    return store.now()
