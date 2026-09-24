"""Teacher demonstrations — a correct, verified answer for each frontier task.

For each task the teacher (a frontier model) writes a reply; it is accepted only when it passes the
task's own hard checks (verifiable — an unverified demo trains nothing). Accepted demos are stored
as run results under `model_spec = "teacher:<spec>"`, so a preference pair forms naturally between
the teacher's passing reply and the student's failing one. For a `needs_solution` task — a failure
with no oracle — an accepted demo is also written as the task's `reference.json` with
`reference_from = "teacher:<spec>"`, so the oracle gate passes and the task turns active. A human-
recorded reference on an active task is never overwritten (only needs_solution tasks are adopted).
"""

from __future__ import annotations

from .. import store
from .. import tasks as tasks_mod
from ..checks import Target, evaluate, passes
from ..config import Settings, load_settings
from ._providers import provider_or_none


def _demo_reply(provider, task: tasks_mod.Task):
    messages = (task.context or {}).get("messages", [])
    tools = (task.context or {}).get("tools") or None
    try:
        return provider.chat(messages, tools=tools)
    except Exception:  # a teacher failure just means no demo for this task
        return None


def _demo_passes(task: tasks_mod.Task, reply, judge_provider) -> bool:
    target = Target(output_text=reply.content or "", tool_calls=reply.tool_calls or [],
                    reference=task.reference)
    return passes(evaluate(task.checks, target, judge_provider=judge_provider), task.checks)


def _adopt_reference(root, task: tasks_mod.Task, reply, model_spec: str) -> None:
    """Write a needs_solution task's oracle from a verified teacher demo (turns it active)."""
    task.reference = {"content": reply.content or "", "tool_calls": reply.tool_calls or []}
    task.reference_from = model_spec
    tasks_mod.write_task(root, task)


def _reply_output(reply) -> dict:
    return {"content": reply.content or "", "tool_calls": reply.tool_calls or []}


def teach(
    conn,
    root: str,
    names: list[str],
    teacher_spec: str | None = None,
    *,
    settings: Settings | None = None,
    teacher_provider=None,
    judge_provider=None,
) -> dict:
    """Produce and store gated teacher demonstrations for `names`. Returns accepted/failed lists."""
    settings = settings or load_settings()
    teacher_spec = teacher_spec or settings.agent_provider
    provider = teacher_provider or provider_or_none(teacher_spec, settings)
    model_spec = f"teacher:{teacher_spec}"
    if provider is None:
        return {"teacher": model_spec, "run": None, "accepted": [], "failed": list(names)}

    run = store.insert_run(conn, store.Run(target="teacher", model_spec=model_spec))
    accepted, failed = [], []
    for name in names:
        ok = _teach_one(conn, root, run.id, name, provider, judge_provider, model_spec)
        (accepted if ok else failed).append(name)
    store.finish_run(conn, run.id)
    return {"teacher": model_spec, "run": run.id, "accepted": accepted, "failed": failed}


def _teach_one(conn, root, run_id, name, provider, judge_provider, model_spec) -> bool:
    """Store a gated demo for one task; adopt it as the oracle when the task needs a solution."""
    task = tasks_mod.get_task(root, name)
    reply = _demo_reply(provider, task) if task is not None else None
    if reply is None or not _demo_passes(task, reply, judge_provider):
        return False
    store.insert_result(conn, store.Result(
        run_id=run_id, task=name, passed=1, reward=1.0, output=_reply_output(reply)))
    if task.status == "needs_solution":
        _adopt_reference(root, task, reply, model_spec)
    return True


def demo_replies(conn, run_id: str) -> dict[str, dict]:
    """The stored teacher demo outputs for a run, keyed by task (for dataset assembly)."""
    return {r.task: r.output for r in store.list_results(conn, run_id) if r.output}


def escalate(conn, root, names: list[str], hub_base_url: str = "") -> list[str]:
    """Open an interview room on each contested task; return the room URLs to print."""
    from ..interview import rooms

    urls = []
    for name in names:
        room = rooms.open(conn, name, f"resolve frontier task {name}")
        urls.append(f"{hub_base_url}/rooms/{room.id}")
    return urls
