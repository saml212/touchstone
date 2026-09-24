"""Sample — run the student, then probe its edges with teacher-generated variants.

Sample runs the candidate (`student_spec`) on a benchmark, records difficulty, and for each active
task the student fails asks the teacher for up to N perturbed variants (a paraphrased user turn, a
renamed tool, or a tightened constraint — the teacher sees the privileged source, the student later
sees only the task). Every variant is a new task directory carrying `parent_task` / `generated_by`
provenance and a `generation.json` (teacher spec, method, the SHA-256 of the parent context — hashes
only, never contents, and the two validation rewards). A variant must pass the oracle/nop gate to
count; the rest are discarded. The student is then run on the surviving variants and the frontier —
every task with `0 < pass_rate < 1` plus the only-incumbent set — is returned. Deterministic with
the `scripted` provider: no teacher JSON parses to a mechanical paraphrase, so a variant always
appears.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil

import tomli_w

from .. import store
from .. import tasks as tasks_mod
from ..bench import benchmark, runner
from ..config import Settings, load_settings
from ..llm.prompt import extract_json
from ..messages import text_of
from .frontier import frontier_split, mean_pass_rate, record_run, write_loop_state

_VARIANTS_TARGET = "__sample_variants__"


# ---- teacher instructions --------------------------------------------------


def _teacher_prompt(task: tasks_mod.Task, n: int) -> list[dict]:
    user = _last_user(task)
    tools = [t.get("function", {}).get("name") or t.get("name")
             for t in (task.context or {}).get("tools") or []]
    system = (
        "You harden an eval by perturbing a task so a weaker model is more likely to trip. "
        "Reply with ONLY a JSON array of at most "
        f"{n} objects, each one of: "
        '{"method":"paraphrase","user":"<reworded user turn, same intent>"} | '
        '{"method":"rename_tool","from":"<tool>","to":"<new name>"} | '
        '{"method":"tighten","contains":"<phrase the good reply already contains>"}.'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"USER TURN:\n{user}\n\nTOOLS: {tools}"}]


def _last_user(task: tasks_mod.Task) -> str:
    for m in reversed((task.context or {}).get("messages", [])):
        if m.get("role") == "user":
            return text_of(m)
    return ""


def _teacher_instructions(provider, task: tasks_mod.Task, n: int) -> list[dict]:
    """Up to N perturbation instructions from the teacher, or one mechanical paraphrase offline."""
    parsed = _ask_teacher(provider, task, n) if provider is not None else []
    usable = [i for i in parsed if isinstance(i, dict) and i.get("method") in _APPLIERS]
    if usable:
        return usable[:n]
    return [{"method": "paraphrase", "user": f"{_last_user(task)} Please help me with this."}]


def _ask_teacher(provider, task: tasks_mod.Task, n: int) -> list:
    try:
        reply = provider.chat(_teacher_prompt(task, n))
        items = json.loads(extract_json(reply.content or "") or "null")
    except Exception:  # a teacher failure degrades to the mechanical fallback, never crashes
        return []
    return items if isinstance(items, list) else []


# ---- variant construction --------------------------------------------------


def _apply_paraphrase(v: tasks_mod.Task, instr: dict) -> bool:
    text = instr.get("user")
    if not isinstance(text, str) or not text:
        return False
    for m in reversed((v.context or {}).get("messages", [])):
        if m.get("role") == "user":
            m["content"] = text
            return True
    return False


def _rename_in_checks(checks, old: str, new: str) -> None:
    for c in checks:
        if c.params.get("name") == old:
            c.params["name"] = new
        order = c.params.get("order")
        if isinstance(order, list):
            c.params["order"] = [new if x == old else x for x in order]


def _apply_rename_tool(v: tasks_mod.Task, instr: dict) -> bool:
    old, new = instr.get("from"), instr.get("to")
    if not (isinstance(old, str) and isinstance(new, str) and old and new):
        return False
    for tool in (v.context or {}).get("tools") or []:
        fn = tool.get("function", tool)
        if fn.get("name") == old:
            fn["name"] = new
    for tc in (v.reference or {}).get("tool_calls") or []:
        if tc.get("name") == old:
            tc["name"] = new
    _rename_in_checks(v.checks, old, new)
    return True


def _apply_tighten(v: tasks_mod.Task, instr: dict) -> bool:
    phrase = instr.get("contains")
    if not isinstance(phrase, str) or not phrase:
        return False
    from ..checks import Check

    v.checks.append(Check(kind="contains", params={"values": [phrase], "mode": "any"},
                          name=f"mentions {phrase}"[:60], severity="hard", source="mined",
                          because="teacher tightened the constraint"))
    return True


_APPLIERS = {"paraphrase": _apply_paraphrase, "rename_tool": _apply_rename_tool,
             "tighten": _apply_tighten}


def _variant_of(parent: tasks_mod.Task, name: str, teacher_spec: str) -> tasks_mod.Task:
    return tasks_mod.Task(
        name=name, kind="variant", tags=list(parent.tags or []),
        description=f"Variant of {parent.name}.",
        context=copy.deepcopy(parent.context or {}),
        reference=copy.deepcopy(parent.reference),
        checks=[copy.deepcopy(c) for c in parent.checks],
        parent_task=parent.name, generated_by=teacher_spec)


def _context_sha256(task: tasks_mod.Task) -> str:
    payload = json.dumps(task.context or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_generation(root, variant: tasks_mod.Task, parent: tasks_mod.Task,
                      teacher_spec: str, method: str) -> None:
    provenance = {
        "parent_task": parent.name,
        "teacher": teacher_spec,
        "method": method,
        "parent_context_sha256": _context_sha256(parent),
        "created_at": store.now(),
        "validation": {
            "oracle_pass": variant.status != "needs_solution",
            "nop_fail": variant.status == "active",
        },
    }
    path = tasks_mod.tasks_dir(root) / variant.name / "generation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_variant(root, parent, instr, index, teacher_spec) -> str | None:
    """Build, gate and persist one variant; return its name when active, else discard it."""
    method = instr["method"]
    variant = _variant_of(parent, f"{parent.name}--v{index}", teacher_spec)
    if not _APPLIERS[method](variant, instr):
        return None
    tasks_mod.write_task(root, variant, preserve=False)
    variant = tasks_mod.read_task(tasks_mod.tasks_dir(root) / variant.name)
    _write_generation(root, variant, parent, teacher_spec, method)
    if variant.status != "active":
        shutil.rmtree(tasks_mod.tasks_dir(root) / variant.name, ignore_errors=True)
        return None
    return variant.name


def _variants_of_parent(root, provider, parent, teacher_spec, n) -> list[str]:
    names = []
    for i, instr in enumerate(_teacher_instructions(provider, parent, n)):
        name = _make_variant(root, parent, instr, i, teacher_spec)
        if name is not None:
            names.append(name)
    return names


def _generate_variants(root, provider, parent_names, teacher_spec, n) -> list[str]:
    created: list[str] = []
    for parent_name in parent_names:
        parent = tasks_mod.get_task(root, parent_name)
        if parent is None or parent.parent_task:  # never perturb a variant of a variant
            continue
        created += _variants_of_parent(root, provider, parent, teacher_spec, n)
    return created


# ---- orchestration ---------------------------------------------------------


def _run_on(conn, root, target, student_spec, concurrency, timeout, judge_provider) -> store.Run:
    run = runner.run(conn, root, target, student_spec, concurrency=concurrency,
                     timeout=timeout, judge_provider=judge_provider)
    record_run(conn, root, run)
    return run


def _run_on_tasks(conn, root, names, student_spec, concurrency, timeout, judge_provider):
    """Run the student over an explicit task list via a throwaway benchmark spec."""
    if not names:
        return None
    path = benchmark.spec_path(root, _VARIANTS_TARGET)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps({"tasks": [f"tasks/{n}" for n in names]}), encoding="utf-8")
    try:
        return _run_on(conn, root, _VARIANTS_TARGET, student_spec, concurrency, timeout,
                       judge_provider)
    finally:
        path.unlink(missing_ok=True)


def _teacher_provider(teacher_spec: str, settings: Settings):
    from ..llm import provider_from_spec

    try:
        return provider_from_spec(teacher_spec, settings)
    except Exception:  # no key/binary: fall back to mechanical variants
        return None


def sample(
    conn,
    root: str,
    benchmark_name: str,
    student_spec: str,
    *,
    teacher_spec: str | None = None,
    variants: int = 1,
    settings: Settings | None = None,
    teacher_provider=None,
    judge_provider=None,
    concurrency: int = 4,
    timeout: float = 60,
) -> dict:
    """Run the student, generate teacher variants of its failures, and return the frontier."""
    settings = settings or load_settings()
    teacher_spec = teacher_spec or settings.agent_provider
    if teacher_provider is None:
        teacher_provider = _teacher_provider(teacher_spec, settings)

    prev = mean_pass_rate(conn, root, student_spec)
    student_run = _run_on(conn, root, benchmark_name, student_spec, concurrency, timeout,
                          judge_provider)
    failing = [r.task for r in store.list_results(conn, student_run.id) if not r.passed]

    created = _generate_variants(root, teacher_provider, failing, teacher_spec, variants)
    variant_run = _run_on_tasks(conn, root, created, student_spec, concurrency, timeout,
                                judge_provider)

    split = frontier_split(conn, root, student_spec)
    front = sorted(set(split["learnability"]) | set(split["only_incumbent"]))
    write_loop_state(root, benchmark_name, student=student_spec,
                                  last_sample=store.now(), frontier_size=len(front),
                                  prev_pass_rate=round(prev, 4))
    return {
        "benchmark": benchmark_name,
        "student": student_spec,
        "teacher": teacher_spec,
        "student_run": student_run.id,
        "variant_run": variant_run.id if variant_run else None,
        "failing": failing,
        "variants_created": created,
        "frontier": front,
        "frontier_split": split,
    }
