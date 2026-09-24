"""Task directories: the source of truth for authored tasks and their checks.

One directory per task, laid out exactly as Harbor expects, so `harbor run -p tasks/<name>` runs
it unchanged. `task.toml` is a strict superset of Harbor's 1.4 schema — Harbor's own sections plus
everything Touchstone-specific under the free-form `[metadata.touchstone]` table, including the
`[[metadata.touchstone.check]]` blocks a PM reads aloud. `write_task` also drops the checks the
container verifier needs under `tests/` (which is the only tree Harbor mounts into the verifier),
so each directory is self-contained.

`write_task` re-materialises a task in place: interview/manual check blocks and the measured
difficulty cache already on disk are preserved, mined/policy blocks are replaced. A task is a work
queue, not a pass/fail flag — `validate` sorts it into one of three statuses at write time:
`active` (its reference scores 1 as the oracle and an empty reply scores 0 as the nop),
`needs_checks` (an empty reply already passes every hard check, so the task measures nothing yet),
or `needs_solution` (the recorded reply fails its own hard checks — a failure with no oracle, so a
teacher or a human must supply one). Only `active` tasks enter a benchmark.
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from .checks import Check, Target, evaluate, passes
from .messages import context_text, text_of

_CHECKS_SRC = Path(__file__).resolve().parent / "checks"
PRESERVE_SOURCES = frozenset({"interview", "manual"})

DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
# jsonschema/simpleeval back the json_schema and expr check kinds; harmless otherwise.
RUN pip install --no-cache-dir jsonschema simpleeval
"""

TEST_SH = """\
#!/bin/bash
# Score /app/output.json against the task's checks; write /logs/verifier/reward.txt + rewards.json.
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/verify.py
"""

GITIGNORE = "__pycache__/\n*.pyc\n.DS_Store\n"


@dataclass
class Task:
    name: str
    episode_id: str | None = None
    cut_span_id: str | None = None
    kind: str = "replay"
    tags: list = field(default_factory=list)
    status: str = "active"  # active | needs_checks | needs_solution
    status_reason: str = ""  # why the task landed in its queue
    description: str = ""
    context: dict = field(default_factory=dict)  # {messages, tools}
    reference: dict | None = None
    checks: list[Check] = field(default_factory=list)
    parent_task: str | None = None  # the task a Sample variant was generated from
    generated_by: str | None = None  # the teacher spec that generated a variant
    reference_from: str | None = None  # a teacher spec, when the oracle is a teacher demo
    difficulty: dict = field(default_factory=dict)  # measured pass rate per model_spec (cache)


_SLUG_CAP = 80  # leaves room for "-turn-<n>[-<span6>]" under the 100-char cap


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return s or "task"


def task_name(episode, span, *, turn: int, disambiguate: bool = False) -> str:
    """Human-readable, deterministic, ≤100-char, filesystem-safe name for a cut task.

    `<episode-slug>-turn-<n>`, where `n` is the 1-based assistant-turn index within the episode —
    a name a human can read and a benchmark can list. When several episodes share the same slug
    the bare name would collide across them, so `disambiguate` appends the first six characters of
    the span id to keep every directory unique. Deterministic for a given episode/turn, so
    re-mining and `tasks sync` rebuild the identical name.
    """
    name = f"{_slug(episode.name)[:_SLUG_CAP]}-turn-{turn}"
    if disambiguate:
        name = f"{name}-{span.id.lower()[:6]}"
    return name[:100]


def tasks_dir(root: str | Path) -> Path:
    return Path(root) / "tasks"


# ---- reference gate + status ------------------------------------------------


def _reward(checks: list[Check], output_text: str, tool_calls: list[dict], reference,
            ctx_text: str = "") -> float:
    gradable = [c for c in checks if c.kind != "judge"]  # judge needs an LLM; skipped offline
    target = Target(output_text=output_text, tool_calls=tool_calls, reference=reference,
                    context_text=ctx_text)
    return 1.0 if passes(evaluate(gradable, target), gradable) else 0.0


def validate(task: Task) -> tuple[str, str]:
    """(status, reason): sort a task into its work queue by the oracle/nop gate.

    `needs_solution` — the recorded reply fails its own hard checks (no oracle yet).
    `needs_checks`   — an empty reply already passes every hard check (measures nothing yet).
    `active`         — oracle scores 1 and nop scores 0. Only these enter a benchmark.
    """
    ref = task.reference or {"content": "", "tool_calls": []}
    calls = ref.get("tool_calls") or []
    ctx = context_text(task.context)
    if _reward(task.checks, text_of(ref), calls, ref, ctx) < 1.0:
        return "needs_solution", "the recorded reply fails its own hard checks"
    if _reward(task.checks, "", [], ref, ctx) >= 1.0:
        return "needs_checks", "an empty reply already passes every hard check"
    return "active", ""


# ---- write ------------------------------------------------------------------


def _dedupe(checks: list[Check]) -> list[Check]:
    seen: set[tuple] = set()
    out: list[Check] = []
    for c in checks:
        key = (c.kind, json.dumps(c.params or {}, sort_keys=True), c.name)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


_OPTIONAL_TS_KEYS = ("episode_id", "cut_span_id", "status_reason",
                     "parent_task", "generated_by", "reference_from")


def _touchstone_meta(task: Task) -> dict:
    ts: dict = {
        "kind": task.kind,
        "tags": list(task.tags or []),
        "status": task.status,
        "context_file": "context.json",
        "reference_file": "reference.json",
    }
    for key in _OPTIONAL_TS_KEYS:
        if getattr(task, key):
            ts[key] = getattr(task, key)
    if task.difficulty:
        ts["difficulty"] = dict(task.difficulty)
    ts["check"] = [c.to_toml() for c in task.checks]
    return ts


def _task_toml(task: Task) -> str:
    doc = {
        "schema_version": "1.4",
        "task": {
            "name": f"touchstone/{task.name}",
            "version": "1.0.0",
            "description": task.description or f"Replay task {task.name}.",
            "keywords": list(task.tags or []),
        },
        "metadata": {"touchstone": _touchstone_meta(task)},
        "verifier": {"timeout_sec": 900.0},
        "agent": {"timeout_sec": 900.0},
        "environment": {"build_timeout_sec": 600.0},
    }
    return tomli_w.dumps(doc)


def _instruction_md(task: Task) -> str:
    messages = (task.context or {}).get("messages", [])
    tools = (task.context or {}).get("tools") or []
    lines = [f"# {task.name}", ""]
    systems = [m for m in messages if m.get("role") == "system"]
    if systems:
        lines += ["## System", *(text_of(m) for m in systems), ""]
    convo = [m for m in messages if m.get("role") != "system"]
    if convo:
        turns = [f"**{m.get('role', 'user')}:** {text_of(m)}" for m in convo]
        lines += ["## Conversation so far", *turns, ""]
    if tools:
        lines += ["## Tools available (JSON schemas)", "```json",
                  json.dumps(tools, ensure_ascii=False, indent=2), "```", ""]
    lines += [
        "## Your task",
        "Produce the assistant's next reply for the conversation above.",
        "",
        "Write your final answer to `/app/output.json` as a JSON object:",
        "```json",
        '{"content": "<your reply text>", '
        '"tool_calls": [{"name": "<tool>", "arguments": "<json string>"}]}',
        "```",
        "Use an empty list for `tool_calls` if you call no tools.",
    ]
    return "\n".join(lines) + "\n"


def _solve_sh(task: Task) -> str:
    reference = task.reference or {"content": "", "tool_calls": []}
    payload = json.dumps(reference, ensure_ascii=False)
    return (
        "#!/bin/bash\n"
        "# Oracle: write the recorded reference answer as the output.\n"
        "set -euo pipefail\n"
        f"cat > /app/output.json <<'TOUCHSTONE_EOF'\n{payload}\nTOUCHSTONE_EOF\n"
    )


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _write_tests(task_dir: Path, task: Task) -> None:
    """The verifier tree — the only files Harbor mounts into the verifier container.

    A `task.toml` copy (non-judge checks; judge needs an LLM) and `reference.json` copy travel
    here so `verify.py` reads its checks from `task.toml`, exactly as the root file does.
    """
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    _write(tests / "test.sh", TEST_SH, executable=True)
    shutil.copyfile(Path(__file__).resolve().parent / "_verify.py", tests / "verify.py")
    container = Task(name=task.name, kind=task.kind, tags=task.tags, status=task.status,
                     reference=task.reference,
                     checks=[c for c in task.checks if c.kind != "judge"])
    _write(tests / "task.toml", _task_toml(container))
    _write(tests / "reference.json",
           json.dumps(task.reference or {"content": "", "tool_calls": []}, ensure_ascii=False))
    # The flattened user/system text an expr check reads as `context_text` inside the container.
    _write(tests / "context_text.txt", context_text(task.context))
    vendor = tests / "touchstone_checks"
    vendor.mkdir(parents=True, exist_ok=True)
    _write(vendor / "__init__.py", "")
    for name in ("dsl.py", "run.py"):
        shutil.copyfile(_CHECKS_SRC / name, vendor / name)


def preserved_checks(root: str | Path, name: str) -> list[Check]:
    """The interview/manual check blocks already authored on a task (survive re-materialisation)."""
    task_dir = tasks_dir(root) / name
    if not (task_dir / "task.toml").exists():
        return []
    return [c for c in read_task(task_dir).checks if c.source in PRESERVE_SOURCES]


def preserved_difficulty(root: str | Path, name: str) -> dict:
    """The measured difficulty cache already on a task (survives re-materialisation)."""
    task_dir = tasks_dir(root) / name
    if not (task_dir / "task.toml").exists():
        return {}
    return dict(read_task(task_dir).difficulty)


def write_task(root: str | Path, task: Task, *, preserve: bool = True) -> Path:
    """Write (or re-materialise) `task` under `root/tasks/<name>/`.

    With `preserve` (the default), interview/manual check blocks already on disk are kept and
    mined/policy blocks are replaced — the sync semantics. Set `preserve=False` to write
    `task.checks` verbatim (used when removing a check). The task is gated (oracle=1, nop=0).
    """
    task_dir = tasks_dir(root) / task.name
    kept = preserved_checks(root, task.name) if preserve else []
    task.checks = _dedupe(list(task.checks) + kept)
    if preserve:
        task.difficulty = {**preserved_difficulty(root, task.name), **(task.difficulty or {})}
    task.status, task.status_reason = validate(task)

    task_dir.mkdir(parents=True, exist_ok=True)
    _write(task_dir / "task.toml", _task_toml(task))
    _write(task_dir / "instruction.md", _instruction_md(task))
    _write(task_dir / "context.json",
           json.dumps(task.context or {}, ensure_ascii=False, indent=2) + "\n")
    _write(task_dir / "reference.json",
           json.dumps(task.reference or {"content": "", "tool_calls": []},
                      ensure_ascii=False, indent=2) + "\n")
    _write(task_dir / "environment" / "Dockerfile", DOCKERFILE)
    _write(task_dir / "solution" / "solve.sh", _solve_sh(task), executable=True)
    _write(task_dir / ".gitignore", GITIGNORE)
    _write_tests(task_dir, task)
    return task_dir


def append_check(root: str | Path, name: str, check: Check) -> Path:
    """Add one check (e.g. an interview commit) to an existing task, idempotently."""
    task = read_task(tasks_dir(root) / name)
    task.checks.append(check)
    return write_task(root, task)


def remove_check(root: str | Path, name: str, check_name: str) -> Task:
    """Drop the check block with `check_name` from a task and rewrite it verbatim."""
    task = read_task(tasks_dir(root) / name)
    task.checks = [c for c in task.checks if c.name != check_name]
    write_task(root, task, preserve=False)
    return read_task(tasks_dir(root) / name)


# ---- read -------------------------------------------------------------------


def _load_json(task_dir: Path, filename: str, default):
    path = task_dir / filename
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_task(task_dir: str | Path) -> Task:
    task_dir = Path(task_dir)
    doc = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    ts = doc.get("metadata", {}).get("touchstone", {})
    return Task(
        name=task_dir.name,
        episode_id=ts.get("episode_id"),
        cut_span_id=ts.get("cut_span_id"),
        kind=ts.get("kind", "replay"),
        tags=list(ts.get("tags", [])),
        status=ts.get("status", "active"),
        status_reason=ts.get("status_reason", ts.get("reason", "")),
        description=doc.get("task", {}).get("description", ""),
        context=_load_json(task_dir, ts.get("context_file", "context.json"), {}),
        reference=_load_json(task_dir, ts.get("reference_file", "reference.json"), None),
        checks=[Check.from_toml(b) for b in ts.get("check", [])],
        parent_task=ts.get("parent_task"),
        generated_by=ts.get("generated_by"),
        reference_from=ts.get("reference_from"),
        difficulty=dict(ts.get("difficulty", {})),
    )


def list_tasks(
    root: str | Path, *, tag: str | None = None, active_only: bool = False
) -> list[Task]:
    base = tasks_dir(root)
    if not base.exists():
        return []
    tasks = []
    for d in sorted(base.iterdir()):
        if not (d.is_dir() and (d / "task.toml").exists()):
            continue
        try:
            tasks.append(read_task(d))
        except (tomllib.TOMLDecodeError, OSError, ValueError):
            continue  # one hand-corrupted task must not break enumeration of the rest
    if tag is not None:
        tasks = [t for t in tasks if tag in (t.tags or [])]
    if active_only:
        tasks = [t for t in tasks if t.status == "active"]
    return tasks


def get_task(root: str | Path, name: str) -> Task | None:
    task_dir = tasks_dir(root) / name
    return read_task(task_dir) if (task_dir / "task.toml").exists() else None


# ---- migration --------------------------------------------------------------

_VARIANT_SUFFIX = re.compile(r"-+v(\d+)$")


def _replay_renames(existing: list[Task], new_by_key: dict[tuple, str]) -> dict[str, str]:
    """Replay dirs whose scheme changed, matched to their new name by (episode, span)."""
    out: dict[str, str] = {}
    for task in existing:
        new = new_by_key.get((task.episode_id, task.cut_span_id))
        if task.kind != "variant" and new and new != task.name:
            out[task.name] = new
    return out


def _variant_renames(existing: list[Task], parents: dict[str, str]) -> dict[str, str]:
    """Variant dirs follow their parent's new name and drop the legacy `--v` separator."""
    out: dict[str, str] = {}
    for task in existing:
        m = _VARIANT_SUFFIX.search(task.name)
        if task.kind != "variant" or not task.parent_task or not m:
            continue
        new = f"{parents.get(task.parent_task, task.parent_task)}-v{m.group(1)}"
        if new != task.name:
            out[task.name] = new
    return out


def _plan_renames(existing: list[Task], new_by_key: dict[tuple, str]) -> dict[str, str]:
    """old dir name -> new dir name for every replay task and its variants."""
    renames = _replay_renames(existing, new_by_key)
    renames.update(_variant_renames(existing, renames))
    return renames


def _retarget_one_variant(root: str | Path, task_dir: Path, task: Task, parents: dict[str, str]):
    """Point one moved variant at its parent's new name, on disk and in generation.json."""
    new_parent = parents.get(task.parent_task or "", task.parent_task)
    if task.kind != "variant" or new_parent == task.parent_task:
        return
    task.parent_task = new_parent
    write_task(root, task, preserve=False)
    data = _load_json(task_dir, "generation.json", None)
    if isinstance(data, dict):
        data["parent_task"] = new_parent
        (task_dir / "generation.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _retarget_variants(root: str | Path, renames: dict[str, str]) -> None:
    base = tasks_dir(root)
    for new in renames.values():
        task_dir = base / new
        if (task_dir / "task.toml").exists():
            _retarget_one_variant(root, task_dir, read_task(task_dir), renames)


def _rewrite_benchmarks(root: str | Path, renames: dict[str, str]) -> None:
    """Rewrite the `tasks = [...]` list in every benchmark that names a renamed task."""
    bdir = Path(root) / "benchmarks"
    for spec in sorted(bdir.glob("*.toml")) if bdir.exists() else []:
        try:
            doc = tomllib.loads(spec.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        listed = doc.get("tasks")
        if not isinstance(listed, list):
            continue
        updated = [
            f"tasks/{renames[entry.removeprefix('tasks/')]}"
            if entry.removeprefix("tasks/") in renames
            else entry
            for entry in listed
        ]
        if updated != listed:
            doc["tasks"] = updated
            spec.write_text(tomli_w.dumps(doc), encoding="utf-8")


def migrate_names(root: str | Path, new_by_key: dict[tuple, str]) -> dict[str, str]:
    """Rename task dirs that predate the `<episode-slug>-turn-<n>` scheme, in place.

    `new_by_key` maps (episode_id, cut_span_id) -> the new replay-task name (the freshly cut
    tasks already carry it). Each old dir is moved to its new name — git sees a rename, so the
    history stays legible — variants are retargeted at their parents, and benchmark task lists are
    rewritten. Idempotent: once every dir already carries its new name the map is empty and
    nothing is touched.
    """
    base = tasks_dir(root)
    if not base.exists():
        return {}
    renames = _plan_renames(list_tasks(root), new_by_key)
    if not renames:
        return {}
    for old, new in renames.items():
        old_dir, new_dir = base / old, base / new
        if old_dir.exists() and not new_dir.exists():
            old_dir.rename(new_dir)
    _retarget_variants(root, renames)
    _rewrite_benchmarks(root, renames)
    return renames
