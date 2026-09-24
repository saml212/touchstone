"""Task directories: the source of truth for authored tasks and their checks.

One directory per task, laid out exactly as Harbor expects, so `harbor run -p tasks/<name>` runs
it unchanged. `task.toml` is a strict superset of Harbor's 1.4 schema — Harbor's own sections plus
everything Touchstone-specific under the free-form `[metadata.touchstone]` table, including the
`[[metadata.touchstone.check]]` blocks a PM reads aloud. `write_task` also drops the checks the
container verifier needs under `tests/` (which is the only tree Harbor mounts into the verifier),
so each directory is self-contained.

`write_task` re-materialises a task in place: interview/manual check blocks already on disk are
preserved, mined/policy blocks are replaced. A task is gated at write time — its reference must
score 1 (oracle) and an empty reply must score 0 (nop) — and marked `status = "rejected"` otherwise.
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
    status: str = "active"  # active | rejected
    reason: str = ""  # why a task was rejected
    description: str = ""
    context: dict = field(default_factory=dict)  # {messages, tools}
    reference: dict | None = None
    checks: list[Check] = field(default_factory=list)


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return s or "task"


def task_name(episode, span) -> str:
    """Stable, sortable, filesystem-safe name for the task cut at `span` of `episode`.

    Deterministic for a given episode/span: the span id (a sortable ULID) anchors it, so
    re-mining and `tasks sync` rebuild the identical directory name.
    """
    return f"{_slug(episode.name)}-{span.id.lower()}"


def tasks_dir(root: str | Path) -> Path:
    return Path(root) / "tasks"


# ---- reference gate + status ------------------------------------------------


def _reward(checks: list[Check], output_text: str, tool_calls: list[dict], reference) -> float:
    gradable = [c for c in checks if c.kind != "judge"]  # judge needs an LLM; skipped offline
    target = Target(output_text=output_text, tool_calls=tool_calls, reference=reference)
    return 1.0 if passes(evaluate(gradable, target), gradable) else 0.0


def gate(task: Task) -> tuple[str, str]:
    """(status, reason): a task counts only if its reference scores 1 and an empty reply 0."""
    ref = task.reference or {"content": "", "tool_calls": []}
    content = ref.get("content") or ""
    calls = ref.get("tool_calls") or []
    if _reward(task.checks, content, calls, ref) < 1.0:
        return "rejected", "oracle: the reference does not satisfy its own hard checks"
    if _reward(task.checks, "", [], ref) >= 1.0:
        return "rejected", "nop: an empty reply already passes every hard check"
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


def _task_toml(task: Task) -> str:
    ts: dict = {
        "kind": task.kind,
        "tags": list(task.tags or []),
        "status": task.status,
        "context_file": "context.json",
        "reference_file": "reference.json",
    }
    if task.episode_id:
        ts["episode_id"] = task.episode_id
    if task.cut_span_id:
        ts["cut_span_id"] = task.cut_span_id
    if task.reason:
        ts["reason"] = task.reason
    ts["check"] = [c.to_toml() for c in task.checks]

    doc = {
        "schema_version": "1.4",
        "task": {
            "name": f"touchstone/{task.name}",
            "version": "1.0.0",
            "description": task.description or f"Replay task {task.name}.",
            "keywords": list(task.tags or []),
        },
        "metadata": {"touchstone": ts},
        "verifier": {"timeout_sec": 900.0},
        "agent": {"timeout_sec": 900.0},
        "environment": {"build_timeout_sec": 600.0},
    }
    return tomli_w.dumps(doc)


def _content(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _instruction_md(task: Task) -> str:
    messages = (task.context or {}).get("messages", [])
    tools = (task.context or {}).get("tools") or []
    lines = [f"# {task.name}", ""]
    systems = [m for m in messages if m.get("role") == "system"]
    if systems:
        lines += ["## System", *(_content(m.get("content")) for m in systems), ""]
    convo = [m for m in messages if m.get("role") != "system"]
    if convo:
        turns = [f"**{m.get('role', 'user')}:** {_content(m.get('content'))}" for m in convo]
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


def write_task(root: str | Path, task: Task, *, preserve: bool = True) -> Path:
    """Write (or re-materialise) `task` under `root/tasks/<name>/`.

    With `preserve` (the default), interview/manual check blocks already on disk are kept and
    mined/policy blocks are replaced — the sync semantics. Set `preserve=False` to write
    `task.checks` verbatim (used when removing a check). The task is gated (oracle=1, nop=0).
    """
    task_dir = tasks_dir(root) / task.name
    kept = preserved_checks(root, task.name) if preserve else []
    task.checks = _dedupe(list(task.checks) + kept)
    task.status, task.reason = gate(task)

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
        reason=ts.get("reason", ""),
        description=doc.get("task", {}).get("description", ""),
        context=_load_json(task_dir, ts.get("context_file", "context.json"), {}),
        reference=_load_json(task_dir, ts.get("reference_file", "reference.json"), None),
        checks=[Check.from_toml(b) for b in ts.get("check", [])],
    )


def list_tasks(
    root: str | Path, *, tag: str | None = None, active_only: bool = False
) -> list[Task]:
    base = tasks_dir(root)
    if not base.exists():
        return []
    tasks = [read_task(d) for d in sorted(base.iterdir())
             if d.is_dir() and (d / "task.toml").exists()]
    if tag is not None:
        tasks = [t for t in tasks if tag in (t.tags or [])]
    if active_only:
        tasks = [t for t in tasks if t.status == "active"]
    return tasks


def get_task(root: str | Path, name: str) -> Task | None:
    task_dir = tasks_dir(root) / name
    return read_task(task_dir) if (task_dir / "task.toml").exists() else None
