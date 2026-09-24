"""Export a benchmark's tasks as Harbor task directories.

Layout and `task.toml` fields match Harbor's `harbor task init` template
(`harbor/src/harbor/cli/template-task/`): `task.toml` (schema_version 1.4, with `[metadata]`,
`[verifier]`, `[agent]`, `[environment]` sections), `instruction.md`, `.gitignore`,
`environment/Dockerfile`, `tests/test.sh`, `tests/test_outputs.py`, `solution/solve.sh`. The
verifier reward lands in `/logs/verifier/reward.txt` (1/0), exactly as the template's test.sh does.

The container carries a vendored copy of `touchstone/checks/{dsl,run}.py` under
`tests/touchstone_checks/`, so it depends on neither the network nor the `touchstone` package.
`judge` checks need an LLM and are dropped from the export (skipped in-container by design).
Re-exporting a task overwrites its directory cleanly.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .. import store

_CHECKS_SRC = Path(__file__).resolve().parent.parent / "checks"

TASK_TOML = """\
schema_version = "1.4"

[metadata]
name = "{name}"
touchstone_task_id = "{task_id}"
touchstone_episode_id = "{episode}"
tags = {tags}

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 600.0
"""

DOCKERFILE = """\
FROM python:3.12-slim

WORKDIR /app

# jsonschema/simpleeval back the json_schema and expr check kinds; harmless otherwise.
RUN pip install --no-cache-dir jsonschema simpleeval
"""

TEST_SH = """\
#!/bin/bash
# Score /app/output.json against the task's checks and write /logs/verifier/reward.txt.
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/test_outputs.py
"""

TEST_OUTPUTS_PY = '''\
"""Standalone verifier: evaluate /app/output.json against the vendored checks, write the reward.

Runnable with plain `python tests/test_outputs.py`. Paths are overridable via env vars so the same
file works in a Harbor container (defaults) and in a local subprocess test.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from touchstone_checks.dsl import Check, Target  # noqa: E402
from touchstone_checks.run import evaluate, passes  # noqa: E402


def main() -> int:
    output_path = Path(os.environ.get("TOUCHSTONE_OUTPUT", "/app/output.json"))
    checks_path = Path(os.environ.get("TOUCHSTONE_CHECKS", str(HERE / "checks.json")))
    reward_dir = Path(os.environ.get("TOUCHSTONE_REWARD_DIR", "/logs/verifier"))
    reward_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(checks_path.read_text())
    checks = [Check.from_dict(c) for c in spec["checks"]]
    try:
        out = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        out = {}
    if not isinstance(out, dict):
        out = {}

    target = Target(
        output_text=out.get("content") or "",
        tool_calls=out.get("tool_calls") or [],
        reference=spec.get("reference"),
    )
    results = evaluate(checks, target, judge_provider=None)
    ok = passes(results, checks)

    (reward_dir / "reward.txt").write_text(f"{1.0 if ok else 0.0}\\n")
    per_check = {r.check_id: r.passed for r in results}
    (reward_dir / "rewards.json").write_text(json.dumps({"reward": 1.0 if ok else 0.0,
                                                         "checks": per_check}))
    for r in results:
        print(f"{r.check_id}: {r.passed} — {r.evidence}")
    print(f"REWARD {1.0 if ok else 0.0}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
'''

GITIGNORE = "__pycache__/\n*.pyc\n.DS_Store\n"

_MARKER = ".touchstone-task"


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s or "task"


def _instruction_md(task: store.Task) -> str:
    ctx = task.context or {}
    messages = ctx.get("messages", [])
    tools = ctx.get("tools") or []

    lines = [f"# {task.name}", ""]
    systems = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    if systems:
        lines.append("## System")
        for m in systems:
            lines.append(_content(m.get("content")))
        lines.append("")
    if convo:
        lines.append("## Conversation so far")
        for m in convo:
            lines.append(f"**{m.get('role', 'user')}:** {_content(m.get('content'))}")
        lines.append("")
    if tools:
        lines.append("## Tools available (JSON schemas)")
        lines.append("```json")
        lines.append(json.dumps(tools, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
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


def _content(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _checks_spec(conn, task: store.Task) -> dict:
    checks = []
    for cid in task.check_ids or []:
        row = store.get_check(conn, cid)
        if row is None or not row.enabled or row.kind == "judge":
            continue
        checks.append({"id": row.id, "name": row.name, "kind": row.kind, "params": row.params,
                       "applies_to": row.applies_to, "severity": row.severity})
    return {"reference": task.reference, "checks": checks}


def _solve_sh(task: store.Task) -> str:
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


def _export_task(conn, task: store.Task, task_dir: Path) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    _write(task_dir / _MARKER, task.id + "\n")
    _write(task_dir / "task.toml", TASK_TOML.format(
        name=task.name, episode=task.episode_id or "-", task_id=task.id,
        tags=json.dumps(task.tags or []),
    ))
    _write(task_dir / "instruction.md", _instruction_md(task))
    _write(task_dir / ".gitignore", GITIGNORE)
    _write(task_dir / "environment" / "Dockerfile", DOCKERFILE)
    _write(task_dir / "tests" / "test.sh", TEST_SH, executable=True)
    _write(task_dir / "tests" / "test_outputs.py", TEST_OUTPUTS_PY)
    _write(task_dir / "tests" / "checks.json",
           json.dumps(_checks_spec(conn, task), ensure_ascii=False, indent=2))
    _write(task_dir / "solution" / "solve.sh", _solve_sh(task), executable=True)

    vendor = task_dir / "tests" / "touchstone_checks"
    vendor.mkdir(parents=True)
    _write(vendor / "__init__.py", "")
    for name in ("dsl.py", "run.py"):
        shutil.copyfile(_CHECKS_SRC / name, vendor / name)


def export(conn, benchmark_id: str, out_dir: str | Path) -> list[Path]:
    """Write one Harbor task directory per task in the benchmark. Returns the dirs written."""
    bench = store.get_benchmark(conn, benchmark_id)
    if bench is None:
        raise ValueError(f"no benchmark {benchmark_id!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    used: set[str] = set()
    for tid in bench.task_ids:
        task = store.get_task(conn, tid)
        if task is None:
            continue
        slug = f"{_slug(task.name)}-{task.id[-6:].lower()}"
        while slug in used:  # ids are unique, but guard the suffix collision anyway
            slug += "x"
        used.add(slug)
        task_dir = out / slug
        _export_task(conn, task, task_dir)
        written.append(task_dir)

    keep = {d.resolve() for d in written}
    for child in out.iterdir():
        if child.is_dir() and (child / _MARKER).exists() and child.resolve() not in keep:
            shutil.rmtree(child)
    return written
