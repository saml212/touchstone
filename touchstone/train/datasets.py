"""Turn a benchmark (its task directories + stored run results) into training datasets.

Three files plus a manifest, all under `<out_dir>`:

- `sft.jsonl`     one row per task whose reference passes every attached hard check: the canonical
                  context messages + tools and the reference reply as the completion.
- `preference.jsonl`  DPO/ORPO pairs {prompt, chosen, rejected}: a passing candidate reply chosen
                  over a failing one (from stored run results), plus the reference chosen over each
                  failing candidate reply on tasks whose reference is good.
- `rl_tasks.jsonl`  one row per task: name, context, tools, and the serialized attached checks
                  a reward verifier evaluates.
- `manifest.json`  target name, timestamps, and row counts.

Every file is written atomically (temp file + `os.replace`), so a reader never sees a half-written
file and a re-run overwrites cleanly.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import store
from .. import tasks as tasks_mod
from ..bench import benchmark
from ..checks import Check, Target, evaluate, passes
from ..messages import text_of


@dataclass
class DatasetBundle:
    benchmark_id: str  # the target name
    benchmark_name: str
    out_dir: Path
    counts: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return s or "benchmark"


def default_out_dir(db_path: str | Path, target: str) -> Path:
    """`<db-dir>/train/<target-slug>` — sits beside the SQLite db under `.touchstone/`."""
    return Path(db_path).expanduser().parent / "train" / _slug(target)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _reference_msg(task: tasks_mod.Task) -> dict:
    ref = task.reference or {}
    return {"role": "assistant", "content": ref.get("content") or "",
            "tool_calls": ref.get("tool_calls") or []}


def _reference_passes(task: tasks_mod.Task) -> bool:
    """Whether the recorded reference satisfies every attached hard check (no judge provider)."""
    checks = [c for c in task.checks if c.kind != "judge"]
    if not checks:
        return True
    ref = task.reference or {}
    target = Target(output_text=text_of(ref),
                    tool_calls=ref.get("tool_calls") or [], reference=ref)
    return passes(evaluate(checks, target), checks)


def _serialized_checks(checks: list[Check]) -> list[dict]:
    return [{"name": c.name, "kind": c.kind, "params": c.params,
             "applies_to": c.applies_to, "severity": c.severity} for c in checks]


def _output_key(msg: dict) -> str:
    return json.dumps({"content": msg.get("content") or "",
                       "tool_calls": msg.get("tool_calls") or []}, sort_keys=True)


def _record_result(r: store.Result, out: dict[str, dict[str, list[dict]]]) -> None:
    if r.error or not r.output:
        return
    reply = {"content": r.output.get("content") or "",
             "tool_calls": r.output.get("tool_calls") or []}
    side = out.setdefault(r.task, {"pass": [], "fail": []})["pass" if r.passed else "fail"]
    if all(_output_key(reply) != _output_key(x) for x in side):
        side.append(reply)


def _candidate_outcomes(conn, target: str) -> dict[str, dict[str, list[dict]]]:
    """Per task, the distinct candidate replies that passed and that failed, across all runs.

    The `reference` model spec is excluded — its reply is the reference, handled separately."""
    out: dict[str, dict[str, list[dict]]] = {}
    for run in (r for r in store.list_runs(conn, target) if r.model_spec != "reference"):
        for r in store.list_results(conn, run.id):
            _record_result(r, out)
    return out


def _pair_row(task, prompt, tools, chosen, rejected, source) -> dict:
    return {"task": task.name, "prompt": prompt, "tools": tools,
            "chosen": chosen, "rejected": rejected, "source": source}


def _emit_pair(rows: list[dict], seen: set, row: dict) -> None:
    ck, rk = _output_key(row["chosen"]), _output_key(row["rejected"])
    key = (row["task"], ck, rk)
    if ck == rk or key in seen:
        return
    seen.add(key)
    rows.append(row)


def _task_pairs(task, bucket: dict, rows: list[dict], seen: set) -> None:
    prompt = (task.context or {}).get("messages", [])
    tools = (task.context or {}).get("tools") or []
    for chosen in bucket["pass"]:
        for rejected in bucket["fail"]:
            _emit_pair(rows, seen, _pair_row(task, prompt, tools, chosen, rejected, "candidates"))
    # The reference is a trusted "chosen" only when it passes the task's hard checks.
    if bucket["fail"] and _reference_passes(task):
        ref = _reference_msg(task)
        for rejected in bucket["fail"]:
            _emit_pair(rows, seen, _pair_row(task, prompt, tools, ref, rejected, "reference"))


def _preference_rows(conn, target: str, tasks: list[tasks_mod.Task]) -> list[dict]:
    outcomes = _candidate_outcomes(conn, target)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for task in tasks:
        _task_pairs(task, outcomes.get(task.name, {"pass": [], "fail": []}), rows, seen)
    return rows


def prepare(conn, root, target: str, out_dir: str | Path,
            *, only: set[str] | None = None) -> DatasetBundle:
    """Write sft/preference/rl datasets + manifest for `target` into `out_dir`.

    `only` restricts the rows to a subset of task names (Distill passes the frontier), so a
    Distill round packages exactly the tasks the student is still failing."""
    tasks = [tasks_mod.read_task(d) for d in benchmark.resolve(root, target)]
    if only is not None:
        tasks = [t for t in tasks if t.name in only]
    if not tasks:
        raise ValueError(f"target {target!r} resolves to no active tasks")
    out = Path(out_dir).expanduser()

    sft_rows, rl_rows = [], []
    for task in tasks:
        ctx_messages = (task.context or {}).get("messages", [])
        ctx_tools = (task.context or {}).get("tools") or []
        if _reference_passes(task):
            sft_rows.append({"task": task.name, "messages": ctx_messages, "tools": ctx_tools,
                             "completion": _reference_msg(task)})
        rl_rows.append({"task": task.name, "messages": ctx_messages, "tools": ctx_tools,
                        "reference": task.reference, "checks": _serialized_checks(task.checks)})

    pref_rows = _preference_rows(conn, target, tasks)

    paths = {
        "sft": out / "sft.jsonl",
        "preference": out / "preference.jsonl",
        "rl_tasks": out / "rl_tasks.jsonl",
        "manifest": out / "manifest.json",
    }
    counts = {"sft": len(sft_rows), "preference": len(pref_rows),
              "rl_tasks": len(rl_rows), "tasks": len(tasks)}

    atomic_write(paths["sft"], _jsonl(sft_rows))
    atomic_write(paths["preference"], _jsonl(pref_rows))
    atomic_write(paths["rl_tasks"], _jsonl(rl_rows))
    atomic_write(paths["manifest"], json.dumps({
        "target": target,
        "created_at": store.now(),
        "counts": counts,
        "files": {k: v.name for k, v in paths.items() if k != "manifest"},
    }, ensure_ascii=False, indent=2) + "\n")

    return DatasetBundle(benchmark_id=target, benchmark_name=target, out_dir=out,
                         counts=counts, paths=paths)
