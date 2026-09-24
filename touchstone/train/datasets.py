"""Turn a benchmark (its tasks, checks, and run results) into training datasets.

Three files plus a manifest, all under `<out_dir>`:

- `sft.jsonl`     one row per task whose reference passes every attached hard check: the canonical
                  context messages + tools and the reference reply as the completion.
- `preference.jsonl`  DPO/ORPO pairs {prompt, chosen, rejected}: a passing candidate reply chosen
                  over a failing one (from stored run results), plus the reference chosen over each
                  failing candidate reply on tasks whose reference is good.
- `rl_tasks.jsonl`  one row per task: id, context, tools, and the serialized attached checks that a
                  reward verifier evaluates.
- `manifest.json`  benchmark id/name, timestamps, and row counts.

Every file is written atomically (temp file + `os.replace`), so a reader never sees a half-written
file and a re-run overwrites cleanly.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import store
from ..checks import Check as DslCheck
from ..checks import Target, evaluate, passes


@dataclass
class DatasetBundle:
    benchmark_id: str
    benchmark_name: str
    out_dir: Path
    counts: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return s or "benchmark"


def default_out_dir(db_path: str | Path, benchmark: store.Benchmark) -> Path:
    """`<db-dir>/train/<benchmark-slug>` — sits beside the SQLite db under `.touchstone/`."""
    return Path(db_path).expanduser().parent / "train" / _slug(benchmark.name)


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


def _enabled_checks(conn, task: store.Task) -> list[DslCheck]:
    checks = []
    for cid in task.check_ids or []:
        row = store.get_check(conn, cid)
        if row is not None and row.enabled:
            checks.append(DslCheck.from_dict(asdict(row)))
    return checks


def _reference_msg(task: store.Task) -> dict:
    ref = task.reference or {}
    return {"role": "assistant", "content": ref.get("content") or "",
            "tool_calls": ref.get("tool_calls") or []}


def _reference_passes(task: store.Task, checks: list[DslCheck]) -> bool:
    """Whether the recorded reference satisfies every attached hard check (no judge provider)."""
    if not checks:
        return True
    ref = task.reference or {}
    target = Target(output_text=ref.get("content") or "",
                    tool_calls=ref.get("tool_calls") or [], reference=ref)
    return passes(evaluate(checks, target), checks)


def _serialized_checks(checks: list[DslCheck]) -> list[dict]:
    return [{"id": c.id, "name": c.name, "kind": c.kind, "params": c.params,
             "applies_to": c.applies_to, "severity": c.severity} for c in checks]


def _output_key(msg: dict) -> str:
    return json.dumps({"content": msg.get("content") or "",
                       "tool_calls": msg.get("tool_calls") or []}, sort_keys=True)


def _benchmark_runs(conn, bench: store.Benchmark) -> list[store.Run]:
    """Runs for this benchmark. Runs record the benchmark id/name as it was passed to `bench run`,
    so resolve each run's stored key back to a benchmark and match on the canonical id."""
    runs = []
    for run in store.list_runs(conn):
        resolved = store.get_benchmark(conn, run.benchmark_id)
        if resolved is not None and resolved.id == bench.id:
            runs.append(run)
    return runs


def _candidate_outcomes(conn, bench: store.Benchmark) -> dict[str, dict[str, list[dict]]]:
    """Per task, the distinct candidate replies that passed and that failed, across all runs.

    The `reference` model spec is excluded — its reply is the reference, handled separately."""
    out: dict[str, dict[str, list[dict]]] = {}
    for run in _benchmark_runs(conn, bench):
        if run.model_spec == "reference":
            continue
        for r in store.list_results(conn, run.id):
            if r.error or not r.output:
                continue
            reply = {"content": r.output.get("content") or "",
                     "tool_calls": r.output.get("tool_calls") or []}
            bucket = out.setdefault(r.task_id, {"pass": [], "fail": []})
            side = bucket["pass" if r.passed else "fail"]
            if all(_output_key(reply) != _output_key(x) for x in side):
                side.append(reply)
    return out


def _preference_rows(conn, bench: store.Benchmark, tasks: list[store.Task]) -> list[dict]:
    outcomes = _candidate_outcomes(conn, bench)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def emit(task, prompt, tools, chosen, rejected, source):
        if _output_key(chosen) == _output_key(rejected):
            return
        key = (task.id, _output_key(chosen), _output_key(rejected))
        if key in seen:
            return
        seen.add(key)
        rows.append({"task_id": task.id, "prompt": prompt, "tools": tools,
                     "chosen": chosen, "rejected": rejected, "source": source})

    for task in tasks:
        prompt = (task.context or {}).get("messages", [])
        tools = (task.context or {}).get("tools") or []
        bucket = outcomes.get(task.id, {"pass": [], "fail": []})
        for chosen in bucket["pass"]:
            for rejected in bucket["fail"]:
                emit(task, prompt, tools, chosen, rejected, "candidates")
        # The reference is a trusted "chosen" only when it passes the task's hard checks.
        if bucket["fail"] and _reference_passes(task, _enabled_checks(conn, task)):
            ref = _reference_msg(task)
            for rejected in bucket["fail"]:
                emit(task, prompt, tools, ref, rejected, "reference")
    return rows


def prepare(conn, benchmark_id: str, out_dir: str | Path) -> DatasetBundle:
    """Write sft/preference/rl datasets + manifest for `benchmark_id` into `out_dir`."""
    bench = store.get_benchmark(conn, benchmark_id)
    if bench is None:
        raise ValueError(f"no benchmark {benchmark_id!r}")
    out = Path(out_dir).expanduser()
    tasks = [t for t in (store.get_task(conn, tid) for tid in bench.task_ids) if t is not None]

    sft_rows, rl_rows = [], []
    for task in tasks:
        checks = _enabled_checks(conn, task)
        ctx_messages = (task.context or {}).get("messages", [])
        ctx_tools = (task.context or {}).get("tools") or []
        if _reference_passes(task, checks):
            sft_rows.append({"task_id": task.id, "messages": ctx_messages, "tools": ctx_tools,
                             "completion": _reference_msg(task)})
        rl_rows.append({"task_id": task.id, "name": task.name, "messages": ctx_messages,
                        "tools": ctx_tools, "reference": task.reference,
                        "checks": _serialized_checks(checks)})

    pref_rows = _preference_rows(conn, bench, tasks)

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
        "benchmark_id": bench.id,
        "benchmark_name": bench.name,
        "created_at": store.now(),
        "counts": counts,
        "files": {k: v.name for k, v in paths.items() if k != "manifest"},
    }, ensure_ascii=False, indent=2) + "\n")

    return DatasetBundle(
        benchmark_id=bench.id, benchmark_name=bench.name, out_dir=out,
        counts=counts, paths=paths,
    )
