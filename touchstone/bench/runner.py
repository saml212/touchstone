"""Replay a target's tasks against a candidate model and store per-task results.

A target (a benchmark name, a `tasks/` path, or a glob) resolves to task directories; each task's
recorded context is replayed to the provider, the reply becomes a checks `Target`, the task's checks
are evaluated, and a `Result` (with the real `reward`) is stored — plus a portable copy under
`.touchstone/runs/<id>/`. Results insert as they complete, so a killed run leaves partial data.
Provider failures (after the provider's own retries) become a result with `error` set and reward 0;
the run continues. Determinism: with the `scripted` provider a rerun yields identical results.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

from .. import store
from .. import tasks as tasks_mod
from ..checks import Target, evaluate, passes
from ..llm import provider_from_spec
from . import benchmark, pricing


def _accepts_task(fn) -> bool:
    """Whether a provider method takes the optional `task` kwarg (or **kwargs to absorb it)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "task" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


async def _call(provider, messages, tools, timeout, task):
    """Await achat, or run a sync-only provider's chat in a thread. Enforce a per-task timeout.

    `task` reaches only providers that declare it (e.g. `reference`, which replays the task's
    recorded reply); every other provider is called with its unchanged signature.
    """
    achat = getattr(provider, "achat", None)
    if achat is not None and inspect.iscoroutinefunction(achat):
        kwargs = {"tools": tools, "timeout": timeout}
        if _accepts_task(achat):
            kwargs["task"] = task
        coro = achat(messages, **kwargs)
    elif _accepts_task(provider.chat):
        coro = asyncio.to_thread(provider.chat, messages, tools, False, timeout, task)
    else:
        coro = asyncio.to_thread(provider.chat, messages, tools, False, timeout)
    return await asyncio.wait_for(coro, timeout)


def _target(task: tasks_mod.Task, reply) -> Target:
    messages = list((task.context or {}).get("messages", []))
    messages.append({"role": "assistant", "content": reply.content,
                     "tool_calls": reply.tool_calls})
    return Target(
        output_text=reply.content or "",
        tool_calls=reply.tool_calls or [],
        messages=messages,
        reference=task.reference,
    )


async def _run_task(run_id, model_spec, provider, task, timeout, judge_provider, results) -> None:
    checks = task.checks
    started = time.monotonic()
    error = None
    reply = None
    try:
        reply = await _call(provider, (task.context or {}).get("messages", []),
                            (task.context or {}).get("tools") or None, timeout, task)
    except TimeoutError:
        error = f"timed out after {timeout}s"
    except Exception as exc:  # provider error after its own retries; the run must continue
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = int((time.monotonic() - started) * 1000)

    if error is not None:
        results.append(store.Result(run_id=run_id, task=task.name, passed=0, reward=0.0,
                                    error=error, latency_ms=latency_ms))
        return
    outcomes = evaluate(checks, _target(task, reply), judge_provider=judge_provider)
    ok = passes(outcomes, checks)
    kinds = {c.id: c.kind for c in checks}
    results.append(store.Result(
        run_id=run_id,
        task=task.name,
        passed=1 if ok else 0,
        reward=1.0 if ok else 0.0,
        check_results={r.check_id: {"passed": r.passed, "evidence": r.evidence,
                                    "kind": kinds.get(r.check_id, "")} for r in outcomes},
        output={"content": reply.content, "tool_calls": reply.tool_calls},
        latency_ms=latency_ms,
        cost_usd=pricing.cost_usd(model_spec, reply.usage),
    ))


async def _run_async(model_spec, provider, tasks, concurrency, timeout, judge_provider, results):
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(task):
        async with sem:
            await _run_task(results["run_id"], model_spec, provider, task, timeout,
                            judge_provider, results["rows"])

    await asyncio.gather(*(guarded(t) for t in tasks))


def _run_dir(root, run_id: str) -> Path:
    return Path(root) / ".touchstone" / "runs" / run_id


def _write_run_files(root, run: store.Run, results: list[store.Result]) -> None:
    """The portable copy the DB indexes: run.json + results.jsonl under .touchstone/runs/<id>/."""
    run_dir = _run_dir(root, run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "id": run.id, "target": run.target, "model_spec": run.model_spec,
        "started_at": run.started_at, "finished_at": run.finished_at, "meta": run.meta,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [json.dumps({
        "task": r.task, "reward": r.reward, "passed": bool(r.passed),
        "check_results": r.check_results, "latency_ms": r.latency_ms,
        "cost_usd": r.cost_usd, "error": r.error,
    }, ensure_ascii=False) for r in results]
    (run_dir / "results.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def start(
    conn,
    root: str,
    target: str,
    model_spec: str,
    *,
    concurrency: int = 4,
    timeout: float = 60,
) -> store.Run:
    """Create the Run row (validating the target resolves to tasks) without executing anything."""
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    task_dirs = benchmark.resolve(root, target)
    if not task_dirs:
        raise ValueError(f"target {target!r} resolves to no active tasks")
    return store.insert_run(conn, store.Run(
        target=target, model_spec=model_spec,
        meta={"concurrency": concurrency, "timeout": timeout, "task_count": len(task_dirs)},
    ))


def execute(conn, root: str, run: store.Run, *, judge_provider=None, provider=None) -> store.Run:
    """Replay a started run's tasks, storing results as they finish, then mark the run done."""
    provider = provider or provider_from_spec(run.model_spec)
    tasks = [tasks_mod.read_task(d) for d in benchmark.resolve(root, run.target)]
    concurrency = run.meta.get("concurrency", 4)
    timeout = run.meta.get("timeout", 60)
    state = {"run_id": run.id, "rows": []}
    asyncio.run(_run_async(run.model_spec, provider, tasks, concurrency, timeout,
                           judge_provider, state))
    for result in state["rows"]:
        store.insert_result(conn, result)
    store.finish_run(conn, run.id)
    finished = store.get_run(conn, run.id)
    _write_run_files(root, finished, store.list_results(conn, run.id))
    return finished


def run(
    conn,
    root: str,
    target: str,
    model_spec: str,
    *,
    concurrency: int = 4,
    timeout: float = 60,
    judge_provider=None,
    provider=None,
) -> store.Run:
    """Replay every task in `target` against `model_spec` and store a Run + its Results."""
    run_row = start(conn, root, target, model_spec, concurrency=concurrency, timeout=timeout)
    return execute(conn, root, run_row, judge_provider=judge_provider, provider=provider)


def result_view(result: store.Result) -> dict:
    """A result's deterministic fields (latency excluded) for rerun-equality assertions."""
    return {
        "task": result.task,
        "passed": result.passed,
        "reward": result.reward,
        "check_results": result.check_results,
        "output": result.output,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }
