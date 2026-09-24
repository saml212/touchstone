"""Replay a benchmark's tasks against a candidate model and store per-task results.

Each task's recorded context (messages + tools) is replayed to the provider, the reply becomes a
checks `Target`, the task's enabled checks are evaluated, and a `Result` is stored. Results insert
as they complete, so a killed run leaves partial data; `finish_run` marks a run done. Provider
failures (after the provider's own retries) become a result with `error` set and `passed=0` — the
run continues. Determinism: with the `scripted` provider a rerun yields identical results (latency
aside, which is wall-clock).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import asdict

from .. import store
from ..checks import Check as DslCheck
from ..checks import Target, evaluate, passes
from ..llm import provider_from_spec
from . import pricing


def _task_checks(conn, task: store.Task) -> list[DslCheck]:
    checks = []
    for cid in task.check_ids or []:
        row = store.get_check(conn, cid)
        if row is None or not row.enabled:
            continue
        checks.append(DslCheck.from_dict({
            "kind": row.kind, "params": row.params, "id": row.id, "name": row.name,
            "applies_to": row.applies_to, "severity": row.severity,
        }))
    return checks


async def _call(provider, messages, tools, timeout):
    """Await achat, or run a sync-only provider's chat in a thread. Enforce a per-task timeout."""
    achat = getattr(provider, "achat", None)
    if achat is not None and inspect.iscoroutinefunction(achat):
        coro = achat(messages, tools=tools, timeout=timeout)
    else:
        coro = asyncio.to_thread(provider.chat, messages, tools, False, timeout)
    return await asyncio.wait_for(coro, timeout)


def _target(task: store.Task, reply) -> Target:
    messages = list((task.context or {}).get("messages", []))
    messages.append({"role": "assistant", "content": reply.content,
                     "tool_calls": reply.tool_calls})
    return Target(
        output_text=reply.content or "",
        tool_calls=reply.tool_calls or [],
        messages=messages,
        reference=task.reference,
    )


async def _run_task(conn, run_id, model_spec, provider, task, timeout, judge_provider) -> None:
    checks = _task_checks(conn, task)
    started = time.monotonic()
    error = None
    reply = None
    try:
        reply = await _call(provider, (task.context or {}).get("messages", []),
                            (task.context or {}).get("tools") or None, timeout)
    except TimeoutError:
        error = f"timed out after {timeout}s"
    except Exception as exc:  # provider error after its own retries; the run must continue
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = int((time.monotonic() - started) * 1000)

    if error is not None:
        result = store.Result(run_id=run_id, task_id=task.id, passed=0, error=error,
                              latency_ms=latency_ms)
    else:
        target = _target(task, reply)
        results = evaluate(checks, target, judge_provider=judge_provider)
        result = store.Result(
            run_id=run_id,
            task_id=task.id,
            passed=1 if passes(results, checks) else 0,
            check_results=[asdict(r) for r in results],
            output={"content": reply.content, "tool_calls": reply.tool_calls},
            latency_ms=latency_ms,
            cost_usd=pricing.cost_usd(model_spec, reply.usage),
        )
    store.insert_result(conn, result)


async def _run_async(conn, run, tasks, provider, concurrency, timeout, judge_provider) -> None:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(task):
        async with sem:
            await _run_task(conn, run.id, run.model_spec, provider, task, timeout, judge_provider)

    await asyncio.gather(*(guarded(t) for t in tasks))


def start(
    conn,
    benchmark_id: str,
    model_spec: str,
    *,
    concurrency: int = 4,
    timeout: float = 60,
) -> store.Run:
    """Create the Run row (validating the benchmark) and return it without executing anything."""
    bench = store.get_benchmark(conn, benchmark_id)
    if bench is None:
        raise ValueError(f"no benchmark {benchmark_id!r}")
    tasks = [t for t in (store.get_task(conn, tid) for tid in bench.task_ids) if t is not None]
    return store.insert_run(conn, store.Run(
        benchmark_id=benchmark_id, model_spec=model_spec,
        meta={"concurrency": concurrency, "timeout": timeout, "task_count": len(tasks)},
    ))


def execute(conn, run: store.Run, *, judge_provider=None, provider=None) -> store.Run:
    """Replay a started run's tasks, storing results as they finish, then mark the run done."""
    provider = provider or provider_from_spec(run.model_spec)
    bench = store.get_benchmark(conn, run.benchmark_id)
    tasks = [t for t in (store.get_task(conn, tid) for tid in bench.task_ids) if t is not None]
    concurrency = run.meta.get("concurrency", 4)
    timeout = run.meta.get("timeout", 60)
    asyncio.run(_run_async(conn, run, tasks, provider, concurrency, timeout, judge_provider))
    store.finish_run(conn, run.id)
    return store.get_run(conn, run.id)


def run(
    conn,
    benchmark_id: str,
    model_spec: str,
    *,
    concurrency: int = 4,
    timeout: float = 60,
    judge_provider=None,
    provider=None,
) -> store.Run:
    """Replay every task in `benchmark_id` against `model_spec` and store a Run + its Results."""
    run_row = start(conn, benchmark_id, model_spec, concurrency=concurrency, timeout=timeout)
    return execute(conn, run_row, judge_provider=judge_provider, provider=provider)


def result_view(result: store.Result) -> dict:
    """A result's deterministic fields (latency excluded) for rerun-equality assertions."""
    return {
        "task_id": result.task_id,
        "passed": result.passed,
        "check_results": result.check_results,
        "output": result.output,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }
