"""Export a captured episode to Harbor ATIF JSON (Agent Trajectory Interchange Format).

Matched against Harbor's pydantic models at
`~/Pebble/Github/harbor/src/harbor/models/trajectories/` (root `trajectory.py`), schema
`ATIF-v1.8`. Those models set `extra="forbid"`, so we emit only known fields.

Mapping: the first llm span's `input.messages` become the leading system/user/agent steps; each
llm span becomes an agent step (assistant text + tool_calls + token metrics); tool spans between
one llm span and the next become that step's `observation.results`, with `source_call_id` matched
to a tool_call id by name. Multi-turn user messages injected mid-episode are not reconstructed
(single simplification; noted in tests).
"""

from __future__ import annotations

import json

from .. import store

SCHEMA_VERSION = "ATIF-v1.8"


def _as_message(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _args_dict(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"_raw": arguments}


def _tool_calls(span: store.Span) -> list[dict]:
    calls = []
    for i, tc in enumerate(span.output.get("message", {}).get("tool_calls") or []):
        calls.append(
            {
                "tool_call_id": tc.get("id") or f"{span.id}-{i}",
                "function_name": tc.get("name") or "tool",
                "arguments": _args_dict(tc.get("arguments", "")),
            }
        )
    return calls


def _agent_step(step_id: int, span: store.Span, model: str | None) -> dict:
    msg = span.output.get("message", {})
    step: dict = {
        "step_id": step_id,
        "source": "agent",
        "message": _as_message(msg.get("content", "")),
    }
    if span.model or model:
        step["model_name"] = span.model or model
    calls = _tool_calls(span)
    if calls:
        step["tool_calls"] = calls
    if span.tokens_in is not None and span.tokens_out is not None:
        step["metrics"] = {"prompt_tokens": span.tokens_in, "completion_tokens": span.tokens_out}
    if span.error:
        step["extra"] = {"error": span.error}
    return step


def _observation(tool_spans: list[store.Span], step: dict) -> dict | None:
    if not tool_spans:
        return None
    by_name = {c["function_name"]: c["tool_call_id"] for c in step.get("tool_calls", [])}
    results = []
    for ts in tool_spans:
        name = ts.input.get("name")
        content = ts.output.get("result") if not ts.error else ts.error
        results.append({
            "source_call_id": by_name.get(name),
            "content": _as_message(content if content is not None else ""),
        })
    return {"results": results}


def to_atif(conn, episode_id: str) -> dict:
    ep = store.get_episode(conn, episode_id)
    if ep is None:
        raise ValueError(f"no episode {episode_id!r}")
    spans = store.list_spans(conn, episode_id)
    llm_spans = [s for s in spans if s.kind == "llm"]
    model = llm_spans[0].model if llm_spans else None

    steps: list[dict] = []

    # Leading conversation from the first llm call's message history.
    if llm_spans:
        for msg in llm_spans[0].input.get("messages", []):
            role = msg.get("role")
            content = _as_message(msg.get("content", ""))
            if role == "system":
                steps.append({"step_id": 0, "source": "system", "message": content})
            elif role == "user":
                steps.append({"step_id": 0, "source": "user", "message": content})
            elif role == "assistant":
                steps.append({"step_id": 0, "source": "agent", "message": content})
            elif role in ("tool", "function") and steps and steps[-1]["source"] == "agent":
                steps[-1].setdefault("observation", {"results": []})["results"].append(
                    {"source_call_id": None, "content": content}
                )

    # Each llm span -> an agent step; tool spans in between -> that step's observation.
    for i, span in enumerate(llm_spans):
        step = _agent_step(0, span, model)
        lo = _span_index(spans, span)
        hi = _span_index(spans, llm_spans[i + 1]) if i + 1 < len(llm_spans) else len(spans)
        between = [s for s in spans[lo + 1 : hi] if s.kind == "tool"]
        obs = _observation(between, step)
        if obs:
            step["observation"] = obs
        steps.append(step)

    # Renumber sequentially from 1 (ATIF requires step_ids 1..N).
    for n, step in enumerate(steps, start=1):
        step["step_id"] = n

    total_in = sum(s.tokens_in or 0 for s in llm_spans)
    total_out = sum(s.tokens_out or 0 for s in llm_spans)

    traj: dict = {
        "schema_version": SCHEMA_VERSION,
        "session_id": ep.id,
        "trajectory_id": ep.id,
        "agent": {
            "name": ep.meta.get("agent", "touchstone-app") if ep.meta else "touchstone-app",
            "version": "1",
            "model_name": model,
        },
        "steps": steps or [{"step_id": 1, "source": "user", "message": ep.name}],
        "final_metrics": {
            "total_prompt_tokens": total_in,
            "total_completion_tokens": total_out,
            "total_steps": len(steps),
        },
    }
    notes = ep.name
    if ep.outcome_label:
        notes += f" ({ep.outcome_label}={ep.outcome_score})"
    traj["notes"] = notes
    return traj


def _span_index(spans: list[store.Span], span: store.Span) -> int:
    for i, s in enumerate(spans):
        if s.id == span.id:
            return i
    return -1


def validate(traj: dict) -> None:
    """Structural ATIF validation (used by tests when Harbor's models are not importable)."""
    assert traj["schema_version"] == SCHEMA_VERSION
    agent = traj["agent"]
    assert agent["name"] and agent["version"]
    steps = traj["steps"]
    assert steps, "at least one step required"
    for i, step in enumerate(steps, start=1):
        assert step["step_id"] == i, f"step_id must be {i}, got {step['step_id']}"
        assert step["source"] in ("system", "user", "agent")
        assert isinstance(step["message"], str)
        if step["source"] != "agent":
            assert "tool_calls" not in step and "metrics" not in step
        ids = {c["tool_call_id"] for c in step.get("tool_calls", [])}
        for r in step.get("observation", {}).get("results", []):
            scid = r.get("source_call_id")
            assert scid is None or scid in ids, "observation source_call_id must match a tool_call"


def export_atif(conn, episode_id: str, path) -> dict:
    from pathlib import Path

    traj = to_atif(conn, episode_id)
    Path(path).write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    return traj
