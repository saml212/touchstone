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
from ..messages import text_of

SCHEMA_VERSION = "ATIF-v1.8"

_IMG_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
              "gif": "image/gif", "webp": "image/webp"}


def _image_source(part: dict) -> dict | None:
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        src = part.get("source")
        url = src.get("url") if isinstance(src, dict) else src if isinstance(src, str) else None
    if not isinstance(url, str):
        return None
    media = _IMG_MEDIA.get(url.rsplit(".", 1)[-1].split("?")[0].lower())
    return {"media_type": media, "path": url} if media else None


def _content_parts(content: list) -> list[dict] | None:
    """Canonical parts -> ATIF ContentParts, or None if any part can't be represented faithfully."""
    parts = []
    for p in content:
        if p.get("type") == "text":
            parts.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            source = _image_source(p)
            if source is None:
                return None
            parts.append({"type": "image", "source": source})
        else:
            return None  # audio/file without a Harbor-valid source: fall back to text
    return parts


def _atif_message(msg: dict):
    """A step message: a string for text, else ContentParts when every part is representable."""
    content = (msg or {}).get("content")
    if isinstance(content, list):
        parts = _content_parts(content)
        if parts is not None:
            return parts
    return text_of(msg)


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


def _reasoning_text(msg: dict) -> str | None:
    texts = [b.get("thinking", "") for b in msg.get("reasoning") or []
             if b.get("type") == "thinking" and b.get("thinking")]
    return "\n".join(texts) or None


def _metrics(span: store.Span) -> dict | None:
    cached = (span.output.get("usage") or {}).get("cached_tokens")
    fields = {"prompt_tokens": span.tokens_in, "completion_tokens": span.tokens_out,
              "cached_tokens": cached, "cost_usd": span.cost_usd}
    kept = {k: v for k, v in fields.items() if v is not None}
    return kept or None


def _agent_step(step_id: int, span: store.Span, model: str | None) -> dict:
    msg = span.output.get("message", {})
    step: dict = {"step_id": step_id, "source": "agent", "message": _atif_message(msg)}
    if span.model or model:
        step["model_name"] = span.model or model
    calls = _tool_calls(span)
    if calls:
        step["tool_calls"] = calls
    reasoning = _reasoning_text(msg)
    if reasoning:
        step["reasoning_content"] = reasoning
    metrics = _metrics(span)
    if metrics:
        step["metrics"] = metrics
    extra = {}
    if span.output.get("stop_reason"):
        extra["stop_reason"] = span.output["stop_reason"]
    if span.error:
        extra["error"] = span.error
    if extra:
        step["extra"] = extra
    return step


def _observation(tool_spans: list[store.Span], step: dict) -> dict | None:
    if not tool_spans:
        return None
    ids = {c["tool_call_id"] for c in step.get("tool_calls", [])}
    by_name = {c["function_name"]: c["tool_call_id"] for c in step.get("tool_calls", [])}
    results = []
    for ts in tool_spans:
        content = ts.output.get("result") if not ts.error else ts.error
        call_id = ts.tool_call_id if ts.tool_call_id in ids else by_name.get(ts.input.get("name"))
        results.append({
            "source_call_id": call_id,
            "content": content if isinstance(content, str) else json.dumps(content or ""),
        })
    return {"results": results}


_ROLE_SOURCE = {"system": "system", "user": "user", "assistant": "agent"}


def _history_calls(msg: dict) -> list[dict]:
    return [{"tool_call_id": tc.get("id"), "function_name": tc.get("name") or "tool",
             "arguments": _args_dict(tc.get("arguments", ""))}
            for tc in msg.get("tool_calls") or []]


def _leading_steps(first_span: store.Span) -> list[dict]:
    """The system/user/agent steps reconstructed from the first model call's message history."""
    steps: list[dict] = []
    for msg in first_span.input.get("messages", []):
        role = msg.get("role")
        source = _ROLE_SOURCE.get(role)
        if source:
            step = {"step_id": 0, "source": source, "message": _atif_message(msg)}
            if source == "agent" and msg.get("tool_calls"):
                step["tool_calls"] = _history_calls(msg)
            steps.append(step)
        elif role in ("tool", "function") and steps and steps[-1]["source"] == "agent":
            ids = {c["tool_call_id"] for c in steps[-1].get("tool_calls", [])}
            cid = msg.get("tool_call_id") if msg.get("tool_call_id") in ids else None
            steps[-1].setdefault("observation", {"results": []})["results"].append(
                {"source_call_id": cid, "content": text_of(msg)}
            )
    return steps


def _body_steps(spans: list[store.Span], llm_spans: list[store.Span], model: str | None) -> list:
    """Each model span becomes an agent step; tool spans in between become its observation."""
    steps: list[dict] = []
    for i, span in enumerate(llm_spans):
        step = _agent_step(0, span, model)
        lo = _span_index(spans, span)
        hi = _span_index(spans, llm_spans[i + 1]) if i + 1 < len(llm_spans) else len(spans)
        between = [s for s in spans[lo + 1 : hi] if s.kind == "tool"]
        obs = _observation(between, step)
        if obs:
            step["observation"] = obs
        steps.append(step)
    return steps


def to_atif(conn, episode_id: str) -> dict:
    ep = store.get_episode(conn, episode_id)
    if ep is None:
        raise ValueError(f"no episode {episode_id!r}")
    spans = store.list_spans(conn, episode_id)
    llm_spans = [s for s in spans if s.kind == "model"]
    model = llm_spans[0].model if llm_spans else None

    steps: list[dict] = []
    if llm_spans:
        steps += _leading_steps(llm_spans[0])
    steps += _body_steps(spans, llm_spans, model)

    # Renumber sequentially from 1 (ATIF requires step_ids 1..N).
    for n, step in enumerate(steps, start=1):
        step["step_id"] = n

    total_in = sum(s.tokens_in or 0 for s in llm_spans)
    total_out = sum(s.tokens_out or 0 for s in llm_spans)
    total_cached = sum((s.output.get("usage") or {}).get("cached_tokens") or 0 for s in llm_spans)
    total_cost = sum(s.cost_usd or 0 for s in llm_spans)

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
            "total_cached_tokens": total_cached or None,
            "total_cost_usd": total_cost or None,
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
        assert isinstance(step["message"], str | list)
        if isinstance(step["message"], list):
            for part in step["message"]:
                assert part["type"] in ("text", "image", "audio")
        if step["source"] != "agent":
            assert not any(k in step for k in ("tool_calls", "metrics", "reasoning_content"))
        ids = {c["tool_call_id"] for c in step.get("tool_calls", [])}
        for r in step.get("observation", {}).get("results", []):
            scid = r.get("source_call_id")
            assert scid is None or scid in ids, "observation source_call_id must match a tool_call"


def export_atif(conn, episode_id: str, path) -> dict:
    from pathlib import Path

    traj = to_atif(conn, episode_id)
    Path(path).write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    return traj
