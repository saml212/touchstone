"""Export a captured episode to Harbor ATIF JSON (Agent Trajectory Interchange Format).

Matched against Harbor's pydantic models at
`~/Pebble/Github/harbor/src/harbor/models/trajectories/` (root `trajectory.py`), schema
`ATIF-v1.8`. Those models set `extra="forbid"`, so we emit only known fields.

Mapping: the first llm span's `input.messages` become the leading system/user/agent steps; each
llm span becomes an agent step (assistant text + tool_calls + token metrics); tool spans between
one llm span and the next become that step's `observation.results`, with `source_call_id` matched
to a tool_call id (by id first, then by tool name). Multi-turn user messages injected mid-episode
are not reconstructed (single simplification; noted in tests).
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


def _step_extra(span: store.Span) -> dict:
    extra = {}
    if span.output.get("stop_reason"):
        extra["stop_reason"] = span.output["stop_reason"]
    if span.error:
        extra["error"] = span.error
    return extra


def _agent_step(step_id: int, span: store.Span, model: str | None) -> dict:
    msg = span.output.get("message", {})
    step: dict = {"step_id": step_id, "source": "agent", "message": _atif_message(msg)}
    optional = {
        "model_name": span.model or model,
        "tool_calls": _tool_calls(span),
        "reasoning_content": _reasoning_text(msg),
        "metrics": _metrics(span),
        "extra": _step_extra(span),
    }
    for key, value in optional.items():
        if value:
            step[key] = value
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


def _leading_step(msg: dict) -> dict | None:
    """A system/user/agent history message -> a leading step, or None for a non-role message."""
    source = _ROLE_SOURCE.get(msg.get("role"))
    if not source:
        return None
    step = {"step_id": 0, "source": source, "message": _atif_message(msg)}
    if source == "agent" and msg.get("tool_calls"):
        step["tool_calls"] = _history_calls(msg)
    return step


def _attach_history_result(steps: list[dict], msg: dict) -> None:
    """Fold a history tool/function message into the preceding agent step's observation."""
    if msg.get("role") not in ("tool", "function") or not steps or steps[-1]["source"] != "agent":
        return
    ids = {c["tool_call_id"] for c in steps[-1].get("tool_calls", [])}
    cid = msg.get("tool_call_id") if msg.get("tool_call_id") in ids else None
    steps[-1].setdefault("observation", {"results": []})["results"].append(
        {"source_call_id": cid, "content": text_of(msg)}
    )


def _leading_steps(first_span: store.Span) -> list[dict]:
    """The system/user/agent steps reconstructed from the first model call's message history."""
    steps: list[dict] = []
    for msg in first_span.input.get("messages", []):
        step = _leading_step(msg)
        if step:
            steps.append(step)
        else:
            _attach_history_result(steps, msg)
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


def _final_metrics(llm_spans: list[store.Span], total_steps: int) -> dict:
    cached = sum((s.output.get("usage") or {}).get("cached_tokens") or 0 for s in llm_spans)
    cost = sum(s.cost_usd or 0 for s in llm_spans)
    return {
        "total_prompt_tokens": sum(s.tokens_in or 0 for s in llm_spans),
        "total_completion_tokens": sum(s.tokens_out or 0 for s in llm_spans),
        "total_steps": total_steps,
        "total_cached_tokens": cached or None,
        "total_cost_usd": cost or None,
    }


def _episode_notes(ep: store.Episode) -> str:
    if ep.outcome_label:
        return f"{ep.name} ({ep.outcome_label}={ep.outcome_score})"
    return ep.name


def _agent_name(ep: store.Episode) -> str:
    return ep.meta.get("agent", "touchstone-app") if ep.meta else "touchstone-app"


def to_atif(conn, episode_id: str) -> dict:
    ep = store.get_episode(conn, episode_id)
    if ep is None:
        raise ValueError(f"no episode {episode_id!r}")
    spans = store.list_spans(conn, episode_id)
    llm_spans = [s for s in spans if s.kind == "model"]
    model = llm_spans[0].model if llm_spans else None

    steps: list[dict] = _leading_steps(llm_spans[0]) if llm_spans else []
    steps += _body_steps(spans, llm_spans, model)
    for n, step in enumerate(steps, start=1):  # ATIF requires step_ids 1..N
        step["step_id"] = n

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": ep.id,
        "trajectory_id": ep.id,
        "agent": {"name": _agent_name(ep), "version": "1", "model_name": model},
        "steps": steps or [{"step_id": 1, "source": "user", "message": ep.name}],
        "final_metrics": _final_metrics(llm_spans, len(steps)),
        "notes": _episode_notes(ep),
    }


def _span_index(spans: list[store.Span], span: store.Span) -> int:
    for i, s in enumerate(spans):
        if s.id == span.id:
            return i
    return -1


def _validate_message(message) -> None:
    assert isinstance(message, str | list)
    if isinstance(message, list):
        for part in message:
            assert part["type"] in ("text", "image", "audio")


def _validate_observation(step: dict) -> None:
    ids = {c["tool_call_id"] for c in step.get("tool_calls", [])}
    for r in step.get("observation", {}).get("results", []):
        scid = r.get("source_call_id")
        assert scid is None or scid in ids, "observation source_call_id must match a tool_call"


def _validate_step(step: dict, i: int) -> None:
    assert step["step_id"] == i, f"step_id must be {i}, got {step['step_id']}"
    assert step["source"] in ("system", "user", "agent")
    _validate_message(step["message"])
    if step["source"] != "agent":
        assert not any(k in step for k in ("tool_calls", "metrics", "reasoning_content"))
    _validate_observation(step)


def validate(traj: dict) -> None:
    """Structural ATIF validation (used by tests when Harbor's models are not importable)."""
    assert traj["schema_version"] == SCHEMA_VERSION
    agent = traj["agent"]
    assert agent["name"] and agent["version"]
    steps = traj["steps"]
    assert steps, "at least one step required"
    for i, step in enumerate(steps, start=1):
        _validate_step(step, i)


# `import_trajectory` (ATIF -> episode, the inverse of `to_atif`) lives in `atif_import`;
# re-exported so callers keep importing it from `harbor.atif`.
from .atif_import import import_trajectory  # noqa: E402, F401
