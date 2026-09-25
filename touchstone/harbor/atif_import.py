"""Reconstruct a captured episode from an ATIF trajectory — the inverse of `atif.to_atif`.

Entry point: `import_trajectory(conn, traj)`. System/user steps rebuild the message history;
each agent step becomes one model span (with its usage), and its observation results become tool
spans linked by `source_call_id`. Re-exported from `harbor.atif` so callers import it from there.
"""

from __future__ import annotations

import json

from .. import store


def _message_text(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "".join(p.get("text", "") for p in message if p.get("type") == "text")
    return ""


def _import_tool_calls(step: dict) -> list[dict]:
    return [{"id": c.get("tool_call_id"), "name": c.get("function_name") or "tool",
             "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}
            for c in step.get("tool_calls") or []]


def _import_usage(metrics: dict) -> dict | None:
    fields = {"tokens_in": metrics.get("prompt_tokens"),
              "tokens_out": metrics.get("completion_tokens"),
              "cached_tokens": metrics.get("cached_tokens")}
    kept = {k: v for k, v in fields.items() if v is not None}
    return kept or None


def _import_agent_step(conn, episode_id: str, history: list[dict], step: dict) -> None:
    tool_calls = _import_tool_calls(step)
    metrics = step.get("metrics") or {}
    assistant = {"role": "assistant", "content": _message_text(step.get("message")),
                 "tool_calls": tool_calls}
    output: dict = {"message": assistant}
    usage = _import_usage(metrics)
    if usage:
        output["usage"] = usage
    store.insert_span(conn, store.Span(
        episode_id=episode_id, kind="model", name=step.get("model_name") or "model",
        model=step.get("model_name"), input={"messages": list(history), "tools": [], "params": {}},
        output=output, tokens_in=metrics.get("prompt_tokens"),
        tokens_out=metrics.get("completion_tokens"), cost_usd=metrics.get("cost_usd")))
    history.append(assistant)
    names = {c["id"]: c["name"] for c in tool_calls}
    for result in (step.get("observation") or {}).get("results", []):
        cid = result.get("source_call_id")
        content = result.get("content")
        store.insert_span(conn, store.Span(
            episode_id=episode_id, kind="tool", name=names.get(cid, "tool"),
            input={"name": names.get(cid)}, output={"result": content}, tool_call_id=cid))
        history.append({"role": "tool", "tool_call_id": cid, "content": content})


def import_trajectory(conn, traj: dict) -> store.Episode:
    """Reconstruct an episode (and its spans) from an ATIF trajectory — the inverse of `to_atif`."""
    agent = traj.get("agent") or {}
    meta = {"agent": agent["name"]} if agent.get("name") else {}
    ep = store.insert_episode(conn, store.Episode(
        name=traj.get("notes") or traj.get("session_id") or "imported", source="atif", meta=meta))
    history: list[dict] = []
    for step in traj.get("steps", []):
        source = step.get("source")
        if source == "system":
            history.append({"role": "system", "content": _message_text(step.get("message"))})
        elif source == "user":
            history.append({"role": "user", "content": _message_text(step.get("message"))})
        elif source == "agent":
            _import_agent_step(conn, ep.id, history, step)
    return ep
