"""Reconstruct recorded tool calls from the trace DB.

Recordings hold only model spans. A tool call lives in a model span's `output.message.tool_calls`;
its result is a `role=tool` message in a later span's `input.messages`, matched by `tool_call_id`.
`tool_events` pairs them into (tool, arguments, output) triples — the raw material for sort,
simulate, and fidelity. Arguments and outputs are JSON-decoded when possible, else kept raw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .. import store


@dataclass
class ToolEvent:
    tool: str
    arguments: object  # decoded JSON (usually a dict), or the raw value when not JSON
    output: object  # decoded JSON, or the raw string
    episode: str


def _decode(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _calls_in(output: dict) -> list[dict]:
    message = output.get("message") if isinstance(output, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return calls if isinstance(calls, list) else []


def _tool_result(msg) -> tuple[str, object] | None:
    if isinstance(msg, dict) and msg.get("role") == "tool":
        cid = msg.get("tool_call_id")
        if cid is not None:
            return cid, _decode(msg.get("content"))
    return None


def _results_in(inp: dict) -> dict[str, object]:
    messages = inp.get("messages") if isinstance(inp, dict) else None
    results: dict[str, object] = {}
    for msg in messages or []:
        pair = _tool_result(msg)
        if pair and pair[0] not in results:
            results[pair[0]] = pair[1]
    return results


def _episode_events(spans) -> list[ToolEvent]:
    calls: dict[str, dict] = {}
    order: list[str] = []
    results: dict[str, object] = {}
    for span in spans:
        for call in _calls_in(span.output or {}):
            cid = call.get("id")
            if cid and cid not in calls:
                calls[cid] = call
                order.append(cid)
        results.update(_results_in(span.input or {}))
    events = []
    for cid in order:
        call = calls[cid]
        events.append(ToolEvent(tool=call.get("name"), arguments=_decode(call.get("arguments")),
                                output=results.get(cid), episode=""))
    return events


def tool_events(conn) -> list[ToolEvent]:
    """Every recorded tool call across all episodes, in recording order, with its returned value."""
    out: list[ToolEvent] = []
    for episode in store.list_episodes(conn):
        for event in _episode_events(store.list_spans(conn, episode.id)):
            event.episode = episode.id
            out.append(event)
    return out


def recorded_tool_names(events: list[ToolEvent]) -> set[str]:
    return {e.tool for e in events if e.tool}
