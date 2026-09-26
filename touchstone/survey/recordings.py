"""Reconstruct recorded tool calls from the trace DB.

Recordings hold only model spans. A tool call lives in a model span's `output.message.tool_calls`;
its result is a `role=tool` message in a later span's `input.messages`, matched by `tool_call_id`.
`tool_events` pairs them into (tool, arguments, output) triples — the raw material for sort,
simulate, and fidelity. Arguments and outputs are JSON-decoded when possible, else kept raw.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
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


def decode_output(value):
    """Public alias: JSON-decode a string result when possible, else keep it. Recorded outputs are
    stored decoded, so a replayed value is decoded the same way before being compared to one."""
    return _decode(value)


def _calls_in(output: dict) -> list[dict]:
    message = output.get("message") if isinstance(output, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return calls if isinstance(calls, list) else []


def _tool_result(msg) -> tuple[str | None, str | None, object] | None:
    """A tool/function result message -> (tool_call_id, name, decoded output), or None. The legacy
    role=function form carries a name but no id, so both are reported for pairing."""
    if isinstance(msg, dict) and msg.get("role") == "tool":
        return msg.get("tool_call_id"), msg.get("name"), _decode(msg.get("content"))
    return None


def _results_in(inp: dict) -> list[tuple[str | None, str | None, object]]:
    messages = inp.get("messages") if isinstance(inp, dict) else None
    return [r for msg in (messages or []) if (r := _tool_result(msg))]


def _collect_calls(span, calls: dict, order: list) -> None:
    for call in _calls_in(span.output or {}):
        cid = call.get("id")
        if cid and cid not in calls:
            calls[cid] = call
            order.append(cid)


def _collect_results(span, by_id: dict, by_name: dict) -> None:
    for cid, name, output in _results_in(span.input or {}):
        if cid is not None:
            by_id.setdefault(cid, output)
        elif name:
            by_name[name].append(output)


def _collect(spans) -> tuple[dict, list, dict, dict]:
    """Across the episode's spans: the tool calls (by id, in order), results keyed by id, and
    id-less results queued by tool name (for the legacy function form that threads no id)."""
    calls: dict[str, dict] = {}
    order: list[str] = []
    by_id: dict[str, object] = {}
    by_name: dict[str, deque] = defaultdict(deque)
    for span in spans:
        _collect_calls(span, calls, order)
        _collect_results(span, by_id, by_name)
    return calls, order, by_id, by_name


def _output_for(cid: str, name: str, by_id: dict, by_name: dict):
    """A call's recorded output: by id when linked, else the next id-less result of that name."""
    if cid in by_id:
        return by_id[cid]
    if by_name.get(name):
        return by_name[name].popleft()
    return None


def _episode_events(spans) -> list[ToolEvent]:
    calls, order, by_id, by_name = _collect(spans)
    events = []
    for cid in order:
        call = calls[cid]
        name = call.get("name")
        events.append(ToolEvent(tool=name, arguments=_decode(call.get("arguments")),
                                output=_output_for(cid, name, by_id, by_name), episode=""))
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
