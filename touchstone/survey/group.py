"""Group recorded episodes by job-to-be-done, one survey-provider call over a compact listing.

The provider sees, per episode, only its id, scrubbed first user turn, tool names used, and outcome
label — never the episode contents. It answers with {"groups": [{"label", "slug", "episodes"}]},
validated against `GROUP_SCHEMA` and retried once. Episodes with no tool call and no outcome go to a
`no_job` bucket (reported, never made into tasks). The result is cached in `groups.json`.

`variant_episodes` then picks the real cases a group becomes tasks from: resolved episodes, ranked
by tool-call count, one per distinct seed signature, capped at three (one otherwise).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jsonschema import ValidationError, validate

from .. import store
from ..llm.prompt import extract_json
from .provider import SurveyProvider
from .recordings import ToolEvent
from .scrub import Scrubber
from .writes import atomic_write_json

_MAX_VARIANTS = 3

GROUP_SCHEMA = {
    "type": "object",
    "required": ["groups"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "slug", "episodes"],
                "properties": {
                    "label": {"type": "string"},
                    "slug": {"type": "string"},
                    "episodes": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

GROUP_PROMPT = """You are grouping recorded conversations of an AI agent by the job the user wanted
done (e.g. "check an order's status", "issue a refund"). Below is one line per conversation: its id,
the user's first message, the tools the agent used, and the outcome.

Reply with ONE JSON object and nothing else (no prose, no markdown fences):
{{"groups": [{{"label": "<plain-english job name>", "slug": "<kebab-case-slug>",
              "episodes": ["<id>", ...]}}]}}

Put every listed conversation in exactly one group. Group by the job-to-be-done, not by wording.
Keep labels short and human. Use only the ids shown. Return only the JSON object.

## Conversations
{listing}"""


def _first_user(spans: list[store.Span]) -> str:
    for span in spans:
        for msg in (span.input or {}).get("messages", []):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"])
    return ""


def _events_by_episode(events: list[ToolEvent]) -> dict[str, list[ToolEvent]]:
    by_ep: dict[str, list[ToolEvent]] = defaultdict(list)
    for event in events:
        by_ep[event.episode].append(event)
    return by_ep


def _episode_row(ep: store.Episode, spans: list[store.Span], evs: list[ToolEvent],
                 scrub: Scrubber) -> dict:
    tools = sorted({e.tool for e in evs if e.tool})
    return {"id": ep.id, "first_user": scrub.text(_first_user(spans)),
            "tools": tools, "outcome": ep.outcome_label or ""}


def _is_no_job(row: dict) -> bool:
    return not row["tools"] and not row["outcome"]


def _listing(conn, events: list[ToolEvent], scrub: Scrubber) -> tuple[list[dict], list[str]]:
    by_ep = _events_by_episode(events)
    rows, no_job = [], []
    for ep in store.list_episodes(conn):
        row = _episode_row(ep, store.list_spans(conn, ep.id), by_ep.get(ep.id, []), scrub)
        if _is_no_job(row):
            no_job.append(row["id"])
        else:
            rows.append(row)
    return rows, no_job


def _render_listing(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        tools = ", ".join(r["tools"]) or "none"
        outcome = r["outcome"] or "unknown"
        lines.append(f"- {r['id']}: user={r['first_user']!r} tools=[{tools}] outcome={outcome}")
    return "\n".join(lines)


def _parse_groups(text: str) -> dict:
    raw = extract_json(text)
    if raw is None:
        raise ValueError("the group answer contained no JSON object")
    data = json.loads(raw)
    validate(data, GROUP_SCHEMA)
    return data


def _obtain_groups(provider: SurveyProvider, repo: Path, prompt: str) -> dict:
    text = provider.run(prompt, repo)
    try:
        return _parse_groups(text)
    except (ValueError, ValidationError) as first:
        retry = (f"{prompt}\n\nYour previous answer was invalid: {first}\n"
                 "Return only the corrected JSON object.")
        return _parse_groups(provider.run(retry, repo))


def _dedup_slugs(groups: list[dict]) -> None:
    seen: dict[str, int] = {}
    for group in groups:
        base = group["slug"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            group["slug"] = f"{base}-{seen[base]}"


def _clean_groups(groups: list[dict], known: set[str]) -> list[dict]:
    out = []
    for group in sorted(groups, key=lambda g: g.get("slug", "")):
        eps = [e for e in group.get("episodes", []) if e in known]
        if eps:
            out.append({"label": group["label"], "slug": group["slug"], "episodes": sorted(eps)})
    _dedup_slugs(out)
    return out


def group_episodes(conn, events: list[ToolEvent], provider: SurveyProvider, repo: Path,
                   out_dir: Path, scrub: Scrubber, force: bool = False) -> dict:
    """Cluster episodes by job-to-be-done. Cached in groups.json; reused unless force."""
    path = out_dir / "groups.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    rows, no_job = _listing(conn, events, scrub)
    known = {r["id"] for r in rows}
    if rows:
        answer = _obtain_groups(provider, repo, GROUP_PROMPT.format(listing=_render_listing(rows)))
        groups = _clean_groups(answer.get("groups", []), known)
    else:
        groups = []
    data = {"groups": groups, "no_job": sorted(no_job)}
    atomic_write_json(path, data)
    return data


def _seed_signature(evs: list[ToolEvent]) -> tuple:
    values: set[str] = set()
    for event in evs:
        args = event.arguments if isinstance(event.arguments, dict) else {}
        values.update(str(v) for v in args.values() if isinstance(v, str | int | float))
    return tuple(sorted(values))


def variant_episodes(group: dict, events_by_ep: dict[str, list[ToolEvent]],
                     resolved: set[str]) -> list[str]:
    """The real cases this group becomes tasks from: resolved, ranked by tool-call count, one per
    distinct seed signature, capped at three (one when fewer than three distinct signatures)."""
    eligible = [e for e in group["episodes"] if e in resolved]
    ranked = sorted(eligible, key=lambda e: (-len(events_by_ep.get(e, [])), e))
    picked, seen_sigs = [], set()
    for ep in ranked:
        sig = _seed_signature(events_by_ep.get(ep, []))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            picked.append(ep)
    return picked[:_MAX_VARIANTS] if len(picked) >= _MAX_VARIANTS else picked[:1]
