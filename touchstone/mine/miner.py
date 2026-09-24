"""Propose checks from captured episodes: statistics (no LLM) and one LLM pass.

`mine_stats` derives checks from outcome-correlated tool use, output JSON-ness, length, PII, and
recurring phrases. `mine_llm` asks a provider for a JSON array of DSL checks and keeps only those
that validate. `dedupe` drops proposals identical to an existing check, and `mine` orchestrates the
whole thing: stats always, LLM unless suppressed, insert as disabled `mined` checks, cut tasks.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field

from .. import store
from ..checks import Check as DslCheck
from ..checks import find_pii
from ..checks.dsl import PARAM_SPEC
from . import cut as cut_mod
from .codebase import Snippet

log = logging.getLogger("touchstone.mine")

_LLM_EPISODE_SAMPLE = 20
_LLM_MAX_PROPOSALS = 25
_LLM_SNIPPET_CAP = 20
_WORD = re.compile(r"[A-Za-z0-9']+")

_TOOL_GOOD_FRAC = 0.8
_TOOL_BAD_FRAC = 0.3
_JSON_FRAC = 0.9
_PHRASE_GOOD_FRAC = 0.6
_PHRASE_BAD_FRAC = 0.1
_LENGTH_SLACK = 1.25


@dataclass
class Proposal:
    kind: str
    params: dict = field(default_factory=dict)
    applies_to: str = "final"
    severity: str = "hard"
    name: str = ""
    rationale: str = ""
    origin: str = "stats"  # stats | llm
    support: dict = field(default_factory=dict)

    def dsl(self) -> DslCheck:
        return DslCheck(
            kind=self.kind,
            params=self.params,
            name=self.name,
            applies_to=self.applies_to,
            severity=self.severity,
        )

    def store_check(self) -> store.Check:
        return store.Check(
            name=self.name or self.kind,
            kind=self.kind,
            params=self.params,
            applies_to=self.applies_to,
            severity=self.severity,
            source="mined",
            rationale=self.rationale,
            enabled=0,
        )

    @property
    def support_count(self) -> int:
        ids = self.support.get("episode_ids")
        return len(ids) if isinstance(ids, list) else 0


# ---- episode view ----------------------------------------------------------


@dataclass
class _View:
    episode: store.Episode
    tool_names: list[str]
    final_output: str
    good: bool | None  # None = unlabeled


def _is_good(ep: store.Episode) -> bool | None:
    if ep.outcome_score is None:
        return None
    return ep.outcome_score >= 0.5


def _final_output(spans: list[store.Span]) -> str:
    for span in reversed(spans):
        if span.kind == "llm":
            msg = (span.output or {}).get("message") or {}
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _view(conn, ep: store.Episode) -> _View:
    spans = store.list_spans(conn, ep.id)
    tool_names = [s.name for s in spans if s.kind == "tool"]
    return _View(ep, tool_names, _final_output(spans), _is_good(ep))


# ---- statistics ------------------------------------------------------------


def mine_stats(conn, episodes: list[store.Episode]) -> list[Proposal]:
    views = [_view(conn, ep) for ep in episodes]
    good = [v for v in views if v.good is True]
    bad = [v for v in views if v.good is False]

    proposals: list[Proposal] = []
    proposals += _tool_proposals(good, bad)
    proposals += _json_proposal(good)
    proposals += _length_proposal(good)
    proposals += _pii_proposal(views)
    proposals += _phrase_proposals(good, bad)
    return proposals


def _tool_proposals(good: list[_View], bad: list[_View]) -> list[Proposal]:
    names = sorted({n for v in good + bad for n in v.tool_names})
    out: list[Proposal] = []
    for name in names:
        g_ids = [v.episode.id for v in good if name in v.tool_names]
        b_ids = [v.episode.id for v in bad if name in v.tool_names]
        g_frac = len(g_ids) / len(good) if good else 0.0
        b_frac = len(b_ids) / len(bad) if bad else 0.0
        if good and g_frac >= _TOOL_GOOD_FRAC and b_frac < _TOOL_BAD_FRAC:
            out.append(
                Proposal(
                    kind="tool_called",
                    params={"name": name},
                    applies_to="tool_calls",
                    name=f"calls {name}",
                    rationale=(
                        f"{name} used in {g_frac:.0%} of good vs {b_frac:.0%} of bad episodes"
                    ),
                    support={
                        "episode_ids": g_ids,
                        "counts": {"good": len(g_ids), "bad": len(b_ids)},
                    },
                )
            )
        elif bad and b_frac >= _TOOL_GOOD_FRAC and g_frac < _TOOL_BAD_FRAC:
            out.append(
                Proposal(
                    kind="tool_not_called",
                    params={"name": name},
                    applies_to="tool_calls",
                    name=f"avoids {name}",
                    rationale=(
                        f"{name} used in {b_frac:.0%} of bad vs {g_frac:.0%} of good episodes"
                    ),
                    support={
                        "episode_ids": b_ids,
                        "counts": {"good": len(g_ids), "bad": len(b_ids)},
                    },
                )
            )
    return out


def _json_objects(views: list[_View]) -> list[dict]:
    objs = []
    for v in views:
        if not v.final_output:
            continue
        try:
            parsed = json.loads(v.final_output)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            objs.append(parsed)
    return objs


def _json_proposal(good: list[_View]) -> list[Proposal]:
    with_output = [v for v in good if v.final_output]
    if not with_output:
        return []
    objs = _json_objects(with_output)
    if len(objs) / len(with_output) < _JSON_FRAC:
        return []
    keys: set[str] | None = None
    for obj in objs:
        keys = set(obj) if keys is None else (keys & set(obj))
    keys = keys or set()
    schema = {"type": "object", "properties": {k: {} for k in sorted(keys)}}
    if keys:
        schema["required"] = sorted(keys)
    return [
        Proposal(
            kind="json_schema",
            params={"schema": schema},
            name="output is JSON",
            rationale=f"{len(objs)}/{len(with_output)} good outputs parse as JSON objects",
            support={
                "episode_ids": [v.episode.id for v in with_output],
                "counts": {"json": len(objs)},
            },
        )
    ]


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = math.ceil(q * len(sorted_values)) - 1
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return sorted_values[idx]


def _length_proposal(good: list[_View]) -> list[Proposal]:
    lengths = sorted(len(v.final_output) for v in good if v.final_output)
    if not lengths:
        return []
    p99 = _percentile(lengths, 0.99)
    limit = int(math.ceil(p99 * _LENGTH_SLACK)) or 1
    slack_pct = int((_LENGTH_SLACK - 1) * 100)
    return [
        Proposal(
            kind="max_length",
            params={"max": limit},
            severity="soft",
            name="output length bound",
            rationale=f"p99 good output length {p99} chars, +{slack_pct}% slack",
            support={
                "episode_ids": [v.episode.id for v in good if v.final_output],
                "counts": {"p99": p99},
            },
        )
    ]


def _pii_proposal(views: list[_View]) -> list[Proposal]:
    hit_ids: list[str] = []
    kinds: set[str] = set()
    for v in views:
        found = {k: vals for k, vals in find_pii(v.final_output).items() if vals}
        if found:
            hit_ids.append(v.episode.id)
            kinds |= set(found)
    if not hit_ids:
        return []
    return [
        Proposal(
            kind="no_pii",
            params={"kinds": sorted(kinds)},
            name="no PII in output",
            rationale=f"{', '.join(sorted(kinds))} leaked in {len(hit_ids)} output(s)",
            support={"episode_ids": hit_ids, "counts": {"leaks": len(hit_ids)}},
        )
    ]


def _joined(text: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(text))


def _ngrams(text: str, lo: int = 3, hi: int = 6) -> set[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    grams: set[str] = set()
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            grams.add(" ".join(words[i : i + n]))
    return grams


def _distinct_phrases(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Greedily keep phrases that do not heavily overlap an already-kept phrase (word sets)."""
    kept: list[tuple[str, float]] = []
    kept_words: list[set[str]] = []
    for gram, frac in ranked:
        words = set(gram.split())
        if any(len(words & prev) / min(len(words), len(prev)) >= 0.6 for prev in kept_words):
            continue
        kept.append((gram, frac))
        kept_words.append(words)
    return kept


def _phrase_proposals(good: list[_View], bad: list[_View]) -> list[Proposal]:
    if not good:
        return []
    good_grams = [_ngrams(v.final_output) for v in good]
    bad_grams = [_ngrams(v.final_output) for v in bad]
    counts: dict[str, int] = {}
    for grams in good_grams:
        for g in grams:
            counts[g] = counts.get(g, 0) + 1

    qualifying: list[tuple[str, float]] = []
    for gram, gc in counts.items():
        g_frac = gc / len(good)
        if g_frac < _PHRASE_GOOD_FRAC:
            continue
        b_frac = (sum(gram in gg for gg in bad_grams) / len(bad)) if bad else 0.0
        if b_frac <= _PHRASE_BAD_FRAC:
            qualifying.append((gram, g_frac))

    qualifying.sort(key=lambda gf: (-gf[1], -len(gf[0]), gf[0]))
    chosen = _distinct_phrases(qualifying)

    out: list[Proposal] = []
    for gram, g_frac in chosen[:5]:
        hits = [v.episode.id for v in good if gram in _joined(v.final_output)]
        out.append(
            Proposal(
                kind="contains",
                params={"values": [gram], "mode": "any"},
                severity="soft",
                name=f"says '{gram}'",
                rationale=f"'{gram}' appears in {g_frac:.0%} of good outputs, rare in bad",
                support={"episode_ids": hits, "counts": {"good": len(hits)}},
            )
        )
    return out


# ---- LLM proposals ---------------------------------------------------------


def _render_episode(conn, ep: store.Episode) -> str:
    spans = store.list_spans(conn, ep.id)
    user = ""
    for span in spans:
        if span.kind == "llm":
            for m in (span.input or {}).get("messages", []):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    user = m["content"]
    tools = sorted({s.name for s in spans if s.kind == "tool"})
    if ep.outcome_label:
        outcome = ep.outcome_label
    elif ep.outcome_score is not None:
        outcome = f"scored {ep.outcome_score}"
    else:
        outcome = "unlabeled"
    lines = [
        f"- user: {user[:300]}",
        f"  tools_called: {', '.join(tools) or 'none'}",
        f"  final_output: {_final_output(spans)[:300]}",
        f"  outcome: {outcome}",
    ]
    return "\n".join(lines)


def _llm_prompt(conn, episodes: list[store.Episode], snippets: list[Snippet]) -> str:
    ep_block = "\n".join(_render_episode(conn, ep) for ep in episodes[:_LLM_EPISODE_SAMPLE])
    code_block = "\n".join(
        f"- {s.path}:{s.line} ({s.kind})\n{s.text[:400]}" for s in snippets[:_LLM_SNIPPET_CAP]
    )
    return (
        "Propose checks in the Touchstone DSL that separate good agent behaviour from bad, "
        "based on these episodes and code.\n\n"
        f"EPISODES:\n{ep_block}\n\n"
        f"CODE (system prompts / tool schemas):\n{code_block or 'none'}\n\n"
        "Reply with ONLY a JSON array. Each element: "
        '{"kind": <dsl kind>, "params": {...}, "applies_to": "final|any_turn|tool_calls", '
        '"severity": "hard|soft", "rationale": <str>, "support": {"episode_ids": [...]}}. '
        "Kinds and their exact params:\n"
        + "\n".join(f"- {k}: {v}" for k, v in PARAM_SPEC.items() if k != "judge")
        + "\nDo not invent other kinds or param names."
    )


def mine_llm(
    conn, provider, episodes: list[store.Episode], code_snippets: list[Snippet]
) -> list[Proposal]:
    if provider is None:
        return []
    prompt = _llm_prompt(conn, episodes, code_snippets)
    messages = [
        {"role": "system", "content": "You author precise, testable evaluation checks."},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = provider.chat(messages)
    except Exception as exc:  # provider/transport failure must never crash mining
        log.warning("mine_llm provider error: %s", exc)
        return []
    return _parse_proposals(reply.content or "")


def _parse_proposals(text: str) -> list[Proposal]:
    from ..llm.prompt import extract_json

    obj_str = extract_json(text)
    if obj_str is None:
        log.warning("mine_llm reply had no JSON")
        return []
    try:
        items = json.loads(obj_str)
    except json.JSONDecodeError as exc:
        log.warning("mine_llm reply was not valid JSON: %s", exc)
        return []
    if not isinstance(items, list):
        log.warning("mine_llm reply was not a JSON array")
        return []

    out: list[Proposal] = []
    for item in items:
        if len(out) >= _LLM_MAX_PROPOSALS:
            break
        proposal = _proposal_from_item(item)
        if proposal is not None:
            out.append(proposal)
    return out


def _proposal_from_item(item) -> Proposal | None:
    if not isinstance(item, dict):
        log.warning("mine_llm dropped non-object proposal")
        return None
    try:
        check = DslCheck.from_dict(item)
        check.validate()
    except (ValueError, TypeError) as exc:
        log.warning("mine_llm dropped invalid proposal (%s): %r", exc, item.get("kind"))
        return None
    support = item.get("support")
    return Proposal(
        kind=check.kind,
        params=check.params,
        applies_to=check.applies_to,
        severity=check.severity,
        name=check.name or check.kind,
        rationale=str(item.get("rationale", "")),
        origin="llm",
        support=support if isinstance(support, dict) else {},
    )


# ---- dedupe + orchestration ------------------------------------------------


def _norm(kind: str, params: dict) -> str:
    return kind + ":" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)


def dedupe(existing_checks, proposals: list[Proposal]) -> list[Proposal]:
    seen = {_norm(c.kind, c.params or {}) for c in existing_checks}
    out: list[Proposal] = []
    for p in proposals:
        key = _norm(p.kind, p.params)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def mine(
    conn,
    *,
    provider=None,
    code_snippets: list[Snippet] | None = None,
    no_llm: bool = False,
    limit: int | None = None,
) -> dict:
    episodes = store.list_episodes(conn)
    if limit is not None:
        episodes = episodes[:limit]

    stats = mine_stats(conn, episodes)
    llm = [] if no_llm else mine_llm(conn, provider, episodes, code_snippets or [])
    fresh = dedupe(store.list_checks(conn), stats + llm)
    for proposal in fresh:
        store.insert_check(conn, proposal.store_check())

    tasks = cut_mod.cut_tasks(conn, episodes)
    return {
        "proposals": fresh,
        "stats": len(stats),
        "llm": len(llm),
        "inserted": len(fresh),
        "tasks_cut": len(tasks),
    }
