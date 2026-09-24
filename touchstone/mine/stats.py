"""Statistical check proposals — no LLM, verifiable-first.

`mine_stats` derives checks from captured episodes in trust order: outcome-correlated tool use
(rank 2), state assertions on tool-call arguments held across every good episode (rank 3), output
JSON-ness (rank 4), recurring phrases and a length bound (ranks 5-6), and PII safety (rank 7). Every
proposal carries a `confidence` (programmatic high, statistical medium); `Proposal.policy` auto-
enables the programmatic ranks and safety while the statistical ranks stay disabled pending review.

Tool calls and their results are read from the model spans' canonical messages (assistant
`tool_calls` and the following `role: tool` messages) — the primary and often only source, since
most instrumented apps call `touchstone.trace()` without decorating their tool functions. Tool spans
are an additional source when present, never the only one. `Proposal` is the shared type both this
module and `mine/llm.py` produce.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from .. import store
from ..checks import SAFETY_KINDS, find_pii
from ..checks import Check as DslCheck
from ..policies import Policy

_WORD = re.compile(r"[A-Za-z0-9']+")

_TOOL_GOOD_FRAC = 0.8
_TOOL_BAD_FRAC = 0.3
_JSON_FRAC = 0.9
_PHRASE_GOOD_FRAC = 0.6
_PHRASE_BAD_FRAC = 0.1
_LENGTH_SLACK = 1.25
_STATE_CONFIDENCE = 0.9  # rank 3: a field consistently set across every good episode

# Verifiable-first is an ordering: ranks 2-4 (programmatic + executable) and the safety
# invariants auto-enable; the statistical ranks 5-6 (phrases, length) stay disabled pending
# review. Rank 1 (CI oracle/nop) is the task gate; rank 8 (judge) is only proposed in interviews.
_AUTO_ENABLE_KINDS = SAFETY_KINDS | {
    "tool_called", "tool_not_called", "tool_order", "json_schema", "expr",
}


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
    confidence: float | None = None  # 0-1: how sure the miner is (programmatic high, judge low)

    def check(self) -> DslCheck:
        c = DslCheck(
            kind=self.kind,
            params=self.params,
            name=self.name or self.kind,
            applies_to=self.applies_to,
            severity=self.severity,
            because=self.rationale,
            source="mined",
            confidence=self.confidence,
        )
        c.id = c.name
        return c

    def policy(self) -> Policy:
        """Verifiable-first: ranks 2-4 and safety auto-enable; statistical ranks stay disabled."""
        return Policy(check=self.check(), enabled=self.kind in _AUTO_ENABLE_KINDS)

    @property
    def support_count(self) -> int:
        ids = self.support.get("episode_ids")
        return len(ids) if isinstance(ids, list) else 0


# ---- episode view ----------------------------------------------------------


@dataclass
class _View:
    episode: store.Episode
    tool_names: list[str]
    tool_calls: list[dict]  # {id, name, arguments} across the whole episode
    final_output: str
    good: bool | None  # None = unlabeled


def _is_good(ep: store.Episode) -> bool | None:
    if ep.outcome_score is None:
        return None
    return ep.outcome_score >= 0.5


def final_output(spans: list[store.Span]) -> str:
    """The last model span's textual content — the episode's final assistant reply."""
    for span in reversed(spans):
        if span.kind == "model":
            msg = (span.output or {}).get("message") or {}
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _tool_calls_from_messages(messages: list[dict]) -> dict:
    """Assistant tool_calls in a message list, keyed by call id (or name+args when id is absent)."""
    calls: dict = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            key = tc.get("id") or f"{tc.get('name')}|{tc.get('arguments')}"
            calls[key] = {"id": tc.get("id"), "name": tc.get("name"),
                          "arguments": tc.get("arguments") or ""}
    return calls


def _episode_tool_calls(spans: list[store.Span]) -> list[dict]:
    """Every tool call in an episode, gathered from the model spans' canonical messages (assistant
    tool_calls) and, when present, from tool spans. Most apps only call `touchstone.trace()`, so
    the messages — not tool spans — are the primary and often only source of tool activity."""
    calls: dict = {}
    for span in spans:
        if span.kind == "model":
            messages = list((span.input or {}).get("messages", []))
            out = (span.output or {}).get("message")
            if out:
                messages.append(out)
            calls.update(_tool_calls_from_messages(messages))
        elif span.kind == "tool":  # an extra source when the app decorated its tools
            calls.setdefault(span.id, {
                "id": span.tool_call_id, "name": span.name,
                "arguments": json.dumps(span.input or {}, ensure_ascii=False)})
    return list(calls.values())


def _view(conn, ep: store.Episode) -> _View:
    spans = store.list_spans(conn, ep.id)
    calls = _episode_tool_calls(spans)
    tool_names = sorted({c["name"] for c in calls if c.get("name")})
    return _View(ep, tool_names, calls, final_output(spans), _is_good(ep))


# ---- statistics ------------------------------------------------------------


def mine_stats(conn, episodes: list[store.Episode]) -> list[Proposal]:
    views = [_view(conn, ep) for ep in episodes]
    good = [v for v in views if v.good is True]
    bad = [v for v in views if v.good is False]

    proposals: list[Proposal] = []
    proposals += _tool_proposals(good, bad)
    proposals += _state_proposals(good)
    proposals += _json_proposal(good)
    proposals += _length_proposal(good)
    proposals += _pii_proposal(views)
    proposals += _phrase_proposals(good, bad)
    return proposals


def _proposal(kind: str, name: str, rationale: str, ids: list[str], counts: dict, **kw) -> Proposal:
    """A stats Proposal with the shared support shape (supporting episode ids + counts)."""
    return Proposal(
        kind=kind, name=name, rationale=rationale,
        support={"episode_ids": ids, "counts": counts}, **kw,
    )


def _fraction(views: list[_View], predicate) -> tuple[list[str], float]:
    """The episode ids in `views` where `predicate` holds, and their fraction of `views`."""
    ids = [v.episode.id for v in views if predicate(v)]
    return ids, (len(ids) / len(views) if views else 0.0)


def _tool_proposals(good: list[_View], bad: list[_View]) -> list[Proposal]:
    names = sorted({n for v in good + bad for n in v.tool_names})
    out: list[Proposal] = []
    for name in names:
        g_ids, g_frac = _fraction(good, lambda v, n=name: n in v.tool_names)
        b_ids, b_frac = _fraction(bad, lambda v, n=name: n in v.tool_names)
        counts = {"good": len(g_ids), "bad": len(b_ids)}
        if good and g_frac >= _TOOL_GOOD_FRAC and b_frac < _TOOL_BAD_FRAC:
            out.append(_proposal(
                "tool_called", f"calls {name}",
                f"{name} used in {g_frac:.0%} of good vs {b_frac:.0%} of bad episodes",
                g_ids, counts, params={"name": name}, applies_to="tool_calls",
                confidence=round(g_frac - b_frac, 2),
            ))
        elif bad and b_frac >= _TOOL_GOOD_FRAC and g_frac < _TOOL_BAD_FRAC:
            out.append(_proposal(
                "tool_not_called", f"avoids {name}",
                f"{name} used in {b_frac:.0%} of bad vs {g_frac:.0%} of good episodes",
                b_ids, counts, params={"name": name}, applies_to="tool_calls",
                confidence=round(b_frac - g_frac, 2),
            ))
    return out


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _common_arg_fields(good: list[_View], name: str) -> set[str]:
    """Argument field names present in every good episode's call(s) to `name`."""
    per_episode: list[set[str]] = []
    for v in good:
        keys: set[str] = set()
        for c in v.tool_calls:
            if c.get("name") == name:
                keys |= set(_parse_arguments(c.get("arguments")))
        per_episode.append(keys)
    return set.intersection(*per_episode) if per_episode else set()


def _state_assertion(name: str, field_name: str, good: list[_View]) -> Proposal:
    needle = f'"{field_name}"'
    expr = (f"any(t.get('name') == {name!r} and {needle!r} in (t.get('arguments') or '') "
            f"for t in tools)")
    return _proposal(
        "expr", f"{name} sets {field_name}",
        f"every good episode called {name} with a {field_name!r} argument",
        [v.episode.id for v in good], {"good": len(good)},
        params={"expr": expr}, applies_to="tool_calls", confidence=_STATE_CONFIDENCE,
    )


def _state_proposals(good: list[_View]) -> list[Proposal]:
    """Rank 3: assert a tool is called with an argument field that is set across all good episodes.

    Only fires when the tool is called in *every* good episode and the field appears in every such
    call, so the assertion holds across the whole good set (it drops otherwise)."""
    if len(good) < 2:  # a single episode is not enough repetition to trust a state assertion
        return []
    per_episode = [{c["name"] for c in v.tool_calls if c.get("name")} for v in good]
    common_tools = set.intersection(*per_episode) if per_episode else set()
    out: list[Proposal] = []
    for name in sorted(common_tools):
        for field_name in sorted(_common_arg_fields(good, name)):
            out.append(_state_assertion(name, field_name, good))
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
    return [_proposal(
        "json_schema", "output is JSON",
        f"{len(objs)}/{len(with_output)} good outputs parse as JSON objects",
        [v.episode.id for v in with_output], {"json": len(objs)},
        params={"schema": schema}, confidence=round(len(objs) / len(with_output), 2),
    )]


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
    return [_proposal(
        "max_length", "output length bound",
        f"p99 good output length {p99} chars, +{slack_pct}% slack",
        [v.episode.id for v in good if v.final_output], {"p99": p99},
        params={"max": limit}, severity="soft", confidence=0.3,
    )]


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
    return [_proposal(
        "no_pii", "no PII in output",
        f"{', '.join(sorted(kinds))} leaked in {len(hit_ids)} output(s)",
        hit_ids, {"leaks": len(hit_ids)}, params={"kinds": sorted(kinds)}, confidence=1.0,
    )]


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


def _phrase_counts(good_grams: list[set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for grams in good_grams:
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
    return counts


def _qualifying_phrases(
    counts: dict[str, int], good: list[_View], bad: list[_View], bad_grams: list[set[str]]
) -> list[tuple[str, float]]:
    """Phrases in >= _PHRASE_GOOD_FRAC of good outputs and <= _PHRASE_BAD_FRAC of bad ones."""
    qualifying: list[tuple[str, float]] = []
    for gram, gc in counts.items():
        g_frac = gc / len(good)
        if g_frac < _PHRASE_GOOD_FRAC:
            continue
        b_frac = (sum(gram in gg for gg in bad_grams) / len(bad)) if bad else 0.0
        if b_frac <= _PHRASE_BAD_FRAC:
            qualifying.append((gram, g_frac))
    return qualifying


def _phrase_proposals(good: list[_View], bad: list[_View]) -> list[Proposal]:
    if not good:
        return []
    good_grams = [_ngrams(v.final_output) for v in good]
    bad_grams = [_ngrams(v.final_output) for v in bad]
    qualifying = _qualifying_phrases(_phrase_counts(good_grams), good, bad, bad_grams)
    qualifying.sort(key=lambda gf: (-gf[1], -len(gf[0]), gf[0]))
    chosen = _distinct_phrases(qualifying)

    out: list[Proposal] = []
    for gram, g_frac in chosen[:5]:
        hits = [v.episode.id for v in good if gram in _joined(v.final_output)]
        out.append(_proposal(
            "contains", f"says '{gram}'",
            f"'{gram}' appears in {g_frac:.0%} of good outputs, rare in bad",
            hits, {"good": len(hits)},
            params={"values": [gram], "mode": "any"}, severity="soft",
            confidence=round(g_frac, 2),
        ))
    return out
