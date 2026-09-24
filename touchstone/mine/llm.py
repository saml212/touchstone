"""The LLM mining pass: prompt a provider for DSL checks, parse and validate the reply.

`mine_llm` renders a sample of episodes and any discovered code (system prompts, tool schemas) into
one prompt, asks the `agent_provider` for a JSON array of checks in the Touchstone DSL, and keeps
only the proposals that validate. A provider or transport failure never crashes mining — it logs and
returns nothing. Proposals are the same `Proposal` type `mine/stats.py` produces, marked
medium-trust until a human reviews them.
"""

from __future__ import annotations

import json
import logging

from .. import store
from ..checks import Check as DslCheck
from ..checks.dsl import PARAM_SPEC
from .codebase import Snippet
from .stats import Proposal, final_output

log = logging.getLogger("touchstone.mine")

_LLM_EPISODE_SAMPLE = 20
_LLM_MAX_PROPOSALS = 25
_LLM_SNIPPET_CAP = 20
_LLM_CONFIDENCE = 0.5  # LLM proposals are medium-trust until a human reviews them


def _user_texts(span: store.Span) -> list[str]:
    return [
        m["content"]
        for m in (span.input or {}).get("messages", [])
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]


def _last_user_message(spans: list[store.Span]) -> str:
    user = ""
    for span in spans:
        if span.kind != "model":
            continue
        for text in _user_texts(span):
            user = text
    return user


def _outcome_text(ep: store.Episode) -> str:
    if ep.outcome_label:
        return ep.outcome_label
    if ep.outcome_score is not None:
        return f"scored {ep.outcome_score}"
    return "unlabeled"


def _render_episode(conn, ep: store.Episode) -> str:
    spans = store.list_spans(conn, ep.id)
    user = _last_user_message(spans)
    tools = sorted({s.name for s in spans if s.kind == "tool"})
    outcome = _outcome_text(ep)
    lines = [
        f"- user: {user[:300]}",
        f"  tools_called: {', '.join(tools) or 'none'}",
        f"  final_output: {final_output(spans)[:300]}",
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
        confidence=_LLM_CONFIDENCE,
    )
