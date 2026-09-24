"""The interview policy: turn a room conversation into committed checks.

`Interviewer` opens with a factual summary of the task and asks one concrete question, then on each
turn either drafts checks (via the LLM, asked for JSON only), commits the current draft when a
participant confirms, or — when participants disagree — asks the group to settle it before writing
anything down. Draft checks are stored `enabled=0` and linked to the room; committing flips them to
`enabled=1` and attaches them to the task. Malformed LLM output never raises: it degrades to a plain
clarifying question. No state machine framework — just a short, ordered set of rules per turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .. import store
from ..checks import Check as DslCheck
from ..checks.dsl import PARAM_SPEC
from ..llm.prompt import extract_json
from . import rooms

_CONFIRM = ("that's right", "thats right", "sounds right", "lgtm", "yes", "yep", "yeah",
            "agreed", "agree", "correct", "commit")
_OBJECT = ("disagree", "not right", "wrong", "instead", "actually no", "don't", "dont", "no")

# Words that carry no new rule, so a confirmation built only from these is a plain "yes".
_AMEND_STOP = set(_CONFIRM) | {
    "commit", "both", "please", "also", "then", "and", "ok", "okay", "sure", "too",
    "the", "a", "an", "it", "that", "this", "is", "to", "do", "fine", "good", "great",
}

_CLARIFY = "What should it have done differently — and is that a hard rule or a preference?"


@dataclass
class AgentTurn:
    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}


def _check_to_dict(c: store.Check) -> dict:
    return {"id": c.id, "name": c.name, "kind": c.kind, "params": c.params,
            "applies_to": c.applies_to, "severity": c.severity,
            "rationale": c.rationale, "enabled": bool(c.enabled)}


def _has(text: str, phrases) -> bool:
    low = text.lower()
    for phrase in phrases:
        if " " in phrase:
            if phrase in low:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", low):
            return True
    return False


class Interviewer:
    def __init__(self, provider, conn, room: store.Room) -> None:
        self.provider = provider
        self.conn = conn
        self.room = room

    # -- opening -------------------------------------------------------------

    def open_statement(self) -> str:
        task = store.get_task(self.conn, self.room.task_id) if self.room.task_id else None
        if task is None:
            return (
                f"Let's define what good looks like for “{self.room.topic}”. "
                f"{_CLARIFY}"
            )
        lines = [f"We're reviewing task {task.name} — topic: {self.room.topic}."]
        last_user = self._last_user_turn(task)
        if last_user:
            lines.append(f"The user asked: {last_user}")
        lines.append(self._what_model_did(task))
        outcome = self._outcome(task)
        if outcome:
            lines.append(outcome)
        attached = self._attached_kinds(task)
        lines.append(
            f"Checks already attached: {attached}." if attached else "No checks attached yet."
        )
        lines.append(_CLARIFY)
        return " ".join(lines)

    def _last_user_turn(self, task: store.Task) -> str:
        for msg in reversed((task.context or {}).get("messages", [])):
            if msg.get("role") == "user":
                return _clip(msg.get("content", ""))
        return ""

    def _what_model_did(self, task: store.Task) -> str:
        ref = task.reference or {}
        calls = [tc.get("name") for tc in ref.get("tool_calls") or [] if tc.get("name")]
        content = _clip(ref.get("content", ""))
        if calls and content:
            return f"The model called {', '.join(calls)} and replied: {content}"
        if calls:
            return f"The model called {', '.join(calls)}."
        if content:
            return f"The model replied: {content}"
        return "The model produced no visible output."

    def _outcome(self, task: store.Task) -> str:
        ep = store.get_episode(self.conn, task.episode_id) if task.episode_id else None
        if ep is None or (ep.outcome_label is None and ep.outcome_score is None):
            return ""
        label = ep.outcome_label or "unlabelled"
        score = "" if ep.outcome_score is None else f" (score {ep.outcome_score:g})"
        return f"Outcome: {label}{score}."

    def _attached_kinds(self, task: store.Task) -> str:
        kinds = []
        for cid in task.check_ids or []:
            c = store.get_check(self.conn, cid)
            if c:
                kinds.append(c.kind)
        return ", ".join(kinds)

    # -- per-turn policy -----------------------------------------------------

    def respond(self, history: list[dict]) -> AgentTurn:
        new = _since_agent(history)
        if any(_is_cmd(m, "/done") for m in new):
            rooms.close(self.conn, self.room.id)
            return AgentTurn(say="Closing the room — the committed checks are saved. Thanks all.")

        affirm, object_ = self._stances(new)
        if affirm and object_:
            return self._ask_to_settle(affirm, object_)

        explicit = any(_is_cmd(m, "/commit") for m in new)
        if explicit or (affirm and not object_):
            # "Yes, but with X" — a confirmation carrying a new rule must be revised through the
            # LLM before it's committed, so the committed check reflects the amendment, not the
            # stale draft. A bare "yes" commits the draft as-is.
            if not explicit and self._has_amendment(new):
                return self._revise_then_commit(history)
            return self._commit_draft()

        return self._llm_turn(history)

    def _has_amendment(self, new: list[dict]) -> bool:
        for m in new:
            if m.get("role") != "user" or not _has(m.get("text", ""), _CONFIRM):
                continue
            words = re.findall(r"[a-z0-9']+", m.get("text", "").lower())
            if len([w for w in words if w not in _AMEND_STOP]) >= 3:
                return True
        return False

    def _revise_then_commit(self, history: list[dict]) -> AgentTurn:
        if self.provider is None:  # no LLM to revise with: commit what's already drafted
            return self._commit_draft()
        self._llm_turn(history)  # persists the revised draft
        return self._commit_draft()

    def _stances(self, new: list[dict]) -> tuple[set[str], set[str]]:
        affirm: set[str] = set()
        object_: set[str] = set()
        for m in new:
            if m.get("role") != "user":
                continue
            text = m.get("text", "")
            speaker = m.get("speaker", "someone")
            if _has(text, _OBJECT):
                object_.add(speaker)
            elif _has(text, _CONFIRM):
                affirm.add(speaker)
        return affirm, object_

    def _ask_to_settle(self, affirm: set[str], object_: set[str]) -> AgentTurn:
        names = sorted(affirm | object_)
        who = " and ".join(names) if len(names) == 2 else ", ".join(names)
        return AgentTurn(
            say=(f"{who} — you're not agreed yet. Can you settle whether this should be a hard "
                 f"rule before I commit it?"),
            draft=self._draft_dicts(),
        )

    def _commit_draft(self) -> AgentTurn:
        draft = self._draft_checks()
        if not draft:
            return AgentTurn(
                say="There's no draft to commit yet — tell me the rule and I'll draft it."
            )
        committed = [self._commit(c) for c in draft]
        names = ", ".join(c.get("name") or c["kind"] for c in committed)
        return AgentTurn(
            say=f"Committed: {names}. Anything else, or /done to close?",
            commit=committed,
        )

    def _llm_turn(self, history: list[dict]) -> AgentTurn:
        if self.provider is None:
            return AgentTurn(say=_CLARIFY, draft=self._draft_dicts())
        try:
            reply = self.provider.chat(self._prompt(history))
            data = json.loads(extract_json(reply.content) or reply.content)
        except Exception:
            return AgentTurn(say=_CLARIFY, draft=self._draft_dicts())
        if not isinstance(data, dict):
            return AgentTurn(say=_CLARIFY, draft=self._draft_dicts())

        for raw in _as_list(data.get("draft")):
            self._persist(raw, enabled=False)
        committed = [self._commit(raw) for raw in _as_list(data.get("commit"))
                     if self._persist(raw, enabled=True)]
        say = data.get("say") if isinstance(data.get("say"), str) and data["say"] else _CLARIFY
        return AgentTurn(say=say, draft=self._draft_dicts(),
                         commit=[c for c in committed if c])

    def _prompt(self, history: list[dict]) -> list[dict]:
        spec = "\n".join(f"  {k}: {v}" for k, v in PARAM_SPEC.items())
        system = (
            "You are Touchstone's interviewer. You turn stakeholders' opinions about an agent's "
            "behaviour into concrete checks. Reply with JSON ONLY, no prose around it, shaped as:\n"
            '{"say": "one short reply/question", '
            '"draft": [{"kind": ..., "params": {...}, "name": "...", "severity": "hard|soft", '
            '"applies_to": "final|any_turn|tool_calls", "rationale": "..."}], "commit": []}\n'
            "Prefer a programmatic kind (contains, regex, tool_called, json_schema, expr, …) that "
            "states the rule exactly; use 'judge' ONLY when no programmatic kind can express it. "
            "Only draft checks; leave commit empty — a human confirms before committing. Ask one "
            "concrete question at a time. Address people by name. Available check kinds and their "
            f"params:\n{spec}"
        )
        convo = "\n".join(f"{m.get('speaker', '?')} ({m.get('role', '?')}): {m.get('text', '')}"
                          for m in history)
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nRespond with JSON."}]

    # -- draft / commit state (store-backed) ---------------------------------

    def _room_checks(self) -> list[store.Check]:
        ids = store.list_room_check_ids(self.conn, self.room.id)
        return [c for c in (store.get_check(self.conn, i) for i in ids) if c]

    def _draft_checks(self) -> list[store.Check]:
        return [c for c in self._room_checks() if not c.enabled]

    def _draft_dicts(self) -> list[dict]:
        return [_check_to_dict(c) for c in self._draft_checks()]

    def _find_room_check(self, kind: str, params: dict) -> store.Check | None:
        target = json.dumps(params, sort_keys=True)
        for c in self._room_checks():
            if c.kind == kind and json.dumps(c.params, sort_keys=True) == target:
                return c
        return None

    def _persist(self, raw, enabled: bool) -> store.Check | None:
        if not isinstance(raw, dict):
            return None
        try:
            dc = DslCheck.from_dict(raw)
            dc.validate()
        except (ValueError, TypeError, KeyError):
            return None
        existing = self._find_room_check(dc.kind, dc.params)
        if existing is not None:
            if enabled and not existing.enabled:
                return self._enable(existing.id)
            return existing
        check = store.insert_check(
            self.conn,
            store.Check(
                name=raw.get("name") or dc.kind,
                kind=dc.kind,
                params=dc.params,
                applies_to=dc.applies_to,
                severity=dc.severity,
                source="interview",
                rationale=raw.get("rationale", "") or "",
                enabled=1 if enabled else 0,
            ),
        )
        store.link_room_check(self.conn, self.room.id, check.id)
        if enabled:
            self._attach_to_task(check.id)
        return check

    def _commit(self, raw_or_check) -> dict:
        if isinstance(raw_or_check, store.Check):
            return _check_to_dict(self._enable(raw_or_check.id))
        check = self._persist(raw_or_check, enabled=True)
        return _check_to_dict(check) if check else {}

    def _enable(self, check_id: str) -> store.Check:
        store.set_check_enabled(self.conn, check_id, True)
        self._attach_to_task(check_id)
        return store.get_check(self.conn, check_id)

    def _attach_to_task(self, check_id: str) -> None:
        if not self.room.task_id:
            return
        task = store.get_task(self.conn, self.room.task_id)
        if task is None:
            return
        ids = list(task.check_ids or [])
        if check_id not in ids:
            ids.append(check_id)
            store.update_task(self.conn, task.id, check_ids=ids)


def _since_agent(history: list[dict]) -> list[dict]:
    idx = -1
    for i, m in enumerate(history):
        if m.get("role") == "assistant":
            idx = i
    return history[idx + 1:]


def _is_cmd(msg: dict, cmd: str) -> bool:
    return msg.get("role") == "user" and msg.get("text", "").strip().lower().startswith(cmd)


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _clip(text, limit: int = 160) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
