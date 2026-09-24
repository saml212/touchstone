"""The interview policy: turn a room conversation into committed checks written to files.

`Interviewer` opens with a factual summary of the task and asks one concrete question, then on each
turn either drafts checks (via the LLM, asked for JSON only), commits the current draft when a
participant confirms, or — when participants disagree — asks the group to settle it before writing
anything down. The draft lives in room state (a `draft` room message); committing writes a
`[[metadata.touchstone.check]]` block into the task's `task.toml` (source `interview`), or, when the
LLM flags a check `policy: true` ("always"/"every task"), an enabled policy in `checks.toml`. A bare
"yes" commits the current draft as-is; a confirmation carrying an amendment is revised through the
LLM first. Malformed LLM output never raises: it degrades to a plain clarifying question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .. import store
from .. import tasks as tasks_mod
from ..checks import Check as DslCheck
from ..checks.dsl import PARAM_SPEC
from ..llm.prompt import extract_json
from ..policies import Policy, read_policies, write_policies
from . import rooms


@dataclass
class AgentTurn:
    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}

_CONFIRM = ("that's right", "thats right", "sounds right", "lgtm", "yes", "yep", "yeah",
            "agreed", "agree", "correct", "commit")
_OBJECT = ("disagree", "not right", "wrong", "instead", "actually no", "don't", "dont", "no")

# Words that carry no new rule, so a confirmation built only from these is a plain "yes".
_AMEND_STOP = set(_CONFIRM) | {
    "commit", "both", "please", "also", "then", "and", "ok", "okay", "sure", "too",
    "the", "a", "an", "it", "that", "this", "is", "to", "do", "fine", "good", "great",
}

_CLARIFY = "What should it have done differently — and is that a hard rule or a preference?"


def _has(text: str, phrases) -> bool:
    low = text.lower()
    for phrase in phrases:
        if " " in phrase:
            if phrase in low:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", low):
            return True
    return False


def _check_dict(check: DslCheck, *, policy: bool = False) -> dict:
    return {"name": check.name or check.kind, "kind": check.kind, "params": check.params,
            "applies_to": check.applies_to, "severity": check.severity, "source": check.source,
            "rule": check.rule, "because": check.because, "policy": policy}


class Interviewer:
    def __init__(self, provider, conn, room: store.Room, root) -> None:
        self.provider = provider
        self.conn = conn
        self.room = room
        self.root = root

    def _task(self):
        if not self.room.task_id:
            return None
        return tasks_mod.get_task(self.root, self.room.task_id)

    # -- opening -------------------------------------------------------------

    def open_statement(self) -> str:
        task = self._task()
        if task is None:
            return f"Let's define what good looks like for “{self.room.topic}”. {_CLARIFY}"
        lines = [f"We're reviewing task {task.name} — topic: {self.room.topic}."]
        last_user = self._last_user_turn(task)
        if last_user:
            lines.append(f"The user asked: {last_user}")
        lines.append(self._what_model_did(task))
        outcome = self._outcome(task)
        if outcome:
            lines.append(outcome)
        kinds = ", ".join(c.kind for c in task.checks)
        lines.append(f"Checks already attached: {kinds}." if kinds else "No checks attached yet.")
        lines.append(_CLARIFY)
        return " ".join(lines)

    def _last_user_turn(self, task) -> str:
        for msg in reversed((task.context or {}).get("messages", [])):
            if msg.get("role") == "user":
                return _clip(msg.get("content", ""))
        return ""

    def _what_model_did(self, task) -> str:
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

    def _outcome(self, task) -> str:
        ep = store.get_episode(self.conn, task.episode_id) if task.episode_id else None
        if ep is None or (ep.outcome_label is None and ep.outcome_score is None):
            return ""
        label = ep.outcome_label or "unlabelled"
        score = "" if ep.outcome_score is None else f" (score {ep.outcome_score:g})"
        return f"Outcome: {label}{score}."

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
            return self._confirm_turn(history, new, explicit)

        return self._llm_turn(history)

    def _confirm_turn(self, history: list[dict], new: list[dict], explicit: bool) -> AgentTurn:
        # "Yes, but with X" — a confirmation carrying a new rule must be revised through the LLM
        # first, so the committed check reflects the amendment, not the stale draft.
        if not explicit and self._has_amendment(new) and self.provider is not None:
            self._llm_turn(history)  # persists the revised draft
        return self._commit_draft()

    def _has_amendment(self, new: list[dict]) -> bool:
        for m in new:
            if m.get("role") != "user" or not _has(m.get("text", ""), _CONFIRM):
                continue
            words = re.findall(r"[a-z0-9']+", m.get("text", "").lower())
            if len([w for w in words if w not in _AMEND_STOP]) >= 3:
                return True
        return False

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
            draft=self.draft(),
        )

    def _commit_draft(self) -> AgentTurn:
        draft = self.draft()
        if not draft:
            return AgentTurn(say="There's no draft to commit yet — tell me the rule to draft.")
        committed = [self._commit(d) for d in draft]
        committed = [c for c in committed if c]
        self._set_draft([])
        names = ", ".join(c.get("name") or c["kind"] for c in committed)
        return AgentTurn(say=f"Committed: {names}. Anything else, or /done to close?",
                         commit=committed)

    def _llm_turn(self, history: list[dict]) -> AgentTurn:
        if self.provider is None:
            return AgentTurn(say=_CLARIFY, draft=self.draft())
        try:
            reply = self.provider.chat(self._prompt(history))
            data = json.loads(extract_json(reply.content) or reply.content)
        except Exception:
            return AgentTurn(say=_CLARIFY, draft=self.draft())
        if not isinstance(data, dict):
            return AgentTurn(say=_CLARIFY, draft=self.draft())

        draft = [d for d in (_normalize(raw) for raw in _as_list(data.get("draft"))) if d]
        self._set_draft(draft)
        committed = [c for c in (self._commit(raw) for raw in _as_list(data.get("commit"))) if c]
        say = data.get("say") if isinstance(data.get("say"), str) and data["say"] else _CLARIFY
        return AgentTurn(say=say, draft=draft, commit=committed)

    def _prompt(self, history: list[dict]) -> list[dict]:
        spec = "\n".join(f"  {k}: {v}" for k, v in PARAM_SPEC.items())
        system = (
            "You are Touchstone's interviewer. You turn stakeholders' opinions about an agent's "
            "behaviour into concrete checks. Reply with JSON ONLY, no prose around it, shaped as:\n"
            '{"say": "one short reply/question", '
            '"draft": [{"kind": ..., "params": {...}, "name": "...", "rule": "...", '
            '"severity": "hard|soft", "applies_to": "final|any_turn|tool_calls", '
            '"because": "...", "policy": false}], "commit": []}\n'
            "Set \"policy\": true when the stakeholder says the rule applies to EVERY task "
            "(\"always\", \"every task\"); otherwise false so it attaches to this task only. "
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

    # -- draft / commit state ------------------------------------------------

    def draft(self) -> list[dict]:
        """The current uncommitted draft, from the latest `draft` room message."""
        for m in reversed(store.list_room_messages(self.conn, self.room.id)):
            if m.role == "draft":
                try:
                    return json.loads(m.text)
                except json.JSONDecodeError:
                    return []
        return []

    def _set_draft(self, checks: list[dict]) -> None:
        rooms.post(self.conn, self.room.id, "agent", "draft",
                   json.dumps(checks, ensure_ascii=False))

    def _commit(self, raw: dict) -> dict:
        """Write one confirmed check to the task (or to checks.toml when flagged a policy)."""
        norm = _normalize(raw)
        if norm is None:
            return {}
        check = DslCheck(kind=norm["kind"], params=norm["params"], name=norm["name"],
                         applies_to=norm["applies_to"], severity=norm["severity"],
                         rule=norm.get("rule", ""), because=norm.get("because", ""),
                         source="interview")
        check.id = check.name
        if norm.get("policy"):
            self._commit_policy(check)
        elif self.room.task_id:
            tasks_mod.append_check(self.root, self.room.task_id, check)
        return _check_dict(check, policy=bool(norm.get("policy")))

    def _commit_policy(self, check: DslCheck) -> None:
        policies = read_policies(self.root)
        key = (check.kind, json.dumps(check.params, sort_keys=True))
        for p in policies:
            if (p.check.kind, json.dumps(p.check.params, sort_keys=True)) == key:
                p.enabled = True
                write_policies(self.root, policies)
                return
        write_policies(self.root, policies + [Policy(check=check, enabled=True)])

    def committed(self) -> list[dict]:
        """The interview checks written to this room's task (for the room panel)."""
        task = self._task()
        return [_check_dict(c) for c in (task.checks if task else []) if c.source == "interview"]


def _normalize(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    try:
        check = DslCheck.from_dict(raw)
        check.validate()
    except (ValueError, TypeError, KeyError):
        return None
    return {"name": raw.get("name") or check.kind, "kind": check.kind, "params": check.params,
            "applies_to": check.applies_to, "severity": check.severity,
            "rule": raw.get("rule", "") or "",
            "because": raw.get("because") or raw.get("rationale") or "",
            "policy": bool(raw.get("policy"))}


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
