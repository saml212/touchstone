"""The review-room agent.

v3 stage 1 keeps a minimal, importable conversational agent so the rooms server, speech, and
realtime bridges keep working. The verifier-correction logic (read a task and its trials, propose a
criterion change, trigger `harbor job regrade`, record trust) is rewritten in stage 4 against the
Harbor job directories and the `reviews` table. Until then the agent opens with a plain statement
and answers each turn with a short LLM reply, drafting and committing nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .. import store
from ..llm.prompt import extract_json
from . import rooms

_CLARIFY = "What should good look like here — and how would you know it when you saw it?"


@dataclass
class AgentTurn:
    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}


class Interviewer:
    def __init__(self, provider, conn, room: store.Room, root) -> None:
        self.provider = provider
        self.conn = conn
        self.room = room
        self.root = root

    def open_statement(self) -> str:
        return f"Let's review “{self.room.topic}”. {_CLARIFY}"

    def respond(self, history: list[dict]) -> AgentTurn:
        new = _since_agent(history)
        if any(_is_cmd(m, "/done") for m in new):
            rooms.close(self.conn, self.room.id)
            return AgentTurn(say="Closing the room — thanks all.")
        return self._reply(history)

    def _reply(self, history: list[dict]) -> AgentTurn:
        if self.provider is None:
            return AgentTurn(say=_CLARIFY)
        try:
            reply = self.provider.chat(self._prompt(history))
            data = json.loads(extract_json(reply.content) or reply.content)
            say = data.get("say") if isinstance(data, dict) else None
        except Exception:
            say = None
        return AgentTurn(say=say if isinstance(say, str) and say else _CLARIFY)

    def _prompt(self, history: list[dict]) -> list[dict]:
        system = (
            "You are Touchstone's review agent. You help product people say, in plain words, what "
            "good behaviour looks like for their AI agent. Reply with JSON ONLY: "
            '{"say": "one short reply or question"}. Ask one concrete question at a time and '
            "address people by name."
        )
        convo = "\n".join(f"{m.get('speaker', '?')} ({m.get('role', '?')}): {m.get('text', '')}"
                          for m in history)
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nRespond with JSON."}]

    def draft(self) -> list[dict]:
        """The current uncommitted draft, from the latest `draft` room message (empty in v3 s1)."""
        for m in reversed(store.list_room_messages(self.conn, self.room.id)):
            if m.role == "draft":
                try:
                    return json.loads(m.text)
                except json.JSONDecodeError:
                    return []
        return []

    def committed(self) -> list[dict]:
        """Verifier criteria committed from this room (rewritten in stage 4)."""
        return []


def _since_agent(history: list[dict]) -> list[dict]:
    idx = -1
    for i, m in enumerate(history):
        if m.get("role") == "assistant":
            idx = i
    return history[idx + 1:]


def _is_cmd(msg: dict, cmd: str) -> bool:
    return msg.get("role") == "user" and msg.get("text", "").strip().lower().startswith(cmd)
