"""Deterministic provider for tests and the zero-key demo.

Reply is a pure function of the last user message. `rules` maps a substring to a canned reply
(text, a tool call, or malformed JSON), letting tests and the demo pin exact behavior. With no
matching rule the reply is derived from a stable hash of the last user message, so reruns are
byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .base import Reply


@dataclass
class Rule:
    substring: str
    content: str | None = None
    tool_call: dict | None = None  # {"name": str, "arguments": dict|str}
    malformed_json: bool = False


@dataclass
class ScriptedProvider:
    rules: list[Rule] = field(default_factory=list)
    name: str = "scripted"

    def _last_user(self, messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        return ""

    def _rule_reply(self, rule: Rule, want_json: bool) -> Reply:
        if rule.malformed_json:
            return Reply(content='{"ok": true,', usage={"tokens_in": 5, "tokens_out": 3})
        if rule.tool_call is not None:
            args = rule.tool_call.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            return Reply(
                content=rule.content or "",
                tool_calls=[{"name": rule.tool_call["name"], "arguments": args}],
                usage={"tokens_in": 8, "tokens_out": 6},
            )
        content = rule.content or ""
        if want_json and content:
            content = json.dumps({"answer": content}, ensure_ascii=False)
        return Reply(content=content, usage={"tokens_in": 8, "tokens_out": 6})

    def _fallback_reply(self, text: str, want_json: bool) -> Reply:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        content = f"scripted-reply-{digest[:12]}"
        if want_json:
            content = json.dumps({"echo_hash": digest[:12]}, ensure_ascii=False)
        tin = len(text.split())
        return Reply(content=content, usage={"tokens_in": tin, "tokens_out": len(content.split())})

    def _reply(self, messages: list[dict], want_json: bool) -> Reply:
        text = self._last_user(messages)
        for rule in self.rules:
            if rule.substring in text:
                return self._rule_reply(rule, want_json)
        return self._fallback_reply(text, want_json)

    def chat(self, messages, tools=None, json=False, timeout=60) -> Reply:
        return self._reply(messages, json)

    async def achat(self, messages, tools=None, json=False, timeout=60) -> Reply:
        return self._reply(messages, json)
