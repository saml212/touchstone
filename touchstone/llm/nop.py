"""The nop baseline provider: always an empty reply, no tool calls.

It is one half of the survey's task gate — a task counts only when the oracle scores 1 and the nop
scores 0, because a task an empty answer already passes measures nothing. Selected with `-m nop`;
argument-free, so Harbor's `nop/x` maps cleanly. Entry point: `NopProvider().chat(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Reply


@dataclass
class NopProvider:
    name: str = "nop"

    def chat(self, messages, tools=None, json=False, timeout=60) -> Reply:
        return Reply(content="", tool_calls=[])

    async def achat(self, messages, tools=None, json=False, timeout=60) -> Reply:
        return Reply(content="", tool_calls=[])
