"""The incumbent baseline provider: replay a task's own recorded reference as the model reply.

Selected with `-m reference`, it stands for "what the traced agent actually did". The runner passes
the task on the provider call path (other providers ignore the extra `task` kwarg); this returns the
task's stored reference message verbatim — the honest incumbent to prove a cheaper candidate
against. Argument-free, so any model part is ignored. Entry point: `ReferenceProvider().chat(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Reply


@dataclass
class ReferenceProvider:
    name: str = "reference"

    def _reply(self, task) -> Reply:
        reference = (getattr(task, "reference", None) or {}) if task is not None else {}
        return Reply(
            content=reference.get("content") or "",
            tool_calls=list(reference.get("tool_calls") or []),
        )

    def chat(self, messages, tools=None, json=False, timeout=60, task=None) -> Reply:
        return self._reply(task)

    async def achat(self, messages, tools=None, json=False, timeout=60, task=None) -> Reply:
        return self._reply(task)
