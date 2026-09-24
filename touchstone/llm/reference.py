"""The incumbent baseline: replay a task's own recorded reference as the model reply.

`reference` is the model spec for "what the traced agent actually did". The runner passes the task
on its provider call path (other providers ignore the extra `task` kwarg); this provider returns the
task's stored reference message verbatim. Because mining attaches a check to a task only when the
reference passes it, `bench run -m reference` passes every non-failure task by construction — it is
the honest incumbent to prove a cheaper candidate against, and a way to confirm the attached checks
are satisfiable.
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
