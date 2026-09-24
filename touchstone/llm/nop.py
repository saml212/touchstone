"""The nop baseline: an empty reply, no tool calls.

`nop` is the second half of the task-generation gate from the Harbor RFC — the oracle
(`reference`) must score 1 and the nop must score 0. A task an empty reply already passes measures
nothing, so it never enters a benchmark. `tasks.validate` computes the nop verdict inline; this
provider makes `bench run -m nop` available and lets Sample run the same empty attempt through the
shared runner path.
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
