"""Provider protocol + Reply. Every provider (scripted now; real ones later) returns a Reply."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "arguments": str}]
    usage: dict | None = None  # {"tokens_in": int, "tokens_out": int}
    raw: Any = None


@runtime_checkable
class Provider(Protocol):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json: bool = False,
        timeout: float = 60,
    ) -> Reply: ...

    async def achat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json: bool = False,
        timeout: float = 60,
    ) -> Reply: ...
