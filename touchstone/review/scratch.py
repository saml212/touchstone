"""The room's scratch state: the trial being discussed, the proposal read back, what was applied.
Kept as a role="draft" message so the UI reads it and the agent survives a restart."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Scratch:
    current: dict | None = None   # {"task", "trial"}
    proposed: object = None       # a change (object or list) awaiting confirmation
    readback: str = ""
    applied: list = None          # summaries of applied changes this room

    def to_dict(self) -> dict:
        return {"current": self.current, "proposed": self.proposed,
                "readback": self.readback, "applied": self.applied or []}


