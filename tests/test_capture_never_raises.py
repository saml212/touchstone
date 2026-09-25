"""Capture must never raise into the user's application (capture/spans.py module docstring).

The SDK-patch path was already guarded, but the explicit-instrumentation surface — `episode()`,
`@tool`, `outcome()`, `record_llm_call()` — funnelled db failures straight into the caller. Attack:
point capture at a db path that cannot be created (its parent is a file, so `mkdir` raises
NotADirectoryError on the first connect). Every public capture entry must swallow it, warn once, and
let the app keep running — and a decorated tool must still return its real result.
"""

from __future__ import annotations

import touchstone
from touchstone.capture import context


def _unwritable_db(tmp_path) -> str:
    blocker = tmp_path / "afile"
    blocker.write_text("x")  # a regular file used as a parent dir -> mkdir/connect fails
    return str(blocker / "sub" / "touchstone.db")


def _point_capture_at(db: str) -> None:
    context._db_path = None
    context._local.__dict__.pop("conn", None)
    context._untracked_ids.clear()
    touchstone.trace(db)


def test_trace_on_unwritable_db_does_not_raise(tmp_path):
    _point_capture_at(_unwritable_db(tmp_path))  # trace() itself must not connect/raise


def test_episode_tool_outcome_survive_unwritable_db(tmp_path, caplog):
    _point_capture_at(_unwritable_db(tmp_path))

    @touchstone.tool
    def add(a, b):
        return a + b

    ran = []
    with touchstone.episode("ep"):  # entering the room must not raise
        ran.append(add(2, 3))  # the tool's own result must survive a failed recording
        touchstone.outcome(1.0, "ok")  # scoring must not raise
    touchstone.record_llm_call("m", [{"role": "user", "content": "hi"}], "yo")

    assert ran == [5]
    assert any("touchstone" in r.message for r in caplog.records)
