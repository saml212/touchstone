"""The project's next-step hint — one function the Overview page and the bare `touchstone` share.

v3: the loop is capture -> survey -> bench -> review -> train. Stage 1 ships capture, bench, and the
review rooms; the hint names the next real move from the trace count.
"""

from __future__ import annotations

from . import store


def project_signals(conn) -> dict:
    """The counts `next_step` reads."""
    return {
        "episodes": len(store.list_episodes(conn)),
        "rooms": len(store.list_rooms(conn)),
    }


def next_step(s: dict) -> str:
    """The one line the UI and CLI show. `s` is a `project_signals` dict."""
    if not s["episodes"]:
        return "Capture traces first — run touchstone demo, or add touchstone.trace() to your app."
    return "Benchmark a model — run touchstone bench -m <provider/model>."
