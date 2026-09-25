"""Pure helpers for the review agent's replies: grounding a trial in its scores, surfacing a tool
error, detecting `/done`, and turning room messages into chat messages. Kept out of `review.agent`
so the state machine stays small and these stay unit-testable on their own.
"""

from __future__ import annotations

import json

FALLBACK = "Let me look at that."


def error_of(result: str) -> str | None:
    """The first line of a tool result's `error`, or None when the call succeeded."""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    err = data.get("error") if isinstance(data, dict) else None
    return err.splitlines()[0] if err else None


def tool_error_reply(error: str | None) -> str:
    """What to say when the loop ends on a tool error or runs out of steps with no spoken reply."""
    if not error:
        return FALLBACK
    return (f"I couldn't apply that change: {error} "
            "Tell me the check by number and I'll try again.")


def reward_pct(reward) -> str:
    return "unknown" if reward is None else f"{round(reward * 100)}%"


def _passed(score) -> bool:
    return score is True or score == 1


def grounding_line(detail: dict) -> str:
    """The verifier's actual result in one line: reward %, each criterion pass/fail, empty note."""
    parts = [f"The verifier scored this {reward_pct(detail.get('reward'))}."]
    crits = detail.get("criteria") or []
    if crits:
        marks = "; ".join(f"{c.get('description', '?')} — "
                          f"{'passed' if _passed(c.get('score')) else 'failed'}" for c in crits)
        parts.append(f"Checks: {marks}.")
    if not detail.get("trajectory"):
        parts.append("The agent did nothing that was recorded.")
    return " ".join(parts)


def wants_done(history: list[dict]) -> bool:
    """The participants asked to close the room since the agent last spoke (`/done`)."""
    idx = max((i for i, m in enumerate(history) if m.get("role") == "assistant"), default=-1)
    return any(m.get("role") == "user" and m.get("text", "").strip().lower().startswith("/done")
               for m in history[idx + 1:])


def as_messages(history: list[dict]) -> list[dict]:
    """Room messages -> chat messages: participants are users, the agent is the assistant."""
    out = []
    for m in history:
        role = "assistant" if m.get("role") == "assistant" else "user"
        who = m.get("speaker", "")
        text = m.get("text", "")
        out.append({"role": role, "content": f"{who}: {text}" if role == "user" else text})
    return out


def applied_reply(result: dict) -> str:
    """The read-out after a successful apply, composed here so it never depends on the model
    having a step left: what was applied where, what the regrade moved, what could not regrade."""
    tasks = result.get("applied_to") or []
    lines = [f"Applied to {', '.join(tasks)}." if tasks else "Applied."]
    deltas = result.get("deltas") or []
    if deltas:
        moved = "; ".join(f"{d['task']} {reward_pct(d['before'])} → {reward_pct(d['after'])}"
                          for d in deltas)
        lines.append(f"Regraded: {moved}. Nothing else moved.")
    elif result.get("job"):
        lines.append("Regraded: no reward changed.")
    failed = result.get("failed") or []
    if failed:
        broken = "; ".join(f"{f['task']} ({f['error']})" for f in failed)
        lines.append(f"Could not regrade: {broken}")
    if result.get("reverted"):
        lines.append("That change broke the verifier on every task it touched, so I put the files "
                     "back as they were.")
    return " ".join(lines) + " Want to look at the next trial?"
