"""The plain read-back of a change — what a product person says yes to before it is applied.

The check is rendered in plain words (its description, or the check in English) AND its literal
rewardkit arguments, so a person hears the words while a reader sees the exact values that will be
written: "the agent used order_status — rk.trajectory_tool_used('order_status')". This is how a
placeholder or a nonsense argument becomes visible before anyone agrees to it."""

from __future__ import annotations

from pathlib import Path

from .changes import ChangeError, _render_call, as_list


def _literal(change: dict) -> str:
    """The exact ``rk.<fn>(...)`` source the change would write, appended so the values are seen —
    or "" for an `expected`-only edit (which keeps the existing call) or a non-call change."""
    params = change.get("params") or {}
    if not params.get("fn"):
        return ""
    try:
        return f" — {_render_call(params)}"
    except ChangeError:
        return ""


def _describe_one(change: dict) -> str:
    op = change.get("op", "edit")
    file = change.get("file", "?")
    if op == "text":
        return (f"rewrite the {Path(file).stem} — this changes what the customer asks, not how it "
                "is scored, so rewards will not move until the tasks are re-run")
    if Path(file).name == "reward.toml":
        return f"set the {change.get('criterion')} weight to {change.get('weight')}"
    if op == "add":
        return f"add a check: {_summarise_target(change)}{_literal(change)}"
    if op == "remove":
        return f"remove check {change.get('criterion')}"
    n, target = change.get("criterion"), _summarise_target(change)
    return f"change check {n} to {target}{_literal(change)}"


_WORDS = {
    "sqlite_query_equals": lambda a: f"the query {a[1]!r} returns {a[2]!r}",
    "trajectory_tool_used": lambda a: f"the agent used {a[0]}",
    "trajectory_tool_not_used": lambda a: f"the agent did not use {a[0]}",
}


def _summarise_target(change: dict) -> str:
    """Plain words for the read-back: the change's own description first, else the check in
    words (never raw code — a product person has to say yes to this sentence)."""
    params = change.get("params", {})
    description = change.get("description") or params.get("description")
    if description:
        return description
    if "expected" in params:
        return f"now expects {params['expected']!r}"
    fn, args = params.get("fn"), params.get("args", [])
    words = _WORDS.get(fn)
    try:
        return words(args) if words else _render_call(params)
    except (IndexError, ChangeError):
        return _render_call(params) if fn else ""


def describe(change) -> str:
    """A plain read-back of what the change does, for the agent to say before applying it."""
    return "; ".join(_describe_one(c) for c in as_list(change))
