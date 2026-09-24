"""`checks.toml`: policies applied to every task — mined safety + interview "always" rules.

A policy is a `Check` plus an `enabled` flag. Mining appends proposals disabled; `checks enable`
flips them. When a task is materialised (mine, interview, `tasks sync`) every enabled policy whose
reference gate passes is copied into the task's `task.toml` with `source = "policy"`, so each task
directory is self-contained. Editing `checks.toml` and re-running `tasks sync` re-materialises.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import tomli_w

from .checks import SAFETY_KINDS, Check, Target, evaluate
from .messages import text_of


def policies_path(root: str | Path) -> Path:
    return Path(root) / "checks.toml"


@dataclass
class Policy:
    check: Check
    enabled: bool = False


def read_policies(root: str | Path) -> list[Policy]:
    path = policies_path(root)
    if not path.exists():
        return []
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    return [Policy(check=Check.from_toml(b), enabled=bool(b.get("enabled", False)))
            for b in doc.get("check", [])]


def write_policies(root: str | Path, policies: list[Policy]) -> Path:
    blocks = []
    for p in policies:
        block = p.check.to_toml()
        block["enabled"] = p.enabled
        blocks.append(block)
    path = policies_path(root)
    path.write_text(tomli_w.dumps({"check": blocks}), encoding="utf-8")
    return path


def gate_check(check: Check, reference: dict, messages: list[dict], failure: bool) -> bool:
    """Reference-consistent attachment: a non-safety check attaches only if the reference passes it.

    Safety kinds ("avoid this") always attach; a failure task carries safety checks only.
    """
    if failure:
        return check.kind in SAFETY_KINDS
    if check.kind in SAFETY_KINDS:
        return True
    target = Target(output_text=text_of(reference),
                    tool_calls=reference.get("tool_calls") or [], messages=messages,
                    reference=reference)
    result = evaluate([check], target)[0]
    return bool(result.passed) or (check.kind == "judge" and bool(reference.get("content")))


def get_policy(root: str | Path, name: str) -> Policy | None:
    return next((p for p in read_policies(root) if p.check.name == name), None)


def add_policy(root: str | Path, policy: Policy) -> None:
    write_policies(root, read_policies(root) + [policy])


def set_enabled(root: str | Path, name: str, enabled: bool) -> bool:
    policies = read_policies(root)
    hit = next((p for p in policies if p.check.name == name), None)
    if hit is None:
        return False
    hit.enabled = enabled
    write_policies(root, policies)
    return True


def enable_all_mined(root: str | Path) -> int:
    policies = read_policies(root)
    changed = [p for p in policies if p.check.source == "mined" and not p.enabled]
    for p in changed:
        p.enabled = True
    write_policies(root, policies)
    return len(changed)


def materialize(task, policies: list[Policy]) -> list[Check]:
    """The policy checks to write into `task`: every enabled policy the reference gate lets in,
    tagged `source = "policy"`. Interview/manual blocks are preserved separately by write_task."""
    reference = task.reference or {"content": "", "tool_calls": []}
    messages = (task.context or {}).get("messages", [])
    failure = "failure" in (task.tags or [])
    return [replace(p.check, source="policy")
            for p in policies if p.enabled and gate_check(p.check, reference, messages, failure)]
