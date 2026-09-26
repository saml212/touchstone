"""Edit a task's TOML verifier files: a ``reward.toml`` dimension weight, or a judge criterion.

Split out of `review.changes` so its criteria-``.py`` round-tripping stays small. Both editors read
the file, refuse anything that is not the rewardkit shape they expect, and write it back atomically.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from ..survey.writes import atomic_write
from .guard import ChangeError


def load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ChangeError(f"{path.name} could not be parsed; edit it by hand.") from exc


# ---- reward.toml dimension weights -----------------------------------------


def apply_reward(path: Path, criterion, weight) -> None:
    doc = load_toml(path)
    rewards = doc.get("reward")
    if not (isinstance(rewards, list) and rewards and isinstance(rewards[0].get("weights"), dict)):
        raise ChangeError(f"{path.name} is not a rewardkit reward.toml; edit it by hand.")
    weights = rewards[0]["weights"]
    if criterion not in weights:
        raise ChangeError(f"'{criterion}' is not a dimension in {path.name}.")
    if weight is None:
        raise ChangeError("a weight change needs a `weight` value.")
    weights[criterion] = float(weight)
    atomic_write(path, tomli_w.dumps(doc))


# ---- judge criteria toml ---------------------------------------------------


def _match_judge(items: list[dict], criterion) -> int:
    for i, item in enumerate(items):
        if item.get("description") == criterion:
            return i
    raise ChangeError(f"no judge criterion matches '{criterion}'.")


def _judge_entry(change: dict) -> dict:
    entry = {"description": change.get("description", ""),
             "type": (change.get("params") or {}).get("type", "binary")}
    points = (change.get("params") or {}).get("points")
    if points is not None:
        entry["points"] = points
    return entry


def apply_judge(path: Path, change: dict) -> None:
    doc = load_toml(path)
    if "judge" not in doc or not isinstance(doc.get("criterion"), list):
        raise ChangeError(f"{path.name} is not a rewardkit judge file; edit it by hand.")
    items = doc["criterion"]
    op = change["op"]
    if op == "add":
        items.append(_judge_entry(change))
    elif op == "edit":
        items[_match_judge(items, change.get("criterion"))] = _judge_entry(change)
    elif op == "remove":
        items.pop(_match_judge(items, change.get("criterion")))
    else:
        raise ChangeError(f"unknown op {op!r} for a judge file.")
    atomic_write(path, tomli_w.dumps({"judge": doc["judge"], "criterion": items}))
