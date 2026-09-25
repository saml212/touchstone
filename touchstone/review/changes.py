"""Criterion edits as data, applied to the `tests/` files `harbor/rewardkit.py` wrote.

A change is a small dict the review agent drafts and reads back before it is applied:

- rewardkit criteria line (``tests/correctness/*.py``): ``{"op": "add"|"edit"|"remove", "file",
  "criterion" (1-based index for edit/remove), "params": {"fn", "args": [...]}}``.
- a dimension weight (``tests/reward.toml``): ``{"op": "edit", "file", "criterion": <dimension>,
  "weight": <float>}``.
- a judge criterion (a ``[[criterion]]`` toml): ``{"op": "add"|"edit"|"remove", "file",
  "criterion": <description to match>, "description", "weight", "params": {"type", "points"}}``.
- instruction/persona wording: ``{"op": "text", "file", "text"}``.

The engine round-trips exactly the shapes rewardkit writes: a ``.py`` criteria file must be
``import rewardkit as rk`` followed by ``rk.<fn>(...)`` calls and nothing else; a hand-edited or
otherwise unparseable file is refused with a clear message so a human edits it instead. Paths are
relative to the task directory.
"""

from __future__ import annotations

import ast
import shutil
import tempfile
import tomllib
from pathlib import Path

import tomli_w

from ..harbor import rewardkit
from ..survey import descriptions
from ..survey.writes import atomic_write


class ChangeError(ValueError):
    """A change could not be applied safely; the message is one clear sentence for the human."""


# ---- rewardkit criteria .py files ------------------------------------------


def parse_criteria(path: Path) -> list[str]:
    """The ``rk.<fn>(...)`` call sources in a rewardkit criteria file, or refuse a hand edit."""
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ChangeError(f"{path.name} is not valid Python; edit it by hand.") from exc
    calls: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Import) and _imports_rewardkit(node):
            continue
        segment = _rk_call_source(node, source)
        if segment is None:
            raise ChangeError(
                f"{path.name} was hand-edited beyond rewardkit calls; edit it by hand.")
        calls.append(segment)
    return calls


def _imports_rewardkit(node: ast.Import) -> bool:
    return any(a.name == "rewardkit" and a.asname == "rk" for a in node.names)


def _rk_call_source(node: ast.stmt, source: str) -> str | None:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    func = node.value.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id == "rk"):
        return None
    return ast.get_source_segment(source, node.value)


# The rewardkit built-ins a change may call (docs: built-in criteria). A model that invents a name
# or prefixes it with "rk." is corrected or refused here, before anything is written.
KNOWN_CRITERIA = {
    "sqlite_query_equals", "json_key_equals", "csv_cell_equals", "xlsx_cell_equals",
    "file_exists", "file_contains", "trajectory_tool_used", "trajectory_tool_not_used",
    "trajectory_turn_count", "http_get_status",
}


def _render_call(params: dict) -> str:
    fn = (params.get("fn") or "").removeprefix("rk.")
    if not fn:
        raise ChangeError("a criterion change needs params.fn (the rewardkit function).")
    if fn not in KNOWN_CRITERIA:
        raise ChangeError(f"unknown criterion {fn!r}; use one of {sorted(KNOWN_CRITERIA)}.")
    args = ", ".join(repr(a) for a in params.get("args", []))
    return f"rk.{fn}({args})"


def _with_expected(source: str, expected) -> str:
    """The same call with its last argument (what the check expects) replaced — the edit a product
    person makes most: "the number should be X". Keeps the db, query, and function untouched."""
    node = ast.parse(source).body[0].value
    old = node.args[-1].value if node.args and isinstance(node.args[-1], ast.Constant) else None
    if isinstance(old, (int, float)) and isinstance(expected, str):
        try:
            expected = float(expected) if "." in expected else int(expected)
        except ValueError as exc:
            raise ChangeError(f"expected {expected!r} is not a number like the current "
                              "value.") from exc
    node.args[-1] = ast.Constant(value=expected)
    return ast.unparse(node)


def _index(criterion, count: int) -> int:
    try:
        idx = int(criterion) - 1
    except (TypeError, ValueError) as exc:
        raise ChangeError("edit/remove needs `criterion` as the 1-based criterion number.") from exc
    if not 0 <= idx < count:
        raise ChangeError(f"there is no criterion {criterion} in this file.")
    return idx


def _apply_py(path: Path, op: str, criterion, params: dict | None) -> None:
    calls = parse_criteria(path)
    if op == "add":
        calls.append(_render_call(params or {}))
    elif op == "edit":
        i = _index(criterion, len(calls))
        params = params or {}
        calls[i] = (_with_expected(calls[i], params["expected"]) if "expected" in params
                    else _render_call(params))
    elif op == "remove":
        calls.pop(_index(criterion, len(calls)))
    else:
        raise ChangeError(f"unknown op {op!r} for a criteria file.")
    rewardkit.write_criteria(path.parent, path.stem, calls)


# ---- reward.toml dimension weights -----------------------------------------


def _load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ChangeError(f"{path.name} could not be parsed; edit it by hand.") from exc


def _apply_reward(path: Path, criterion, weight) -> None:
    doc = _load_toml(path)
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


def _apply_judge(path: Path, change: dict) -> None:
    doc = _load_toml(path)
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


# ---- text (instruction / persona) ------------------------------------------


def _apply_text(path: Path, text) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ChangeError("a text change needs the replacement `text`.")
    atomic_write(path, text.strip() + "\n")


# ---- dispatch --------------------------------------------------------------


def _resolve(task_dir: Path, rel: str) -> Path:
    path = (task_dir / rel).resolve()
    if not str(path).startswith(str(task_dir.resolve())):
        raise ChangeError("a change may only touch files inside the task.")
    return path


def _apply_one(task_dir: Path, change: dict) -> str:
    rel = change.get("file")
    if not rel:
        raise ChangeError("a change needs a `file`.")
    path = _resolve(task_dir, rel)
    op = change.get("op", "edit")
    if op == "text":
        _apply_text(path, change.get("text"))
    elif path.name == "reward.toml":
        _apply_reward(path, change.get("criterion"), change.get("weight"))
    elif path.suffix == ".py":
        _apply_py(path, op, change.get("criterion"), change.get("params"))
        _sync_descriptions(task_dir, rel, path, change)
    elif path.suffix == ".toml":
        _apply_judge(path, change)
    else:
        raise ChangeError(f"don't know how to change {path.name}.")
    return rel


def _change_desc(change: dict) -> str:
    """The description for an added/edited criterion: the change's text, else one from the call."""
    if change.get("description"):
        return str(change["description"])
    params = change.get("params") or {}
    return descriptions.describe_call(params.get("fn", ""), params.get("args", []))


def _sync_descriptions(task_dir: Path, rel: str, path: Path, change: dict) -> None:
    """Keep tests/descriptions.toml aligned after a .py criteria edit: indexes shift on add/remove,
    and a new or edited criterion takes the change's description (or one derived from its call)."""
    tests_dir = task_dir / "tests"
    mapping = descriptions.load(tests_dir)
    prefix = f"{rel}:"
    entries = {int(k[len(prefix):]): v for k, v in mapping.items() if k.startswith(prefix)}
    for k in [k for k in mapping if k.startswith(prefix)]:
        del mapping[k]
    op = change.get("op", "edit")
    if op == "remove":
        i = int(change["criterion"])
        entries.pop(i, None)
        entries = {(x - 1 if x > i else x): d for x, d in entries.items()}
    elif op == "edit":
        entries[int(change["criterion"])] = _change_desc(change) or \
            entries.get(int(change["criterion"]), "")
    elif op == "add":
        entries[len(parse_criteria(path))] = _change_desc(change) or "an added check"
    for idx, desc in entries.items():
        mapping[descriptions.key(rel, idx)] = desc
    descriptions.write(tests_dir, mapping)


def as_list(change) -> list[dict]:
    """A change may be one edit or several applied together; normalise to a list."""
    if isinstance(change, dict):
        return [change]
    if isinstance(change, list) and all(isinstance(c, dict) for c in change):
        return list(change)
    raise ChangeError("a change must be an object or a list of objects.")


def apply(task_dir: str | Path, change) -> list[str]:
    """Apply a change (or list of changes) to a task's files; return the files touched, or raise."""
    task_dir = Path(task_dir)
    return [_apply_one(task_dir, c) for c in as_list(change)]


def validate(task_dir: str | Path, change) -> None:
    """Dry-run a change against a throwaway copy of the task; raise ChangeError with a precise
    message if it cannot be applied (wrong shape, unknown op, missing file, out-of-range criterion).
    Nothing in the real task is written — the review agent validates before it reads a change back
    and again before it commits, so a malformed draft never reaches disk."""
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise ChangeError(f"there is no task {task_dir.name!r} to change.")
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "task"
        shutil.copytree(task_dir, clone)
        apply(clone, change)


# ---- read-back -------------------------------------------------------------


def _describe_one(change: dict) -> str:
    op = change.get("op", "edit")
    file = change.get("file", "?")
    if op == "text":
        return f"rewrite {Path(file).stem}"
    if Path(file).name == "reward.toml":
        return f"set the {change.get('criterion')} weight to {change.get('weight')}"
    if op == "add":
        return f"add a check: {_summarise_target(change)}"
    if op == "remove":
        return f"remove check {change.get('criterion')}"
    return f"change check {change.get('criterion')} to {_summarise_target(change)}"


_WORDS = {
    "sqlite_query_equals": lambda a: f"the query {a[1]!r} returns {a[2]!r}",
    "trajectory_tool_used": lambda a: f"the agent used {a[0]}",
    "trajectory_tool_not_used": lambda a: f"the agent did not use {a[0]}",
}


def _summarise_target(change: dict) -> str:
    """Plain words for the read-back: the change's own description first, else the check in
    words (never raw code — a product person has to say yes to this sentence)."""
    if change.get("description"):
        return change["description"]
    params = change.get("params", {})
    fn, args = params.get("fn"), params.get("args", [])
    words = _WORDS.get(fn)
    try:
        return words(args) if words else _render_call(params)
    except (IndexError, ChangeError):
        return _render_call(params) if fn else ""


def describe(change) -> str:
    """A plain read-back of what the change does, for the agent to say before applying it."""
    return "; ".join(_describe_one(c) for c in as_list(change))
