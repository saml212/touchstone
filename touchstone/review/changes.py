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
from pathlib import Path

import tomli_w

from ..harbor import rewardkit
from ..survey import descriptions
from ..survey.writes import atomic_write
from . import guard, tomls
from .guard import ChangeError  # a single ChangeError type across changes + guard

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


def _tests_root(path: Path) -> Path | None:
    """The `tests/` directory a criteria file lives under, or None if it is not under one."""
    return next((p for p in path.parents if p.name == "tests"), None)


def existing_criteria_files(tests_root: Path | None) -> list[str]:
    """The criteria files (`tests/**/*.py`) that already exist, task-relative, for a clear error."""
    if tests_root is None or not tests_root.is_dir():
        return []
    task_dir = tests_root.parent
    return sorted(p.relative_to(task_dir).as_posix() for p in tests_root.rglob("*.py"))


def _missing_criteria_msg(rel: str, tests_root: Path | None) -> str:
    files = existing_criteria_files(tests_root)
    have = ", ".join(files) if files else "none yet"
    return (f"there is no {rel} to change; this task's checks are in: {have}. Edit one of those, "
            "or add a new check.")


def _ensure_reward_dimension(tests_root: Path, path: Path) -> None:
    """A new criteria file in a fresh dimension dir must be weighted in reward.toml, the same wiring
    the survey writes — else Harbor never aggregates it into the reward."""
    parts = path.relative_to(tests_root).parts
    if len(parts) < 2:
        return
    dim, reward = parts[0], tests_root / "reward.toml"
    if not reward.is_file():
        dims = sorted({d.name for d in tests_root.iterdir() if d.is_dir()} | {dim})
        rewardkit.write_reward_toml(tests_root, dims)
        return
    doc = tomls.load_toml(reward)
    rewards = doc.get("reward")
    if isinstance(rewards, list) and rewards and isinstance(rewards[0].get("weights"), dict):
        weights = rewards[0]["weights"]
        if dim not in weights:
            weights[dim] = 1.0
            atomic_write(reward, tomli_w.dumps(doc))


def _read_or_create(path: Path, op: str, rel: str) -> list[str]:
    """The file's criteria calls, creating an empty rewardkit file (import + reward wiring) for an
    `add` to a task that never had this check, or refusing an edit/remove on an absent file."""
    if path.exists():
        return parse_criteria(path)
    if op != "add":
        raise ChangeError(_missing_criteria_msg(rel, _tests_root(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tests_root = _tests_root(path)
    if tests_root is not None:
        _ensure_reward_dimension(tests_root, path)
    return []


def _task_dir_of(path: Path) -> Path:
    root = _tests_root(path)
    return root.parent if root is not None else path.parent


def _guard_call(path: Path, params: dict) -> None:
    """Refuse a full criterion call with placeholder or (for sqlite) unreal arguments, before it is
    rendered — the `expected` shortcut keeps an already-accepted call, so it is not re-guarded."""
    guard.check_call(_task_dir_of(path), (params.get("fn") or "").removeprefix("rk."),
                     params.get("args", []))


def _apply_py(path: Path, op: str, criterion, params: dict | None, rel: str) -> None:
    calls = _read_or_create(path, op, rel)
    if op == "add":
        _guard_call(path, params or {})
        calls.append(_render_call(params or {}))
    elif op == "edit":
        i = _index(criterion, len(calls))
        params = params or {}
        if "expected" in params:
            calls[i] = _with_expected(calls[i], params["expected"])
        else:
            _guard_call(path, params)
            calls[i] = _render_call(params)
    elif op == "remove":
        calls.pop(_index(criterion, len(calls)))
    else:
        raise ChangeError(f"unknown op {op!r} for a criteria file.")
    rewardkit.write_criteria(path.parent, path.stem, calls)


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
        tomls.apply_reward(path, change.get("criterion"), change.get("weight"))
    elif path.suffix == ".py":
        _apply_py(path, op, change.get("criterion"), change.get("params"), rel)
        _sync_descriptions(task_dir, rel, path, change)
    elif path.suffix == ".toml":
        tomls.apply_judge(path, change)
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


def _check_intent(change, intent: str) -> None:
    """Refuse a criterion whose KIND contradicts the person's words (a tool-use disagreement drafted
    as a sqlite query, say). Only runs when `intent` carries the person's words — an apply, whose
    go-ahead is a bare "yes", passes no intent and so is judged on shape alone."""
    if not intent:
        return
    for c in as_list(change):
        fn = (c.get("params") or {}).get("fn")
        if fn:
            guard.check_intent(fn, intent)


def validate(task_dir: str | Path, change, intent: str = "") -> None:
    """Dry-run a change against a throwaway copy of the task; raise ChangeError with a precise
    message if it cannot be applied (wrong shape, unknown op, missing file, out-of-range criterion,
    placeholder/unreal arguments, or a kind that contradicts `intent`, the person's words).
    Nothing in the real task is written — the review agent validates before it reads a change back
    and again before it commits, so a malformed draft never reaches disk."""
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise ChangeError(f"there is no task {task_dir.name!r} to change.")
    _check_intent(change, intent)
    guard.check_queries(task_dir, change)  # run the SELECT against the real state.db schema
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "task"
        shutil.copytree(task_dir, clone)
        apply(clone, change)
