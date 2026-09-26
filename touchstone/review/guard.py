"""Reject a criterion change the reviewer's model drafted badly, before it touches a task's files.

Three guards, each raising ChangeError with one plain sentence the tool result hands back so the
model can retry once with real values (and, failing that, the reviewer tells the person it could
not build a check it trusts):

- placeholder text pasted straight from the schema (``<db path…>``, ``…``, ``""``, "TODO"),
- a ``sqlite_query_equals`` whose db path is not one of this dataset's simulator ``state.db`` files
  or whose query does not start with ``SELECT``,
- a criterion whose KIND contradicts the person's words — a "the agent used tool X" disagreement
  drafted as a sqlite query, or a "the record should be Y" disagreement drafted as a tool check.

The intent table (`_INTENT`) maps the person's words to the criterion KIND they asked for; the
kind of each rewardkit built-in is `_FN_KIND`. A definite intent that disagrees with the drafted
`fn` is refused with a hint naming the built-in and file to use instead.
"""

from __future__ import annotations

import ast
import re
import shutil
import sqlite3
import tempfile
import tomllib
from pathlib import Path


class ChangeError(ValueError):
    """A change could not be applied safely; the message is one clear sentence for the human."""


# ---- placeholder text ------------------------------------------------------

_ANGLE = re.compile(r"<[A-Za-z][^<>]{0,80}>")  # "<db path from a sibling check>", "<full SQL>"
_PLACEHOLDER_WORDS = ("placeholder", "todo")


def _placeholder_reason(value) -> str | None:
    """Why `value` reads as a stand-in rather than a real argument, or None when it is fine."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return "an empty string"
    if _ANGLE.search(s):
        return "a <…> placeholder"
    if s in {"...", "…"} or "…" in s:
        return "an ellipsis placeholder"
    if any(w in s.lower() for w in _PLACEHOLDER_WORDS):
        return "placeholder text"
    return None


def check_placeholders(args: list) -> None:
    for value in args:
        reason = _placeholder_reason(value)
        if reason is not None:
            raise ChangeError(
                f"{value!r} is {reason}, not a real value — give the actual value (copy the db "
                "path and table from a sibling check, or state the value the person expects).")


# ---- sqlite_query_equals: real db path + a SELECT --------------------------


def _normalise_db(path: str) -> str:
    return str(path).replace("\\", "/").removeprefix("/app/").lstrip("/")


def _artifact_dbs(task_dir: Path) -> set[str]:
    toml = task_dir / "task.toml"
    if not toml.is_file():
        return set()
    try:
        data = tomllib.loads(toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    arts = data.get("artifacts", [])
    return {_normalise_db(a) for a in arts if isinstance(a, str) and a.endswith("state.db")}


def _sibling_dbs(task_dir: Path) -> set[str]:
    """Every db path a `sqlite_query_equals` in this task's tests already targets."""
    out: set[str] = set()
    tests = task_dir / "tests"
    if not tests.is_dir():
        return out
    for py in tests.rglob("*.py"):
        out |= _dbs_in_source(py.read_text(encoding="utf-8", errors="ignore"))
    return out


def _dbs_in_source(source: str) -> set[str]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(module):
        if (isinstance(node, ast.Call) and _is_rk(node.func, "sqlite_query_equals")
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(_normalise_db(node.args[0].value))
    return out


def _is_rk(func, name: str) -> bool:
    return (isinstance(func, ast.Attribute) and func.attr == name
            and isinstance(func.value, ast.Name) and func.value.id == "rk")


def valid_db_paths(task_dir: Path) -> list[str]:
    """This task's simulator ``state.db`` paths — from its task.toml artifacts and any sibling
    sqlite check — that a new state criterion may query. Normalised (no ``/app/`` prefix)."""
    return sorted(_artifact_dbs(task_dir) | _sibling_dbs(task_dir))


def _check_sqlite(task_dir: Path, args: list) -> None:
    if len(args) < 2:
        raise ChangeError("a sqlite check needs a db path, a SELECT query, and the expected value.")
    db, query = args[0], args[1]
    valid = valid_db_paths(task_dir)
    if not isinstance(db, str) or _normalise_db(db) not in valid:
        known = ", ".join(valid) if valid else "none are known for this task"
        raise ChangeError(
            f"{db!r} is not one of this task's simulator databases ({known}); copy the db path "
            "from a sibling state check.")
    if not isinstance(query, str) or not query.lstrip().upper().startswith("SELECT"):
        raise ChangeError(f"a sqlite check's query must start with SELECT; got {query!r}.")


def check_call(task_dir: Path, fn: str, args: list) -> None:
    """Placeholder + (for sqlite) db-path/SELECT validation for a criterion's real arguments."""
    check_placeholders(args)
    if fn == "sqlite_query_equals":
        _check_sqlite(task_dir, args)


# ---- sqlite_query_equals: the query must run against the real schema --------

_TABLE_RE = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _sim_db_file(task_dir: Path, db: str) -> Path | None:
    """The on-disk simulator ``state.db`` a check queries — the dataset's shared file at
    ``<dataset>/simulators/…``, not a per-task copy — or None when it has not been materialised."""
    path = task_dir.parent.parent / _normalise_db(db)
    return path if path.is_file() else None


def _schema_reason(conn: sqlite3.Connection, query: str, exc: sqlite3.Error) -> str:
    """A rejection that names the real columns, so a wrong column (``refunded_amount`` for
    ``refunded``) is fixed against the actual schema rather than silently failing every regrade."""
    match = _TABLE_RE.search(query)
    if match:
        cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({match.group(1)})")]
        if cols:
            return (f"that query does not run against the data ({exc}); the {match.group(1)} table "
                    f"has columns {', '.join(cols)} — use one of those.")
    return f"that query does not run against the data ({exc})."


def _run_select(db_file: Path, query: str) -> None:
    """Run the SELECT against a fresh read-only copy of the state.db; reject a query the schema does
    not support (with the real columns) or one that matches no row (a wrong id or value)."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "state.db"
        shutil.copy2(db_file, copy)
        conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            _execute_select(conn, query)
        finally:
            conn.close()


def _execute_select(conn: sqlite3.Connection, query: str) -> None:
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.Error as exc:
        raise ChangeError(_schema_reason(conn, query, exc)) from exc
    if not rows:
        raise ChangeError(f"that query matched no row ({query!r}) — check the id and column "
                          "against the order data before it can grade anything.")


def _as_change_list(change) -> list:
    if isinstance(change, dict):
        return [change]
    return [c for c in change if isinstance(c, dict)] if isinstance(change, list) else []


def check_queries(task_dir: Path, change) -> None:
    """Run every ``sqlite_query_equals`` SELECT in a change against the real materialised state.db,
    so a wrong column or an id that matches no row is refused before the change is read back."""
    for c in _as_change_list(change):
        _check_one_query(task_dir, c)


def _check_one_query(task_dir: Path, change: dict) -> None:
    params = change.get("params") or {}
    if (params.get("fn") or "").removeprefix("rk.") != "sqlite_query_equals":
        return
    args = params.get("args") or []
    if len(args) < 2 or not isinstance(args[0], str) or not isinstance(args[1], str):
        return
    db_file = _sim_db_file(task_dir, args[0])
    if db_file is not None:
        _run_select(db_file, args[1])


# ---- intent -> criterion kind ----------------------------------------------

_FN_KIND = {
    "sqlite_query_equals": "state", "json_key_equals": "state",
    "csv_cell_equals": "state", "xlsx_cell_equals": "state",
    "trajectory_tool_used": "trajectory", "trajectory_tool_not_used": "trajectory",
    "trajectory_turn_count": "trajectory",
    "file_contains": "answer", "file_exists": "answer", "http_get_status": "answer",
}

# Words that name a kind, checked in this priority order — a tool-use phrase wins over a state one
# ("looked up the order with order_status" is a tool-use disagreement, not a state one).
_INTENT: list[tuple[str, tuple[str, ...]]] = [
    ("trajectory", ("tool", "used ", "use ", "using", "call", "called", "invoke", "invoked",
                     "look up", "looked up", "lookup", "did not use", "didn't use", "not use",
                     "without", "avoid", "skip")),
    ("answer", ("reply", "respond", "response", "answer", "mention", "said", "say ",
                "tell the customer", "message to", "wording of the reply")),
    ("state", ("record", "field", "should be", "should show", "should read", "value", "set to",
               "column", " row", "database", "status", "refund", "balance", "amount", "shows ",
               "marked", "stored", "in the db", "the order should")),
]

_HINT = {
    ("trajectory", "state"): (
        "This reads as a tool-use check (whether the agent used a tool), but a sqlite query grades "
        "a stored value. Use rk.trajectory_tool_used('<tool>') (or trajectory_tool_not_used) in "
        "tests/correctness/trajectory.py instead."),
    ("state", "trajectory"): (
        "This reads as a check on a stored value, but a trajectory check only grades which tools "
        "ran. Use rk.sqlite_query_equals, copying the db path and table from a sibling check in "
        "tests/correctness/state.py, instead."),
    ("answer", "state"): (
        "This reads as a check on the reply text, not a stored value. Use rk.file_contains on the "
        "answer file instead."),
    ("answer", "trajectory"): (
        "This reads as a check on the reply text, not which tools ran. Use rk.file_contains on the "
        "answer file instead."),
}


def intended_kind(words: str) -> str | None:
    """The criterion KIND the person's words ask for, or None when the words don't clearly say."""
    said = f" {(words or '').lower()} "
    for kind, markers in _INTENT:
        if any(m in said for m in markers):
            return kind
    return None


def check_intent(fn: str, words: str) -> None:
    """Refuse a criterion whose kind contradicts a definite intent, with a hint to the right one."""
    want = intended_kind(words)
    got = _FN_KIND.get((fn or "").removeprefix("rk."))
    if want and got and want != got:
        raise ChangeError(_HINT.get((want, got),
                          f"the person asked for a {want} check but this is a {got} check."))
