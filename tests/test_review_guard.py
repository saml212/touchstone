"""guard.py refuses a criterion change the model drafted badly — placeholder or unreal arguments,
a sqlite check that names no real database or is not a SELECT, and a kind that contradicts the
person's words — each with a plain sentence the reviewer can retry from or relay."""

from __future__ import annotations

import sqlite3

import pytest

from touchstone.harbor import rewardkit
from touchstone.review import changes, guard
from touchstone.review.guard import ChangeError


def _task(tmp_path, *, artifacts=True, sibling=True):
    d = tmp_path / "tasks" / "check-order-status-1"
    (d / "tests" / "correctness").mkdir(parents=True)
    toml = '[metadata.touchstone]\njob = "Check order status"\ntools = ["order_status"]\n'
    if artifacts:
        toml = 'artifacts = ["/app/simulators/orders_api/state.db"]\n' + toml
    (d / "task.toml").write_text(toml, encoding="utf-8")
    if sibling:
        rewardkit.write_criteria(d / "tests" / "correctness", "state", [
            "rk.sqlite_query_equals('simulators/orders_api/state.db', "
            "\"SELECT refunded FROM orders WHERE id='B1791'\", 0.0)"])
    rewardkit.write_reward_toml(d / "tests", ["correctness"])
    return d


# ---- placeholder rejections (one per stand-in kind) ------------------------


@pytest.mark.parametrize("bad", ["<db path from a sibling check>", "<full SQL>", "",
                                 "...", "…", "TODO", "placeholder value"])
def test_placeholder_arguments_are_refused(bad):
    with pytest.raises(ChangeError):
        guard.check_placeholders([bad])


def test_real_arguments_pass_the_placeholder_guard():
    guard.check_placeholders(["order_status", "simulators/orders_api/state.db", 0.0])


def test_a_less_than_clause_is_not_mistaken_for_a_placeholder():
    # "amount < 5 AND qty > 0" contains < and > but is a real predicate, not a <token>.
    guard.check_placeholders(["SELECT n FROM t WHERE amount < 5 AND qty > 0"])


# ---- sqlite: real db path + a SELECT ---------------------------------------


def test_sqlite_rejects_a_db_path_that_is_not_a_simulator_db(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(ChangeError, match="not one of this task's simulator databases"):
        guard.check_call(task, "sqlite_query_equals",
                         ["nonsense.db", "SELECT 1 FROM orders", 1])


def test_sqlite_rejects_a_query_that_is_not_a_select(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(ChangeError, match="must start with SELECT"):
        guard.check_call(task, "sqlite_query_equals",
                         ["simulators/orders_api/state.db", "DELETE FROM orders", 1])


def test_sqlite_accepts_a_db_path_from_task_toml_artifacts(tmp_path):
    task = _task(tmp_path, sibling=False)  # only the artifact names the db
    guard.check_call(task, "sqlite_query_equals",
                     ["simulators/orders_api/state.db", "SELECT refunded FROM orders", 0.0])


def test_sqlite_accepts_a_db_path_from_a_sibling_check(tmp_path):
    task = _task(tmp_path, artifacts=False)  # only a sibling names the db
    guard.check_call(task, "sqlite_query_equals",
                     ["/app/simulators/orders_api/state.db", "SELECT refunded FROM orders", 0.0])


def test_valid_db_paths_normalises_and_dedupes(tmp_path):
    assert guard.valid_db_paths(_task(tmp_path)) == ["simulators/orders_api/state.db"]


# ---- sqlite: the query must run against the real schema --------------------


def _with_state_db(tmp_path, rows=(("B1614", "processing", 44.07, 0.0),)):
    """A task whose simulator state.db really exists, so a SELECT can be run against it."""
    task = _task(tmp_path)
    db = tmp_path / "simulators" / "orders_api" / "state.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders "
                 "(id TEXT PRIMARY KEY, status TEXT, total REAL, refunded REAL)")
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return task


def _sqlite_change(query):
    return {"op": "add", "file": "tests/correctness/state.py",
            "params": {"fn": "sqlite_query_equals",
                       "args": ["simulators/orders_api/state.db", query, 0.0]}}


def test_query_with_a_real_column_passes(tmp_path):
    task = _with_state_db(tmp_path)
    guard.check_queries(task, _sqlite_change("SELECT refunded FROM orders WHERE id='B1614'"))


def test_query_with_a_wrong_column_is_rejected_with_the_real_columns(tmp_path):
    task = _with_state_db(tmp_path)
    with pytest.raises(ChangeError, match="refunded") as exc:
        guard.check_queries(task, _sqlite_change(
            "SELECT refunded_amount FROM orders WHERE id='B1614'"))
    assert "refunded_amount" not in str(exc.value).split("columns")[1]  # not in the columns list


def test_query_matching_no_row_is_rejected(tmp_path):
    task = _with_state_db(tmp_path)
    with pytest.raises(ChangeError, match="matched no row"):
        guard.check_queries(task, _sqlite_change("SELECT refunded FROM orders WHERE id='NOPE'"))


def test_query_check_is_skipped_when_the_db_is_not_materialised(tmp_path):
    task = _task(tmp_path)  # no simulators/ dir on disk -> nothing to run, no false reject
    guard.check_queries(task, _sqlite_change("SELECT anything FROM orders"))


def test_validate_rejects_the_live_wrong_column_regression(tmp_path):
    task = _with_state_db(tmp_path)
    with pytest.raises(changes.ChangeError, match="orders table has columns"):
        changes.validate(task, _sqlite_change(
            "SELECT refunded_amount FROM orders WHERE id='B1614'"))


# ---- intent -> kind --------------------------------------------------------


def test_tool_use_words_read_as_a_trajectory_intent():
    assert guard.intended_kind(
        "the agent should have looked up the order with order_status before answering") \
        == "trajectory"


def test_stored_value_words_read_as_a_state_intent():
    assert guard.intended_kind("the order B1791 should show refunded 0.0 after this") == "state"


def test_vague_words_have_no_definite_intent():
    assert guard.intended_kind("that's just wrong") is None


def test_a_tool_use_ask_drafted_as_sqlite_is_refused_with_a_hint():
    with pytest.raises(ChangeError, match="trajectory_tool_used"):
        guard.check_intent("sqlite_query_equals", "the agent should have used order_status")


def test_a_stored_value_ask_drafted_as_a_tool_check_is_refused(tmp_path):
    with pytest.raises(ChangeError, match="sqlite_query_equals"):
        guard.check_intent("trajectory_tool_used", "the order B1791 should show refunded 0.0")


def test_the_right_kind_for_the_intent_passes():
    guard.check_intent("trajectory_tool_used", "the agent should have used order_status")
    guard.check_intent("sqlite_query_equals", "the order B1791 should show refunded 0.0")


# ---- through the public changes.validate contract --------------------------


def test_validate_relays_a_tool_use_disagreement_drafted_as_sqlite(tmp_path):
    task = _task(tmp_path)
    change = {"op": "add", "file": "tests/correctness/state.py",
              "params": {"fn": "sqlite_query_equals",
                         "args": ["simulators/orders_api/state.db",
                                  "SELECT 1 FROM orders", "order_status was used"]}}
    with pytest.raises(changes.ChangeError, match="trajectory_tool_used"):
        changes.validate(task, change,
                         intent="the agent should have used order_status before answering")


def test_validate_refuses_the_live_placeholder_regression(tmp_path):
    task = _task(tmp_path)
    bad = {"op": "add", "file": "tests/correctness/state.py",
           "params": {"fn": "sqlite_query_equals",
                      "args": ["<db path from a sibling check>", "<full SQL>",
                               "order_status tool was used"]}}
    with pytest.raises(changes.ChangeError):
        changes.validate(task, bad)


def test_validate_accepts_a_real_trajectory_tool_use_check(tmp_path):
    task = _task(tmp_path)
    good = {"op": "add", "file": "tests/correctness/trajectory.py",
            "description": "the agent looked up the order",
            "params": {"fn": "trajectory_tool_used", "args": ["order_status"]}}
    changes.validate(task, good,
                     intent="the agent should have used order_status before answering")
