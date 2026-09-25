"""changes.py round-trips the tests/ files rewardkit writes, and refuses hand-edited ones."""

from __future__ import annotations

import tomllib

import pytest

from touchstone.harbor import rewardkit
from touchstone.review import changes


def _task(tmp_path):
    d = tmp_path / "tasks" / "refund"
    (d / "tests" / "correctness").mkdir(parents=True)
    rewardkit.write_criteria(d / "tests" / "correctness", "state", [
        "rk.sqlite_query_equals('simulators/orders_service/state.db', \"SELECT refunded FROM orders"
        " WHERE id='B6991'\", 183.18)",
        "rk.sqlite_query_equals('simulators/orders_service/state.db', \"SELECT COUNT(*) FROM emails"
        "\", 1)",
    ])
    rewardkit.write_reward_toml(d / "tests", ["correctness", "safety"])
    return d


def _lines(path):
    return [ln for ln in path.read_text().splitlines() if ln.startswith("rk.")]


def test_edit_a_criteria_line_roundtrips(tmp_path):
    task = _task(tmp_path)
    query = "SELECT refunded FROM orders WHERE id='B6991'"
    touched = changes.apply(task, {"op": "edit", "file": "tests/correctness/state.py",
                                   "criterion": 1,
                                   "params": {"fn": "sqlite_query_equals",
                                              "args": ["simulators/orders_service/state.db",
                                                       query, 200.0]}})
    assert touched == ["tests/correctness/state.py"]
    lines = _lines(task / "tests" / "correctness" / "state.py")
    assert "200.0" in lines[0] and len(lines) == 2
    # re-parseable after the edit (true round trip)
    changes.apply(task, {"op": "remove", "file": "tests/correctness/state.py", "criterion": 2})
    assert len(_lines(task / "tests" / "correctness" / "state.py")) == 1


def test_add_and_remove_a_criteria_line(tmp_path):
    task = _task(tmp_path)
    changes.apply(task, {"op": "add", "file": "tests/correctness/state.py",
                         "params": {"fn": "trajectory_tool_not_used", "args": ["escalate"]}})
    lines = _lines(task / "tests" / "correctness" / "state.py")
    assert lines[-1] == "rk.trajectory_tool_not_used('escalate')"


def test_reward_weight_edit(tmp_path):
    task = _task(tmp_path)
    changes.apply(task, {"op": "edit", "file": "tests/reward.toml",
                         "criterion": "safety", "weight": 2})
    doc = tomllib.loads((task / "tests" / "reward.toml").read_text())
    assert doc["reward"][0]["weights"]["safety"] == 2.0


def test_text_edit_rewrites_instruction(tmp_path):
    task = _task(tmp_path)
    (task / "instruction.md").write_text("old\n")
    changes.apply(task, {"op": "text", "file": "instruction.md", "text": "  Refund order B6991.  "})
    assert (task / "instruction.md").read_text() == "Refund order B6991.\n"


def test_judge_criterion_add_edit_remove(tmp_path):
    task = _task(tmp_path)
    judge = task / "tests" / "quality" / "judge.toml"
    judge.parent.mkdir(parents=True)
    rewardkit.write_judge(judge, model="openai/gpt-4o-mini", files=["output.json"],
                          criteria=[{"description": "answer is polite", "type": "binary"}])
    changes.apply(task, {"op": "edit", "file": "tests/quality/judge.toml",
                         "criterion": "answer is polite",
                         "description": "answer is warm and clear"})
    doc = tomllib.loads(judge.read_text())
    assert doc["criterion"][0]["description"] == "answer is warm and clear"
    changes.apply(task, {"op": "add", "file": "tests/quality/judge.toml",
                         "description": "answer cites the order number"})
    assert len(tomllib.loads(judge.read_text())["criterion"]) == 2


def test_refuses_hand_edited_criteria_file(tmp_path):
    task = _task(tmp_path)
    (task / "tests" / "correctness" / "state.py").write_text(
        "import rewardkit as rk\nx = 1\nrk.sqlite_query_equals('a', 'b', 1)\n")
    with pytest.raises(changes.ChangeError, match="hand-edited"):
        changes.apply(task, {"op": "remove", "file": "tests/correctness/state.py", "criterion": 1})


def test_refuses_bad_index_and_missing_dimension(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(changes.ChangeError, match="no criterion 9"):
        changes.apply(task, {"op": "remove", "file": "tests/correctness/state.py", "criterion": 9})
    with pytest.raises(changes.ChangeError, match="not a dimension"):
        changes.apply(task, {"op": "edit", "file": "tests/reward.toml",
                             "criterion": "speed", "weight": 1})


def test_refuses_escape_outside_task(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(changes.ChangeError, match="inside the task"):
        changes.apply(task, {"op": "text", "file": "../../evil.md", "text": "x"})


def test_describe_reads_back_in_plain_words(tmp_path):
    text = changes.describe([
        {"op": "edit", "file": "tests/reward.toml", "criterion": "safety", "weight": 2},
        {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
         "params": {"fn": "sqlite_query_equals", "args": ["db", "q", 200.0]}}])
    assert "safety weight to 2" in text
    assert "change check 1" in text


def test_descriptions_stay_in_sync_on_add_edit_remove(tmp_path):
    from touchstone.survey import descriptions
    task = _task(tmp_path)
    tests = task / "tests"
    # seed two descriptions matching the two seeded state criteria
    descriptions.write(tests, {"tests/correctness/state.py:1": "the refund is 183.18",
                               "tests/correctness/state.py:2": "one email was sent"})
    # add a third check with its own description
    changes.apply(task, {"op": "add", "file": "tests/correctness/state.py",
                         "description": "a tracking ticket was opened",
                         "params": {"fn": "sqlite_query_equals",
                                    "args": ["s/state.db", "SELECT COUNT(*) FROM tickets", 1]}})
    d = tomllib.loads((tests / "descriptions.toml").read_text())
    assert d["tests/correctness/state.py:3"] == "a tracking ticket was opened"
    # remove #1 -> #2 and #3 shift down to #1 and #2
    changes.apply(task, {"op": "remove", "file": "tests/correctness/state.py", "criterion": 1})
    d = tomllib.loads((tests / "descriptions.toml").read_text())
    assert d["tests/correctness/state.py:1"] == "one email was sent"
    assert d["tests/correctness/state.py:2"] == "a tracking ticket was opened"
    assert "tests/correctness/state.py:3" not in d
    # edit #2 without a description -> derived from the call, path-free
    changes.apply(task, {"op": "edit", "file": "tests/correctness/state.py", "criterion": 2,
                         "params": {"fn": "trajectory_tool_used", "args": ["refund"]}})
    d = tomllib.loads((tests / "descriptions.toml").read_text())
    assert d["tests/correctness/state.py:2"] == "the agent used refund"


def test_apply_list_applies_both(tmp_path):
    task = _task(tmp_path)
    touched = changes.apply(task, [
        {"op": "edit", "file": "tests/reward.toml", "criterion": "safety", "weight": 3},
        {"op": "remove", "file": "tests/correctness/state.py", "criterion": 2}])
    assert set(touched) == {"tests/reward.toml", "tests/correctness/state.py"}


def test_describe_uses_plain_words_never_raw_code():
    from touchstone.review import changes

    edit = {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
            "params": {"fn": "sqlite_query_equals",
                       "args": ["db", "SELECT refunded FROM orders WHERE id='B1'", 183.18]}}
    assert "rk." not in changes.describe(edit)
    assert "returns 183.18" in changes.describe(edit)
    assert changes.describe({**edit, "description": "order B1 shows $183.18 refunded"}) == \
        "change check 1 to order B1 shows $183.18 refunded"


def test_expected_shortcut_keeps_the_query_and_coerces_numbers(tmp_path):
    from touchstone.harbor import rewardkit
    from touchstone.review import changes

    tests = tmp_path / "tests" / "correctness"
    tests.mkdir(parents=True)
    sql = "SELECT refunded FROM orders WHERE id='B1'"
    line = f"rk.sqlite_query_equals('s/state.db', {sql!r}, 100.0)"
    rewardkit.write_criteria(tests, "state", [line])
    changes.apply(tmp_path, {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
                             "params": {"expected": "183.18"}})
    line = (tests / "state.py").read_text().splitlines()[-1]
    assert line.endswith("WHERE id='B1'\", 183.18)") and "SELECT refunded FROM orders" in line


def test_criterion_names_are_normalised_or_refused(tmp_path):
    import pytest

    from touchstone.review import changes

    assert changes._render_call({"fn": "rk.file_exists", "args": ["x"]}) == "rk.file_exists('x')"
    with pytest.raises(changes.ChangeError):
        changes._render_call({"fn": "made_up", "args": []})


def test_read_back_uses_a_description_inside_params_and_words_an_expected_edit():
    from touchstone.review import changes

    edit = {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
            "params": {"expected": 183.18, "description": "order B6991 refunded 183.18"}}
    assert changes.describe(edit) == "change check 1 to order B6991 refunded 183.18"
    del edit["params"]["description"]
    assert changes.describe(edit) == "change check 1 to now expects 183.18"
