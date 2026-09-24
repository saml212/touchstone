"""Difficulty tracking, the failure frontier, and the loop's stopping criteria."""

from pathlib import Path

from touchstone import store, tasks
from touchstone.checks import Check
from touchstone.loop import (
    check_stop,
    frontier,
    frontier_split,
    mean_pass_rate,
    record_run,
    write_loop_state,
)
from touchstone.loop.frontier import read_loop_state


def _active_task(root, name):
    tasks.write_task(root, tasks.Task(
        name=name,
        context={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        reference={"content": "", "tool_calls": [{"name": "refund", "arguments": "{}"}]},
        checks=[Check(kind="tool_called", params={"name": "refund"}, name="calls refund",
                      source="policy")]))


def _difficulty(conn, model, rows):
    for task, outcomes in rows.items():
        for passed in outcomes:
            store.upsert_difficulty(conn, task, model, passed=passed)


def test_frontier_splits_learnability_and_only_incumbent(tmp_path, conn):
    root = str(tmp_path)
    for n in ("t-learn", "t-fail", "t-learned"):
        _active_task(root, n)
    _difficulty(conn, "m", {"t-learn": [True, False], "t-fail": [False], "t-learned": [True]})

    assert frontier(conn, root, "m") == ["t-fail", "t-learn"]
    split = frontier_split(conn, root, "m")
    assert split["learnability"] == ["t-learn"]
    assert split["only_incumbent"] == ["t-fail"]
    assert abs(mean_pass_rate(conn, root, "m") - 0.5) < 1e-9  # (0.5 + 0 + 1) / 3


def test_record_run_writes_difficulty_and_file_cache(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t1")
    run = store.insert_run(conn, store.Run(target="b", model_spec="m"))
    store.insert_result(conn, store.Result(run_id=run.id, task="t1", passed=1, reward=1.0))
    store.finish_run(conn, run.id)

    record_run(conn, root, store.get_run(conn, run.id))
    assert store.get_difficulty(conn, "t1", "m").pass_rate == 1.0
    assert tasks.read_task(Path(root) / "tasks" / "t1").difficulty["m"] == 1.0


def test_stop_when_frontier_empty(tmp_path, conn):
    stop, reason = check_stop(conn, str(tmp_path), "never-sampled")
    assert stop and "empty" in reason


def test_stop_on_teacher_failure_cost_and_delta(tmp_path, conn):
    root = str(tmp_path)
    _active_task(root, "t-learn")
    _difficulty(conn, "m", {"t-learn": [True, False]})  # a non-empty frontier

    assert check_stop(conn, root, "m", teacher_all_failed=True)[0]
    assert check_stop(conn, root, "m", round_cost=1.0, cost_cap=0.5)[0]
    # mean pass rate is 0.5; a prev of 0.5 means zero improvement -> stop
    stop, reason = check_stop(conn, root, "m", prev_pass_rate=0.5, delta_threshold=0.01)
    assert stop and "delta" in reason
    # a big improvement since prev keeps the loop running
    assert not check_stop(conn, root, "m", prev_pass_rate=0.0, delta_threshold=0.01)[0]


def test_loop_state_roundtrip(tmp_path):
    root = str(tmp_path)
    write_loop_state(root, "bench", last_sample="t0", frontier_size=3)
    write_loop_state(root, "bench", last_distill="t1")
    state = read_loop_state(root, "bench")
    assert state == {"last_sample": "t0", "frontier_size": 3, "last_distill": "t1"}
