"""The next-step hint walks the v2 work queues, then the Sample/Distill loop, in one order."""

from __future__ import annotations

from touchstone import overview, store, tasks


def _signals(**kw) -> dict:
    base = dict(episodes=0, tasks=0, active=0, needs_checks=0, needs_solution=0,
                runs=0, sampled=False, frontier=0)
    base.update(kw)
    return base


def test_no_episodes_points_at_capture():
    assert "Capture traces" in overview.next_step(_signals())


def test_episodes_but_no_tasks_points_at_mine():
    assert "touchstone mine" in overview.next_step(_signals(episodes=5))


def test_largest_queue_wins_needs_checks_over_needs_solution():
    hint = overview.next_step(_signals(episodes=5, tasks=9, needs_checks=6, needs_solution=2))
    assert hint == "6 tasks need a positive rule — open an interview."


def test_needs_solution_wins_when_it_is_the_larger_pile():
    hint = overview.next_step(_signals(episodes=5, tasks=9, needs_checks=1, needs_solution=4))
    assert hint == "4 tasks need a solution — Ask teacher."


def test_active_tasks_without_a_run_point_at_run():
    hint = overview.next_step(_signals(episodes=5, tasks=3, active=3))
    assert "Run a model" in hint and "3 active tasks" in hint


def test_runs_without_a_sample_point_at_sample():
    hint = overview.next_step(_signals(episodes=5, tasks=3, active=3, runs=1))
    assert "Sample a candidate" in hint


def test_a_nonempty_frontier_points_at_distill():
    hint = overview.next_step(
        _signals(episodes=5, tasks=3, active=3, runs=1, sampled=True, frontier=2))
    assert hint == "2 tasks on the frontier — Distill them into training data."


def test_closed_frontier_is_the_terminal_state():
    hint = overview.next_step(
        _signals(episodes=5, tasks=3, active=3, runs=1, sampled=True, frontier=0))
    assert "you're set" in hint


def test_singular_task_is_not_pluralized():
    assert overview.next_step(_signals(episodes=1, tasks=1, needs_checks=1)).startswith("1 task ")


def test_project_signals_reads_queues_and_loop_state(tmp_path):
    conn = store.connect(str(tmp_path / ".touchstone" / "touchstone.db"))
    store.insert_episode(conn, store.Episode(name="e1", outcome_label="ok"))
    tasks.write_task(tmp_path, tasks.Task(name="a-01", status="active"))
    from touchstone.loop.frontier import write_loop_state
    write_loop_state(tmp_path, "demo", last_sample="2026-09-24T00:00:00Z", frontier_size=3)
    s = overview.project_signals(conn, tmp_path)
    conn.close()
    assert s["episodes"] == 1 and s["tasks"] == 1
    assert s["sampled"] is True and s["frontier"] == 3
