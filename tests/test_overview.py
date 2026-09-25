"""The next-step hint: capture first, then benchmark."""

from __future__ import annotations

from touchstone import overview, store


def test_no_episodes_points_at_capture():
    assert "Capture traces" in overview.next_step({"episodes": 0, "rooms": 0})


def test_episodes_point_at_bench():
    assert "touchstone bench" in overview.next_step({"episodes": 5, "rooms": 0})


def test_project_signals_counts_episodes_and_rooms(tmp_path):
    conn = store.connect(str(tmp_path / ".touchstone" / "touchstone.db"))
    store.insert_episode(conn, store.Episode(name="e1", outcome_label="ok"))
    store.insert_room(conn, store.Room(task_id=None, topic="review"))
    s = overview.project_signals(conn)
    conn.close()
    assert s == {"episodes": 1, "rooms": 1}
