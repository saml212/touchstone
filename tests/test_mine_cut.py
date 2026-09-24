import pytest

from touchstone import store
from touchstone.mine import build_tasks


@pytest.fixture
def demo_conn(demo_db):
    return demo_db(30)


def test_build_one_task_per_assistant_turn(demo_conn):
    eps = store.list_episodes(demo_conn)
    tasks = build_tasks(demo_conn, eps)
    model_spans = sum(
        1 for ep in eps for s in store.list_spans(demo_conn, ep.id) if s.kind == "model")
    assert len(tasks) == model_spans > len(eps)
    assert all(t.kind == "replay" for t in tasks)
    assert all("messages" in t.context and "tools" in t.context for t in tasks)
    assert all(t.checks == [] for t in tasks)  # checks are attached by materialize, not here


def test_task_names_are_stable_and_unique(demo_conn):
    eps = store.list_episodes(demo_conn)
    names1 = [t.name for t in build_tasks(demo_conn, eps)]
    names2 = [t.name for t in build_tasks(demo_conn, eps)]
    assert names1 == names2  # deterministic
    assert len(names1) == len(set(names1))  # unique


def test_reference_is_the_recorded_assistant_message(demo_conn):
    tasks = build_tasks(demo_conn, store.list_episodes(demo_conn))
    with_tool = next(t for t in tasks if t.reference["tool_calls"])
    assert with_tool.reference["tool_calls"][0].get("name")
    assert "content" in with_tool.reference


def test_failure_tag_and_outcome_label_in_tags(demo_conn):
    tasks = build_tasks(demo_conn, store.list_episodes(demo_conn))
    failures = [t for t in tasks if "failure" in (t.tags or [])]
    assert failures  # demo has escalations/failures
    for t in failures:
        assert t.tags[0] in ("escalated", "failed")


def test_build_skips_episode_without_llm_span(conn):
    ep = store.insert_episode(conn, store.Episode(name="toolonly", outcome_score=1.0))
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="tool", name="x"))
    assert build_tasks(conn, store.list_episodes(conn)) == []
