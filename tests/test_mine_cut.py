import pytest

from touchstone import store
from touchstone.demo import run_demo
from touchstone.mine import cut_tasks


@pytest.fixture
def demo_conn(traced):
    run_demo(n=30)
    conn = store.connect(traced)
    yield conn
    conn.close()


def test_final_cut_one_task_per_episode(demo_conn):
    eps = store.list_episodes(demo_conn)
    tasks = cut_tasks(demo_conn, eps)
    assert len(tasks) == len(eps)
    assert all(t.kind == "replay" for t in tasks)
    assert all("messages" in t.context and "tools" in t.context for t in tasks)


def test_every_turn_produces_more_tasks(demo_conn):
    eps = store.list_episodes(demo_conn)
    final = cut_tasks(demo_conn, eps)
    every = cut_tasks(demo_conn, eps, every_turn=True)
    assert len(every) > len(final)


def test_cut_is_idempotent_and_refreshes_check_ids(demo_conn):
    eps = store.list_episodes(demo_conn)
    first = cut_tasks(demo_conn, eps)
    ids_first = sorted(t.id for t in first)

    # Enable a check, re-cut: same task rows, now carrying the check.
    check = store.insert_check(
        demo_conn, store.Check(name="c", kind="no_pii", params={}, enabled=1)
    )
    second = cut_tasks(demo_conn, eps)
    assert sorted(t.id for t in second) == ids_first  # no duplicate tasks
    assert len(store.list_tasks(demo_conn)) == len(eps)
    good = next(t for t in second if "failure" not in (t.tags or []))
    assert check.id in good.check_ids


def test_failure_tasks_exclude_reference_derived_checks(demo_conn):
    # A positive expectation (contains) and a safety check (no_pii), both enabled.
    contains = store.insert_check(
        demo_conn, store.Check(name="c", kind="contains", params={"values": ["ok"]}, enabled=1)
    )
    no_pii = store.insert_check(
        demo_conn, store.Check(name="p", kind="no_pii", params={}, enabled=1)
    )
    tasks = cut_tasks(demo_conn, store.list_episodes(demo_conn))

    failure = next(t for t in tasks if "failure" in (t.tags or []))
    good = next(t for t in tasks if "failure" not in (t.tags or []))

    assert contains.id not in failure.check_ids
    assert no_pii.id in failure.check_ids
    assert contains.id in good.check_ids and no_pii.id in good.check_ids

    # Reference is still recorded on failure tasks; only checks differ.
    assert failure.reference is not None


def test_failure_tag_and_outcome_label_in_tags(demo_conn):
    tasks = cut_tasks(demo_conn, store.list_episodes(demo_conn))
    failures = [t for t in tasks if "failure" in (t.tags or [])]
    assert failures  # demo has escalations/failures
    for t in failures:
        assert t.tags[0] in ("escalated", "failed")


def test_cut_skips_episode_without_llm_span(conn):
    ep = store.insert_episode(conn, store.Episode(name="toolonly", outcome_score=1.0))
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="tool", name="x"))
    assert cut_tasks(conn, store.list_episodes(conn)) == []
