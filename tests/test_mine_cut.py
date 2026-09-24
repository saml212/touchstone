import pytest

from touchstone import store
from touchstone.mine import cut_tasks


@pytest.fixture
def demo_conn(demo_db):
    return demo_db(30)


def test_cut_one_task_per_assistant_turn(demo_conn):
    eps = store.list_episodes(demo_conn)
    tasks = cut_tasks(demo_conn, eps)
    llm_spans = sum(1 for ep in eps for s in store.list_spans(demo_conn, ep.id) if s.kind == "llm")
    assert len(tasks) == llm_spans > len(eps)
    assert all(t.kind == "replay" for t in tasks)
    assert all("messages" in t.context and "tools" in t.context for t in tasks)


def test_checks_attach_only_where_the_reference_passes(demo_conn):
    called = store.insert_check(
        demo_conn,
        store.Check(
            name="t",
            kind="tool_called",
            params={"name": "order_status"},
            applies_to="tool_calls",
            enabled=1,
        ),
    )
    tasks = cut_tasks(demo_conn, store.list_episodes(demo_conn))
    with_check = [t for t in tasks if called.id in t.check_ids]
    assert with_check
    for t in with_check:
        assert any(tc["name"] == "order_status" for tc in t.reference["tool_calls"])
    without = [t for t in tasks if called.id not in t.check_ids and "failure" not in t.tags]
    assert without and all(
        not any(tc["name"] == "order_status" for tc in t.reference["tool_calls"]) for t in without
    )


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
    assert len(store.list_tasks(demo_conn)) == len(ids_first)
    good = next(t for t in second if "failure" not in (t.tags or []))
    assert check.id in good.check_ids


def test_failure_tasks_exclude_reference_derived_checks(demo_conn):
    # A positive expectation (contains) and a safety check (no_pii), both enabled.
    contains = store.insert_check(
        demo_conn,
        store.Check(name="c", kind="contains", params={"values": ["all sorted"]}, enabled=1),
    )
    no_pii = store.insert_check(
        demo_conn, store.Check(name="p", kind="no_pii", params={}, enabled=1)
    )
    tasks = cut_tasks(demo_conn, store.list_episodes(demo_conn))

    failure = next(t for t in tasks if "failure" in (t.tags or []))
    good = next(t for t in tasks if "failure" not in (t.tags or []) and t.reference["content"])

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
