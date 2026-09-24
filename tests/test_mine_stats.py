import pytest

from touchstone import store
from touchstone.mine import mine_stats


@pytest.fixture
def demo_conn(demo_db):
    return demo_db(30)


def _by_kind(proposals):
    out = {}
    for p in proposals:
        out.setdefault(p.kind, []).append(p)
    return out


def test_stats_on_demo_are_sensible(demo_conn):
    props = mine_stats(demo_conn, store.list_episodes(demo_conn))
    by = _by_kind(props)

    # The demo always leaks an email in some resolved outputs.
    assert "no_pii" in by
    assert by["no_pii"][0].severity == "hard"
    assert by["no_pii"][0].support_count >= 1

    # order_status is called in every resolved episode, never in escalations.
    called = {p.params["name"] for p in by.get("tool_called", [])}
    assert "order_status" in called

    # escalate is the failure signature -> propose not calling it.
    not_called = {p.params["name"] for p in by.get("tool_not_called", [])}
    assert "escalate" in not_called

    # Length bound is a soft guardrail near the good-output size, not absurd.
    max_len = by["max_length"][0]
    assert max_len.severity == "soft"
    assert 30 <= max_len.params["max"] <= 300

    # Recurring good phrasing surfaces as a soft contains check.
    assert "contains" in by
    assert all(p.severity == "soft" for p in by["contains"])


def test_stats_are_deterministic(demo_conn):
    a = mine_stats(demo_conn, store.list_episodes(demo_conn))
    b = mine_stats(demo_conn, store.list_episodes(demo_conn))
    assert [(p.kind, p.params) for p in a] == [(p.kind, p.params) for p in b]


def test_stats_empty_db(conn):
    assert mine_stats(conn, []) == []


def test_unlabeled_episodes_produce_no_tool_or_phrase_stats(conn):
    # Episodes with no outcome are unlabeled: no good/bad split, so no correlation checks.
    ep = store.insert_episode(conn, store.Episode(name="e"))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="llm", name="m",
        input={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        output={"message": {"role": "assistant", "content": "plain text", "tool_calls": []}},
    ))
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="tool", name="lookup"))
    props = mine_stats(conn, store.list_episodes(conn))
    assert all(p.kind not in ("tool_called", "tool_not_called", "contains") for p in props)


def test_json_schema_proposed_when_good_outputs_are_json(conn):
    for i in range(4):
        ep = store.insert_episode(
            conn, store.Episode(name=f"e{i}", outcome_score=1.0, outcome_label="good")
        )
        store.insert_span(conn, store.Span(
            episode_id=ep.id, kind="llm", name="m",
            input={"messages": [], "tools": []},
            output={"message": {"role": "assistant",
                                "content": '{"status": "ok", "id": 1}', "tool_calls": []}},
        ))
    props = mine_stats(conn, store.list_episodes(conn))
    schema = next(p for p in props if p.kind == "json_schema")
    assert set(schema.params["schema"]["required"]) == {"status", "id"}
