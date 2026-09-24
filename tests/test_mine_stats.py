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
        episode_id=ep.id, kind="model", name="m",
        input={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        output={"message": {"role": "assistant", "content": "plain text", "tool_calls": []}},
    ))
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="tool", name="lookup"))
    props = mine_stats(conn, store.list_episodes(conn))
    assert all(p.kind not in ("tool_called", "tool_not_called", "contains") for p in props)


def _messages_only_episode(conn, name, *, good, call_refund):
    """An episode whose tool activity lives ONLY in the model spans' messages (no tool spans) —
    the shape of an app that added touchstone.trace() but never decorated its tool functions."""
    score = 1.0 if good else 0.0
    ep = store.insert_episode(conn, store.Episode(
        name=name, outcome_score=score, outcome_label="good" if good else "bad"))
    tool_calls = ([{"id": "c1", "name": "refund", "arguments": '{"order_id": "A1", "amount": 5}'}]
                  if call_refund else [])
    # first turn: the assistant calls refund; the recorded result is a following role:tool message
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="m1",
        input={"messages": [{"role": "user", "content": "refund my order please"}], "tools": []},
        output={"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}))
    history = [{"role": "user", "content": "refund my order please"},
               {"role": "assistant", "content": "", "tool_calls": tool_calls}]
    if call_refund:
        history.append({"role": "tool", "tool_call_id": "c1",
                        "content": '{"refunded": 5, "currency": "USD"}'})
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="m2",
        input={"messages": history, "tools": []},
        output={"message": {"role": "assistant", "content": "Done, refunded.", "tool_calls": []}}))
    return ep


def test_tool_and_state_proposals_from_model_span_messages_without_tool_spans(conn):
    # Two good episodes call refund with an amount; two bad ones never call it. No tool spans exist.
    for i in range(2):
        _messages_only_episode(conn, f"g{i}", good=True, call_refund=True)
    for i in range(2):
        _messages_only_episode(conn, f"b{i}", good=False, call_refund=False)
    assert not [s for e in store.list_episodes(conn)
                for s in store.list_spans(conn, e.id) if s.kind == "tool"]

    props = mine_stats(conn, store.list_episodes(conn))
    by = _by_kind(props)
    # rank 2: tool_called from the assistant tool_calls in the messages
    assert "refund" in {p.params["name"] for p in by.get("tool_called", [])}
    # rank 3: a state assertion that refund is always called with the recorded fields
    state = [p for p in by.get("expr", []) if "refund" in p.name]
    assert state and state[0].confidence == 0.9
    assert all("amount" in p.name or "order_id" in p.name for p in state)
    # confidence is stamped on the programmatic tool proposal
    assert by["tool_called"][0].confidence is not None


def test_json_schema_proposed_when_good_outputs_are_json(conn):
    for i in range(4):
        ep = store.insert_episode(
            conn, store.Episode(name=f"e{i}", outcome_score=1.0, outcome_label="good")
        )
        store.insert_span(conn, store.Span(
            episode_id=ep.id, kind="model", name="m",
            input={"messages": [], "tools": []},
            output={"message": {"role": "assistant",
                                "content": '{"status": "ok", "id": 1}', "tool_calls": []}},
        ))
    props = mine_stats(conn, store.list_episodes(conn))
    schema = next(p for p in props if p.kind == "json_schema")
    assert set(schema.params["schema"]["required"]) == {"status", "id"}
