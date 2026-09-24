import json

from touchstone import store
from touchstone.checks import Check
from touchstone.llm import Rule, ScriptedProvider
from touchstone.mine import Proposal, dedupe, mine_llm

_ARRAY = [
    {"kind": "contains", "params": {"values": ["thanks"]}, "severity": "soft",
     "applies_to": "final", "rationale": "polite closing", "support": {"episode_ids": ["e1"]}},
    {"kind": "contains", "params": {}},  # invalid: missing values
    {"kind": "tool_called", "params": {"name": "refund"}, "applies_to": "tool_calls",
     "rationale": "refund path expected"},
]


def _provider(content: str) -> ScriptedProvider:
    return ScriptedProvider(rules=[Rule(substring="Propose checks", content=content)])


def _episode(conn):
    ep = store.insert_episode(
        conn, store.Episode(name="e", outcome_label="good", outcome_score=1.0)
    )
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="llm", name="m",
        input={"messages": [{"role": "user", "content": "help"}], "tools": []},
        output={"message": {"role": "assistant", "content": "sure", "tool_calls": []}},
    ))
    return store.list_episodes(conn)


def test_mine_llm_keeps_valid_drops_invalid(conn):
    eps = _episode(conn)
    provider = _provider(json.dumps(_ARRAY))
    props = mine_llm(conn, provider, eps, [])
    kinds = sorted(p.kind for p in props)
    assert kinds == ["contains", "tool_called"]  # invalid contains dropped
    assert all(p.origin == "llm" for p in props)
    assert props[0].support == {"episode_ids": ["e1"]}


def test_mine_llm_none_provider_returns_empty(conn):
    assert mine_llm(conn, None, _episode(conn), []) == []


def test_mine_llm_malformed_reply_is_empty_no_raise(conn):
    eps = _episode(conn)
    provider = ScriptedProvider(rules=[Rule(substring="Propose checks", malformed_json=True)])
    assert mine_llm(conn, provider, eps, []) == []


def test_mine_llm_non_array_reply_is_empty(conn):
    eps = _episode(conn)
    props = mine_llm(conn, _provider('{"kind": "contains"}'), eps, [])
    assert props == []


def test_mine_llm_provider_exception_is_swallowed(conn):
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("network down")

    assert mine_llm(conn, Boom(), _episode(conn), []) == []


def test_mine_llm_caps_at_25(conn):
    big = [{"kind": "contains", "params": {"values": [f"w{i}"]}} for i in range(40)]
    props = mine_llm(conn, _provider(json.dumps(big)), _episode(conn), [])
    assert len(props) == 25


def test_dedupe_drops_matches_against_existing_checks():
    existing = [Check(name="r", kind="tool_called", params={"name": "refund"})]
    proposals = [
        Proposal(kind="contains", params={"values": ["thanks"]}),
        Proposal(kind="tool_called", params={"name": "refund"}),  # identical to existing
    ]
    kept = dedupe(existing, proposals)
    assert [p.kind for p in kept] == ["contains"]


def test_dedupe_collapses_duplicates_within_batch():
    proposals = [
        Proposal(kind="max_length", params={"max": 100}),
        Proposal(kind="max_length", params={"max": 100}),
    ]
    assert len(dedupe([], proposals)) == 1


def test_dedupe_param_key_order_insensitive():
    existing = [Check(name="j", kind="json_schema", params={"a": 1, "b": 2})]
    proposals = [Proposal(kind="json_schema", params={"b": 2, "a": 1})]
    assert dedupe(existing, proposals) == []
