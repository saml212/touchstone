import json

import pytest

from touchstone import store
from touchstone.interview import rooms
from touchstone.interview.agent import Interviewer
from touchstone.llm import Rule, ScriptedProvider

DRAFT_JSON = json.dumps(
    {
        "say": "Should every reply mention the refund policy?",
        "draft": [
            {
                "kind": "contains",
                "params": {"values": ["refund"], "mode": "any"},
                "name": "mentions refund",
                "severity": "hard",
                "rationale": "stakeholders want the refund policy stated",
            }
        ],
        "commit": [],
    }
)


def _task_room(conn):
    ep = store.insert_episode(
        conn, store.Episode(name="support#1", outcome_label="unresolved", outcome_score=0.2)
    )
    task = store.insert_task(
        conn,
        store.Task(
            name="support#1",
            episode_id=ep.id,
            context={
                "messages": [{"role": "user", "content": "I want my money back"}],
                "tools": [],
            },
            reference={"content": "I cannot help with that.", "tool_calls": []},
        ),
    )
    room = rooms.open(conn, task_id=task.id, topic="refund tone")
    return task, room


def _provider():
    return ScriptedProvider(rules=[
        Rule(substring="refund policy", content=DRAFT_JSON),
        Rule(substring="gibberish", malformed_json=True),
    ])


def _hist(*items):
    return [{"speaker": s, "role": r, "text": t} for s, r, t in items]


def test_open_statement_summarizes_task(conn):
    task, room = _task_room(conn)
    text = Interviewer(_provider(), conn, room).open_statement()
    assert task.name in text
    assert "money back" in text
    assert "unresolved" in text
    assert "?" in text  # asks a concrete question


def test_draft_then_yes_commits_check(conn):
    task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)

    turn = agent.respond(_hist(("sam", "user", "it must mention the refund policy")))
    assert len(turn.draft) == 1
    assert turn.commit == []
    drafts = [c for c in store.list_checks(conn) if not c.enabled]
    assert len(drafts) == 1 and drafts[0].source == "interview"

    history = _hist(
        ("sam", "user", "it must mention the refund policy"),
        ("agent", "assistant", turn.say),
        ("sam", "user", "yes, exactly"),
    )
    turn2 = agent.respond(history)
    assert len(turn2.commit) == 1
    committed = [c for c in store.list_checks(conn, enabled=True) if c.source == "interview"]
    assert len(committed) == 1
    assert committed[0].kind == "contains"
    assert committed[0].id in store.get_task(conn, task.id).check_ids


def test_disagreement_blocks_commit_and_names_both(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)
    first = agent.respond(_hist(("sam", "user", "it must mention the refund policy")))

    history = _hist(
        ("sam", "user", "it must mention the refund policy"),
        ("agent", "assistant", first.say),
        ("alice", "user", "yes agreed"),
        ("bob", "user", "no, that should stay a preference not a hard rule"),
    )
    turn = agent.respond(history)
    assert turn.commit == []
    assert "alice" in turn.say and "bob" in turn.say
    assert not [c for c in store.list_checks(conn, enabled=True) if c.source == "interview"]


def test_malformed_json_falls_back_to_a_question(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)
    turn = agent.respond(_hist(("sam", "user", "some gibberish here")))
    assert turn.commit == []
    assert turn.say.endswith("?")
    assert store.list_checks(conn) == []


def test_slash_commit_commits_the_draft(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)
    first = agent.respond(_hist(("sam", "user", "it must mention the refund policy")))
    history = _hist(
        ("sam", "user", "it must mention the refund policy"),
        ("agent", "assistant", first.say),
        ("sam", "user", "/commit"),
    )
    turn = agent.respond(history)
    assert len(turn.commit) == 1
    assert [c for c in store.list_checks(conn, enabled=True) if c.source == "interview"]


def test_slash_done_closes_the_room(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)
    turn = agent.respond(_hist(("sam", "user", "/done")))
    assert store.get_room(conn, room.id).closed_at is not None
    assert "clos" in turn.say.lower()


def test_commit_intent_with_no_draft_is_a_gentle_nudge(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(_provider(), conn, room)
    turn = agent.respond(_hist(("sam", "user", "/commit")))
    assert turn.commit == []
    assert "draft" in turn.say.lower()


def test_no_provider_degrades_to_a_question(conn):
    _task, room = _task_room(conn)
    agent = Interviewer(None, conn, room)
    turn = agent.respond(_hist(("sam", "user", "tell me more")))
    assert turn.say.endswith("?")
    assert turn.commit == []


@pytest.mark.parametrize("bad", ["", None])
def test_open_statement_without_task(conn, bad):
    room = rooms.open(conn, task_id=bad, topic="general quality")
    text = Interviewer(_provider(), conn, room).open_statement()
    assert "general quality" in text


def test_prompt_prefers_programmatic_kinds_over_judge(conn):
    _task, room = _task_room(conn)
    prompt = Interviewer(_provider(), conn, room)._prompt(_hist(("sam", "user", "hi")))
    system = prompt[0]["content"]
    assert "programmatic" in system.lower()
    assert "judge" in system.lower() and "only" in system.lower()


class _CountingProvider:
    """Wraps a scripted provider and counts how many times the LLM is consulted."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chat(self, messages, tools=None, json=False, timeout=60):
        self.calls += 1
        return self.inner.chat(messages, tools, json, timeout)


DRAFT1 = json.dumps({"say": "Draft?", "draft": [
    {"kind": "contains", "params": {"values": ["cancelled"], "mode": "any"},
     "name": "says cancelled", "severity": "hard"}], "commit": []})
REVISED = json.dumps({"say": "Revised.", "draft": [
    {"kind": "contains", "params": {"values": ["canceled"], "mode": "any"},
     "name": "says canceled (one L)", "severity": "hard"}], "commit": []})


def _amend_provider():
    # The "one L" rule must precede "cancelled": the post-amendment convo contains both, and the
    # first matching rule wins.
    return _CountingProvider(ScriptedProvider(rules=[
        Rule(substring="one L", content=REVISED),
        Rule(substring="cancelled", content=DRAFT1),
    ]))


def test_bare_yes_commits_the_draft_without_consulting_the_llm(conn):
    _task, room = _task_room(conn)
    provider = _amend_provider()
    agent = Interviewer(provider, conn, room)
    first = agent.respond(_hist(("sam", "user", "it must say cancelled")))
    assert len(first.draft) == 1
    calls_after_draft = provider.calls

    history = _hist(
        ("sam", "user", "it must say cancelled"),
        ("agent", "assistant", first.say),
        ("sam", "user", "yes"),
    )
    turn = agent.respond(history)
    assert len(turn.commit) == 1
    assert provider.calls == calls_after_draft  # a bare yes never re-consults the LLM


def test_confirm_with_amendment_revises_through_llm_then_commits_both(conn):
    _task, room = _task_room(conn)
    provider = _amend_provider()
    agent = Interviewer(provider, conn, room)
    first = agent.respond(_hist(("sam", "user", "it must say cancelled")))
    assert len(first.draft) == 1
    calls_after_draft = provider.calls

    history = _hist(
        ("sam", "user", "it must say cancelled"),
        ("agent", "assistant", first.say),
        ("sam", "user", "yes, canceled with one L is fine too, commit both"),
    )
    turn = agent.respond(history)
    assert provider.calls > calls_after_draft  # the amendment was revised through the LLM
    assert len(turn.commit) == 2
    committed = [c for c in store.list_checks(conn, enabled=True) if c.source == "interview"]
    values = {tuple(c.params["values"]) for c in committed}
    assert values == {("cancelled",), ("canceled",)}
