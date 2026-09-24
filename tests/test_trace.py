import touchstone
from touchstone.capture import context


def test_trace_twice_is_idempotent(db):
    context._untracked_ids.clear()
    a = touchstone.trace(db)
    b = touchstone.trace(db)
    assert a["db"] == b["db"] == db


def test_trace_never_raises_when_no_sdk_installed(db):
    # openai/anthropic are not project dependencies, so neither should be importable here.
    info = touchstone.trace(db)
    assert info["patched"] == []


def test_record_llm_call_without_episode_uses_untracked(db):
    context._untracked_ids.clear()
    context._local.__dict__.pop("conn", None)
    touchstone.trace(db)
    touchstone.record_llm_call("m", [{"role": "user", "content": "x"}], "y")
    from touchstone import store

    conn = context.get_conn()
    eps = store.list_episodes(conn)
    assert any(e.source == "untracked" for e in eps)
