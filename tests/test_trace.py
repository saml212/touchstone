import touchstone
from touchstone.capture import context


def test_trace_twice_is_idempotent(db):
    context._untracked_ids.clear()
    a = touchstone.trace(db)
    b = touchstone.trace(db)
    assert a["db"] == b["db"] == db


def test_trace_never_raises_for_absent_sdk(db):
    # openai is a dev dependency (real-SDK smoke), anthropic is not installed: trace must not
    # raise for the absent one and returns whatever it could patch.
    info = touchstone.trace(db)
    assert isinstance(info["patched"], list)
    assert "anthropic" not in info["patched"]


def test_record_llm_call_without_episode_uses_untracked(db):
    context._untracked_ids.clear()
    context._local.__dict__.pop("conn", None)
    touchstone.trace(db)
    touchstone.record_llm_call("m", [{"role": "user", "content": "x"}], "y")
    from touchstone import store

    conn = context.get_conn()
    eps = store.list_episodes(conn)
    assert any(e.source == "untracked" for e in eps)
