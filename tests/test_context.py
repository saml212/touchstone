import asyncio

import touchstone
from touchstone import store
from touchstone.capture import context


def test_episode_records_spans_under_it(traced):
    conn = context.get_conn()
    with touchstone.episode("job", meta={"agent": "x"}) as ep:
        touchstone.record_llm_call("m", [{"role": "user", "content": "hi"}], "hello")
        ep.outcome(1.0, "resolved")
    spans = store.list_spans(conn, ep.id)
    assert len(spans) == 1 and spans[0].kind == "model"
    assert store.get_episode(conn, ep.id).outcome_label == "resolved"


def test_untracked_episode_for_spans_outside_any_episode(traced):
    conn = context.get_conn()
    touchstone.record_llm_call("m", [{"role": "user", "content": "loose"}], "ok")
    untracked = [e for e in store.list_episodes(conn) if e.source == "untracked"]
    assert len(untracked) == 1 and untracked[0].name.startswith("untracked-")
    # A second loose span reuses the same untracked episode.
    touchstone.record_llm_call("m", [{"role": "user", "content": "loose2"}], "ok")
    assert len(store.list_spans(conn, untracked[0].id)) == 2


def test_nested_episodes_restore_outer(traced):
    conn = context.get_conn()
    with touchstone.episode("outer") as outer:
        with touchstone.episode("inner") as inner:
            touchstone.record_llm_call("m", [{"role": "user", "content": "i"}], "x")
        touchstone.record_llm_call("m", [{"role": "user", "content": "o"}], "y")
    assert len(store.list_spans(conn, inner.id)) == 1
    assert len(store.list_spans(conn, outer.id)) == 1


def test_tool_decorator_sync_records_arguments_and_result(traced):
    conn = context.get_conn()

    @touchstone.tool
    def add(a, b):
        return a + b

    with touchstone.episode("t") as ep:
        assert add(2, 3) == 5
    span = store.list_spans(conn, ep.id)[0]
    assert span.kind == "tool" and span.name == "add"
    assert span.input["arguments"] == {"a": 2, "b": 3}
    assert span.output["result"] == 5


def test_tool_decorator_records_error_and_reraises(traced):
    conn = context.get_conn()

    @touchstone.tool
    def boom():
        raise ValueError("nope")

    with touchstone.episode("t") as ep:
        try:
            boom()
            raise AssertionError("should have raised")
        except ValueError:
            pass
    span = store.list_spans(conn, ep.id)[0]
    assert "ValueError" in span.error and span.output == {}


def test_tool_decorator_async(traced):
    conn = context.get_conn()

    @touchstone.tool
    async def afetch(x):
        await asyncio.sleep(0)
        return {"got": x}

    async def scenario():
        with touchstone.episode("t") as ep:
            assert await afetch("v") == {"got": "v"}
        return ep

    ep = asyncio.run(scenario())
    span = store.list_spans(conn, ep.id)[0]
    assert span.input["arguments"] == {"x": "v"} and span.output["result"] == {"got": "v"}


def test_tool_span_auto_links_to_model_call_by_name(traced):
    conn = context.get_conn()

    @touchstone.tool
    def lookup(q):
        return "answer"

    with touchstone.episode("t") as ep:
        touchstone.record_llm_call(
            "m", [{"role": "user", "content": "go"}],
            {"content": "", "tool_calls": [
                {"id": "call_ab", "name": "lookup", "arguments": '{"q": 1}'}]})
        lookup(1)
    tool_span = next(s for s in store.list_spans(conn, ep.id) if s.kind == "tool")
    assert tool_span.tool_call_id == "call_ab"


def test_tool_span_explicit_tool_call_id_wins_and_is_not_passed_through(traced):
    conn = context.get_conn()
    seen = {}

    @touchstone.tool
    def lookup(q):
        seen["kwargs_had_tool_call_id"] = False
        return q

    with touchstone.episode("t") as ep:
        lookup(7, tool_call_id="explicit_1")
    tool_span = next(s for s in store.list_spans(conn, ep.id) if s.kind == "tool")
    assert tool_span.tool_call_id == "explicit_1"
    assert tool_span.input["arguments"] == {"q": 7}  # tool_call_id not bound to the tool


def test_parallel_same_tool_calls_link_to_distinct_ids(traced):
    conn = context.get_conn()

    @touchstone.tool
    def lookup(q):
        return q

    with touchstone.episode("t") as ep:
        touchstone.record_llm_call(
            "m", [{"role": "user", "content": "go"}],
            {"content": "", "tool_calls": [
                {"id": "a", "name": "lookup", "arguments": '{"q": 1}'},
                {"id": "b", "name": "lookup", "arguments": '{"q": 2}'}]})
        lookup(1)
        lookup(2)
    ids = [s.tool_call_id for s in store.list_spans(conn, ep.id) if s.kind == "tool"]
    assert ids == ["a", "b"]  # each dispatch consumes the next unresolved call
