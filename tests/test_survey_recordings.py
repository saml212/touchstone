from touchstone import store
from touchstone.survey.recordings import recorded_tool_names, tool_events


def _model_span(episode_id, input_messages, tool_calls, content=""):
    return store.Span(
        episode_id=episode_id, kind="model", name="gpt-4o-mini",
        input={"messages": input_messages, "tools": [], "params": {}},
        output={"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}},
    )


def test_tool_events_pairs_calls_with_results(conn):
    ep = store.insert_episode(conn, store.Episode(name="ep1"))
    # span 1: model asks for order_status
    store.insert_span(conn, _model_span(
        ep.id,
        [{"role": "user", "content": "where is B1?"}],
        [{"id": "c1", "name": "order_status", "arguments": '{"order_id": "B1"}'}],
    ))
    # span 2: result of c1 is in the input; model then refunds
    store.insert_span(conn, _model_span(
        ep.id,
        [{"role": "user", "content": "where is B1?"},
         {"role": "tool", "tool_call_id": "c1", "content": '{"id": "B1", "total": 10.0}'}],
        [{"id": "c2", "name": "refund", "arguments": '{"order_id": "B1", "amount": 10.0}'}],
    ))
    # span 3: result of c2 present, no more calls
    store.insert_span(conn, _model_span(
        ep.id,
        [{"role": "tool", "tool_call_id": "c1", "content": '{"id": "B1", "total": 10.0}'},
         {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true, "refunded": 10.0}'}],
        [], content="done",
    ))

    events = tool_events(conn)
    assert [e.tool for e in events] == ["order_status", "refund"]
    assert events[0].arguments == {"order_id": "B1"}
    assert events[0].output == {"id": "B1", "total": 10.0}
    assert events[1].output == {"ok": True, "refunded": 10.0}
    assert events[0].episode == ep.id
    assert recorded_tool_names(events) == {"order_status", "refund"}


def test_non_json_arguments_kept_raw(conn):
    ep = store.insert_episode(conn, store.Episode(name="ep2"))
    store.insert_span(conn, _model_span(
        ep.id, [{"role": "user", "content": "hi"}],
        [{"id": "c1", "name": "echo", "arguments": "just a string"}],
    ))
    events = tool_events(conn)
    assert events[0].arguments == "just a string"
    assert events[0].output is None  # no result recorded


def test_id_less_tool_result_pairs_by_name(conn):
    # The deprecated OpenAI function-calling API returns the result as role="function" with a name
    # but no id, and does not re-send the assistant call in the follow-up messages. After capture
    # canonicalizes it (function -> tool), the result is an id-less tool message: it must still pair
    # to its call by tool name, not be dropped (which left the tool's output null before this fix).
    ep = store.insert_episode(conn, store.Episode(name="ep-fn"))
    store.insert_span(conn, _model_span(
        ep.id, [{"role": "user", "content": "weather in Paris?"}],
        [{"id": "c1", "name": "get_today_weather", "arguments": '{"location": "Paris"}'}],
    ))
    store.insert_span(conn, _model_span(
        ep.id,
        [{"role": "user", "content": "weather in Paris?"},
         {"role": "tool", "name": "get_today_weather", "content": '{"temperature": 12}'}],
        [], content="It is 12C.",
    ))
    events = tool_events(conn)
    assert [e.tool for e in events] == ["get_today_weather"]
    assert events[0].output == {"temperature": 12}
