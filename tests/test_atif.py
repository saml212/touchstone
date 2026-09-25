import json

import touchstone
from touchstone.capture import context
from touchstone.harbor import atif


def _build_episode():
    with touchstone.episode("support-A1", meta={"agent": "demo"}) as ep:
        messages = [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": "Where is order A1?"},
        ]
        touchstone.record_llm_call(
            "scripted", messages,
            {"content": "", "tool_calls": [
                {"id": "call_1", "name": "order_status", "arguments": '{"order_id":"A1"}'}],
             "usage": {"tokens_in": 5, "tokens_out": 2}},
        )

        @touchstone.tool
        def order_status(order_id):
            return {"status": "shipped"}

        order_status("A1")
        touchstone.record_llm_call("scripted", messages, "It shipped, arriving in 2 days.")
    return ep


def test_atif_round_trip_and_structure(traced):
    ep = _build_episode()
    conn = context.get_conn()
    traj = atif.to_atif(conn, ep.id)

    reloaded = json.loads(json.dumps(traj))
    atif.validate(reloaded)

    assert reloaded["schema_version"] == "ATIF-v1.8"
    sources = [s["source"] for s in reloaded["steps"]]
    assert sources == ["system", "user", "agent", "agent"]

    tool_step = reloaded["steps"][2]
    assert tool_step["tool_calls"][0]["function_name"] == "order_status"
    assert tool_step["tool_calls"][0]["arguments"] == {"order_id": "A1"}
    obs = tool_step["observation"]["results"][0]
    assert obs["source_call_id"] == "call_1"
    assert "shipped" in obs["content"]
    assert reloaded["final_metrics"]["total_prompt_tokens"] == 5


def _projection(conn, episode_id):
    """The parts round-tripping must preserve: assistant outputs, tool results, and token usage."""
    spans = touchstone.store.list_spans(conn, episode_id)
    models = [s for s in spans if s.kind == "model"]
    assistants = [(m.output["message"].get("content", ""),
                   [(tc.get("id"), tc.get("name"))
                    for tc in m.output["message"].get("tool_calls") or []])
                  for m in models]
    tools = [s.tool_call_id for s in spans if s.kind == "tool"]  # ATIF stringifies results
    usage = (sum(m.tokens_in or 0 for m in models), sum(m.tokens_out or 0 for m in models))
    return assistants, tools, usage


def test_import_trajectory_round_trips_messages_tool_ids_and_usage(traced):
    ep = _build_episode()
    conn = context.get_conn()
    traj = atif.to_atif(conn, ep.id)

    ep2 = atif.import_trajectory(conn, traj)
    assert _projection(conn, ep2.id) == _projection(conn, ep.id)
    # re-exporting the imported episode yields the same steps and metrics
    traj2 = atif.to_atif(conn, ep2.id)
    assert traj2["steps"] == traj["steps"]
    assert traj2["final_metrics"] == traj["final_metrics"]


def test_atif_empty_episode_has_fallback_step(traced):
    conn = context.get_conn()
    with touchstone.episode("empty") as ep:
        pass
    traj = atif.to_atif(conn, ep.id)
    atif.validate(traj)
    assert len(traj["steps"]) == 1


def test_atif_interleaves_mid_conversation_user_turns(conn):
    from touchstone import store
    ep = store.insert_episode(conn, store.Episode(id="mt", name="mt"))
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="user", name="user",
                                       input={"content": "book a flight"}, output={}))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="gpt",
        input={"messages": [{"role": "system", "content": "assistant"},
                            {"role": "user", "content": "book a flight"}]},
        output={"message": {"content": "Sure, who is flying?", "tool_calls": []}}))
    # a second user turn injected mid-conversation, then the agent acts
    store.insert_span(conn, store.Span(episode_id=ep.id, kind="user", name="user",
                                       input={"content": "Ben Cole, aisle seat"}, output={}))
    store.insert_span(conn, store.Span(
        episode_id=ep.id, kind="model", name="gpt",
        input={"messages": [{"role": "user", "content": "Ben Cole, aisle seat"}]},
        output={"message": {"content": "Booked.", "tool_calls": []}}))
    traj = atif.to_atif(conn, ep.id)
    atif.validate(traj)
    sources = [s["source"] for s in traj["steps"]]
    assert sources == ["system", "user", "agent", "user", "agent"]
    assert traj["steps"][3]["message"] == "Ben Cole, aisle seat"


def test_atif_matches_harbor_models_when_available(traced):
    harbor = __import__("importlib").util.find_spec("harbor")
    if harbor is None:
        return  # harbor not installed in this env; structural validate() already covered it
    from harbor.models.trajectories.trajectory import Trajectory

    ep = _build_episode()
    conn = context.get_conn()
    Trajectory.model_validate(atif.to_atif(conn, ep.id))


def _episode_with_reasoning_and_parts():
    with touchstone.episode("vision", meta={"agent": "demo"}) as ep:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "http://x/pic.png"}}]},
        ]
        touchstone.record_llm_call(
            "gpt-4o-mini", messages,
            {"content": "a cat", "usage": {"tokens_in": 10, "tokens_out": 3}})
    conn = context.get_conn()
    span = [s for s in context.get_conn().execute("SELECT id FROM spans WHERE kind='model'")][0][0]
    out = touchstone.store.get_span(conn, span).output
    out["message"]["reasoning"] = [{"type": "thinking", "thinking": "furry"}]
    out["stop_reason"] = "stop"
    out["usage"] = {"cached_tokens": 4}
    touchstone.store.update_span(conn, span, output=out, cost_usd=0.0002)
    return ep


def test_atif_reasoning_stop_reason_metrics_and_parts(traced):
    ep = _episode_with_reasoning_and_parts()
    traj = atif.to_atif(context.get_conn(), ep.id)
    atif.validate(traj)

    user_step = traj["steps"][0]
    assert user_step["message"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image", "source": {"media_type": "image/png", "path": "http://x/pic.png"}}]

    agent_step = traj["steps"][1]
    assert agent_step["reasoning_content"] == "furry"
    assert agent_step["extra"]["stop_reason"] == "stop"
    assert agent_step["metrics"]["cached_tokens"] == 4
    assert agent_step["metrics"]["cost_usd"] == 0.0002
    assert traj["final_metrics"]["total_cached_tokens"] == 4
    assert traj["final_metrics"]["total_cost_usd"] == 0.0002


def test_atif_parallel_tool_results_link_by_id(traced):
    with touchstone.episode("parallel") as ep:
        touchstone.record_llm_call(
            "m", [{"role": "user", "content": "go"}],
            {"content": "", "tool_calls": [
                {"id": "a", "name": "lookup", "arguments": '{"q": 1}'},
                {"id": "b", "name": "lookup", "arguments": '{"q": 2}'}]})

        @touchstone.tool
        def lookup(q):
            return f"result-{q}"

        lookup(1)
        lookup(2)
    traj = atif.to_atif(context.get_conn(), ep.id)
    atif.validate(traj)
    agent_step = next(s for s in traj["steps"] if s.get("observation"))
    results = agent_step["observation"]["results"]
    assert {r["source_call_id"] for r in results} == {"a", "b"}  # distinct, id-linked
