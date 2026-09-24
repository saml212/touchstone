import json

import touchstone
from touchstone.capture import atif, context


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


def test_atif_empty_episode_has_fallback_step(traced):
    conn = context.get_conn()
    with touchstone.episode("empty") as ep:
        pass
    traj = atif.to_atif(conn, ep.id)
    atif.validate(traj)
    assert len(traj["steps"]) == 1


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
