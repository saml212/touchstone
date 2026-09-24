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
