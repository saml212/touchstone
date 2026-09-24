"""Real-SDK smoke test: traces an actual gpt-4o-mini chat (parallel tool calls) and a Responses
API call, then asserts the stored spans linked tool_call ids and captured usage.

Skipped unless TOUCHSTONE_LIVE=1. Costs well under a cent. The key resolves from
$OPENAI_API_KEY or the macOS Keychain entry `rockie-openai-api-key`.
"""

import os

import pytest

import touchstone
from touchstone import store
from touchstone.capture import context
from touchstone.llm.keychain import secret

pytestmark = pytest.mark.skipif(
    os.environ.get("TOUCHSTONE_LIVE") != "1", reason="live SDK smoke; set TOUCHSTONE_LIVE=1")

_WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def test_live_openai_parallel_tools_and_responses(tmp_path):
    import openai

    key = secret("OPENAI_API_KEY", "rockie-openai-api-key")
    assert key, "no OpenAI API key in env or Keychain"

    db = str(tmp_path / "live.db")
    context._local.__dict__.pop("conn", None)
    touchstone.trace(db)
    client = openai.OpenAI(api_key=key)

    convo = [{"role": "user", "content":
              "Call get_weather for BOTH London and Paris in a single turn."}]
    with touchstone.episode("live-smoke") as ep:
        first = client.chat.completions.create(
            model="gpt-4o-mini", messages=convo, tools=_WEATHER_TOOL, tool_choice="required")
        # The example app's pattern: append the SDK's own message object (pydantic, no .get),
        # then tool-result dicts carrying tool_call_id, and call create again.
        assistant = first.choices[0].message
        convo.append(assistant)
        for call in assistant.tool_calls:
            convo.append({"role": "tool", "tool_call_id": call.id, "content": "18C and sunny"})
        client.chat.completions.create(model="gpt-4o-mini", messages=convo)
        client.responses.create(model="gpt-4o-mini", input="Reply with the single word: hello.")

    conn = store.connect(db)
    spans = [s for s in store.list_spans(conn, ep.id) if s.kind == "model"]
    conn.close()
    assert len(spans) == 3

    chat = spans[0]
    calls = chat.output["message"]["tool_calls"]
    assert len(calls) >= 2, "expected parallel tool calls"
    assert all(c["id"] for c in calls), "every tool call has a linked id"
    assert len({c["id"] for c in calls}) == len(calls), "ids are distinct"
    assert chat.output["stop_reason"] == "tool_calls"
    assert chat.tokens_in and chat.tokens_out and chat.cost_usd

    # Second call's stored context canonicalized the appended pydantic message + tool results.
    followup = spans[1].input["messages"]
    stored_assistant = next(m for m in followup if m.get("tool_calls"))
    assert {c["id"] for c in stored_assistant["tool_calls"]} == {c["id"] for c in calls}
    tools = [m for m in followup if m["role"] == "tool"]
    assert len(tools) == len(calls) and all(t.get("tool_call_id") for t in tools)

    responses = spans[2]
    assert responses.output["message"]["content"]
    assert responses.tokens_in and responses.tokens_out
