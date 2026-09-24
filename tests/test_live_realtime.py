"""Live check: open a real OpenAI Realtime session and prove the model calls draft_check.

Skipped unless TOUCHSTONE_LIVE=1. Costs cents. The key resolves from $OPENAI_API_KEY or the macOS
Keychain entry `rockie-openai-api-key`. We send a text turn (deterministic, no audio conversion) and
assert a `draft_check` tool call comes back with the four Touchstone tools declared on the session.
"""

import asyncio
import json
import os

import pytest
import websockets

from touchstone.interview.realtime import TOOLS
from touchstone.llm.keychain import secret

pytestmark = pytest.mark.skipif(
    os.environ.get("TOUCHSTONE_LIVE") != "1", reason="live realtime; set TOUCHSTONE_LIVE=1")

MODEL = "gpt-realtime-2.1-mini"
_ASK = ("Draft a check that the agent's reply must mention the order id. "
        "Call the draft_check tool now with kind=contains.")


async def _draft_check_call() -> dict:
    key = secret("OPENAI_API_KEY", "rockie-openai-api-key")
    assert key, "no OpenAI API key in env or Keychain"
    url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    headers = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": "You are Touchstone's interviewer. When asked, call draft_check.",
            "output_modalities": ["text"], "tools": TOOLS, "tool_choice": "auto"}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": _ASK}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        async with asyncio.timeout(30):
            async for raw in ws:
                event = json.loads(raw)
                if event.get("type") == "response.function_call_arguments.done":
                    return event
                if event.get("type") == "error":
                    raise AssertionError(f"realtime error: {event}")
    raise AssertionError("no function call arrived")


def test_live_realtime_calls_draft_check():
    event = asyncio.run(_draft_check_call())
    assert event["name"] == "draft_check"
    args = json.loads(event["arguments"])
    assert args.get("kind")  # the model filled in a check kind
