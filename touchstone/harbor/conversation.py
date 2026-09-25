"""The recorded tool-calling loop, driven one user turn at a time.

`run_until_reply` runs model/tool steps until the assistant answers with no tool call or the step
budget (model calls left in the whole conversation) is spent, recording each model call and tool
result as a span. `record_user` appends a user turn and records it as a `kind="user"` span so every
turn of a simulated-user conversation lands in the ATIF trajectory. Shared by the single-turn
replica loop (`harbor.agent`) and the multi-turn ACP server (`harbor.acp_server`): the same loop,
whether the whole goal arrives in one message or the user reveals it across turns.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex

from .. import store

OUTPUT_DEFAULT = "/app/output.json"  # where the verifier collects the run's answer


def final_text(messages: list[dict]) -> str:
    """The last assistant reply — the run's answer, written to output.json for the verifier."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


async def write_output(environment, answer: str) -> None:
    """Write {"answer": <final reply>} to output.json inside the sandbox, so the verifier can
    collect the /app/output.json artifact and grade the answer. Without it `harbor job regrade`
    refuses the trial. Packaged mode writes the same file from the customer's own entrypoint."""
    payload = json.dumps({"answer": answer}, ensure_ascii=False)
    path = os.environ.get("TOUCHSTONE_OUTPUT", OUTPUT_DEFAULT)
    await environment.exec(
        f"mkdir -p $(dirname {shlex.quote(path)}) && printf %s {shlex.quote(payload)} "
        f"> {shlex.quote(path)}")


def record_user(conn, episode_id: str, messages: list[dict], text: str) -> None:
    """Append a user turn to the running conversation and record it as its own span."""
    store.insert_span(conn, store.Span(episode_id=episode_id, kind="user", name="user",
                                       input={"content": text}, output={}))
    messages.append({"role": "user", "content": text})


async def run_until_reply(provider, config, messages: list[dict], environment, conn,
                          episode_id: str, remaining: int) -> tuple[str, dict, int]:
    """Drive the loop for one user turn. Returns (assistant text, usage totals, model calls used).

    `remaining` is the model-call budget left across the whole conversation, so a multi-turn run
    never exceeds `max_steps` model calls in total."""
    totals = {"tokens_in": 0, "tokens_out": 0}
    used, text = 0, ""
    while used < remaining:
        reply = await asyncio.to_thread(provider.chat, messages, config.tools or None)
        used += 1
        _add_usage(totals, reply.usage)
        _record_model_span(conn, episode_id, list(messages), config.tools, reply)
        messages.append({"role": "assistant", "content": reply.content,
                         "tool_calls": reply.tool_calls})
        text = reply.content or text
        if not reply.tool_calls:
            break
        for tool_call in reply.tool_calls:
            await run_tool(config, tool_call, environment, messages, conn, episode_id)
    return text, totals, used


async def run_tool(config, tool_call, environment, messages, conn, episode_id) -> None:
    name = tool_call.get("name") or "tool"
    try:
        arguments = json.loads(tool_call.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = {}
    result = await _dispatch(config.call, name, arguments, environment)
    result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    call_id = tool_call.get("id")
    store.insert_span(conn, store.Span(
        episode_id=episode_id, kind="tool", name=name,
        input={"name": name, "arguments": arguments},
        output={"result": result}, tool_call_id=call_id))
    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})


async def _dispatch(call, name: str, arguments: dict, environment):
    """Call the task's tool dispatch off the event loop unless it is a coroutine function."""
    if call is None:
        return ""
    if inspect.iscoroutinefunction(call):
        return await call(name, arguments, environment)
    return await asyncio.to_thread(call, name, arguments, environment)


def _record_model_span(conn, episode_id: str, history: list[dict], tools, reply) -> None:
    output: dict = {"message": {"role": "assistant", "content": reply.content,
                                "tool_calls": reply.tool_calls}}
    if reply.usage:
        output["usage"] = reply.usage
    store.insert_span(conn, store.Span(
        episode_id=episode_id, kind="model", name="model",
        input={"messages": history, "tools": tools or [], "params": {}}, output=output,
        tokens_in=(reply.usage or {}).get("tokens_in"),
        tokens_out=(reply.usage or {}).get("tokens_out")))


def _add_usage(totals: dict, usage: dict | None) -> None:
    if usage:
        totals["tokens_in"] += usage.get("tokens_in") or 0
        totals["tokens_out"] += usage.get("tokens_out") or 0
