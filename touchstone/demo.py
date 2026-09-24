"""Zero-key demo: a scripted support agent producing real captured episodes.

Everything runs through the real capture path — `@touchstone.tool` for tools and
`touchstone.record_llm_call` for model turns — so the demo exercises the same code an
instrumented app would. Deterministic: same seed => same episodes (ids aside).
"""

from __future__ import annotations

import json
import random

import touchstone

SYSTEM = "You are a customer support agent. Look up orders, issue refunds, escalate when needed."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "order_status",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
        },
    },
]

SCENARIOS = [
    "Where is my order {oid}?",
    "I want a refund for order {oid}, it arrived broken.",
    "My order {oid} is late and I'm furious.",
    "Can you cancel order {oid}?",
    "Order {oid} charged me twice.",
]


@touchstone.tool
def order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


@touchstone.tool
def refund(order_id: str, amount: float) -> dict:
    return {"order_id": order_id, "refunded": amount, "currency": "USD"}


@touchstone.tool
def escalate(reason: str) -> dict:
    return {"escalated": True, "reason": reason}


TOOL_FNS = {"order_status": order_status, "refund": refund, "escalate": escalate}


def _turn(messages: list[dict], content: str = "", tool_calls: list[dict] | None = None) -> None:
    """One assistant turn through the real capture path, appended to the running conversation."""
    reply = {"content": content, "tool_calls": tool_calls or []}
    touchstone.record_llm_call("scripted:demo", list(messages), reply, tools=TOOLS)
    messages.append({"role": "assistant", "content": content, "tool_calls": reply["tool_calls"]})
    for call in reply["tool_calls"]:
        result = TOOL_FNS[call["name"]](**call["arguments"])
        messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result)})


def _resolved_episode(rng, messages: list[dict], oid: str, scenario: str, i: int) -> None:
    calls = [{"name": "order_status", "arguments": {"order_id": oid}}]
    if "refund" in scenario or "twice" in scenario:
        amount = round(rng.uniform(10, 200), 2)
        calls.append({"name": "refund", "arguments": {"order_id": oid, "amount": amount}})
    _turn(messages, tool_calls=calls)
    if i % 7 == 0:
        final = f"All sorted. Confirmation emailed to customer{i}@example.com."
    else:
        final = "All sorted — is there anything else I can help with?"
    _turn(messages, content=final)
    touchstone.outcome(1.0, "resolved")


def _escalated_episode(rng, messages: list[dict], scenario: str) -> None:
    _turn(messages, tool_calls=[{"name": "escalate", "arguments": {"reason": scenario}}])
    _turn(messages, content="Escalating to a human.")
    touchstone.outcome(0.0, "escalated" if rng.random() < 0.6 else "failed")


def run_demo(n: int = 30, seed: int = 1729) -> int:
    rng = random.Random(seed)
    for i in range(n):
        oid = f"A{rng.randint(1000, 9999)}"
        scenario = rng.choice(SCENARIOS).format(oid=oid)
        with touchstone.episode(f"support-{oid}-{i}", meta={"agent": "demo-support"}):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": scenario},
            ]
            if rng.random() < 0.70:
                _resolved_episode(rng, messages, oid, scenario, i)
            else:
                _escalated_episode(rng, messages, scenario)
    return n
