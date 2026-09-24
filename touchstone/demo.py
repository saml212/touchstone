"""Zero-key demo: a scripted support agent producing real captured episodes.

Everything runs through the real capture path — `@touchstone.tool` for tools and
`touchstone.record_llm_call` for model turns — so the demo exercises the same code an
instrumented app would. Deterministic: same seed => same episodes (ids aside).
"""

from __future__ import annotations

import random

import touchstone
from touchstone.llm import provider_from_spec

SYSTEM = "You are a customer support agent. Look up orders, issue refunds, escalate when needed."

TOOLS = [
    {"type": "function", "function": {"name": "order_status",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "refund", "parameters": {"type": "object",
        "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "escalate",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}}},
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


def run_demo(n: int = 30, seed: int = 1729) -> int:
    provider = provider_from_spec("scripted")
    rng = random.Random(seed)
    for i in range(n):
        oid = f"A{rng.randint(1000, 9999)}"
        scenario = rng.choice(SCENARIOS).format(oid=oid)
        with touchstone.episode(f"support-{oid}-{i}", meta={"agent": "demo-support"}):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": scenario},
            ]
            reply = provider.chat(messages, tools=TOOLS)
            touchstone.record_llm_call("scripted:demo", messages, reply, tools=TOOLS)

            if rng.random() < 0.70:
                order_status(oid)
                if "refund" in scenario or "twice" in scenario:
                    refund(oid, round(rng.uniform(10, 200), 2))
                if i % 7 == 0:
                    final = f"All sorted. Confirmation emailed to customer{i}@example.com."
                else:
                    final = "All sorted — is there anything else I can help with?"
                touchstone.record_llm_call("scripted:demo", messages, final)
                touchstone.outcome(1.0, "resolved")
            else:
                escalate(scenario)
                label = "escalated" if rng.random() < 0.6 else "failed"
                touchstone.record_llm_call("scripted:demo", messages, "Escalating to a human.")
                touchstone.outcome(0.0, label)
    return n
