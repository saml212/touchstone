"""Token pricing for cost estimates. Unknown models return None — a guessed cost is worse than none.

Prices are USD per one million tokens (input, output), from each vendor's public list. The model
name is matched as a substring of the run's `model_spec` (e.g. "openai:gpt-4o-mini"), longest key
first so "gpt-4o-mini" wins over "gpt-4o".
"""

from __future__ import annotations

# (input_per_mtok, output_per_mtok)
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "deepseek-chat": (0.27, 1.10),
}

_BY_LENGTH = sorted(PRICES, key=len, reverse=True)


def price_for(model_spec: str) -> tuple[float, float] | None:
    for key in _BY_LENGTH:
        if key in model_spec:
            return PRICES[key]
    return None


def cost_usd(model_spec: str, usage: dict | None) -> float | None:
    """Cost of one call, or None when the model is unpriced or usage is missing."""
    price = price_for(model_spec)
    if price is None or not usage:
        return None
    tokens_in = usage.get("tokens_in")
    tokens_out = usage.get("tokens_out")
    if tokens_in is None and tokens_out is None:
        return None
    in_price, out_price = price
    return (tokens_in or 0) / 1e6 * in_price + (tokens_out or 0) / 1e6 * out_price
