"""Token pricing and cost estimation.

Lives in app/ rather than eval/ because cost-per-query is a product metric, and
app/ must not depend on eval/. The eval harness reuses it for judge cost.

Costs are ESTIMATES: token counts x published list rates. They exclude batching
discounts, cached-input pricing, and any negotiated terms.
"""

from __future__ import annotations

# USD per 1,000,000 tokens, as (input, output).
#
# ⚠️ Last checked 2026-07-20. Provider pricing changes — verify against the
# vendor's pricing page before publishing numbers, or set PRICE_INPUT_PER_1M /
# PRICE_OUTPUT_PER_1M in .env to override.
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _rates(model: str) -> tuple[float, float] | None:
    """Look up (input, output) rates, tolerating version suffixes."""
    from app.config import settings

    if settings.price_input_per_1m is not None and settings.price_output_per_1m is not None:
        return (settings.price_input_per_1m, settings.price_output_per_1m)

    if not model:
        return None
    if model in PRICING_PER_1M:
        return PRICING_PER_1M[model]
    # e.g. "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini"
    for known, rates in PRICING_PER_1M.items():
        if model.startswith(known):
            return rates
    return None


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost, or None if the model's rates are unknown.

    Returning None rather than 0.0 or a guess keeps an unknown model visible
    instead of silently reporting a wrong (and flatteringly small) number.
    """
    rates = _rates(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 8)
