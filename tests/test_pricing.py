"""Tests for cost estimation."""

from app.pricing import estimate_cost_usd


def test_known_model_cost_is_computed():
    # 1M input + 1M output on gpt-4o-mini = 0.15 + 0.60
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_versioned_model_name_resolves_to_base_rates():
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15


def test_unknown_model_returns_none_not_a_guess():
    """An unknown model must be visible, not silently priced at zero."""
    assert estimate_cost_usd("some-unreleased-model", 1_000_000, 1_000_000) is None


def test_zero_tokens_costs_nothing():
    assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0
