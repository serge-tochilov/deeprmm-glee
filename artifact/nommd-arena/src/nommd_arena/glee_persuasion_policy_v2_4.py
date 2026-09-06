"""Longitudinal seller control for GLEE Persuasion v2.4."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

from .glee_persuasion_twin_v2 import PersuasionContext


SCHEMA_VERSION = 1
POLICY_VERSION = "persuasion-longitudinal-seller-v2.4"


def _finite_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


@dataclass(frozen=True)
class PersuasionSellerPolicyConfig:
    """Frozen thresholds for comparing immediate sales with reputation value."""

    deception_advantage_margin: float = 0.05
    no_response_min_passes: int = 4
    no_response_max_smoothed_buy_rate: float = 0.2
    no_response_routing: str = "advisory-only"

    def validate(self) -> None:
        if not 0 <= self.deception_advantage_margin <= 1:
            raise ValueError("deception_advantage_margin must lie in [0, 1]")
        if self.no_response_min_passes < 1:
            raise ValueError("no_response_min_passes must be positive")
        if not 0 <= self.no_response_max_smoothed_buy_rate <= 1:
            raise ValueError("no_response_max_smoothed_buy_rate must lie in [0, 1]")
        if self.no_response_routing not in {"advisory-only", "deterministic-quality-consistent"}:
            raise ValueError("no_response_routing must be advisory-only or deterministic-quality-consistent")


def canonical_seller_message(signal_polarity: str) -> str:
    """Return a legal quality-consistent fallback realization for one semantic signal."""
    if signal_polarity == "positive":
        return "I recommend buying this product; it is high quality."
    if signal_polarity == "negative":
        return "I do not recommend buying this product; it is low quality."
    raise ValueError(f"unsupported seller signal polarity: {signal_polarity}")


def seller_longitudinal_control(context: PersuasionContext, *, current_quality: str | None, buyer_response_forecasts: Mapping[str, object], config: PersuasionSellerPolicyConfig | None = None) -> dict[str, object]:
    """Select a semantic seller signal by comparing a low-item harvest with future reputation at risk."""
    selected = config or PersuasionSellerPolicyConfig()
    selected.validate()
    positive = buyer_response_forecasts.get("positive") if isinstance(buyer_response_forecasts.get("positive"), Mapping) else {}
    negative = buyer_response_forecasts.get("negative") if isinstance(buyer_response_forecasts.get("negative"), Mapping) else {}
    positive_buy_probability = _finite_probability(positive.get("buy_probability"))
    negative_buy_probability = _finite_probability(negative.get("buy_probability"))
    remaining_rounds = max(0, context.total_rounds - context.round_number)
    expected_future_high_items = context.prior_high_probability * remaining_rounds
    response_count = context.prior_buys + context.prior_passes
    smoothed_buy_rate = (1 + context.prior_buys) / (2 + response_count)
    no_response_regime = context.prior_passes >= selected.no_response_min_passes and smoothed_buy_rate <= selected.no_response_max_smoothed_buy_rate
    trust_state = "breached" if context.positive_observed_low > 0 else "verified" if context.positive_observed_high > 0 else "unverified"
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "status": "available" if current_quality in {"high", "low"} else "unavailable-current-quality",
        "current_quality": current_quality,
        "trust_state": trust_state,
        "response_regime": "no-response" if no_response_regime else "responsive-or-uncertain",
        "prior_buys": context.prior_buys,
        "prior_passes": context.prior_passes,
        "smoothed_current_game_buy_rate": round(smoothed_buy_rate, 6),
        "remaining_rounds_after_current": remaining_rounds,
        "expected_future_high_items": round(expected_future_high_items, 6),
        "positive_buy_probability": positive_buy_probability,
        "negative_buy_probability": negative_buy_probability,
        "selected_signal_polarity": None,
        "selection_reason": "current quality unavailable",
        "authority": "none",
        "config": asdict(selected),
        "boundary": "The calculation values reputation through expected future high-quality sales. It is a conservative continuation proxy, not an equilibrium proof or a claim that deception is always harmful.",
    }
    if current_quality not in {"high", "low"}:
        return result
    if current_quality == "high":
        result.update({"selected_signal_polarity": "positive", "selection_reason": "sell the current high-quality item while preserving a quality-consistent reputation", "authority": "semantic-authority"})
        return result
    if positive_buy_probability is None or negative_buy_probability is None:
        result.update({"selected_signal_polarity": "negative", "selection_reason": "preserve reputation because candidate response probabilities are unavailable", "authority": "semantic-authority"})
        return result
    immediate_deception_advantage = max(0.0, positive_buy_probability - negative_buy_probability)
    future_reputation_sales_at_risk = positive_buy_probability * expected_future_high_items
    harvest_margin = immediate_deception_advantage - future_reputation_sales_at_risk
    harvest_eligible = harvest_margin > selected.deception_advantage_margin
    result.update(
        {
            "immediate_deception_sale_advantage": round(immediate_deception_advantage, 6),
            "future_reputation_sales_at_risk": round(future_reputation_sales_at_risk, 6),
            "harvest_margin_after_reputation_cost": round(harvest_margin, 6),
            "harvest_eligible": harvest_eligible,
        }
    )
    if harvest_eligible:
        result.update({"selected_signal_polarity": "positive", "selection_reason": "modeled immediate low-item sale advantage exceeds expected future high-item reputation value plus margin", "authority": "semantic-authority"})
    else:
        result.update({"selected_signal_polarity": "negative", "selection_reason": "expected future high-item reputation value meets or exceeds the modeled low-item deception advantage", "authority": "semantic-authority"})
    return result
