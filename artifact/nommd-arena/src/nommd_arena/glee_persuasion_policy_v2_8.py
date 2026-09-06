"""Bounded no-response routing for GLEE Persuasion v2.8."""

from __future__ import annotations

from typing import Mapping

from .glee_persuasion_policy_v2_4 import PersuasionSellerPolicyConfig
from .glee_persuasion_policy_v2_7 import canonical_seller_message, seller_longitudinal_control as seller_longitudinal_control_v2_7
from .glee_persuasion_twin_v2 import PersuasionContext


SCHEMA_VERSION = 1
POLICY_VERSION = "persuasion-bounded-no-response-routing-v2.8"


def seller_longitudinal_control(context: PersuasionContext, *, current_quality: str | None, buyer_response_forecasts: Mapping[str, object], config: PersuasionSellerPolicyConfig | None = None) -> dict[str, object]:
    """Bypass cloud deliberation after a causally observed current-game no-response boundary."""
    selected = config or PersuasionSellerPolicyConfig()
    selected.validate()
    result = seller_longitudinal_control_v2_7(context, current_quality=current_quality, buyer_response_forecasts=buyer_response_forecasts, config=selected)
    routing_active = selected.no_response_routing == "deterministic-quality-consistent" and result.get("response_regime") == "no-response" and current_quality in {"high", "low"}
    result.update(
        {
            "policy_version": POLICY_VERSION,
            "no_response_routing": selected.no_response_routing,
            "no_response_routing_active": routing_active,
            "no_response_boundary": "The route activates only after the configured number of visible current-game passes and a sufficiently low smoothed current-game buy rate. It is an efficiency boundary for this trajectory, not a claim that the buyer has an immutable type.",
        }
    )
    if not routing_active:
        return result
    selected_polarity = "positive" if current_quality == "high" else "negative"
    result.update(
        {
            "selected_signal_polarity": selected_polarity,
            "selection_reason": "current-game no-response boundary activates deterministic quality-consistent signaling",
            "authority": "bounded-no-response-authority",
        }
    )
    return result
