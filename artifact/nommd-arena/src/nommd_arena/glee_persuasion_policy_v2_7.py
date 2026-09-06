"""Known-identity adaptive pooling for GLEE Persuasion v2.7."""

from __future__ import annotations

from typing import Mapping

from .glee_persuasion_policy_v2_4 import PersuasionSellerPolicyConfig
from .glee_persuasion_policy_v2_6 import canonical_seller_message, seller_longitudinal_control as seller_longitudinal_control_v2_6
from .glee_persuasion_twin_v2 import PersuasionContext


SCHEMA_VERSION = 1
POLICY_VERSION = "persuasion-ki-adaptive-pooling-v2.7"


def seller_longitudinal_control(context: PersuasionContext, *, current_quality: str | None, buyer_response_forecasts: Mapping[str, object], config: PersuasionSellerPolicyConfig | None = None) -> dict[str, object]:
    """Retain v2.6 pooling until a known buyer passes one pooled positive signal, then separate permanently."""
    result = seller_longitudinal_control_v2_6(context, current_quality=current_quality, buyer_response_forecasts=buyer_response_forecasts, config=config)
    pooling_eligible = result.get("prior_positive_pooling_eligible") is True
    known_buyer_rejected_pooling = pooling_eligible and context.opponent_named and context.positive_passes > 0
    result.update(
        {
            "policy_version": POLICY_VERSION,
            "adaptive_pooling_scope": "known-identity eligible seller games only",
            "positive_signal_passes_observed": context.positive_passes,
            "known_buyer_rejected_pooling": known_buyer_rejected_pooling,
            "adaptive_separation_active": known_buyer_rejected_pooling,
            "adaptive_boundary": "A known buyer's first pass after a positive pooled signal permanently changes the remainder of an eligible game to quality-consistent separation. Hidden-identity sellers and all ineligible states retain v2.6 behavior.",
        }
    )
    if not known_buyer_rejected_pooling or current_quality not in {"high", "low"}:
        return result
    selected_polarity = "positive" if current_quality == "high" else "negative"
    result.update(
        {
            "selected_signal_polarity": selected_polarity,
            "selection_reason": "known buyer positive pooled signal pass activates one-way quality-consistent separation",
            "authority": "semantic-authority",
        }
    )
    return result
