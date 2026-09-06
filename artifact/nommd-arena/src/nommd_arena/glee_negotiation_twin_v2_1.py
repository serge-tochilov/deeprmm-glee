"""Exact-surplus and monotone-response extension of the GLEE Negotiation twin."""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from .glee_negotiation_twin_v2 import GaussianComponent, GaussianMixture, NegotiationContext, NegotiationDecisionRow, NegotiationModelConfig, NegotiationOpponentModelV2, PriorObservation, _optional_distance, _scaled_weights, _sha


SCHEMA_VERSION = 1
ENGINE_VERSION = "v2.1"
MODEL_VERSION = f"negotiation-twin-{ENGINE_VERSION}-development"
BASE_MODEL_VERSION = "negotiation-twin-v2.0-development"


@dataclass(frozen=True)
class NegotiationModelConfigV21(NegotiationModelConfig):
    """One audit-derived extension of the frozen v2.0 configuration."""

    proposal_uncertainty_contraction: float = 0.8201719692016325

    def validate(self) -> None:
        super().validate()
        if not 0 < self.proposal_uncertainty_contraction < 1:
            raise ValueError("proposal_uncertainty_contraction must lie in (0, 1)")


def response_context_similarity(current: NegotiationContext, previous: NegotiationContext) -> float:
    """Compare response contexts without the numeric offer coordinate that the monotone curve models."""
    value = 1.0
    value *= 1.0 if current.opponent_role == previous.opponent_role else 0.2
    value *= 1.0 if current.complete_information == previous.complete_information else 0.48
    value *= 1.0 if current.horizon_known == previous.horizon_known else 0.62
    value *= 1.0 if current.messages_allowed == previous.messages_allowed else 0.85
    value *= math.exp(-abs(current.round_phase - previous.round_phase) / 0.38)
    value *= math.exp(-0.08 * abs(math.log(current.our_value / previous.our_value)))
    if current.horizon_known and previous.horizon_known:
        if current.max_rounds is None or previous.max_rounds is None:
            value *= 0.7
        else:
            value *= math.exp(-abs(math.log(current.max_rounds / previous.max_rounds)) / 1.2)
    value *= _optional_distance(current.previous_opponent_demand, previous.previous_opponent_demand, 0.4, 0.72)
    value *= _optional_distance(current.previous_our_demand, previous.previous_our_demand, 0.4, 0.72)
    if current.previous_opponent_response != previous.previous_opponent_response:
        value *= 0.78
    if current.current_offer_message_act != previous.current_offer_message_act:
        value *= 0.88
    return max(value, 1e-12)


def weighted_isotonic_probability(points: Sequence[tuple[float, bool, float]], query: float) -> tuple[float | None, float, int]:
    """Fit deterministic weighted PAVA and linearly interpolate its monotone fitted levels."""
    aggregated: dict[float, list[float]] = {}
    for coordinate, accepted, weight in points:
        if not math.isfinite(coordinate) or not math.isfinite(weight) or weight <= 0:
            continue
        totals = aggregated.setdefault(float(coordinate), [0.0, 0.0])
        totals[0] += weight
        totals[1] += weight * float(accepted)
    if not aggregated:
        return None, 0.0, 0
    coordinates = sorted(aggregated)
    blocks: list[list[float | int]] = []
    for index, coordinate in enumerate(coordinates):
        weight, weighted_success = aggregated[coordinate]
        blocks.append([index, index, weight, weighted_success])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left[3]) / float(left[2])
            right_mean = float(right[3]) / float(right[2])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [[int(left[0]), int(right[1]), float(left[2]) + float(right[2]), float(left[3]) + float(right[3])]]
    fitted = [0.0] * len(coordinates)
    for start, end, weight, weighted_success in blocks:
        probability = float(weighted_success) / float(weight)
        for index in range(int(start), int(end) + 1):
            fitted[index] = probability
    if query <= coordinates[0]:
        probability = fitted[0]
    elif query >= coordinates[-1]:
        probability = fitted[-1]
    else:
        right_index = bisect.bisect_right(coordinates, query)
        left_index = right_index - 1
        span = coordinates[right_index] - coordinates[left_index]
        fraction = (query - coordinates[left_index]) / span
        probability = fitted[left_index] + fraction * (fitted[right_index] - fitted[left_index])
    return probability, sum(value[0] for value in aggregated.values()), len(coordinates)


def affine_contract_mixture(distribution: GaussianMixture, contraction: float) -> GaussianMixture:
    """Contract the entire mixture around its median while preserving weights and modal order."""
    if not 0 < contraction < 1:
        raise ValueError("contraction must lie in (0, 1)")
    center = distribution.quantile(0.5)
    components = [
        GaussianComponent(
            mean=center + contraction * (component.mean - center),
            sigma=contraction * component.sigma,
            weight=component.weight,
            source="audit-affine-contraction",
        )
        for component in distribution.components
    ]
    return GaussianMixture(components)


class NegotiationOpponentModelV21(NegotiationOpponentModelV2):
    """Retain v2.0 ablations while adding monotone responses and calibrated proposal width."""

    def __init__(self, config: NegotiationModelConfigV21 | None = None) -> None:
        selected = config or NegotiationModelConfigV21()
        selected.validate()
        super().__init__(selected)
        self.config: NegotiationModelConfigV21 = selected

    def _response_prior_weights(
        self,
        current: NegotiationDecisionRow,
        observations: Sequence[PriorObservation],
        *,
        current_global_game_index: int,
        current_target_game_index: int,
        scope: Literal["population", "target"],
    ) -> list[tuple[NegotiationDecisionRow, float]]:
        selected: list[tuple[NegotiationDecisionRow, float]] = []
        for observation in observations:
            previous = observation.row
            if previous.action_type != "response":
                continue
            if scope == "population":
                age = max(0, current_global_game_index - observation.global_game_index - 1)
                recency = self.config.population_game_decay**age
            else:
                age = max(0, current_target_game_index - observation.opponent_game_index - 1)
                recency = self.config.target_game_decay**age
            selected.append((previous, recency * response_context_similarity(current.context, previous.context)))
        return selected

    def _response_prefix_weights(self, current: NegotiationDecisionRow, prefix: Sequence[NegotiationDecisionRow]) -> list[tuple[NegotiationDecisionRow, float]]:
        matching = [row for row in prefix if row.action_type == "response"]
        return [
            (previous, self.config.same_game_strength * self.config.same_game_decay**index * response_context_similarity(current.context, previous.context))
            for index, previous in enumerate(reversed(matching))
        ]

    def response_forecast(
        self,
        row: NegotiationDecisionRow,
        *,
        population_prior: Sequence[PriorObservation],
        target_prior: Sequence[PriorObservation],
        prefix: Sequence[NegotiationDecisionRow],
        current_global_game_index: int,
        current_target_game_index: int,
    ) -> dict[str, dict[str, float | int | str]]:
        forecasts: dict[str, dict[str, float | int | str]] = dict(
            super().response_forecast(
                row,
                population_prior=population_prior,
                target_prior=target_prior,
                prefix=prefix,
                current_global_game_index=current_global_game_index,
                current_target_game_index=current_target_game_index,
            )
        )
        population = _scaled_weights(
            self._response_prior_weights(row, population_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="population"),
            self.config.population_equivalent_rows,
        )
        target = self._response_prior_weights(row, target_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="target")
        same_game = self._response_prefix_weights(row, prefix)
        weighted = [*population, *target, *same_game]
        use_exact_surplus = row.offered_opponent_surplus_share is not None
        coordinate_name = "exact_opponent_surplus_share" if use_exact_surplus else "opponent_demand"
        query = row.offered_opponent_surplus_share if use_exact_surplus else row.offered_demand
        if query is None:
            raise ValueError("response row has no usable offer coordinate")
        points: list[tuple[float, bool, float]] = []
        for previous, weight in weighted:
            coordinate = previous.offered_opponent_surplus_share if use_exact_surplus else previous.offered_demand
            if coordinate is not None and previous.accepted is not None:
                points.append((float(coordinate), bool(previous.accepted), weight))
        isotonic, evidence_mass, coordinate_count = weighted_isotonic_probability(points, float(query))
        base_probability = self.config.response_base_probability
        base_mass = self.config.response_base_equivalent_rows
        if isotonic is None:
            probability = base_probability
        else:
            probability = (base_mass * base_probability + evidence_mass * isotonic) / (base_mass + evidence_mass)
        probability = min(1 - 1e-6, max(1e-6, probability))
        forecasts["adaptive_v2_1"] = {
            "probability": probability,
            "effective_mass": base_mass + evidence_mass,
            "evidence_mass": evidence_mass,
            "population_evidence_mass": sum(weight for _previous, weight in population),
            "target_evidence_mass": sum(weight for _previous, weight in target),
            "same_game_evidence_mass": sum(weight for _previous, weight in same_game),
            "coordinate_count": coordinate_count,
            "coordinate": coordinate_name,
        }
        return forecasts

    def proposal_forecast(
        self,
        row: NegotiationDecisionRow,
        *,
        population_prior: Sequence[PriorObservation],
        target_prior: Sequence[PriorObservation],
        prefix: Sequence[NegotiationDecisionRow],
        current_global_game_index: int,
        current_target_game_index: int,
    ) -> dict[str, GaussianMixture]:
        forecasts = super().proposal_forecast(
            row,
            population_prior=population_prior,
            target_prior=target_prior,
            prefix=prefix,
            current_global_game_index=current_global_game_index,
            current_target_game_index=current_target_game_index,
        )
        return {**forecasts, "adaptive_v2_1": affine_contract_mixture(forecasts["adaptive_v2"], self.config.proposal_uncertainty_contraction)}


def model_receipt_v2_1(config: NegotiationModelConfigV21) -> dict[str, object]:
    """Return the stable v2.1 model identity and its frozen v2.0 ancestry."""
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "model_version": MODEL_VERSION,
        "base_model_version": BASE_MODEL_VERSION,
        "mechanisms": ["weighted-isotonic-response", "exact-surplus-when-visible", "audit-affine-proposal-contraction"],
        "config": asdict(config),
        "config_sha256": _sha(asdict(config)),
    }
