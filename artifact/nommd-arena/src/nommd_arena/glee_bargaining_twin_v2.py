"""Adaptive executable opponent model developed separately from sealed bargaining v1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .glee_bargaining_twin import BargainingContext, BargainingDecisionRow, _clamp
from .glee_bargaining_validation import PriorObservation


ENGINE_VERSION = "v2.0"
MODEL_VERSION = f"bargaining-twin-{ENGINE_VERSION}-development"
RESPONSE_EXPERTS = ("hierarchical_program", "recency_kernel", "regularized_tabular", "target_recency")
PROPOSAL_EXPERTS = ("hierarchical_program", "recency_kernel", "regularized_tabular", "mode_kernel")
_SQRT_TWO = math.sqrt(2.0)


@dataclass(frozen=True)
class AdaptiveConfig:
    """Frozen development settings for recency, multimodal prediction, and online expert adaptation."""

    target_recency_decay: float = 0.9
    population_recency_decay: float = 0.997
    proposal_population_equivalent_rows: float = 3.0
    proposal_kernel_sigma: float = 0.025
    proposal_floor_sigma: float = 0.22
    proposal_floor_equivalent_rows: float = 0.35
    response_bandwidth: float = 0.075
    response_prior_equivalent_rows: float = 1.5
    expert_score_decay: float = 0.88
    expert_learning_rate: float = 0.45
    expert_uniform_mix: float = 0.05
    expert_loss_floor: float = -3.0
    expert_loss_ceiling: float = 8.0
    response_prior_weights: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)
    proposal_prior_weights: tuple[float, ...] = (0.45, 0.1, 0.2, 0.25)

    def validate(self) -> None:
        for name in ("target_recency_decay", "population_recency_decay", "expert_score_decay"):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.proposal_population_equivalent_rows <= 0 or self.proposal_kernel_sigma <= 0 or self.proposal_floor_sigma <= 0 or self.proposal_floor_equivalent_rows <= 0:
            raise ValueError("proposal mixture scales must be positive")
        if self.response_bandwidth <= 0 or self.response_prior_equivalent_rows <= 0:
            raise ValueError("response kernel scales must be positive")
        if self.expert_learning_rate < 0 or not 0 <= self.expert_uniform_mix < 1:
            raise ValueError("expert adaptation settings are invalid")
        if self.expert_loss_floor >= self.expert_loss_ceiling:
            raise ValueError("expert loss bounds are inverted")
        if len(self.response_prior_weights) != len(RESPONSE_EXPERTS) or len(self.proposal_prior_weights) != len(PROPOSAL_EXPERTS):
            raise ValueError("expert prior length does not match the model grammar")
        if any(value <= 0 for value in self.response_prior_weights + self.proposal_prior_weights):
            raise ValueError("expert prior weights must be positive")


@dataclass(frozen=True)
class GaussianComponent:
    """One component of a bounded-target Gaussian mixture."""

    mean: float
    sigma: float
    weight: float
    source: str


class GaussianMixture:
    """Small dependency-free mixture retaining policy modes rather than only moments."""

    def __init__(self, components: Sequence[GaussianComponent]) -> None:
        selected = [component for component in components if component.weight > 0 and component.sigma > 0 and math.isfinite(component.mean) and math.isfinite(component.sigma) and math.isfinite(component.weight)]
        if not selected:
            raise ValueError("Gaussian mixture has no valid components")
        total = sum(component.weight for component in selected)
        calculation_weights: dict[tuple[float, float], float] = {}
        for component in selected:
            key = (_clamp(component.mean, 0.0, 1.0), component.sigma)
            calculation_weights[key] = calculation_weights.get(key, 0.0) + component.weight
        self.components = tuple(GaussianComponent(mean, sigma, weight / total, "coalesced-identical-components") for (mean, sigma), weight in calculation_weights.items())
        self.component_count = len(selected)

    @classmethod
    def single(cls, *, mean: float, sigma: float, source: str) -> GaussianMixture:
        return cls((GaussianComponent(mean, sigma, 1.0, source),))

    @classmethod
    def blend(cls, distributions: Mapping[str, GaussianMixture], weights: Mapping[str, float]) -> GaussianMixture:
        components = [
            GaussianComponent(component.mean, component.sigma, float(weights[name]) * component.weight, f"{name}:{component.source}")
            for name, distribution in distributions.items()
            for component in distribution.components
            if float(weights.get(name, 0.0)) > 0
        ]
        result = cls(components)
        result.component_count = sum(distribution.component_count for name, distribution in distributions.items() if float(weights.get(name, 0.0)) > 0)
        return result

    @property
    def mean(self) -> float:
        return sum(component.weight * component.mean for component in self.components)

    @property
    def sigma(self) -> float:
        mean = self.mean
        variance = sum(component.weight * (component.sigma**2 + component.mean**2) for component in self.components) - mean * mean
        return max(0.001, math.sqrt(max(0.0, variance)))

    def density(self, observed: float) -> float:
        return sum(component.weight * math.exp(-0.5 * ((observed - component.mean) / component.sigma) ** 2) / (component.sigma * math.sqrt(2 * math.pi)) for component in self.components)

    def nll(self, observed: float) -> float:
        return -math.log(max(self.density(observed), 1e-300))

    def cdf(self, value: float) -> float:
        return sum(component.weight * 0.5 * (1.0 + math.erf((value - component.mean) / (component.sigma * _SQRT_TWO))) for component in self.components)

    def quantile(self, probability: float) -> float:
        low, high = 0.0, 1.0
        for _iteration in range(48):
            middle = (low + high) / 2
            if self.cdf(middle) < probability:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    def as_dict(self, *, include_components: bool = False) -> dict[str, object]:
        value: dict[str, object] = {"mean": self.mean, "sigma": self.sigma, "q10": self.quantile(0.1), "q50": self.quantile(0.5), "q90": self.quantile(0.9), "component_count": self.component_count}
        if include_components:
            value["components"] = [asdict(component) for component in self.components]
        return value


def _context_similarity(current: BargainingContext, previous: BargainingContext) -> float:
    value = 1.0
    value *= 1.0 if current.complete_information == previous.complete_information else 0.42
    value *= 1.0 if current.horizon_known == previous.horizon_known else 0.7
    value *= 1.0 if current.opponent_player == previous.opponent_player else 0.55
    value *= 1.0 if current.messages_allowed == previous.messages_allowed else 0.85
    value *= 1.0 if (current.round_number == 1) == (previous.round_number == 1) else 0.4
    value *= math.exp(-abs(current.progress - previous.progress) / 0.45)
    if current.opponent_discount is not None and previous.opponent_discount is not None:
        value *= math.exp(-abs(current.opponent_discount - previous.opponent_discount) / 0.18)
    elif (current.opponent_discount is None) != (previous.opponent_discount is None):
        value *= 0.8
    current_previous = current.previous_our_offer_to_opponent_share
    prior_previous = previous.previous_our_offer_to_opponent_share
    if current_previous is not None and prior_previous is not None:
        value *= math.exp(-abs(current_previous - prior_previous) / 0.12)
    elif (current_previous is None) != (prior_previous is None):
        value *= 0.55
    if current.previous_opponent_response != previous.previous_opponent_response:
        value *= 0.75
    return max(value, 1e-5)


def target_response_probability(
    row: BargainingDecisionRow,
    target_prior: Sequence[PriorObservation],
    *,
    current_target_game_index: int,
    base_probability: float,
    config: AdaptiveConfig,
) -> float:
    """Estimate a target-local response while shrinking unsupported regions to the program posterior."""
    offered = float(row.offered_share if row.offered_share is not None else 0.5)
    numerator = config.response_prior_equivalent_rows * base_probability
    denominator = config.response_prior_equivalent_rows
    for observation in target_prior:
        previous = observation.row
        if previous.action_type != "response" or previous.offered_share is None or previous.accepted is None:
            continue
        age = max(0, current_target_game_index - observation.opponent_game_index - 1)
        recency = config.target_recency_decay**age
        offer_similarity = math.exp(-abs(offered - float(previous.offered_share)) / config.response_bandwidth)
        weight = recency * offer_similarity * _context_similarity(row.context, previous.context)
        numerator += weight * float(previous.accepted)
        denominator += weight
    return _clamp(numerator / denominator, 1e-6, 1 - 1e-6)


def mode_kernel_distribution(
    row: BargainingDecisionRow,
    population_prior: Sequence[PriorObservation],
    target_prior: Sequence[PriorObservation],
    *,
    current_global_game_index: int,
    current_target_game_index: int,
    config: AdaptiveConfig,
) -> GaussianMixture:
    """Return a context-weighted mixture centered on causally prior proposal modes."""
    target_components: list[GaussianComponent] = []
    for observation in target_prior:
        previous = observation.row
        if previous.action_type != "proposal" or previous.proposal_share is None:
            continue
        age = max(0, current_target_game_index - observation.opponent_game_index - 1)
        weight = config.target_recency_decay**age * _context_similarity(row.context, previous.context)
        target_components.append(GaussianComponent(float(previous.proposal_share), config.proposal_kernel_sigma, weight, "target-mode"))
    population_raw: list[GaussianComponent] = []
    for observation in population_prior:
        previous = observation.row
        if previous.action_type != "proposal" or previous.proposal_share is None:
            continue
        age = max(0, current_global_game_index - observation.global_game_index - 1)
        weight = config.population_recency_decay**age * _context_similarity(row.context, previous.context)
        population_raw.append(GaussianComponent(float(previous.proposal_share), config.proposal_kernel_sigma, weight, "population-mode"))
    population_total = sum(component.weight for component in population_raw)
    population_scale = config.proposal_population_equivalent_rows / population_total if population_total else 0.0
    population_components = [GaussianComponent(component.mean, component.sigma, component.weight * population_scale, component.source) for component in population_raw]
    evidence_weight = sum(component.weight for component in target_components) + sum(component.weight for component in population_components)
    floor_weight = config.proposal_floor_equivalent_rows * max(1.0, math.sqrt(evidence_weight))
    floor = GaussianComponent(0.5, config.proposal_floor_sigma, floor_weight, "broad-floor")
    return GaussianMixture((*target_components, *population_components, floor))


class AdaptiveExpertState:
    """Opponent-local exponentially discounted log-score state over fixed experts."""

    def __init__(self, names: Sequence[str], priors: Sequence[float], config: AdaptiveConfig) -> None:
        if len(names) != len(priors):
            raise ValueError("expert names and priors have different lengths")
        total = sum(priors)
        if total <= 0:
            raise ValueError("expert prior has no mass")
        self.names = tuple(names)
        self.priors = {name: float(prior) / total for name, prior in zip(names, priors, strict=True)}
        self.scores = {name: 0.0 for name in names}
        self.update_count = 0
        self.config = config

    def weights(self) -> dict[str, float]:
        logits = {name: math.log(self.priors[name]) + self.scores[name] for name in self.names}
        maximum = max(logits.values())
        raw = {name: math.exp(value - maximum) for name, value in logits.items()}
        total = sum(raw.values())
        uniform = 1.0 / len(self.names)
        return {name: (1 - self.config.expert_uniform_mix) * raw[name] / total + self.config.expert_uniform_mix * uniform for name in self.names}

    def update(self, losses: Mapping[str, float]) -> None:
        if any(name not in losses or not math.isfinite(float(losses[name])) for name in self.names):
            raise ValueError("expert update requires one finite loss per expert")
        clipped = {name: _clamp(float(losses[name]), self.config.expert_loss_floor, self.config.expert_loss_ceiling) for name in self.names}
        center = sum(clipped.values()) / len(clipped)
        for name in self.names:
            self.scores[name] = self.config.expert_score_decay * self.scores[name] - self.config.expert_learning_rate * (clipped[name] - center)
        self.update_count += 1

    def as_dict(self) -> dict[str, object]:
        return {"update_count": self.update_count, "weights": self.weights(), "discounted_relative_log_scores": dict(self.scores)}


class AdaptiveOpponentState:
    """Separate response and proposal expert states for one recurring opponent."""

    def __init__(self, config: AdaptiveConfig) -> None:
        config.validate()
        self.response = AdaptiveExpertState(RESPONSE_EXPERTS, config.response_prior_weights, config)
        self.proposal = AdaptiveExpertState(PROPOSAL_EXPERTS, config.proposal_prior_weights, config)

    def as_dict(self) -> dict[str, object]:
        return {"response": self.response.as_dict(), "proposal": self.proposal.as_dict()}


@dataclass(frozen=True)
class ResponseForecast:
    probability: float
    expert_probabilities: dict[str, float]
    expert_weights: dict[str, float]


@dataclass(frozen=True)
class ProposalForecast:
    distribution: GaussianMixture
    expert_weights: dict[str, float]


class AdaptiveBargainingTwinV2:
    """Combine interpretable experts while allowing opponent-local policy-regime shifts."""

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self.config = config or AdaptiveConfig()
        self.config.validate()

    def response_forecast(
        self,
        row: BargainingDecisionRow,
        *,
        target_prior: Sequence[PriorObservation],
        current_target_game_index: int,
        base_probabilities: Mapping[str, float],
        state: AdaptiveOpponentState,
    ) -> ResponseForecast:
        missing = [name for name in RESPONSE_EXPERTS[:-1] if name not in base_probabilities]
        if missing:
            raise ValueError(f"missing response experts: {missing}")
        probabilities = {name: _clamp(float(base_probabilities[name]), 1e-6, 1 - 1e-6) for name in RESPONSE_EXPERTS[:-1]}
        probabilities["target_recency"] = target_response_probability(row, target_prior, current_target_game_index=current_target_game_index, base_probability=probabilities["hierarchical_program"], config=self.config)
        weights = state.response.weights()
        probability = sum(weights[name] * probabilities[name] for name in RESPONSE_EXPERTS)
        return ResponseForecast(_clamp(probability, 1e-6, 1 - 1e-6), probabilities, weights)

    def proposal_forecast(
        self,
        row: BargainingDecisionRow,
        *,
        population_prior: Sequence[PriorObservation],
        target_prior: Sequence[PriorObservation],
        current_global_game_index: int,
        current_target_game_index: int,
        base_distributions: Mapping[str, GaussianMixture],
        state: AdaptiveOpponentState,
    ) -> ProposalForecast:
        missing = [name for name in PROPOSAL_EXPERTS[:-1] if name not in base_distributions]
        if missing:
            raise ValueError(f"missing proposal experts: {missing}")
        distributions = dict(base_distributions)
        distributions["mode_kernel"] = mode_kernel_distribution(row, population_prior, target_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, config=self.config)
        weights = state.proposal.weights()
        return ProposalForecast(GaussianMixture.blend(distributions, weights), weights)
