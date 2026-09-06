"""Causal rolling-origin validation for executable GLEE bargaining models."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .glee_bargaining_twin import (
    MODEL_VERSION,
    BargainingContext,
    BargainingDecisionRow,
    BargainingGameEvidence,
    ProposalParticle,
    TwinConfig,
    _clamp,
    _mixed_population_prior,
    _ood_flags,
    _posterior_from_prior,
    _proposal_log_likelihood,
    _proposal_scores,
    _response_prediction,
    _response_scores,
    _sha,
    _sha_file,
    _support,
    load_bargaining_corpus,
    proposal_particles,
    response_particles,
)
from .glee_validation_plots import write_validation_figures


VALIDATION_SCHEMA_VERSION = 1
VALIDATION_VERSION = "bargaining-validation-v1"
MODEL_NAMES = ("hierarchical_program", "population_program", "opponent_program", "recency_kernel", "regularized_tabular", "unconditional_empirical")
_NORMAL_80 = 1.2815515655446004


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    _atomic_text(path, "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for value in values))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _safe_probability(value: float) -> float:
    return _clamp(value, 1e-6, 1 - 1e-6)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 40.0)))
    exponential = math.exp(max(value, -40.0))
    return exponential / (1.0 + exponential)


def _normal_nll(observed: float, mean: float, sigma: float) -> float:
    residual = (observed - mean) / sigma
    return 0.5 * residual * residual + math.log(sigma * math.sqrt(2 * math.pi))


def _entropy(weights: Sequence[float]) -> float:
    if len(weights) < 2:
        return 0.0
    raw = -sum(weight * math.log(weight) for weight in weights if weight > 0)
    return raw / math.log(len(weights))


def _add_scores(total: list[float], update: Sequence[float]) -> None:
    for index, value in enumerate(update):
        total[index] += value


def _subtract_scores(total: Sequence[float], part: Sequence[float]) -> list[float]:
    return [left - right for left, right in zip(total, part, strict=True)]


def _solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [list(row) + [float(vector[index])] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            augmented[pivot][column] += 1e-8
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-14:
            return [0.0] * size
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column], strict=True)]
    return [augmented[index][-1] for index in range(size)]


@dataclass(frozen=True)
class ValidationConfig:
    """Rolling-origin selection and baseline settings."""

    min_games: int = 10
    warmup_games: int = 4
    population_equivalent_rows: float = 12.0
    target_recency_decay: float = 0.9
    population_recency_decay: float = 0.995
    response_kernel_bandwidth: float = 0.08
    proposal_kernel_bandwidth: float = 0.12
    logistic_l2: float = 0.75
    ridge_l2: float = 0.5
    calibration_bins: int = 10
    surprise_window: int = 12

    def validate(self) -> None:
        if self.min_games <= self.warmup_games:
            raise ValueError("min_games must exceed warmup_games")
        if self.warmup_games < 1:
            raise ValueError("warmup_games must be positive")
        if self.population_equivalent_rows <= 0:
            raise ValueError("population_equivalent_rows must be positive")
        if not 0 < self.target_recency_decay <= 1 or not 0 < self.population_recency_decay <= 1:
            raise ValueError("recency decays must lie in (0, 1]")
        if self.response_kernel_bandwidth <= 0 or self.proposal_kernel_bandwidth <= 0:
            raise ValueError("kernel bandwidths must be positive")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")


@dataclass(frozen=True)
class PriorObservation:
    """One row plus causal indices used only for weighting."""

    row: BargainingDecisionRow
    global_game_index: int
    opponent_game_index: int


@dataclass(frozen=True)
class WeightedRow:
    """One prior row and its baseline fitting weight."""

    row: BargainingDecisionRow
    weight: float


@dataclass(frozen=True)
class GaussianPrediction:
    mean: float
    sigma: float

    @property
    def q10(self) -> float:
        return _clamp(self.mean - _NORMAL_80 * self.sigma, 0.0, 1.0)

    @property
    def q90(self) -> float:
        return _clamp(self.mean + _NORMAL_80 * self.sigma, 0.0, 1.0)


def _ordinary_weights(population: Sequence[PriorObservation], target: Sequence[PriorObservation], config: ValidationConfig) -> tuple[list[WeightedRow], list[WeightedRow]]:
    population_weight = config.population_equivalent_rows / len(population) if population else 0.0
    return ([WeightedRow(observation.row, population_weight) for observation in population], [WeightedRow(observation.row, 1.0) for observation in target])


def _recency_weights(
    population: Sequence[PriorObservation],
    target: Sequence[PriorObservation],
    *,
    current_global_game_index: int,
    current_target_game_index: int,
    config: ValidationConfig,
) -> tuple[list[WeightedRow], list[WeightedRow]]:
    population_raw = [config.population_recency_decay ** max(0, current_global_game_index - observation.global_game_index - 1) for observation in population]
    population_scale = config.population_equivalent_rows / sum(population_raw) if population_raw else 0.0
    weighted_population = [WeightedRow(observation.row, raw * population_scale) for observation, raw in zip(population, population_raw, strict=True)]
    weighted_target = [
        WeightedRow(observation.row, config.target_recency_decay ** max(0, current_target_game_index - observation.opponent_game_index - 1))
        for observation in target
    ]
    return weighted_population, weighted_target


def _response_features(row: BargainingDecisionRow) -> tuple[float, ...]:
    context = row.context
    share = float(row.offered_share if row.offered_share is not None else 0.5)
    previous = context.previous_opponent_offer_share
    return (
        1.0,
        share,
        share * share,
        context.progress,
        float(context.complete_information),
        float(context.horizon_known),
        float(context.opponent_player == "player_1"),
        float(math.isclose(share, 0.5, abs_tol=0.005)),
        float(previous is not None),
        float(previous if previous is not None else 0.5),
        float((context.our_discount if context.our_discount is not None else 0.9) - 0.9),
    )


def _proposal_features(row: BargainingDecisionRow) -> tuple[float, ...]:
    context = row.context
    previous_our = context.previous_our_offer_to_opponent_share
    previous_opponent = context.previous_opponent_offer_share
    return (
        1.0,
        context.progress,
        float(context.complete_information),
        float(context.horizon_known),
        float(context.opponent_player == "player_1"),
        float(previous_our is not None),
        float(previous_our if previous_our is not None else 0.5),
        float(context.previous_opponent_response == "reject"),
        float(context.previous_opponent_response == "accept"),
        float(previous_opponent if previous_opponent is not None else 0.5),
        float((context.our_discount if context.our_discount is not None else 0.9) - 0.9),
    )


class LogisticResponseModel:
    """Small dependency-free weighted logistic regression."""

    def __init__(self, weights: Sequence[float]) -> None:
        self.weights = tuple(weights)

    @classmethod
    def fit(cls, examples: Sequence[WeightedRow], *, l2: float) -> LogisticResponseModel:
        selected = [example for example in examples if example.row.action_type == "response" and example.row.accepted is not None]
        width = len(_response_features(selected[0].row)) if selected else 11
        coefficients = [0.0] * width
        if not selected:
            return cls(coefficients)
        for _iteration in range(12):
            gradient = [0.0] * width
            hessian = [[0.0] * width for _ in range(width)]
            for example in selected:
                features = _response_features(example.row)
                probability = _sigmoid(sum(coefficient * feature for coefficient, feature in zip(coefficients, features, strict=True)))
                label = 1.0 if example.row.accepted else 0.0
                curvature = example.weight * max(probability * (1 - probability), 1e-5)
                for left in range(width):
                    gradient[left] += example.weight * (label - probability) * features[left]
                    for right in range(width):
                        hessian[left][right] += curvature * features[left] * features[right]
            for index in range(1, width):
                gradient[index] -= l2 * coefficients[index]
                hessian[index][index] += l2
            hessian[0][0] += 1e-4
            delta = _solve_linear(hessian, gradient)
            coefficients = [_clamp(value + change, -20.0, 20.0) for value, change in zip(coefficients, delta, strict=True)]
            if max(abs(change) for change in delta) < 1e-6:
                break
        return cls(coefficients)

    def probability(self, row: BargainingDecisionRow) -> float:
        features = _response_features(row)
        return _safe_probability(_sigmoid(sum(coefficient * feature for coefficient, feature in zip(self.weights, features, strict=True))))


class RidgeProposalModel:
    """Small dependency-free weighted ridge proposal regression."""

    def __init__(self, weights: Sequence[float], sigma: float) -> None:
        self.weights = tuple(weights)
        self.sigma = sigma

    @classmethod
    def fit(cls, examples: Sequence[WeightedRow], *, l2: float) -> RidgeProposalModel:
        selected = [example for example in examples if example.row.action_type == "proposal" and example.row.proposal_share is not None]
        width = len(_proposal_features(selected[0].row)) if selected else 11
        if not selected:
            return cls([0.0] * width, 0.16)
        matrix = [[0.0] * width for _ in range(width)]
        vector = [0.0] * width
        for example in selected:
            features = _proposal_features(example.row)
            observed = float(example.row.proposal_share)
            for left in range(width):
                vector[left] += example.weight * features[left] * observed
                for right in range(width):
                    matrix[left][right] += example.weight * features[left] * features[right]
        for index in range(1, width):
            matrix[index][index] += l2
        matrix[0][0] += 1e-4
        coefficients = _solve_linear(matrix, vector)
        residual_weight = 0.0
        residual_sum = 0.0
        for example in selected:
            predicted = sum(coefficient * feature for coefficient, feature in zip(coefficients, _proposal_features(example.row), strict=True))
            residual_sum += example.weight * (float(example.row.proposal_share) - predicted) ** 2
            residual_weight += example.weight
        sigma = _clamp(math.sqrt(residual_sum / residual_weight) if residual_weight else 0.16, 0.03, 0.3)
        return cls(coefficients, sigma)

    def prediction(self, row: BargainingDecisionRow) -> GaussianPrediction:
        mean = sum(coefficient * feature for coefficient, feature in zip(self.weights, _proposal_features(row), strict=True))
        return GaussianPrediction(_clamp(mean, 0.0, 1.0), self.sigma)


class UnconditionalEmpiricalModel:
    """Target-adapted reference without current-state features."""

    def __init__(self, examples: Sequence[WeightedRow]) -> None:
        responses = [example for example in examples if example.row.action_type == "response" and example.row.accepted is not None]
        response_weight = sum(example.weight for example in responses)
        self.acceptance = (0.5 + sum(example.weight * float(bool(example.row.accepted)) for example in responses)) / (1.0 + response_weight)
        proposals = [example for example in examples if example.row.action_type == "proposal" and example.row.proposal_share is not None]
        proposal_weight = sum(example.weight for example in proposals)
        self.proposal_mean = (0.5 + sum(example.weight * float(example.row.proposal_share) for example in proposals)) / (1.0 + proposal_weight)
        variance = (0.04 + sum(example.weight * (float(example.row.proposal_share) - self.proposal_mean) ** 2 for example in proposals)) / (1.0 + proposal_weight)
        self.proposal_sigma = _clamp(math.sqrt(variance), 0.03, 0.3)

    def response_probability(self) -> float:
        return _safe_probability(self.acceptance)

    def proposal_prediction(self) -> GaussianPrediction:
        return GaussianPrediction(_clamp(self.proposal_mean, 0.0, 1.0), self.proposal_sigma)


class RecencyKernelModel:
    """Local empirical interpolation over decayed prior observations."""

    def __init__(self, examples: Sequence[WeightedRow], config: ValidationConfig) -> None:
        self.examples = tuple(examples)
        self.config = config

    @staticmethod
    def _context_similarity(current: BargainingContext, previous: BargainingContext) -> float:
        value = 1.0
        value *= 1.0 if current.complete_information == previous.complete_information else 0.65
        value *= 1.0 if current.horizon_known == previous.horizon_known else 0.75
        value *= 1.0 if current.opponent_player == previous.opponent_player else 0.8
        return value

    def response_probability(self, row: BargainingDecisionRow) -> float:
        numerator = 0.5
        denominator = 1.0
        offered = float(row.offered_share if row.offered_share is not None else 0.5)
        for example in self.examples:
            previous = example.row
            if previous.action_type != "response" or previous.offered_share is None or previous.accepted is None:
                continue
            distance = abs(offered - float(previous.offered_share))
            similarity = math.exp(-distance / self.config.response_kernel_bandwidth) * self._context_similarity(row.context, previous.context)
            weight = example.weight * similarity
            numerator += weight * float(previous.accepted)
            denominator += weight
        return _safe_probability(numerator / denominator)

    def proposal_prediction(self, row: BargainingDecisionRow) -> GaussianPrediction:
        values: list[tuple[float, float]] = [(0.5, 1.0)]
        current_previous = row.context.previous_our_offer_to_opponent_share
        for example in self.examples:
            previous = example.row
            if previous.action_type != "proposal" or previous.proposal_share is None:
                continue
            similarity = self._context_similarity(row.context, previous.context)
            prior_previous = previous.context.previous_our_offer_to_opponent_share
            if current_previous is not None and prior_previous is not None:
                similarity *= math.exp(-abs(current_previous - prior_previous) / self.config.proposal_kernel_bandwidth)
            elif (current_previous is None) != (prior_previous is None):
                similarity *= 0.6
            values.append((float(previous.proposal_share), example.weight * similarity))
        denominator = sum(weight for _value, weight in values)
        mean = sum(value * weight for value, weight in values) / denominator
        variance = sum(weight * (value - mean) ** 2 for value, weight in values) / denominator
        return GaussianPrediction(_clamp(mean, 0.0, 1.0), _clamp(math.sqrt(variance), 0.03, 0.3))


def _particle_proposal_prediction(particles: Sequence[ProposalParticle], weights: Sequence[float], row: BargainingDecisionRow) -> GaussianPrediction:
    means = [particle.mean(row.context) for particle in particles]
    mean = sum(weight * value for weight, value in zip(weights, means, strict=True))
    second = sum(weight * (particle.sigma**2 + value**2) for particle, weight, value in zip(particles, weights, means, strict=True))
    variance = max(0.03**2, second - mean * mean)
    return GaussianPrediction(_clamp(mean, 0.0, 1.0), _clamp(math.sqrt(variance), 0.03, 0.3))


def _response_prediction_record(probability: float, accepted: bool) -> dict[str, float | bool]:
    probability = _safe_probability(probability)
    label = 1.0 if accepted else 0.0
    return {
        "probability": probability,
        "nll": -math.log(probability if accepted else 1 - probability),
        "brier": (probability - label) ** 2,
        "correct": (probability >= 0.5) == accepted,
    }


def _proposal_prediction_record(prediction: GaussianPrediction, observed: float, *, nll: float | None = None) -> dict[str, float | bool]:
    error = prediction.mean - observed
    return {
        "mean": prediction.mean,
        "sigma": prediction.sigma,
        "q10": prediction.q10,
        "q90": prediction.q90,
        "nll": _normal_nll(observed, prediction.mean, prediction.sigma) if nll is None else nll,
        "absolute_error": abs(error),
        "squared_error": error * error,
        "interval_80_covered": prediction.q10 <= observed <= prediction.q90,
    }


def _metric_block(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    responses = [record for record in records if record["action_type"] == "response"]
    proposals = [record for record in records if record["action_type"] == "proposal"]
    for model in MODEL_NAMES:
        response_predictions = [record["predictions"][model] for record in responses]
        proposal_predictions = [record["predictions"][model] for record in proposals]
        result[model] = {
            "response": {
                "count": len(response_predictions),
                "nll": _mean([float(prediction["nll"]) for prediction in response_predictions]),
                "brier": _mean([float(prediction["brier"]) for prediction in response_predictions]),
                "accuracy": _mean([float(bool(prediction["correct"])) for prediction in response_predictions]),
            },
            "proposal": {
                "count": len(proposal_predictions),
                "nll": _mean([float(prediction["nll"]) for prediction in proposal_predictions]),
                "mae": _mean([float(prediction["absolute_error"]) for prediction in proposal_predictions]),
                "rmse": math.sqrt(_mean([float(prediction["squared_error"]) for prediction in proposal_predictions]) or 0.0) if proposal_predictions else None,
                "interval_80_coverage": _mean([float(bool(prediction["interval_80_covered"])) for prediction in proposal_predictions]),
            },
        }
    return result


def _macro_metrics(per_opponent: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for model in MODEL_NAMES:
        result[model] = {}
        for task, metrics in (("response", ("nll", "brier", "accuracy")), ("proposal", ("nll", "mae", "rmse", "interval_80_coverage"))):
            result[model][task] = {"opponent_count": sum(int(opponent["metrics"][model][task]["count"]) > 0 for opponent in per_opponent)}
            for metric in metrics:
                values = [opponent["metrics"][model][task][metric] for opponent in per_opponent if opponent["metrics"][model][task][metric] is not None]
                result[model][task][metric] = _mean([float(value) for value in values])
    return result


def _calibration(records: Sequence[dict[str, Any]], bins: int) -> dict[str, list[dict[str, float | int | None]]]:
    responses = [record for record in records if record["action_type"] == "response"]
    result: dict[str, list[dict[str, float | int | None]]] = {}
    for model in MODEL_NAMES:
        buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
        for record in responses:
            probability = float(record["predictions"][model]["probability"])
            index = min(bins - 1, int(probability * bins))
            buckets[index].append((probability, float(bool(record["actual"]["accepted"]))))
        result[model] = [
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(bucket),
                "mean_prediction": _mean([prediction for prediction, _observed in bucket]),
                "observed_frequency": _mean([observed for _prediction, observed in bucket]),
            }
            for index, bucket in enumerate(buckets)
        ]
    return result


def _expected_calibration_error(calibration: dict[str, list[dict[str, float | int | None]]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for model, buckets in calibration.items():
        total = sum(int(bucket["count"]) for bucket in buckets)
        if total == 0:
            result[model] = None
            continue
        result[model] = sum(
            int(bucket["count"]) * abs(float(bucket["mean_prediction"]) - float(bucket["observed_frequency"]))
            for bucket in buckets
            if int(bucket["count"]) > 0 and bucket["mean_prediction"] is not None and bucket["observed_frequency"] is not None
        ) / total
    return result


def _entropy_by_prior_games(origins: Sequence[dict[str, Any]]) -> list[dict[str, float | int | None]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for origin in origins:
        grouped[int(origin["prior_target_game_count"])].append(origin)
    return [
        {
            "prior_target_game_count": count,
            "origin_count": len(values),
            "response_entropy": _mean([float(value["response_posterior_entropy"]) for value in values]),
            "proposal_entropy": _mean([float(value["proposal_posterior_entropy"]) for value in values]),
        }
        for count, values in sorted(grouped.items())
    ]


def _surprise_series(records: Sequence[dict[str, Any]], window: int) -> list[dict[str, float | int | None]]:
    response_values: deque[float] = deque(maxlen=window)
    proposal_values: deque[float] = deque(maxlen=window)
    result: list[dict[str, float | int | None]] = []
    for index, record in enumerate(records):
        value = float(record["predictions"]["hierarchical_program"]["nll"])
        if record["action_type"] == "response":
            response_values.append(value)
        else:
            proposal_values.append(value)
        result.append(
            {
                "decision_index": index,
                "completed_at": record["completed_at"],
                "response_rolling_nll": _mean(list(response_values)),
                "proposal_rolling_nll": _mean(list(proposal_values)),
                "response_window_count": len(response_values),
                "proposal_window_count": len(proposal_values),
            }
        )
    return result


def _half_change(values: Sequence[float]) -> float | None:
    if len(values) < 4:
        return None
    middle = len(values) // 2
    first = _mean(values[:middle])
    second = _mean(values[middle:])
    return second - first if first is not None and second is not None else None


def _diagnose_opponent(opponent: dict[str, str], records: Sequence[dict[str, Any]], origins: Sequence[dict[str, Any]]) -> dict[str, object]:
    metrics = _metric_block(records)
    response_transfer = None
    proposal_transfer = None
    response_grammar = None
    proposal_grammar = None
    if metrics["hierarchical_program"]["response"]["nll"] is not None and metrics["opponent_program"]["response"]["nll"] is not None:
        response_transfer = float(metrics["hierarchical_program"]["response"]["nll"]) - float(metrics["opponent_program"]["response"]["nll"])
    if metrics["hierarchical_program"]["proposal"]["mae"] is not None and metrics["opponent_program"]["proposal"]["mae"] is not None:
        proposal_transfer = float(metrics["hierarchical_program"]["proposal"]["mae"]) - float(metrics["opponent_program"]["proposal"]["mae"])
    if metrics["hierarchical_program"]["response"]["nll"] is not None and metrics["regularized_tabular"]["response"]["nll"] is not None:
        response_grammar = float(metrics["hierarchical_program"]["response"]["nll"]) - float(metrics["regularized_tabular"]["response"]["nll"])
    if metrics["hierarchical_program"]["proposal"]["mae"] is not None and metrics["regularized_tabular"]["proposal"]["mae"] is not None:
        proposal_grammar = float(metrics["hierarchical_program"]["proposal"]["mae"]) - float(metrics["regularized_tabular"]["proposal"]["mae"])
    response_records = [record for record in records if record["action_type"] == "response"]
    proposal_records = [record for record in records if record["action_type"] == "proposal"]
    response_recent_change = _half_change([float(record["predictions"]["hierarchical_program"]["nll"]) for record in response_records])
    proposal_recent_change = _half_change([float(record["predictions"]["hierarchical_program"]["absolute_error"]) for record in proposal_records])
    ood_rate = _mean([float(bool(record["ood_flags"])) for record in records]) or 0.0
    messaged = [record for record in proposal_records if record["message_act"] != "none"]
    unmessaged = [record for record in proposal_records if record["message_act"] == "none"]
    messaged_error = _mean([float(record["predictions"]["hierarchical_program"]["absolute_error"]) for record in messaged])
    unmessaged_error = _mean([float(record["predictions"]["hierarchical_program"]["absolute_error"]) for record in unmessaged])
    message_gap = messaged_error - unmessaged_error if messaged_error is not None and unmessaged_error is not None else None
    candidates: list[dict[str, object]] = []
    if len(response_records) < 10 or len(proposal_records) < 10:
        candidates.append({"kind": "sparse-effective-evidence", "evidence": {"response_count": len(response_records), "proposal_count": len(proposal_records)}})
    if (response_transfer is not None and response_transfer > 0.05) or (proposal_transfer is not None and proposal_transfer > 0.02):
        candidates.append({"kind": "population-negative-transfer", "evidence": {"response_nll_delta": response_transfer, "proposal_mae_delta": proposal_transfer}})
    if (response_recent_change is not None and response_recent_change > 0.25) or (proposal_recent_change is not None and proposal_recent_change > 0.04):
        candidates.append({"kind": "recent-policy-shift", "evidence": {"response_nll_half_change": response_recent_change, "proposal_mae_half_change": proposal_recent_change}})
    if ood_rate > 0.2:
        candidates.append({"kind": "context-shift", "evidence": {"ood_rate": ood_rate}})
    if len(messaged) >= 3 and message_gap is not None and message_gap > 0.03:
        candidates.append({"kind": "omitted-message-semantics", "evidence": {"messaged_proposal_count": len(messaged), "message_error_gap": message_gap}})
    if ood_rate <= 0.2 and ((response_grammar is not None and response_grammar > 0.05) or (proposal_grammar is not None and proposal_grammar > 0.02)):
        candidates.append({"kind": "candidate-grammar-mismatch", "evidence": {"response_nll_delta_vs_tabular": response_grammar, "proposal_mae_delta_vs_tabular": proposal_grammar}})
    return {
        "opponent": opponent,
        "metrics": metrics,
        "population_transfer": {"response_nll_delta": response_transfer, "proposal_mae_delta": proposal_transfer},
        "program_vs_tabular": {"response_nll_delta": response_grammar, "proposal_mae_delta": proposal_grammar},
        "recent_change": {"response_nll_half_change": response_recent_change, "proposal_mae_half_change": proposal_recent_change},
        "support": {"ood_rate": ood_rate},
        "messages": {"messaged_proposal_count": len(messaged), "unmessaged_proposal_count": len(unmessaged), "message_error_gap": message_gap},
        "posterior": {
            "mean_response_entropy": _mean([float(origin["response_posterior_entropy"]) for origin in origins]),
            "mean_proposal_entropy": _mean([float(origin["proposal_posterior_entropy"]) for origin in origins]),
        },
        "candidate_explanations": candidates,
        "interpretation": "Exploratory residual labels under frozen thresholds; overlapping labels are not causal adjudications.",
    }


class BargainingRollingValidation:
    """Run causally ordered prediction from every eligible post-warmup game."""

    def __init__(
        self,
        *,
        dossier_root: Path,
        output_dir: Path,
        validation_config: ValidationConfig | None = None,
        twin_config: TwinConfig | None = None,
        opponents: set[str] | None = None,
    ) -> None:
        self.dossier_root = dossier_root
        self.output_dir = output_dir
        self.validation_config = validation_config or ValidationConfig()
        self.validation_config.validate()
        self.twin_config = twin_config or TwinConfig()
        self.opponents = opponents
        self.response_programs = response_particles(self.twin_config)
        self.proposal_programs = proposal_particles(self.twin_config)
        self.response_complexities = [particle.complexity for particle in self.response_programs]
        self.proposal_complexities = [particle.complexity for particle in self.proposal_programs]
        self.response_target_prior = _mixed_population_prior([0.0] * len(self.response_programs), row_count=0, complexities=self.response_complexities, config=self.twin_config)
        self.proposal_target_prior = _mixed_population_prior([0.0] * len(self.proposal_programs), row_count=0, complexities=self.proposal_complexities, config=self.twin_config)

    def _eligible(self, games: Sequence[BargainingGameEvidence]) -> dict[str, dict[str, str]]:
        grouped: dict[str, list[BargainingGameEvidence]] = defaultdict(list)
        for game in games:
            grouped[game.opponent_id].append(game)
        return {
            opponent_id: {"id": opponent_id, "name": opponent_games[0].opponent_name}
            for opponent_id, opponent_games in grouped.items()
            if len(opponent_games) >= self.validation_config.min_games and (self.opponents is None or opponent_id in self.opponents or opponent_games[0].opponent_name in self.opponents)
        }

    def _program_weights(
        self,
        *,
        global_response_scores: Sequence[float],
        global_proposal_scores: Sequence[float],
        target_response_scores: Sequence[float],
        target_proposal_scores: Sequence[float],
        global_response_count: int,
        global_proposal_count: int,
        target_response_count: int,
        target_proposal_count: int,
    ) -> dict[str, tuple[list[float], list[float]]]:
        population_response_scores = _subtract_scores(global_response_scores, target_response_scores)
        population_proposal_scores = _subtract_scores(global_proposal_scores, target_proposal_scores)
        population_response_prior = _mixed_population_prior(
            population_response_scores,
            row_count=global_response_count - target_response_count,
            complexities=self.response_complexities,
            config=self.twin_config,
        )
        population_proposal_prior = _mixed_population_prior(
            population_proposal_scores,
            row_count=global_proposal_count - target_proposal_count,
            complexities=self.proposal_complexities,
            config=self.twin_config,
        )
        return {
            "hierarchical_program": (
                _posterior_from_prior(population_response_prior, target_response_scores),
                _posterior_from_prior(population_proposal_prior, target_proposal_scores),
            ),
            "population_program": (population_response_prior, population_proposal_prior),
            "opponent_program": (
                _posterior_from_prior(self.response_target_prior, target_response_scores),
                _posterior_from_prior(self.proposal_target_prior, target_proposal_scores),
            ),
        }

    def _evaluate_game(
        self,
        *,
        game: BargainingGameEvidence,
        global_game_index: int,
        target_game_index: int,
        population_game_count: int,
        target_prior: Sequence[PriorObservation],
        population_prior: Sequence[PriorObservation],
        program_weights: dict[str, tuple[list[float], list[float]]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ordinary_population, ordinary_target = _ordinary_weights(population_prior, target_prior, self.validation_config)
        ordinary_examples = ordinary_population + ordinary_target
        recency_population, recency_target = _recency_weights(
            population_prior,
            target_prior,
            current_global_game_index=global_game_index,
            current_target_game_index=target_game_index,
            config=self.validation_config,
        )
        recency_model = RecencyKernelModel(recency_population + recency_target, self.validation_config)
        logistic_model = LogisticResponseModel.fit(ordinary_examples, l2=self.validation_config.logistic_l2)
        ridge_model = RidgeProposalModel.fit(ordinary_examples, l2=self.validation_config.ridge_l2)
        unconditional_model = UnconditionalEmpiricalModel(ordinary_examples)
        target_rows = [observation.row for observation in target_prior]
        support = _support(target_rows)
        response_entropy = _entropy(program_weights["hierarchical_program"][0])
        proposal_entropy = _entropy(program_weights["hierarchical_program"][1])
        records: list[dict[str, Any]] = []
        for within_game_index, row in enumerate(game.rows):
            record: dict[str, Any] = {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "validation_version": VALIDATION_VERSION,
                "decision_index_within_game": within_game_index,
                "game_id": game.game_id,
                "job_id": game.job_id,
                "job_sha256": game.job_sha256,
                "completed_at": game.completed_at,
                "completion_order": game.completion_order,
                "opponent": {"id": game.opponent_id, "name": game.opponent_name},
                "action_type": row.action_type,
                "prior_target_game_count": target_game_index,
                "prior_population_game_count": population_game_count,
                "ood_flags": _ood_flags(row.context, support),
                "message_act": row.message_act,
                "context": {
                    "round_number": row.context.round_number,
                    "progress": row.context.progress,
                    "complete_information": row.context.complete_information,
                    "horizon_known": row.context.horizon_known,
                    "max_rounds": row.context.max_rounds,
                    "messages_allowed": row.context.messages_allowed,
                    "opponent_player": row.context.opponent_player,
                    "money_to_divide": row.context.money_to_divide,
                    "our_discount": row.context.our_discount,
                    "opponent_discount": row.context.opponent_discount,
                    "previous_opponent_offer_share": row.context.previous_opponent_offer_share,
                    "previous_our_offer_to_opponent_share": row.context.previous_our_offer_to_opponent_share,
                    "previous_opponent_response": row.context.previous_opponent_response,
                },
                "posterior_entropy": {"response": response_entropy, "proposal": proposal_entropy},
                "predictions": {},
            }
            if row.action_type == "response" and row.offered_share is not None and row.accepted is not None:
                record["actual"] = {"offered_share": row.offered_share, "accepted": row.accepted, "decision": row.decision}
                for model in ("hierarchical_program", "population_program", "opponent_program"):
                    probability = _response_prediction(self.response_programs, program_weights[model][0], row.context, float(row.offered_share))
                    record["predictions"][model] = _response_prediction_record(probability, row.accepted)
                record["predictions"]["recency_kernel"] = _response_prediction_record(recency_model.response_probability(row), row.accepted)
                record["predictions"]["regularized_tabular"] = _response_prediction_record(logistic_model.probability(row), row.accepted)
                record["predictions"]["unconditional_empirical"] = _response_prediction_record(unconditional_model.response_probability(), row.accepted)
            elif row.action_type == "proposal" and row.proposal_share is not None:
                observed = float(row.proposal_share)
                record["actual"] = {"proposal_share": observed}
                for model in ("hierarchical_program", "population_program", "opponent_program"):
                    proposal_weights = program_weights[model][1]
                    prediction = _particle_proposal_prediction(self.proposal_programs, proposal_weights, row)
                    exact_nll = -_proposal_log_likelihood(self.proposal_programs, proposal_weights, row)
                    record["predictions"][model] = _proposal_prediction_record(prediction, observed, nll=exact_nll)
                record["predictions"]["recency_kernel"] = _proposal_prediction_record(recency_model.proposal_prediction(row), observed)
                record["predictions"]["regularized_tabular"] = _proposal_prediction_record(ridge_model.prediction(row), observed)
                record["predictions"]["unconditional_empirical"] = _proposal_prediction_record(unconditional_model.proposal_prediction(), observed)
            else:
                raise ValueError(f"unsupported extracted bargaining row in game {game.game_id}")
            records.append(record)
        origin = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_version": VALIDATION_VERSION,
            "game_id": game.game_id,
            "completed_at": game.completed_at,
            "completion_order": game.completion_order,
            "opponent": {"id": game.opponent_id, "name": game.opponent_name},
            "prior_target_game_count": target_game_index,
            "prior_population_game_count": population_game_count,
            "response_posterior_entropy": response_entropy,
            "proposal_posterior_entropy": proposal_entropy,
            "decision_count": len(records),
        }
        return records, origin

    def _roll(self, games: Sequence[BargainingGameEvidence], eligible: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        global_response_scores = [0.0] * len(self.response_programs)
        global_proposal_scores = [0.0] * len(self.proposal_programs)
        global_response_count = 0
        global_proposal_count = 0
        target_response_scores = {opponent_id: [0.0] * len(self.response_programs) for opponent_id in eligible}
        target_proposal_scores = {opponent_id: [0.0] * len(self.proposal_programs) for opponent_id in eligible}
        target_response_count = {opponent_id: 0 for opponent_id in eligible}
        target_proposal_count = {opponent_id: 0 for opponent_id in eligible}
        observations: dict[str, list[PriorObservation]] = defaultdict(list)
        game_counts: dict[str, int] = defaultdict(int)
        records: list[dict[str, Any]] = []
        origins: list[dict[str, Any]] = []
        for global_game_index, game in enumerate(games):
            opponent_id = game.opponent_id
            target_game_index = game_counts[opponent_id]
            if opponent_id in eligible and target_game_index >= self.validation_config.warmup_games:
                population_prior = [observation for other_id, values in observations.items() if other_id != opponent_id for observation in values]
                program_weights = self._program_weights(
                    global_response_scores=global_response_scores,
                    global_proposal_scores=global_proposal_scores,
                    target_response_scores=target_response_scores[opponent_id],
                    target_proposal_scores=target_proposal_scores[opponent_id],
                    global_response_count=global_response_count,
                    global_proposal_count=global_proposal_count,
                    target_response_count=target_response_count[opponent_id],
                    target_proposal_count=target_proposal_count[opponent_id],
                )
                game_records, origin = self._evaluate_game(
                    game=game,
                    global_game_index=global_game_index,
                    target_game_index=target_game_index,
                    population_game_count=global_game_index - target_game_index,
                    target_prior=observations[opponent_id],
                    population_prior=population_prior,
                    program_weights=program_weights,
                )
                records.extend(game_records)
                origins.append(origin)
            response_update = _response_scores(self.response_programs, game.rows)
            proposal_update = _proposal_scores(self.proposal_programs, game.rows)
            _add_scores(global_response_scores, response_update)
            _add_scores(global_proposal_scores, proposal_update)
            game_response_count = sum(row.action_type == "response" for row in game.rows)
            game_proposal_count = sum(row.action_type == "proposal" for row in game.rows)
            global_response_count += game_response_count
            global_proposal_count += game_proposal_count
            if opponent_id in eligible:
                _add_scores(target_response_scores[opponent_id], response_update)
                _add_scores(target_proposal_scores[opponent_id], proposal_update)
                target_response_count[opponent_id] += game_response_count
                target_proposal_count[opponent_id] += game_proposal_count
            observations[opponent_id].extend(PriorObservation(row, global_game_index, target_game_index) for row in game.rows)
            game_counts[opponent_id] += 1
        return records, origins

    @staticmethod
    def _comparison(metrics: dict[str, Any], comparator: str) -> dict[str, float | None]:
        primary = metrics["hierarchical_program"]

        def delta(task: str, metric: str) -> float | None:
            left = primary[task][metric]
            right = metrics[comparator][task][metric]
            return float(left) - float(right) if left is not None and right is not None else None

        return {
            "response_nll_delta": delta("response", "nll"),
            "response_brier_delta": delta("response", "brier"),
            "proposal_nll_delta": delta("proposal", "nll"),
            "proposal_mae_delta": delta("proposal", "mae"),
            "proposal_rmse_delta": delta("proposal", "rmse"),
        }

    def _evaluate(self, records: Sequence[dict[str, Any]], origins: Sequence[dict[str, Any]], eligible: dict[str, dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        records_by_opponent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        origins_by_opponent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_opponent[str(record["opponent"]["id"])].append(record)
        for origin in origins:
            origins_by_opponent[str(origin["opponent"]["id"])].append(origin)
        per_opponent: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for opponent_id, opponent in sorted(eligible.items(), key=lambda item: item[1]["name"].casefold()):
            opponent_records = records_by_opponent[opponent_id]
            metrics = _metric_block(opponent_records)
            per_opponent.append({"opponent": opponent, "origin_count": len(origins_by_opponent[opponent_id]), "decision_count": len(opponent_records), "metrics": metrics})
            diagnostics.append(_diagnose_opponent(opponent, opponent_records, origins_by_opponent[opponent_id]))
        micro = _metric_block(records)
        macro = _macro_metrics(per_opponent)
        comparisons = {
            "micro": {comparator: self._comparison(micro, comparator) for comparator in MODEL_NAMES if comparator != "hierarchical_program"},
            "macro": {comparator: self._comparison(macro, comparator) for comparator in MODEL_NAMES if comparator != "hierarchical_program"},
        }
        response_calibration = _calibration(records, self.validation_config.calibration_bins)
        evaluation = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-bargaining-rolling-origin-evaluation",
            "validation_version": VALIDATION_VERSION,
            "model_version": MODEL_VERSION,
            "frontier": "Every action in a target game is predicted before any row from that game enters target or population evidence.",
            "origin_count": len(origins),
            "decision_count": len(records),
            "opponent_count": len(per_opponent),
            "micro": micro,
            "macro": macro,
            "comparisons": comparisons,
            "per_opponent": per_opponent,
            "response_calibration": response_calibration,
            "response_expected_calibration_error": _expected_calibration_error(response_calibration),
            "entropy_by_prior_games": _entropy_by_prior_games(origins),
            "surprise_series": _surprise_series(records, self.validation_config.surprise_window),
        }
        candidate_counts: dict[str, int] = defaultdict(int)
        for diagnostic in diagnostics:
            for candidate in diagnostic["candidate_explanations"]:
                candidate_counts[str(candidate["kind"])] += 1
        diagnostic_report = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-bargaining-negative-transfer-diagnostics",
            "validation_version": VALIDATION_VERSION,
            "threshold_status": "exploratory-frozen-for-v1",
            "candidate_counts": dict(sorted(candidate_counts.items())),
            "opponents": diagnostics,
            "boundary": "Residual patterns generate overlapping candidate explanations and do not identify hidden causes.",
        }
        return evaluation, diagnostic_report

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty validation directory: {self.output_dir}")
        games, rejected = load_bargaining_corpus(self.dossier_root)
        eligible = self._eligible(games)
        if not eligible:
            raise RuntimeError(f"no opponent has at least {self.validation_config.min_games} bargaining games")
        records, origins = self._roll(games, eligible)
        if not records:
            raise RuntimeError("rolling validation produced no post-warmup decisions")
        evaluation, diagnostics = self._evaluate(records, origins, eligible)
        corpus_receipts = [
            {
                "game_id": game.game_id,
                "opponent_id": game.opponent_id,
                "opponent_name": game.opponent_name,
                "job_id": game.job_id,
                "job_path": game.job_path,
                "job_sha256": game.job_sha256,
                "final_game_sha256": game.final_game_sha256,
            }
            for game in games
        ]
        corpus = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-bargaining-validation-corpus",
            "source_root": str(self.dossier_root.resolve()),
            "game_count": len(games),
            "opponent_count": len({game.opponent_id for game in games}),
            "row_count": sum(len(game.rows) for game in games),
            "eligible_opponents": list(eligible.values()),
            "rejected": list(rejected),
            "receipts": corpus_receipts,
            "corpus_sha256": _sha(corpus_receipts),
        }
        _atomic_json(self.output_dir / "corpus.json", corpus)
        _jsonl(self.output_dir / "predictions.jsonl", records)
        _jsonl(self.output_dir / "origins.jsonl", origins)
        _atomic_json(self.output_dir / "evaluation.json", evaluation)
        _atomic_json(self.output_dir / "diagnostics.json", diagnostics)
        figure_records = write_validation_figures(evaluation=evaluation, diagnostics=diagnostics, records=records, output_dir=self.output_dir / "figures")
        module_path = Path(__file__).resolve()
        project_root = module_path.parents[2]
        source_paths = {
            "validation_module": module_path,
            "twin_module": project_root / "src" / "nommd_arena" / "glee_bargaining_twin.py",
            "plot_module": project_root / "src" / "nommd_arena" / "glee_validation_plots.py",
            "validation_protocol": project_root / "protocols" / "glee-bargaining-validation-v1.md",
            "model_protocol": project_root / "protocols" / "glee-executable-opponent-models-v1.md",
        }
        artifacts = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts.append({"path": str(path.relative_to(self.output_dir)), "sha256": _sha_file(path), "bytes": path.stat().st_size})
        manifest: dict[str, object] = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-bargaining-validation-run",
            "status": "offline-shadow-only",
            "validation_version": VALIDATION_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at": _now(),
            "corpus_sha256": corpus["corpus_sha256"],
            "validation_config": asdict(self.validation_config),
            "twin_config": asdict(self.twin_config),
            "selected_opponents": sorted(self.opponents) if self.opponents is not None else None,
            "implementation_receipts": {
                label: {"path": str(path.relative_to(project_root)), "sha256": _sha_file(path)} for label, path in source_paths.items()
            },
            "artifacts": artifacts,
            "figures": figure_records,
            "summary": {
                "game_count": corpus["game_count"],
                "eligible_opponent_count": len(eligible),
                "origin_count": len(origins),
                "decision_count": len(records),
                "micro_comparisons": evaluation["comparisons"]["micro"],
                "macro_comparisons": evaluation["comparisons"]["macro"],
                "diagnostic_candidate_counts": diagnostics["candidate_counts"],
            },
            "promotion": "No live-policy effect; a prospective pre-outcome shadow suffix and separate review remain mandatory.",
        }
        manifest["manifest_sha256"] = _sha(manifest)
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return {"manifest": manifest, "micro": evaluation["micro"], "macro": evaluation["macro"], "diagnostics": diagnostics["candidate_counts"]}
