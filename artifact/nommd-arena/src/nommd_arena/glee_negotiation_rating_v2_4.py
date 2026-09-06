"""Dependency-free rating-delta surrogate for GLEE Negotiation v2.4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_VERSION = "negotiation-rating-surrogate-v2.4"
VISIBLE_FEATURES = (
    "bias",
    "agreement",
    "walked_away",
    "our_walkaway",
    "seller",
    "complete_information",
    "known_horizon",
    "one_round",
    "round_phase",
    "own_payoff_ratio",
    "own_payoff_ratio_squared",
)
COMPLETE_FEATURES = tuple(name for name in VISIBLE_FEATURES if name != "complete_information") + ("own_realized_share",)
ORACLE_FEATURES = (*VISIBLE_FEATURES, "own_realized_share")


def _other_player(player: str) -> str:
    try:
        return {"player_1": "player_2", "player_2": "player_1"}[player]
    except KeyError as error:
        raise ValueError(f"unsupported player identity: {player}") from error


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else default


def _round_phase(state: Mapping[str, object], round_number: int | None = None) -> float:
    current = max(1, int(round_number or state.get("round") or 1))
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True and isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 1:
        return min(1.0, max(0.0, (current - 1) / (maximum - 1)))
    if state.get("horizon_known") is True and maximum == 1:
        return 1.0
    return 1.0 - math.exp(-max(0, current - 1) / 12.0)


def scenario_features(
    game: Mapping[str, object],
    *,
    outcome: str,
    own_payoff: float,
    opponent_payoff: float | None,
    round_number: int | None = None,
) -> dict[str, float]:
    """Build only features available from the game state and the explicitly supplied scenario."""
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    our_value = max(1e-9, _number(state.get(f"{our_player}_value"), 1.0))
    role = str(state.get(f"{our_player}_role") or "")
    normalized_outcome = outcome.casefold()
    agreement = float(normalized_outcome == "agreement")
    total_payoff = own_payoff + opponent_payoff if opponent_payoff is not None else 0.0
    own_share = own_payoff / total_payoff if opponent_payoff is not None and total_payoff > 0 else 0.0
    payoff_advantage = (own_payoff - opponent_payoff) / total_payoff if opponent_payoff is not None and total_payoff > 0 else 0.0
    own_payoff_ratio = math.tanh(own_payoff / our_value)
    maximum = state.get("max_rounds")
    result = game.get("result") if isinstance(game.get("result"), Mapping) else state.get("result") if isinstance(state.get("result"), Mapping) else {}
    walked_by = str(result.get("walked_away_by") or "") if isinstance(result, Mapping) else ""
    return {
        "bias": 1.0,
        "agreement": agreement,
        "no_deal": float(normalized_outcome == "no_deal"),
        "walked_away": float(normalized_outcome == "walked_away"),
        "our_walkaway": float(normalized_outcome == "walked_away" and walked_by == our_player),
        "seller": float(role == "seller"),
        "complete_information": float(state.get("complete_information") is True),
        "known_horizon": float(state.get("horizon_known") is True),
        "one_round": float(maximum == 1),
        "round_phase": _round_phase(state, round_number),
        "own_payoff_ratio": own_payoff_ratio,
        "own_payoff_ratio_squared": own_payoff_ratio**2,
        "agreement_own_payoff_ratio": agreement * own_payoff_ratio,
        "own_realized_share": own_share,
        "payoff_advantage": payoff_advantage,
        "agreement_payoff_advantage": agreement * payoff_advantage,
        "opponent_value_visible": float(state.get("complete_information") is True and state.get(f"{opponent_player}_value") is not None),
    }


def final_game_sample(final_game: Mapping[str, object], rating_delta: float) -> dict[str, object]:
    """Extract one authenticated terminal outcome and both deployable and oracle feature vectors."""
    state = final_game.get("game_state") if isinstance(final_game.get("game_state"), Mapping) else {}
    result = final_game.get("result") if isinstance(final_game.get("result"), Mapping) else state.get("result") if isinstance(state.get("result"), Mapping) else {}
    our_player = str(final_game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    own_payoff = _number(result.get(f"{our_player}_payoff"))
    opponent_payoff = _number(result.get(f"{opponent_player}_payoff"))
    outcome = str(result.get("outcome") or "")
    features = scenario_features(final_game, outcome=outcome, own_payoff=own_payoff, opponent_payoff=opponent_payoff)
    return {
        "game_id": str(final_game.get("game_id") or ""),
        "rating_delta": float(rating_delta),
        "complete_information": state.get("complete_information") is True,
        "outcome": outcome,
        "own_payoff": own_payoff,
        "opponent_payoff": opponent_payoff,
        "visible_features": {name: features[name] for name in VISIBLE_FEATURES},
        "complete_features": {name: features[name] for name in COMPLETE_FEATURES},
        "oracle_features": {name: features[name] for name in ORACLE_FEATURES},
    }


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...]:
    size = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] += 1e-9
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column], strict=True)]
    return tuple(augmented[index][-1] for index in range(size))


@dataclass(frozen=True)
class RidgeRatingModel:
    """One bounded-feature ridge regression with an unpenalized intercept."""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    ridge_lambda: float
    training_count: int

    def predict(self, features: Mapping[str, float]) -> float:
        return sum(coefficient * float(features.get(name, 0.0)) for name, coefficient in zip(self.feature_names, self.coefficients, strict=True))

    def as_dict(self) -> dict[str, object]:
        return {"feature_names": list(self.feature_names), "coefficients": list(self.coefficients), "ridge_lambda": self.ridge_lambda, "training_count": self.training_count}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RidgeRatingModel:
        return cls(feature_names=tuple(str(name) for name in value["feature_names"]), coefficients=tuple(float(number) for number in value["coefficients"]), ridge_lambda=float(value["ridge_lambda"]), training_count=int(value["training_count"]))


def fit_ridge(samples: Sequence[Mapping[str, object]], *, feature_key: str, feature_names: Sequence[str], ridge_lambda: float) -> RidgeRatingModel:
    """Fit deterministic normal-equation ridge regression without adding a numerical dependency."""
    names = tuple(feature_names)
    if not samples:
        raise ValueError("rating surrogate requires at least one sample")
    size = len(names)
    matrix = [[0.0 for _column in range(size)] for _row in range(size)]
    vector = [0.0 for _column in range(size)]
    for sample in samples:
        features = sample.get(feature_key)
        if not isinstance(features, Mapping):
            raise ValueError(f"rating sample lacks {feature_key}")
        row = [float(features.get(name, 0.0)) for name in names]
        target = float(sample["rating_delta"])
        for left in range(size):
            vector[left] += row[left] * target
            for right in range(size):
                matrix[left][right] += row[left] * row[right]
    for index, name in enumerate(names):
        if name != "bias":
            matrix[index][index] += ridge_lambda
    return RidgeRatingModel(feature_names=names, coefficients=_solve(matrix, vector), ridge_lambda=ridge_lambda, training_count=len(samples))


def regression_metrics(model: RidgeRatingModel, samples: Sequence[Mapping[str, object]], *, feature_key: str) -> dict[str, float | int | None]:
    """Score point errors, direction, and linear association on one chronological block."""
    if not samples:
        return {"count": 0, "mae": None, "rmse": None, "sign_accuracy": None, "correlation": None}
    actual = [float(sample["rating_delta"]) for sample in samples]
    predicted = [model.predict(sample[feature_key]) for sample in samples]
    errors = [estimate - observed for estimate, observed in zip(predicted, actual, strict=True)]
    mean_actual = sum(actual) / len(actual)
    mean_predicted = sum(predicted) / len(predicted)
    covariance = sum((left - mean_actual) * (right - mean_predicted) for left, right in zip(actual, predicted, strict=True))
    variance_actual = sum((value - mean_actual) ** 2 for value in actual)
    variance_predicted = sum((value - mean_predicted) ** 2 for value in predicted)
    correlation = covariance / math.sqrt(variance_actual * variance_predicted) if variance_actual > 0 and variance_predicted > 0 else None
    return {
        "count": len(samples),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error**2 for error in errors) / len(errors)),
        "sign_accuracy": sum((estimate >= 0) == (observed >= 0) for estimate, observed in zip(predicted, actual, strict=True)) / len(actual),
        "correlation": correlation,
    }


@dataclass(frozen=True)
class NegotiationRatingSurrogate:
    """Deployable visible-state models plus their frozen retrospective validation receipt."""

    visible: RidgeRatingModel
    complete: RidgeRatingModel
    validation: dict[str, object]

    def predict_scenario(self, game: Mapping[str, object], *, outcome: str, own_payoff: float, opponent_payoff: float | None, round_number: int | None = None) -> dict[str, object]:
        features = scenario_features(game, outcome=outcome, own_payoff=own_payoff, opponent_payoff=opponent_payoff, round_number=round_number)
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        use_complete = state.get("complete_information") is True and opponent_payoff is not None
        model = self.complete if use_complete else self.visible
        return {"estimated_rating_delta": round(model.predict(features), 4), "model": "complete-visible-payoffs" if use_complete else "visible-state-only", "ridge_lambda": model.ridge_lambda}

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION, "visible": self.visible.as_dict(), "complete": self.complete.as_dict(), "validation": self.validation, "boundary": "Retrospective rating-delta surrogate only; it omits opponent rating and hidden reservation values, supplies no action authority, and cannot identify a causal game-policy effect."}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> NegotiationRatingSurrogate:
        if value.get("schema_version") != SCHEMA_VERSION or value.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported Negotiation rating surrogate")
        return cls(visible=RidgeRatingModel.from_dict(value["visible"]), complete=RidgeRatingModel.from_dict(value["complete"]), validation=dict(value.get("validation") or {}))
