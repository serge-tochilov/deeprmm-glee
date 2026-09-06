"""Offline reconstruction of GLEE displayed-rating changes from immutable game and public receipts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .glee_activity_eda import GLEE_FAMILIES, _file_digest, _read_only_database
from .glee_effective_events import EffectiveEvent, derive_effective_events
from .glee_negotiation_rating_v2_4 import RidgeRatingModel, fit_ridge
from .glee_semantics import model_static_game_context


JOINT_RATING_CONTRACT = "glee-joint-rating-reconstruction-v1"
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
ETA_SCHEDULES = (
    {"name": "constant-0.002", "kind": "constant", "floor": 0.002},
    {"name": "piecewise-linear-120", "kind": "piecewise-linear", "start": 0.01, "floor": 0.002, "decay_games": 120.0},
    {"name": "exponential-tau-15", "kind": "exponential", "start": 0.01, "floor": 0.002, "tau": 15.0},
    {"name": "exponential-tau-20", "kind": "exponential", "start": 0.01, "floor": 0.002, "tau": 20.0},
    {"name": "exponential-tau-30", "kind": "exponential", "start": 0.01, "floor": 0.002, "tau": 30.0},
    {"name": "reciprocal-tau-1", "kind": "reciprocal", "start": 0.01, "floor": 0.002, "tau": 1.0},
    {"name": "reciprocal-tau-3", "kind": "reciprocal", "start": 0.01, "floor": 0.002, "tau": 3.0},
)
BASE_FEATURES = (
    "bias",
    "target_player_1",
    "target_seller",
    "target_buyer",
    "complete_information",
    "known_horizon",
    "messages_allowed",
    "agreement",
    "no_deal",
    "walked_away",
    "timeout",
    "ordinary_completion",
    "target_walked_away",
    "round_phase",
    "round_phase_squared",
    "own_payoff_ratio",
    "own_payoff_ratio_squared",
    "own_payoff_ratio_cubed",
    "opponent_payoff_ratio",
    "payoff_advantage_ratio",
    "own_nonpositive_payoff",
    "bargaining_self_discount",
    "bargaining_opponent_discount",
    "bargaining_log_pool",
    "negotiation_own_value_ratio",
    "negotiation_opponent_value_ratio",
    "negotiation_opponent_value_known",
    "persuasion_quality_probability",
    "persuasion_u_price_ratio",
    "persuasion_v_price_ratio",
    "persuasion_total_rounds_scaled",
    "persuasion_seller_knows_values",
    "persuasion_binary_message",
    "completion_epoch_days",
    "opponent_pregame_rating_known",
    "opponent_pregame_rating_scaled",
)
DIRECT_FEATURES = (*BASE_FEATURES, "pregame_rating_scaled", "pregame_rating_squared", "log_game_count", "display_shrinkage_gap", "payoff_rating_interaction")
STRUCTURAL_FEATURES = BASE_FEATURES
RATING_FEATURES = ("pregame_rating_scaled", "pregame_rating_squared", "log_game_count", "display_shrinkage_gap", "payoff_rating_interaction")
FAMILY_BASE_FEATURES = {
    "bargaining": (
        "bias", "target_player_1", "complete_information", "known_horizon", "messages_allowed", "agreement", "no_deal", "walked_away", "timeout", "target_walked_away", "round_phase", "round_phase_squared", "own_payoff_ratio", "own_payoff_ratio_squared", "own_payoff_ratio_cubed", "opponent_payoff_ratio", "payoff_advantage_ratio", "own_nonpositive_payoff", "bargaining_self_discount", "bargaining_opponent_discount", "bargaining_log_pool", "completion_epoch_days", "opponent_pregame_rating_known", "opponent_pregame_rating_scaled",
    ),
    "negotiation": (
        "bias", "target_seller", "complete_information", "known_horizon", "messages_allowed", "agreement", "no_deal", "walked_away", "timeout", "target_walked_away", "round_phase", "round_phase_squared", "own_payoff_ratio", "own_payoff_ratio_squared", "own_payoff_ratio_cubed", "opponent_payoff_ratio", "payoff_advantage_ratio", "own_nonpositive_payoff", "negotiation_own_value_ratio", "negotiation_opponent_value_ratio", "negotiation_opponent_value_known", "completion_epoch_days", "opponent_pregame_rating_known", "opponent_pregame_rating_scaled",
    ),
    "persuasion": (
        "bias", "target_seller", "timeout", "ordinary_completion", "round_phase", "own_payoff_ratio", "own_payoff_ratio_squared", "own_payoff_ratio_cubed", "opponent_payoff_ratio", "payoff_advantage_ratio", "own_nonpositive_payoff", "persuasion_quality_probability", "persuasion_u_price_ratio", "persuasion_v_price_ratio", "persuasion_total_rounds_scaled", "persuasion_seller_knows_values", "persuasion_binary_message", "completion_epoch_days", "opponent_pregame_rating_known", "opponent_pregame_rating_scaled",
    ),
}
FAMILY_DIRECT_FEATURES = {family: (*features, *RATING_FEATURES) for family, features in FAMILY_BASE_FEATURES.items()}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _iso_to_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player identity: {player}")


def display_to_raw(display_rating: float, games_played: int) -> float:
    """Invert GLEE's published sample-size display shrinkage."""
    if games_played <= 0:
        if games_played == 0 and abs(display_rating - 1000.0) < 1e-9:
            return 1000.0
        raise ValueError("display inversion requires a positive game count")
    return 1000.0 + (display_rating - 1000.0) * (games_played + 30.0) / games_played


def raw_to_display(raw_rating: float, games_played: int) -> float:
    """Apply GLEE's published sample-size display shrinkage."""
    if games_played < 0:
        raise ValueError("game count cannot be negative")
    if games_played == 0:
        return 1000.0
    return 1000.0 + games_played / (games_played + 30.0) * (raw_rating - 1000.0)


def implied_adjusted_percentile(pre_display: float, post_display: float, games_before: int, eta: float) -> float:
    """Invert one unclamped clean public update under a candidate learning rate."""
    if eta <= 0:
        raise ValueError("learning rate must be positive")
    raw_before = display_to_raw(pre_display, games_before)
    raw_after = display_to_raw(post_display, games_before + 1)
    game_rating = raw_before + (raw_after - raw_before) / eta
    return 0.5 + (game_rating - 2000.0) / 8000.0


def predict_display_delta(pre_display: float, games_before: int, adjusted_percentile: float, eta: float) -> float:
    """Propagate one bounded adjusted-percentile prediction through the published rating equations."""
    if eta <= 0:
        raise ValueError("learning rate must be positive")
    raw_before = display_to_raw(pre_display, games_before)
    bounded_percentile = min(1.0, max(0.0, adjusted_percentile))
    game_rating = 2000.0 + 8000.0 * (bounded_percentile - 0.5)
    raw_after = min(5000.0, max(100.0, raw_before + eta * (game_rating - raw_before)))
    return raw_to_display(raw_after, games_before + 1) - pre_display


def eta_for_game_count(schedule: Mapping[str, object], games_before: int) -> float:
    """Evaluate one declared candidate schedule at the pregame count."""
    if games_before < 0:
        raise ValueError("game count cannot be negative")
    kind = str(schedule["kind"])
    floor = float(schedule["floor"])
    if kind == "constant":
        return floor
    start = float(schedule["start"])
    if kind == "piecewise-linear":
        progress = min(1.0, games_before / float(schedule["decay_games"]))
        return start + progress * (floor - start)
    if kind == "exponential":
        return floor + (start - floor) * math.exp(-games_before / float(schedule["tau"]))
    if kind == "reciprocal":
        return floor + (start - floor) / (1.0 + games_before / float(schedule["tau"]))
    raise ValueError(f"unsupported learning-rate schedule: {kind}")


def _round_phase(state: Mapping[str, object], result: Mapping[str, object]) -> tuple[float, int, int | None]:
    current = int(result.get("agreed_round") or result.get("rounds_played") or state.get("round") or 1)
    maximum_value = result.get("rounds_total") or state.get("total_rounds") or state.get("max_rounds")
    maximum = int(maximum_value) if isinstance(maximum_value, int) and not isinstance(maximum_value, bool) and maximum_value > 0 else None
    if maximum is not None and maximum > 1:
        phase = min(1.0, max(0.0, (current - 1.0) / (maximum - 1.0)))
    elif maximum == 1:
        phase = 1.0
    else:
        phase = 1.0 - math.exp(-max(0, current - 1) / 12.0)
    return phase, current, maximum


def _terminal_view(payload: Mapping[str, object], perspective_player: str) -> dict[str, object]:
    """Project one terminal game from either player's perspective without inventing hidden configuration values."""
    family = str(payload.get("game_family") or "")
    if family not in GLEE_FAMILIES:
        raise ValueError(f"unsupported game family: {family}")
    opponent_player = _other_player(perspective_player)
    state = payload.get("game_state") if isinstance(payload.get("game_state"), Mapping) else {}
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else state.get("result") if isinstance(state.get("result"), Mapping) else {}
    own_payoff = _number(result.get(f"{perspective_player}_payoff"))
    opponent_payoff = _number(result.get(f"{opponent_player}_payoff"))
    outcome = str(result.get("outcome") or "unknown").casefold()
    role = str(state.get(f"{perspective_player}_role") or ("seller" if family == "persuasion" and perspective_player == "player_1" else "buyer" if family == "persuasion" else perspective_player)).casefold()
    phase, rounds_played, rounds_total = _round_phase(state, result)
    agreed_price = _optional_number(result.get("agreed_price"))
    own_value = _optional_number(state.get(f"{perspective_player}_value"))
    opponent_value = _optional_number(state.get(f"{opponent_player}_value"))
    if family == "negotiation" and outcome == "agreement" and agreed_price is not None:
        if role == "seller":
            own_value = own_value if own_value is not None else agreed_price - own_payoff
            opponent_value = opponent_value if opponent_value is not None else agreed_price + opponent_payoff
        elif role == "buyer":
            own_value = own_value if own_value is not None else agreed_price + own_payoff
            opponent_value = opponent_value if opponent_value is not None else agreed_price - opponent_payoff
    if family == "bargaining":
        scale = max(1.0, abs(_number(state.get("money_to_divide"), 1.0)))
    elif family == "negotiation":
        scale = max(1.0, *(abs(value) for value in (own_payoff, opponent_payoff, agreed_price or 0.0, own_value or 0.0, opponent_value or 0.0)))
    else:
        unit_scale = max(1.0, abs(_number(state.get("product_price"), 1.0)), abs(_number(state.get("u"))), abs(_number(state.get("v"))))
        scale = unit_scale * max(1, rounds_total or rounds_played)
    own_ratio = math.tanh(own_payoff / scale)
    opponent_ratio = math.tanh(opponent_payoff / scale)
    walked_by = str(result.get("walked_away_by") or "")
    context_payload = dict(payload)
    context_payload["your_player"] = perspective_player
    static_context = model_static_game_context(context_payload)
    price = max(1.0, abs(_number(state.get("product_price"), 1.0)))
    self_discount = _number(state.get("delta_1" if perspective_player == "player_1" else "delta_2")) if family == "bargaining" else 0.0
    opponent_discount = _number(state.get("delta_2" if perspective_player == "player_1" else "delta_1")) if family == "bargaining" else 0.0
    seller_knows = state.get("seller_knows_buyer_values", state.get("is_seller_know_cv")) is True
    features = {name: 0.0 for name in BASE_FEATURES}
    features.update(
        {
            "bias": 1.0,
            "target_player_1": float(perspective_player == "player_1"),
            "target_seller": float(role == "seller"),
            "target_buyer": float(role == "buyer"),
            "complete_information": float(state.get("complete_information") is True),
            "known_horizon": float(state.get("horizon_known") is True),
            "messages_allowed": float(state.get("messages_allowed") is True),
            "agreement": float(outcome == "agreement"),
            "no_deal": float(outcome == "no_deal"),
            "walked_away": float(outcome == "walked_away"),
            "timeout": float(outcome == "timeout"),
            "ordinary_completion": float(outcome == "completed"),
            "target_walked_away": float(outcome == "walked_away" and walked_by == perspective_player),
            "round_phase": phase,
            "round_phase_squared": phase * phase,
            "own_payoff_ratio": own_ratio,
            "own_payoff_ratio_squared": own_ratio * own_ratio,
            "own_payoff_ratio_cubed": own_ratio * own_ratio * own_ratio,
            "opponent_payoff_ratio": opponent_ratio,
            "payoff_advantage_ratio": math.tanh((own_payoff - opponent_payoff) / scale),
            "own_nonpositive_payoff": float(own_payoff <= 0.0),
            "bargaining_self_discount": self_discount,
            "bargaining_opponent_discount": opponent_discount,
            "bargaining_log_pool": math.log10(max(1.0, abs(_number(state.get("money_to_divide"), 1.0)))) / 7.0 if family == "bargaining" else 0.0,
            "negotiation_own_value_ratio": math.tanh((own_value or 0.0) / scale) if family == "negotiation" and own_value is not None else 0.0,
            "negotiation_opponent_value_ratio": math.tanh((opponent_value or 0.0) / scale) if family == "negotiation" and opponent_value is not None else 0.0,
            "negotiation_opponent_value_known": float(family == "negotiation" and opponent_value is not None),
            "persuasion_quality_probability": _number(state.get("p")) if family == "persuasion" else 0.0,
            "persuasion_u_price_ratio": math.tanh(_number(state.get("u")) / price) if family == "persuasion" else 0.0,
            "persuasion_v_price_ratio": math.tanh(_number(state.get("v")) / price) if family == "persuasion" else 0.0,
            "persuasion_total_rounds_scaled": min(2.0, (rounds_total or rounds_played) / 50.0) if family == "persuasion" else 0.0,
            "persuasion_seller_knows_values": float(family == "persuasion" and seller_knows),
            "persuasion_binary_message": float(family == "persuasion" and str(state.get("seller_message_type") or "").casefold() == "binary"),
        }
    )
    return {
        "family": family,
        "perspective_player": perspective_player,
        "opponent_player": opponent_player,
        "role": role,
        "outcome": outcome,
        "own_payoff": own_payoff,
        "opponent_payoff": opponent_payoff,
        "rounds_played": rounds_played,
        "rounds_total": rounds_total,
        "observed_static_context": static_context,
        "observed_configuration_sha256": _digest(static_context),
        "base_features": features,
    }


def _sample_features(view: Mapping[str, object], target_event: EffectiveEvent, opponent_event: EffectiveEvent | None, *, completed_at: float, epoch_origin: float) -> tuple[dict[str, float], dict[str, float]]:
    if target_event.previous_rating_anchor is None or target_event.current_rating is None:
        raise ValueError("rating sample lacks public rating anchors")
    base = {name: float(view["base_features"][name]) for name in BASE_FEATURES}
    base["completion_epoch_days"] = (completed_at - epoch_origin) / 86400.0
    if opponent_event is not None and opponent_event.previous_rating_anchor is not None:
        base["opponent_pregame_rating_known"] = 1.0
        base["opponent_pregame_rating_scaled"] = (opponent_event.previous_rating_anchor - 2000.0) / 1000.0
    direct = dict(base)
    pregame_scaled = (target_event.previous_rating_anchor - 2000.0) / 1000.0
    direct.update(
        {
            "pregame_rating_scaled": pregame_scaled,
            "pregame_rating_squared": pregame_scaled * pregame_scaled,
            "log_game_count": math.log1p(target_event.previous_high_water) / 10.0,
            "display_shrinkage_gap": 30.0 / (target_event.previous_high_water + 30.0),
            "payoff_rating_interaction": pregame_scaled * base["own_payoff_ratio"],
        }
    )
    return direct, base


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _correlation(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(actual) < 2:
        return None
    mean_actual = statistics.fmean(actual)
    mean_predicted = statistics.fmean(predicted)
    covariance = sum((left - mean_actual) * (right - mean_predicted) for left, right in zip(actual, predicted, strict=True))
    variance_actual = sum((value - mean_actual) ** 2 for value in actual)
    variance_predicted = sum((value - mean_predicted) ** 2 for value in predicted)
    return covariance / math.sqrt(variance_actual * variance_predicted) if variance_actual > 0 and variance_predicted > 0 else None


def prediction_metrics(actual: Sequence[float], predicted: Sequence[float], *, interval_radius_80: float | None = None, interval_radius_95: float | None = None) -> dict[str, object]:
    """Score point predictions and optional symmetric residual intervals."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    if not actual:
        return {"count": 0, "mae": None, "rmse": None, "mean_error": None, "sign_accuracy": None, "correlation": None, "interval_80_coverage": None, "interval_95_coverage": None}
    errors = [estimate - observed for estimate, observed in zip(predicted, actual, strict=True)]
    return {
        "count": len(actual),
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "mean_error": statistics.fmean(errors),
        "sign_accuracy": statistics.fmean(float((estimate >= 0.0) == (observed >= 0.0)) for estimate, observed in zip(predicted, actual, strict=True)),
        "correlation": _correlation(actual, predicted),
        "interval_80_radius": interval_radius_80,
        "interval_80_coverage": statistics.fmean(float(abs(error) <= interval_radius_80) for error in errors) if interval_radius_80 is not None else None,
        "interval_95_radius": interval_radius_95,
        "interval_95_coverage": statistics.fmean(float(abs(error) <= interval_radius_95) for error in errors) if interval_radius_95 is not None else None,
    }


def chronological_game_splits(samples: Sequence[Mapping[str, object]], *, train_fraction: float, calibration_fraction: float) -> dict[str, str]:
    """Assign whole games to family-stratified chronological train, calibration, and test blocks."""
    if train_fraction <= 0 or calibration_fraction <= 0 or train_fraction + calibration_fraction >= 1:
        raise ValueError("chronological split fractions must leave positive train, calibration, and test blocks")
    by_family: dict[str, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        game_id = str(sample["game_id"])
        family = str(sample["family"])
        completed_at = float(sample["completed_at_timestamp"])
        by_family[family][game_id] = min(completed_at, by_family[family].get(game_id, completed_at))
    assignments: dict[str, str] = {}
    for family, games in by_family.items():
        ordered = sorted(games, key=lambda game_id: (games[game_id], game_id))
        if len(ordered) < 5:
            raise ValueError(f"{family} needs at least 5 games for a 3-way chronological split")
        train_end = max(1, min(len(ordered) - 2, math.floor(train_fraction * len(ordered))))
        calibration_end = max(train_end + 1, min(len(ordered) - 1, math.floor((train_fraction + calibration_fraction) * len(ordered))))
        for index, game_id in enumerate(ordered):
            assignments[game_id] = "train" if index < train_end else "calibration" if index < calibration_end else "test"
    return assignments


def _model_rows(samples: Sequence[Mapping[str, object]], *, split: str, family: str, feature_key: str, target_key: str = "rating_delta") -> list[dict[str, object]]:
    return [{"rating_delta": float(sample[target_key]), "features": sample[feature_key], "sample": sample} for sample in samples if sample["split"] == split and sample["family"] == family]


def _metric_slices(rows: Sequence[Mapping[str, object]], predictions: Sequence[float], *, interval_80: float | None, interval_95: float | None) -> dict[str, object]:
    output: dict[str, object] = {"all": prediction_metrics([float(row["rating_delta"]) for row in rows], predictions, interval_radius_80=interval_80, interval_radius_95=interval_95)}
    for perspective in ("self", "opponent"):
        indexes = [index for index, row in enumerate(rows) if row["sample"]["target_scope"] == perspective]
        output[perspective] = prediction_metrics([float(rows[index]["rating_delta"]) for index in indexes], [predictions[index] for index in indexes], interval_radius_80=interval_80, interval_radius_95=interval_95)
    return output


def fit_direct_models(samples: Sequence[Mapping[str, object]]) -> tuple[dict[str, RidgeRatingModel], dict[str, object]]:
    """Select one family-specific direct ridge model on calibration and score the untouched suffix."""
    models: dict[str, RidgeRatingModel] = {}
    report: dict[str, object] = {"families": {}}
    pooled_actual: list[float] = []
    pooled_predicted: list[float] = []
    pooled_zero: list[float] = []
    pooled_mean: list[float] = []
    for family in GLEE_FAMILIES:
        train = _model_rows(samples, split="train", family=family, feature_key="direct_features")
        calibration = _model_rows(samples, split="calibration", family=family, feature_key="direct_features")
        test = _model_rows(samples, split="test", family=family, feature_key="direct_features")
        candidates = []
        for ridge_lambda in RIDGE_GRID:
            model = fit_ridge(train, feature_key="features", feature_names=FAMILY_DIRECT_FEATURES[family], ridge_lambda=ridge_lambda)
            predictions = [model.predict(row["features"]) for row in calibration]
            metrics = prediction_metrics([float(row["rating_delta"]) for row in calibration], predictions)
            candidates.append((float(metrics["mae"]), float(metrics["rmse"]), ridge_lambda, model, metrics))
        candidates.sort(key=lambda row: row[:3])
        _mae, _rmse, ridge_lambda, model, calibration_metrics = candidates[0]
        models[family] = model
        calibration_residuals = [abs(model.predict(row["features"]) - float(row["rating_delta"])) for row in calibration]
        radius_80 = _quantile(calibration_residuals, 0.8)
        radius_95 = _quantile(calibration_residuals, 0.95)
        predictions = [model.predict(row["features"]) for row in test]
        actual = [float(row["rating_delta"]) for row in test]
        training_mean = statistics.fmean(float(row["rating_delta"]) for row in train)
        report["families"][family] = {
            "ridge_lambda": ridge_lambda,
            "counts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
            "calibration": calibration_metrics,
            "test": _metric_slices(test, predictions, interval_80=radius_80, interval_95=radius_95),
            "test_zero_baseline": prediction_metrics(actual, [0.0] * len(actual)),
            "test_training_mean_baseline": prediction_metrics(actual, [training_mean] * len(actual)),
            "model": model.as_dict(),
        }
        pooled_actual.extend(actual)
        pooled_predicted.extend(predictions)
        pooled_zero.extend([0.0] * len(actual))
        pooled_mean.extend([training_mean] * len(actual))
    report["pooled_test"] = prediction_metrics(pooled_actual, pooled_predicted)
    report["pooled_test_zero_baseline"] = prediction_metrics(pooled_actual, pooled_zero)
    report["pooled_test_training_mean_baseline"] = prediction_metrics(pooled_actual, pooled_mean)
    return models, report


def _structural_target(sample: Mapping[str, object], schedule: Mapping[str, object]) -> float:
    games_before = int(sample["pregame_game_count"])
    return implied_adjusted_percentile(float(sample["pregame_display_rating"]), float(sample["postgame_display_rating"]), games_before, eta_for_game_count(schedule, games_before))


def _structural_predictions(model: RidgeRatingModel, rows: Sequence[Mapping[str, object]], schedule: Mapping[str, object]) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    percentiles: list[float] = []
    for row in rows:
        sample = row["sample"]
        percentile = model.predict(row["features"])
        percentiles.append(percentile)
        games_before = int(sample["pregame_game_count"])
        deltas.append(predict_display_delta(float(sample["pregame_display_rating"]), games_before, percentile, eta_for_game_count(schedule, games_before)))
    return deltas, percentiles


def fit_structural_models(samples: Sequence[Mapping[str, object]]) -> tuple[dict[str, RidgeRatingModel], dict[str, object]]:
    """Select a shared high-count learning rate and family-specific adjusted-percentile surfaces."""
    candidates: list[tuple[float, float, dict[str, RidgeRatingModel], dict[str, object]]] = []
    for schedule in ETA_SCHEDULES:
        family_models: dict[str, RidgeRatingModel] = {}
        family_report: dict[str, object] = {}
        pooled_actual: list[float] = []
        pooled_predicted: list[float] = []
        for family in GLEE_FAMILIES:
            train = _model_rows(samples, split="train", family=family, feature_key="structural_features")
            calibration = _model_rows(samples, split="calibration", family=family, feature_key="structural_features")
            train_targets = [{**row, "rating_delta": _structural_target(row["sample"], schedule)} for row in train]
            model_candidates = []
            for ridge_lambda in RIDGE_GRID:
                model = fit_ridge(train_targets, feature_key="features", feature_names=FAMILY_BASE_FEATURES[family], ridge_lambda=ridge_lambda)
                predictions, percentiles = _structural_predictions(model, calibration, schedule)
                metrics = prediction_metrics([float(row["rating_delta"]) for row in calibration], predictions)
                clip_fraction = statistics.fmean(float(value < 0.0 or value > 1.0) for value in percentiles) if percentiles else 0.0
                model_candidates.append((float(metrics["mae"]), float(metrics["rmse"]), clip_fraction, ridge_lambda, model, metrics))
            model_candidates.sort(key=lambda row: row[:4])
            _mae, _rmse, clip_fraction, ridge_lambda, model, metrics = model_candidates[0]
            family_models[family] = model
            family_report[family] = {"ridge_lambda": ridge_lambda, "calibration": metrics, "calibration_predicted_percentile_outside_unit_interval": clip_fraction}
            predictions, _percentiles = _structural_predictions(model, calibration, schedule)
            pooled_actual.extend(float(row["rating_delta"]) for row in calibration)
            pooled_predicted.extend(predictions)
        pooled = prediction_metrics(pooled_actual, pooled_predicted)
        implied = [_structural_target(sample, schedule) for sample in samples if sample["split"] == "train"]
        diagnostics = {
            "schedule": dict(schedule),
            "eta_at_game_0": eta_for_game_count(schedule, 0),
            "eta_at_game_30": eta_for_game_count(schedule, 30),
            "eta_at_game_60": eta_for_game_count(schedule, 60),
            "eta_at_game_120": eta_for_game_count(schedule, 120),
            "eta_at_game_500": eta_for_game_count(schedule, 500),
            "pooled_calibration": pooled,
            "families": family_report,
            "train_implied_percentile_below_zero": statistics.fmean(float(value < 0.0) for value in implied),
            "train_implied_percentile_above_one": statistics.fmean(float(value > 1.0) for value in implied),
            "train_implied_percentile_p01": _quantile(implied, 0.01),
            "train_implied_percentile_p50": _quantile(implied, 0.5),
            "train_implied_percentile_p99": _quantile(implied, 0.99),
        }
        candidates.append((float(pooled["mae"]), float(pooled["rmse"]), family_models, diagnostics))
    candidates.sort(key=lambda row: (row[0], row[1], str(row[3]["schedule"]["name"])))
    _mae, _rmse, selected_models, selected = candidates[0]
    selected_schedule = dict(selected["schedule"])
    report: dict[str, object] = {
        "selected_eta_schedule": selected_schedule,
        "eta_support": "DeepRMM-01 contributes no sample below 500 prior games; only 8 clean opponent samples begin below 120. This is too little support to identify the organizer's early schedule precisely, so schedule comparison is diagnostic and the high-count floor dominates fit quality.",
        "calibration_mae_gap_to_second_schedule": candidates[1][0] - candidates[0][0] if len(candidates) > 1 else None,
        "calibration_mae_range_across_schedules": candidates[-1][0] - candidates[0][0] if len(candidates) > 1 else None,
        "candidate_grid": [{key: value for key, value in diagnostics.items() if key != "families"} for _left, _right, _models, diagnostics in candidates],
        "families": {},
    }
    pooled_actual: list[float] = []
    pooled_predicted: list[float] = []
    for family in GLEE_FAMILIES:
        model = selected_models[family]
        calibration = _model_rows(samples, split="calibration", family=family, feature_key="structural_features")
        test = _model_rows(samples, split="test", family=family, feature_key="structural_features")
        calibration_predictions, calibration_percentiles = _structural_predictions(model, calibration, selected_schedule)
        calibration_residuals = [abs(prediction - float(row["rating_delta"])) for prediction, row in zip(calibration_predictions, calibration, strict=True)]
        radius_80 = _quantile(calibration_residuals, 0.8)
        radius_95 = _quantile(calibration_residuals, 0.95)
        predictions, percentiles = _structural_predictions(model, test, selected_schedule)
        actual = [float(row["rating_delta"]) for row in test]
        train = _model_rows(samples, split="train", family=family, feature_key="structural_features")
        implied = [_structural_target(row["sample"], selected_schedule) for row in train]
        report["families"][family] = {
            "ridge_lambda": model.ridge_lambda,
            "counts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
            "calibration": prediction_metrics([float(row["rating_delta"]) for row in calibration], calibration_predictions),
            "calibration_predicted_percentile_outside_unit_interval": statistics.fmean(float(value < 0.0 or value > 1.0) for value in calibration_percentiles),
            "test": _metric_slices(test, predictions, interval_80=radius_80, interval_95=radius_95),
            "test_predicted_percentile_outside_unit_interval": statistics.fmean(float(value < 0.0 or value > 1.0) for value in percentiles),
            "train_implied_percentile_outside_unit_interval": statistics.fmean(float(value < 0.0 or value > 1.0) for value in implied),
            "model": model.as_dict(),
        }
        pooled_actual.extend(actual)
        pooled_predicted.extend(predictions)
    report["pooled_test"] = prediction_metrics(pooled_actual, pooled_predicted)
    return selected_models, report


def prediction_rows(samples: Sequence[Mapping[str, object]], direct_models: Mapping[str, RidgeRatingModel], structural_models: Mapping[str, RidgeRatingModel], schedule: Mapping[str, object]) -> list[dict[str, object]]:
    """Emit compact per-sample predictions for residual audit without copying terminal archives."""
    rows: list[dict[str, object]] = []
    for sample in samples:
        family = str(sample["family"])
        actual = float(sample["rating_delta"])
        direct = direct_models[family].predict(sample["direct_features"])
        games_before = int(sample["pregame_game_count"])
        percentile = structural_models[family].predict(sample["structural_features"])
        structural = predict_display_delta(float(sample["pregame_display_rating"]), games_before, percentile, eta_for_game_count(schedule, games_before))
        rows.append(
            {
                "contract": JOINT_RATING_CONTRACT,
                "schema_version": 1,
                "game_id": sample["game_id"],
                "family": family,
                "completed_at": sample["completed_at"],
                "split": sample["split"],
                "target_scope": sample["target_scope"],
                "target_player": sample["target_player"],
                "target_public_player_id": sample["target_public_player_id"],
                "target_event_id": sample["target_event_id"],
                "role": sample["terminal"]["role"],
                "outcome": sample["terminal"]["outcome"],
                "complete_information": bool(sample["direct_features"]["complete_information"]),
                "configuration_seen_in_train": bool(sample["configuration_seen_in_train"]),
                "pregame_game_count": games_before,
                "actual_rating_delta": actual,
                "direct_prediction": direct,
                "direct_error": direct - actual,
                "structural_prediction": structural,
                "structural_error": structural - actual,
                "predicted_adjusted_percentile": percentile,
                "eta": eta_for_game_count(schedule, games_before),
            }
        )
    return rows


def _prediction_group(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    actual = [float(row["actual_rating_delta"]) for row in rows]
    direct = [float(row["direct_prediction"]) for row in rows]
    structural = [float(row["structural_prediction"]) for row in rows]
    return {"count": len(rows), "direct": prediction_metrics(actual, direct), "structural": prediction_metrics(actual, structural)}


def residual_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Describe untouched-test residuals, support, and paired joint errors without selecting a new model."""
    test = [row for row in rows if row["split"] == "test"]

    def groups(key: str) -> dict[str, object]:
        values: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in test:
            values[str(row[key])].append(row)
        return {value: _prediction_group(group) for value, group in sorted(values.items())}

    count_bands: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in test:
        count = int(row["pregame_game_count"])
        band = "lt-120" if count < 120 else "120-to-499" if count < 500 else "500-to-1999" if count < 2000 else "ge-2000"
        count_bands[band].append(row)
    direct_wins = sum(abs(float(row["direct_error"])) < abs(float(row["structural_error"])) for row in test)
    structural_wins = sum(abs(float(row["structural_error"])) < abs(float(row["direct_error"])) for row in test)
    ties = len(test) - direct_wins - structural_wins
    by_game: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in test:
        by_game[str(row["game_id"])][str(row["target_scope"])] = row
    pairs = [value for value in by_game.values() if set(value) == {"self", "opponent"}]
    pair_audit: dict[str, object] = {"count": len(pairs)}
    for prefix, key in (("actual", "actual_rating_delta"), ("direct_prediction", "direct_prediction"), ("structural_prediction", "structural_prediction"), ("direct_residual", "direct_error"), ("structural_residual", "structural_error")):
        pair_audit[f"{prefix}_correlation"] = _correlation([float(value["self"][key]) for value in pairs], [float(value["opponent"][key]) for value in pairs])
    extremes = sorted(test, key=lambda row: max(abs(float(row["direct_error"])), abs(float(row["structural_error"]))), reverse=True)[:20]
    return {
        "test_samples": len(test),
        "by_scope": groups("target_scope"),
        "by_outcome": groups("outcome"),
        "by_role": groups("role"),
        "by_complete_information": groups("complete_information"),
        "by_configuration_seen_in_train": groups("configuration_seen_in_train"),
        "by_game_count_band": {band: _prediction_group(group) for band, group in sorted(count_bands.items())},
        "paired_absolute_error": {"direct_better": direct_wins, "structural_better": structural_wins, "tie": ties},
        "joint_test_pairs": pair_audit,
        "largest_residuals": [
            {
                "game_id": row["game_id"],
                "family": row["family"],
                "target_scope": row["target_scope"],
                "role": row["role"],
                "outcome": row["outcome"],
                "actual_rating_delta": row["actual_rating_delta"],
                "direct_prediction": row["direct_prediction"],
                "structural_prediction": row["structural_prediction"],
                "pregame_game_count": row["pregame_game_count"],
            }
            for row in extremes
        ],
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(path, "".join(_canonical(value) + "\n" for value in values))


@dataclass(frozen=True)
class _Archive:
    path: Path
    digest: str
    payload: dict[str, object]


class GleeJointRatingAnalysis:
    """Build and evaluate a shadow-only direct and structural joint rating model."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        history_database: Path,
        game_archive_root: Path,
        activity_summary: Path,
        assignment_dir: Path,
        output_dir: Path,
        train_fraction: float = 0.6,
        calibration_fraction: float = 0.2,
        self_delta_tolerance: float = 0.11,
        opponent_assignment_probability: float = 0.95,
    ) -> None:
        if self_delta_tolerance < 0 or not 0 < opponent_assignment_probability <= 1:
            raise ValueError("rating reconstruction thresholds are invalid")
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.activity_summary_path = activity_summary.resolve()
        self.assignment_dir = assignment_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.train_fraction = train_fraction
        self.calibration_fraction = calibration_fraction
        self.self_delta_tolerance = self_delta_tolerance
        self.opponent_assignment_probability = opponent_assignment_probability

    def _load_sources(self) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        activity = json.loads(self.activity_summary_path.read_text(encoding="utf-8"))
        assignment_manifest_path = self.assignment_dir / "manifest.json"
        assignment_manifest = json.loads(assignment_manifest_path.read_text(encoding="utf-8"))
        if activity.get("contract") != "glee-activity-eda-v2" or assignment_manifest.get("contract") != "glee-joint-game-assignment-v1":
            raise ValueError("rating reconstruction requires corrected activity and joint-assignment receipts")
        frontier = int(assignment_manifest["frontier_sequence"])
        if frontier != int(activity["source_frontier"]["frontier_sequence"]):
            raise ValueError("activity and assignment frontiers differ")
        for name in ("self-alignments.jsonl", "assignments-evidence-conditioned.jsonl"):
            expected = assignment_manifest["artifacts"][name]["sha256"]
            if _file_digest(self.assignment_dir / name) != expected:
                raise RuntimeError(f"assignment artifact hash mismatch: {name}")
        alignments = [json.loads(line) for line in (self.assignment_dir / "self-alignments.jsonl").read_text(encoding="utf-8").splitlines()]
        assignment_rows = {str(row["game_id"]): row for row in (json.loads(line) for line in (self.assignment_dir / "assignments-evidence-conditioned.jsonl").read_text(encoding="utf-8").splitlines())}
        history = _read_only_database(self.history_database)
        try:
            history_rows = {
                str(row["game_id"]): dict(row)
                for row in history.execute("SELECT game_id, game_family, completed_at, rating_delta, revision, record_sha256 FROM games WHERE completed_at IS NOT NULL ORDER BY completed_at, game_id")
            }
        finally:
            history.close()
        return activity, assignment_manifest, alignments, assignment_rows, history_rows

    def _events(self, *, frontier: int, selected_ids: set[str], expected_source_hash: str, expected_event_hash: str) -> tuple[dict[str, EffectiveEvent], dict[str, object]]:
        selected: dict[str, EffectiveEvent] = {}

        def retain(event: EffectiveEvent) -> None:
            event_id = str(event.source_change_sequence)
            if event_id in selected_ids:
                selected[event_id] = event

        reporter = _read_only_database(self.reporter_database)
        try:
            reporter.execute("BEGIN")
            summary = derive_effective_events(reporter, frontier_sequence=frontier, event_sink=retain)
            reporter.execute("ROLLBACK")
        finally:
            reporter.close()
        if summary["source_rows_sha256"] != expected_source_hash or summary["effective_events_sha256"] != expected_event_hash:
            raise RuntimeError("rating reconstruction does not reproduce the assignment event substrate")
        missing = sorted(selected_ids - set(selected), key=lambda value: int(value))
        if missing:
            raise RuntimeError(f"selected public event IDs are absent from the frozen substrate: {missing[:5]}")
        return selected, summary

    def _archive(self, alignment: Mapping[str, object], cache: dict[str, _Archive]) -> _Archive:
        game_id = str(alignment["game_id"])
        if game_id in cache:
            return cache[game_id]
        relative = alignment.get("archive_path")
        expected = alignment.get("archive_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("archive reference is missing")
        path = self.game_archive_root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = _digest(payload)
        if digest != expected or str(payload.get("game_id") or "") != game_id or str(payload.get("game_family") or "") != str(alignment["family"]):
            raise RuntimeError(f"terminal archive receipt mismatch: {game_id}")
        archive = _Archive(path=path, digest=digest, payload=payload)
        cache[game_id] = archive
        return archive

    def _build_samples(
        self,
        *,
        activity: Mapping[str, object],
        alignments: Sequence[Mapping[str, object]],
        assignments: Mapping[str, Mapping[str, object]],
        history_rows: Mapping[str, Mapping[str, object]],
        events: Mapping[str, EffectiveEvent],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        epoch_origin = _iso_to_timestamp(str(activity["source_frontier"]["first_started_at"]))
        archive_cache: dict[str, _Archive] = {}
        samples: list[dict[str, object]] = []
        receipts: list[dict[str, object]] = []
        exclusions: Counter[str] = Counter()
        scope_counts: Counter[str] = Counter()
        clean_event_by_game: dict[str, EffectiveEvent] = {}
        exact_opponent_event_by_game: dict[str, EffectiveEvent] = {}
        alignment_by_game = {str(row["game_id"]): row for row in alignments}
        for game_id, alignment in alignment_by_game.items():
            event_id = alignment.get("self_event_id")
            event = events.get(str(event_id)) if event_id is not None else None
            history = history_rows.get(game_id)
            if event is None:
                exclusions["self-event-unmatched"] += 1
                continue
            history_delta = _optional_number(history.get("rating_delta")) if history is not None else None
            clean_delta = event.clean_rating_delta
            if history_delta is None:
                exclusions["self-history-delta-missing"] += 1
                continue
            if event.games_delta != 1 or clean_delta is None or event.previous_rating_anchor is None or event.current_rating is None:
                exclusions["self-public-event-not-clean-single"] += 1
                continue
            if abs(history_delta - clean_delta) > self.self_delta_tolerance:
                exclusions["self-authenticated-public-delta-mismatch"] += 1
                continue
            clean_event_by_game[game_id] = event
        for game_id, assignment in assignments.items():
            true_ids = tuple(str(value) for value in assignment.get("true_public_player_ids") or ())
            top = assignment.get("top_public_players") or []
            top_probability = _number(top[0].get("probability")) if top and isinstance(top[0], Mapping) else 0.0
            event_id = assignment.get("map_event_id")
            event = events.get(str(event_id)) if event_id is not None else None
            if assignment.get("identity_scope") != "known" or len(true_ids) != 1:
                exclusions["opponent-identity-not-unique"] += 1
                continue
            if assignment.get("map_public_player_id") != true_ids[0] or top_probability < self.opponent_assignment_probability:
                exclusions["opponent-assignment-below-clean-threshold"] += 1
                continue
            if event is None or event.player_id != true_ids[0] or event.games_delta != 1 or event.clean_rating_delta is None or event.previous_rating_anchor is None or event.current_rating is None:
                exclusions["opponent-public-event-not-clean-single"] += 1
                continue
            exact_opponent_event_by_game[game_id] = event
        for game_id, alignment in sorted(alignment_by_game.items(), key=lambda item: (str(item[1]["completed_at"]), item[0])):
            self_event = clean_event_by_game.get(game_id)
            opponent_event = exact_opponent_event_by_game.get(game_id)
            if self_event is None and opponent_event is None:
                continue
            history = history_rows.get(game_id)
            if history is None:
                exclusions["history-row-missing"] += int(self_event is not None) + int(opponent_event is not None)
                continue
            try:
                archive = self._archive(alignment, archive_cache)
            except (OSError, json.JSONDecodeError, ValueError, RuntimeError):
                exclusions["terminal-archive-invalid"] += int(self_event is not None) + int(opponent_event is not None)
                continue
            self_player = str(archive.payload.get("your_player") or "")
            if self_player not in {"player_1", "player_2"}:
                exclusions["terminal-self-player-invalid"] += int(self_event is not None) + int(opponent_event is not None)
                continue
            completed_at = _iso_to_timestamp(str(alignment["completed_at"]))
            perspectives = []
            if self_event is not None:
                perspectives.append(("self", self_player, self_event, opponent_event, _optional_number(history.get("rating_delta"))))
            if opponent_event is not None:
                perspectives.append(("opponent", _other_player(self_player), opponent_event, self_event, None))
            for target_scope, target_player, target_event, other_event, authenticated_delta in perspectives:
                try:
                    view = _terminal_view(archive.payload, target_player)
                    direct_features, structural_features = _sample_features(view, target_event, other_event, completed_at=completed_at, epoch_origin=epoch_origin)
                except (KeyError, TypeError, ValueError):
                    exclusions["terminal-projection-invalid"] += 1
                    continue
                sample = {
                    "contract": JOINT_RATING_CONTRACT,
                    "schema_version": 1,
                    "game_id": game_id,
                    "family": str(alignment["family"]),
                    "completed_at": str(alignment["completed_at"]),
                    "completed_at_timestamp": completed_at,
                    "target_scope": target_scope,
                    "target_player": target_player,
                    "target_public_player_id": target_event.player_id,
                    "target_event_id": str(target_event.source_change_sequence),
                    "target_event_frontier": target_event.frontier_sequence,
                    "rating_delta": float(target_event.clean_rating_delta),
                    "authenticated_history_delta": authenticated_delta,
                    "pregame_game_count": target_event.previous_high_water,
                    "postgame_game_count": target_event.new_high_water,
                    "pregame_display_rating": target_event.previous_rating_anchor,
                    "postgame_display_rating": target_event.current_rating,
                    "opponent_pregame_rating_known": other_event is not None and other_event.previous_rating_anchor is not None,
                    "opponent_pregame_display_rating": other_event.previous_rating_anchor if other_event is not None else None,
                    "opponent_assignment_probability": _number((assignments.get(game_id, {}).get("top_public_players") or [{}])[0].get("probability")) if opponent_event is not None else None,
                    "terminal": {key: value for key, value in view.items() if key != "base_features"},
                    "direct_features": direct_features,
                    "structural_features": structural_features,
                }
                samples.append(sample)
                scope_counts[target_scope] += 1
                receipts.append(
                    {
                        "contract": JOINT_RATING_CONTRACT,
                        "schema_version": 1,
                        "game_id": game_id,
                        "family": sample["family"],
                        "completed_at": sample["completed_at"],
                        "target_scope": target_scope,
                        "target_player": target_player,
                        "target_public_player_id": target_event.player_id,
                        "target_event_id": sample["target_event_id"],
                        "archive_path": str(archive.path.relative_to(self.game_archive_root)),
                        "archive_sha256": archive.digest,
                        "history_revision": int(history["revision"]),
                        "history_record_sha256": str(history["record_sha256"]),
                        "authenticated_history_delta": authenticated_delta,
                        "public_clean_rating_delta": sample["rating_delta"],
                        "pregame_game_count": sample["pregame_game_count"],
                        "pregame_display_rating": sample["pregame_display_rating"],
                        "postgame_display_rating": sample["postgame_display_rating"],
                        "observed_configuration_sha256": sample["terminal"]["observed_configuration_sha256"],
                        "target_provenance": "authenticated-self-plus-clean-public" if target_scope == "self" else "unique-label-capacity-assigned-clean-public",
                    }
                )
        split_by_game = chronological_game_splits(samples, train_fraction=self.train_fraction, calibration_fraction=self.calibration_fraction)
        for sample in samples:
            sample["split"] = split_by_game[str(sample["game_id"])]
        train_configurations = {family: {str(sample["terminal"]["observed_configuration_sha256"]) for sample in samples if sample["family"] == family and sample["split"] == "train"} for family in GLEE_FAMILIES}
        for sample in samples:
            sample["configuration_seen_in_train"] = str(sample["terminal"]["observed_configuration_sha256"]) in train_configurations[str(sample["family"])]
        receipt_by_key = {(str(sample["game_id"]), str(sample["target_scope"])): str(sample["split"]) for sample in samples}
        for receipt in receipts:
            receipt["split"] = receipt_by_key[(str(receipt["game_id"]), str(receipt["target_scope"]))]
        game_count_support = {}
        for scope in ("self", "opponent"):
            counts = [int(sample["pregame_game_count"]) for sample in samples if sample["target_scope"] == scope]
            game_count_support[scope] = {
                "count": len(counts),
                "minimum": min(counts),
                "median": statistics.median(counts),
                "maximum": max(counts),
                "below_30": sum(value < 30 for value in counts),
                "below_60": sum(value < 60 for value in counts),
                "below_120": sum(value < 120 for value in counts),
                "below_500": sum(value < 500 for value in counts),
            }
        joint_games = defaultdict(dict)
        for sample in samples:
            joint_games[str(sample["game_id"])][str(sample["target_scope"])] = float(sample["rating_delta"])
        complete_pairs = [value for value in joint_games.values() if set(value) == {"self", "opponent"}]
        joint_delta = {
            "complete_pairs": len(complete_pairs),
            "both_positive": sum(value["self"] > 0 and value["opponent"] > 0 for value in complete_pairs),
            "both_negative": sum(value["self"] < 0 and value["opponent"] < 0 for value in complete_pairs),
            "opposite_sign": sum(value["self"] * value["opponent"] < 0 for value in complete_pairs),
            "zero_involved": sum(value["self"] == 0 or value["opponent"] == 0 for value in complete_pairs),
            "correlation": _correlation([value["self"] for value in complete_pairs], [value["opponent"] for value in complete_pairs]),
        }
        configuration_support = {}
        for family in GLEE_FAMILIES:
            family_samples = [sample for sample in samples if sample["family"] == family]
            test_samples = [sample for sample in family_samples if sample["split"] == "test"]
            configuration_support[family] = {
                "unique_all": len({str(sample["terminal"]["observed_configuration_sha256"]) for sample in family_samples}),
                "unique_train": len(train_configurations[family]),
                "test_samples": len(test_samples),
                "test_seen_in_train": sum(bool(sample["configuration_seen_in_train"]) for sample in test_samples),
            }
        inventory = {
            "sample_count": len(samples),
            "logical_games": len({str(sample["game_id"]) for sample in samples}),
            "by_scope": dict(sorted(scope_counts.items())),
            "by_family_scope": {family: dict(sorted(Counter(str(sample["target_scope"]) for sample in samples if sample["family"] == family).items())) for family in GLEE_FAMILIES},
            "by_split": dict(sorted(Counter(str(sample["split"]) for sample in samples).items())),
            "minimum_pregame_game_count": min(int(sample["pregame_game_count"]) for sample in samples),
            "maximum_pregame_game_count": max(int(sample["pregame_game_count"]) for sample in samples),
            "game_count_support": game_count_support,
            "joint_delta_pairs": joint_delta,
            "configuration_support": configuration_support,
            "exclusions": dict(sorted(exclusions.items())),
            "archive_files_loaded": len(archive_cache),
        }
        return samples, receipts, inventory

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        inventory = summary["inventory"]
        direct = summary["direct_model"]
        structural = summary["structural_model"]
        return "\n".join(
            [
                "# GLEE joint rating-delta reconstruction v1",
                "",
                f"**Status:** Offline shadow reconstruction through public reporter frontier `{summary['frontier_sequence']}`; it starts no matchmaking, sends no model call, changes no live prompt or action, and publishes no dossier or identity update.",
                "",
                "## Evidence",
                "",
                f"The clean corpus contains {inventory['sample_count']:,} player-perspective samples from {inventory['logical_games']:,} whole games: {inventory['by_scope'].get('self', 0):,} DeepRMM-01 samples anchored by authenticated history plus public score, and {inventory['by_scope'].get('opponent', 0):,} uniquely labeled, capacity-assigned opponent samples. DeepRMM-01 begins at {inventory['game_count_support']['self']['minimum']:,} prior games; clean opponent samples range down to {inventory['game_count_support']['opponent']['minimum']:,}, but only {inventory['game_count_support']['opponent']['below_120']:,} lie below 120. Public single-game targets that disagree with authenticated self deltas by more than `0.11`, aggregate pulses, identity collisions, and low-confidence joins are excluded rather than repaired by assumption.",
                "",
                "Every perspective from the same game stays in one family-stratified chronological block. Models train on the first 60% of games, select ridge strength and residual intervals on the next 20%, and report once on the final 20%. The target is the corrected clean public displayed-rating change; the independently authenticated self delta checks its provenance but is rounded more coarsely.",
                "",
                "## Direct baseline",
                "",
                f"The family-specific direct ridge baseline reaches pooled untouched-test MAE `{direct['pooled_test']['mae']:.4f}`, RMSE `{direct['pooled_test']['rmse']:.4f}`, sign accuracy `{direct['pooled_test']['sign_accuracy']:.3%}`, and correlation `{direct['pooled_test']['correlation']:.4f}`. A zero-change baseline has MAE `{direct['pooled_test_zero_baseline']['mae']:.4f}`; a family training-mean baseline has MAE `{direct['pooled_test_training_mean_baseline']['mae']:.4f}`.",
                "",
                "## Structural inversion",
                "",
                f"The structural ladder selects `{structural['selected_eta_schedule']['name']}` on calibration, inverts public display shrinkage to an implied adjusted percentile, fits family-specific percentile surfaces, and propagates predictions back through the published equations. It reaches pooled untouched-test MAE `{structural['pooled_test']['mae']:.4f}`, RMSE `{structural['pooled_test']['rmse']:.4f}`, sign accuracy `{structural['pooled_test']['sign_accuracy']:.3%}`, and correlation `{structural['pooled_test']['correlation']:.4f}`.",
                "",
                "The archive begins after DeepRMM-01 already exceeded 500 games in every family, while only 8 clean opponent samples cover the nominal early schedule below 120 games. Constant, piecewise-linear, exponential, and reciprocal candidates are therefore compared, but their early behavior is weakly identified and their common high-count floor dominates the result. The selected schedule is an empirical reconstruction under the tested percentile surface, not proof of the organizer's exact implementation. Out-of-range implied or predicted percentiles remain diagnostics; only predictions are bounded when propagated through the official game-rating range.",
                "",
                "## Boundary",
                "",
                "This receipt establishes a chronological self-and-clean-opponent baseline. It does not yet turn ambiguous public events into labels, reconstruct the organizer's hourly opponent-strength model, or establish that an opponent optimizes rating. No model in this artifact has live authority. The next step is residual and support analysis, then out-of-fold opponent-feasibility features only if the clean model improves on simpler baselines.",
                "",
            ]
        )

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"joint-rating output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        activity, assignment_manifest, alignments, assignments, history_rows = self._load_sources()
        frontier = int(assignment_manifest["frontier_sequence"])
        selected_ids = {str(row["self_event_id"]) for row in alignments if row.get("self_event_id") is not None}
        selected_ids.update(str(row["map_event_id"]) for row in assignments.values() if row.get("map_event_id") is not None)
        events, effective_summary = self._events(frontier=frontier, selected_ids=selected_ids, expected_source_hash=str(assignment_manifest["source_rows_sha256"]), expected_event_hash=str(assignment_manifest["effective_events_sha256"]))
        samples, receipts, inventory = self._build_samples(activity=activity, alignments=alignments, assignments=assignments, history_rows=history_rows, events=events)
        direct_models, direct_report = fit_direct_models(samples)
        structural_models, structural_report = fit_structural_models(samples)
        predictions = prediction_rows(samples, direct_models, structural_models, structural_report["selected_eta_schedule"])
        residuals = residual_audit(predictions)
        summary: dict[str, object] = {
            "contract": JOINT_RATING_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only",
            "frontier_sequence": frontier,
            "parameters": {
                "train_fraction": self.train_fraction,
                "calibration_fraction": self.calibration_fraction,
                "test_fraction": 1.0 - self.train_fraction - self.calibration_fraction,
                "self_delta_tolerance": self.self_delta_tolerance,
                "opponent_assignment_probability": self.opponent_assignment_probability,
                "ridge_grid": list(RIDGE_GRID),
                "eta_schedules": [dict(schedule) for schedule in ETA_SCHEDULES],
            },
            "sources": {
                "reporter_database": str(self.reporter_database),
                "history_database": str(self.history_database),
                "game_archive_root": str(self.game_archive_root),
                "activity_summary": {"path": str(self.activity_summary_path), "sha256": _file_digest(self.activity_summary_path)},
                "assignment_dir": str(self.assignment_dir),
                "assignment_manifest_sha256": _file_digest(self.assignment_dir / "manifest.json"),
                "source_rows_sha256": effective_summary["source_rows_sha256"],
                "effective_events_sha256": effective_summary["effective_events_sha256"],
            },
            "inventory": inventory,
            "direct_model": direct_report,
            "structural_model": structural_report,
            "residual_audit": residuals,
            "promotion": {
                "live_authority": False,
                "opponent_ambiguous_events_used": False,
                "early_eta_schedule_identified": False,
                "next_gate": "residual audit and clean opponent-feasibility evaluation",
            },
        }
        _write_jsonl(self.output_dir / "rating-game-receipts.jsonl", receipts)
        _write_jsonl(self.output_dir / "joint-rating-samples.jsonl", samples)
        _write_jsonl(self.output_dir / "predictions.jsonl", predictions)
        selected_event_ids = {str(sample["target_event_id"]) for sample in samples}
        _write_jsonl(self.output_dir / "public-rating-events.jsonl", (events[event_id].as_dict() for event_id in sorted(selected_event_ids, key=int)))
        _write_json(self.output_dir / "direct-model.json", {"contract": JOINT_RATING_CONTRACT, "schema_version": 1, "feature_names_by_family": {family: list(features) for family, features in FAMILY_DIRECT_FEATURES.items()}, "models": {family: model.as_dict() for family, model in direct_models.items()}, "validation": direct_report})
        _write_json(self.output_dir / "structural-model.json", {"contract": JOINT_RATING_CONTRACT, "schema_version": 1, "feature_names_by_family": {family: list(features) for family, features in FAMILY_BASE_FEATURES.items()}, "eta_schedule": structural_report["selected_eta_schedule"], "models": {family: model.as_dict() for family, model in structural_models.items()}, "validation": structural_report})
        _write_json(self.output_dir / "summary.json", summary)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifact_names = ("README.md", "summary.json", "rating-game-receipts.jsonl", "public-rating-events.jsonl", "joint-rating-samples.jsonl", "predictions.jsonl", "direct-model.json", "structural-model.json")
        manifest = {
            "contract": JOINT_RATING_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "parameters": summary["parameters"],
            "source_rows_sha256": effective_summary["source_rows_sha256"],
            "effective_events_sha256": effective_summary["effective_events_sha256"],
            "assignment_manifest_sha256": summary["sources"]["assignment_manifest_sha256"],
            "implementation_sha256": {
                "glee_joint_rating_analysis.py": _file_digest(Path(__file__)),
                "glee_effective_events.py": _file_digest(Path(__file__).with_name("glee_effective_events.py")),
                "glee_semantics.py": _file_digest(Path(__file__).with_name("glee_semantics.py")),
                "glee_negotiation_rating_v2_4.py": _file_digest(Path(__file__).with_name("glee_negotiation_rating_v2_4.py")),
            },
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifact_names},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": JOINT_RATING_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": frontier, "inventory": inventory, "direct_model": direct_report, "structural_model": structural_report, "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
