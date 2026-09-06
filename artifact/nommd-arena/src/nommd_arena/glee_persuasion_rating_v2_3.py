"""Scale-normalized displayed-rating estimates for live GLEE Persuasion v2.3."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .glee_joint_rating_analysis import _terminal_view
from .glee_negotiation_rating_v2_4 import RidgeRatingModel, fit_ridge, regression_metrics


SCHEMA_VERSION = 1
MODEL_VERSION = "persuasion-rating-surrogate-v2.3"
RIDGE_LAMBDA = 10.0
FEATURE_NAMES = (
    "bias",
    "target_seller",
    "ordinary_completion",
    "round_phase",
    "own_payoff_ratio",
    "own_payoff_ratio_squared",
    "own_payoff_ratio_cubed",
    "opponent_payoff_ratio",
    "payoff_advantage_ratio",
    "own_nonpositive_payoff",
    "persuasion_quality_probability",
    "persuasion_u_price_ratio",
    "persuasion_v_price_ratio",
    "persuasion_total_rounds_scaled",
    "persuasion_seller_knows_values",
    "persuasion_binary_message",
)


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported Persuasion player: {player}")


def _role_players(game: Mapping[str, object]) -> tuple[str, str]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    for player in ("player_1", "player_2"):
        if str(state.get(f"{player}_role") or "").casefold() == "seller":
            return player, _other_player(player)
    return "player_1", "player_2"


def _history_totals(state: Mapping[str, object]) -> tuple[float, float]:
    seller = _finite(state.get("seller_total_payoff"), float("nan"))
    buyer = _finite(state.get("buyer_total_payoff"), float("nan"))
    history = state.get("history") if isinstance(state.get("history"), list) else []
    if not math.isfinite(seller):
        seller = sum(_finite(entry.get("seller_payoff")) for entry in history if isinstance(entry, Mapping))
    if not math.isfinite(buyer):
        buyer = sum(_finite(entry.get("buyer_payoff")) for entry in history if isinstance(entry, Mapping))
    return seller, buyer


def terminal_scenario(game: Mapping[str, object], *, bought: bool, quality: str | None) -> dict[str, Any]:
    """Construct the stop-now terminal branch after the current Persuasion round."""
    terminal = copy.deepcopy(dict(game))
    state = terminal.get("game_state") if isinstance(terminal.get("game_state"), dict) else {}
    terminal["game_state"] = state
    seller_player, buyer_player = _role_players(terminal)
    seller_total, buyer_total = _history_totals(state)
    price = _finite(state.get("product_price"))
    if bought:
        quality_value = _finite(state.get("v")) if quality == "high" else _finite(state.get("u"))
        seller_total += price
        buyer_total += quality_value - price
    round_number = max(1, int(_finite(state.get("round"), 1.0)))
    total_rounds = max(round_number, int(_finite(state.get("total_rounds"), float(round_number))))
    result = {
        "outcome": "completed",
        "rounds_played": round_number,
        "rounds_total": total_rounds,
        f"{seller_player}_payoff": seller_total,
        f"{buyer_player}_payoff": buyer_total,
    }
    state.update({"phase": "completed", "result": copy.deepcopy(result), "seller_total_payoff": seller_total, "buyer_total_payoff": buyer_total})
    terminal.update({"status": "completed", "result": result})
    terminal.pop("valid_actions", None)
    terminal.pop("phase", None)
    return terminal


def terminal_features(terminal: Mapping[str, object]) -> dict[str, float]:
    view = _terminal_view(terminal, str(terminal.get("your_player") or ""))
    base = view["base_features"]
    return {name: float(base[name]) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class PersuasionRatingSurrogate:
    """One frozen point estimator with a chronological retrospective receipt."""

    model: RidgeRatingModel
    validation: dict[str, object]

    def predict_terminal(self, terminal: Mapping[str, object]) -> dict[str, object]:
        return {
            "estimated_rating_delta": round(self.model.predict(terminal_features(terminal)), 4),
            "model": MODEL_VERSION,
            "ridge_lambda": self.model.ridge_lambda,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "model": self.model.as_dict(),
            "validation": copy.deepcopy(self.validation),
            "boundary": "Retrospective scale-normalized point estimate only; it omits opponent rating, supplies no deterministic action authority, and non-final stop-now scenarios are not continuation values.",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PersuasionRatingSurrogate:
        if value.get("schema_version") != SCHEMA_VERSION or value.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported Persuasion rating surrogate")
        return cls(model=RidgeRatingModel.from_dict(value["model"]), validation=copy.deepcopy(dict(value.get("validation") or {})))


def fit_persuasion_rating_surrogate(embedded_games: Sequence[Mapping[str, object]], rating_deltas: Mapping[str, object]) -> PersuasionRatingSurrogate | None:
    """Fit a fixed-regularization model and preserve a final chronological suffix for evaluation."""
    samples: list[dict[str, object]] = []
    for embedded in embedded_games:
        game = embedded.get("final_game") if isinstance(embedded.get("final_game"), Mapping) else None
        if game is None:
            continue
        game_id = str(game.get("game_id") or "")
        target = rating_deltas.get(game_id)
        if not isinstance(target, Mapping):
            continue
        delta = target.get("rating_delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            continue
        try:
            features = terminal_features(game)
        except (KeyError, TypeError, ValueError):
            continue
        samples.append({"game_id": game_id, "completed_at": str(target.get("completed_at") or embedded.get("completed_at") or ""), "features": features, "rating_delta": float(delta)})
    samples.sort(key=lambda row: (str(row["completed_at"]), str(row["game_id"])))
    if len(samples) < 20:
        return None
    held_count = max(5, len(samples) // 5)
    development = samples[:-held_count]
    held_out = samples[-held_count:]
    development_model = fit_ridge(development, feature_key="features", feature_names=FEATURE_NAMES, ridge_lambda=RIDGE_LAMBDA)
    validation = {
        "design": "fixed ridge lambda inherited from the frozen joint-rating Persuasion model; final 20% chronological suffix untouched by fitting",
        "samples": len(samples),
        "development_samples": len(development),
        "held_out_samples": len(held_out),
        "held_out": regression_metrics(development_model, held_out, feature_key="features"),
        "latest_completed_at": str(samples[-1]["completed_at"]),
    }
    return PersuasionRatingSurrogate(model=fit_ridge(samples, feature_key="features", feature_names=FEATURE_NAMES, ridge_lambda=RIDGE_LAMBDA), validation=validation)


def persuasion_rating_decision_surface(game: Mapping[str, object], decision_facts: Mapping[str, object], surrogate: PersuasionRatingSurrogate | None) -> dict[str, object]:
    """Expose bounded branch estimates for the current decision without treating them as continuation values."""
    if surrogate is None:
        return {"status": "unavailable", "reason": "seed-has-no-supported-rating-surrogate"}
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or game.get("phase") or "")
    round_number = max(1, int(_finite(state.get("round"), 1.0)))
    total_rounds = max(round_number, int(_finite(state.get("total_rounds"), float(round_number))))

    def estimate(*, bought: bool, quality: str | None) -> dict[str, object]:
        return surrogate.predict_terminal(terminal_scenario(game, bought=bought, quality=quality))

    pass_estimate = estimate(bought=False, quality=None)
    buy_low = estimate(bought=True, quality="low")
    buy_high = estimate(bought=True, quality="high")
    result: dict[str, object] = {
        "status": "retrospective-surrogate",
        "frontier": "computed-before-model-inference",
        "round": round_number,
        "total_rounds": total_rounds,
        "horizon_semantics": "exact current-action terminal branches" if round_number >= total_rounds else "stop-now branch diagnostic; future-round continuation is omitted",
        "pass": pass_estimate,
        "buy_if_low": buy_low,
        "buy_if_high": buy_high,
        "authority": "advisory-only",
        "validation": copy.deepcopy(surrogate.validation.get("held_out")),
        "boundary": "The estimates are noisy displayed-rating diagnostics. They do not override payoff dominance, causal buyer or seller models, legal-action guards, or deadline safety.",
    }
    if action_type == "buyer_decision":
        reliability = decision_facts.get("seller_reliability_forecast") if isinstance(decision_facts.get("seller_reliability_forecast"), Mapping) else {}
        probability_high = min(1.0, max(0.0, _finite(reliability.get("posterior_high_probability"), _finite(state.get("p"), 0.5))))
        expected_buy = probability_high * float(buy_high["estimated_rating_delta"]) + (1.0 - probability_high) * float(buy_low["estimated_rating_delta"])
        result["buyer_decision"] = {
            "posterior_high_probability": round(probability_high, 6),
            "expected_buy_rating_delta": round(expected_buy, 4),
            "pass_rating_delta": pass_estimate["estimated_rating_delta"],
            "expected_buy_minus_pass": round(expected_buy - float(pass_estimate["estimated_rating_delta"]), 4),
        }
    elif action_type in {"seller_recommendation", "seller_message"}:
        forecasts = decision_facts.get("buyer_response_forecasts") if isinstance(decision_facts.get("buyer_response_forecasts"), Mapping) else {}
        quality = str(state.get("current_quality") or "").casefold()
        probability_high = 1.0 if quality == "high" else 0.0 if quality == "low" else min(1.0, max(0.0, _finite(state.get("p"), 0.5)))
        buy_delta = probability_high * float(buy_high["estimated_rating_delta"]) + (1.0 - probability_high) * float(buy_low["estimated_rating_delta"])
        candidates = {}
        for label in ("positive", "negative"):
            forecast = forecasts.get(label) if isinstance(forecasts.get(label), Mapping) else {}
            buy_probability = min(1.0, max(0.0, _finite(forecast.get("buy_probability"), 0.5)))
            expected = buy_probability * buy_delta + (1.0 - buy_probability) * float(pass_estimate["estimated_rating_delta"])
            candidates[label] = {"buy_probability": round(buy_probability, 6), "expected_rating_delta": round(expected, 4)}
        result["seller_signal_candidates"] = {"current_quality": quality if quality in {"high", "low"} else None, "quality_high_probability": round(probability_high, 6), "candidates": candidates}
    return result
