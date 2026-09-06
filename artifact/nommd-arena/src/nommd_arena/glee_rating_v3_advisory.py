"""Dual-channel live adapter for sealed GLEE rating-model v3 forecasts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glee_persuasion_rating_v2_3 import terminal_scenario
from .glee_rating_canary import PublicRatingReader, _accepted_terminal
from .glee_rating_effects import DEFAULT_RATING_EFFECT_CORRECTIONS_PATH
from .glee_rating_v3 import RATING_V3_MODEL_VERSION, load_rating_v3_release
from .glee_rating_v3_shadow import RatingV3ShadowRegistry


RATING_V3_ADVISORY_CONTRACT = "glee-rating-v3-dual-channel-advisory-v1"
RATING_V3_ADVISORY_SCHEMA_VERSION = 1
_BARGAINING_SHARES = (0.1, 0.2, 0.25, 1 / 3, 0.4, 0.5, 0.6, 0.75, 0.9)
_BARGAINING_DISCOUNT_SUPPORT = (0.8, 0.9, 0.95, 1.0)
_REPORTER_CONTRACT = "glee-arena-reporter-v1"
_REPORTER_KIND = "glee-arena-family-frontier"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player identity: {player}")


def _identity_scope(game: Mapping[str, object]) -> str:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
    return "hidden" if opponent.get("type") == "hidden" or not str(opponent.get("name") or "").strip() else "known"


def _compact_prediction(prediction: Mapping[str, object]) -> dict[str, object]:
    if prediction.get("status") != "available":
        return {"status": str(prediction.get("status") or "unavailable"), "reason": prediction.get("reason")}
    interval = prediction.get("interval_80") if isinstance(prediction.get("interval_80"), Sequence) else ()
    return {
        "status": "available",
        "predicted_self_rating_delta": round(float(prediction["predicted_delta"]), 4),
        "interval_80": [round(float(value), 4) for value in interval[:2]],
        "own_payoff": round(float(prediction["own_payoff"]), 6),
        "opponent_payoff": round(float(prediction["opponent_payoff"]), 6),
        "blended_payoff_percentile": round(float(prediction["blended_percentile"]), 6),
        "global_rank_support": int(prediction.get("global_support") or 0),
        "recent_24h_support": int(prediction.get("support_24h") or 0),
        "recent_6h_support": int(prediction.get("support_6h") or 0),
    }


def _terminal_no_deal(game: Mapping[str, object], *, outcome: str = "no_deal") -> dict[str, object]:
    terminal = copy.deepcopy(dict(game))
    state = terminal.get("game_state") if isinstance(terminal.get("game_state"), dict) else {}
    terminal["game_state"] = state
    result = {"outcome": outcome, "rounds_played": int(state.get("round") or 1), "player_1_payoff": 0.0, "player_2_payoff": 0.0}
    state.update({"phase": "completed", "result": copy.deepcopy(result)})
    terminal.update({"status": "completed", "result": result})
    terminal.pop("valid_actions", None)
    terminal.pop("phase", None)
    return terminal


def _negotiation_agreement_terminal(game: Mapping[str, object], *, price: float) -> dict[str, object]:
    terminal = copy.deepcopy(dict(game))
    state = terminal.get("game_state") if isinstance(terminal.get("game_state"), dict) else {}
    terminal["game_state"] = state
    roles = {player: str(state.get(f"{player}_role") or "").casefold() for player in ("player_1", "player_2")}
    values = {player: _finite(state.get(f"{player}_value")) for player in ("player_1", "player_2")}
    if set(roles.values()) != {"seller", "buyer"} or any(value is None for value in values.values()):
        raise ValueError("a Negotiation agreement branch requires both visible reservation values")
    payoffs = {}
    for player in ("player_1", "player_2"):
        payoffs[player] = price - float(values[player]) if roles[player] == "seller" else float(values[player]) - price
    result = {"outcome": "agreement", "agreed_round": int(state.get("round") or 1), "agreed_price": price, "player_1_payoff": payoffs["player_1"], "player_2_payoff": payoffs["player_2"]}
    state.update({"phase": "completed", "result": copy.deepcopy(result)})
    terminal.update({"status": "completed", "result": result})
    terminal.pop("valid_actions", None)
    terminal.pop("phase", None)
    return terminal


class RatingV3Advisory:
    """Seal prospective forecasts and expose compact branch advice without guard authority."""

    def __init__(self, *, model_root: Path, registry_path: Path, reporter_root: Path, history_path: Path, reporter_max_age_s: float = 30.0, corrections_path: Path = DEFAULT_RATING_EFFECT_CORRECTIONS_PATH) -> None:
        self.model_root = model_root.resolve()
        self.model, self.release = load_rating_v3_release(self.model_root)
        self.model_cutoff = str(self.model.source["cutoff"])
        self.corrections_path = corrections_path.resolve()
        self.registry = RatingV3ShadowRegistry(registry_path, model_sha256=str(self.release["model_sha256"]), model_cutoff=self.model_cutoff, corrections_path=self.corrections_path)
        self.reporter = PublicRatingReader(reporter_root, max_age_s=reporter_max_age_s)
        self.history_path = history_path.resolve()
        self._observed_terminal_games: set[str] = set()
        self._last_reconcile = float("-inf")
        self._replay_registered_terminals()

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "contract": RATING_V3_ADVISORY_CONTRACT,
            "schema_version": RATING_V3_ADVISORY_SCHEMA_VERSION,
            "release": self.release["release"],
            "model_sha256": self.release["model_sha256"],
            "model_cutoff": self.model_cutoff,
            "registry_path": str(self.registry.path),
            "history_path": str(self.history_path),
            "rating_effect_corrections_path": str(self.corrections_path),
            "rating_effect_corrections": len(self.registry.corrections),
            "prompt_authority": "advisory-only",
            "guard_authority": "none",
            "shadow_evaluation": "immutable-pre-outcome-registration",
        }

    @staticmethod
    def _own_rating(family: str, sensor_frontier: Mapping[str, object] | None, fallback_stats: Mapping[str, object] | None) -> dict[str, object]:
        stats = sensor_frontier.get("stats") if isinstance(sensor_frontier, Mapping) and isinstance(sensor_frontier.get("stats"), Mapping) else fallback_stats if isinstance(fallback_stats, Mapping) else {}
        scores = stats.get("scores") if isinstance(stats.get("scores"), Mapping) else {}
        score = scores.get(family) if isinstance(scores.get(family), Mapping) else {}
        rating = _finite(score.get("rating"))
        games_played = score.get("games_played")
        if rating is None or isinstance(games_played, bool) or not isinstance(games_played, int) or games_played < 1:
            return {"status": "unavailable", "reason": f"self-{family}-rating-state-unavailable"}
        return {
            "status": "available",
            "source": "authenticated-sensor-frontier" if isinstance(sensor_frontier, Mapping) else "authenticated-supervisor-stats-cache",
            "rating": rating,
            "games_played": games_played,
            "sensor_sequence": sensor_frontier.get("sequence") if isinstance(sensor_frontier, Mapping) else None,
            "observed_at": sensor_frontier.get("fetched_at") if isinstance(sensor_frontier, Mapping) else None,
            "frontier_sha256": sensor_frontier.get("frontier_sha256") if isinstance(sensor_frontier, Mapping) else None,
            "stats_sha256": _sha(stats),
        }

    def _family_public_context(self, family: str, *, observed_at: str) -> dict[str, object]:
        path = self.reporter.current_path
        if not path.is_file():
            return {"status": "unavailable", "reason": "reporter-frontier-missing"}
        frontier = json.loads(path.read_text(encoding="utf-8"))
        if frontier.get("contract") != _REPORTER_CONTRACT or frontier.get("kind") != _REPORTER_KIND or frontier.get("schema_version") != 2:
            raise RuntimeError("unsupported public reporter frontier")
        if frontier.get("frontier_sha256") != _sha({key: value for key, value in frontier.items() if key != "frontier_sha256"}):
            raise RuntimeError("public reporter frontier hash mismatch")
        completed_at = str(frontier["completed_at"])
        age_s = max(0.0, _timestamp(observed_at) - _timestamp(completed_at))
        if age_s > self.reporter.max_age_s:
            return {"status": "unavailable", "reason": "reporter-frontier-stale", "age_s": age_s}
        family_frontier = frontier.get("families", {}).get(family)
        if not isinstance(family_frontier, Mapping) or family_frontier.get("poll", {}).get("status") != "ok":
            return {"status": "unavailable", "reason": "family-poll-unavailable"}
        sequence = int(frontier["sequence"])
        rows = [entry.get("row") for entry in family_frontier.get("rows", ()) if isinstance(entry, Mapping) and entry.get("last_observed_sequence") == sequence and isinstance(entry.get("row"), Mapping)]
        ratings = [value for row in rows if row.get("is_baseline") is not True and (value := _finite(row.get("rating"))) is not None]
        return {"status": "available", "family_rating_median": statistics.median(ratings) if ratings else None, "visible_agents": len(ratings), "reporter_sequence": sequence, "observed_at": completed_at, "age_s": age_s}

    def capture_game(self, game: Mapping[str, object], *, package_context: object, sensor_frontier: Mapping[str, object] | None, fallback_stats: Mapping[str, object] | None, observed_at: str) -> bool:
        game_id = str(game.get("game_id") or "")
        family = str(game.get("game_family") or "")
        target_player = str(game.get("your_player") or "")
        if not game_id or family not in self.model.families or target_player not in {"player_1", "player_2"}:
            raise ValueError("rating v3 advisory received an unsupported game identity")
        if self.registry.game_context(game_id) is not None:
            return False
        own = self._own_rating(family, sensor_frontier, fallback_stats)
        identity = package_context.get("identity_resolution") if isinstance(package_context, Mapping) and isinstance(package_context.get("identity_resolution"), Mapping) else {}
        public_id = str(identity.get("public_player_id")) if identity.get("status") == "exact-current-label" and identity.get("public_player_id") else None
        opponent = self.reporter.player(family, public_id, now=observed_at) if public_id is not None else {"status": "unavailable", "reason": "identity-not-exact", "identity_resolution": identity.get("status")}
        public_context = self._family_public_context(family, observed_at=observed_at)
        model_context = {
            "identity_scope": _identity_scope(game),
            "family_rating_median": public_context.get("family_rating_median") if public_context.get("status") == "available" else None,
            "opponent_rating_observed": opponent.get("status") == "available",
            "opponent_rating": opponent.get("rating") if opponent.get("status") == "available" else None,
            "traffic_300": 0,
            "traffic_1800": 0,
            "fleet_300": 0,
            "fleet_1800": 0,
            "unavailable_dynamic_features": ["traffic_300", "traffic_1800", "fleet_300", "fleet_1800"],
        }
        context = {
            "contract": RATING_V3_ADVISORY_CONTRACT,
            "game_id": game_id,
            "family": family,
            "target_player": target_player,
            "observed_at": observed_at,
            "self": own,
            "opponent": opponent,
            "public_family_context": public_context,
            "model_context": model_context,
            "visible_game_sha256": _sha(game),
            "prompt_authority": "advisory-only",
            "guard_authority": "none",
        }
        return self.registry.register_game_context(game_id=game_id, family=family, target_player=target_player, observed_at=observed_at, context=context)

    def _predict(self, terminal: Mapping[str, object], context: Mapping[str, object], *, terminal_at: str) -> dict[str, object]:
        own = context.get("self") if isinstance(context.get("self"), Mapping) else {}
        if own.get("status") != "available":
            return {"status": "unavailable", "reason": own.get("reason", "pregame-self-rating-unavailable")}
        opponent = context.get("opponent") if isinstance(context.get("opponent"), Mapping) else {}
        dynamic = context.get("model_context") if isinstance(context.get("model_context"), Mapping) else {}
        return self.model.predict_terminal(
            terminal,
            target_player=str(context["target_player"]),
            target_rating=float(own["rating"]),
            target_games=int(own["games_played"]),
            terminal_at=_timestamp(terminal_at),
            other_rating=float(opponent["rating"]) if opponent.get("status") == "available" else None,
            context=dynamic,
        )

    @staticmethod
    def _bargaining_probability(advisor_handle: object | None, share: float) -> float | None:
        if advisor_handle is None:
            return None
        try:
            evaluation = advisor_handle.rollout.evaluate_offer(advisor_handle.context, share)
        except Exception:
            return None
        probability = _finite(evaluation.get("opponent_accept_probability_conservative")) if isinstance(evaluation, Mapping) else None
        return min(1.0, max(0.0, probability)) if probability is not None else None

    def _bargaining_terminal_prediction(self, game: Mapping[str, object], context: Mapping[str, object], *, opponent_share: float, terminal_at: str) -> dict[str, object]:
        """Predict an accepted branch exactly when visible and marginalize a semantically hidden discount otherwise."""
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        round_number = int(state.get("round") or 1)
        if round_number <= 1 or state.get("complete_information") is True:
            return _compact_prediction(self._predict(_accepted_terminal(game, opponent_share=opponent_share), context, terminal_at=terminal_at))
        your_player = str(game.get("your_player") or "")
        opponent_player = _other_player(your_player)
        own_discount_field = "delta_1" if your_player == "player_1" else "delta_2"
        opponent_discount_field = "delta_1" if opponent_player == "player_1" else "delta_2"
        own_discount = _finite(state.get(own_discount_field))
        if own_discount is None or not 0 < own_discount <= 1:
            return {"status": "unavailable", "reason": "own Bargaining discount factor is unavailable at a discounted round"}
        branches: list[dict[str, object]] = []
        for latent_discount in _BARGAINING_DISCOUNT_SUPPORT:
            branch_game = copy.deepcopy(dict(game))
            branch_state = branch_game.get("game_state") if isinstance(branch_game.get("game_state"), dict) else {}
            branch_game["game_state"] = branch_state
            branch_state[own_discount_field] = own_discount
            branch_state[opponent_discount_field] = latent_discount
            prediction = _compact_prediction(self._predict(_accepted_terminal(branch_game, opponent_share=opponent_share), context, terminal_at=terminal_at))
            branches.append({"opponent_discount": latent_discount, "weight": round(1.0 / len(_BARGAINING_DISCOUNT_SUPPORT), 6), "prediction": prediction})
        available = [branch for branch in branches if branch["prediction"].get("status") == "available"]
        marginalization = {
            "status": "marginalized-hidden-opponent-discount",
            "latent_field": opponent_discount_field,
            "support": list(_BARGAINING_DISCOUNT_SUPPORT),
            "weights": [round(1.0 / len(_BARGAINING_DISCOUNT_SUPPORT), 6)] * len(_BARGAINING_DISCOUNT_SUPPORT),
            "source": "empirical discrete platform support",
            "semantics": "The opponent discount remains hidden; the advisory integrates over every observed platform value and does not promote one guessed value.",
        }
        if len(available) != len(branches):
            return {"status": "unavailable", "reason": "one or more hidden-discount branches are unavailable", "marginalization": marginalization, "branches": branches}
        predictions = [branch["prediction"] for branch in available]
        own_payoffs = [float(prediction["own_payoff"]) for prediction in predictions]
        if max(own_payoffs) - min(own_payoffs) > 1e-6:
            raise RuntimeError("hidden opponent discounts changed the target player's accepted payoff")
        intervals = [prediction.get("interval_80") for prediction in predictions]
        lower = min(float(interval[0]) for interval in intervals if isinstance(interval, Sequence) and len(interval) >= 2)
        upper = max(float(interval[1]) for interval in intervals if isinstance(interval, Sequence) and len(interval) >= 2)
        opponent_payoffs = [float(prediction["opponent_payoff"]) for prediction in predictions]
        deltas = [float(prediction["predicted_self_rating_delta"]) for prediction in predictions]
        return {
            "status": "available",
            "predicted_self_rating_delta": round(statistics.fmean(deltas), 4),
            "predicted_self_rating_delta_range": [round(min(deltas), 4), round(max(deltas), 4)],
            "interval_80": [round(lower, 4), round(upper, 4)],
            "own_payoff": round(statistics.fmean(own_payoffs), 6),
            "opponent_payoff": round(statistics.fmean(opponent_payoffs), 6),
            "opponent_payoff_range": [round(min(opponent_payoffs), 6), round(max(opponent_payoffs), 6)],
            "blended_payoff_percentile": round(statistics.fmean(float(prediction["blended_payoff_percentile"]) for prediction in predictions), 6),
            "global_rank_support": min(int(prediction.get("global_rank_support") or 0) for prediction in predictions),
            "recent_24h_support": min(int(prediction.get("recent_24h_support") or 0) for prediction in predictions),
            "recent_6h_support": min(int(prediction.get("recent_6h_support") or 0) for prediction in predictions),
            "marginalization": marginalization,
            "branches": branches,
        }

    def _bargaining_advisory(self, game: Mapping[str, object], context: Mapping[str, object], *, terminal_at: str, advisor_handle: object | None) -> dict[str, object]:
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or game.get("phase") or "")
        if action_type == "decision":
            last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else {}
            your_player = str(game.get("your_player") or "")
            opponent_player = _other_player(your_player)
            money = _finite(state.get("money_to_divide"))
            opponent_gain = _finite(last_offer.get(f"{opponent_player}_gain"))
            if opponent_gain is None:
                opponent_gain = _finite(last_offer.get("bob_gain" if opponent_player == "player_2" else "alice_gain"))
            if money in (None, 0.0) or opponent_gain is None:
                return {"status": "unavailable", "reason": "current Bargaining offer cannot be normalized"}
            share = opponent_gain / money
            accept = self._bargaining_terminal_prediction(game, context, opponent_share=share, terminal_at=terminal_at)
            final_round = state.get("horizon_known") is True and isinstance(state.get("max_rounds"), int) and int(state.get("round") or 1) >= int(state["max_rounds"])
            reject = _compact_prediction(self._predict(_terminal_no_deal(game), context, terminal_at=terminal_at)) if final_round else {"status": "continuation-omitted"}
            return {"status": "available", "action_type": "decision", "current_offer": {"opponent_share": round(share, 6), "accept": accept, "reject": reject}, "horizon_semantics": "exact terminal branches" if final_round else "accept is terminal; reject continuation is omitted"}
        modeled_share = None
        prompt_context = getattr(advisor_handle, "prompt_context", {}) if advisor_handle is not None else {}
        continuation = prompt_context.get("behavioral_continuation") if isinstance(prompt_context, Mapping) and isinstance(prompt_context.get("behavioral_continuation"), Mapping) else {}
        modeled = continuation.get("modeled_offer_policy") if isinstance(continuation.get("modeled_offer_policy"), Mapping) else {}
        modeled_share = _finite(modeled.get("opponent_share"))
        shares = set(_BARGAINING_SHARES)
        if modeled_share is not None and 0 < modeled_share < 1:
            shares.add(modeled_share)
        rows = []
        for share in sorted(shares):
            prediction = self._bargaining_terminal_prediction(game, context, opponent_share=share, terminal_at=terminal_at)
            probability = self._bargaining_probability(advisor_handle, share)
            delta = _finite(prediction.get("predicted_self_rating_delta"))
            rows.append({"opponent_share": round(share, 6), "if_accepted": prediction, "opponent_accept_probability": round(probability, 6) if probability is not None else None, "myopic_probability_weighted_delta": round(probability * delta, 4) if probability is not None and delta is not None else None})
        rating_rows = [row for row in rows if _finite(row["if_accepted"].get("predicted_self_rating_delta")) is not None]
        weighted_rows = [row for row in rows if _finite(row.get("myopic_probability_weighted_delta")) is not None]
        return {
            "status": "available",
            "action_type": "offer",
            "candidate_surface": rows,
            "highest_if_accepted_delta_share": max(rating_rows, key=lambda row: float(row["if_accepted"]["predicted_self_rating_delta"]))["opponent_share"] if rating_rows else None,
            "highest_myopic_probability_weighted_share": max(weighted_rows, key=lambda row: float(row["myopic_probability_weighted_delta"]))["opponent_share"] if weighted_rows else None,
            "horizon_semantics": "each row is the immediate agreement branch; rejection continuation is omitted",
        }

    @staticmethod
    def _negotiation_candidate_prices(game: Mapping[str, object], advisor_handle: object | None, advisor_context: object) -> list[tuple[float, float | None]]:
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        prices: dict[float, float | None] = {}
        last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else {}
        if (price := _finite(last_offer.get("price"))) is not None and price > 0:
            prices[price] = None
        shadow = getattr(advisor_handle, "forecast_receipt", {}).get("shadow_utility_rollout") if advisor_handle is not None and isinstance(getattr(advisor_handle, "forecast_receipt", {}), Mapping) else {}
        candidates = shadow.get("candidates") if isinstance(shadow, Mapping) and isinstance(shadow.get("candidates"), Sequence) else ()
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or (price := _finite(candidate.get("price"))) is None or price <= 0:
                continue
            response = candidate.get("opponent_response") if isinstance(candidate.get("opponent_response"), Mapping) else {}
            probability = _finite(response.get("weighted"))
            prices[price] = min(1.0, max(0.0, probability)) if probability is not None else prices.get(price)
        facts = advisor_context.get("deterministic_decision_facts") if isinstance(advisor_context, Mapping) and isinstance(advisor_context.get("deterministic_decision_facts"), Mapping) else {}
        for control_name, price_name in (("reciprocal_concession_control", "maximum_next_offer"), ("reciprocal_concession_control", "minimum_next_offer")):
            control = facts.get(control_name) if isinstance(facts.get(control_name), Mapping) else {}
            if (price := _finite(control.get(price_name))) is not None and price > 0:
                prices.setdefault(price, None)
        return sorted(prices.items())

    def _negotiation_advisory(self, game: Mapping[str, object], context: Mapping[str, object], *, terminal_at: str, advisor_handle: object | None, advisor_context: object) -> dict[str, object]:
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        no_deal = _compact_prediction(self._predict(_terminal_no_deal(game), context, terminal_at=terminal_at))
        if state.get("complete_information") is not True or any(_finite(state.get(f"{player}_value")) is None for player in ("player_1", "player_2")):
            return {"status": "partial", "no_deal": no_deal, "agreement_branches": [], "reason": "opponent reservation value is hidden; v3 does not invent the missing terminal payoff", "horizon_semantics": "no-deal is a stop-now diagnostic unless the horizon ends now"}
        rows = []
        for price, probability in self._negotiation_candidate_prices(game, advisor_handle, advisor_context):
            prediction = _compact_prediction(self._predict(_negotiation_agreement_terminal(game, price=price), context, terminal_at=terminal_at))
            delta = _finite(prediction.get("predicted_self_rating_delta"))
            rows.append({"price": round(price, 6), "if_accepted": prediction, "opponent_accept_probability": round(probability, 6) if probability is not None else None, "myopic_probability_weighted_delta": round(probability * delta, 4) if probability is not None and delta is not None else None})
        return {"status": "available", "no_deal": no_deal, "agreement_branches": rows, "horizon_semantics": "agreement rows are exact immediate terminal branches; rejected-offer continuation and future rounds are omitted"}

    def _persuasion_advisory(self, game: Mapping[str, object], context: Mapping[str, object], *, terminal_at: str, advisor_context: object) -> dict[str, object]:
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        pass_prediction = _compact_prediction(self._predict(terminal_scenario(game, bought=False, quality=None), context, terminal_at=terminal_at))
        low_prediction = _compact_prediction(self._predict(terminal_scenario(game, bought=True, quality="low"), context, terminal_at=terminal_at))
        high_prediction = _compact_prediction(self._predict(terminal_scenario(game, bought=True, quality="high"), context, terminal_at=terminal_at))
        result: dict[str, object] = {"status": "available", "pass": pass_prediction, "buy_if_low": low_prediction, "buy_if_high": high_prediction, "horizon_semantics": "current-round stop-now branches; future-round continuation is omitted unless this is the final round"}
        action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or game.get("phase") or "")
        if action_type == "buyer_decision":
            reliability = advisor_context.get("seller_reliability_forecast") if isinstance(advisor_context, Mapping) and isinstance(advisor_context.get("seller_reliability_forecast"), Mapping) else {}
            probability_high = _finite(reliability.get("posterior_high_probability"))
            if probability_high is None:
                probability_high = _finite(state.get("p")) or 0.5
            probability_high = min(1.0, max(0.0, probability_high))
            low_delta = _finite(low_prediction.get("predicted_self_rating_delta"))
            high_delta = _finite(high_prediction.get("predicted_self_rating_delta"))
            expected = probability_high * high_delta + (1 - probability_high) * low_delta if low_delta is not None and high_delta is not None else None
            result["buyer_decision"] = {"posterior_high_probability": round(probability_high, 6), "expected_buy_rating_delta": round(expected, 4) if expected is not None else None, "pass_rating_delta": pass_prediction.get("predicted_self_rating_delta")}
        else:
            response = advisor_context.get("buyer_response_forecasts") if isinstance(advisor_context, Mapping) and isinstance(advisor_context.get("buyer_response_forecasts"), Mapping) else {}
            quality = str(state.get("current_quality") or "").casefold()
            buy_prediction = high_prediction if quality == "high" else low_prediction if quality == "low" else None
            buy_delta = _finite(buy_prediction.get("predicted_self_rating_delta")) if isinstance(buy_prediction, Mapping) else None
            pass_delta = _finite(pass_prediction.get("predicted_self_rating_delta"))
            candidates = {}
            for label in ("positive", "negative"):
                forecast = response.get(label) if isinstance(response.get(label), Mapping) else {}
                probability = _finite(forecast.get("buy_probability"))
                expected = probability * buy_delta + (1 - probability) * pass_delta if probability is not None and buy_delta is not None and pass_delta is not None else None
                candidates[label] = {"buy_probability": round(probability, 6) if probability is not None else None, "myopic_expected_rating_delta": round(expected, 4) if expected is not None else None}
            result["seller_signal_candidates"] = {"current_quality": quality if quality in {"high", "low"} else None, "candidates": candidates}
        return result

    def turn_advisory(self, *, turn_id: str, game: Mapping[str, object], observed_at: str, bargaining_advisor_handle: object | None = None, negotiation_advisor_handle: object | None = None, negotiation_advisor_context: object = None, persuasion_advisor_context: object = None) -> dict[str, object]:
        existing = self.registry.turn_advisory(turn_id)
        if existing is not None:
            return existing
        game_id = str(game.get("game_id") or "")
        context = self.registry.game_context(game_id)
        if context is None:
            raise RuntimeError("rating v3 turn has no immutable pregame context")
        family = str(game.get("game_family") or "")
        if family == "bargaining":
            branches = self._bargaining_advisory(game, context, terminal_at=observed_at, advisor_handle=bargaining_advisor_handle)
        elif family == "negotiation":
            branches = self._negotiation_advisory(game, context, terminal_at=observed_at, advisor_handle=negotiation_advisor_handle, advisor_context=negotiation_advisor_context)
        elif family == "persuasion":
            branches = self._persuasion_advisory(game, context, terminal_at=observed_at, advisor_context=persuasion_advisor_context)
        else:
            raise ValueError(f"unsupported rating v3 advisory family: {family}")
        advisory = {
            "contract": RATING_V3_ADVISORY_CONTRACT,
            "schema_version": RATING_V3_ADVISORY_SCHEMA_VERSION,
            "status": branches.get("status", "available"),
            "model_version": RATING_V3_MODEL_VERSION,
            "release": self.release["release"],
            "frontier": "registered-before-model-inference",
            "game_id": game_id,
            "turn_id": turn_id,
            "family": family,
            "branches": branches,
            "instruction": "Treat these as fallible rating-objective evidence alongside payoff and opponent-response evidence. Do not optimize an immediate branch as if omitted continuation were zero.",
            "prompt_authority": "advisory-only",
            "guard_authority": "none",
        }
        self.registry.register_turn_advisory(turn_id=turn_id, game_id=game_id, observed_at=observed_at, advisory=advisory)
        return advisory

    def _observe_forecast(self, record: Mapping[str, object]) -> bool:
        game_id = str(record.get("game_id") or "")
        if not game_id or game_id in self._observed_terminal_games:
            return False
        family = str(record.get("family") or "")
        forecast = record.get("forecast") if isinstance(record.get("forecast"), Mapping) else {}
        if forecast.get("status") != "available" or family not in self.model.families:
            return False
        self.model.families[family].payoff_index.observe(str(forecast["configuration_sha256"]), float(forecast["own_payoff"]), _timestamp(str(record["terminal_at"])))
        self._observed_terminal_games.add(game_id)
        return True

    def _replay_registered_terminals(self) -> None:
        for record in self.registry.terminal_forecasts():
            self._observe_forecast(record)

    def register_terminal(self, terminal: Mapping[str, object], *, terminal_at: str) -> bool:
        game_id = str(terminal.get("game_id") or "")
        existing = self.registry.terminal_forecast(game_id)
        if existing is not None:
            if existing.get("terminal_sha256") != _sha(terminal):
                raise RuntimeError(f"conflicting rating v3 terminal retry: {game_id}")
            self._observe_forecast(existing)
            return False
        context = self.registry.game_context(game_id)
        if context is None:
            raise RuntimeError("rating v3 terminal has no immutable pregame context")
        forecast = self._predict(terminal, context, terminal_at=terminal_at)
        if forecast.get("status") != "available":
            raise RuntimeError(f"rating v3 terminal forecast unavailable: {forecast.get('reason')}")
        registered_at = _now()
        inserted = self.registry.register(game_id=game_id, family=str(terminal["game_family"]), target_player=str(context["target_player"]), terminal_at=terminal_at, registered_at=registered_at, terminal_sha256=_sha(terminal), forecast=forecast)
        self._observe_forecast({"game_id": game_id, "family": terminal["game_family"], "terminal_at": terminal_at, "forecast": forecast})
        return inserted

    def reconcile(self, *, force: bool = False) -> dict[str, object]:
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - self._last_reconcile < 10.0:
            return {"status": "throttled", "matured": 0}
        self._last_reconcile = now
        if not self.history_path.is_file():
            return {"status": "unavailable", "reason": "rating-history-database-missing", "matured": 0}
        return {"status": "available", **self.registry.mature_from_history(self.history_path)}

    def status(self) -> dict[str, object]:
        return {**self.manifest_receipt(), "registry": self.registry.summary(), "replayed_terminal_games": len(self._observed_terminal_games)}

    def close(self) -> None:
        return None
