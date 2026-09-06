"""Bounded v2.17 Bargaining intervention over monotone identity packages and a frozen rating model."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from .glee_policy import normalize_action
from .glee_statistical_package import STATISTICAL_DECISION_FORECAST_CONTRACT, _isotonic_nondecreasing


INTERVENTION_CONTRACT = "glee-bargaining-intervention-v2.17"
MINIMUM_PACKAGE_SUPPORT = 2.0
MAXIMUM_PACKAGE_BLEND_WEIGHT = 0.35
MINIMUM_PACKAGE_VALUE_IMPROVEMENT = 0.02
MINIMUM_NONTERMINAL_OWN_SHARE = 0.20
EXTREME_OPPONENT_SHARE = 0.05
CONTINUATION_TOLERANCE = 0.01
RATING_POINT_CATASTROPHE = -5.0

_EMBEDDED_POLICY = {
    "schema_version": 1,
    "contract": "glee-bargaining-live-policy-v1",
    "revision": "embedded-bargaining-v2.17",
    "parameters": {
        "minimum_package_support": MINIMUM_PACKAGE_SUPPORT,
        "maximum_package_blend_weight": MAXIMUM_PACKAGE_BLEND_WEIGHT,
        "minimum_package_value_improvement": MINIMUM_PACKAGE_VALUE_IMPROVEMENT,
        "minimum_nonterminal_own_share": MINIMUM_NONTERMINAL_OWN_SHARE,
        "extreme_opponent_share": EXTREME_OPPONENT_SHARE,
        "continuation_tolerance": CONTINUATION_TOLERANCE,
        "rating_point_catastrophe": RATING_POINT_CATASTROPHE,
    },
    "features": {"selected_curve_comparator": "nearest-grid", "patient_acceptance_guard": "continuation-nondominated"},
}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _live_policy(context: Mapping[str, object]) -> Mapping[str, object]:
    policy = context.get("live_policy")
    return policy if isinstance(policy, Mapping) and policy.get("contract") == "glee-bargaining-live-policy-v1" and isinstance(policy.get("parameters"), Mapping) and isinstance(policy.get("features"), Mapping) else _EMBEDDED_POLICY


def _parameter(policy: Mapping[str, object], name: str) -> float:
    parameters = policy.get("parameters") if isinstance(policy.get("parameters"), Mapping) else {}
    value = _finite(parameters.get(name))
    fallback = _finite(_EMBEDDED_POLICY["parameters"][name])
    if value is None or fallback is None:
        raise RuntimeError(f"Bargaining live-policy parameter {name} is unavailable")
    return value


def _state(game: Mapping[str, object]) -> Mapping[str, object]:
    return game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}


def _player(game: Mapping[str, object]) -> str:
    state = _state(game)
    return str(game.get("your_player") or state.get("current_player") or "")


def _gain(source: Mapping[str, object], player: str) -> float | None:
    keys = ("player_1_gain", "alice_gain") if player in {"player_1", "alice"} else ("player_2_gain", "bob_gain") if player in {"player_2", "bob"} else ()
    return next((numeric for key in keys if (numeric := _finite(source.get(key))) is not None), None)


def _other(player: str) -> str:
    return "player_2" if player in {"player_1", "alice"} else "player_1"


def _opponent_share(game: Mapping[str, object], action: Mapping[str, object]) -> float | None:
    state = _state(game)
    money = _finite(state.get("money_to_divide"))
    player = _player(game)
    if money is None or money <= 0 or player not in {"player_1", "player_2", "alice", "bob"}:
        return None
    source = state.get("last_offer") if str(action.get("decision") or "").casefold() in {"accept", "acceptoffer"} and isinstance(state.get("last_offer"), Mapping) else action
    gain = _gain(source, _other(player))
    return gain / money if gain is not None else None


def _own_share_of_current_offer(game: Mapping[str, object]) -> float | None:
    state = _state(game)
    offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else {}
    money = _finite(state.get("money_to_divide"))
    gain = _gain(offer, _player(game))
    return gain / money if money is not None and money > 0 and gain is not None else None


def _offer_for_opponent_share(game: Mapping[str, object], share: float) -> dict[str, Any] | None:
    state = _state(game)
    money = _finite(state.get("money_to_divide"))
    player = _player(game)
    if money is None or money <= 0 or player not in {"player_1", "player_2", "alice", "bob"}:
        return None
    bounded = min(1.0, max(0.0, float(share)))
    if player in {"player_1", "alice"}:
        candidate = {"alice_gain": money * (1.0 - bounded), "bob_gain": money * bounded}
    else:
        candidate = {"alice_gain": money * bounded, "bob_gain": money * (1.0 - bounded)}
    return normalize_action(dict(game), candidate)


def _policy_bounds(advisor_context: object) -> tuple[float, float, bool]:
    context = advisor_context if isinstance(advisor_context, Mapping) else {}
    continuation = context.get("behavioral_continuation") if isinstance(context.get("behavioral_continuation"), Mapping) else {}
    guard = continuation.get("policy_guard") if isinstance(continuation.get("policy_guard"), Mapping) else {}
    minimum = _finite(guard.get("minimum_opponent_share"))
    maximum = _finite(guard.get("maximum_opponent_share"))
    loss = guard.get("loss_minimization_control") if isinstance(guard.get("loss_minimization_control"), Mapping) else {}
    return max(0.0, minimum if minimum is not None else 0.0), min(1.0, maximum if maximum is not None else 1.0), loss.get("status") == "active"


def _package_curve(package_context: object, advisor_handle: object, advisor_context: object, policy: Mapping[str, object]) -> dict[str, object]:
    package = package_context if isinstance(package_context, Mapping) else {}
    local = package.get("decision_local_model") if isinstance(package.get("decision_local_model"), Mapping) else {}
    identity = package.get("identity_resolution") if isinstance(package.get("identity_resolution"), Mapping) else {}
    rows = local.get("response_to_our_offer") if local.get("contract") == STATISTICAL_DECISION_FORECAST_CONTRACT and isinstance(local.get("response_to_our_offer"), list) else []
    exact_identity = identity.get("status") == "exact-current-label" and bool(identity.get("public_player_id"))
    minimum_support = _parameter(policy, "minimum_package_support")
    maximum_blend_weight = _parameter(policy, "maximum_package_blend_weight")
    minimum_value_improvement = _parameter(policy, "minimum_package_value_improvement")
    minimum, maximum, loss_active = _policy_bounds(advisor_context)
    projected: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        share = _finite(raw.get("representative_opponent_share"))
        package_probability = _finite(raw.get("hierarchical_identity_accept_probability"))
        if share is None or package_probability is None:
            continue
        behavior: Mapping[str, object] = {}
        try:
            behavior = advisor_handle.rollout.evaluate_offer(advisor_handle.context, share)
        except Exception:
            behavior = {}
        specialized_probability = _finite(behavior.get("opponent_accept_probability_conservative"))
        accepted_value = _finite(behavior.get("accepted_value"))
        rejected_value = _finite(behavior.get("rejected_path_value"))
        specialized_value = _finite(behavior.get("expected_value"))
        exact_support = float(raw.get("exact_direct_support") or 0.0)
        nearby_support = float(raw.get("nearby_direct_effective_support") or 0.0)
        support = exact_support + nearby_support
        eligible = exact_identity and raw.get("evidence_scope") != "population-only" and support >= minimum_support and specialized_probability is not None and accepted_value is not None and rejected_value is not None and minimum - 1e-12 <= share <= maximum + 1e-12 and not loss_active
        blend_weight = min(maximum_blend_weight, maximum_blend_weight * support / 5.0) if eligible else 0.0
        projected.append(
            {
                "opponent_share": round(share, 6),
                "evidence_scope": raw.get("evidence_scope"),
                "exact_direct_support": int(exact_support),
                "nearby_direct_effective_support": round(nearby_support, 6),
                "effective_support": round(support, 6),
                "package_probability_raw": raw.get("raw_hierarchical_identity_accept_probability"),
                "package_probability_monotone": round(package_probability, 6),
                "specialized_probability_raw": round(specialized_probability, 6) if specialized_probability is not None else None,
                "package_blend_weight": round(blend_weight, 6),
                "accepted_value": round(accepted_value, 6) if accepted_value is not None else None,
                "rejected_path_value": round(rejected_value, 6) if rejected_value is not None else None,
                "specialized_expected_value_raw": round(specialized_value, 6) if specialized_value is not None else None,
                "authority_eligible": eligible,
            }
        )
    projected.sort(key=lambda row: float(row["opponent_share"]))
    probability_rows = [row for row in projected if _finite(row.get("specialized_probability_raw")) is not None]
    specialized_monotone = _isotonic_nondecreasing([float(row["specialized_probability_raw"]) for row in probability_rows], [1.0] * len(probability_rows)) if probability_rows else []
    blended_raw = []
    for row, specialized_probability in zip(probability_rows, specialized_monotone, strict=True):
        package_probability = float(row["package_probability_monotone"])
        weight = float(row["package_blend_weight"])
        row["specialized_probability_monotone"] = round(specialized_probability, 6)
        blended_raw.append((1.0 - weight) * specialized_probability + weight * package_probability)
    blended_monotone = _isotonic_nondecreasing(blended_raw, [max(1.0, float(row.get("effective_support") or 0.0)) for row in probability_rows]) if probability_rows else []
    for row, blended_probability in zip(probability_rows, blended_monotone, strict=True):
        accepted_value = float(row["accepted_value"])
        rejected_value = float(row["rejected_path_value"])
        specialized_probability = float(row["specialized_probability_monotone"])
        row["blended_probability"] = round(blended_probability, 6)
        row["specialized_expected_value"] = round(specialized_probability * accepted_value + (1.0 - specialized_probability) * rejected_value, 6)
        row["blended_expected_value"] = round(blended_probability * accepted_value + (1.0 - blended_probability) * rejected_value, 6)
        row["policy_curve_projection"] = "specialized and final blended acceptance probabilities are nondecreasing in opponent share"
    eligible_rows = [row for row in projected if row.get("authority_eligible") is True and _finite(row.get("blended_expected_value")) is not None]
    recommendation = max(eligible_rows, key=lambda row: (float(row["blended_expected_value"]), float(row.get("blended_probability") or 0.0), -float(row["opponent_share"]))) if eligible_rows else None
    return {
        "identity_resolution": deepcopy(dict(identity)),
        "identity_exact": exact_identity,
        "policy_bounds": {"minimum_opponent_share": minimum, "maximum_opponent_share": maximum, "loss_minimization_active": loss_active},
        "curve": projected,
        "recommendation": deepcopy(recommendation),
        "authority": "bounded-offer-candidate-scoring" if recommendation is not None else "diagnostic-only",
        "gate": {"minimum_effective_support": minimum_support, "maximum_blend_weight": maximum_blend_weight, "minimum_expected_value_improvement": minimum_value_improvement},
    }


def _rating_surface(game: Mapping[str, object], advisor_handle: object, rating_canary: object | None, *, observed_at: str) -> dict[str, object]:
    if rating_canary is None:
        return {"status": "unavailable", "reason": "legacy-rating-authority-disabled"}
    try:
        context = rating_canary.registry.game_context(str(game.get("game_id") or ""))
        if not isinstance(context, Mapping):
            return {"status": "unavailable", "reason": "pregame-rating-context-unavailable"}
        action = {"decision": "accept"} if str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or "") == "decision" else {}
        rows, recommendations = rating_canary._candidate_surface(game, action, advisor_handle, context, terminal_at=observed_at)
    except Exception as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return {
        "status": "available",
        "source_seed_sha256": getattr(getattr(rating_canary, "predictor", None), "seed_sha256", None),
        "candidate_surface": rows,
        "recommendations": recommendations,
        "pregame_context_status": "available",
    }


def build_bargaining_v217_context(*, game: Mapping[str, object], package_context: object, advisor_handle: object, advisor_context: object, rating_canary: object | None, observed_at: str, live_policy: Mapping[str, object] | None = None) -> dict[str, object]:
    """Freeze all v2.17 intervention inputs before model inference and before the current action is selected."""
    policy = deepcopy(dict(live_policy)) if isinstance(live_policy, Mapping) else deepcopy(_EMBEDDED_POLICY)
    return {
        "contract": INTERVENTION_CONTRACT,
        "frontier": "current-turn-before-model-inference",
        "live_policy": policy,
        "package": _package_curve(package_context, advisor_handle, advisor_context, policy),
        "rating": _rating_surface(game, advisor_handle, rating_canary, observed_at=observed_at),
        "authority_scope": {"package": "exact-identity offer candidate scoring only", "rating": "strongly negative nonterminal acceptance only", "nominal_fallback": "sub-20-percent acceptance when rating is unavailable and rejection continuation is non-dominated"},
    }


def _nearest_curve_row(context: Mapping[str, object], share: float) -> Mapping[str, object] | None:
    package = context.get("package") if isinstance(context.get("package"), Mapping) else {}
    rows = package.get("curve") if isinstance(package.get("curve"), list) else []
    eligible = [row for row in rows if isinstance(row, Mapping) and _finite(row.get("opponent_share")) is not None]
    return min(eligible, key=lambda row: abs(float(row["opponent_share"]) - share)) if eligible else None


def _curve_value(context: Mapping[str, object], share: float, field: str, comparator: str) -> float | None:
    package = context.get("package") if isinstance(context.get("package"), Mapping) else {}
    raw_rows = package.get("curve") if isinstance(package.get("curve"), list) else []
    rows = sorted(((float(row["opponent_share"]), value) for row in raw_rows if isinstance(row, Mapping) and (value := _finite(row.get(field))) is not None and _finite(row.get("opponent_share")) is not None), key=lambda pair: pair[0])
    if not rows:
        return None
    if comparator != "linear-interpolation":
        return min(rows, key=lambda pair: abs(pair[0] - share))[1]
    if share <= rows[0][0]:
        return rows[0][1]
    if share >= rows[-1][0]:
        return rows[-1][1]
    for (lower_share, lower_value), (upper_share, upper_value) in zip(rows, rows[1:]):
        if lower_share <= share <= upper_share:
            if math.isclose(lower_share, upper_share, rel_tol=0.0, abs_tol=1e-12):
                return lower_value
            weight = (share - lower_share) / (upper_share - lower_share)
            return lower_value + weight * (upper_value - lower_value)
    return None


def _rejected_own_offer_shares(game: Mapping[str, object]) -> list[float]:
    state = _state(game)
    history = state.get("history") if isinstance(state.get("history"), list) else []
    player = _player(game)
    money = _finite(state.get("money_to_divide"))
    if money is None or money <= 0:
        return []
    result = []
    for entry in history:
        if not isinstance(entry, Mapping) or str(entry.get("decision") or "").casefold() != "reject":
            continue
        offer = entry.get("offer") if isinstance(entry.get("offer"), Mapping) else {}
        if str(entry.get("proposer") or offer.get("proposer") or "") != player:
            continue
        gain = _gain(offer, _other(player))
        if gain is not None:
            result.append(gain / money)
    return result


def _later_round_available(game: Mapping[str, object]) -> bool:
    state = _state(game)
    round_number = state.get("round")
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True and isinstance(round_number, int) and not isinstance(round_number, bool) and isinstance(maximum, int) and not isinstance(maximum, bool):
        return round_number < maximum
    return not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 99


def _replacement_share(context: Mapping[str, object], advisor_context: object, rejected: list[float], policy: Mapping[str, object]) -> float:
    package = context.get("package") if isinstance(context.get("package"), Mapping) else {}
    recommendation = package.get("recommendation") if isinstance(package.get("recommendation"), Mapping) else {}
    candidates = [_finite(recommendation.get("opponent_share"))]
    advisor = advisor_context if isinstance(advisor_context, Mapping) else {}
    continuation = advisor.get("behavioral_continuation") if isinstance(advisor.get("behavioral_continuation"), Mapping) else {}
    modeled = continuation.get("modeled_offer_policy") if isinstance(continuation.get("modeled_offer_policy"), Mapping) else {}
    response = advisor.get("response_to_our_numeric_offer") if isinstance(advisor.get("response_to_our_numeric_offer"), Mapping) else {}
    myopic = response.get("myopic_no_continuation_candidate") if isinstance(response.get("myopic_no_continuation_candidate"), Mapping) else {}
    candidates.extend((_finite(modeled.get("opponent_share")), _finite(myopic.get("opponent_share")), 0.5))
    minimum, maximum, _loss = _policy_bounds(advisor_context)
    extreme_opponent_share = _parameter(policy, "extreme_opponent_share")
    minimum_nonterminal_own_share = _parameter(policy, "minimum_nonterminal_own_share")
    safe_minimum = max(extreme_opponent_share, minimum)
    safe_maximum = min(1.0 - minimum_nonterminal_own_share, maximum)
    if safe_minimum > safe_maximum:
        safe_minimum, safe_maximum = extreme_opponent_share, 1.0 - minimum_nonterminal_own_share
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate < safe_minimum - 1e-12 or candidate > safe_maximum + 1e-12:
            continue
        bounded = candidate
        if not any(math.isclose(bounded, previous, rel_tol=0.0, abs_tol=0.005) and (previous <= extreme_opponent_share + 1e-12 or previous >= 1.0 - minimum_nonterminal_own_share - 1e-12) for previous in rejected):
            return bounded
    return min(safe_maximum, max(safe_minimum, 0.5))


def _terminal_positive_policy(advisor_context: object) -> bool:
    advisor = advisor_context if isinstance(advisor_context, Mapping) else {}
    continuation = advisor.get("behavioral_continuation") if isinstance(advisor.get("behavioral_continuation"), Mapping) else {}
    guard = continuation.get("policy_guard") if isinstance(continuation.get("policy_guard"), Mapping) else {}
    return str(guard.get("force_accept_reason") or "") in {"positive payoff inside the observed round-99 terminal window", "positive known-final-round payoff"}


def _response_rating_loss_assessment(context: Mapping[str, object], policy: Mapping[str, object]) -> dict[str, object]:
    rating = context.get("rating") if isinstance(context.get("rating"), Mapping) else {}
    rows = rating.get("candidate_surface") if isinstance(rating.get("candidate_surface"), list) else []
    row = rows[0] if len(rows) == 1 and isinstance(rows[0], Mapping) else next((value for value in rows if isinstance(value, Mapping) and value.get("submitted") is True), None)
    self_forecast = row.get("forecasts_if_accepted", {}).get("self", {}) if isinstance(row, Mapping) and isinstance(row.get("forecasts_if_accepted"), Mapping) and isinstance(row.get("forecasts_if_accepted", {}).get("self"), Mapping) else {}
    point = _finite(self_forecast.get("predicted_delta"))
    interval = self_forecast.get("interval_80") if isinstance(self_forecast.get("interval_80"), list) else []
    upper = _finite(interval[1]) if len(interval) == 2 else None
    catastrophe = _parameter(policy, "rating_point_catastrophe")
    strong_loss = self_forecast.get("status") == "available" and ((point is not None and point <= catastrophe) or (point is not None and point < 0 and upper is not None and upper < 0))
    features = policy.get("features") if isinstance(policy.get("features"), Mapping) else {}
    mode = str(features.get("response_rating_loss_guard") or "bounded-authoritative")
    return {
        "mode": mode,
        "signal": "strong-rating-loss" if strong_loss else "none",
        "would_reject_under_v2_20": strong_loss,
        "predicted_delta": point,
        "interval_80_upper": upper,
        "catastrophe_threshold": catastrophe,
    }


def _rating_v3_low_share_loss_assessment(envelope: object, policy: Mapping[str, object], own_share: float | None) -> dict[str, object]:
    """Read one current-offer v3 branch without treating omitted continuation as zero."""
    features = policy.get("features") if isinstance(policy.get("features"), Mapping) else {}
    mode = str(features.get("rating_v3_low_share_guard") or "shadow-only")
    advisory = getattr(envelope, "rating_v3_advisory", None)
    branches = advisory.get("branches") if isinstance(advisory, Mapping) and advisory.get("family") == "bargaining" else {}
    current = branches.get("current_offer") if isinstance(branches, Mapping) and branches.get("action_type") == "decision" and isinstance(branches.get("current_offer"), Mapping) else {}
    accept = current.get("accept") if isinstance(current.get("accept"), Mapping) else {}
    point = _finite(accept.get("predicted_self_rating_delta"))
    interval = accept.get("interval_80") if isinstance(accept.get("interval_80"), (list, tuple)) else ()
    upper = _finite(interval[1]) if len(interval) == 2 else None
    low_share = own_share is not None and own_share < 0.5 - 1e-12
    bounded_loss = accept.get("status") == "available" and point is not None and point < 0 and upper is not None and upper < 0
    return {
        "mode": mode,
        "status": "available" if accept.get("status") == "available" else "unavailable",
        "own_share": own_share,
        "low_share": low_share,
        "predicted_delta": point,
        "interval_80_upper": upper,
        "bounded_loss": bounded_loss,
        "would_reject": mode == "bounded-authoritative" and low_share and bounded_loss,
        "continuation_boundary": "evaluated only after behavioral rejection is non-dominated",
    }


def apply_bargaining_v217_intervention(envelope: object, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply the frozen package and response rules after every inherited policy branch."""
    game = getattr(envelope, "game", {})
    context = getattr(envelope, "bargaining_intervention_context", None)
    if not isinstance(game, Mapping) or game.get("game_family") != "bargaining" or not isinstance(context, Mapping) or context.get("contract") != INTERVENTION_CONTRACT:
        return candidate, []
    advisor_context = getattr(envelope, "bargaining_advisor_context", None)
    policy = _live_policy(context)
    features = policy.get("features") if isinstance(policy.get("features"), Mapping) else {}
    minimum_value_improvement = _parameter(policy, "minimum_package_value_improvement")
    minimum_nonterminal_own_share = _parameter(policy, "minimum_nonterminal_own_share")
    extreme_opponent_share = _parameter(policy, "extreme_opponent_share")
    continuation_tolerance = _parameter(policy, "continuation_tolerance")
    action = dict(candidate)
    applied: list[str] = []
    action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or "")
    if action_type == "offer":
        selected_share = _opponent_share(game, action)
        package = context.get("package") if isinstance(context.get("package"), Mapping) else {}
        recommendation = package.get("recommendation") if isinstance(package.get("recommendation"), Mapping) else {}
        recommended_share = _finite(recommendation.get("opponent_share"))
        recommended_value = _finite(recommendation.get("blended_expected_value"))
        if selected_share is not None and recommended_share is not None and recommended_value is not None and package.get("authority") == "bounded-offer-candidate-scoring":
            comparator = str(features.get("selected_curve_comparator") or "nearest-grid")
            selected_value = _curve_value(context, selected_share, "blended_expected_value", comparator)
            if selected_value is None:
                selected_value = _curve_value(context, selected_share, "specialized_expected_value", comparator)
            if selected_value is not None and recommended_value > selected_value + minimum_value_improvement + 1e-12 and not math.isclose(selected_share, recommended_share, rel_tol=0.0, abs_tol=0.005):
                replacement = _offer_for_opponent_share(game, recommended_share)
                if replacement is not None:
                    action = replacement
                    selected_share = recommended_share
                    applied.append("bargaining_v217_exact_package_offer_candidate")
        rejected = _rejected_own_offer_shares(game)
        if selected_share is not None and rejected and selected_share > 1.0 - minimum_nonterminal_own_share + 1e-12 and _later_round_available(game):
            replacement = _offer_for_opponent_share(game, _replacement_share(context, advisor_context, rejected, policy))
            if replacement is not None:
                action = replacement
                selected_share = _opponent_share(game, action)
                applied.append("bargaining_v217_post_rejection_capitulation_guard")
        repeated_extreme = selected_share is not None and any(math.isclose(selected_share, previous, rel_tol=0.0, abs_tol=0.005) for previous in rejected) and (selected_share <= extreme_opponent_share + 1e-12 or selected_share >= 1.0 - minimum_nonterminal_own_share - 1e-12)
        if repeated_extreme and _later_round_available(game):
            replacement = _offer_for_opponent_share(game, _replacement_share(context, advisor_context, rejected, policy))
            if replacement is not None and not math.isclose(float(_opponent_share(game, replacement) or -1.0), float(selected_share), rel_tol=0.0, abs_tol=0.005):
                action = replacement
                applied.append("bargaining_v217_rejected_extreme_offer_repetition_guard")
        return action, applied
    if action_type != "decision" or not _later_round_available(game) or _terminal_positive_policy(advisor_context):
        return action, applied
    advisor = advisor_context if isinstance(advisor_context, Mapping) else {}
    continuation = advisor.get("behavioral_continuation") if isinstance(advisor.get("behavioral_continuation"), Mapping) else {}
    comparison = continuation.get("decision_comparison") if isinstance(continuation.get("decision_comparison"), Mapping) else {}
    accept_value = _finite(comparison.get("accept_now_value"))
    reject_value = _finite(comparison.get("reject_value"))
    consistency = continuation.get("consistency_check") if isinstance(continuation.get("consistency_check"), Mapping) else {}
    coherence_mode = str(features.get("patient_acceptance_guard") or "continuation-nondominated")
    if str(action.get("decision") or "").casefold() in {"reject", "rejectoffer"} and coherence_mode in {"accept-if-next-settlement-no-better", "bidirectional-continuation-coherence"} and consistency.get("modeled_next_offer_same_or_worse_for_us") is True and accept_value is not None and reject_value is not None and accept_value > reject_value + continuation_tolerance:
        return normalize_action(dict(game), {"decision": "accept"}), ["bargaining_live_policy_patient_no_better_next_settlement_guard"]
    if str(action.get("decision") or "").casefold() not in {"accept", "acceptoffer"}:
        return action, applied
    if coherence_mode == "bidirectional-continuation-coherence" and accept_value is not None and reject_value is not None and reject_value > accept_value + max(continuation_tolerance, minimum_value_improvement) + 1e-12:
        return normalize_action(dict(game), {"decision": "reject"}), ["bargaining_live_policy_materially_dominated_acceptance_guard"]
    continuation_non_dominated = accept_value is not None and reject_value is not None and reject_value + continuation_tolerance >= accept_value
    if not continuation_non_dominated:
        return action, applied
    rating_loss = _response_rating_loss_assessment(context, policy)
    if rating_loss["would_reject_under_v2_20"] is True and rating_loss["mode"] == "bounded-authoritative":
        return normalize_action(dict(game), {"decision": "reject"}), ["bargaining_v217_response_rating_loss_guard"]
    own_share = _own_share_of_current_offer(game)
    rating_v3_loss = _rating_v3_low_share_loss_assessment(envelope, policy, own_share)
    if rating_v3_loss["would_reject"] is True:
        return normalize_action(dict(game), {"decision": "reject"}), ["bargaining_v3_low_share_rating_loss_guard"]
    rating = context.get("rating") if isinstance(context.get("rating"), Mapping) else {}
    rows = rating.get("candidate_surface") if isinstance(rating.get("candidate_surface"), list) else []
    row = rows[0] if len(rows) == 1 and isinstance(rows[0], Mapping) else next((value for value in rows if isinstance(value, Mapping) and value.get("submitted") is True), None)
    self_forecast = row.get("forecasts_if_accepted", {}).get("self", {}) if isinstance(row, Mapping) and isinstance(row.get("forecasts_if_accepted"), Mapping) and isinstance(row.get("forecasts_if_accepted", {}).get("self"), Mapping) else {}
    if self_forecast.get("status") != "available" and own_share is not None and own_share < minimum_nonterminal_own_share - 1e-12:
        return normalize_action(dict(game), {"decision": "reject"}), ["bargaining_v217_response_nominal_share_fallback"]
    return action, applied


def intervention_receipt(envelope: object, decision: object) -> dict[str, object] | None:
    """Return one compact causal receipt joining frozen inputs to the selected outward action."""
    context = getattr(envelope, "bargaining_intervention_context", None)
    if not isinstance(context, Mapping) or context.get("contract") != INTERVENTION_CONTRACT:
        return None
    safeguards = list(getattr(decision, "deterministic_safeguards", []) or [])
    game = getattr(envelope, "game", {})
    action = getattr(decision, "action", {})
    selected_share = _opponent_share(game, action) if isinstance(game, Mapping) and isinstance(action, Mapping) else None
    package = context.get("package") if isinstance(context.get("package"), Mapping) else {}
    selected_package_row = _nearest_curve_row(context, selected_share) if selected_share is not None else None
    policy = _live_policy(context)
    features = policy.get("features") if isinstance(policy.get("features"), Mapping) else {}
    comparator = str(features.get("selected_curve_comparator") or "nearest-grid")
    selected_package_value = _curve_value(context, selected_share, "blended_expected_value", comparator) if selected_share is not None else None
    rating = context.get("rating") if isinstance(context.get("rating"), Mapping) else {}
    rating_rows = rating.get("candidate_surface") if isinstance(rating.get("candidate_surface"), list) else []
    response_rating = rating_rows[0] if len(rating_rows) == 1 and isinstance(rating_rows[0], Mapping) else None
    response_rating_guard = _response_rating_loss_assessment(context, policy)
    own_share = _own_share_of_current_offer(game) if isinstance(game, Mapping) else None
    rating_v3_guard = _rating_v3_low_share_loss_assessment(envelope, policy, own_share)
    return {
        "contract": INTERVENTION_CONTRACT,
        "frontier": context.get("frontier"),
        "live_policy": {"revision": policy.get("revision"), "release_sha256": policy.get("release_sha256"), "features": deepcopy(features)},
        "action": deepcopy(action),
        "proposal": deepcopy(getattr(decision, "proposal", None)),
        "interventions": [value for value in safeguards if str(value).startswith(("bargaining_v217_", "bargaining_v3_", "bargaining_live_policy_"))],
        "package": {"identity_resolution": deepcopy(package.get("identity_resolution")), "authority": package.get("authority"), "gate": deepcopy(package.get("gate")), "recommendation": deepcopy(package.get("recommendation")), "selected_curve_comparator": comparator, "selected_interpolated_value": selected_package_value, "selected_row": deepcopy(selected_package_row)},
        "rating": {"status": rating.get("status"), "source_seed_sha256": rating.get("source_seed_sha256"), "response_candidate": deepcopy(response_rating), "response_guard": response_rating_guard, "rating_v3_low_share_guard": rating_v3_guard, "recommendations": deepcopy(rating.get("recommendations"))},
    }
