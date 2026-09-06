"""Stateless family-routed turn solvers for the parallel GLEE supervisor."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable, Mapping

from .glee_advisor_contracts import NEGOTIATION_ADVISOR_MODEL_VERSION, NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE, PERSUASION_ADVISOR_MODEL_VERSION
from .glee_bargaining_intervention import apply_bargaining_v217_intervention
from .glee_dossier import DossierSnapshot
from .glee_message_style import opponent_message_policy_signal
from .glee_meta_controller_v2 import FAMILY_SELECTOR_POLICY_CONTRACT, ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT, SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT, FrozenCandidateSet, GuardedSingleAction, build_conditional_surface, build_family_candidate_evidence, build_persuasion_buyer_continuation_selector_payload, build_planner_payload_v15, build_public_self_mirror_admissibility, build_selector_payload_v15, freeze_fixed_candidates, freeze_planner_candidates, negotiation_reject_counteroffer_action, negotiation_reject_counteroffer_frontier
from .glee_nommd import glee_action_model, nommd_candidate_plan_model
from .glee_policy import action_model, analytic_bargaining_opening, analytic_bargaining_reference, apply_deterministic_safeguards, normalize_action, safe_action
from .glee_sequence_shadow import terra_synthetic_feature_bundle
from .glee_selector_backend import SelectorBackend, TerraSelectorBackend, build_selector_backend_request
from .glee_semantics import bargaining_plateau_summary, bargaining_semantic_staircase_summary, canonical_bargaining_facts, model_game_semantics, model_official_prompt, model_visible_game_state
from .glee_statistical_package import _isotonic_nondecreasing, compact_opponent_decision_forecast
from .immutable_pack import object_sha256
from .models import TetradUpdate

FAMILY_ROLES = {
    "bargaining": "glee_nommd_bargaining",
    "negotiation": "glee_nommd_negotiation",
    "persuasion": "glee_nommd_persuasion",
}

CAPACITY_MODEL_CHAIN = (
    ("terra-high", "gpt-5.6-terra", "high"),
    ("luna-high", "gpt-5.6-luna", "high"),
)

BARGAINING_CAPACITY_MODEL_CHAIN = CAPACITY_MODEL_CHAIN
BARGAINING_ADVISOR_MODEL_VERSION = "bargaining-live-advisor-v2.17"

_CODEX_MODEL_CAPACITY_MESSAGE = "selected model is at capacity"
_BARGAINING_MINIMUM_MATERIAL_OWN_SHARE = 0.05
_BARGAINING_SEMANTIC_STAIRCASE_MAXIMUM_RECOMMENDATION_SHIFT = 0.02
_PERSUASION_ADVISORY_ANCHOR_AUTHORITY = "advisory-anchor"
_PERSUASION_BOUNDED_AUTHORITY = "bounded-authoritative"
_BARGAINING_NUMERIC_OVERRIDE_MINIMUM_TWIN_GAIN = 0.02
_BARGAINING_NUMERIC_OVERRIDE_MINIMUM_SEQUENCE_GAIN = 0.01
_BARGAINING_NUMERIC_OVERRIDE_MAXIMUM_COMPONENT_REGRESSION = 0.005
_PERSUASION_POLARITY_OVERRIDE_BASE_BUY_MARGIN = 0.12
_PERSUASION_POLARITY_OVERRIDE_EARLY_HORIZON_MARGIN = 0.08


@dataclass(frozen=True)
class TurnEnvelope:
    """One immutable task given to a worker with no credentials or write handles."""

    game: dict[str, Any]
    snapshot: DossierSnapshot
    deadline_at_monotonic: float
    bargaining_advisor_context: dict[str, object] | None = None
    bargaining_advisor_handle: Any | None = None
    negotiation_advisor_context: dict[str, object] | None = None
    negotiation_advisor_handle: Any | None = None
    negotiation_live_policy: dict[str, object] | None = None
    persuasion_advisor_context: dict[str, object] | None = None
    persuasion_advisor_handle: Any | None = None
    bargaining_analytic_authority_context: dict[str, object] | None = None
    bargaining_analytic_authority_prepared: bool = False
    opponent_decision_forecast: dict[str, object] | None = None
    opponent_account_hypothesis: dict[str, object] | None = None
    rating_v3_advisory: dict[str, object] | None = None
    bargaining_intervention_context: dict[str, object] | None = None
    message_style_profile: dict[str, object] | None = None


@dataclass(frozen=True)
class WorkerDecision:
    action: dict[str, Any]
    proposal: dict[str, Any] | None
    tetrad_update: TetradUpdate | None
    tetrad_transport_issues: list[str]
    deterministic_safeguards: list[str]
    fallback: bool
    fallback_reason: str | None
    role: str
    elapsed_s: float
    call_metadata: dict[str, object] | None
    selection_branch: str = "single"
    branch_receipts: list[dict[str, object]] | None = None
    bargaining_advisor_submission: dict[str, object] | None = None
    negotiation_advisor_submission: dict[str, object] | None = None
    persuasion_advisor_submission: dict[str, object] | None = None

    def receipt(self) -> dict[str, object]:
        return {
            "action": self.action,
            "proposal": self.proposal,
            "tetrad_update": self.tetrad_update.model_dump(mode="json") if self.tetrad_update is not None else None,
            "tetrad_transport_issues": self.tetrad_transport_issues,
            "deterministic_safeguards": self.deterministic_safeguards,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "role": self.role,
            "elapsed_s": self.elapsed_s,
            "call_metadata": self.call_metadata,
            "selection_branch": self.selection_branch,
            "branch_receipts": self.branch_receipts,
            "bargaining_advisor_submission": self.bargaining_advisor_submission,
            "negotiation_advisor_submission": self.negotiation_advisor_submission,
            "persuasion_advisor_submission": self.persuasion_advisor_submission,
        }


def _worker_payload(envelope: TurnEnvelope) -> dict[str, object]:
    game = envelope.game
    operational_context = envelope.snapshot.memory_context
    payload: dict[str, object] = {}
    canonical_bargaining = canonical_bargaining_facts(game)
    reference = analytic_bargaining_reference(game)
    analytic_authority = _resolved_bargaining_analytic_authority(envelope)
    if game["game_family"] == "bargaining":
        payload["bargaining_decision_control"] = _bargaining_decision_control(envelope, reference=reference, analytic_authority=analytic_authority)
        if canonical_bargaining is not None:
            payload["canonical_bargaining_facts"] = canonical_bargaining
    payload.update({
        "turn_receipt": {
            "turn_id": envelope.snapshot.turn_id,
            "state_hash": envelope.snapshot.state_hash,
            "snapshot_id": envelope.snapshot.snapshot_id,
        },
        "game_family": game["game_family"],
        "your_player": game["your_player"],
        "phase": game["phase"],
        "opponent": game.get("opponent"),
        "official_prompt": model_official_prompt(game),
        "visible_game_state": model_visible_game_state(game),
        "valid_actions": game["valid_actions"],
    })
    payload["opponent_context"] = {
        "identity_scope": operational_context.get("opponent_identity_scope"),
        "prior_game_count": operational_context.get("opponent_prior_game_count"),
    }
    if isinstance(operational_context.get("global_tactic_memory"), dict):
        payload["global_tactic_memory"] = copy.deepcopy(operational_context["global_tactic_memory"])
    semantics = model_game_semantics(game)
    if semantics is not None:
        payload["game_semantics"] = semantics
    if canonical_bargaining is not None and game["game_family"] != "bargaining":
        payload["canonical_bargaining_facts"] = canonical_bargaining
    if reference is not None:
        analytic: dict[str, object] = {
            "method": reference.method,
            "round": reference.round_number,
            "proposer": reference.proposer,
            "responder": reference.responder,
            "delta_1": reference.delta_1,
            "delta_2": reference.delta_2,
            "max_rounds": reference.max_rounds,
            "equilibrium_offer": reference.equilibrium_offer,
            "responder_continuation_threshold": reference.responder_gain,
            "calculation_precomputed": True,
        }
        if reference.max_rounds is None:
            analytic["stationary_equilibrium_precomputed"] = True
        else:
            analytic["full_backward_induction_precomputed"] = True
            analytic["indifference_tie_break"] = "accept-at-continuation-value"
        if game["valid_actions"]["type"] == "decision" and game.get("your_player") == reference.responder:
            last_offer = game["game_state"].get("last_offer") or {}
            own_gain = float(last_offer.get(f"{reference.responder}_gain", 0))
            analytic["current_offer_own_gain"] = own_gain
            analytic["stationary_equilibrium_reference_decision"] = "accept" if own_gain >= reference.responder_gain - 1e-9 else "reject"
        selected_policy = str(analytic_authority.get("selected_policy") or "") if isinstance(analytic_authority, dict) else ""
        authority_selected = isinstance(analytic_authority, dict) and isinstance(analytic_authority.get("selected_action"), dict)
        exact_reference_selected = authority_selected and selected_policy in {"exact-asymmetric-patient-boundary", "exact-analytic-reference", "exact-after-rejected-nonexact-probe", "recomputed-exact-after-opponent-deviation"}
        analytic["reference_role"] = "authoritative" if exact_reference_selected else "advisory-only"
        if isinstance(analytic_authority, dict) and analytic_authority.get("authority_invalidated") is True:
            analytic["reference_role_reason"] = "current-opponent rejection invalidated this exact acceptance assumption"
        payload["analytic_bargaining_reference"] = analytic
        if canonical_bargaining is not None:
            canonical_bargaining["continuation_reference_status"] = "precomputed-authoritative" if exact_reference_selected else "precomputed-advisory"
    elif canonical_bargaining is not None:
        canonical_bargaining["continuation_reference_status"] = "unavailable"
    if envelope.bargaining_advisor_context is not None:
        payload["bargaining_opponent_model_v2"] = envelope.bargaining_advisor_context
    advisor_context = envelope.bargaining_advisor_context if game["game_family"] == "bargaining" else envelope.negotiation_advisor_context if game["game_family"] == "negotiation" else envelope.persuasion_advisor_context
    decision_forecast = envelope.opponent_decision_forecast or compact_opponent_decision_forecast(envelope.snapshot.memory_context.get("opponent_statistical_package"), game, advisor_context)
    if decision_forecast is not None:
        payload["opponent_statistical_decision_forecast"] = decision_forecast
    if envelope.opponent_account_hypothesis is not None:
        payload["opponent_account_hypothesis"] = envelope.opponent_account_hypothesis
    if envelope.rating_v3_advisory is not None:
        payload["rating_v3_advisory"] = envelope.rating_v3_advisory
    if envelope.negotiation_advisor_context is not None:
        payload["negotiation_opponent_model_v2"] = envelope.negotiation_advisor_context
    if game["game_family"] == "persuasion":
        if envelope.persuasion_advisor_context is not None:
            payload["persuasion_decision_facts"] = envelope.persuasion_advisor_context
        else:
            from .glee_persuasion_live_v2 import persuasion_decision_facts

            payload["persuasion_decision_facts"] = persuasion_decision_facts(game, ())
    if analytic_authority is not None:
        payload["bargaining_analytic_authority"] = analytic_authority
    if envelope.message_style_profile is not None:
        payload["message_realization_profile"] = {
            "profile_id": envelope.message_style_profile.get("profile_id"),
            "prompt_contract": envelope.message_style_profile.get("prompt_contract"),
            "scope": "surface realization only; preserve the selected action, recommendation polarity, and communicative intent; do not mention this profile",
        }
    message_policy_signal = opponent_message_policy_signal(game)
    if message_policy_signal is not None:
        payload["opponent_message_policy_signal"] = message_policy_signal
    return payload


def build_worker_prompt(envelope: TurnEnvelope) -> str:
    """Serialize the official turn and pinned memory identically for every effort branch."""
    preserve_authority_order = envelope.game.get("game_family") == "bargaining"
    return json.dumps(_worker_payload(envelope), ensure_ascii=False, separators=(",", ":"), sort_keys=not preserve_authority_order)


def _metadata(value: object) -> dict[str, object] | None:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)} if value is not None else None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _numeric_offer_for_opponent_share(game: dict[str, Any], opponent_share: float) -> dict[str, Any] | None:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    money = _finite_float(state.get("money_to_divide"))
    if money is None or money <= 0 or not 0 <= opponent_share <= 1:
        return None
    player = str(game.get("your_player") or state.get("current_player") or "")
    if player == "player_1":
        return {"alice_gain": money * (1 - opponent_share), "bob_gain": money * opponent_share}
    if player == "player_2":
        return {"alice_gain": money * opponent_share, "bob_gain": money * (1 - opponent_share)}
    return None


def _offer_action_for_opponent_share(envelope: TurnEnvelope, opponent_share: float) -> dict[str, Any] | None:
    action = _numeric_offer_for_opponent_share(envelope.game, opponent_share)
    return normalize_action(envelope.game, action) if action is not None else None


def _offer_opponent_share(game: dict[str, Any], action: dict[str, Any]) -> float | None:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    money = _finite_float(state.get("money_to_divide"))
    player = str(game.get("your_player") or state.get("current_player") or "")
    if money is None or money <= 0 or player not in {"player_1", "player_2"}:
        return None
    aliases = ("bob_gain", "player_2_gain") if player == "player_1" else ("alice_gain", "player_1_gain")
    amount = next((_finite_float(action.get(key)) for key in aliases if _finite_float(action.get(key)) is not None), None)
    return amount / money if amount is not None else None


def _offer_evaluation(envelope: TurnEnvelope, action: dict[str, Any]) -> dict[str, object] | None:
    handle = envelope.bargaining_advisor_handle
    if handle is None or not hasattr(handle, "submission_prediction"):
        return None
    try:
        prediction = handle.submission_prediction(action)
    except Exception:
        return None
    if not isinstance(prediction, dict):
        return None
    evaluation = prediction.get("behavioral_offer_evaluation")
    return dict(evaluation) if isinstance(evaluation, dict) else None


def _rejection_path_offer_prediction(envelope: TurnEnvelope, action: dict[str, Any]) -> dict[str, object] | None:
    handle = envelope.bargaining_advisor_handle
    if handle is None or not hasattr(handle, "rejection_path_offer_prediction"):
        return None
    try:
        prediction = handle.rejection_path_offer_prediction(action)
    except Exception:
        return None
    return dict(prediction) if isinstance(prediction, dict) else None


def _next_round_exact_offer(envelope: TurnEnvelope) -> dict[str, object] | None:
    game = copy.deepcopy(envelope.game)
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else None
    player = str(game.get("your_player") or (state or {}).get("current_player") or "")
    round_number = (state or {}).get("round")
    if state is None or player not in {"player_1", "player_2"} or isinstance(round_number, bool) or not isinstance(round_number, int):
        return None
    state["round"] = round_number + 1
    state["phase"] = "offer"
    state["proposer"] = player
    state["current_player"] = player
    state.pop("last_offer", None)
    game["phase"] = "offer"
    game["valid_actions"] = {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}}
    reference = analytic_bargaining_reference(game)
    if reference is None or reference.proposer != player:
        return None
    return {
        "round": reference.round_number,
        "method": reference.method,
        "action": dict(reference.equilibrium_offer),
    }


def _bargaining_control_parameter(envelope: TurnEnvelope, name: str, default: float) -> float:
    context = envelope.bargaining_advisor_context
    parameters = context.get("control_parameters") if isinstance(context, dict) else None
    value = _finite_float(parameters.get(name)) if isinstance(parameters, dict) else None
    return value if value is not None and 0 <= value < 1 else default


def _adaptive_bargaining_decision_margin(envelope: TurnEnvelope, *, accept_value: float | None, reject_value: float | None, comparison: Mapping[str, object] | None) -> dict[str, object]:
    """Scale exact-policy override evidence by current value, observed responses, horizon pressure, and recoverability."""
    base = _bargaining_control_parameter(envelope, "decision_override_additional_margin", 0.05)
    value_scale = max(abs(accept_value or 0.0), abs(reject_value or 0.0))
    scale_multiplier = min(1.2, max(0.8, value_scale / 0.5))
    scale_margin = base * scale_multiplier
    state = envelope.game.get("game_state") if isinstance(envelope.game.get("game_state"), dict) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    observed_responses = sum(isinstance(row, Mapping) and str(row.get("decision") or "").casefold() in {"accept", "reject", "acceptoffer", "rejectoffer", "walkaway"} for row in history)
    evidence_multiplier = 1.25 if observed_responses == 0 else 1.0 if observed_responses == 1 else 0.8
    round_number = state.get("round")
    maximum_rounds = state.get("max_rounds")
    remaining_rounds = maximum_rounds - round_number if state.get("horizon_known") is True and isinstance(round_number, int) and not isinstance(round_number, bool) and isinstance(maximum_rounds, int) and not isinstance(maximum_rounds, bool) else None
    horizon_multiplier = 0.6 if remaining_rounds is not None and remaining_rounds <= 1 else 0.8 if remaining_rounds is not None and remaining_rounds <= 3 else 1.0
    reservation = comparison.get("rejected_offer_reservation") if isinstance(comparison, Mapping) and isinstance(comparison.get("rejected_offer_reservation"), Mapping) else {}
    recoverable = reservation.get("status") == "available" and reservation.get("active_constraint") is True
    recoverability_multiplier = 1.2 if recoverable else 1.0
    margin = min(0.15, max(0.01, scale_margin * evidence_multiplier * horizon_multiplier * recoverability_multiplier))
    return {
        "margin": margin,
        "configured_ceiling": base,
        "current_value_scale": value_scale,
        "value_scale_multiplier": scale_multiplier,
        "scale_margin": scale_margin,
        "observed_response_count": observed_responses,
        "evidence_multiplier": evidence_multiplier,
        "remaining_authenticated_rounds": remaining_rounds,
        "horizon_multiplier": horizon_multiplier,
        "recoverable_rejected_value": recoverable,
        "recoverability_multiplier": recoverability_multiplier,
    }


def _normalized_historical_offer(game: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any] | None:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    money = _finite_float(state.get("money_to_divide"))
    if money is None or money <= 0:
        return None
    alice = _finite_float(offer.get("player_1_gain"))
    if alice is None:
        alice = _finite_float(offer.get("alice_gain"))
    bob = _finite_float(offer.get("player_2_gain"))
    if bob is None:
        bob = _finite_float(offer.get("bob_gain"))
    if alice is None and bob is not None:
        alice = money - bob
    if bob is None and alice is not None:
        bob = money - alice
    if alice is None or bob is None or not math.isclose(alice + bob, money, rel_tol=0.0, abs_tol=1e-7):
        return None
    historical_game = copy.deepcopy(game)
    historical_game["phase"] = "offer"
    historical_game["valid_actions"] = {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}}
    historical_game["game_state"]["phase"] = "offer"
    return normalize_action(historical_game, {"alice_gain": alice, "bob_gain": bob})


def _rejected_exact_reference_evidence(envelope: TurnEnvelope) -> list[dict[str, object]]:
    """Find current-game rejections that directly falsify an exact authority proposal."""
    game = envelope.game
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    player = str(game.get("your_player") or state.get("current_player") or "")
    money = _finite_float(state.get("money_to_divide"))
    if not isinstance(history, list) or player not in {"player_1", "player_2"} or money is None or money <= 0:
        return []
    tolerance = _bargaining_control_parameter(envelope, "exact_offer_match_share_tolerance", 0.000001)
    evidence: list[dict[str, object]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or str(entry.get("decision") or "").casefold() != "reject":
            continue
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        proposer = str(entry.get("proposer") or (offer or {}).get("proposer") or "")
        if offer is None or proposer != player:
            continue
        round_number = entry.get("round") or offer.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
            continue
        historical_game = copy.deepcopy(game)
        historical_state = historical_game["game_state"]
        historical_game["phase"] = "offer"
        historical_game["valid_actions"] = {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}}
        historical_game["your_player"] = player
        historical_state.update({"current_player": player, "history": copy.deepcopy(history[:index]), "last_offer": None, "phase": "offer", "proposer": player, "round": round_number})
        reference = analytic_bargaining_reference(historical_game)
        actual = _normalized_historical_offer(historical_game, offer)
        if reference is None or reference.proposer != player or actual is None:
            continue
        if reference.max_rounds is None:
            own_discount = reference.delta_1 if player == "player_1" else reference.delta_2
            opponent_discount = reference.delta_2 if player == "player_1" else reference.delta_1
            patient_self_boundary = math.isclose(own_discount, 1.0, rel_tol=0.0, abs_tol=1e-12) and 0 <= opponent_discount < 1
            if not patient_self_boundary:
                continue
        exact = normalize_action(historical_game, reference.equilibrium_offer)
        maximum_difference = max(abs(float(actual[key]) - float(exact[key])) / money for key in ("alice_gain", "bob_gain"))
        if maximum_difference <= tolerance + 1e-12:
            evidence.append(
                {
                    "round": round_number,
                    "actual_offer": actual,
                    "exact_offer": exact,
                    "maximum_share_difference": maximum_difference,
                    "match_tolerance": tolerance,
                    "opponent_response": "reject",
                }
            )
    return evidence


def _rejected_nonexact_probe_evidence(envelope: TurnEnvelope) -> list[dict[str, object]]:
    """Find rejected departures from the exact capped reference without claiming how they were generated."""
    game = envelope.game
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    player = str(game.get("your_player") or state.get("current_player") or "")
    money = _finite_float(state.get("money_to_divide"))
    if not isinstance(history, list) or player not in {"player_1", "player_2"} or money is None or money <= 0:
        return []
    tolerance = _bargaining_control_parameter(envelope, "exact_offer_match_share_tolerance", 0.000001)
    evidence: list[dict[str, object]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or str(entry.get("decision") or "").casefold() != "reject":
            continue
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        proposer = str(entry.get("proposer") or (offer or {}).get("proposer") or "")
        if offer is None or proposer != player:
            continue
        round_number = entry.get("round") or offer.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
            continue
        historical_game = copy.deepcopy(game)
        historical_state = historical_game["game_state"]
        historical_game["phase"] = "offer"
        historical_game["valid_actions"] = {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}}
        historical_game["your_player"] = player
        historical_state.update({"current_player": player, "history": copy.deepcopy(history[:index]), "last_offer": None, "phase": "offer", "proposer": player, "round": round_number})
        reference = analytic_bargaining_reference(historical_game)
        actual = _normalized_historical_offer(historical_game, offer)
        if reference is None or reference.max_rounds is None or reference.proposer != player or actual is None:
            continue
        exact = normalize_action(historical_game, reference.equilibrium_offer)
        maximum_difference = max(abs(float(actual[key]) - float(exact[key])) / money for key in ("alice_gain", "bob_gain"))
        if maximum_difference > tolerance + 1e-12:
            evidence.append(
                {
                    "round": round_number,
                    "actual_offer": actual,
                    "exact_offer": exact,
                    "maximum_share_difference": maximum_difference,
                    "match_tolerance": tolerance,
                    "opponent_response": "reject",
                    "classification": "rejected-nonexact-probe",
                }
            )
    return evidence


def _patient_boundary_response_policy(envelope: TurnEnvelope, invalidation_evidence: list[dict[str, object]]) -> dict[str, object] | None:
    """Apply the inherited patient settlement boundary after exact-boundary deviation."""
    if not invalidation_evidence or envelope.game.get("valid_actions", {}).get("type") != "decision":
        return None
    state = envelope.game.get("game_state") if isinstance(envelope.game.get("game_state"), dict) else {}
    history = state.get("history")
    current = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
    player = str(envelope.game.get("your_player") or state.get("current_player") or "")
    money = _finite_float(state.get("money_to_divide"))
    if not isinstance(history, list) or current is None or player not in {"player_1", "player_2"} or money is None or money <= 0:
        return None
    opponent = "player_2" if player == "player_1" else "player_1"
    current_proposer = str(current.get("proposer") or state.get("proposer") or "")
    if current_proposer != opponent:
        return None
    current_round = current.get("round") or state.get("round")
    if isinstance(current_round, bool) or not isinstance(current_round, int):
        return None
    own_keys = ("player_1_gain", "alice_gain") if player == "player_1" else ("player_2_gain", "bob_gain")

    def own_share(offer: dict[str, Any]) -> float | None:
        for key in own_keys:
            gain = _finite_float(offer.get(key))
            if gain is not None:
                return gain / money
        return None

    current_share = own_share(current)
    if current_share is None:
        return None
    tolerance = _bargaining_control_parameter(envelope, "exact_offer_match_share_tolerance", 0.000001)
    minimum_share = _bargaining_control_parameter(envelope, "patient_minimum_own_settlement_share", 0.6)
    accept = current_share + tolerance >= minimum_share
    return {
        "contract": "patient-settlement-boundary-v2",
        "policy_regime": "patient",
        "minimum_own_settlement_share": minimum_share,
        "current_offer_our_share": current_share,
        "selected_policy": "patient-settlement-boundary-accept" if accept else "patient-settlement-boundary-reject",
        "selected_action": normalize_action(envelope.game, {"decision": "accept" if accept else "reject"}),
        "reason": "accept at or above the patient settlement boundary" if accept else "preserve the patient option below the settlement boundary",
    }


def _bargaining_analytic_authority(envelope: TurnEnvelope) -> dict[str, object] | None:
    """Select bounded-horizon authority or the exact unbounded patient-self boundary."""
    game = envelope.game
    reference = analytic_bargaining_reference(game)
    if reference is None:
        return None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    if state.get("complete_information") is not True:
        return None
    player = str(game.get("your_player") or state.get("current_player") or "")
    own_discount = reference.delta_1 if player == "player_1" else reference.delta_2 if player == "player_2" else None
    if own_discount is None:
        return None
    opponent_discount = reference.delta_2 if player == "player_1" else reference.delta_1 if player == "player_2" else None
    patient_self_boundary = reference.max_rounds is None and math.isclose(own_discount, 1.0, rel_tol=0.0, abs_tol=1e-12) and opponent_discount is not None and 0 <= opponent_discount < 1
    if reference.max_rounds is None and not patient_self_boundary:
        return None
    if reference.max_rounds is not None and state.get("horizon_known") is not True:
        return None
    action_type = str(game.get("valid_actions", {}).get("type") or "")
    if patient_self_boundary:
        comparison_rule = "the exact patient-self opening remains authoritative until rejected; later decisions use the explicit patient settlement boundary"
    elif action_type == "offer":
        comparison_rule = "a behavioral offer candidate may override the exact offer only when its modeled expected-value advantage strictly exceeds one-round discount cost"
    else:
        comparison_rule = "when exact policy rejects, compare acceptance against the projected controller offer path and require the configured margin; when exact policy accepts, a behavioral rejection must still clear one-round discount cost plus that margin"
    result: dict[str, object] = {
        "contract": "bargaining-analytic-authority-v7",
        "scope": "unbounded-complete-information-patient-self-boundary" if patient_self_boundary else "capped-complete-information",
        "action_type": action_type,
        "exact_method": reference.method,
        "own_discount": own_discount,
        "value_units": "fraction-of-initial-pool",
        "comparison_rule": comparison_rule,
        "override_authorized": False,
    }
    invalidation_evidence = _rejected_exact_reference_evidence(envelope)
    behavioral_invalidation_evidence = _rejected_nonexact_probe_evidence(envelope)
    finite_exact_invalidated = bool(invalidation_evidence) and not patient_self_boundary
    result.update(
        {
            "authority_invalidated": bool(invalidation_evidence),
            "authority_invalidation_reason": "opponent-rejected-unbounded-patient-opening-in-current-game" if invalidation_evidence and patient_self_boundary else "opponent-rejected-matching-finite-exact-offer-in-current-game" if finite_exact_invalidated else None,
            "authority_invalidation_evidence": invalidation_evidence,
            "exact_deviation_observed": bool(invalidation_evidence),
            "exact_deviation_interpretation": "retain the recomputed finite solution as an advisory reference, but revoke its outward authority because this opponent rejected an offer generated by its acceptance assumptions" if finite_exact_invalidated else None,
            "behavioral_authority_invalidated": bool(behavioral_invalidation_evidence),
            "behavioral_authority_invalidation_reason": "opponent-rejected-nonexact-probe-in-current-game" if behavioral_invalidation_evidence else None,
            "behavioral_authority_invalidation_evidence": behavioral_invalidation_evidence,
        }
    )
    if patient_self_boundary:
        if action_type == "offer" and reference.proposer == player:
            exact_action = normalize_action(game, reference.equilibrium_offer)
        elif action_type == "decision" and reference.responder == player:
            last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
            own_gain = _finite_float(last_offer.get("player_1_gain" if player == "player_1" else "player_2_gain"))
            if own_gain is None:
                own_gain = _finite_float(last_offer.get("alice_gain" if player == "player_1" else "bob_gain"))
            if own_gain is None:
                return None
            exact_action = normalize_action(game, {"decision": "accept" if own_gain >= reference.responder_gain - 1e-9 else "reject"})
        else:
            return None
        response_policy = _patient_boundary_response_policy(envelope, invalidation_evidence)
        bounded_selected = response_policy.get("selected_action") if isinstance(response_policy, dict) else None
        result.update(
            {
                "exact_action": exact_action,
                "behavioral_candidate_action": None,
                "quantification_status": "not-applicable-exact-patient-boundary" if not invalidation_evidence else "exact-authority-falsified-current-game",
                "post_invalidation_policy": response_policy,
                "selected_policy": str(response_policy.get("selected_policy")) if isinstance(response_policy, dict) and isinstance(bounded_selected, dict) else "authority-invalidated-current-game-rejection" if invalidation_evidence else "exact-asymmetric-patient-boundary",
                "selected_action": bounded_selected if isinstance(bounded_selected, dict) else None if invalidation_evidence else exact_action,
            }
        )
        return result
    advisor_context = envelope.bargaining_advisor_context
    advisor_available = isinstance(advisor_context, dict) and advisor_context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and advisor_context.get("status") == "available"
    continuation = advisor_context.get("behavioral_continuation") if advisor_available else None
    if action_type == "offer" and reference.proposer == player:
        exact_action = normalize_action(game, reference.equilibrium_offer)
        result["exact_action"] = exact_action
        selected_action = exact_action
        policy = continuation.get("modeled_offer_policy") if isinstance(continuation, dict) else None
        opponent_share = _finite_float(policy.get("opponent_share")) if isinstance(policy, dict) else None
        behavioral_action = _offer_action_for_opponent_share(envelope, opponent_share) if opponent_share is not None else None
        result["behavioral_candidate_action"] = behavioral_action
        exact_evaluation = _offer_evaluation(envelope, exact_action)
        behavioral_evaluation = _offer_evaluation(envelope, behavioral_action) if behavioral_action is not None else None
        exact_value = _finite_float(exact_evaluation.get("expected_value")) if exact_evaluation is not None else None
        behavioral_value = _finite_float(behavioral_evaluation.get("expected_value")) if behavioral_evaluation is not None else None
        exact_accepted_value = _finite_float(exact_evaluation.get("accepted_value")) if exact_evaluation is not None else None
        discount_cost = exact_accepted_value * (1 - own_discount) if exact_accepted_value is not None else None
        advantage = behavioral_value - exact_value if behavioral_value is not None and exact_value is not None else None
        tolerance = max(1e-9, abs(discount_cost or 0.0) * 1e-9)
        override = behavioral_action is not None and advantage is not None and discount_cost is not None and advantage > discount_cost + tolerance
        if override and not behavioral_invalidation_evidence:
            selected_action = behavioral_action
        result.update(
            {
                "exact_expected_value": exact_value,
                "behavioral_expected_value": behavioral_value,
                "behavioral_advantage": advantage,
                "one_round_discount_cost": discount_cost,
                "comparison_tolerance": tolerance,
                "quantification_status": "available" if advantage is not None and discount_cost is not None else "unavailable",
                "override_authorized": override and not bool(behavioral_invalidation_evidence) and not finite_exact_invalidated,
                "selected_policy": "finite-exact-authority-revoked-after-rejection" if finite_exact_invalidated else "exact-after-rejected-nonexact-probe" if override and behavioral_invalidation_evidence else "quantified-behavioral-override" if override else "exact-analytic-reference",
                "selected_action": None if finite_exact_invalidated else selected_action,
            }
        )
        return result
    if action_type == "decision" and reference.responder == player:
        last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
        own_key = "player_1_gain" if player == "player_1" else "player_2_gain"
        alias_key = "alice_gain" if player == "player_1" else "bob_gain"
        own_gain = _finite_float(last_offer.get(own_key))
        if own_gain is None:
            own_gain = _finite_float(last_offer.get(alias_key))
        if own_gain is None:
            return None
        exact_decision = "accept" if own_gain >= reference.responder_gain - 1e-9 else "reject"
        exact_action = normalize_action(game, {"decision": exact_decision})
        comparison = continuation.get("decision_comparison") if isinstance(continuation, dict) else None
        modeled = str(comparison.get("modeled_preference") or "").casefold() if isinstance(comparison, dict) else ""
        behavioral_action = normalize_action(game, {"decision": modeled}) if modeled in {"accept", "reject"} else None
        accept_value = _finite_float(comparison.get("accept_now_value")) if isinstance(comparison, dict) else None
        reject_value = _finite_float(comparison.get("reject_value")) if isinstance(comparison, dict) else None
        discount_cost = accept_value * (1 - own_discount) if accept_value is not None else None
        policy_guard = continuation.get("policy_guard") if isinstance(continuation, dict) else None
        margin_assessment = _adaptive_bargaining_decision_margin(envelope, accept_value=accept_value, reject_value=reject_value, comparison=comparison)
        additional_margin = float(margin_assessment["margin"])
        behavioral_override_authorized = isinstance(policy_guard, dict) and policy_guard.get("mode") == "thresholded-authority-eligible" and policy_guard.get("behavioral_override_eligible") is True and str(policy_guard.get("recommended_decision") or "").casefold() == modeled
        rejection_path: dict[str, object] | None = None
        projected_reject_value = reject_value
        if exact_decision == "reject":
            exact_offer = _next_round_exact_offer(envelope)
            exact_offer_action = exact_offer.get("action") if isinstance(exact_offer, dict) else None
            exact_prediction = _rejection_path_offer_prediction(envelope, exact_offer_action) if isinstance(exact_offer_action, dict) else None
            exact_evaluation = exact_prediction.get("behavioral_offer_evaluation") if isinstance(exact_prediction, dict) and isinstance(exact_prediction.get("behavioral_offer_evaluation"), dict) else None
            exact_offer_value = _finite_float(exact_evaluation.get("expected_value")) if isinstance(exact_evaluation, dict) else None
            exact_offer_accepted_value = _finite_float(exact_evaluation.get("accepted_value")) if isinstance(exact_evaluation, dict) else None
            modeled_offer = continuation.get("modeled_offer_policy_after_rejection") if isinstance(continuation, dict) else None
            modeled_offer_share = _finite_float(modeled_offer.get("opponent_share")) if isinstance(modeled_offer, dict) else None
            modeled_offer_action = _numeric_offer_for_opponent_share(game, modeled_offer_share) if modeled_offer_share is not None else None
            modeled_prediction = _rejection_path_offer_prediction(envelope, modeled_offer_action) if isinstance(modeled_offer_action, dict) else None
            modeled_evaluation = modeled_prediction.get("behavioral_offer_evaluation") if isinstance(modeled_prediction, dict) and isinstance(modeled_prediction.get("behavioral_offer_evaluation"), dict) else None
            modeled_offer_value = _finite_float(modeled_evaluation.get("expected_value")) if isinstance(modeled_evaluation, dict) else reject_value
            modeled_offer_advantage = modeled_offer_value - exact_offer_value if modeled_offer_value is not None and exact_offer_value is not None else None
            next_offer_discount_cost = exact_offer_accepted_value * (1 - own_discount) if exact_offer_accepted_value is not None else None
            next_offer_tolerance = max(1e-9, abs(next_offer_discount_cost or 0.0) * 1e-9)
            next_behavioral_override_candidate = modeled_offer_action is not None and modeled_offer_advantage is not None and next_offer_discount_cost is not None and modeled_offer_advantage > next_offer_discount_cost + next_offer_tolerance
            next_behavioral_override = next_behavioral_override_candidate and not bool(behavioral_invalidation_evidence)
            if exact_offer_value is not None:
                projected_reject_value = modeled_offer_value if next_behavioral_override and modeled_offer_value is not None else exact_offer_value
            rejection_path = {
                "contract": "bargaining-controller-rejection-path-v1",
                "exact_next_offer": exact_offer,
                "exact_next_offer_prediction": exact_prediction,
                "exact_next_offer_expected_value": exact_offer_value,
                "behavioral_next_offer": modeled_offer_action,
                "behavioral_next_offer_prediction": modeled_prediction,
                "behavioral_next_offer_expected_value": modeled_offer_value,
                "behavioral_next_offer_advantage": modeled_offer_advantage,
                "next_offer_one_round_discount_cost": next_offer_discount_cost,
                "next_offer_behavioral_override_candidate": next_behavioral_override_candidate,
                "next_offer_behavioral_override_authorized": next_behavioral_override,
                "projected_next_offer_policy": "behavioral-override" if next_behavioral_override else "finite-exact-reference" if exact_offer_value is not None else "behavioral-continuation-fallback",
                "projected_reject_value": projected_reject_value,
            }
        exact_value = accept_value if exact_decision == "accept" else projected_reject_value
        behavioral_value = accept_value if modeled == "accept" else reject_value if modeled == "reject" else None
        advantage = behavioral_value - exact_value if behavioral_value is not None and exact_value is not None else None
        action_path_aligned_acceptance = exact_decision == "reject" and modeled == "accept" and rejection_path is not None and rejection_path.get("exact_next_offer_expected_value") is not None
        required_advantage = additional_margin if action_path_aligned_acceptance else (discount_cost + additional_margin) if discount_cost is not None else None
        tolerance = max(1e-9, abs(required_advantage or 0.0) * 1e-9)
        override_candidate = behavioral_action is not None and modeled != exact_decision and advantage is not None and required_advantage is not None and advantage > required_advantage + tolerance and behavioral_override_authorized
        override = override_candidate and not behavioral_invalidation_evidence
        selected_action = behavioral_action if override else exact_action
        result.update(
            {
                "exact_action": exact_action,
                "behavioral_candidate_action": behavioral_action,
                "exact_expected_value": exact_value,
                "exact_expected_value_source": "projected-controller-rejection-path" if action_path_aligned_acceptance else "current-acceptance" if exact_decision == "accept" else "behavioral-continuation-fallback",
                "behavioral_expected_value": behavioral_value,
                "behavioral_advantage": advantage,
                "one_round_discount_cost": discount_cost,
                "decision_override_additional_margin": additional_margin,
                "decision_override_margin_assessment": margin_assessment,
                "required_behavioral_advantage": required_advantage,
                "action_path_aligned_acceptance_comparison": action_path_aligned_acceptance,
                "controller_rejection_path": rejection_path,
                "comparison_tolerance": tolerance,
                "quantification_status": "available" if advantage is not None and discount_cost is not None else "unavailable",
                "override_authorized": override and not finite_exact_invalidated,
                "behavioral_override_authorized_by_advisor": behavioral_override_authorized,
                "selected_policy": "finite-exact-authority-revoked-after-rejection" if finite_exact_invalidated else "exact-after-rejected-nonexact-probe" if behavioral_invalidation_evidence and override_candidate else "quantified-behavioral-decision-override" if override else "exact-analytic-reference",
                "selected_action": None if finite_exact_invalidated else selected_action,
            }
        )
        return result
    return None


def _resolved_bargaining_analytic_authority(envelope: TurnEnvelope) -> dict[str, object] | None:
    if envelope.bargaining_analytic_authority_prepared:
        return envelope.bargaining_analytic_authority_context
    return _bargaining_analytic_authority(envelope)


def _bargaining_decision_control(envelope: TurnEnvelope, *, reference: Any | None, analytic_authority: dict[str, object] | None) -> dict[str, object]:
    """Put the current Bargaining action hierarchy in the first model-facing payload field."""
    game = envelope.game
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    action_type = str(game.get("valid_actions", {}).get("type") or "")
    player = str(game.get("your_player") or state.get("current_player") or "")
    reference_action: dict[str, Any] | None = None
    if reference is not None and action_type == "offer" and reference.proposer == player:
        reference_action = normalize_action(game, reference.equilibrium_offer)
    elif reference is not None and action_type == "decision" and reference.responder == player:
        last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
        own_key = "player_1_gain" if player == "player_1" else "player_2_gain"
        alias_key = "alice_gain" if player == "player_1" else "bob_gain"
        own_gain = _finite_float(last_offer.get(own_key))
        if own_gain is None:
            own_gain = _finite_float(last_offer.get(alias_key))
        if own_gain is not None:
            reference_action = normalize_action(game, {"decision": "accept" if own_gain >= reference.responder_gain - 1e-9 else "reject"})
    continuation = envelope.bargaining_advisor_context.get("behavioral_continuation") if isinstance(envelope.bargaining_advisor_context, dict) else None
    behavioral_action: dict[str, Any] | None = None
    if action_type == "decision":
        comparison = continuation.get("decision_comparison") if isinstance(continuation, dict) else None
        modeled = str(comparison.get("modeled_preference") or "").casefold() if isinstance(comparison, dict) else ""
        if modeled in {"accept", "reject"}:
            behavioral_action = normalize_action(game, {"decision": modeled})
    elif action_type == "offer":
        policy = continuation.get("modeled_offer_policy") if isinstance(continuation, dict) else None
        if not isinstance(policy, dict) and isinstance(continuation, dict):
            policy = continuation.get("modeled_offer_policy_after_rejection")
        opponent_share = _finite_float(policy.get("opponent_share")) if isinstance(policy, dict) else None
        if opponent_share is not None:
            behavioral_action = _offer_action_for_opponent_share(envelope, opponent_share)
    conflict = reference_action is not None and behavioral_action is not None and not _same_policy_action(game, reference_action, behavioral_action)
    selected = analytic_authority.get("selected_action") if isinstance(analytic_authority, dict) else None
    selected_policy = str(analytic_authority.get("selected_policy") or "") if isinstance(analytic_authority, dict) else ""
    if isinstance(selected, dict):
        scope = str(analytic_authority.get("scope") or "")
        if selected_policy in {"patient-settlement-boundary-accept", "patient-settlement-boundary-reject"}:
            reason = "deterministic-patient-settlement-boundary"
        elif selected_policy == "recomputed-exact-after-opponent-deviation":
            reason = "recomputed-current-finite-subgame-after-opponent-deviation"
        elif scope == "unbounded-complete-information-patient-self-boundary":
            reason = "exact-patient-self-boundary"
        elif action_type == "decision" and analytic_authority.get("override_authorized") is True:
            reason = "quantified-behavioral-decision-advantage-clears-discount-cost-plus-margin"
        elif action_type == "decision":
            reason = "exact-finite-horizon-decision; behavioral-advantage-did-not-clear-threshold"
        elif analytic_authority.get("override_authorized") is True:
            reason = "quantified-behavioral-offer-advantage-clears-discount-cost"
        else:
            reason = "exact-finite-horizon-offer-reference"
        authority_status = "authoritative"
        selected_action: dict[str, object] | None = copy.deepcopy(selected)
        instruction = "Submit selected_action numerically; any optional message must remain consistent with it."
    elif isinstance(analytic_authority, dict) and analytic_authority.get("authority_invalidated") is True:
        authority_status = "advisory-only"
        selected_action = None
        reason = "exact-authority-invalidated-by-current-opponent-rejection"
        instruction = "The exact reference remains a calculation but has no action authority after this opponent rejected its matching offer; compare behavioral and authenticated current-game evidence."
    elif reference is not None:
        own_discount = reference.delta_1 if player == "player_1" else reference.delta_2 if player == "player_2" else None
        opponent_discount = reference.delta_2 if player == "player_1" else reference.delta_1 if player == "player_2" else None
        if reference.max_rounds is None and own_discount is not None and 0 < own_discount < 1 and opponent_discount is not None and math.isclose(opponent_discount, 1.0, rel_tol=0.0, abs_tol=1e-12):
            reason = "opponent-patient-unbounded-reference-is-advisory"
        elif reference.max_rounds is None:
            reason = "unbounded-reference-outside-exact-authority-scope"
        else:
            reason = "reference-outside-authenticated-finite-horizon-authority-scope"
        authority_status = "advisory-only"
        selected_action = None
        instruction = "No action is selected by the analytic reference; compare all authenticated and model-derived evidence under the Bargaining policy."
    else:
        authority_status = "unavailable"
        selected_action = None
        reason = "no-analytic-reference"
        instruction = "No analytic action is selected; reason from authenticated state, behavioral evidence, and policy safeguards."
    control: dict[str, object] = {
        "contract": "bargaining-decision-control-v4",
        "authority_status": authority_status,
        "selected_action": selected_action,
        "authority_reason": reason,
        "instruction": instruction,
        "action_type": action_type,
        "analytic_reference_role": "authoritative" if authority_status == "authoritative" and selected_policy in {"exact-asymmetric-patient-boundary", "exact-analytic-reference", "exact-after-rejected-nonexact-probe", "recomputed-exact-after-opponent-deviation"} else "advisory-only" if reference is not None else "unavailable",
        "stationary_equilibrium_reference_action": reference_action,
        "behavioral_recommendation": behavioral_action,
        "reference_behavior_conflict": conflict,
        "authority_invalidated": bool(isinstance(analytic_authority, dict) and analytic_authority.get("authority_invalidated") is True),
        "authority_invalidation_evidence": copy.deepcopy(analytic_authority.get("authority_invalidation_evidence")) if isinstance(analytic_authority, dict) else None,
        "behavioral_authority_invalidated": bool(isinstance(analytic_authority, dict) and analytic_authority.get("behavioral_authority_invalidated") is True),
        "behavioral_authority_invalidation_evidence": copy.deepcopy(analytic_authority.get("behavioral_authority_invalidation_evidence")) if isinstance(analytic_authority, dict) else None,
    }
    if action_type == "decision" and isinstance(reference_action, dict):
        control["stationary_equilibrium_reference_decision"] = reference_action.get("decision")
    return control


def _prepare_bargaining_analytic_authority(envelope: TurnEnvelope) -> TurnEnvelope:
    if envelope.bargaining_analytic_authority_prepared:
        return envelope
    authority = _bargaining_analytic_authority(envelope)
    return replace(
        envelope,
        bargaining_analytic_authority_context=authority,
        bargaining_analytic_authority_prepared=True,
    )


def prepare_worker_envelope(envelope: TurnEnvelope) -> TurnEnvelope:
    """Freeze every deterministic pre-Terra context that is shared by shadow capture and model inference."""
    return _prepare_bargaining_analytic_authority(envelope)


def worker_payload(envelope: TurnEnvelope) -> dict[str, object]:
    """Expose the exact structured pre-Terra payload for causal feature capture and prompt serialization."""
    return _worker_payload(envelope)


def _same_policy_action(game: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> bool:
    action_type = str(game.get("valid_actions", {}).get("type") or "")
    if action_type == "offer":
        pairs = [(_finite_float(left.get(key)), _finite_float(right.get(key))) for key in ("alice_gain", "bob_gain")]
        return all(left_value is not None and right_value is not None and math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-7) for left_value, right_value in pairs)
    if action_type == "decision":
        return str(left.get("decision") or "").casefold() == str(right.get("decision") or "").casefold()
    return left == right


def _apply_bargaining_analytic_authority(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    authority = _resolved_bargaining_analytic_authority(envelope)
    if authority is None or not isinstance(authority.get("selected_action"), dict):
        return None
    selected = dict(authority["selected_action"])
    if _same_policy_action(envelope.game, candidate, selected):
        return candidate, []
    if authority.get("selected_policy") in {"patient-settlement-boundary-accept", "patient-settlement-boundary-reject"}:
        name = "bargaining_v210_patient_settlement_boundary"
    elif authority.get("selected_policy") == "recomputed-exact-after-opponent-deviation":
        name = "bargaining_v210_recomputed_exact_subgame"
    elif authority.get("selected_policy") == "exact-asymmetric-patient-boundary":
        name = "bargaining_v25_exact_patient_boundary"
    elif authority.get("selected_policy") == "quantified-behavioral-decision-override":
        name = "bargaining_v213_action_path_decision_override"
    else:
        name = "bargaining_v23_behavioral_override" if authority.get("override_authorized") is True else "bargaining_v23_exact_analytic_authority"
    return normalize_action(envelope.game, selected), [name]


def _zero_inference_model_capacity(error: Exception) -> dict[str, object] | None:
    """Recognize only an explicit Codex capacity failure before any inference item."""
    receipt = getattr(error, "provider_receipt", None)
    if not isinstance(receipt, dict) or not isinstance(receipt.get("event_stream"), str):
        return None
    events: list[dict[str, Any]] = []
    for line in receipt["event_stream"].splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        events.append(event)
    allowed = {"thread.started", "turn.started", "error", "turn.failed"}
    event_types = [str(event.get("type") or "") for event in events]
    if not events or any(event_type not in allowed for event_type in event_types):
        return None
    terminal = [event for event in events if event.get("type") in {"error", "turn.failed"}]
    if {str(event.get("type")) for event in terminal} != {"error", "turn.failed"}:
        return None
    diagnostic = " ".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in terminal).casefold()
    if _CODEX_MODEL_CAPACITY_MESSAGE not in diagnostic:
        return None
    return {
        "classification": "explicit-zero-inference-model-capacity",
        "event_types": event_types,
        "diagnostic": "Selected model is at capacity. Please try a different model.",
    }


def _mentions_model_capacity(error: Exception) -> bool:
    receipt = getattr(error, "provider_receipt", None)
    material = f"{type(error).__name__}: {error}"
    if isinstance(receipt, dict):
        material = f"{material}\n{json.dumps(receipt, ensure_ascii=False, sort_keys=True)}"
    return _CODEX_MODEL_CAPACITY_MESSAGE in material.casefold()


def _is_timeout_failure(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    receipt = getattr(error, "provider_receipt", None)
    material = f"{type(error).__name__}: {error}"
    if isinstance(receipt, dict):
        material = f"{material}\n{receipt.get('stderr') or ''}"
    lowered = material.casefold()
    return "timed out" in lowered or "timeout expired" in lowered


def _bargaining_coordinates(envelope: TurnEnvelope) -> tuple[dict[str, Any], float, str, str] | None:
    state = envelope.game.get("game_state") if isinstance(envelope.game.get("game_state"), dict) else {}
    money = _finite_float(state.get("money_to_divide"))
    if money is None or money <= 0:
        return None
    player = str(envelope.game.get("your_player") or state.get("current_player") or "")
    if player in {"player_1", "alice"}:
        own_key, opponent_key = "alice_gain", "bob_gain"
    elif player in {"player_2", "bob"}:
        own_key, opponent_key = "bob_gain", "alice_gain"
    else:
        return None
    return state, money, own_key, opponent_key


def _apply_bargaining_material_payoff_guard(envelope: TurnEnvelope, candidate: dict[str, Any], *, safeguard: str) -> tuple[dict[str, Any], list[str]]:
    """Reject a nonterminal acceptance below the scale-free material-payoff floor."""
    if envelope.game.get("game_family") != "bargaining":
        return candidate, []
    coordinates = _bargaining_coordinates(envelope)
    if coordinates is None:
        return candidate, []
    state, money, own_key, _ = coordinates
    action = dict(candidate)
    action_type = str(envelope.game.get("valid_actions", {}).get("type") or "")
    if action_type != "decision" or str(action.get("decision") or "").casefold() != "accept":
        return action, []
    last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
    own_gain = _finite_float(last_offer.get(own_key))
    if own_gain is None:
        own_gain = _finite_float(last_offer.get("player_1_gain" if own_key == "alice_gain" else "player_2_gain"))
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    guard = continuation.get("policy_guard") if isinstance(continuation, dict) else None
    force_accept_reason = str(guard.get("force_accept_reason") or "") if isinstance(guard, dict) else ""
    terminal_positive_accept = force_accept_reason in {"positive payoff inside the observed round-99 terminal window", "positive known-final-round payoff"}
    minimum_accept_share = 0.0 if terminal_positive_accept else _BARGAINING_MINIMUM_MATERIAL_OWN_SHARE
    if own_gain is not None and own_gain / money <= minimum_accept_share - 1e-12:
        return normalize_action(envelope.game, {"decision": "reject"}), [safeguard]
    return action, []


def _apply_bargaining_v25_guard(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply the cumulative scale-free catastrophic-loss and bounded-concession controls."""
    if envelope.game.get("game_family") != "bargaining":
        return candidate, []
    coordinates = _bargaining_coordinates(envelope)
    if coordinates is None:
        return candidate, []
    state, money, own_key, opponent_key = coordinates
    action = dict(candidate)
    action_type = str(envelope.game.get("valid_actions", {}).get("type") or "")
    if action_type == "decision":
        action, material_applied = _apply_bargaining_material_payoff_guard(envelope, action, safeguard="bargaining_v25_reject_immaterial_own_payoff")
        if material_applied:
            return action, material_applied
        last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
        own_gain = _finite_float(last_offer.get(own_key))
        if own_gain is None:
            own_gain = _finite_float(last_offer.get("player_1_gain" if own_key == "alice_gain" else "player_2_gain"))
        context = envelope.bargaining_advisor_context
        continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
        guard = continuation.get("policy_guard") if isinstance(continuation, dict) else None
        decision = str(action.get("decision") or "").casefold()
        force_accept_reason = str(guard.get("force_accept_reason") or "") if isinstance(guard, dict) else ""
        environmental_terminal_accept = force_accept_reason == "positive payoff inside the observed round-99 terminal window"
        known_final_accept = force_accept_reason == "positive known-final-round payoff"
        terminal_positive_accept = environmental_terminal_accept or known_final_accept
        patient_settlement_accept = force_accept_reason == "patient settlement boundary met"
        patient_deadlock_accept = force_accept_reason == "persistent patient deadlock settlement"
        patient_reciprocal_window_accept = force_accept_reason == "temporary reciprocal-concession window"
        minimum_accept_share = 0.0 if terminal_positive_accept else _BARGAINING_MINIMUM_MATERIAL_OWN_SHARE
        own_share = own_gain / money if own_gain is not None else None
        force_accept_eligible = own_share is not None and (own_share > 1e-12 if terminal_positive_accept else own_share + 1e-12 >= minimum_accept_share)
        if decision == "accept" and isinstance(guard, dict) and guard.get("force_reject") is True:
            return normalize_action(envelope.game, {"decision": "reject"}), ["bargaining_v210_patient_settlement_reject"]
        if decision in {"reject", "walkaway"} and force_accept_eligible and isinstance(guard, dict) and guard.get("force_accept") is True:
            safeguard = "bargaining_v210_environmental_terminal_accept" if environmental_terminal_accept else "bargaining_v212_known_final_accept" if known_final_accept else "bargaining_v212_patient_deadlock_accept" if patient_deadlock_accept else "bargaining_v215_reciprocal_concession_window_accept" if patient_reciprocal_window_accept else "bargaining_v210_patient_settlement_accept" if patient_settlement_accept else "bargaining_v212_current_utility_settlement_accept"
            return normalize_action(envelope.game, {"decision": "accept"}), [safeguard]
        return action, []
    if action_type != "offer":
        return action, []
    current = _finite_float(action.get(opponent_key))
    if current is None:
        return action, []
    original_share = current / money
    target_share = min(original_share, 1 - _BARGAINING_MINIMUM_MATERIAL_OWN_SHARE)
    applied: list[str] = []
    if target_share < original_share - 1e-12:
        applied.append("bargaining_v25_material_own_share_floor")
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    guard = continuation.get("policy_guard") if isinstance(continuation, dict) else None
    if isinstance(guard, dict) and guard.get("mode") == "bounded-authoritative":
        share_tolerance = _bargaining_control_parameter(envelope, "exact_offer_match_share_tolerance", 0.000001)
        minimum = _finite_float(guard.get("minimum_opponent_share"))
        maximum = _finite_float(guard.get("maximum_opponent_share"))
        loss_minimization = guard.get("loss_minimization_control") if isinstance(guard.get("loss_minimization_control"), dict) else {}
        loss_target = _finite_float(loss_minimization.get("target_opponent_share")) if loss_minimization.get("status") == "active" else None
        if loss_target is not None and 0 <= loss_target <= 1:
            if not math.isclose(target_share, loss_target, rel_tol=0.0, abs_tol=1e-12):
                target_share = loss_target
                applied.append("bargaining_v212_loss_minimization_offer")
        else:
            if minimum is not None and 0 <= minimum <= 1 and target_share < minimum - share_tolerance:
                target_share = minimum
                applied.append("bargaining_v25_opponent_share_floor")
            reciprocal = continuation.get("reciprocal_concession_control") if isinstance(continuation.get("reciprocal_concession_control"), dict) else {}
            reciprocal_cap = _finite_float(reciprocal.get("maximum_next_opponent_share")) if reciprocal.get("status") == "available" else None
            if reciprocal_cap is not None and target_share > reciprocal_cap + 1e-12:
                target_share = reciprocal_cap
                applied.append("bargaining_v25_reciprocal_concession_bound")
            recovery = continuation.get("rejected_value_recovery_control") if isinstance(continuation.get("rejected_value_recovery_control"), dict) else {}
            recovery_cap = _finite_float(recovery.get("maximum_current_offer_opponent_share")) if recovery.get("status") == "available" else None
            if recovery_cap is not None and target_share > recovery_cap + 1e-12:
                target_share = recovery_cap
                applied.append("bargaining_v25_rejected_value_recovery")
            if maximum is not None and 0 <= maximum <= 1 and target_share > maximum + 1e-12:
                target_share = maximum
                if "bargaining_v25_material_own_share_floor" not in applied and "bargaining_v25_reciprocal_concession_bound" not in applied and "bargaining_v25_rejected_value_recovery" not in applied:
                    applied.append("bargaining_v25_bounded_offer")
    if not applied:
        return action, []
    if math.isclose(target_share, original_share, rel_tol=0.0, abs_tol=1e-12):
        return action, []
    target = min(money, max(0.0, target_share * money))
    action[opponent_key] = target
    action[own_key] = money - target
    return normalize_action(envelope.game, action), applied


def _apply_bargaining_v27_offer_value_guard(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Reject a large advisor-estimated value regression from the frozen behavioral offer."""
    if envelope.game.get("game_family") != "bargaining" or envelope.game.get("valid_actions", {}).get("type") != "offer":
        return candidate, []
    context = envelope.bargaining_advisor_context
    if not isinstance(context, dict) or context.get("model_version") != BARGAINING_ADVISOR_MODEL_VERSION or context.get("status") != "available":
        return candidate, []
    continuation = context.get("behavioral_continuation")
    control = continuation.get("offer_expected_value_regression_control") if isinstance(continuation, dict) else None
    if not isinstance(control, dict) or control.get("status") != "available":
        return candidate, []
    reference_share = _finite_float(control.get("reference_opponent_share"))
    margin = _finite_float(control.get("maximum_expected_value_regression"))
    reference_action = _offer_action_for_opponent_share(envelope, reference_share) if reference_share is not None else None
    if reference_action is None or margin is None or not 0 <= margin < 1:
        return candidate, []
    candidate_evaluation = _offer_evaluation(envelope, candidate)
    reference_evaluation = _offer_evaluation(envelope, reference_action)
    candidate_value = _finite_float(candidate_evaluation.get("expected_value")) if candidate_evaluation is not None else None
    reference_value = _finite_float(reference_evaluation.get("expected_value")) if reference_evaluation is not None else None
    if candidate_value is None or reference_value is None:
        return candidate, []
    tolerance = max(1e-9, abs(reference_value) * 1e-9)
    if reference_value - candidate_value <= margin + tolerance:
        return candidate, []
    guarded, bounded = _apply_bargaining_v25_guard(envelope, reference_action)
    if isinstance(candidate.get("message"), str):
        guarded = {**guarded, "message": candidate["message"]}
    return normalize_action(envelope.game, guarded), ["bargaining_v27_offer_expected_value_regression", *bounded]


def _apply_bargaining_v22_guard(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Compatibility alias for tests and receipts written before v2.5."""
    return _apply_bargaining_v25_guard(envelope, candidate)


def _apply_negotiation_v28_guard(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if envelope.game.get("game_family") != "negotiation":
        return candidate, []
    context = envelope.negotiation_advisor_context
    handle = envelope.negotiation_advisor_handle
    if not isinstance(context, dict) or context.get("model_version") != NEGOTIATION_ADVISOR_MODEL_VERSION or context.get("status") != "available" or handle is None or not hasattr(handle, "guard_action"):
        return candidate, []
    action, applied = handle.guard_action(candidate)
    return normalize_action(envelope.game, action), list(applied)


def _persuasion_buyer_economic_control(envelope: TurnEnvelope) -> Mapping[str, object] | None:
    context = envelope.persuasion_advisor_context
    if not isinstance(context, Mapping):
        return None
    control = context.get("buyer_economic_control")
    return control if isinstance(control, Mapping) and control.get("contract") == "glee-persuasion-buyer-economic-control-v1" else None


def _apply_persuasion_v27_guard(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if envelope.game.get("game_family") != "persuasion":
        return candidate, []
    guarded = normalize_action(envelope.game, candidate)
    applied: list[str] = []
    action_type = str((envelope.game.get("valid_actions") or {}).get("type") or envelope.game.get("phase") or "")
    economic = _persuasion_buyer_economic_control(envelope)
    if action_type == "buyer_decision" and isinstance(economic, Mapping):
        persuasion_context = envelope.persuasion_advisor_context if isinstance(envelope.persuasion_advisor_context, Mapping) else {}
        persuasion_authority = persuasion_context.get("authority") if isinstance(persuasion_context.get("authority"), Mapping) else {}
        authoritative_selected = persuasion_authority.get("selected_action")
        robust_selected = economic.get("robust_uncertainty_action")
        if not isinstance(authoritative_selected, Mapping) and isinstance(robust_selected, Mapping) and economic.get("common_unit_continuation_lower_bound_over_price") is None:
            selected = normalize_action(envelope.game, dict(robust_selected))
            if guarded != selected:
                guarded = selected
                applied.append("persuasion_buyer_robust_uncertainty_boundary")
        terminal_selected = economic.get("terminal_selected_action") if economic.get("terminal") is True else None
        if isinstance(terminal_selected, Mapping):
            selected = normalize_action(envelope.game, dict(terminal_selected))
            if guarded != selected:
                guarded = selected
                applied.append("persuasion_buyer_terminal_expected_value")
        if str(guarded.get("decision") or "").casefold() == "yes" and economic.get("buy_economically_admissible") is False:
            guarded = normalize_action(envelope.game, {"decision": "no"})
            applied.append("persuasion_buyer_material_negative_purchase")
    context = envelope.persuasion_advisor_context
    handle = envelope.persuasion_advisor_handle
    if not isinstance(context, dict) or context.get("model_version") != PERSUASION_ADVISOR_MODEL_VERSION or context.get("status") != "available" or handle is None or not hasattr(handle, "guard_action"):
        return guarded, applied
    action, advisor_applied = handle.guard_action(guarded)
    return normalize_action(envelope.game, action), [*applied, *list(advisor_applied)]


def _bargaining_settlement_policy_preempts_analytic(envelope: TurnEnvelope, continuation: dict[str, Any], guard: dict[str, Any], action_type: str) -> bool:
    """Limit settlement precedence to the inherited environmental policy or absent, falsified, or terminal finite authority."""
    if continuation.get("round_cap_source") == "observed-platform-termination":
        return True
    authority = _resolved_bargaining_analytic_authority(envelope)
    if not isinstance(authority, dict) or not isinstance(authority.get("selected_action"), dict):
        return True
    if authority.get("authority_invalidated") is True:
        return True
    if action_type == "decision":
        return str(guard.get("force_accept_reason") or "") in {"positive payoff inside the observed round-99 terminal window", "positive known-final-round payoff"}
    if action_type == "offer":
        loss = guard.get("loss_minimization_control") if isinstance(guard.get("loss_minimization_control"), dict) else {}
        return loss.get("status") == "active" and loss.get("terminal_window_trigger") is True
    return False


def _apply_worker_safeguards(envelope: TurnEnvelope, candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    negotiation_policy = envelope.negotiation_live_policy if isinstance(envelope.negotiation_live_policy, dict) else {}
    negotiation_features = negotiation_policy.get("features") if isinstance(negotiation_policy.get("features"), dict) else {}
    preserve_positive_negotiation_exit = negotiation_features.get(NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE) == "strategic-model"
    action, applied = apply_deterministic_safeguards(envelope.game, candidate, preserve_positive_negotiation_exit=preserve_positive_negotiation_exit)
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    if isinstance(continuation, dict):
        guard = continuation.get("policy_guard") if isinstance(continuation.get("policy_guard"), dict) else {}
        loss = guard.get("loss_minimization_control") if isinstance(guard.get("loss_minimization_control"), dict) else {}
        action_type = str(envelope.game.get("valid_actions", {}).get("type") or "")
        frozen_policy_authority = (action_type == "decision" and (guard.get("force_accept") is True or guard.get("force_reject") is True)) or (action_type == "offer" and loss.get("status") == "active")
        if frozen_policy_authority and _bargaining_settlement_policy_preempts_analytic(envelope, continuation, guard, action_type):
            action, policy_applied = _apply_bargaining_v25_guard(envelope, action)
            action, intervention_applied = apply_bargaining_v217_intervention(envelope, action)
            return action, [*applied, *policy_applied, *intervention_applied]
    analytic = _apply_bargaining_analytic_authority(envelope, action)
    if analytic is not None:
        action, authority_applied = analytic
        action, material_applied = _apply_bargaining_material_payoff_guard(envelope, action, safeguard="bargaining_v215_post_analytic_material_payoff_floor")
        action, intervention_applied = apply_bargaining_v217_intervention(envelope, action)
        return action, [*applied, *authority_applied, *material_applied, *intervention_applied]
    action, advisor_applied = _apply_bargaining_v25_guard(envelope, action)
    action, value_applied = _apply_bargaining_v27_offer_value_guard(envelope, action)
    action, intervention_applied = apply_bargaining_v217_intervention(envelope, action)
    action, negotiation_applied = _apply_negotiation_v28_guard(envelope, action)
    action, persuasion_applied = _apply_persuasion_v27_guard(envelope, action)
    return action, [*applied, *advisor_applied, *value_applied, *intervention_applied, *negotiation_applied, *persuasion_applied]


def _candidate_from_parsed(envelope: TurnEnvelope, parsed: Any, *, role: str, elapsed_s: float, call_metadata: object) -> WorkerDecision:
    proposal = parsed.action.model_dump(exclude_none=True)
    normalized = normalize_action(envelope.game, parsed.action)
    action, safeguards = _apply_worker_safeguards(envelope, normalized)
    return WorkerDecision(
        action=action,
        proposal=proposal,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role=role,
        elapsed_s=round(elapsed_s, 6),
        call_metadata=_metadata(call_metadata),
    )


def _bargaining_action_has_message_channel(game: dict[str, Any]) -> bool:
    if game.get("valid_actions", {}).get("type") != "offer":
        return False
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    if state.get("messages_allowed") is False:
        return False
    fields = game.get("valid_actions", {}).get("fields")
    return not isinstance(fields, dict) or not fields or "message" in fields


def _authoritative_message_free_bargaining_decision(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Return a corrected authoritative action directly when inference cannot add a message."""
    if envelope.game.get("game_family") != "bargaining" or _bargaining_action_has_message_channel(envelope.game):
        return None
    authority = _resolved_bargaining_analytic_authority(envelope)
    selected = authority.get("selected_action") if isinstance(authority, dict) else None
    if not isinstance(selected, dict):
        return None
    started = time.monotonic()
    action, safeguards = _apply_worker_safeguards(envelope, normalize_action(envelope.game, selected))
    receipt = {
        "branch": "bargaining-authority",
        "status": "succeeded",
        "cloud_inference": "bypassed",
        "reason": "corrected-authoritative-action-without-message-channel",
        "authority": copy.deepcopy(authority),
        "action": action,
        "deterministic_safeguards": safeguards,
    }
    return WorkerDecision(
        action=action,
        proposal=copy.deepcopy(selected),
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_bargaining_analytic_authority",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="bargaining-authority",
        branch_receipts=[receipt],
    )


def _authoritative_bargaining_settlement_policy_action(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Bypass inference when the frozen settlement policy selects a complete numeric action."""
    game = envelope.game
    if game.get("game_family") != "bargaining":
        return None
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    guard = continuation.get("policy_guard") if isinstance(continuation, dict) else None
    if not isinstance(guard, dict):
        return None
    action_type = str(game.get("valid_actions", {}).get("type") or "")
    selected: dict[str, object] | None = None
    reason: str | None = None
    if action_type == "decision":
        if guard.get("force_accept") is True:
            selected = {"decision": "accept"}
            reason = str(guard.get("force_accept_reason") or "frozen policy selected acceptance")
        elif guard.get("force_reject") is True:
            selected = {"decision": "reject"}
            reason = str(guard.get("force_reject_reason") or "frozen policy selected rejection")
    elif action_type == "offer" and not _bargaining_action_has_message_channel(game):
        loss_minimization = guard.get("loss_minimization_control") if isinstance(guard.get("loss_minimization_control"), dict) else {}
        target = _finite_float(loss_minimization.get("target_opponent_share")) if loss_minimization.get("status") == "active" else None
        if target is not None:
            selected = _offer_action_for_opponent_share(envelope, target)
            reason = f"cap-aware {loss_minimization.get('stage') or 'loss-minimization'} allocation"
    if not isinstance(selected, dict):
        return None
    if not _bargaining_settlement_policy_preempts_analytic(envelope, continuation, guard, action_type):
        return None
    started = time.monotonic()
    proposal = normalize_action(game, selected)
    action, safeguards = apply_deterministic_safeguards(game, proposal)
    action, bounded = _apply_bargaining_v25_guard(envelope, action)
    action, intervention = apply_bargaining_v217_intervention(envelope, action)
    safeguards = [*safeguards, *bounded, *intervention]
    receipt = {
        "branch": "bargaining-v2.17-policy",
        "status": "succeeded",
        "cloud_inference": "bypassed",
        "reason": reason,
        "action": action,
        "deterministic_safeguards": safeguards,
    }
    return WorkerDecision(
        action=action,
        proposal=proposal,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_bargaining_v217_policy",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="bargaining-v2.17-policy",
        branch_receipts=[receipt],
    )


def _deterministic_bargaining_plateau_offer(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Bypass cloud prose generation when both numeric positions and the advisor recommendation are unchanged."""
    game = envelope.game
    if game.get("game_family") != "bargaining" or game.get("valid_actions", {}).get("type") != "offer" or not _bargaining_action_has_message_channel(game):
        return None
    plateau = bargaining_plateau_summary(game)
    if plateau is None:
        return None
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    policy = continuation.get("modeled_offer_policy") if isinstance(continuation, dict) else None
    opponent_share = _finite_float(policy.get("opponent_share")) if isinstance(policy, dict) else None
    recommended = _offer_action_for_opponent_share(envelope, opponent_share) if opponent_share is not None else None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    player = str(game.get("your_player") or state.get("current_player") or "")
    if recommended is None or not isinstance(history, list) or player not in {"player_1", "player_2"}:
        return None
    latest_offer = next(
        (
            entry["offer"]
            for entry in reversed(history)
            if isinstance(entry, dict)
            and isinstance(entry.get("offer"), dict)
            and str(entry.get("proposer") or entry["offer"].get("proposer") or "") == player
        ),
        None,
    )
    if not isinstance(latest_offer, dict):
        return None
    previous = _normalized_historical_offer(game, latest_offer)
    if previous is None or not _same_policy_action(game, recommended, previous):
        return None
    started = time.monotonic()
    proposal = {**recommended, "message": "My numeric proposal is unchanged."}
    action, safeguards = _apply_worker_safeguards(envelope, normalize_action(game, proposal))
    if not _same_policy_action(game, action, recommended):
        return None
    receipt = {
        "branch": "bargaining-plateau",
        "status": "succeeded",
        "cloud_inference": "bypassed",
        "reason": "bilateral numeric plateau and unchanged frozen-advisor recommendation make another prose paraphrase non-informative",
        "plateau": plateau,
        "action": action,
        "deterministic_safeguards": safeguards,
    }
    return WorkerDecision(
        action=action,
        proposal=proposal,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_bargaining_plateau",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="bargaining-plateau",
        branch_receipts=[receipt],
    )


def _deterministic_bargaining_semantic_staircase_offer(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Bypass redundant prose inference while a low-information staircase leaves the frozen numeric policy locally stable."""
    game = envelope.game
    if game.get("game_family") != "bargaining" or game.get("valid_actions", {}).get("type") != "offer" or not _bargaining_action_has_message_channel(game):
        return None
    staircase = bargaining_semantic_staircase_summary(game)
    if staircase is None:
        return None
    context = envelope.bargaining_advisor_context
    continuation = context.get("behavioral_continuation") if isinstance(context, dict) and context.get("model_version") == BARGAINING_ADVISOR_MODEL_VERSION and context.get("status") == "available" else None
    policy = continuation.get("modeled_offer_policy") if isinstance(continuation, dict) else None
    opponent_share = _finite_float(policy.get("opponent_share")) if isinstance(policy, dict) else None
    recommended = _offer_action_for_opponent_share(envelope, opponent_share) if opponent_share is not None else None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    player = str(game.get("your_player") or state.get("current_player") or "")
    if recommended is None or not isinstance(history, list) or player not in {"player_1", "player_2"}:
        return None
    latest_offer = next(
        (
            entry["offer"]
            for entry in reversed(history)
            if isinstance(entry, dict)
            and isinstance(entry.get("offer"), dict)
            and str(entry.get("proposer") or entry["offer"].get("proposer") or "") == player
        ),
        None,
    )
    if not isinstance(latest_offer, dict):
        return None
    previous = _normalized_historical_offer(game, latest_offer)
    previous_share = _offer_opponent_share(game, previous) if previous is not None else None
    recommended_share = _offer_opponent_share(game, recommended)
    if previous_share is None or recommended_share is None:
        return None
    recommendation_shift = abs(recommended_share - previous_share)
    if recommendation_shift > _BARGAINING_SEMANTIC_STAIRCASE_MAXIMUM_RECOMMENDATION_SHIFT + 1e-12:
        return None
    started = time.monotonic()
    proposal = {**recommended, "message": "The numeric terms are updated; my strategic position otherwise remains unchanged."}
    action, safeguards = _apply_worker_safeguards(envelope, normalize_action(game, proposal))
    if not _same_policy_action(game, action, recommended):
        return None
    receipt = {
        "branch": "bargaining-semantic-staircase",
        "status": "succeeded",
        "cloud_inference": "bypassed",
        "reason": "a stable low-information opponent staircase left the frozen-advisor recommendation within the local policy-stability boundary",
        "staircase": staircase,
        "previous_opponent_share": previous_share,
        "recommended_opponent_share": recommended_share,
        "absolute_recommendation_shift": recommendation_shift,
        "maximum_recommendation_shift": _BARGAINING_SEMANTIC_STAIRCASE_MAXIMUM_RECOMMENDATION_SHIFT,
        "action": action,
        "deterministic_safeguards": safeguards,
    }
    return WorkerDecision(
        action=action,
        proposal=proposal,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_bargaining_semantic_staircase",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="bargaining-semantic-staircase",
        branch_receipts=[receipt],
    )


def _preinference_bargaining_decision(envelope: TurnEnvelope) -> WorkerDecision | None:
    return _authoritative_bargaining_settlement_policy_action(envelope) or _authoritative_message_free_bargaining_decision(envelope) or _deterministic_bargaining_plateau_offer(envelope) or _deterministic_bargaining_semantic_staircase_offer(envelope)


def _preinference_persuasion_decision(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Bypass cloud inference only when a frozen Persuasion control grants hard action authority."""
    if envelope.game.get("game_family") != "persuasion":
        return None
    context = envelope.persuasion_advisor_context
    authority = context.get("authority") if isinstance(context, dict) else None
    if not isinstance(authority, dict):
        return None
    action_type = str((envelope.game.get("valid_actions") or {}).get("type") or envelope.game.get("phase") or "")
    selected = authority.get("selected_action") if isinstance(authority, dict) else None
    economic = _persuasion_buyer_economic_control(envelope)
    if action_type == "buyer_decision" and isinstance(economic, Mapping) and economic.get("terminal") is True and isinstance(economic.get("terminal_selected_action"), Mapping):
        selected = dict(economic["terminal_selected_action"])
        branch = "persuasion-buyer-terminal-expected-value"
    elif action_type in {"seller_message", "seller_recommendation"} and authority.get("action_authority") == _PERSUASION_ADVISORY_ANCHOR_AUTHORITY:
        state = envelope.game.get("game_state") if isinstance(envelope.game.get("game_state"), dict) else {}
        round_number = state.get("round")
        total_rounds = state.get("total_rounds")
        terminal_binary_anchor = action_type == "seller_recommendation" and isinstance(round_number, int) and not isinstance(round_number, bool) and isinstance(total_rounds, int) and not isinstance(total_rounds, bool) and round_number >= total_rounds
        if not terminal_binary_anchor:
            return None
        selected = _persuasion_advisory_anchor_action(envelope)
        if not isinstance(selected, dict):
            return None
        branch = "persuasion-v2.7-terminal-binary-anchor"
    elif action_type in {"seller_message", "seller_recommendation"} and authority.get("action_authority") == _PERSUASION_BOUNDED_AUTHORITY and authority.get("policy_scope") == "bounded-no-response-authority":
        selected = _persuasion_seller_policy_action(envelope, allowed_authorities={_PERSUASION_BOUNDED_AUTHORITY})
        if not isinstance(selected, dict):
            return None
        branch = "persuasion-v2.8-no-response-authority"
    else:
        if not isinstance(selected, dict):
            return None
        branch = "persuasion-v2.7-authority"
    started = time.monotonic()
    proposal = normalize_action(envelope.game, selected)
    action, safeguards = _apply_worker_safeguards(envelope, proposal)
    return WorkerDecision(
        action=action,
        proposal=proposal,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_persuasion_buyer_terminal_expected_value" if branch == "persuasion-buyer-terminal-expected-value" else "glee_persuasion_v27_terminal_anchor" if branch.endswith("terminal-binary-anchor") else "glee_persuasion_v28_no_response_authority" if branch.endswith("no-response-authority") else "glee_persuasion_v28_authority",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch=branch,
        branch_receipts=[{"branch": branch, "status": "succeeded", "cloud_inference": "bypassed", "reason": "terminal posterior expected-value comparison with exact-tie pass" if branch == "persuasion-buyer-terminal-expected-value" else authority.get("reason"), "economic_control": copy.deepcopy(dict(economic)) if branch == "persuasion-buyer-terminal-expected-value" and isinstance(economic, Mapping) else None, "action": action, "deterministic_safeguards": safeguards}],
    )


def _preinference_decision(envelope: TurnEnvelope) -> WorkerDecision | None:
    if envelope.game.get("game_family") == "bargaining":
        return _preinference_bargaining_decision(envelope)
    if envelope.game.get("game_family") == "persuasion":
        return _preinference_persuasion_decision(envelope)
    return None


def _persuasion_seller_policy_action(envelope: TurnEnvelope, *, allowed_authorities: set[str]) -> dict[str, Any] | None:
    """Resolve one selected seller polarity under an explicitly supplied authority set."""
    if envelope.game.get("game_family") != "persuasion":
        return None
    context = envelope.persuasion_advisor_context
    if not isinstance(context, dict) or context.get("model_version") != PERSUASION_ADVISOR_MODEL_VERSION or context.get("status") != "available":
        return None
    authority = context.get("authority")
    if not isinstance(authority, dict) or authority.get("action_authority") not in allowed_authorities:
        return None
    action_type = str((envelope.game.get("valid_actions") or {}).get("type") or envelope.game.get("phase") or "")
    try:
        if action_type == "seller_recommendation" and isinstance(authority.get("selected_action"), dict):
            return normalize_action(envelope.game, authority["selected_action"])
        polarity = str(authority.get("selected_signal_polarity") or "")
        if action_type == "seller_message" and polarity in {"positive", "negative"}:
            from .glee_persuasion_policy_v2_8 import canonical_seller_message

            return normalize_action(envelope.game, {"message": canonical_seller_message(polarity)})
    except (TypeError, ValueError):
        return None
    return None


def _persuasion_advisory_anchor_action(envelope: TurnEnvelope) -> dict[str, Any] | None:
    """Resolve the v2.8 seller recommendation as a legal fallback without granting it action authority."""
    return _persuasion_seller_policy_action(envelope, allowed_authorities={_PERSUASION_ADVISORY_ANCHOR_AUTHORITY})


def _negotiation_one_round_advisory_anchor_action(envelope: TurnEnvelope) -> dict[str, Any] | None:
    """Resolve the hidden-value one-round calibrated price as a reference candidate and safe fallback."""
    if envelope.game.get("game_family") != "negotiation":
        return None
    context = envelope.negotiation_advisor_context
    if not isinstance(context, dict) or context.get("model_version") != NEGOTIATION_ADVISOR_MODEL_VERSION or context.get("status") != "available":
        return None
    facts = context.get("deterministic_decision_facts")
    control = facts.get("one_round_seller_control") if isinstance(facts, Mapping) else None
    if not isinstance(control, Mapping) or control.get("status") != "advisory" or control.get("mode") != "incomplete-information-calibrated-markup":
        return None
    target = _finite_float(control.get("target_price"))
    if target is None:
        return None
    try:
        return normalize_action(envelope.game, {"product_price": target})
    except (TypeError, ValueError):
        return None


def _meta15_candidate_seed_specs(envelope: TurnEnvelope, baseline: WorkerDecision) -> tuple[list[dict[str, object]], bool]:
    """Return planner-visible seeds and whether they must be present in the frozen candidate set."""
    action_type = str((envelope.game.get("valid_actions") or {}).get("type") or envelope.game.get("phase") or "")
    if envelope.game.get("game_family") == "negotiation" and action_type == "decision":
        raw_specs: list[dict[str, object]] = [
            {"action": {"decision": "AcceptOffer"}, "purpose": "accept the current binding offer at its exact terminal own surplus"},
            {"action": {"decision": "WalkAway"}, "purpose": "end negotiation now at exact zero terminal surplus"},
            {"action": dict(baseline.action), "purpose": "guarded deterministic fallback"},
        ]
        handle = envelope.negotiation_advisor_handle
        shadow = handle.forecast_receipt.get("shadow_utility_rollout") if handle is not None and isinstance(getattr(handle, "forecast_receipt", None), Mapping) else None
        best = shadow.get("best_candidate") if isinstance(shadow, Mapping) and isinstance(shadow.get("best_candidate"), Mapping) else None
        price = _finite_float(best.get("price")) if isinstance(best, Mapping) else None
        if negotiation_reject_counteroffer_frontier(envelope.game) and price is not None:
            raw_specs.append({"action": {"decision": "RejectOffer", "product_price": price}, "purpose": "reject and submit the established advisor's best bounded-continuation counteroffer"})
        specs: list[dict[str, object]] = []
        seen: set[str] = set()
        for raw in raw_specs:
            try:
                action = normalize_action(envelope.game, dict(raw["action"]))
            except (TypeError, ValueError):
                continue
            digest = object_sha256(action)
            if digest in seen:
                continue
            seen.add(digest)
            specs.append({"action": action, "purpose": raw["purpose"]})
        return specs[:5], True
    negotiation_anchor = _negotiation_one_round_advisory_anchor_action(envelope)
    if negotiation_anchor is not None:
        return ([{"action": negotiation_anchor, "purpose": "calibrated hidden-value one-round seller reference and deterministic fallback"}], True)
    anchor = _persuasion_advisory_anchor_action(envelope)
    if anchor is None:
        return ([{"action": dict(baseline.action), "purpose": "guarded deterministic fallback and selector-gate reference"}], True)
    specs: list[dict[str, object]] = [{"action": anchor, "purpose": "v2.7 longitudinal-policy advisory anchor"}]
    if action_type == "seller_recommendation":
        anchor_decision = str(anchor.get("decision") or "")
        alternative = {"decision": "no" if anchor_decision == "yes" else "yes"}
        specs.append({"action": normalize_action(envelope.game, alternative), "purpose": "complete binary alternative to the v2.7 advisory anchor"})
    elif action_type == "seller_message":
        from .glee_persuasion_twin_v2 import classify_persuasion_signal

        anchor_polarity = classify_persuasion_signal(anchor.get("message"), channel="text")[0]
        if anchor_polarity in {"positive", "negative"}:
            alternative_polarity = "negative" if anchor_polarity == "positive" else "positive"
            from .glee_persuasion_policy_v2_7 import canonical_seller_message

            specs.append({"action": normalize_action(envelope.game, {"message": canonical_seller_message(alternative_polarity)}), "purpose": "opposite-polarity canonical alternative to the v2.7 advisory anchor"})
    return specs, True


def _meta15_family_candidate_evidence(envelope: TurnEnvelope, candidate_set: Any) -> dict[str, object] | None:
    """Project established family-advisor evidence onto every frozen candidate."""
    family = str(envelope.game.get("game_family") or "")
    handle = envelope.bargaining_advisor_handle if family == "bargaining" else envelope.negotiation_advisor_handle if family == "negotiation" else None
    if handle is None:
        return None
    evidence: list[dict[str, object]] = []
    for candidate in candidate_set.candidates:
        action = dict(candidate.action)
        if family == "bargaining" and hasattr(handle, "submission_prediction"):
            prediction = handle.submission_prediction(action)
            evaluation = prediction.get("behavioral_offer_evaluation") if isinstance(prediction, Mapping) else None
            if not isinstance(evaluation, Mapping):
                raise ValueError("Bargaining family advisor returned no candidate offer evaluation")
            evidence.append(
                {
                    "source": "bargaining-live-advisor-v2.17",
                    "behavioral_offer_evaluation": dict(evaluation),
                    "direct_acceptance_probability": prediction.get("opponent_acceptance_probability_v2"),
                    "response_expert_probabilities": copy.deepcopy(prediction.get("response_expert_probabilities")),
                    "response_expert_weights": copy.deepcopy(prediction.get("response_expert_weights")),
                    "authority": "established-bounded-behavioral-evidence",
                }
            )
        elif family == "negotiation" and hasattr(handle, "candidate_selection_evidence"):
            value = handle.candidate_selection_evidence(action)
            if not isinstance(value, Mapping):
                raise ValueError("Negotiation family advisor returned malformed candidate evidence")
            evidence.append(dict(value))
        else:
            return None
    if family == "negotiation":
        role = str((envelope.negotiation_advisor_context or {}).get("deterministic_decision_facts", {}).get("our_role") or "")
        grouped: dict[float, list[int]] = {}
        for index, value in enumerate(evidence):
            evaluation = value.get("evaluation") if isinstance(value.get("evaluation"), Mapping) else {}
            response = evaluation.get("opponent_response") if isinstance(evaluation.get("opponent_response"), Mapping) else {}
            price = _finite_float(evaluation.get("price"))
            probability = _finite_float(response.get("weighted"))
            if price is not None and probability is not None:
                grouped.setdefault(price, []).append(index)
        prices = sorted(grouped)
        raw_by_price = [sum(float(evidence[index]["evaluation"]["opponent_response"]["weighted"]) for index in grouped[price]) / len(grouped[price]) for price in prices]
        if role == "seller":
            projected_by_price = list(reversed(_isotonic_nondecreasing(list(reversed(raw_by_price)), [float(len(grouped[price])) for price in reversed(prices)])))
        else:
            projected_by_price = _isotonic_nondecreasing(raw_by_price, [float(len(grouped[price])) for price in prices])
        for price, projected in zip(prices, projected_by_price, strict=True):
            for index in grouped[price]:
                evaluation = evidence[index]["evaluation"]
                response = evaluation["opponent_response"]
                raw_probability = float(response["weighted"])
                response["weighted_raw"] = round(raw_probability, 6)
                response["weighted_price_monotone"] = round(projected, 6)
                response["bounded_message_residual"] = round(max(-0.03, min(0.03, raw_probability - projected)), 6)
                response["weighted"] = round(projected, 6)
                response["projection_rule"] = "nonincreasing in price for a seller proposal; nondecreasing in price for a buyer proposal; message residual is retained separately as a bounded diagnostic"
                own_surplus = float(evaluation.get("own_surplus_if_accepted") or 0.0)
                continuation = evaluation.get("conditional_rejection_continuation") if isinstance(evaluation.get("conditional_rejection_continuation"), Mapping) else {}
                continuation_value = float(continuation.get("bounded_continuation_value") or 0.0)
                evaluation["bounded_expected_value_raw"] = evaluation.get("bounded_expected_value")
                evaluation["bounded_expected_value"] = round(projected * own_surplus + (1.0 - projected) * continuation_value, 6)
                evidence[index]["authority"] = "candidate-specific-role-monotone-bounded-continuation-evidence"
    return build_family_candidate_evidence(candidate_set=candidate_set, evidence=evidence)


def _meta15_candidate_response_surface(*, envelope: TurnEnvelope, candidate_set: FrozenCandidateSet, conditional_client: Any, worker_turn: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Forecast response-bearing candidates while representing terminal Negotiation actions without a fictitious opponent reply."""
    if candidate_set.family != "negotiation" or candidate_set.action_type != "decision":
        receipt = conditional_client.forecast_candidates(game=envelope.game, turn_id=envelope.snapshot.turn_id, synthetic_features=terra_synthetic_feature_bundle(worker_turn), candidate_set=candidate_set)
        rows = receipt.get("rows")
        forecasts = [dict(row["forecast"]) for row in rows if isinstance(row, Mapping) and isinstance(row.get("forecast"), Mapping)] if isinstance(rows, list) else []
        return receipt, build_conditional_surface(candidate_set=candidate_set, forecasts=forecasts)
    counters = tuple(candidate for candidate in candidate_set.candidates if negotiation_reject_counteroffer_action(candidate.action))
    counter_forecasts: dict[int, dict[str, object]] = {}
    counter_receipt: dict[str, object] | None = None
    if len(counters) == 1:
        candidate = counters[0]
        handle = envelope.negotiation_advisor_handle
        if handle is None or not hasattr(handle, "candidate_selection_evidence"):
            raise ValueError("single Negotiation counteroffer requires established family-advisor evidence")
        evidence = handle.candidate_selection_evidence(dict(candidate.action))
        evaluation = evidence.get("evaluation") if isinstance(evidence, Mapping) and isinstance(evidence.get("evaluation"), Mapping) else None
        response = evaluation.get("opponent_response") if isinstance(evaluation, Mapping) and isinstance(evaluation.get("opponent_response"), Mapping) else None
        accept_probability = _finite_float(response.get("weighted")) if isinstance(response, Mapping) else None
        if accept_probability is None or not 0.0 <= accept_probability <= 1.0:
            raise ValueError("single Negotiation counteroffer has no bounded family-advisor acceptance probability")
        forecast = {
            "authority": "prospective-shadow-only",
            "labels": ["accept", "reject", "walkaway"],
            "response_probabilities": [accept_probability, 1.0 - accept_probability, 0.0],
            "source": "established-negotiation-family-advisor",
            "interpretation": "A single counteroffer has no comparative conditional-twin surface. Reuse its frozen family-advisor acceptance projection; the residual is represented as rejection and no separate walkaway mass is invented.",
        }
        counter_forecasts[candidate.index] = forecast
        counter_receipt = {
            "status": "locally-projected-single-counter",
            "candidate_index": candidate.index,
            "action_sha256": candidate.action_sha256,
            "accept_probability": accept_probability,
            "source": "negotiation-advisor-candidate-selection-evidence",
            "conditional_service_called": False,
        }
    elif counters:
        projected_counters = tuple(replace(candidate, index=index) for index, candidate in enumerate(counters))
        identity = {"family": candidate_set.family, "action_type": candidate_set.action_type, "candidates": [candidate.receipt() for candidate in projected_counters]}
        counter_set = FrozenCandidateSet(family=candidate_set.family, action_type=candidate_set.action_type, candidates=projected_counters, candidate_set_sha256=object_sha256(identity))
        counter_receipt = conditional_client.forecast_candidates(game=envelope.game, turn_id=envelope.snapshot.turn_id, synthetic_features=terra_synthetic_feature_bundle(worker_turn), candidate_set=counter_set)
        rows = counter_receipt.get("rows")
        if not isinstance(rows, list) or len(rows) != len(counters):
            raise ValueError("Negotiation counteroffer forecast does not cover its candidate subset")
        for candidate, projected, row in zip(counters, projected_counters, rows, strict=True):
            forecast = row.get("forecast") if isinstance(row, Mapping) else None
            if not isinstance(row, Mapping) or row.get("candidate_index") != projected.index or row.get("action_sha256") != projected.action_sha256 or not isinstance(forecast, Mapping):
                raise ValueError("Negotiation counteroffer forecast is malformed")
            counter_forecasts[candidate.index] = dict(forecast)
    forecasts: list[dict[str, object]] = []
    for candidate in candidate_set.candidates:
        if candidate.index in counter_forecasts:
            forecasts.append(counter_forecasts[candidate.index])
            continue
        decision = str(candidate.action.get("decision") or "")
        forecasts.append({"authority": "exact-terminal-value-no-opponent-response", "target_status": "terminal-self-action", "decision": decision, "labels": [], "response_probabilities": [], "interpretation": "AcceptOffer, WalkAway, or a terminal bare rejection ends this response frontier; compare its exact terminal value through family evidence rather than inventing an opponent response."})
    surface = build_conditional_surface(candidate_set=candidate_set, forecasts=forecasts)
    receipt = {"contract": "glee-negotiation-mixed-categorical-candidate-surface-v1", "candidate_set_sha256": candidate_set.candidate_set_sha256, "counteroffer_candidate_count": len(counters), "terminal_candidate_count": len(candidate_set.candidates) - len(counters), "counteroffer_forecast_receipt": counter_receipt, "rows": copy.deepcopy(surface["rows"]), "authority": "counteroffer-response-shadow-plus-exact-terminal-value-boundary"}
    return receipt, surface


def _meta15_public_self_mirror_admissibility(*, candidate_set: Any, conditional_surface: Mapping[str, object] | None, family_evidence: Mapping[str, object] | None, buyer_path: bool = False) -> dict[str, object]:
    """Bound public-expectedness authority to an economically near-equivalent candidate set."""
    values: list[float | None] = []
    family = str(candidate_set.family)
    source = "unavailable"
    cap = 0.0
    units = "none"
    if buyer_path:
        values = [None for _candidate in candidate_set.candidates]
        source = "Persuasion buyer path has no common-unit candidate utility estimate"
    elif family == "bargaining":
        source = "bargaining-live-advisor behavioral expected value"
        cap = 0.01
        units = "normalized expected own payoff"
        for candidate in candidate_set.candidates:
            evidence = _meta15_aligned_row(family_evidence, candidate.index, payload_key="evidence")
            evaluation = evidence.get("behavioral_offer_evaluation") if isinstance(evidence, Mapping) and isinstance(evidence.get("behavioral_offer_evaluation"), Mapping) else None
            values.append(_finite_float(evaluation.get("expected_value")) if isinstance(evaluation, Mapping) else None)
    elif family == "negotiation":
        source = "negotiation-live-advisor bounded complete-game expected value"
        units = "native expected surplus"
        for candidate in candidate_set.candidates:
            evidence = _meta15_aligned_row(family_evidence, candidate.index, payload_key="evidence")
            evaluation = evidence.get("evaluation") if isinstance(evidence, Mapping) and isinstance(evidence.get("evaluation"), Mapping) else None
            values.append(_finite_float(evaluation.get("bounded_expected_value")) if isinstance(evaluation, Mapping) else None)
        finite = [value for value in values if value is not None]
        cap = 0.02 * max(1.0, max(abs(value) for value in finite)) if finite else 0.0
    elif family == "persuasion":
        source = "candidate-conditioned current purchase probability"
        cap = 0.03
        units = "current buy probability"
        for candidate in candidate_set.candidates:
            forecast = _meta15_aligned_row(conditional_surface, candidate.index, payload_key="forecast")
            values.append(_meta15_probability(forecast, "buy"))
    else:
        values = [None for _candidate in candidate_set.candidates]
    return build_public_self_mirror_admissibility(candidate_set=candidate_set, utility_values=values, absolute_regret_cap=cap, value_units=units, evidence_source=source)


def _meta15_aligned_row(surface: Mapping[str, object] | None, candidate_index: int, *, payload_key: str) -> dict[str, object] | None:
    rows = surface.get("rows") if isinstance(surface, Mapping) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("candidate_index") == candidate_index and isinstance(row.get(payload_key), Mapping):
            return dict(row[payload_key])
    return None


def _meta15_probability(forecast: Mapping[str, object] | None, label: str, *, probability_key: str = "response_probabilities") -> float | None:
    labels = forecast.get("labels") if isinstance(forecast, Mapping) else None
    probabilities = forecast.get(probability_key) if isinstance(forecast, Mapping) else None
    if not isinstance(labels, list) or not isinstance(probabilities, list) or label not in labels or len(labels) != len(probabilities):
        return None
    value = _finite_float(probabilities[labels.index(label)])
    return value if value is not None and 0 <= value <= 1 else None


def _meta15_bargaining_pair(action: Mapping[str, object]) -> tuple[float, float] | None:
    alice = _finite_float(action.get("alice_gain"))
    bob = _finite_float(action.get("bob_gain"))
    return (alice, bob) if alice is not None and bob is not None else None


def _meta15_candidates_with_bargaining_pair(candidate_set: Any, pair: tuple[float, float]) -> list[Any]:
    matches: list[Any] = []
    for candidate in candidate_set.candidates:
        candidate_pair = _meta15_bargaining_pair(candidate.action)
        if candidate_pair is not None and all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(candidate_pair, pair, strict=True)):
            matches.append(candidate)
    return matches


def _meta15_bargaining_component_value(evaluation: Mapping[str, object] | None, forecast: Mapping[str, object] | None, probability_key: str) -> float | None:
    accepted = _finite_float(evaluation.get("accepted_value")) if isinstance(evaluation, Mapping) else None
    rejected = _finite_float(evaluation.get("rejected_path_value")) if isinstance(evaluation, Mapping) else None
    accept_probability = _meta15_probability(forecast, "accept", probability_key=probability_key)
    reject_probability = _meta15_probability(forecast, "reject", probability_key=probability_key)
    if accepted is None or rejected is None or accept_probability is None or reject_probability is None:
        return None
    return accept_probability * accepted + reject_probability * rejected


def _meta15_bargaining_selector_gate(*, envelope: TurnEnvelope, baseline: WorkerDecision, selected: Any, candidate_set: Any, conditional_surface: Mapping[str, object], family_evidence: Mapping[str, object] | None) -> tuple[dict[str, Any], list[str], dict[str, object]]:
    selected_action = dict(selected.candidate.action)
    selected_pair = _meta15_bargaining_pair(selected_action)
    baseline_pair = _meta15_bargaining_pair(baseline.action)
    receipt: dict[str, object] = {
        "contract": FAMILY_SELECTOR_POLICY_CONTRACT,
        "family": "bargaining",
        "policy": "numeric-override-support-and-confidence-gate-v2",
        "selected_candidate_index": selected.candidate.index,
        "minimum_blended_expected_value_gain": _BARGAINING_NUMERIC_OVERRIDE_MINIMUM_TWIN_GAIN,
        "minimum_sequence_expected_value_gain": _BARGAINING_NUMERIC_OVERRIDE_MINIMUM_SEQUENCE_GAIN,
        "maximum_component_or_advisor_regression": _BARGAINING_NUMERIC_OVERRIDE_MAXIMUM_COMPONENT_REGRESSION,
    }
    if selected_pair is None or baseline_pair is None:
        receipt.update({"status": "not-applicable", "reason": "numeric offer pair unavailable"})
        return selected_action, [], receipt
    if all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(selected_pair, baseline_pair, strict=True)):
        receipt.update({"status": "passed", "reason": "same numeric allocation; wording remains selector-controlled"})
        return selected_action, [], receipt
    baseline_candidates = _meta15_candidates_with_bargaining_pair(candidate_set, baseline_pair)
    selected_forecast = _meta15_aligned_row(conditional_surface, selected.candidate.index, payload_key="forecast")
    selected_family = _meta15_aligned_row(family_evidence, selected.candidate.index, payload_key="evidence")
    selected_evaluation = selected_family.get("behavioral_offer_evaluation") if isinstance(selected_family, Mapping) and isinstance(selected_family.get("behavioral_offer_evaluation"), Mapping) else None
    baseline_controls: list[tuple[Any, dict[str, object] | None, Mapping[str, object] | None]] = []
    for candidate in baseline_candidates:
        forecast = _meta15_aligned_row(conditional_surface, candidate.index, payload_key="forecast")
        family = _meta15_aligned_row(family_evidence, candidate.index, payload_key="evidence")
        evaluation = family.get("behavioral_offer_evaluation") if isinstance(family, Mapping) and isinstance(family.get("behavioral_offer_evaluation"), Mapping) else None
        baseline_controls.append((candidate, forecast, evaluation))
    receipt["baseline_numeric_comparator_candidate_indexes"] = [candidate.index for candidate in baseline_candidates]
    gains: dict[str, float | None] = {}
    for label, key in (("blended", "response_probabilities"), ("sequence", "sequence_probabilities"), ("engineered", "engineered_probabilities")):
        selected_value = _meta15_bargaining_component_value(selected_evaluation, selected_forecast, key)
        baseline_values = [_meta15_bargaining_component_value(evaluation, forecast, key) for _candidate, forecast, evaluation in baseline_controls]
        baseline_value = max(value for value in baseline_values if value is not None) if baseline_values and all(value is not None for value in baseline_values) else None
        gains[label] = selected_value - baseline_value if selected_value is not None and baseline_value is not None else None
    selected_advisor = _finite_float(selected_evaluation.get("expected_value")) if isinstance(selected_evaluation, Mapping) else None
    baseline_advisors = [_finite_float(evaluation.get("expected_value")) if isinstance(evaluation, Mapping) else None for _candidate, _forecast, evaluation in baseline_controls]
    baseline_advisor = max(value for value in baseline_advisors if value is not None) if baseline_advisors and all(value is not None for value in baseline_advisors) else None
    gains["family_advisor"] = selected_advisor - baseline_advisor if selected_advisor is not None and baseline_advisor is not None else None
    receipt["expected_value_gains_over_strongest_baseline_numeric_control"] = {key: round(value, 6) if value is not None else None for key, value in gains.items()}
    authorized = gains["blended"] is not None and gains["sequence"] is not None and gains["engineered"] is not None and gains["family_advisor"] is not None and gains["blended"] >= _BARGAINING_NUMERIC_OVERRIDE_MINIMUM_TWIN_GAIN and gains["sequence"] >= _BARGAINING_NUMERIC_OVERRIDE_MINIMUM_SEQUENCE_GAIN and gains["engineered"] >= -_BARGAINING_NUMERIC_OVERRIDE_MAXIMUM_COMPONENT_REGRESSION and gains["family_advisor"] >= -_BARGAINING_NUMERIC_OVERRIDE_MAXIMUM_COMPONENT_REGRESSION
    if authorized:
        receipt.update({"status": "passed", "reason": "numeric override clears blended, sequence, engineered, and established-advisor support margins against every available baseline-numeric wording control"})
        return selected_action, [], receipt
    guarded = dict(baseline.action)
    if isinstance(selected_action.get("message"), str):
        guarded["message"] = selected_action["message"]
    guarded, safeguards = _apply_worker_safeguards(envelope, normalize_action(envelope.game, guarded))
    applied = ["bargaining_meta15_numeric_override_support_gate", *safeguards]
    receipt.update({"status": "numeric-reverted", "reason": "numeric override lacks calibrated multi-model support; selector wording is preserved", "submitted_action": guarded})
    return guarded, applied, receipt


def _meta15_persuasion_polarity(action: Mapping[str, object], action_type: str) -> str:
    if action_type == "seller_recommendation":
        return "positive" if str(action.get("decision") or "").casefold() == "yes" else "negative" if str(action.get("decision") or "").casefold() == "no" else "unknown"
    from .glee_persuasion_twin_v2 import classify_persuasion_signal

    return classify_persuasion_signal(action.get("message"), channel="text")[0]


def _meta15_persuasion_selector_gate(*, envelope: TurnEnvelope, selected: Any, candidate_set: Any, conditional_surface: Mapping[str, object]) -> tuple[dict[str, Any], list[str], dict[str, object]]:
    selected_action = dict(selected.candidate.action)
    action_type = str((envelope.game.get("valid_actions") or {}).get("type") or envelope.game.get("phase") or "")
    anchor_action = _persuasion_advisory_anchor_action(envelope)
    receipt: dict[str, object] = {"contract": FAMILY_SELECTOR_POLICY_CONTRACT, "family": "persuasion", "policy": "polarity-override-calibrated-margin-v1", "selected_candidate_index": selected.candidate.index}
    if anchor_action is None:
        receipt.update({"status": "not-applicable", "reason": "no v2.7 advisory anchor"})
        return selected_action, [], receipt
    selected_polarity = _meta15_persuasion_polarity(selected_action, action_type)
    anchor_polarity = _meta15_persuasion_polarity(anchor_action, action_type)
    receipt.update({"selected_polarity": selected_polarity, "anchor_polarity": anchor_polarity})
    if selected_polarity == anchor_polarity and selected_polarity in {"positive", "negative"}:
        receipt.update({"status": "passed", "reason": "anchor polarity retained; exact wording remains selector-controlled"})
        return selected_action, [], receipt
    anchor_candidate = next((candidate for candidate in candidate_set.candidates if _meta15_persuasion_polarity(candidate.action, action_type) == anchor_polarity), None)
    selected_forecast = _meta15_aligned_row(conditional_surface, selected.candidate.index, payload_key="forecast")
    anchor_forecast = _meta15_aligned_row(conditional_surface, anchor_candidate.index, payload_key="forecast") if anchor_candidate is not None else None
    selected_buy = _meta15_probability(selected_forecast, "buy")
    anchor_buy = _meta15_probability(anchor_forecast, "buy")
    selected_sequence = _meta15_probability(selected_forecast, "buy", probability_key="sequence_probabilities")
    anchor_sequence = _meta15_probability(anchor_forecast, "buy", probability_key="sequence_probabilities")
    state = envelope.game.get("game_state") if isinstance(envelope.game.get("game_state"), dict) else {}
    round_number = state.get("round")
    total_rounds = state.get("total_rounds")
    remaining_fraction = 0.0
    if isinstance(round_number, int) and not isinstance(round_number, bool) and isinstance(total_rounds, int) and not isinstance(total_rounds, bool) and total_rounds > 1:
        remaining_fraction = max(0.0, min(1.0, (total_rounds - round_number) / (total_rounds - 1)))
    required_margin = _PERSUASION_POLARITY_OVERRIDE_BASE_BUY_MARGIN + _PERSUASION_POLARITY_OVERRIDE_EARLY_HORIZON_MARGIN * remaining_fraction
    blended_gain = selected_buy - anchor_buy if selected_buy is not None and anchor_buy is not None else None
    sequence_gain = selected_sequence - anchor_sequence if selected_sequence is not None and anchor_sequence is not None else None
    receipt.update({"remaining_horizon_fraction": round(remaining_fraction, 6), "required_buy_probability_gain": round(required_margin, 6), "selected_minus_anchor_buy_probability": round(blended_gain, 6) if blended_gain is not None else None, "selected_minus_anchor_sequence_buy_probability": round(sequence_gain, 6) if sequence_gain is not None else None})
    authorized = selected_polarity in {"positive", "negative"} and anchor_polarity in {"positive", "negative"} and blended_gain is not None and sequence_gain is not None and blended_gain >= required_margin and sequence_gain >= 0.75 * required_margin
    if authorized:
        receipt.update({"status": "passed", "reason": "opposite polarity clears the horizon-scaled blended and sequence expected-sale margins"})
        return selected_action, [], receipt
    guarded, safeguards = _apply_worker_safeguards(envelope, normalize_action(envelope.game, anchor_action))
    applied = ["persuasion_meta15_polarity_override_margin", *safeguards]
    receipt.update({"status": "polarity-reverted", "reason": "opposite polarity lacks the required calibrated expected-sale margin", "submitted_action": guarded})
    return guarded, applied, receipt


def _meta15_negotiation_selector_gate(*, envelope: TurnEnvelope, baseline: WorkerDecision, selected: Any) -> tuple[dict[str, Any], list[str], dict[str, object]]:
    """Require uncertainty-adjusted rating support before a low-share acceptance reverses continuation."""
    selected_action = dict(selected.candidate.action)
    receipt: dict[str, object] = {"contract": FAMILY_SELECTOR_POLICY_CONTRACT, "family": "negotiation", "policy": "categorical-common-frontier-with-low-share-rating-bound-v1", "selected_candidate_index": selected.candidate.index}
    if selected_action.get("decision") != "AcceptOffer" or baseline.action.get("decision") == "AcceptOffer":
        receipt.update({"status": "passed", "reason": "the low-share categorical reversal gate is not applicable"})
        return selected_action, [], receipt
    context = envelope.negotiation_advisor_context if isinstance(envelope.negotiation_advisor_context, Mapping) else {}
    facts = context.get("deterministic_decision_facts") if isinstance(context.get("deterministic_decision_facts"), Mapping) else {}
    complete = facts.get("complete_information_surplus") if isinstance(facts.get("complete_information_surplus"), Mapping) else {}
    own_share = _finite_float(complete.get("current_offer_own_surplus_share"))
    if own_share is None or own_share >= 0.5:
        receipt.update({"status": "passed", "reason": "the accepted offer is not a known sub-half surplus settlement"})
        return selected_action, [], receipt
    advisory = envelope.rating_v3_advisory if isinstance(envelope.rating_v3_advisory, Mapping) else {}
    branches = advisory.get("branches") if isinstance(advisory.get("branches"), Mapping) else {}
    agreement_rows = branches.get("agreement_branches") if isinstance(branches.get("agreement_branches"), list) else []
    current_price = _finite_float(facts.get("current_offer_price"))
    agreement = next((row.get("if_accepted") for row in agreement_rows if isinstance(row, Mapping) and current_price is not None and (price := _finite_float(row.get("price"))) is not None and math.isclose(price, current_price, rel_tol=1e-9, abs_tol=1e-6) and isinstance(row.get("if_accepted"), Mapping)), None)
    no_deal = branches.get("no_deal") if isinstance(branches.get("no_deal"), Mapping) else None
    agreement_interval = agreement.get("interval_80") if isinstance(agreement, Mapping) and isinstance(agreement.get("interval_80"), list) else None
    no_deal_interval = no_deal.get("interval_80") if isinstance(no_deal, Mapping) and isinstance(no_deal.get("interval_80"), list) else None
    agreement_lower = _finite_float(agreement_interval[0]) if agreement_interval and len(agreement_interval) == 2 else None
    no_deal_lower = _finite_float(no_deal_interval[0]) if no_deal_interval and len(no_deal_interval) == 2 else None
    required_advantage = 0.5
    receipt.update({"current_offer_own_surplus_share": round(own_share, 6), "agreement_interval_80_lower": agreement_lower, "no_deal_interval_80_lower": no_deal_lower, "required_lower_bound_advantage": required_advantage})
    if agreement_lower is not None and no_deal_lower is not None and agreement_lower > no_deal_lower + required_advantage:
        receipt.update({"status": "passed", "reason": "the low-share agreement clears the uncertainty-adjusted lower-bound advantage"})
        return selected_action, [], receipt
    guarded, safeguards = _apply_worker_safeguards(envelope, normalize_action(envelope.game, dict(baseline.action)))
    receipt.update({"status": "categorical-reverted", "reason": "a sparse low-share rating point estimate cannot reverse continuation without a material 80% lower-bound advantage", "submitted_action": guarded})
    return guarded, ["negotiation_meta15_low_share_rating_uncertainty_gate", *safeguards], receipt


def _meta15_apply_family_selector_policy(*, envelope: TurnEnvelope, baseline: WorkerDecision, selected: Any, candidate_set: Any, conditional_surface: Mapping[str, object], family_evidence: Mapping[str, object] | None) -> tuple[dict[str, Any], list[str], dict[str, object]]:
    family = str(envelope.game.get("game_family") or "")
    if family == "bargaining":
        return _meta15_bargaining_selector_gate(envelope=envelope, baseline=baseline, selected=selected, candidate_set=candidate_set, conditional_surface=conditional_surface, family_evidence=family_evidence)
    if family == "persuasion":
        return _meta15_persuasion_selector_gate(envelope=envelope, selected=selected, candidate_set=candidate_set, conditional_surface=conditional_surface)
    if family == "negotiation":
        return _meta15_negotiation_selector_gate(envelope=envelope, baseline=baseline, selected=selected)
    return dict(selected.candidate.action), [], {"contract": FAMILY_SELECTOR_POLICY_CONTRACT, "family": family, "policy": "candidate-specific-continuation-evidence-v1", "status": "selector-controlled", "reason": "No family-specific post-selector override applies"}


def _fallback_decision(envelope: TurnEnvelope, reason: str) -> WorkerDecision:
    started = time.monotonic()
    normalized = normalize_action(envelope.game, _persuasion_advisory_anchor_action(envelope) or _negotiation_one_round_advisory_anchor_action(envelope) or safe_action(envelope.game))
    action, safeguards = _apply_worker_safeguards(envelope, normalized)
    return WorkerDecision(
        action=action,
        proposal=None,
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=True,
        fallback_reason=reason,
        role=FAMILY_ROLES[str(envelope.game["game_family"])],
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="deterministic",
        branch_receipts=[],
    )


def _persuasion_buyer_continuation_total_variation(candidate_set: FrozenCandidateSet, continuation_surface: Mapping[str, object]) -> float:
    rows = continuation_surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("buyer-continuation surface does not cover its frozen candidate set")
    by_decision: dict[str, list[float]] = {}
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        forecast = row.get("forecast") if isinstance(row, Mapping) else None
        probabilities = forecast.get("response_probabilities") if isinstance(forecast, Mapping) else None
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(probabilities, list) or len(probabilities) != 3:
            raise ValueError("buyer-continuation surface lost candidate alignment")
        decision = str(candidate.action.get("decision") or "").casefold()
        by_decision[decision] = [float(value) for value in probabilities]
    if set(by_decision) != {"yes", "no"}:
        raise ValueError("buyer-continuation sensitivity requires exact buy and pass candidates")
    return 0.5 * sum(abs(left - right) for left, right in zip(by_decision["yes"], by_decision["no"], strict=True))


def _persuasion_buyer_selector_control(envelope: TurnEnvelope, candidate_set: FrozenCandidateSet, continuation_surface: Mapping[str, object]) -> dict[str, object]:
    economic = _persuasion_buyer_economic_control(envelope)
    if not isinstance(economic, Mapping):
        raise ValueError("Persuasion buyer selector has no frozen economic control")
    expected_ratio = _finite_float(economic.get("expected_surplus_over_price"))
    band = _finite_float(economic.get("indifference_band_ratio"))
    material_loss = _finite_float(economic.get("material_loss_ratio"))
    minimum_sensitivity = _finite_float(economic.get("continuation_minimum_total_variation"))
    if expected_ratio is None or band is None or material_loss is None or minimum_sensitivity is None or band < 0 or material_loss < 0 or not 0 <= minimum_sensitivity <= 1:
        raise ValueError("Persuasion buyer economic control is malformed")
    sensitivity = _persuasion_buyer_continuation_total_variation(candidate_set, continuation_surface)
    materially_negative = expected_ratio <= -material_loss
    economically_close = abs(expected_ratio) <= band
    sensitive = sensitivity >= minimum_sensitivity
    common_unit_lower_bound = _finite_float(economic.get("common_unit_continuation_lower_bound_over_price"))
    robust_action = economic.get("robust_uncertainty_action") if isinstance(economic.get("robust_uncertainty_action"), Mapping) else economic.get("local_expected_value_action")
    local_action = dict(robust_action) if isinstance(robust_action, Mapping) else {"decision": "yes" if expected_ratio > 0.0 else "no"}
    selector_eligible = economically_close and sensitive and not materially_negative and common_unit_lower_bound is not None
    if materially_negative:
        reason = "materially negative immediate expected surplus has no common-unit bounded continuation offset"
    elif not economically_close:
        reason = "immediate expected surplus lies outside the frozen indifference band"
    elif not sensitive:
        reason = "candidate-conditioned next-signal distributions do not clear the material sensitivity threshold"
    elif common_unit_lower_bound is None:
        reason = "predictive next-signal change has no lower-bounded continuation value in current-payoff units"
    else:
        reason = "economically close current value and material next-signal sensitivity have a common-unit continuation lower bound"
    return {
        "contract": "glee-persuasion-buyer-selector-gate-v1",
        "expected_surplus_over_price": round(expected_ratio, 9),
        "indifference_band_ratio": band,
        "material_loss_ratio": material_loss,
        "materially_negative_purchase": materially_negative,
        "continuation_total_variation": round(sensitivity, 9),
        "continuation_minimum_total_variation": minimum_sensitivity,
        "economically_close": economically_close,
        "continuation_sensitive": sensitive,
        "selector_eligible": selector_eligible,
        "local_expected_value_action": local_action,
        "common_unit_continuation_lower_bound_over_price": common_unit_lower_bound,
        "rating_authority": "advisory-only; point estimates cannot reverse the economic gate",
        "reason": reason,
    }


def _amount(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def analytic_cold_opening_decision(envelope: TurnEnvelope) -> WorkerDecision | None:
    """Bypass inference for one identifiable opening with no opponent-specific prior game."""
    prior_games = envelope.snapshot.memory_context.get("opponent_prior_game_count")
    if isinstance(prior_games, bool) or not isinstance(prior_games, int) or prior_games != 0:
        return None
    opening = analytic_bargaining_opening(envelope.game)
    if opening is None:
        return None
    started = time.monotonic()
    normalized = normalize_action(envelope.game, opening.action)
    action, safeguards = _apply_worker_safeguards(envelope, normalized)
    receipt = {
        "branch": "analytic-cold",
        "status": "succeeded",
        "method": opening.method,
        "proposer": opening.proposer,
        "delta_1": opening.delta_1,
        "delta_2": opening.delta_2,
        "max_rounds": opening.max_rounds,
        "predicted_responder_gain": opening.responder_gain,
        "action": action,
        "deterministic_safeguards": safeguards,
    }
    return WorkerDecision(
        action=action,
        proposal=dict(opening.action),
        tetrad_update=None,
        tetrad_transport_issues=[],
        deterministic_safeguards=safeguards,
        fallback=False,
        fallback_reason=None,
        role="glee_analytic_bargaining",
        elapsed_s=round(time.monotonic() - started, 6),
        call_metadata=None,
        selection_branch="analytic-cold",
        branch_receipts=[receipt],
    )


class GleeTurnWorker:
    """Make one bounded model call and return a proposal without touching GLEE or memory."""

    def __init__(self, *, model_runner: Any, model: str, effort: str, minimum_start_budget_s: float = 10.0) -> None:
        self.model_runner = model_runner
        self.model = model
        self.effort = effort
        self.minimum_start_budget_s = minimum_start_budget_s

    def fallback(self, envelope: TurnEnvelope, reason: str) -> WorkerDecision:
        return _fallback_decision(envelope, reason)

    def solve(self, envelope: TurnEnvelope) -> WorkerDecision:
        started = time.monotonic()
        envelope = _prepare_bargaining_analytic_authority(envelope)
        preinference = _preinference_decision(envelope)
        if preinference is not None:
            return replace(preinference, elapsed_s=round(time.monotonic() - started, 6))
        remaining = envelope.deadline_at_monotonic - started
        if remaining <= self.minimum_start_budget_s:
            return self.fallback(envelope, f"deadline budget too small for inference: {remaining:.3f}s")
        family = str(envelope.game["game_family"])
        role = FAMILY_ROLES[family]
        proposal: dict[str, Any] | None = None
        safeguards: list[str] = []
        metadata: dict[str, object] | None = None
        try:
            model_cls = glee_action_model(action_model(envelope.game))
            parsed, call_metadata = self.model_runner.call_structured(
                role,
                build_worker_prompt(envelope),
                model_cls,
                model=self.model,
                effort=self.effort,
            )
            proposal = parsed.action.model_dump(exclude_none=True)
            normalized = normalize_action(envelope.game, parsed.action)
            action, safeguards = _apply_worker_safeguards(envelope, normalized)
            metadata = _metadata(call_metadata)
            return WorkerDecision(
                action=action,
                proposal=proposal,
                tetrad_update=None,
                tetrad_transport_issues=[],
                deterministic_safeguards=safeguards,
                fallback=False,
                fallback_reason=None,
                role=role,
                elapsed_s=round(time.monotonic() - started, 6),
                call_metadata=metadata,
            )
        except Exception as error:
            fallback = self.fallback(envelope, f"{type(error).__name__}: {error}")
            return WorkerDecision(
                action=fallback.action,
                proposal=proposal,
                tetrad_update=None,
                tetrad_transport_issues=[],
                deterministic_safeguards=fallback.deterministic_safeguards,
                fallback=True,
                fallback_reason=fallback.fallback_reason,
                role=role,
                elapsed_s=round(time.monotonic() - started, 6),
                call_metadata=metadata,
            )


class MaxHighGleeTurnWorker:
    """Race Max with High except in explicitly configured High-only families."""

    def __init__(
        self,
        *,
        runner_factory: Callable[[int], Any],
        model: str,
        max_effort: str = "max",
        high_effort: str = "high",
        model_timeout_s: float = 108.0,
        finalization_margin_s: float = 12.0,
        minimum_model_budget_s: float = 5.0,
        bargaining_opening_policy: str = "rmm",
        high_only_families: tuple[str, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if model_timeout_s <= 0:
            raise ValueError("the shared effort-race timeout must be positive")
        if finalization_margin_s < 0:
            raise ValueError("finalization margin cannot be negative")
        if minimum_model_budget_s <= 0:
            raise ValueError("minimum model budget must be positive")
        if not high_effort.strip():
            raise ValueError("High effort must be nonempty")
        if bargaining_opening_policy not in {"analytic-cold", "rmm"}:
            raise ValueError("bargaining_opening_policy must be 'analytic-cold' or 'rmm'")
        unknown_high_only_families = sorted(set(high_only_families) - set(FAMILY_ROLES))
        if unknown_high_only_families:
            raise ValueError(f"unsupported High-only families: {unknown_high_only_families}")
        self.runner_factory = runner_factory
        self.model = model
        self.max_effort = max_effort
        self.high_effort = high_effort
        self.model_timeout_s = model_timeout_s
        self.finalization_margin_s = finalization_margin_s
        self.minimum_model_budget_s = minimum_model_budget_s
        self.minimum_start_budget_s = finalization_margin_s + minimum_model_budget_s
        self.bargaining_opening_policy = bargaining_opening_policy
        self.high_only_families = frozenset(high_only_families)
        self.clock = clock

    def fallback(self, envelope: TurnEnvelope, reason: str) -> WorkerDecision:
        return _fallback_decision(envelope, reason)

    @staticmethod
    def _timeout(value: float) -> int:
        return max(1, math.floor(value))

    def _run_branch(
        self,
        *,
        branch: str,
        effort: str,
        timeout_s: int,
        role: str,
        body: str,
        model_cls: type[Any],
        envelope: TurnEnvelope,
    ) -> tuple[WorkerDecision | None, dict[str, object]]:
        started = self.clock()
        metadata: object = None
        try:
            runner = self.runner_factory(timeout_s)
            parsed, metadata = runner.call_structured(role, body, model_cls, model=self.model, effort=effort)
            candidate = _candidate_from_parsed(envelope, parsed, role=role, elapsed_s=self.clock() - started, call_metadata=metadata)
            return candidate, {
                "branch": branch,
                "status": "succeeded",
                "timeout_s": timeout_s,
                "effort": effort,
                "elapsed_s": candidate.elapsed_s,
                "proposal": candidate.proposal,
                "guarded_action": candidate.action,
                "deterministic_safeguards": candidate.deterministic_safeguards,
                "tetrad_update": candidate.tetrad_update.model_dump(mode="json") if candidate.tetrad_update is not None else None,
                "transport_issues": candidate.tetrad_transport_issues,
                "call_metadata": candidate.call_metadata,
            }
        except Exception as error:
            return None, {
                "branch": branch,
                "status": "failed",
                "timeout_s": timeout_s,
                "effort": effort,
                "elapsed_s": round(self.clock() - started, 6),
                "error": f"{type(error).__name__}: {error}",
                "call_metadata": _metadata(metadata),
            }

    def solve(self, envelope: TurnEnvelope) -> WorkerDecision:
        started = self.clock()
        if self.bargaining_opening_policy == "analytic-cold":
            analytic = analytic_cold_opening_decision(envelope)
            if analytic is not None:
                return analytic
        envelope = _prepare_bargaining_analytic_authority(envelope)
        preinference = _preinference_decision(envelope)
        if preinference is not None:
            return replace(preinference, elapsed_s=round(self.clock() - started, 6))
        budget_deadline = min(envelope.deadline_at_monotonic - self.finalization_margin_s, started + self.model_timeout_s)
        initial_budget = budget_deadline - started
        baseline = self.fallback(envelope, "neither effort branch produced a valid action")
        branch_receipts: list[dict[str, object]] = [{
            "branch": "deterministic",
            "status": "ready",
            "action": baseline.action,
            "deterministic_safeguards": baseline.deterministic_safeguards,
        }]
        if initial_budget < self.minimum_model_budget_s:
            return replace(
                baseline,
                fallback_reason=f"deadline budget too small for parallel inference: {initial_budget:.3f}s",
                elapsed_s=round(self.clock() - started, 6),
                branch_receipts=branch_receipts,
            )

        family = str(envelope.game["game_family"])
        role = FAMILY_ROLES[family]
        body = build_worker_prompt(envelope)
        model_cls = glee_action_model(action_model(envelope.game))
        shared_timeout = self._timeout(budget_deadline - self.clock())
        branches = [("high", self.high_effort)] if family in self.high_only_families else [("high", self.high_effort), ("max", self.max_effort)]
        candidates: dict[str, WorkerDecision | None] = {}
        with ThreadPoolExecutor(max_workers=len(branches), thread_name_prefix="glee-effort") as pool:
            futures = [
                (
                    branch,
                    pool.submit(
                        self._run_branch,
                        branch=branch,
                        effort=effort,
                        timeout_s=shared_timeout,
                        role=role,
                        body=body,
                        model_cls=model_cls,
                        envelope=envelope,
                    ),
                )
                for branch, effort in branches
            ]
            for branch, future in futures:
                candidate, receipt = future.result()
                candidates[branch] = candidate
                branch_receipts.append(receipt)

        selected = candidates.get("max") or candidates.get("high") or baseline
        selection_branch = next((branch for branch in ("max", "high") if candidates.get(branch) is not None), "deterministic")
        fallback_reason: str | None = None
        if selection_branch == "deterministic":
            failures = [f"{receipt['branch']}: {receipt.get('error')}" for receipt in branch_receipts if receipt.get("status") == "failed"]
            fallback_reason = "; ".join(failures) or "no effort branch had enough time to return a valid action"
        return WorkerDecision(
            action=selected.action,
            proposal=selected.proposal,
            tetrad_update=selected.tetrad_update,
            tetrad_transport_issues=selected.tetrad_transport_issues,
            deterministic_safeguards=selected.deterministic_safeguards,
            fallback=selection_branch == "deterministic",
            fallback_reason=fallback_reason,
            role=selected.role,
            elapsed_s=round(self.clock() - started, 6),
            call_metadata=selected.call_metadata,
            selection_branch=selection_branch,
            branch_receipts=branch_receipts,
        )


class CapacityFallbackGleeTurnWorker:
    """Try a fixed all-family model chain, with one bounded same-model primary retry."""

    def __init__(
        self,
        *,
        runner_factory: Callable[[int], Any],
        model_chain: tuple[tuple[str, str, str], ...] = CAPACITY_MODEL_CHAIN,
        model_timeout_s: float = 100.0,
        finalization_margin_s: float = 12.0,
        minimum_model_budget_s: float = 5.0,
        minimum_retry_budget_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if model_timeout_s <= 0:
            raise ValueError("the capacity-chain budget must be positive")
        if finalization_margin_s < 0:
            raise ValueError("finalization margin cannot be negative")
        if minimum_model_budget_s <= 0:
            raise ValueError("minimum model budget must be positive")
        if minimum_retry_budget_s < minimum_model_budget_s:
            raise ValueError("minimum retry budget cannot be smaller than the minimum model budget")
        if not model_chain:
            raise ValueError("the capacity model chain cannot be empty")
        if len({branch for branch, _model, _effort in model_chain}) != len(model_chain):
            raise ValueError("capacity model-chain branch names must be unique")
        if any(not branch.strip() or not model.strip() or not effort.strip() for branch, model, effort in model_chain):
            raise ValueError("every capacity model-chain field must be nonempty")
        self.runner_factory = runner_factory
        self.model_chain = tuple(model_chain)
        self.model_timeout_s = model_timeout_s
        self.finalization_margin_s = finalization_margin_s
        self.minimum_model_budget_s = minimum_model_budget_s
        self.minimum_retry_budget_s = minimum_retry_budget_s
        self.minimum_start_budget_s = finalization_margin_s + minimum_model_budget_s
        self.clock = clock

    @property
    def manifest_chain(self) -> list[dict[str, str]]:
        return [{"branch": branch, "model": model, "effort": effort} for branch, model, effort in self.model_chain]

    @staticmethod
    def _timeout(value: float) -> int:
        return max(1, math.floor(value))

    def fallback(self, envelope: TurnEnvelope, reason: str) -> WorkerDecision:
        return _fallback_decision(envelope, reason)

    def solve(self, envelope: TurnEnvelope) -> WorkerDecision:
        started = self.clock()
        family = str(envelope.game.get("game_family") or "")
        if family not in FAMILY_ROLES:
            return self.fallback(envelope, f"the capacity fallback chain does not support family {family!r}")
        if family == "bargaining":
            envelope = _prepare_bargaining_analytic_authority(envelope)
        preinference = _preinference_decision(envelope)
        if preinference is not None:
            return replace(preinference, elapsed_s=round(self.clock() - started, 6))
        budget_deadline = min(envelope.deadline_at_monotonic - self.finalization_margin_s, started + self.model_timeout_s)
        baseline = self.fallback(envelope, "no capacity-chain model produced a valid action")
        branch_receipts: list[dict[str, object]] = [
            {
                "branch": "deterministic",
                "status": "ready",
                "action": baseline.action,
                "deterministic_safeguards": baseline.deterministic_safeguards,
            }
        ]
        initial_budget = budget_deadline - self.clock()
        if initial_budget < self.minimum_model_budget_s:
            return replace(
                baseline,
                fallback_reason=f"deadline budget too small for capacity-chain inference: {initial_budget:.3f}s",
                elapsed_s=round(self.clock() - started, 6),
                branch_receipts=branch_receipts,
            )
        role = FAMILY_ROLES[family]
        body = build_worker_prompt(envelope)
        model_cls = glee_action_model(action_model(envelope.game))
        failures: list[str] = []
        attempts = [(branch, model, effort, False) for branch, model, effort in self.model_chain]
        terra_retried = False
        attempt_index = 0
        while attempt_index < len(attempts):
            branch, model, effort, is_retry = attempts[attempt_index]
            attempt_index += 1
            remaining = budget_deadline - self.clock()
            if remaining < self.minimum_model_budget_s:
                branch_receipts.append(
                    {
                        "branch": branch,
                        "model": model,
                        "effort": effort,
                        "status": "skipped",
                        "reason": f"remaining model budget {remaining:.3f}s is below {self.minimum_model_budget_s:.3f}s",
                    }
                )
                failures.append(f"{branch}: insufficient remaining model budget")
                break
            timeout_s = self._timeout(remaining)
            branch_started = self.clock()
            metadata: object = None
            try:
                runner = self.runner_factory(timeout_s)
                parsed, metadata = runner.call_structured(role, body, model_cls, model=model, effort=effort)
                candidate = _candidate_from_parsed(envelope, parsed, role=role, elapsed_s=self.clock() - branch_started, call_metadata=metadata)
            except Exception as error:
                capacity = _zero_inference_model_capacity(error)
                capacity_mentioned = _mentions_model_capacity(error)
                timeout_failure = _is_timeout_failure(error)
                retry_budget = budget_deadline - self.clock()
                terra_retry_eligible = branch == self.model_chain[0][0] and not terra_retried and capacity is None and not capacity_mentioned and not timeout_failure and retry_budget >= self.minimum_retry_budget_s
                receipt: dict[str, object] = {
                    "branch": branch,
                    "model": model,
                    "effort": effort,
                    "status": "failed",
                    "timeout_s": timeout_s,
                    "elapsed_s": round(self.clock() - branch_started, 6),
                    "error": f"{type(error).__name__}: {error}",
                    "capacity_fallback_eligible": capacity is not None,
                    "capacity_proof": capacity,
                    "capacity_diagnostic_without_zero_inference_proof": capacity_mentioned and capacity is None,
                    "timeout_failure": timeout_failure,
                    "terra_retry_eligible": terra_retry_eligible,
                    "call_metadata": _metadata(metadata),
                }
                branch_receipts.append(receipt)
                failures.append(f"{branch}: {receipt['error']}")
                if capacity is None:
                    if terra_retry_eligible:
                        terra_retried = True
                        attempts.insert(attempt_index, (f"{branch}-retry", model, effort, True))
                        continue
                    break
                continue
            branch_receipts.append(
                {
                    "branch": branch,
                    "model": model,
                    "effort": effort,
                    "same_model_retry": is_retry,
                    "status": "succeeded",
                    "timeout_s": timeout_s,
                    "elapsed_s": candidate.elapsed_s,
                    "proposal": candidate.proposal,
                    "guarded_action": candidate.action,
                    "deterministic_safeguards": candidate.deterministic_safeguards,
                    "tetrad_update": candidate.tetrad_update.model_dump(mode="json") if candidate.tetrad_update is not None else None,
                    "transport_issues": candidate.tetrad_transport_issues,
                    "call_metadata": candidate.call_metadata,
                }
            )
            return WorkerDecision(
                action=candidate.action,
                proposal=candidate.proposal,
                tetrad_update=candidate.tetrad_update,
                tetrad_transport_issues=candidate.tetrad_transport_issues,
                deterministic_safeguards=candidate.deterministic_safeguards,
                fallback=False,
                fallback_reason=None,
                role=candidate.role,
                elapsed_s=round(self.clock() - started, 6),
                call_metadata=candidate.call_metadata,
                selection_branch=branch,
                branch_receipts=branch_receipts,
            )
        return replace(
            baseline,
            fallback_reason="; ".join(failures) or "no capacity-chain model had enough time to run",
            elapsed_s=round(self.clock() - started, 6),
            branch_receipts=branch_receipts,
        )


class MetaControllerV15GleeTurnWorker:
    """Run independent cloud planning, one local conditional batch, and constrained cloud selection."""

    def __init__(
        self,
        *,
        runner_factory: Callable[[int], Any],
        conditional_client: Any,
        self_mirror_client: Any | None = None,
        model: str = "gpt-5.6-terra",
        effort: str = "high",
        model_timeout_s: float = 108.0,
        planner_timeout_s: float = 48.0,
        finalization_margin_s: float = 12.0,
        minimum_planner_budget_s: float = 5.0,
        minimum_selector_budget_s: float = 12.0,
        selector_backend: SelectorBackend | None = None,
        ineligible_model_chain: tuple[tuple[str, str, str], ...] = CAPACITY_MODEL_CHAIN,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if model_timeout_s <= 0 or planner_timeout_s <= 0:
            raise ValueError("1.5-round model budgets must be positive")
        if finalization_margin_s < 0 or minimum_planner_budget_s <= 0 or minimum_selector_budget_s <= 0:
            raise ValueError("1.5-round reserve budgets are invalid")
        if planner_timeout_s + minimum_selector_budget_s > model_timeout_s:
            raise ValueError("planner timeout plus minimum selector budget exceeds the shared model window")
        if not model.strip() or not effort.strip():
            raise ValueError("1.5-round cloud model and effort must be nonempty")
        self.runner_factory = runner_factory
        self.conditional_client = conditional_client
        self.self_mirror_client = self_mirror_client
        self.model = model
        self.effort = effort
        self.model_timeout_s = float(model_timeout_s)
        self.planner_timeout_s = float(planner_timeout_s)
        self.finalization_margin_s = float(finalization_margin_s)
        self.minimum_planner_budget_s = float(minimum_planner_budget_s)
        self.minimum_selector_budget_s = float(minimum_selector_budget_s)
        self.conditional_reserve_s = max(1.0, float(getattr(conditional_client, "timeout_s", 3.0)))
        self.self_mirror_reserve_s = max(1.0, float(getattr(self_mirror_client, "timeout_s", 0.0))) if self_mirror_client is not None else 0.0
        self.minimum_start_budget_s = self.finalization_margin_s + self.minimum_planner_budget_s + self.conditional_reserve_s + self.self_mirror_reserve_s + self.minimum_selector_budget_s
        self.clock = clock
        self.selector_backend = selector_backend or TerraSelectorBackend(runner_factory=runner_factory, model=model, effort=effort, clock=clock)
        self.single_call_worker = CapacityFallbackGleeTurnWorker(runner_factory=runner_factory, model_chain=ineligible_model_chain, model_timeout_s=model_timeout_s, finalization_margin_s=finalization_margin_s, clock=clock)

    @property
    def manifest_receipt(self) -> dict[str, object]:
        return {
            "contract": ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT,
            "model": self.model,
            "effort": self.effort,
            "model_timeout_s": self.model_timeout_s,
            "planner_timeout_s": self.planner_timeout_s,
            "minimum_selector_budget_s": self.minimum_selector_budget_s,
            "conditional_reserve_s": self.conditional_reserve_s,
            "self_mirror_reserve_s": self.self_mirror_reserve_s,
            "public_self_mirror": copy.deepcopy(dict(self.self_mirror_client.receipt)) if self.self_mirror_client is not None else None,
            "selector_authority_contract": SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT,
            "family_selector_policy_contract": FAMILY_SELECTOR_POLICY_CONTRACT,
            "selector_backend": copy.deepcopy(dict(self.selector_backend.manifest_receipt)),
            "ineligible_model_chain": self.single_call_worker.manifest_chain,
            "fallback": "existing guarded deterministic action",
            "ineligible_turn_worker": "capacity-chain-v1",
        }

    def _record_self_mirror_selection(self, *, envelope: TurnEnvelope, surface: Mapping[str, object] | None, candidate_set: Any, selected_candidate: Any | None, submitted_action: Mapping[str, object], branch_receipts: list[dict[str, object]]) -> None:
        if self.self_mirror_client is None or surface is None:
            return
        candidate = selected_candidate
        if candidate is None:
            submitted_sha256 = object_sha256(dict(submitted_action))
            candidate = next((value for value in candidate_set.candidates if value.action_sha256 == submitted_sha256), None)
        try:
            receipt = self.self_mirror_client.record_selection(
                turn_id=envelope.snapshot.turn_id,
                forecast_sha256=str(surface["forecast_sha256"]),
                selected_candidate_index=candidate.index if candidate is not None else None,
                selected_action_sha256=candidate.action_sha256 if candidate is not None else None,
                submitted_action=dict(submitted_action),
            )
            branch_receipts.append({"branch": "public-self-mirror-selection", "status": "recorded", "receipt": receipt})
        except Exception as error:
            branch_receipts.append({"branch": "public-self-mirror-selection", "status": "failed-advisory", "error": f"{type(error).__name__}: {error}"})

    @staticmethod
    def _timeout(value: float) -> int:
        return max(1, math.floor(value))

    @staticmethod
    def _eligible(envelope: TurnEnvelope) -> bool:
        game = envelope.game
        family = str(game.get("game_family") or "")
        action_type = str((game.get("valid_actions") or {}).get("type") or game.get("phase") or "")
        if family == "bargaining":
            return action_type == "offer"
        if family == "negotiation":
            return action_type in {"offer", "decision"}
        if family != "persuasion" or action_type not in {"seller_message", "seller_recommendation"}:
            return False
        state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
        player = str(game.get("your_player") or "")
        return str(state.get(f"{player}_role") or "").casefold() == "seller"

    @staticmethod
    def _buyer_continuation_eligible(envelope: TurnEnvelope) -> bool:
        game = envelope.game
        if game.get("game_family") != "persuasion":
            return False
        action_type = str((game.get("valid_actions") or {}).get("type") or game.get("phase") or "")
        state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
        player = str(game.get("your_player") or "")
        round_number = state.get("round")
        total_rounds = state.get("total_rounds")
        if action_type != "buyer_decision" or str(state.get(f"{player}_role") or "").casefold() != "buyer":
            return False
        if isinstance(round_number, bool) or not isinstance(round_number, int) or isinstance(total_rounds, bool) or not isinstance(total_rounds, int) or not 1 <= round_number < total_rounds:
            return False
        context = envelope.persuasion_advisor_context
        if context is None:
            from .glee_persuasion_live_v2 import persuasion_decision_facts

            context = persuasion_decision_facts(game, ())
        authority = context.get("authority") if isinstance(context, Mapping) else None
        economic = context.get("buyer_economic_control") if isinstance(context, Mapping) else None
        return isinstance(authority, Mapping) and authority.get("action_authority") == "advisory-only" and authority.get("selected_action") is None and isinstance(economic, Mapping) and economic.get("contract") == "glee-persuasion-buyer-economic-control-v1"

    def fallback(self, envelope: TurnEnvelope, reason: str) -> WorkerDecision:
        return _fallback_decision(envelope, reason)

    def _failed(self, *, baseline: WorkerDecision, started: float, branch_receipts: list[dict[str, object]], reason: str) -> WorkerDecision:
        return replace(baseline, fallback_reason=reason, elapsed_s=round(self.clock() - started, 6), branch_receipts=branch_receipts)

    def _solve_persuasion_buyer_continuation(self, envelope: TurnEnvelope, *, started: float) -> WorkerDecision:
        """Select buy or pass after local candidate-conditioned next-seller-signal inference."""
        budget_deadline = min(envelope.deadline_at_monotonic - self.finalization_margin_s, started + self.model_timeout_s)
        baseline = self.fallback(envelope, "buyer-continuation selector did not commit a candidate")
        branch_receipts: list[dict[str, object]] = [{"branch": "deterministic", "status": "ready", "action": baseline.action, "deterministic_safeguards": baseline.deterministic_safeguards}]
        try:
            candidate_set = freeze_fixed_candidates(
                game=envelope.game,
                candidate_specs=(
                    {"action": {"decision": "yes"}, "purpose": "buy the current product"},
                    {"action": {"decision": "no"}, "purpose": "pass on the current product"},
                ),
                guard=lambda action: apply_deterministic_safeguards(envelope.game, action),
            )
        except Exception as error:
            branch_receipts.append({"branch": "buyer-candidate-freeze", "status": "failed", "error": f"{type(error).__name__}: {error}"})
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"buyer candidate freeze failed: {type(error).__name__}: {error}")
        branch_receipts.append({"branch": "buyer-candidate-freeze", "status": "succeeded", "candidate_set": candidate_set.receipt(), "planner_call": "not-applicable-exhaustive-binary-action-set"})
        conditional_started = self.clock()
        try:
            conditional_receipt = self.conditional_client.forecast_buyer_continuation(game=envelope.game, turn_id=envelope.snapshot.turn_id, candidate_set=candidate_set)
        except Exception as error:
            branch_receipts.append({"branch": "persuasion-buyer-continuation", "status": "failed", "elapsed_s": round(self.clock() - conditional_started, 6), "error": f"{type(error).__name__}: {error}"})
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"buyer-continuation inference failed: {type(error).__name__}: {error}")
        branch_receipts.append({"branch": "persuasion-buyer-continuation", "status": "succeeded", "elapsed_s": round(self.clock() - conditional_started, 6), "receipt": conditional_receipt})
        try:
            selector_control = _persuasion_buyer_selector_control(envelope, candidate_set, conditional_receipt)
        except Exception as error:
            branch_receipts.append({"branch": "persuasion-buyer-selector-gate", "status": "failed", "error": f"{type(error).__name__}: {error}"})
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"buyer selector gate failed: {type(error).__name__}: {error}")
        branch_receipts.append({"branch": "persuasion-buyer-selector-gate", "status": "selector-eligible" if selector_control["selector_eligible"] else "resolved-locally", "control": selector_control})
        if selector_control["selector_eligible"] is not True:
            proposal = normalize_action(envelope.game, dict(selector_control["local_expected_value_action"]))
            action, safeguards = _apply_worker_safeguards(envelope, proposal)
            return WorkerDecision(
                action=action,
                proposal=proposal,
                tetrad_update=None,
                tetrad_transport_issues=[],
                deterministic_safeguards=safeguards,
                fallback=False,
                fallback_reason=None,
                role="glee_persuasion_buyer_economic_control",
                elapsed_s=round(self.clock() - started, 6),
                call_metadata=None,
                selection_branch="persuasion-buyer-economic-local",
                branch_receipts=branch_receipts,
            )
        self_mirror_surface: dict[str, object] | None = None
        self_mirror_admissibility: dict[str, object] | None = None
        if self.self_mirror_client is not None:
            mirror_started = self.clock()
            try:
                self_mirror_surface = self.self_mirror_client.forecast_candidates(game=envelope.game, turn_id=envelope.snapshot.turn_id, candidate_set=candidate_set)
                self_mirror_admissibility = _meta15_public_self_mirror_admissibility(candidate_set=candidate_set, conditional_surface=None, family_evidence=None, buyer_path=True)
                branch_receipts.append({"branch": "public-self-mirror", "status": "succeeded-diagnostic-only", "elapsed_s": round(self.clock() - mirror_started, 6), "surface": self_mirror_surface, "admissibility": self_mirror_admissibility})
            except Exception as error:
                self_mirror_surface = None
                self_mirror_admissibility = None
                branch_receipts.append({"branch": "public-self-mirror", "status": "failed-advisory", "elapsed_s": round(self.clock() - mirror_started, 6), "error": f"{type(error).__name__}: {error}"})
        try:
            selector_payload = build_persuasion_buyer_continuation_selector_payload(worker_payload=worker_payload(envelope), candidate_set=candidate_set, continuation_surface=conditional_receipt, public_self_mirror_surface=self_mirror_surface, public_self_mirror_admissibility=self_mirror_admissibility)
            selector_payload["buyer_selector_gate"] = copy.deepcopy(selector_control)
        except Exception as error:
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"buyer selector payload failed: {type(error).__name__}: {error}")
        selector_remaining = budget_deadline - self.clock()
        if selector_remaining < self.minimum_selector_budget_s:
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"deadline budget too small for buyer-continuation selector: {selector_remaining:.3f}s")
        fallback = next((candidate for candidate in candidate_set.candidates if dict(candidate.action) == baseline.action), candidate_set.candidates[0])
        selector_started = self.clock()
        selector_metadata: object = None
        selector_role = "glee_persuasion_buyer_continuation_selector"
        selector_backend_receipt: Mapping[str, object] | None = None
        try:
            selector_request = build_selector_backend_request(candidate_set=candidate_set, selector_payload=selector_payload, fallback_candidate_id=fallback.action_sha256, timeout_s=selector_remaining)
            selector_outcome = self.selector_backend.select(selector_request)
            if self.clock() > budget_deadline:
                raise TimeoutError("buyer-continuation selector completed after the reserved finalization boundary")
            selected = selector_outcome.selected
            selector_metadata = selector_outcome.metadata
            selector_role = selector_outcome.role
            selector_backend_receipt = selector_outcome.receipt
        except Exception as error:
            branch_receipts.append({"branch": "persuasion-buyer-continuation-selector", "status": "failed", "role": selector_role, "timeout_s": self._timeout(selector_remaining), "elapsed_s": round(self.clock() - selector_started, 6), "backend": copy.deepcopy(dict(self.selector_backend.manifest_receipt)), "error": f"{type(error).__name__}: {error}", "call_metadata": _metadata(selector_metadata)})
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"buyer-continuation selector failed: {type(error).__name__}: {error}")
        final_action, safeguards = _apply_worker_safeguards(envelope, dict(selected.candidate.action))
        branch_receipts.append({"branch": "persuasion-buyer-continuation-selector", "status": "succeeded", "role": selector_role, "timeout_s": self._timeout(selector_remaining), "elapsed_s": round(self.clock() - selector_started, 6), "selected_candidate": selected.candidate.receipt(), "backend": copy.deepcopy(dict(self.selector_backend.manifest_receipt)), "backend_result": copy.deepcopy(dict(selector_backend_receipt or {})), "call_metadata": _metadata(selector_metadata)})
        self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=selected.candidate, submitted_action=final_action, branch_receipts=branch_receipts)
        return WorkerDecision(
            action=final_action,
            proposal=dict(selected.candidate.action),
            tetrad_update=None,
            tetrad_transport_issues=[],
            deterministic_safeguards=[*selected.candidate.safeguards, *safeguards],
            fallback=False,
            fallback_reason=None,
            role=selector_role,
            elapsed_s=round(self.clock() - started, 6),
            call_metadata={"selector": _metadata(selector_metadata)},
            selection_branch="persuasion-buyer-continuation-selector",
            branch_receipts=branch_receipts,
        )

    def solve(self, envelope: TurnEnvelope) -> WorkerDecision:
        started = self.clock()
        family = str(envelope.game.get("game_family") or "")
        if family == "bargaining":
            envelope = _prepare_bargaining_analytic_authority(envelope)
        preinference = _preinference_decision(envelope)
        if preinference is not None:
            return replace(preinference, elapsed_s=round(self.clock() - started, 6))
        if self._buyer_continuation_eligible(envelope):
            return self._solve_persuasion_buyer_continuation(envelope, started=started)
        if not self._eligible(envelope):
            return self.single_call_worker.solve(envelope)
        budget_deadline = min(envelope.deadline_at_monotonic - self.finalization_margin_s, started + self.model_timeout_s)
        baseline = self.fallback(envelope, "1.5-round controller did not commit a selected candidate")
        branch_receipts: list[dict[str, object]] = [
            {"branch": "deterministic", "status": "ready", "action": baseline.action, "deterministic_safeguards": baseline.deterministic_safeguards}
        ]
        worker_turn = worker_payload(envelope)
        candidate_seed_specs, require_candidate_seeds = _meta15_candidate_seed_specs(envelope, baseline)
        planner_payload = build_planner_payload_v15(worker_payload=worker_turn, deterministic_candidates=[dict(spec["action"]) for spec in candidate_seed_specs])
        planner_available = budget_deadline - self.clock() - self.conditional_reserve_s - self.self_mirror_reserve_s - self.minimum_selector_budget_s
        planner_timeout = min(self.planner_timeout_s, planner_available)
        if planner_timeout < self.minimum_planner_budget_s:
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"deadline budget too small for 1.5-round planner: {planner_timeout:.3f}s")
        planner_started = self.clock()
        planner_metadata: object = None
        planner_role = f"glee_meta_controller_v2_15_planner_{family}"
        try:
            planner_runner = self.runner_factory(self._timeout(planner_timeout))
            planner_parsed, planner_metadata = planner_runner.call_structured(
                planner_role,
                json.dumps(planner_payload, ensure_ascii=False, separators=(",", ":")),
                nommd_candidate_plan_model(action_model(envelope.game)),
                model=self.model,
                effort=self.effort,
            )
            def candidate_guard(action: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
                guarded, safeguards = _apply_worker_safeguards(envelope, action)
                return guarded, safeguards

            candidate_set = freeze_planner_candidates(game=envelope.game, parsed=planner_parsed, guard=candidate_guard, required_candidates=candidate_seed_specs if require_candidate_seeds else ())
        except GuardedSingleAction as constrained:
            candidate = constrained.candidate_set.candidates[0]
            final_action, safeguards = _apply_worker_safeguards(envelope, dict(candidate.action))
            branch_receipts.append({"branch": "meta15-planner", "status": "succeeded-guarded-single-action", "role": planner_role, "timeout_s": self._timeout(planner_timeout), "elapsed_s": round(self.clock() - planner_started, 6), "receipt": constrained.receipt, "call_metadata": _metadata(planner_metadata)})
            return WorkerDecision(
                action=final_action,
                proposal=dict(candidate.action),
                tetrad_update=None,
                tetrad_transport_issues=[],
                deterministic_safeguards=[*candidate.safeguards, *safeguards],
                fallback=False,
                fallback_reason=None,
                role=f"glee_guarded_single_action_{family}",
                elapsed_s=round(self.clock() - started, 6),
                call_metadata={"planner": _metadata(planner_metadata)},
                selection_branch="deterministic-single-action",
                branch_receipts=branch_receipts,
            )
        except Exception as error:
            branch_receipts.append({"branch": "meta15-planner", "status": "failed", "role": planner_role, "timeout_s": self._timeout(planner_timeout), "elapsed_s": round(self.clock() - planner_started, 6), "error": f"{type(error).__name__}: {error}", "call_metadata": _metadata(planner_metadata)})
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"1.5-round planner failed: {type(error).__name__}: {error}")
        branch_receipts.append(
            {
                "branch": "meta15-planner",
                "status": "succeeded",
                "role": planner_role,
                "timeout_s": self._timeout(planner_timeout),
                "elapsed_s": round(self.clock() - planner_started, 6),
                "candidate_set": candidate_set.receipt(),
                "call_metadata": _metadata(planner_metadata),
            }
        )
        conditional_started = self.clock()
        try:
            conditional_receipt, conditional_surface = _meta15_candidate_response_surface(envelope=envelope, candidate_set=candidate_set, conditional_client=self.conditional_client, worker_turn=worker_turn)
        except Exception as error:
            branch_receipts.append({"branch": "conditional-twin", "status": "failed", "elapsed_s": round(self.clock() - conditional_started, 6), "error": f"{type(error).__name__}: {error}"})
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"conditional twin failed: {type(error).__name__}: {error}")
        branch_receipts.append({"branch": "conditional-twin", "status": "succeeded", "elapsed_s": round(self.clock() - conditional_started, 6), "receipt": conditional_receipt, "surface": conditional_surface})
        family_evidence: dict[str, object] | None = None
        try:
            family_evidence = _meta15_family_candidate_evidence(envelope, candidate_set)
            branch_receipts.append({"branch": "family-candidate-evidence", "status": "succeeded" if family_evidence is not None else "not-applicable", "evidence": family_evidence})
        except Exception as error:
            branch_receipts.append({"branch": "family-candidate-evidence", "status": "failed-advisory", "error": f"{type(error).__name__}: {error}"})
        self_mirror_surface: dict[str, object] | None = None
        self_mirror_admissibility: dict[str, object] | None = None
        if self.self_mirror_client is not None:
            mirror_started = self.clock()
            try:
                self_mirror_surface = self.self_mirror_client.forecast_candidates(game=envelope.game, turn_id=envelope.snapshot.turn_id, candidate_set=candidate_set)
                self_mirror_admissibility = _meta15_public_self_mirror_admissibility(candidate_set=candidate_set, conditional_surface=conditional_surface, family_evidence=family_evidence)
                branch_receipts.append({"branch": "public-self-mirror", "status": "succeeded", "elapsed_s": round(self.clock() - mirror_started, 6), "surface": self_mirror_surface, "admissibility": self_mirror_admissibility})
            except Exception as error:
                self_mirror_surface = None
                self_mirror_admissibility = None
                branch_receipts.append({"branch": "public-self-mirror", "status": "failed-advisory", "elapsed_s": round(self.clock() - mirror_started, 6), "error": f"{type(error).__name__}: {error}"})
        selector_remaining = budget_deadline - self.clock()
        if selector_remaining < self.minimum_selector_budget_s:
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"deadline budget too small for 1.5-round selector: {selector_remaining:.3f}s")
        try:
            selector_payload = build_selector_payload_v15(worker_payload=worker_turn, candidate_set=candidate_set, conditional_surface=conditional_surface, family_candidate_evidence=family_evidence, public_self_mirror_surface=self_mirror_surface, public_self_mirror_admissibility=self_mirror_admissibility)
        except Exception as error:
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"selector payload failed: {type(error).__name__}: {error}")
        selector_started = self.clock()
        selector_metadata: object = None
        selector_role = f"glee_meta_controller_v2_15_selector_{family}"
        selector_backend_receipt: Mapping[str, object] | None = None
        try:
            selector_request = build_selector_backend_request(
                candidate_set=candidate_set,
                selector_payload=selector_payload,
                fallback_candidate_id=candidate_set.candidates[0].action_sha256,
                timeout_s=selector_remaining,
            )
            selector_outcome = self.selector_backend.select(selector_request)
            if self.clock() > budget_deadline:
                raise TimeoutError("selector backend completed after the reserved finalization boundary")
            selected = selector_outcome.selected
            selector_metadata = selector_outcome.metadata
            selector_role = selector_outcome.role
            selector_backend_receipt = selector_outcome.receipt
        except Exception as error:
            branch_receipts.append({"branch": "meta15-selector", "status": "failed", "role": selector_role, "timeout_s": self._timeout(selector_remaining), "elapsed_s": round(self.clock() - selector_started, 6), "authority_contract": selector_payload["selector_authority_contract"], "backend": copy.deepcopy(dict(self.selector_backend.manifest_receipt)), "error": f"{type(error).__name__}: {error}", "call_metadata": _metadata(selector_metadata)})
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=None, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"1.5-round selector failed: {type(error).__name__}: {error}")
        branch_receipts.append(
            {
                "branch": "meta15-selector",
                "status": "succeeded",
                "role": selector_role,
                "timeout_s": self._timeout(selector_remaining),
                "elapsed_s": round(self.clock() - selector_started, 6),
                "authority_contract": selector_payload["selector_authority_contract"],
                "selected_candidate": selected.candidate.receipt(),
                "backend": copy.deepcopy(dict(self.selector_backend.manifest_receipt)),
                "backend_result": copy.deepcopy(dict(selector_backend_receipt or {})),
                "call_metadata": _metadata(selector_metadata),
            }
        )
        try:
            final_action, family_safeguards, family_policy_receipt = _meta15_apply_family_selector_policy(envelope=envelope, baseline=baseline, selected=selected, candidate_set=candidate_set, conditional_surface=conditional_surface, family_evidence=family_evidence)
        except Exception as error:
            branch_receipts.append({"branch": "family-selector-policy", "status": "failed", "contract": FAMILY_SELECTOR_POLICY_CONTRACT, "error": f"{type(error).__name__}: {error}"})
            self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=selected.candidate, submitted_action=baseline.action, branch_receipts=branch_receipts)
            return self._failed(baseline=baseline, started=started, branch_receipts=branch_receipts, reason=f"family selector policy failed: {type(error).__name__}: {error}")
        branch_receipts.append({"branch": "family-selector-policy", **family_policy_receipt})
        self._record_self_mirror_selection(envelope=envelope, surface=self_mirror_surface, candidate_set=candidate_set, selected_candidate=selected.candidate, submitted_action=final_action, branch_receipts=branch_receipts)
        return WorkerDecision(
            action=final_action,
            proposal=dict(selected.candidate.action),
            tetrad_update=None,
            tetrad_transport_issues=[],
            deterministic_safeguards=[*selected.candidate.safeguards, *family_safeguards],
            fallback=False,
            fallback_reason=None,
            role=selector_role,
            elapsed_s=round(self.clock() - started, 6),
            call_metadata={"planner": _metadata(planner_metadata), "selector": _metadata(selector_metadata)},
            selection_branch="meta15-selector",
            branch_receipts=branch_receipts,
        )
