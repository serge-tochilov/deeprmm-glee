"""Fail-closed transport contracts for the 4-stage pre-Terra decision loop."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel

from .glee_policy import normalize_action


TWO_CALL_CONTROLLER_CONTRACT = "glee-pre-terra-two-call-controller-v2.1"
PLANNER_PAYLOAD_CONTRACT = "glee-pre-terra-candidate-planner-payload-v1"
SELECTOR_PAYLOAD_CONTRACT = "glee-pre-terra-candidate-selector-payload-v2"
ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT = "glee-pre-terra-one-and-half-round-controller-v2.2"
INDEPENDENT_PLANNER_PAYLOAD_CONTRACT = "glee-pre-terra-independent-candidate-planner-payload-v1"
CONDITIONAL_SELECTOR_PAYLOAD_CONTRACT = "glee-pre-terra-conditional-candidate-selector-payload-v2"
PERSUASION_BUYER_CONTINUATION_SELECTOR_PAYLOAD_CONTRACT = "glee-persuasion-buyer-continuation-selector-payload-v1"
PERSUASION_BUYER_CONTINUATION_SELECTOR_AUTHORITY_CONTRACT = "glee-persuasion-buyer-continuation-selector-authority-v1"
POLICY_MARGINAL_FORECAST_CONTRACT = "glee-pre-terra-policy-marginal-forecast-v1"
CONDITIONAL_SURFACE_CONTRACT = "glee-pre-terra-candidate-response-surface-v1"
CONDITIONAL_MODEL_CARD_CONTRACT = "glee-pre-terra-conditional-model-card-v1"
SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT = "glee-symmetric-candidate-selector-authority-v1"
FAMILY_CANDIDATE_EVIDENCE_CONTRACT = "glee-family-candidate-decision-evidence-v1"
FAMILY_SELECTOR_POLICY_CONTRACT = "glee-family-selector-policy-v1"
NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT = "glee-negotiation-reject-counteroffer-bridge-v1"
PUBLIC_SELF_MIRROR_SURFACE_CONTRACT = "glee-public-self-mirror-candidate-surface-v1"
PUBLIC_SELF_MIRROR_AUTHORITY = "bounded-public-expectedness-selector-evidence-only"
PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT = "glee-public-self-mirror-economic-admissibility-v1"


_NEW_LEARNED_INPUT_KEYS = {
    "candidate_response_model_card",
    "conditional_opponent_response_surface",
    "family_candidate_decision_evidence",
    "policy_marginal_opponent_forecast",
    "public_self_mirror_candidate_surface",
    "public_self_mirror_economic_admissibility",
}
_NEW_LEARNED_INPUT_CONTRACTS = {
    POLICY_MARGINAL_FORECAST_CONTRACT,
    CONDITIONAL_SURFACE_CONTRACT,
    CONDITIONAL_MODEL_CARD_CONTRACT,
    "glee-pre-terra-action-conditional-release-v2",
    PUBLIC_SELF_MIRROR_SURFACE_CONTRACT,
    PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT,
}


_CONDITIONAL_MODEL_CARDS: dict[str, dict[str, object]] = {
    "bargaining": {
        "direct_response_target_count": 3613,
        "held_out_test_target_count": 855,
        "held_out_test_game_count": 355,
        "held_out_test_accuracy": 0.791813,
        "held_out_test_nll": 0.431513,
        "validation_selected_sequence_weight": 0.479,
        "selector_use": "Use material candidate-conditioned differences as calibration-weighted direct-response evidence after exact arithmetic and hard controls; do not privilege an analytic, deterministic, or planner-first candidate merely because of its provenance.",
    },
    "negotiation": {
        "direct_response_target_count": 1185,
        "held_out_test_target_count": 178,
        "held_out_test_game_count": 178,
        "held_out_test_accuracy": 0.887640,
        "held_out_test_nll": 0.361282,
        "validation_selected_sequence_weight": 0.164,
        "selector_use": "Treat the lower-support surface as calibration-discounted but substantive direct-response evidence; a material coherent difference may determine selection, while exact surplus arithmetic and hard controls remain binding.",
    },
    "persuasion": {
        "direct_response_target_count": 12359,
        "held_out_test_target_count": 1895,
        "held_out_test_game_count": 95,
        "held_out_test_accuracy": 0.925594,
        "held_out_test_nll": 0.196017,
        "validation_selected_sequence_weight": 0.938,
        "selector_use": "Use the surface as the strongest current family-level direct-response evidence; an advisory seller anchor has no selection presumption, while categorical controls and evidence-based longitudinal effects remain binding inputs.",
    },
}


_SELECTOR_FAMILY_OBJECTIVES: dict[str, dict[str, str]] = {
    "bargaining": {
        "objective": "maximize ranking-relevant expected own payoff over the complete remaining game",
        "immediate_value": "accepted own allocation multiplied by predicted acceptance probability, with rejection and walkaway valued through the supplied continuation and terminal mechanics",
        "continuation_evidence": "discounting, remaining rounds, observed response policy, cap-aware settlement mechanics, and current-game evidence",
    },
    "negotiation": {
        "objective": "maximize expected own surplus over the complete remaining game subject to exact role, reservation, and feasibility arithmetic",
        "immediate_value": "own feasible settlement surplus multiplied by predicted acceptance probability, with rejection and walkaway valued through remaining-round continuation",
        "continuation_evidence": "remaining rounds, observed concession policy, feasible surplus bounds, and current-game evidence",
    },
    "persuasion": {
        "objective": "maximize expected seller payoff over the complete remaining game while preserving every hard information and action boundary",
        "immediate_value": "current sale payoff multiplied by predicted purchase probability",
        "continuation_evidence": "remaining products, current-game buyer adaptation, revealed response-by-signal behavior, and an evidence-based estimate of future sale effects",
    },
}


CandidateGuard = Callable[[dict[str, Any]], tuple[dict[str, Any], Sequence[str]]]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return copy.deepcopy(dict(value))


def _new_learned_input_paths(value: object, *, path: str = "authenticated_turn") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        contract = value.get("contract")
        if isinstance(contract, str) and contract in _NEW_LEARNED_INPUT_CONTRACTS:
            paths.append(f"{path}.contract={contract}")
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in _NEW_LEARNED_INPUT_KEYS:
                paths.append(child_path)
            paths.extend(_new_learned_input_paths(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            paths.extend(_new_learned_input_paths(child, path=f"{path}[{index}]"))
    return paths


def _validate_independent_turn(value: Mapping[str, object]) -> dict[str, object]:
    base = _mapping(value, name="worker payload")
    leaked = _new_learned_input_paths(base)
    if leaked:
        raise ValueError(f"1.5-round planner input contains a new learned-model artifact: {leaked[0]}")
    return base


def negotiation_reject_counteroffer_frontier(value: Mapping[str, object]) -> bool:
    """Recognize a legal Negotiation decision schema that can carry a counteroffer."""
    if value.get("game_family") != "negotiation":
        return False
    valid_actions = value.get("valid_actions")
    if not isinstance(valid_actions, Mapping) or valid_actions.get("type") != "decision":
        return False
    fields = valid_actions.get("fields")
    if not isinstance(fields, Mapping) or "product_price" not in fields:
        return False
    state = value.get("game_state")
    if not isinstance(state, Mapping) or state.get("horizon_known") is not True:
        return True
    round_number = state.get("round")
    max_rounds = state.get("max_rounds")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        return False
    return round_number < max_rounds


def negotiation_reject_counteroffer_action(value: Mapping[str, object]) -> bool:
    """Recognize one finite compound rejection and counteroffer action."""
    price = value.get("product_price")
    return value.get("decision") == "RejectOffer" and not isinstance(price, bool) and isinstance(price, (int, float)) and math.isfinite(float(price)) and float(price) >= 0.0


def _direct_response_frontier(value: Mapping[str, object], *, name: str) -> tuple[str, str]:
    family = str(value.get("game_family") or "")
    valid_actions = value.get("valid_actions")
    if not isinstance(valid_actions, Mapping):
        raise ValueError(f"{name} has no valid action contract")
    action_type = str(valid_actions.get("type") or "")
    eligible = (family in {"bargaining", "negotiation"} and action_type == "offer") or (family == "negotiation" and action_type == "decision") or (family == "persuasion" and action_type in {"seller_message", "seller_recommendation"})
    if not eligible:
        raise ValueError(f"{name} is outside the direct-response conditional frontier")
    return family, action_type


def conditional_model_card(family: str) -> dict[str, object]:
    """Expose frozen family calibration without granting the adaptive diagnostics live authority."""
    calibration = _CONDITIONAL_MODEL_CARDS.get(family)
    if calibration is None:
        raise ValueError(f"unsupported conditional model family: {family!r}")
    return {
        "contract": CONDITIONAL_MODEL_CARD_CONTRACT,
        "release": "post-planner-conditional-twin-v3.0-post-planner-all-history-cut-20260816-r1",
        "family": family,
        "authority": "live-advisory-candidate-response-evidence",
        "prediction_target": "direct authenticated opponent response to one exact candidate",
        "not_predicted": ["later continuation", "counterfactual causal effect", "candidate submission latency", "payoff-optimal action"],
        "wording_boundary": "A bounded non-recoverable wording representation is an input. Forecast sensitivity is predictive evidence, not proof that unsubmitted wording would cause the difference.",
        "deployment_status": "active controlled online evaluation under deterministic guards and exact-candidate selector confinement",
        **copy.deepcopy(calibration),
    }


def selector_authority_contract(family: str) -> dict[str, object]:
    """Grant symmetric choice among guarded candidates without weakening hard controls."""
    objective = _SELECTOR_FAMILY_OBJECTIVES.get(family)
    if objective is None:
        raise ValueError(f"unsupported selector-authority family: {family!r}")
    return {
        "contract": SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT,
        "authority": "selector-controls-choice-among-immutable-guarded-candidates",
        "candidate_standing": "symmetric-after-hard-controls",
        "candidate_provenance": "index, planner order, purpose, deterministic origin, analytic origin, reference status, and advisory-anchor status confer no selection presumption",
        "deterministic_candidate": "execution fallback if staged selection fails; during a successful selector call it receives no prior weight",
        "objective": copy.deepcopy(objective),
        "comparison_rule": "compare every candidate in common expected-payoff units using its calibrated direct-response distribution and evidence-based continuation value",
        "lower_immediate_value_rule": "selecting a candidate with materially lower response-weighted immediate value requires a concrete continuation benefit, bounded in the same payoff units, that covers the immediate gap",
        "unsupported_reasons": ["generic trust preservation", "generic fairness", "generic stability", "familiarity with the deterministic or analytic action", "candidate purpose text", "unquantified future value"],
        "horizon_rule": "continuation effects shrink with the remaining horizon and are zero after the terminal response",
        "uncertainty_rule": "discount forecast differences according to family calibration and current evidence, but do not demote a material coherent difference to a tie-break merely because the alternative was not submitted",
        "hard_boundaries": ["legality", "categorical action authority", "bounded-authoritative action control", "exact role and payoff arithmetic", "required terminal action", "planner-candidate safeguards", "submission deadline"],
        "public_self_mirror": {
            "authority": PUBLIC_SELF_MIRROR_AUTHORITY,
            "use": "Within the explicitly marked economically near-equivalent admissible set only, prefer a strategically useful candidate that the public-information mirror assigns lower expectedness when the opponent-response and continuation evidence does not distinguish the candidates materially.",
            "outside_admissible_set": "Ignore public expectedness. It cannot justify an economic sacrifice, categorical reversal, terminal deviation, false reservation claim, account-specific route, or violation of any hard boundary.",
            "epistemic_status": "Predictive population evidence about what DeepRMM publicly appears likely to do; not a causal estimate of opponent confusion and not proof that surprise improves payoff.",
        },
    }


def _validate_marginal_forecast(value: Mapping[str, object]) -> dict[str, object]:
    forecast = _mapping(value, name="policy-marginal forecast")
    if forecast.get("contract") != POLICY_MARGINAL_FORECAST_CONTRACT:
        raise ValueError("policy-marginal forecast has the wrong contract")
    if forecast.get("frontier") != "authenticated-prefix-with-one-unobserved-self-action-bridge":
        raise ValueError("policy-marginal forecast has the wrong causal frontier")
    if forecast.get("authority") != "advisory-prior-only":
        raise ValueError("policy-marginal forecast must remain an advisory prior")
    labels = forecast.get("labels")
    probabilities = forecast.get("response_probabilities")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("policy-marginal response labels are invalid")
    family = str(forecast.get("family") or "")
    expected_labels = ["buy", "pass"] if family == "persuasion" else ["accept", "reject", "walkaway"] if family in {"bargaining", "negotiation"} else []
    if labels != expected_labels:
        raise ValueError("policy-marginal response labels do not match its family")
    if not isinstance(probabilities, list) or len(probabilities) != len(labels):
        raise ValueError("policy-marginal response probabilities are invalid")
    numeric = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in numeric) or abs(sum(numeric) - 1.0) > 1e-5:
        raise ValueError("policy-marginal response probabilities are not normalized")
    forecast["response_probabilities"] = numeric
    return forecast


def policy_marginal_forecast_from_shadow(value: Mapping[str, object]) -> dict[str, object]:
    """Re-contract one v2 missing-bridge shadow prediction as the stage-a advisory prior."""
    raw = _mapping(value, name="missing-bridge shadow prediction")
    family = str(raw.get("family") or "")
    expected_labels = ["buy", "pass"] if family == "persuasion" else ["accept", "reject", "walkaway"] if family in {"bargaining", "negotiation"} else []
    if raw.get("forecast_frontier") != "authenticated-visible-prefix-before-terra" or raw.get("causal_bridge_event_count") != 1 or raw.get("target_kind") != "response" or not expected_labels:
        raise ValueError("shadow prediction is not a one-missing-self-action pre-Terra forecast")
    labels = raw.get("labels")
    probabilities = raw.get("action_probabilities")
    if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels) or not isinstance(probabilities, list) or len(labels) != len(probabilities):
        raise ValueError("shadow prediction has no aligned action distribution")
    probability_by_label = {str(label): float(probability) for label, probability in zip(labels, probabilities, strict=True)}
    selected = [probability_by_label.get(label, 0.0) for label in expected_labels]
    total = sum(selected)
    if total <= 0.0:
        raise ValueError("shadow prediction assigns no mass to direct opponent responses")
    forecast = {
        "contract": POLICY_MARGINAL_FORECAST_CONTRACT,
        "frontier": "authenticated-prefix-with-one-unobserved-self-action-bridge",
        "authority": "advisory-prior-only",
        "family": family,
        "labels": expected_labels,
        "response_probabilities": [value / total for value in selected],
        "source_candidate_id": raw.get("candidate_id"),
        "source_manifest_sha256": raw.get("candidate_manifest_sha256"),
        "source_prediction_sha256": _sha(raw),
        "interpretation": "Opponent response marginalized over the DeepRMM-01 actions represented by the fitted policy; not conditioned on the current candidate.",
    }
    return _validate_marginal_forecast(forecast)


@dataclass(frozen=True)
class PlannedCandidate:
    index: int
    action: Mapping[str, Any]
    purpose: str
    action_sha256: str
    safeguards: tuple[str, ...]

    def receipt(self) -> dict[str, object]:
        return {"candidate_index": self.index, "action": copy.deepcopy(dict(self.action)), "purpose": self.purpose, "action_sha256": self.action_sha256, "planner_candidate_safeguards": list(self.safeguards)}


@dataclass(frozen=True)
class FrozenCandidateSet:
    family: str
    action_type: str
    candidates: tuple[PlannedCandidate, ...]
    candidate_set_sha256: str

    def receipt(self) -> dict[str, object]:
        return {"family": self.family, "action_type": self.action_type, "candidates": [candidate.receipt() for candidate in self.candidates], "candidate_set_sha256": self.candidate_set_sha256}


class GuardedSingleAction(RuntimeError):
    """Signal that distinct proposals intentionally collapsed to one action under hard guards."""

    def __init__(self, candidate_set: FrozenCandidateSet, *, pre_guard_candidate_count: int) -> None:
        message = "the required seed and planner candidates normalized to one authoritative action" if pre_guard_candidate_count == 1 else "hard guards collapsed distinct planner candidates to one authoritative action"
        super().__init__(message)
        self.candidate_set = candidate_set
        interpretation = "The required deterministic seed and every planner candidate normalized to the same guarded outward move. This is a deliberate seed-locked single-action state rather than planner failure." if pre_guard_candidate_count == 1 else "The planner supplied distinct normalized actions, but hard legality or policy guards mapped every surviving action to the same outward move. This is a deliberate single-action state rather than planner failure."
        self.receipt = {
            "status": "guarded-single-action",
            "pre_guard_candidate_count": pre_guard_candidate_count,
            "candidate_set": candidate_set.receipt(),
            "interpretation": interpretation,
        }


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: PlannedCandidate


def build_planner_payload(*, worker_payload: Mapping[str, object], policy_marginal_forecast: Mapping[str, object], deterministic_candidates: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    """Build the first Terra call without granting it final-selection authority."""
    base = _mapping(worker_payload, name="worker payload")
    marginal = _validate_marginal_forecast(policy_marginal_forecast)
    family, action_type = _direct_response_frontier(base, name="worker payload")
    if marginal.get("family") != family:
        raise ValueError("policy-marginal forecast family does not match the authenticated turn")
    seeds = [_mapping(value, name="deterministic candidate") for value in deterministic_candidates]
    return {
        "contract": PLANNER_PAYLOAD_CONTRACT,
        "controller": TWO_CALL_CONTROLLER_CONTRACT,
        "stage": "candidate-planning",
        "authenticated_turn": base,
        "conditional_frontier": {"family": family, "action_type": action_type, "target": "direct opponent response after the candidate"},
        "policy_marginal_opponent_forecast": marginal,
        "candidate_response_model_card": conditional_model_card(family),
        "deterministic_candidate_seeds": seeds,
        "output_boundary": "Return 2 to 5 exact legal candidate actions and a short purpose for each. Do not select, rank, recommend, or mark a winner.",
        "authenticated_turn_sha256": _sha(base),
    }


def build_planner_payload_v15(*, worker_payload: Mapping[str, object], deterministic_candidates: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    """Build the learned-forecast-independent planner call for the 1.5-round arm."""
    base = _validate_independent_turn(worker_payload)
    family, action_type = _direct_response_frontier(base, name="worker payload")
    seeds = [_mapping(value, name="deterministic candidate") for value in deterministic_candidates]
    return {
        "contract": INDEPENDENT_PLANNER_PAYLOAD_CONTRACT,
        "controller": ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT,
        "stage": "independent-candidate-planning",
        "authenticated_turn": base,
        "conditional_frontier": {"family": family, "action_type": action_type, "target": "direct opponent response after the candidate"},
        "deterministic_candidate_seeds": seeds,
        "learned_model_boundary": {
            "new_learned_model_inputs": [],
            "policy_marginal_forecast_visible": False,
            "conditional_forecast_visible": False,
            "conditional_model_card_visible": False,
            "candidate_commitment_precedes": CONDITIONAL_SURFACE_CONTRACT,
        },
        "output_boundary": "Return 2 to 5 exact legal candidate actions and a short purpose for each. Do not select, rank, recommend, or mark a winner.",
        "authenticated_turn_sha256": _sha(base),
    }


def freeze_planner_candidates(*, game: dict[str, Any], parsed: BaseModel, guard: CandidateGuard | None = None, required_candidates: Sequence[Mapping[str, object]] = ()) -> FrozenCandidateSet:
    """Normalize, optionally guard, deduplicate, and freeze required seeds plus the planner's exact actions."""
    family, action_type = _direct_response_frontier(game, name="game")
    raw_candidates = getattr(parsed, "candidates", None)
    if not isinstance(raw_candidates, list) or not 2 <= len(raw_candidates) <= 5:
        raise ValueError("planner must return 2 to 5 candidates")
    required = [_mapping(value, name="required candidate") for value in required_candidates]
    if len(required) > 5:
        raise ValueError("at most 5 required candidates may be frozen")
    frozen: list[PlannedCandidate] = []
    seen: set[str] = set()
    seen_before_guard: set[str] = set()

    def append_candidate(action_value: object, purpose_value: object, *, required_seed: bool) -> None:
        purpose = str(purpose_value or "").strip()
        if not isinstance(action_value, (BaseModel, Mapping)) or not purpose:
            raise ValueError("planner candidate is missing its exact action or purpose")
        action = normalize_action(game, action_value if isinstance(action_value, BaseModel) else dict(action_value))
        pre_guard_digest = _sha(action)
        duplicate_before_guard = pre_guard_digest in seen_before_guard
        seen_before_guard.add(pre_guard_digest)
        safeguards: tuple[str, ...] = ()
        if guard is not None:
            guarded, raw_safeguards = guard(copy.deepcopy(action))
            action = normalize_action(game, guarded)
            safeguards = tuple(str(value) for value in raw_safeguards)
        digest = _sha(action)
        if digest in seen:
            if required_seed and duplicate_before_guard:
                raise ValueError("required candidates are not distinct after normalization and guarding")
            return
        seen.add(digest)
        frozen.append(PlannedCandidate(index=len(frozen), action=MappingProxyType(copy.deepcopy(action)), purpose=purpose, action_sha256=digest, safeguards=safeguards))

    for raw in required:
        append_candidate(raw.get("action"), raw.get("purpose"), required_seed=True)
    for raw in raw_candidates:
        if len(frozen) >= 5:
            break
        append_candidate(getattr(raw, "action", None), getattr(raw, "purpose", ""), required_seed=False)
    if len(frozen) < 2:
        if len(frozen) == 1 and guard is not None and (len(seen_before_guard) >= 2 or bool(required)):
            receipt = {"family": family, "action_type": action_type, "candidates": [candidate.receipt() for candidate in frozen]}
            candidate_set = FrozenCandidateSet(family=family, action_type=action_type, candidates=tuple(frozen), candidate_set_sha256=_sha(receipt))
            raise GuardedSingleAction(candidate_set, pre_guard_candidate_count=len(seen_before_guard))
        raise ValueError("planner produced fewer than 2 unique candidates after normalization and guarding")
    receipt = {"family": family, "action_type": action_type, "candidates": [candidate.receipt() for candidate in frozen]}
    return FrozenCandidateSet(family=family, action_type=action_type, candidates=tuple(frozen), candidate_set_sha256=_sha(receipt))


def freeze_fixed_candidates(*, game: dict[str, Any], candidate_specs: Sequence[Mapping[str, object]], guard: CandidateGuard | None = None) -> FrozenCandidateSet:
    """Normalize and freeze one caller-complete candidate set without a planner call."""
    family = str(game.get("game_family") or "")
    valid_actions = game.get("valid_actions")
    action_type = str(valid_actions.get("type") or game.get("phase") or "") if isinstance(valid_actions, Mapping) else ""
    if not family or not action_type or not 2 <= len(candidate_specs) <= 5:
        raise ValueError("fixed candidate set requires one family, action type, and 2 to 5 candidates")
    frozen: list[PlannedCandidate] = []
    seen: set[str] = set()
    for raw in candidate_specs:
        spec = _mapping(raw, name="fixed candidate")
        purpose = str(spec.get("purpose") or "").strip()
        action_value = spec.get("action")
        if not purpose or not isinstance(action_value, Mapping):
            raise ValueError("fixed candidate is missing its exact action or purpose")
        action = normalize_action(game, dict(action_value))
        safeguards: tuple[str, ...] = ()
        if guard is not None:
            guarded, raw_safeguards = guard(copy.deepcopy(action))
            action = normalize_action(game, guarded)
            safeguards = tuple(str(value) for value in raw_safeguards)
        digest = _sha(action)
        if digest in seen:
            raise ValueError("fixed candidates are not distinct after normalization and guarding")
        seen.add(digest)
        frozen.append(PlannedCandidate(index=len(frozen), action=MappingProxyType(copy.deepcopy(action)), purpose=purpose, action_sha256=digest, safeguards=safeguards))
    receipt = {"family": family, "action_type": action_type, "candidates": [candidate.receipt() for candidate in frozen]}
    return FrozenCandidateSet(family=family, action_type=action_type, candidates=tuple(frozen), candidate_set_sha256=_sha(receipt))


def build_persuasion_buyer_continuation_selector_payload(*, worker_payload: Mapping[str, object], candidate_set: FrozenCandidateSet, continuation_surface: Mapping[str, object], public_self_mirror_surface: Mapping[str, object] | None = None, public_self_mirror_admissibility: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build the selector-only buyer path over an exhaustive immutable buy/pass set."""
    base = _validate_independent_turn(worker_payload)
    valid_actions = base.get("valid_actions")
    if base.get("game_family") != "persuasion" or not isinstance(valid_actions, Mapping) or valid_actions.get("type") != "buyer_decision":
        raise ValueError("buyer-continuation selector requires a Persuasion buyer decision")
    if candidate_set.family != "persuasion" or candidate_set.action_type != "buyer_decision" or {dict(candidate.action).get("decision") for candidate in candidate_set.candidates} != {"yes", "no"} or len(candidate_set.candidates) != 2:
        raise ValueError("buyer-continuation selector requires the exhaustive buy/pass candidate set")
    surface = _mapping(continuation_surface, name="buyer-continuation surface")
    if surface.get("contract") != "glee-persuasion-buyer-continuation-live-v1" or surface.get("authority") != "advisory-next-seller-signal-evidence-only" or surface.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("buyer-continuation surface has the wrong identity or authority")
    rows = surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("buyer-continuation surface does not cover the candidate set")
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        forecast = row.get("forecast") if isinstance(row, Mapping) else None
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(forecast, Mapping):
            raise ValueError("buyer-continuation surface lost candidate alignment")
        if forecast.get("labels") != ["signal_positive", "signal_negative", "signal_unknown"] or forecast.get("authority") != "advisory-next-seller-signal-evidence-only":
            raise ValueError("buyer-continuation candidate forecast is malformed")
        probabilities = forecast.get("response_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in probabilities) or not math.isclose(sum(float(value) for value in probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("buyer-continuation candidate probabilities are invalid")
    facts = base.get("persuasion_decision_facts")
    authority = facts.get("authority") if isinstance(facts, Mapping) else None
    if not isinstance(authority, Mapping) or authority.get("action_authority") != "advisory-only" or authority.get("selected_action") is not None:
        raise ValueError("buyer-continuation selector cannot override a hard Persuasion buyer control")
    payload = {
        "contract": PERSUASION_BUYER_CONTINUATION_SELECTOR_PAYLOAD_CONTRACT,
        "stage": "exhaustive-buyer-candidate-selection",
        "authenticated_turn": base,
        "candidate_set": candidate_set.receipt(),
        "conditional_opponent_response_surface": surface,
        "buyer_continuation_model_card": {
            "release": surface.get("release_id"),
            "authority": "live-advisory-next-seller-signal-evidence",
            "prediction_target": "opponent seller's next-round positive, negative, or unknown signal after the exact candidate buyer action",
            "not_predicted": ["current hidden quality", "realized current buyer payoff", "later seller signals", "causal effect of an unsubmitted buyer action", "payoff-optimal buyer action"],
            "retrospective_test_game_macro_nll": 0.6318,
            "markov_baseline_test_game_macro_nll": 0.7266,
            "paired_bootstrap_arm_minus_baseline_95_percent_interval": [-0.1467, -0.0393],
            "sparse_cell_warning": "The preceding-negative-signal plus buy cell had only 135 examples in the frozen full corpus; use the per-candidate support warning.",
        },
        "selector_authority_contract": {
            "contract": PERSUASION_BUYER_CONTINUATION_SELECTOR_AUTHORITY_CONTRACT,
            "authority": "selector-controls-choice-between-exhaustive-guarded-buy-and-pass-candidates",
            "objective": "maximize the buyer's expected complete-game payoff under visible utilities and bounded evidence",
            "current_payoff_rule": "Immediate expected buy surplus from seller reliability and visible utilities remains primary; pass has zero current payoff.",
            "continuation_rule": "Use the candidate-conditioned next-seller-signal distribution only to estimate bounded future information and payoff. It predicts one next seller signal, not its truth, the future quality, or the complete continuation.",
            "sacrifice_rule": "Choose lower immediate expected payoff only when a concrete bounded future-payoff advantage covers the current gap after uncertainty; generic trust, retaliation, exploration, or model sensitivity does not cover it.",
            "causality_rule": "The forecast is observed-policy predictive evidence, not an identified causal effect of buy versus pass.",
            "hard_boundaries": ["legality", "categorical or bounded-authoritative buyer control", "visible utility arithmetic", "terminal decision", "candidate safeguards", "submission deadline"],
        },
        "learned_model_boundary": {"candidate_set_is_exhaustive": True, "candidate_actions_committed_before_forecast": True, "selector_may_only_return_candidate_index": True},
        "output_boundary": "Return only candidate_index. Select one supplied candidate exactly; do not reproduce, edit, merge, or invent an action.",
        "authenticated_turn_sha256": _sha(base),
    }
    if (public_self_mirror_surface is None) != (public_self_mirror_admissibility is None):
        raise ValueError("buyer selector requires both public self-mirror artifacts or neither")
    if public_self_mirror_surface is not None and public_self_mirror_admissibility is not None:
        payload["public_self_mirror_candidate_surface"] = validate_public_self_mirror_surface(candidate_set=candidate_set, value=public_self_mirror_surface)
        payload["public_self_mirror_economic_admissibility"] = _validate_public_self_mirror_admissibility(candidate_set=candidate_set, value=public_self_mirror_admissibility)
        payload["selector_authority_contract"]["public_self_mirror"] = selector_authority_contract("persuasion")["public_self_mirror"]
        payload["learned_model_boundary"]["public_self_mirror_not_visible_to_candidate_generation"] = True
    return payload


def build_conditional_surface(*, candidate_set: FrozenCandidateSet, forecasts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Align one action-conditioned forecast with every immutable candidate."""
    if len(forecasts) != len(candidate_set.candidates):
        raise ValueError("conditional forecasts do not cover the candidate set")
    expected_labels = ["buy", "pass"] if candidate_set.family == "persuasion" else ["accept", "reject", "walkaway"]
    rows: list[dict[str, object]] = []
    for candidate, raw in zip(candidate_set.candidates, forecasts, strict=True):
        forecast = _mapping(raw, name="conditional forecast")
        terminal_self_action = candidate_set.family == "negotiation" and candidate_set.action_type == "decision" and forecast.get("target_status") == "terminal-self-action"
        if terminal_self_action:
            if forecast.get("authority") != "exact-terminal-value-no-opponent-response" or forecast.get("labels") != [] or forecast.get("response_probabilities") != []:
                raise ValueError("terminal Negotiation candidate forecast is malformed")
            rows.append({"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": forecast})
            continue
        if forecast.get("authority") != "prospective-shadow-only":
            raise ValueError("conditional forecast exceeded its shadow authority")
        labels = forecast.get("labels")
        probabilities = forecast.get("response_probabilities")
        if labels != expected_labels or not isinstance(probabilities, list) or len(labels) != len(probabilities):
            raise ValueError("conditional response distribution is malformed")
        numeric = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in numeric) or abs(sum(numeric) - 1.0) > 1e-5:
            raise ValueError("conditional response distribution is not normalized")
        forecast["response_probabilities"] = numeric
        rows.append({"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": forecast})
    return {
        "contract": CONDITIONAL_SURFACE_CONTRACT,
        "candidate_set_sha256": candidate_set.candidate_set_sha256,
        "rows": rows,
        "authority": "advisory-candidate-response-evidence-only",
    }


def build_family_candidate_evidence(*, candidate_set: FrozenCandidateSet, evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Align established family-advisor evidence with the immutable candidate set."""
    if len(evidence) != len(candidate_set.candidates):
        raise ValueError("family candidate evidence does not cover the candidate set")
    rows: list[dict[str, object]] = []
    for candidate, raw in zip(candidate_set.candidates, evidence, strict=True):
        value = _mapping(raw, name="family candidate evidence")
        try:
            _canonical(value)
        except (TypeError, ValueError) as error:
            raise ValueError("family candidate evidence is not JSON-serializable") from error
        rows.append({"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "evidence": value})
    return {
        "contract": FAMILY_CANDIDATE_EVIDENCE_CONTRACT,
        "family": candidate_set.family,
        "candidate_set_sha256": candidate_set.candidate_set_sha256,
        "rows": rows,
        "authority": "established-family-advisor-candidate-evidence",
    }


def _validate_family_candidate_evidence(*, candidate_set: FrozenCandidateSet, value: Mapping[str, object]) -> dict[str, object]:
    surface = _mapping(value, name="family candidate evidence surface")
    if surface.get("contract") != FAMILY_CANDIDATE_EVIDENCE_CONTRACT or surface.get("family") != candidate_set.family or surface.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("family candidate evidence does not match the immutable candidate set")
    rows = surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("family candidate evidence coverage changed")
    evidence: list[dict[str, object]] = []
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(row.get("evidence"), Mapping):
            raise ValueError("family candidate evidence alignment changed")
        evidence.append(dict(row["evidence"]))
    rebuilt = build_family_candidate_evidence(candidate_set=candidate_set, evidence=evidence)
    if rebuilt != surface:
        raise ValueError("family candidate evidence content changed after validation")
    return surface


def validate_public_self_mirror_surface(*, candidate_set: FrozenCandidateSet, value: Mapping[str, object]) -> dict[str, object]:
    """Validate one prospective public-expectedness surface without reconstructing its model internals."""
    surface = _mapping(value, name="public self-mirror surface")
    if surface.get("contract") != PUBLIC_SELF_MIRROR_SURFACE_CONTRACT or surface.get("authority") != PUBLIC_SELF_MIRROR_AUTHORITY or surface.get("candidate_set_sha256") != candidate_set.candidate_set_sha256 or surface.get("family") != candidate_set.family:
        raise ValueError("public self-mirror surface has the wrong identity or authority")
    rows = surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("public self-mirror surface does not cover the candidate set")
    ordering: list[tuple[float, int, float]] = []
    relative_sum = 0.0
    for candidate, raw in zip(candidate_set.candidates, rows, strict=True):
        forecast = raw.get("forecast") if isinstance(raw, Mapping) else None
        if not isinstance(raw, Mapping) or raw.get("candidate_index") != candidate.index or raw.get("action_sha256") != candidate.action_sha256 or not isinstance(forecast, Mapping):
            raise ValueError("public self-mirror surface lost immutable candidate alignment")
        numeric_keys = ("ensemble_log_expectedness", "relative_expectedness", "expectedness_percentile", "component_log_score_stddev")
        if any(isinstance(forecast.get(key), bool) or not isinstance(forecast.get(key), (int, float)) or not math.isfinite(float(forecast[key])) for key in numeric_keys):
            raise ValueError("public self-mirror surface has a non-finite score")
        relative = float(forecast["relative_expectedness"])
        percentile = float(forecast["expectedness_percentile"])
        rank = forecast.get("expectedness_rank")
        components = forecast.get("component_log_expectedness")
        if not 0.0 <= relative <= 1.0 or not 0.0 <= percentile <= 1.0 or isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= len(rows) or not isinstance(components, Mapping) or len(components) < 2:
            raise ValueError("public self-mirror relative score, rank, or component support is invalid")
        if any(isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(float(component)) for component in components.values()):
            raise ValueError("public self-mirror component score is invalid")
        if forecast.get("population_prediction") is not True or forecast.get("account_prediction") is not None or forecast.get("message_wording_scored") is not False:
            raise ValueError("public self-mirror exceeded its initial population-only boundary")
        ordering.append((float(forecast["ensemble_log_expectedness"]), rank, percentile))
        relative_sum += relative
    if not math.isclose(relative_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("public self-mirror relative expectedness is not normalized")
    for left_index, (left_score, left_rank, left_percentile) in enumerate(ordering):
        for right_score, right_rank, right_percentile in ordering[left_index + 1 :]:
            if left_score == right_score and (left_rank != right_rank or left_percentile != right_percentile):
                raise ValueError("public self-mirror tied scores depend on candidate position")
            if left_score > right_score and not (left_rank < right_rank and left_percentile > right_percentile):
                raise ValueError("public self-mirror expectedness ordering is inconsistent")
            if left_score < right_score and not (left_rank > right_rank and left_percentile < right_percentile):
                raise ValueError("public self-mirror expectedness ordering is inconsistent")
    return surface


def build_public_self_mirror_admissibility(*, candidate_set: FrozenCandidateSet, utility_values: Sequence[float | None], absolute_regret_cap: float, value_units: str, evidence_source: str) -> dict[str, object]:
    """Freeze the economic set inside which public expectedness may affect selection."""
    if len(utility_values) != len(candidate_set.candidates) or not value_units.strip() or not evidence_source.strip():
        raise ValueError("public self-mirror admissibility lacks aligned utility evidence")
    if not math.isfinite(absolute_regret_cap) or absolute_regret_cap < 0.0:
        raise ValueError("public self-mirror regret cap must be finite and nonnegative")
    normalized = [float(value) if value is not None and math.isfinite(float(value)) else None for value in utility_values]
    complete = all(value is not None for value in normalized)
    best = max(value for value in normalized if value is not None) if complete and normalized else None
    rows: list[dict[str, object]] = []
    for candidate, value in zip(candidate_set.candidates, normalized, strict=True):
        regret = best - value if best is not None and value is not None else None
        rows.append(
            {
                "candidate_index": candidate.index,
                "action_sha256": candidate.action_sha256,
                "utility_value": round(value, 8) if value is not None else None,
                "regret_from_best": round(regret, 8) if regret is not None else None,
                "eligible_for_public_expectedness": regret is not None and regret <= absolute_regret_cap + 1e-12,
            }
        )
    eligible_count = sum(row["eligible_for_public_expectedness"] is True for row in rows)
    enabled = complete and eligible_count >= 2
    if not enabled:
        for row in rows:
            row["eligible_for_public_expectedness"] = False
    return {
        "contract": PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT,
        "candidate_set_sha256": candidate_set.candidate_set_sha256,
        "family": candidate_set.family,
        "enabled": enabled,
        "absolute_regret_cap": absolute_regret_cap,
        "value_units": value_units,
        "evidence_source": evidence_source,
        "rows": rows,
        "reason": "at least 2 candidates are economically near-equivalent" if enabled else "public expectedness has no directional authority because a complete near-equivalent set is absent",
    }


def _validate_public_self_mirror_admissibility(*, candidate_set: FrozenCandidateSet, value: Mapping[str, object]) -> dict[str, object]:
    admissibility = _mapping(value, name="public self-mirror admissibility")
    if admissibility.get("contract") != PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT or admissibility.get("candidate_set_sha256") != candidate_set.candidate_set_sha256 or admissibility.get("family") != candidate_set.family:
        raise ValueError("public self-mirror admissibility has the wrong identity")
    rows = admissibility.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("public self-mirror admissibility does not cover the candidate set")
    eligible = 0
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(row.get("eligible_for_public_expectedness"), bool):
            raise ValueError("public self-mirror admissibility lost candidate alignment")
        eligible += int(row["eligible_for_public_expectedness"] is True)
    if admissibility.get("enabled") is True and eligible < 2:
        raise ValueError("public self-mirror authority requires at least 2 economically admissible candidates")
    if admissibility.get("enabled") is not True and eligible:
        raise ValueError("disabled public self-mirror admissibility cannot expose eligible candidates")
    return admissibility


def build_selector_payload(*, worker_payload: Mapping[str, object], policy_marginal_forecast: Mapping[str, object], candidate_set: FrozenCandidateSet, conditional_surface: Mapping[str, object]) -> dict[str, object]:
    """Build the second Terra call while making candidate mutation structurally impossible."""
    base = _mapping(worker_payload, name="worker payload")
    marginal = _validate_marginal_forecast(policy_marginal_forecast)
    family, action_type = _direct_response_frontier(base, name="worker payload")
    if (family, action_type) != (candidate_set.family, candidate_set.action_type):
        raise ValueError("candidate set does not match the authenticated turn frontier")
    if marginal.get("family") != family:
        raise ValueError("policy-marginal forecast family does not match the authenticated turn")
    surface = _mapping(conditional_surface, name="conditional surface")
    if surface.get("contract") != CONDITIONAL_SURFACE_CONTRACT or surface.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("conditional surface does not match the immutable candidate set")
    rows = surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("conditional surface coverage changed")
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256:
            raise ValueError("conditional surface candidate alignment changed")
    rebuilt = build_conditional_surface(candidate_set=candidate_set, forecasts=[dict(row["forecast"]) for row in rows if isinstance(row, Mapping) and isinstance(row.get("forecast"), Mapping)])
    if rebuilt != surface:
        raise ValueError("conditional surface content changed after validation")
    return {
        "contract": SELECTOR_PAYLOAD_CONTRACT,
        "controller": TWO_CALL_CONTROLLER_CONTRACT,
        "stage": "candidate-selection",
        "authenticated_turn": base,
        "conditional_frontier": {"family": family, "action_type": action_type, "target": "direct opponent response after the candidate"},
        "policy_marginal_opponent_forecast": marginal,
        "candidate_response_model_card": conditional_model_card(family),
        "selector_authority_contract": selector_authority_contract(family),
        "candidate_set": candidate_set.receipt(),
        "conditional_opponent_response_surface": surface,
        "output_boundary": "Return only candidate_index. Select one supplied candidate exactly; do not reproduce, edit, merge, or invent an action.",
        "authenticated_turn_sha256": _sha(base),
    }


def build_selector_payload_v15(*, worker_payload: Mapping[str, object], candidate_set: FrozenCandidateSet, conditional_surface: Mapping[str, object], family_candidate_evidence: Mapping[str, object] | None = None, public_self_mirror_surface: Mapping[str, object] | None = None, public_self_mirror_admissibility: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build the 1.5-round selector without reintroducing the policy-marginal prior."""
    base = _validate_independent_turn(worker_payload)
    family, action_type = _direct_response_frontier(base, name="worker payload")
    if (family, action_type) != (candidate_set.family, candidate_set.action_type):
        raise ValueError("candidate set does not match the authenticated turn frontier")
    surface = _mapping(conditional_surface, name="conditional surface")
    if surface.get("contract") != CONDITIONAL_SURFACE_CONTRACT or surface.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("conditional surface does not match the immutable candidate set")
    rows = surface.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError("conditional surface coverage changed")
    for candidate, row in zip(candidate_set.candidates, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256:
            raise ValueError("conditional surface candidate alignment changed")
    rebuilt = build_conditional_surface(candidate_set=candidate_set, forecasts=[dict(row["forecast"]) for row in rows if isinstance(row, Mapping) and isinstance(row.get("forecast"), Mapping)])
    if rebuilt != surface:
        raise ValueError("conditional surface content changed after validation")
    payload = {
        "contract": CONDITIONAL_SELECTOR_PAYLOAD_CONTRACT,
        "controller": ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT,
        "stage": "conditional-candidate-selection",
        "authenticated_turn": base,
        "conditional_frontier": {"family": family, "action_type": action_type, "target": "direct opponent response after the candidate"},
        "candidate_response_model_card": conditional_model_card(family),
        "selector_authority_contract": selector_authority_contract(family),
        "candidate_set": candidate_set.receipt(),
        "conditional_opponent_response_surface": surface,
        "learned_model_boundary": {
            "policy_marginal_forecast_visible": False,
            "candidate_set_committed_before_conditional_forecast": True,
            "selector_may_only_return_candidate_index": True,
        },
        "output_boundary": "Return only candidate_index. Select one supplied candidate exactly; do not reproduce, edit, merge, or invent an action.",
        "authenticated_turn_sha256": _sha(base),
    }
    if family_candidate_evidence is not None:
        payload["family_candidate_decision_evidence"] = _validate_family_candidate_evidence(candidate_set=candidate_set, value=family_candidate_evidence)
    if (public_self_mirror_surface is None) != (public_self_mirror_admissibility is None):
        raise ValueError("selector requires both public self-mirror artifacts or neither")
    if public_self_mirror_surface is not None and public_self_mirror_admissibility is not None:
        payload["public_self_mirror_candidate_surface"] = validate_public_self_mirror_surface(candidate_set=candidate_set, value=public_self_mirror_surface)
        payload["public_self_mirror_economic_admissibility"] = _validate_public_self_mirror_admissibility(candidate_set=candidate_set, value=public_self_mirror_admissibility)
        payload["learned_model_boundary"]["public_self_mirror_not_visible_to_candidate_generation"] = True
    return payload


def select_frozen_candidate(*, candidate_set: FrozenCandidateSet, parsed: BaseModel) -> SelectedCandidate:
    """Resolve the selector index to the exact frozen action."""
    index = getattr(parsed, "candidate_index", None)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidate_set.candidates):
        raise ValueError("selector index does not identify a supplied candidate")
    return SelectedCandidate(candidate=candidate_set.candidates[index])
