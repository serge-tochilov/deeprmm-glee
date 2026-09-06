"""Frozen forecasts, rating estimates, and longitudinal live authority for GLEE Persuasion v2.8."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glee_advisor_contracts import PERSUASION_ADVISOR_MODEL_VERSION as MODEL_VERSION, PERSUASION_LIVE_ENGINE_VERSION as LIVE_ENGINE_VERSION
from .glee_persuasion_policy_v2_4 import PersuasionSellerPolicyConfig
from .glee_persuasion_policy_v2_8 import canonical_seller_message, seller_longitudinal_control
from .glee_persuasion_rating_v2_3 import PersuasionRatingSurrogate, fit_persuasion_rating_surrogate, persuasion_rating_decision_surface
from .glee_persuasion_twin_v2 import BuyerResponseModel, PersuasionContext, PersuasionGameEvidence, PersuasionModelConfig, SellerReliabilityModel, classify_persuasion_signal, context_from_game, extract_persuasion_game, load_persuasion_archive, model_receipt, object_sha256
from .immutable_pack import file_sha256, load_ordered_json_objects, seal_json_objects


SCHEMA_VERSION = 1
SEED_KIND = "glee-persuasion-live-v2-seed"
SEED_REFERENCE_KIND = "glee-persuasion-live-v2-seed-reference"
JOURNAL_KIND = "glee-persuasion-live-v2-completion"
PERSUASION_LIVE_POLICY_CONTRACT = "glee-persuasion-live-policy-v1"
PERSUASION_HOT_PARAMETERS = (
    "authority_min_population_purchased_rows",
    "buyer_buy_margin_ratio",
    "buyer_pass_margin_ratio",
    "buyer_min_current_revealed_purchases",
    "seller_deception_advantage_margin",
    "seller_no_response_min_passes",
    "seller_no_response_max_smoothed_buy_rate",
)
PERSUASION_HOT_FEATURES = ("buyer_channel_scope", "buyer_evidence_boundary", "buyer_pass_scope")
PERSUASION_OPTIONAL_HOT_FEATURES = ("seller_no_response_routing",)
BUYER_INDIFFERENCE_BAND_RATIO = 0.05
BUYER_MATERIAL_LOSS_RATIO = 0.01
BUYER_CONTINUATION_MINIMUM_TOTAL_VARIATION = 0.1


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_reference(source: Path, project_root: Path) -> dict[str, str]:
    source = source.resolve()
    try:
        return {"kind": "project-relative", "path": str(source.relative_to(project_root.resolve()))}
    except ValueError:
        return {"kind": "absolute", "path": str(source)}


def _resolve_source(reference: Mapping[str, object], project_root: Path) -> Path:
    if reference.get("kind") == "absolute":
        return Path(str(reference.get("path") or "")).resolve()
    if reference.get("kind") != "project-relative":
        raise RuntimeError(f"unsupported Persuasion seed source reference: {reference.get('kind')!r}")
    root = project_root.resolve()
    path = (root / str(reference.get("path") or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("Persuasion seed source escapes the project root") from error
    return path


def _load_seed_document(path: Path, project_root: Path) -> tuple[dict[str, Any], Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Persuasion seed is not an object: {path}")
    if value.get("kind") != SEED_REFERENCE_KIND:
        return value, path.resolve()
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Persuasion seed reference has no source: {path}")
    resolved = _resolve_source(source, project_root)
    if not resolved.is_file() or file_sha256(resolved) != value.get("source_file_sha256"):
        raise RuntimeError(f"Persuasion seed source failed verification: {resolved}")
    seed = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(seed, dict) or seed.get("kind") != SEED_KIND or seed.get("seed_sha256") != value.get("seed_sha256"):
        raise RuntimeError(f"Persuasion seed source changed logical identity: {resolved}")
    return seed, resolved


def _seed_objects(seed: Mapping[str, object], seed_path: Path) -> list[dict[str, Any]]:
    inline = seed.get("games")
    if isinstance(inline, list):
        return copy.deepcopy(inline)
    reference = seed.get("games_ref")
    if not isinstance(reference, dict):
        raise RuntimeError(f"Persuasion seed has no game corpus: {seed_path}")
    return load_ordered_json_objects(root=seed_path.parent, reference=reference)


def _seed_digest(seed: Mapping[str, object], seed_path: Path) -> str:
    logical = {key: copy.deepcopy(value) for key, value in seed.items() if key not in {"seed_sha256", "games_ref"}}
    if "games_ref" in seed:
        logical["games"] = _seed_objects(seed, seed_path)
    return _sha(logical)


def install_seed(source: Path, destination: Path, *, project_root: Path) -> None:
    """Install a small immutable reference to one canonical Persuasion seed."""
    seed, seed_source = _load_seed_document(source.resolve(), project_root)
    if seed.get("kind") != SEED_KIND or seed.get("seed_sha256") != _seed_digest(seed, seed_source):
        raise RuntimeError(f"invalid Persuasion advisor seed: {seed_source}")
    reference = {"schema_version": SCHEMA_VERSION, "kind": SEED_REFERENCE_KIND, "source": _source_reference(seed_source, project_root), "source_file_sha256": file_sha256(seed_source), "seed_sha256": seed["seed_sha256"], "model_version": seed.get("model_version"), "corpus_sha256": seed.get("corpus_sha256")}
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != reference:
            raise RuntimeError(f"Persuasion advisor seed reference differs on resume: {destination}")
        return
    _atomic_json(destination, reference)


def _implementation_paths(project_root: Path) -> dict[str, Path]:
    return {
        "live_advisor_module": project_root / "src" / "nommd_arena" / "glee_persuasion_live_v2.py",
        "rating_surrogate_module": project_root / "src" / "nommd_arena" / "glee_persuasion_rating_v2_3.py",
        "advisor_contracts_module": project_root / "src" / "nommd_arena" / "glee_advisor_contracts.py",
        "seller_policy_config_module": project_root / "src" / "nommd_arena" / "glee_persuasion_policy_v2_4.py",
        "seller_policy_module": project_root / "src" / "nommd_arena" / "glee_persuasion_policy_v2_8.py",
        "twin_module": project_root / "src" / "nommd_arena" / "glee_persuasion_twin_v2.py",
        "live_policy_module": project_root / "src" / "nommd_arena" / "glee_live_policy.py",
        "meta_controller_module": project_root / "src" / "nommd_arena" / "glee_meta_controller_v2.py",
        "worker_module": project_root / "src" / "nommd_arena" / "glee_worker.py",
        "supervisor_module": project_root / "src" / "nommd_arena" / "glee_parallel.py",
        "transport_client_module": project_root / "src" / "nommd_arena" / "glee_transport.py",
        "dossier_broker_module": project_root / "src" / "nommd_arena" / "glee_dossier.py",
        "activity_scheduler_module": project_root / "src" / "nommd_arena" / "glee_activity_scheduler.py",
        "persuasion_prompt": project_root / "prompts" / "glee_nommd_persuasion.md",
        "meta_prompt_transport_module": project_root / "src" / "nommd_arena" / "model_runner.py",
        "meta_common_prompt": project_root / "prompts" / "glee_meta_controller_common.md",
        "meta_planner_stage_prompt": project_root / "prompts" / "glee_meta_controller_planner.md",
        "meta_selector_stage_prompt": project_root / "prompts" / "glee_meta_controller_selector.md",
        "meta_planner_prompt": project_root / "prompts" / "glee_meta_controller_planner_persuasion.md",
        "meta_selector_prompt": project_root / "prompts" / "glee_meta_controller_selector_persuasion.md",
        "live_protocol": project_root / "protocols" / "glee-persuasion-live-v2-8.md",
        "meta_controller_protocol": project_root / "protocols" / "glee-terra-meta-controller-live-v1.md",
        "transport_protocol": project_root / "protocols" / "glee-transport-fault-containment-v2.md",
    }


@dataclass(frozen=True)
class PersuasionAuthorityConfig:
    """Conservative boundaries under which local policy may bypass cloud inference."""

    allow_dominance: bool = True
    binary_only: bool = True
    pass_only_on_final_round: bool = True
    evidence_boundary: str = "credible-interval"
    min_current_revealed_purchases: int = 4

    def validate(self) -> None:
        if not self.allow_dominance:
            raise ValueError("payoff dominance is an invariant of the Persuasion authority contract")
        if self.evidence_boundary not in {"credible-interval", "posterior-mean-after-current-support"}:
            raise ValueError("unsupported Persuasion buyer evidence boundary")
        if isinstance(self.min_current_revealed_purchases, bool) or not isinstance(self.min_current_revealed_purchases, int) or self.min_current_revealed_purchases < 0:
            raise ValueError("minimum current revealed purchases must be a nonnegative integer")


def persuasion_turn_config(base_model: PersuasionModelConfig, base_authority: PersuasionAuthorityConfig, base_seller: PersuasionSellerPolicyConfig, live_policy: Mapping[str, object] | None) -> tuple[PersuasionModelConfig, PersuasionAuthorityConfig, PersuasionSellerPolicyConfig, dict[str, object] | None]:
    """Overlay only the stable bounded-control contract for one externally pinned game."""
    if live_policy is None:
        base_model.validate()
        base_authority.validate()
        base_seller.validate()
        return base_model, base_authority, base_seller, None
    if live_policy.get("schema_version") != 1 or live_policy.get("contract") != PERSUASION_LIVE_POLICY_CONTRACT or not isinstance(live_policy.get("revision"), str):
        raise ValueError("Persuasion live policy has an incompatible contract")
    parameters = live_policy.get("parameters")
    features = live_policy.get("features")
    allowed_features = set(PERSUASION_HOT_FEATURES) | set(PERSUASION_OPTIONAL_HOT_FEATURES)
    if not isinstance(parameters, Mapping) or set(parameters) != set(PERSUASION_HOT_PARAMETERS) or not isinstance(features, Mapping) or not set(PERSUASION_HOT_FEATURES).issubset(features) or not set(features).issubset(allowed_features):
        raise ValueError("Persuasion live policy has an incomplete bounded-control set")
    model = replace(base_model, authority_min_effective_support=float(parameters["authority_min_population_purchased_rows"]), buyer_buy_margin_ratio=float(parameters["buyer_buy_margin_ratio"]), buyer_pass_margin_ratio=float(parameters["buyer_pass_margin_ratio"]))
    authority = replace(base_authority, binary_only=features["buyer_channel_scope"] == "binary-only", pass_only_on_final_round=features["buyer_pass_scope"] == "final-round-only", evidence_boundary=str(features["buyer_evidence_boundary"]), min_current_revealed_purchases=int(parameters["buyer_min_current_revealed_purchases"]))
    seller = replace(base_seller, deception_advantage_margin=float(parameters["seller_deception_advantage_margin"]), no_response_min_passes=int(parameters["seller_no_response_min_passes"]), no_response_max_smoothed_buy_rate=float(parameters["seller_no_response_max_smoothed_buy_rate"]), no_response_routing=str(features.get("seller_no_response_routing") or "advisory-only"))
    model.validate()
    authority.validate()
    seller.validate()
    receipt = {
        "contract": PERSUASION_LIVE_POLICY_CONTRACT,
        "revision": live_policy["revision"],
        "release_sha256": live_policy.get("release_sha256"),
        "parameters": {name: parameters[name] for name in PERSUASION_HOT_PARAMETERS},
        "features": {name: features[name] for name in (*PERSUASION_HOT_FEATURES, *PERSUASION_OPTIONAL_HOT_FEATURES) if name in features},
        "scope": "externally pinned per game; preimplemented bounded action controls only",
    }
    return model, authority, seller, receipt


def _candidate_signal(value: str, channel: str) -> tuple[str, str, str]:
    return classify_persuasion_signal(value, channel=channel)


def _expected_surplus(probability_high: float, *, low_value: float, high_value: float, price: float) -> float:
    return probability_high * (high_value - price) + (1 - probability_high) * (low_value - price)


def _buyer_predictability_diagnostic(context: PersuasionContext) -> dict[str, object]:
    """Describe a publicly visible extreme buyer response pattern without changing action authority."""
    total = context.prior_buys + context.prior_passes
    if total == 0:
        pattern = "opening-no-history"
    elif context.prior_passes == 0:
        pattern = "all-buy"
    elif context.prior_buys == 0:
        pattern = "all-pass"
    else:
        pattern = "mixed"
    high_prior = context.prior_high_probability >= 0.75
    extreme = high_prior and total >= 4 and pattern in {"all-buy", "all-pass"}
    return {
        "status": "observed-current-game-diagnostic",
        "high_prior_regime": high_prior,
        "observed_decisions": total,
        "buys": context.prior_buys,
        "passes": context.prior_passes,
        "observed_buy_rate": context.prior_buys / total if total else None,
        "response_pattern": pattern,
        "extreme_policy_visible": extreme,
        "authority": "advisory-only",
        "interpretation": "A long all-buy or all-pass trace in a high-prior game can make the buyer policy easy for the seller to exploit. This visible-history diagnostic does not identify a profitable deviation and cannot override current-payoff arithmetic.",
    }


def persuasion_decision_facts(game: Mapping[str, Any], games: Sequence[PersuasionGameEvidence], *, model_config: PersuasionModelConfig | None = None, authority_config: PersuasionAuthorityConfig | None = None, seller_policy_config: PersuasionSellerPolicyConfig | None = None, rating_surrogate: PersuasionRatingSurrogate | None = None) -> dict[str, object]:
    """Build role-resolved scale-safe facts, shadow forecasts, and narrow authority."""
    config = model_config or PersuasionModelConfig()
    authority = authority_config or PersuasionAuthorityConfig()
    config.validate()
    authority.validate()
    context = context_from_game(game)
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    action_type = str((game.get("valid_actions") or {}).get("type") or game.get("phase") or "")
    scale = max(1.0, abs(context.product_price))
    visible = {
        "prior_buys": context.prior_buys,
        "prior_passes": context.prior_passes,
        "response_by_signal": {
            "positive": {"buys": context.positive_buys, "passes": context.positive_passes},
            "negative": {"buys": context.negative_buys, "passes": context.negative_passes},
            "unknown": {"buys": context.unknown_buys, "passes": context.unknown_passes},
        },
        "revealed_high": context.observed_high,
        "revealed_low": context.observed_low,
        "positive_revealed_high": context.positive_observed_high,
        "positive_revealed_low": context.positive_observed_low,
        "negative_revealed_high": context.negative_observed_high,
        "negative_revealed_low": context.negative_observed_low,
        "passed_quality_labels_used": 0,
    }
    facts: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "available",
        "role": context.our_role,
        "opponent_role": context.opponent_role,
        "channel": context.channel,
        "round": context.round_number,
        "total_rounds": context.total_rounds,
        "round_phase": context.round_phase,
        "prior_high_probability": context.prior_high_probability,
        "normalized_utility": {"low_surplus_over_price": context.normalized_low_surplus, "high_surplus_over_price": context.normalized_high_surplus, "prior_expected_surplus_over_price": context.normalized_prior_surplus},
        "visible_trust_evidence": visible,
        "censoring_contract": "A pass reveals no quality to the buyer model; terminal god's-eye pass labels never enter reliability fitting.",
        "authority": {"selected_action": None, "reason": "advisory-only", "action_authority": "advisory-only"},
    }
    if action_type == "buyer_decision":
        buyer_predictability = _buyer_predictability_diagnostic(context)
        facts["buyer_predictability_diagnostic"] = buyer_predictability
        if context.low_value is None or context.high_value is None:
            facts["status"] = "unavailable-masked-buyer-utilities"
            facts["authority"] = {"selected_action": None, "reason": "buyer utilities are required for a buyer decision", "action_authority": "unavailable"}
            facts["rating_objective_surrogate"] = {"status": "unavailable", "reason": "buyer utilities are required for a rating scenario"}
            return facts
        message = state.get("seller_message")
        polarity, act, fingerprint = classify_persuasion_signal(message, channel=context.channel)
        forecast = SellerReliabilityModel(games, config).predict(context, signal_polarity=polarity, message_fingerprint=fingerprint)
        lower_surplus = _expected_surplus(float(forecast["lower_90"]), low_value=context.low_value, high_value=context.high_value, price=context.product_price)
        upper_surplus = _expected_surplus(float(forecast["upper_90"]), low_value=context.low_value, high_value=context.high_value, price=context.product_price)
        mean_surplus = _expected_surplus(float(forecast["posterior_high_probability"]), low_value=context.low_value, high_value=context.high_value, price=context.product_price)
        facts["current_signal"] = {"polarity": polarity, "act": act, "message_fingerprint": fingerprint}
        facts["seller_reliability_forecast"] = {**forecast, "expected_surplus_over_price": mean_surplus / scale, "lower_expected_surplus_over_price": lower_surplus / scale, "upper_expected_surplus_over_price": upper_surplus / scale}
        expected_surplus_ratio = mean_surplus / scale
        lower_surplus_ratio = lower_surplus / scale
        upper_surplus_ratio = upper_surplus / scale
        uncertainty_close = abs(expected_surplus_ratio) <= BUYER_INDIFFERENCE_BAND_RATIO
        robust_buy = expected_surplus_ratio > 0.0 and (not uncertainty_close or lower_surplus_ratio > 0.0)
        robust_action = {"decision": "yes" if robust_buy else "no"}
        terminal = context.round_number >= context.total_rounds
        facts["buyer_economic_control"] = {
            "contract": "glee-persuasion-buyer-economic-control-v1",
            "price_normalization": scale,
            "expected_surplus": mean_surplus,
            "expected_surplus_over_price": expected_surplus_ratio,
            "lower_expected_surplus_over_price": lower_surplus_ratio,
            "upper_expected_surplus_over_price": upper_surplus_ratio,
            "indifference_band_ratio": BUYER_INDIFFERENCE_BAND_RATIO,
            "inside_uncertainty_indifference_band": uncertainty_close,
            "material_loss_ratio": BUYER_MATERIAL_LOSS_RATIO,
            "continuation_minimum_total_variation": BUYER_CONTINUATION_MINIMUM_TOTAL_VARIATION,
            "common_unit_continuation_lower_bound_over_price": None,
            "buy_economically_admissible": expected_surplus_ratio > -BUYER_MATERIAL_LOSS_RATIO,
            "robust_buy_authorized": robust_buy,
            "robust_uncertainty_action": robust_action,
            "local_expected_value_action": robust_action,
            "terminal": terminal,
            "terminal_selected_action": robust_action if terminal else None,
            "terminal_tie_rule": "pass on an exact tie and inside the 5% uncertainty band unless the posterior 90% lower surplus bound is positive; rating v3 remains advisory",
            "selector_economic_scope": "a cloud reversal additionally requires a lower-bounded continuation value expressed in current-payoff units; predictive next-signal variation alone is advisory",
            "predictability_diagnostic": buyer_predictability,
        }
        low_surplus = min(context.low_value, context.high_value) - context.product_price
        high_surplus = max(context.low_value, context.high_value) - context.product_price
        selected: dict[str, str] | None = None
        reason = "advisory-only"
        if authority.allow_dominance and low_surplus >= 0 and high_surplus > 0:
            selected, reason = {"decision": "yes"}, "buy-dominates-pass-in-every-quality-state"
        elif authority.allow_dominance and high_surplus <= 0 and low_surplus < 0:
            selected, reason = {"decision": "no"}, "pass-dominates-buy-in-every-quality-state"
        elif (not authority.binary_only or context.channel == "binary") and int(forecast["population_matching_purchased_rows"]) >= config.authority_min_effective_support:
            pass_allowed = not authority.pass_only_on_final_round or context.round_number >= context.total_rounds
            if authority.evidence_boundary == "credible-interval":
                if lower_surplus > config.buyer_buy_margin_ratio * scale:
                    selected, reason = {"decision": "yes"}, "posterior-lower-bound-has-positive-current-surplus"
                elif pass_allowed and upper_surplus < -config.buyer_pass_margin_ratio * scale:
                    selected, reason = {"decision": "no"}, "posterior-upper-bound-has-negative-current-surplus"
            elif context.observed_high + context.observed_low >= authority.min_current_revealed_purchases:
                if mean_surplus > config.buyer_buy_margin_ratio * scale:
                    selected, reason = {"decision": "yes"}, "posterior-mean-has-positive-current-surplus-after-current-support"
                elif pass_allowed and mean_surplus < -config.buyer_pass_margin_ratio * scale:
                    selected, reason = {"decision": "no"}, "posterior-mean-has-negative-current-surplus-after-current-support"
        selected_authority = "categorical" if selected is not None and reason in {"buy-dominates-pass-in-every-quality-state", "pass-dominates-buy-in-every-quality-state"} else "bounded-authoritative" if selected is not None else "advisory-only"
        facts["authority"] = {"selected_action": selected, "reason": reason, "action_authority": selected_authority, "text_realization": "not-applicable"}
    elif action_type in {"seller_recommendation", "seller_message"}:
        response_model = BuyerResponseModel(games, config)
        candidate_forecasts: dict[str, object] = {}
        candidates = (("positive", "yes"), ("negative", "no")) if context.channel == "binary" else (("positive", "I recommend buying this product."), ("negative", "I recommend passing on this product."))
        for label, candidate in candidates:
            polarity, act, fingerprint = _candidate_signal(candidate, context.channel)
            candidate_forecasts[label] = response_model.predict(context, signal_polarity=polarity, signal_act=act, message_fingerprint=fingerprint)
        facts["buyer_response_forecasts"] = candidate_forecasts
        current_quality = str(state.get("current_quality") or "")
        facts["seller_private_facts"] = {"current_quality": current_quality if current_quality in {"high", "low"} else None, "seller_knows_buyer_values": context.seller_knows_buyer_values}
        control = seller_longitudinal_control(context, current_quality=current_quality if current_quality in {"high", "low"} else None, buyer_response_forecasts=candidate_forecasts, config=seller_policy_config)
        facts["seller_longitudinal_control"] = control
        polarity = str(control.get("selected_signal_polarity") or "")
        selected = {"decision": "yes" if polarity == "positive" else "no"} if action_type == "seller_recommendation" and polarity in {"positive", "negative"} else None
        bounded_no_response = control.get("authority") == "bounded-no-response-authority"
        action_authority = "bounded-authoritative" if bounded_no_response else "advisory-anchor" if polarity in {"positive", "negative"} else "advisory-only"
        candidate_policy = "deterministic-quality-consistent-no-response" if bounded_no_response else "compare-anchor-with-serious-alternative" if action_authority == "advisory-anchor" else "unresolved"
        facts["authority"] = {"selected_action": selected, "selected_signal_polarity": polarity or None, "reason": control.get("selection_reason"), "action_authority": action_authority, "policy_scope": control.get("authority"), "candidate_policy": candidate_policy, "text_realization": "Use the canonical quality-consistent message without cloud deliberation" if bounded_no_response and action_type == "seller_message" else "Terra compares both signal polarity and wording while retaining the v2.8 recommendation as the fallback anchor" if action_type == "seller_message" and polarity in {"positive", "negative"} else "not-applicable"}
    else:
        facts["status"] = "unsupported-action-type"
        facts["unsupported_action_type"] = action_type
    facts["rating_objective_surrogate"] = persuasion_rating_decision_surface(game, facts, rating_surrogate)
    return facts


@dataclass(frozen=True)
class PersuasionTurnForecast:
    """Prompt facts and shadow forecast frozen before cloud inference."""

    game: dict[str, Any]
    prompt_context: dict[str, object]
    forecast_receipt: dict[str, object]
    seed_sha256: str
    state_revision: int

    def guard_action(self, action: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        from .glee_policy import normalize_action

        authority = self.prompt_context.get("authority") if isinstance(self.prompt_context.get("authority"), Mapping) else {}
        selected = authority.get("selected_action") if isinstance(authority, Mapping) else None
        candidate = normalize_action(self.game, action)
        reason = str(authority.get("reason") or "narrow-authority")
        action_type = str((self.game.get("valid_actions") or {}).get("type") or self.game.get("phase") or "")
        action_authority = str(authority.get("action_authority") or "legacy-authoritative")
        buyer_control = self.prompt_context.get("buyer_economic_control") if isinstance(self.prompt_context.get("buyer_economic_control"), Mapping) else {}
        if action_type in {"seller_message", "seller_recommendation"} and action_authority == "advisory-anchor":
            return candidate, []
        if action_type == "buyer_decision" and isinstance(buyer_control, Mapping):
            robust_selected = buyer_control.get("robust_uncertainty_action")
            if not isinstance(selected, Mapping) and isinstance(robust_selected, Mapping) and buyer_control.get("common_unit_continuation_lower_bound_over_price") is None:
                normalized = normalize_action(self.game, dict(robust_selected))
                if normalized != candidate:
                    return normalized, ["persuasion_buyer_robust_uncertainty_boundary"]
            terminal_selected = buyer_control.get("terminal_selected_action") if buyer_control.get("terminal") is True else None
            if isinstance(terminal_selected, Mapping):
                normalized = normalize_action(self.game, dict(terminal_selected))
                if normalized != candidate:
                    return normalized, ["persuasion_buyer_terminal_expected_value"]
            if str(candidate.get("decision") or "").casefold() == "yes" and buyer_control.get("buy_economically_admissible") is False:
                return normalize_action(self.game, {"decision": "no"}), ["persuasion_buyer_material_negative_purchase"]
        if isinstance(selected, Mapping):
            normalized = normalize_action(self.game, dict(selected))
            if normalized == candidate:
                return candidate, []
            return normalized, [f"persuasion_v27_{reason.replace('-', '_').replace(' ', '_')}"]
        selected_polarity = str(authority.get("selected_signal_polarity") or "")
        if action_type != "seller_message" or selected_polarity not in {"positive", "negative"}:
            return candidate, []
        candidate_polarity = classify_persuasion_signal(candidate.get("message"), channel="text")[0]
        if candidate_polarity == selected_polarity:
            return candidate, []
        replacement = normalize_action(self.game, {"message": canonical_seller_message(selected_polarity)})
        return replacement, [f"persuasion_v27_semantic_{selected_polarity}_{reason.replace('-', '_').replace(' ', '_')}"]

    def submission_prediction(self, action: dict[str, Any]) -> dict[str, object]:
        from .glee_policy import normalize_action

        submitted = normalize_action(self.game, action)
        authority = self.prompt_context.get("authority") if isinstance(self.prompt_context.get("authority"), Mapping) else {}
        selected = authority.get("selected_action") if isinstance(authority, Mapping) else None
        selected_polarity = str(authority.get("selected_signal_polarity") or "") if isinstance(authority, Mapping) else ""
        action_authority = str(authority.get("action_authority") or "legacy-authoritative") if isinstance(authority, Mapping) else "legacy-authoritative"
        warnings: list[str] = []
        advisory_anchor_disposition = "not-applicable"
        if action_authority == "advisory-anchor":
            if isinstance(selected, Mapping):
                advisory_anchor_disposition = "selected" if normalize_action(self.game, dict(selected)) == submitted else "overridden"
            elif selected_polarity in {"positive", "negative"}:
                advisory_anchor_disposition = "selected" if classify_persuasion_signal(submitted.get("message"), channel="text")[0] == selected_polarity else "overridden"
        elif isinstance(selected, Mapping) and normalize_action(self.game, dict(selected)) != submitted:
            warnings.append("submitted_action_differs_from_narrow_authority")
        action_type = str((self.game.get("valid_actions") or {}).get("type") or self.game.get("phase") or "")
        if action_authority != "advisory-anchor" and action_type == "seller_message" and selected_polarity in {"positive", "negative"} and classify_persuasion_signal(submitted.get("message"), channel="text")[0] != selected_polarity:
            warnings.append("submitted_message_differs_from_longitudinal_semantic_authority")
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "frontier": "computed-after-action-normalization-from-the-frozen-pre-inference-forecast",
            "forecast_id": self.forecast_receipt["forecast_id"],
            "seed_sha256": self.seed_sha256,
            "state_revision": self.state_revision,
            "game_id": self.game.get("game_id"),
            "submitted_action": submitted,
            "authority_selected_action": copy.deepcopy(selected),
            "authority_selected_signal_polarity": selected_polarity or None,
            "action_authority": action_authority,
            "advisory_anchor_disposition": advisory_anchor_disposition,
            "status": "consistent" if not warnings else "material-warning",
            "warnings": warnings,
        }


class PersuasionLiveSeed:
    """Seal all completed archived Persuasion games at one causal frontier."""

    def __init__(self, *, output_path: Path, project_root: Path, game_archive_root: Path, rating_history_path: Path, model_config: PersuasionModelConfig | None = None, authority_config: PersuasionAuthorityConfig | None = None, seller_policy_config: PersuasionSellerPolicyConfig | None = None) -> None:
        self.output_path = output_path
        self.project_root = project_root
        self.game_archive_root = game_archive_root
        self.rating_history_path = rating_history_path
        self.model_config = model_config or PersuasionModelConfig()
        self.authority_config = authority_config or PersuasionAuthorityConfig()
        self.seller_policy_config = seller_policy_config or PersuasionSellerPolicyConfig()
        self.model_config.validate()
        self.authority_config.validate()
        self.seller_policy_config.validate()

    def run(self) -> dict[str, object]:
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite Persuasion seed: {self.output_path}")
        games, rejected = load_persuasion_archive(self.game_archive_root, rating_history_path=self.rating_history_path)
        if not games:
            raise RuntimeError("Persuasion seed has no completed games")
        objects: list[dict[str, Any]] = []
        for game in games:
            final_game = json.loads(Path(game.source_path).read_text(encoding="utf-8"))
            objects.append({"completed_at": game.completed_at, "completion_order": game.completion_order, "final_game_sha256": game.final_game_sha256, "final_game": final_game})
        rating_history = json.loads(self.rating_history_path.read_text(encoding="utf-8"))
        rating_deltas = rating_history.get("game_deltas") if isinstance(rating_history.get("game_deltas"), Mapping) else {}
        rating_surrogate = fit_persuasion_rating_surrogate(objects, rating_deltas)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        games_ref, _unique = seal_json_objects(root=self.output_path.parent, directory=self.output_path.parent / "corpora", prefix="persuasion-games", objects=objects)
        implementations = {label: {"path": str(path.relative_to(self.project_root)), "sha256": file_sha256(path)} for label, path in _implementation_paths(self.project_root).items()}
        seed: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": SEED_KIND,
            "model_version": MODEL_VERSION,
            "generated_at": _now(),
            "frontier": "Only this content-addressed completed-game corpus initializes the epoch; later terminal games enter through the append-only completion journal.",
            "model_config": asdict(self.model_config),
            "authority_config": asdict(self.authority_config),
            "seller_policy_config": asdict(self.seller_policy_config),
            "model_receipt": model_receipt(self.model_config),
            "rating_surrogate": rating_surrogate.as_dict() if rating_surrogate is not None else None,
            "game_count": len(objects),
            "corpus_sha256": _sha([game.final_game_sha256 for game in games]),
            "rejected_source_count": len(rejected),
            "implementation_receipts": implementations,
            "games_ref": games_ref,
        }
        seed["seed_sha256"] = _seed_digest(seed, self.output_path)
        _atomic_json(self.output_path, seed)
        return {"seed_path": str(self.output_path), "seed_sha256": seed["seed_sha256"], "corpus_sha256": seed["corpus_sha256"], "game_count": len(objects), "rejected_source_count": len(rejected), "model_version": MODEL_VERSION}


def reseal_persuasion_seed_implementation(*, source_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Preserve one Persuasion frontier exactly while refreshing implementation receipts."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Persuasion advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path.resolve(), project_root)
    parent_model_version = str(seed.get("model_version") or "")
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or parent_model_version not in {MODEL_VERSION, "persuasion-live-advisor-v2.4", "persuasion-live-advisor-v2.5", "persuasion-live-advisor-v2.7"}:
        raise RuntimeError(f"unsupported parent Persuasion advisor seed: {source_path}")
    if seed.get("seed_sha256") != _seed_digest(seed, seed_source_path):
        raise RuntimeError(f"invalid parent Persuasion advisor seed: {source_path}")
    if "games_ref" in seed and output_path.parent.resolve() != seed_source_path.parent.resolve():
        raise ValueError("a referenced Persuasion corpus can be resealed only beside its source manifest")
    parent_seed_sha256 = str(seed["seed_sha256"])
    if parent_model_version == "persuasion-live-advisor-v2.4":
        authority_config = dict(seed.get("authority_config") or {})
        authority_config.update({"evidence_boundary": "credible-interval", "min_current_revealed_purchases": 4})
        seed["authority_config"] = authority_config
    if parent_model_version != MODEL_VERSION:
        seed["model_version"] = MODEL_VERSION
    seller_policy_config = dict(seed.get("seller_policy_config") or {})
    seller_policy_config.setdefault("no_response_routing", "advisory-only")
    seed["seller_policy_config"] = seller_policy_config
    seed["generated_at"] = _now()
    seed["parent_seed_sha256"] = parent_seed_sha256
    seed["parent_model_version"] = parent_model_version
    seed["reseal_reason"] = "v2.8-bounded-no-response-routing-migration-with-identical-historical-frontier" if parent_model_version != MODEL_VERSION else "implementation-receipt-refresh-with-identical-embedded-historical-frontier"
    seed["implementation_receipts"] = {label: {"path": str(path.relative_to(project_root)), "sha256": file_sha256(path)} for label, path in _implementation_paths(project_root).items()}
    seed["seed_sha256"] = _seed_digest(seed, output_path)
    _atomic_json(output_path, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "path": str(output_path),
        "model_version": MODEL_VERSION,
        "parent_seed_sha256": parent_seed_sha256,
        "seed_sha256": seed["seed_sha256"],
        "corpus_sha256": seed["corpus_sha256"],
        "game_count": seed["game_count"],
    }


def promote_persuasion_seed_journal(*, source_path: Path, journal_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Fold one verified Persuasion completion journal into its exact parent frontier."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Persuasion advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path.resolve(), project_root)
    parent_model_version = str(seed.get("model_version") or "")
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or parent_model_version not in {MODEL_VERSION, "persuasion-live-advisor-v2.7"}:
        raise RuntimeError(f"unsupported parent Persuasion advisor seed: {source_path}")
    if seed.get("seed_sha256") != _seed_digest(seed, seed_source_path):
        raise RuntimeError(f"invalid parent Persuasion advisor seed: {source_path}")
    if "games_ref" in seed and output_path.parent.resolve() != seed_source_path.parent.resolve():
        raise ValueError("a referenced Persuasion corpus can be promoted only beside its source manifest")
    if not journal_path.is_file():
        raise FileNotFoundError(f"Persuasion completion journal does not exist: {journal_path}")

    parent_seed_sha256 = str(seed["seed_sha256"])
    embedded_games = _seed_objects(seed, seed_source_path)
    parent_game_count = len(embedded_games)
    promoted_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        expected_sequence = len(promoted_records) + 1
        if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != JOURNAL_KIND or record.get("model_version") != parent_model_version or record.get("seed_sha256") != parent_seed_sha256:
            raise RuntimeError(f"invalid Persuasion completion journal record at line {line_number}")
        if record.get("journal_sequence") != expected_sequence:
            raise RuntimeError(f"non-contiguous Persuasion completion journal sequence at line {line_number}")
        embedded = record.get("embedded_game")
        if not isinstance(embedded, dict):
            raise RuntimeError(f"Persuasion completion journal record has no embedded game at line {line_number}")
        promoted_records.append(copy.deepcopy(embedded))
    if not promoted_records:
        raise RuntimeError(f"Persuasion completion journal has no records: {journal_path}")

    embedded_games.extend(promoted_records)
    completed_hashes: dict[str, str] = {}
    evidence: list[PersuasionGameEvidence] = []
    for index, embedded in enumerate(embedded_games, start=1):
        final_game = embedded.get("final_game")
        final_game_sha256 = str(embedded.get("final_game_sha256") or "")
        if not isinstance(final_game, dict) or final_game_sha256 != object_sha256(final_game):
            raise RuntimeError(f"invalid embedded Persuasion game at promoted position {index}")
        game_id = str(final_game.get("game_id") or "")
        if not game_id:
            raise RuntimeError(f"embedded Persuasion game has no game ID at promoted position {index}")
        previous = completed_hashes.get(game_id)
        if previous is not None:
            if previous != final_game_sha256:
                raise RuntimeError(f"conflicting completed Persuasion game in promoted frontier: {game_id}")
            raise RuntimeError(f"duplicate completed Persuasion game in promoted frontier: {game_id}")
        completed_hashes[game_id] = final_game_sha256
        evidence.append(extract_persuasion_game(final_game, completed_at=str(embedded.get("completed_at") or ""), completion_order=int(embedded.get("completion_order") or index), source_path=f"promoted/{game_id}.json"))

    promoted_seed: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "model_version": MODEL_VERSION,
        "generated_at": _now(),
        "parent_seed_sha256": parent_seed_sha256,
        "parent_model_version": parent_model_version,
        "frontier": "Only this content-addressed completed-game corpus initializes the epoch; later terminal games enter through the append-only completion journal.",
        "source": {
            "kind": "parent-seed-plus-completion-journal",
            "parent_seed_path": str(seed_source_path),
            "parent_seed_sha256": parent_seed_sha256,
            "parent_game_count": parent_game_count,
            "journal_path": str(journal_path.resolve()),
            "journal_sha256": file_sha256(journal_path),
            "journal_game_count": len(promoted_records),
        },
        "promotion_reason": "completed-epoch-journal-folded-for-an-operational-body-cutover",
        "model_config": copy.deepcopy(seed["model_config"]),
        "authority_config": copy.deepcopy(seed["authority_config"]),
        "seller_policy_config": {**copy.deepcopy(seed["seller_policy_config"]), "no_response_routing": str((seed.get("seller_policy_config") or {}).get("no_response_routing") or "advisory-only")},
        "model_receipt": copy.deepcopy(seed["model_receipt"]),
        "rating_surrogate": copy.deepcopy(seed.get("rating_surrogate")),
        "game_count": len(embedded_games),
        "corpus_sha256": _sha([game.final_game_sha256 for game in evidence]),
        "rejected_source_count": int(seed.get("rejected_source_count") or 0),
        "implementation_receipts": {label: {"path": str(path.relative_to(project_root)), "sha256": file_sha256(path)} for label, path in _implementation_paths(project_root).items()},
    }
    games_reference, _unique_games = seal_json_objects(root=output_path.parent, directory=output_path.parent / "corpora", prefix="persuasion-games", objects=embedded_games)
    promoted_seed["games_ref"] = games_reference
    promoted_seed["seed_sha256"] = _seed_digest(promoted_seed, output_path)
    _atomic_json(output_path, promoted_seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "path": str(output_path),
        "model_version": MODEL_VERSION,
        "parent_seed_sha256": parent_seed_sha256,
        "seed_sha256": promoted_seed["seed_sha256"],
        "corpus_sha256": promoted_seed["corpus_sha256"],
        "parent_game_count": parent_game_count,
        "journal_game_count": len(promoted_records),
        "game_count": len(embedded_games),
    }


class PersuasionLiveAdvisorV2:
    """Serve frozen forecasts and update only from authenticated terminal games."""

    def __init__(self, *, seed_path: Path, journal_path: Path, project_root: Path) -> None:
        self.seed_path = seed_path
        self.journal_path = journal_path
        self.project_root = project_root
        self._lock = threading.RLock()
        self.seed, self.seed_source_path = _load_seed_document(seed_path, project_root)
        self._verify_seed()
        self.seed_sha256 = str(self.seed["seed_sha256"])
        self.model_config = PersuasionModelConfig(**dict(self.seed["model_config"]))
        self.authority_config = PersuasionAuthorityConfig(**dict(self.seed["authority_config"]))
        self.seller_policy_config = PersuasionSellerPolicyConfig(**dict(self.seed["seller_policy_config"]))
        self.model_config.validate()
        self.authority_config.validate()
        self.seller_policy_config.validate()
        rating_value = self.seed.get("rating_surrogate")
        self.rating_surrogate = PersuasionRatingSurrogate.from_dict(rating_value) if isinstance(rating_value, Mapping) else None
        self.games: list[PersuasionGameEvidence] = []
        self.completed_game_hashes: dict[str, str] = {}
        self.journal_sequence = 0
        for embedded in _seed_objects(self.seed, self.seed_source_path):
            self._apply_embedded(embedded)
        self.seed_game_count = len(self.games)
        self._replay_journal()

    def _verify_seed(self) -> None:
        if self.seed.get("schema_version") != SCHEMA_VERSION or self.seed.get("kind") != SEED_KIND or self.seed.get("model_version") != MODEL_VERSION:
            raise RuntimeError(f"unsupported Persuasion advisor seed: {self.seed_path}")
        if self.seed.get("seed_sha256") != _seed_digest(self.seed, self.seed_source_path):
            raise RuntimeError(f"Persuasion seed SHA-256 mismatch: {self.seed_path}")
        for label, receipt in dict(self.seed.get("implementation_receipts") or {}).items():
            path = self.project_root / str(receipt["path"])
            if not path.is_file() or file_sha256(path) != receipt.get("sha256"):
                raise RuntimeError(f"Persuasion implementation receipt mismatch: {label}")

    def _apply_embedded(self, embedded: Mapping[str, Any]) -> None:
        final_game = embedded.get("final_game")
        if not isinstance(final_game, Mapping) or object_sha256(final_game) != embedded.get("final_game_sha256"):
            raise RuntimeError("Persuasion embedded terminal game failed verification")
        game_id = str(final_game.get("game_id") or "")
        digest = str(embedded["final_game_sha256"])
        prior = self.completed_game_hashes.get(game_id)
        if prior is not None:
            if prior != digest:
                raise RuntimeError(f"conflicting Persuasion terminal game: {game_id}")
            return
        evidence = extract_persuasion_game(final_game, completed_at=str(embedded.get("completed_at") or ""), completion_order=int(embedded.get("completion_order") or len(self.games) + 1), source_path="embedded")
        self.games.append(evidence)
        self.completed_game_hashes[game_id] = digest

    def _replay_journal(self) -> None:
        if not self.journal_path.is_file():
            return
        for line_number, line in enumerate(self.journal_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") != JOURNAL_KIND or record.get("seed_sha256") != self.seed_sha256:
                raise RuntimeError(f"invalid Persuasion completion journal record at line {line_number}")
            self.journal_sequence = max(self.journal_sequence, int(record["journal_sequence"]))
            self._apply_embedded(record["embedded_game"])

    def forecast_turn(self, game: dict[str, Any], *, live_policy: Mapping[str, object] | None = None) -> PersuasionTurnForecast:
        with self._lock:
            model_config, authority_config, seller_policy_config, policy_receipt = persuasion_turn_config(self.model_config, self.authority_config, self.seller_policy_config, live_policy)
            facts = persuasion_decision_facts(game, tuple(self.games), model_config=model_config, authority_config=authority_config, seller_policy_config=seller_policy_config, rating_surrogate=self.rating_surrogate)
            facts["live_policy"] = policy_receipt
            revision = len(self.games)
            forecast_id = _sha({"seed_sha256": self.seed_sha256, "state_revision": revision, "game": game, "facts": facts})
            receipt = {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION, "forecast_id": forecast_id, "seed_sha256": self.seed_sha256, "state_revision": revision, "game_id": game.get("game_id"), "frontier": "frozen-before-cloud-inference", "facts_sha256": _sha(facts)}
            return PersuasionTurnForecast(game=copy.deepcopy(game), prompt_context=facts, forecast_receipt=receipt, seed_sha256=self.seed_sha256, state_revision=revision)

    def update_completed_game(self, final_game: dict[str, Any], *, completed_at: str, completion_order: int) -> dict[str, object]:
        with self._lock:
            result = final_game.get("result") if isinstance(final_game.get("result"), Mapping) else {}
            if str(result.get("outcome") or final_game.get("status") or "").casefold() != "completed":
                return {"status": "censored-terminal-state", "game_id": final_game.get("game_id")}
            digest = object_sha256(final_game)
            game_id = str(final_game.get("game_id") or "")
            if game_id in self.completed_game_hashes:
                if self.completed_game_hashes[game_id] != digest:
                    raise RuntimeError(f"conflicting Persuasion completion: {game_id}")
                return {"status": "duplicate", "game_id": game_id, "state_revision": len(self.games)}
            embedded = {"completed_at": completed_at, "completion_order": completion_order, "final_game_sha256": digest, "final_game": copy.deepcopy(final_game)}
            self.journal_sequence += 1
            record = {"schema_version": SCHEMA_VERSION, "kind": JOURNAL_KIND, "model_version": MODEL_VERSION, "seed_sha256": self.seed_sha256, "journal_sequence": self.journal_sequence, "recorded_at": _now(), "embedded_game": embedded}
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(_canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._apply_embedded(embedded)
            return {"status": "applied", "game_id": game_id, "state_revision": len(self.games), "journal_sequence": self.journal_sequence}

    def manifest_receipt(self) -> dict[str, object]:
        return {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION, "seed_sha256": self.seed_sha256, "corpus_sha256": self.seed.get("corpus_sha256"), "seed_game_count": self.seed_game_count, "journal_path": str(self.journal_path), "rating_surrogate": self.rating_surrogate.as_dict() if self.rating_surrogate is not None else None, "hot_policy_support": {"contract": PERSUASION_LIVE_POLICY_CONTRACT, "parameters": list(PERSUASION_HOT_PARAMETERS), "features": list(PERSUASION_HOT_FEATURES), "pinning_owner": "parallel supervisor"}}

    def status(self) -> dict[str, object]:
        roles = Counter(game.opponent_role for game in self.games)
        return {**self.manifest_receipt(), "state_revision": len(self.games), "journal_game_count": len(self.games) - self.seed_game_count, "opponent_role_games": dict(sorted(roles.items()))}
