from types import SimpleNamespace

import pytest

from nommd_arena.glee_bargaining_intervention import INTERVENTION_CONTRACT, apply_bargaining_v217_intervention, build_bargaining_v217_context, intervention_receipt


def _offer_game(*, rejected_opponent_share: float = 0.5, round_number: int = 3) -> dict[str, object]:
    offer = {"round": 1, "proposer": "player_1", "player_1_gain": 100 * (1 - rejected_opponent_share), "player_2_gain": 100 * rejected_opponent_share}
    return {"game_id": "offer", "game_family": "bargaining", "your_player": "player_1", "game_state": {"current_player": "player_1", "money_to_divide": 100.0, "round": round_number, "history": [{"round": 1, "proposer": "player_1", "offer": offer, "decision": "reject"}], "horizon_known": True, "max_rounds": 12}, "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}}}


def _advisor_context() -> dict[str, object]:
    return {"model_version": "bargaining-live-advisor-v2.17", "status": "available", "response_to_our_numeric_offer": {"myopic_no_continuation_candidate": {"opponent_share": 0.45}}, "behavioral_continuation": {"modeled_offer_policy": {"opponent_share": 0.4}, "policy_guard": {"mode": "bounded-authoritative", "minimum_opponent_share": 0.25, "maximum_opponent_share": 0.8, "loss_minimization_control": {"status": "inactive"}}}}


def _context(*, package_authority: bool = False, response_forecast: dict[str, object] | None = None) -> dict[str, object]:
    curve = [
        {"opponent_share": 0.4, "blended_expected_value": 0.2, "specialized_expected_value": 0.2},
        {"opponent_share": 0.55, "blended_expected_value": 0.4, "specialized_expected_value": 0.35},
        {"opponent_share": 0.95, "blended_expected_value": 0.1, "specialized_expected_value": 0.1},
    ]
    rating = {"status": "available", "candidate_surface": [response_forecast] if response_forecast is not None else []}
    return {"contract": INTERVENTION_CONTRACT, "package": {"authority": "bounded-offer-candidate-scoring" if package_authority else "diagnostic-only", "curve": curve, "recommendation": curve[1] if package_authority else None}, "rating": rating}


def _live_policy(*, comparator: str = "nearest-grid", patient_guard: str = "continuation-nondominated", minimum_nonterminal_own_share: float = 0.2, continuation_tolerance: float = 0.01, response_rating_loss_guard: str | None = None, rating_v3_low_share_guard: str | None = None) -> dict[str, object]:
    features = {"selected_curve_comparator": comparator, "patient_acceptance_guard": patient_guard}
    if response_rating_loss_guard is not None:
        features["response_rating_loss_guard"] = response_rating_loss_guard
    if rating_v3_low_share_guard is not None:
        features["rating_v3_low_share_guard"] = rating_v3_low_share_guard
    return {"schema_version": 1, "contract": "glee-bargaining-live-policy-v1", "revision": "test", "parameters": {"minimum_package_support": 2.0, "maximum_package_blend_weight": 0.35, "minimum_package_value_improvement": 0.02, "minimum_nonterminal_own_share": minimum_nonterminal_own_share, "extreme_opponent_share": 0.05, "continuation_tolerance": continuation_tolerance, "rating_point_catastrophe": -5.0}, "features": features}


def test_v217_context_keeps_non_rating_intervention_active_without_legacy_rating_canary() -> None:
    context = build_bargaining_v217_context(game=_offer_game(), package_context={}, advisor_handle=object(), advisor_context=_advisor_context(), rating_canary=None, observed_at="2026-08-25T12:00:00Z", live_policy=_live_policy())
    assert context["contract"] == INTERVENTION_CONTRACT
    assert context["frontier"] == "current-turn-before-model-inference"
    assert context["package"]["authority"] == "diagnostic-only"
    assert context["rating"] == {"status": "unavailable", "reason": "legacy-rating-authority-disabled"}


def test_v217_package_authority_changes_only_a_materially_better_offer_candidate() -> None:
    game = _offer_game()
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_advisor_context(), bargaining_intervention_context=_context(package_authority=True))
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"alice_gain": 60.0, "bob_gain": 40.0})
    assert action == {"alice_gain": pytest.approx(45.0), "bob_gain": pytest.approx(55.0)}
    assert safeguards == ["bargaining_v217_exact_package_offer_candidate"]


def test_live_policy_can_promote_interpolated_package_comparison_without_changing_the_intervention_contract() -> None:
    game = _offer_game()
    context = _context(package_authority=True)
    context["package"]["curve"] = [{"opponent_share": 0.45, "blended_expected_value": 0.3}, {"opponent_share": 0.55, "blended_expected_value": 0.33}]
    context["package"]["recommendation"] = context["package"]["curve"][1]
    context["live_policy"] = _live_policy(comparator="linear-interpolation")
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_advisor_context(), bargaining_intervention_context=context)
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"alice_gain": 50.0, "bob_gain": 50.0})
    assert action == {"alice_gain": 50.0, "bob_gain": 50.0}
    assert safeguards == []
    context["live_policy"] = _live_policy(comparator="nearest-grid")
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"alice_gain": 50.0, "bob_gain": 50.0})
    assert action == {"alice_gain": pytest.approx(45.0), "bob_gain": pytest.approx(55.0)}
    assert safeguards == ["bargaining_v217_exact_package_offer_candidate"]


def test_v217_offer_guards_prevent_post_rejection_giveaway_and_rejected_extreme_repetition() -> None:
    giveaway_game = _offer_game(rejected_opponent_share=0.5)
    giveaway = SimpleNamespace(game=giveaway_game, bargaining_advisor_context=_advisor_context(), bargaining_intervention_context=_context())
    action, safeguards = apply_bargaining_v217_intervention(giveaway, {"alice_gain": 0.0, "bob_gain": 100.0})
    assert action == {"alice_gain": pytest.approx(60.0), "bob_gain": pytest.approx(40.0)}
    assert safeguards == ["bargaining_v217_post_rejection_capitulation_guard"]

    repetition_game = _offer_game(rejected_opponent_share=0.0)
    repetition = SimpleNamespace(game=repetition_game, bargaining_advisor_context=_advisor_context(), bargaining_intervention_context=_context())
    action, safeguards = apply_bargaining_v217_intervention(repetition, {"alice_gain": 100.0, "bob_gain": 0.0})
    assert action == {"alice_gain": pytest.approx(60.0), "bob_gain": pytest.approx(40.0)}
    assert safeguards == ["bargaining_v217_rejected_extreme_offer_repetition_guard"]


def _decision_game(*, round_number: int = 4, max_rounds: int = 10) -> dict[str, object]:
    offer = {"round": round_number, "proposer": "player_1", "player_1_gain": 92.0, "player_2_gain": 8.0}
    return {"game_id": "decision", "game_family": "bargaining", "your_player": "player_2", "game_state": {"current_player": "player_2", "money_to_divide": 100.0, "round": round_number, "last_offer": offer, "history": [], "horizon_known": True, "max_rounds": max_rounds}, "valid_actions": {"type": "decision", "fields": {"decision": "enum"}}}


def _decision_advisor() -> dict[str, object]:
    return {"model_version": "bargaining-live-advisor-v2.17", "status": "available", "behavioral_continuation": {"decision_comparison": {"accept_now_value": 0.04096, "reject_value": 0.046172}, "policy_guard": {"force_accept_reason": "current discounted utility dominates or nearly dominates the conservative post-rejection continuation"}}}


def test_v217_response_guards_reject_rating_catastrophe_and_use_nominal_fallback_only_when_unscored() -> None:
    game = _decision_game()
    strong_loss = {"forecasts_if_accepted": {"self": {"status": "available", "predicted_delta": -6.0, "interval_80": [-8.0, -1.0]}}}
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_decision_advisor(), bargaining_intervention_context=_context(response_forecast=strong_loss))
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "reject"}
    assert safeguards == ["bargaining_v217_response_rating_loss_guard"]

    unavailable = {"forecasts_if_accepted": {"self": {"status": "unavailable", "reason": "hidden-discount"}}}
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_decision_advisor(), bargaining_intervention_context=_context(response_forecast=unavailable))
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "reject"}
    assert safeguards == ["bargaining_v217_response_nominal_share_fallback"]

    final_game = _decision_game(round_number=10, max_rounds=10)
    terminal = SimpleNamespace(game=final_game, bargaining_advisor_context=_decision_advisor(), bargaining_intervention_context=_context(response_forecast=unavailable))
    action, safeguards = apply_bargaining_v217_intervention(terminal, {"decision": "accept"})
    assert action == {"decision": "accept"}
    assert safeguards == []


def test_v221_keeps_the_response_rating_loss_signal_but_removes_its_action_authority() -> None:
    game = _decision_game()
    strong_loss = {"forecasts_if_accepted": {"self": {"status": "available", "predicted_delta": -6.0, "interval_80": [-8.0, -1.0]}}}
    context = _context(response_forecast=strong_loss)
    context["live_policy"] = _live_policy(response_rating_loss_guard="shadow-only")
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_decision_advisor(), bargaining_intervention_context=context)
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "accept"}
    assert safeguards == []
    decision = SimpleNamespace(action=action, proposal={"decision": "accept"}, deterministic_safeguards=safeguards)
    receipt = intervention_receipt(envelope, decision)
    assert receipt is not None
    assert receipt["rating"]["response_guard"] == {"mode": "shadow-only", "signal": "strong-rating-loss", "would_reject_under_v2_20": True, "predicted_delta": -6.0, "interval_80_upper": -1.0, "catastrophe_threshold": -5.0}


def test_v222_rejects_only_a_low_share_acceptance_with_bounded_v3_loss_and_non_dominated_continuation() -> None:
    game = _decision_game(round_number=3)
    game["game_state"]["last_offer"].update({"player_1_gain": 65.0, "player_2_gain": 35.0})
    advisor = _decision_advisor()
    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.35, "reject_value": 0.369075}
    context = _context()
    context["live_policy"] = _live_policy(patient_guard="bidirectional-continuation-coherence", continuation_tolerance=0.05, response_rating_loss_guard="shadow-only", rating_v3_low_share_guard="bounded-authoritative")
    advisory = {"family": "bargaining", "branches": {"status": "available", "action_type": "decision", "current_offer": {"accept": {"status": "available", "predicted_self_rating_delta": -4.7258, "interval_80": [-7.7163, -1.9908]}}}}
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=advisor, bargaining_intervention_context=context, rating_v3_advisory=advisory)

    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})

    assert action == {"decision": "reject"}
    assert safeguards == ["bargaining_v3_low_share_rating_loss_guard"]
    decision = SimpleNamespace(action=action, proposal={"decision": "accept"}, deterministic_safeguards=safeguards)
    receipt = intervention_receipt(envelope, decision)
    assert receipt is not None
    assert receipt["interventions"] == ["bargaining_v3_low_share_rating_loss_guard"]
    assert receipt["rating"]["rating_v3_low_share_guard"]["would_reject"] is True

    advisory["branches"]["current_offer"]["accept"]["interval_80"] = [-4.1395, 1.5861]
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "accept"}
    assert safeguards == []

    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.42, "reject_value": 0.348747}
    game["game_state"]["last_offer"].update({"player_1_gain": 58.0, "player_2_gain": 42.0})
    advisory["branches"]["current_offer"]["accept"]["interval_80"] = [-7.0, -1.0]
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "accept"}
    assert safeguards == []


def test_live_policy_can_accept_when_rejection_only_reproduces_a_no_better_next_settlement() -> None:
    game = _decision_game()
    advisor = _decision_advisor()
    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.416254, "reject_value": 0.320918}
    advisor["behavioral_continuation"]["consistency_check"] = {"modeled_next_offer_same_or_worse_for_us": True}
    context = _context()
    context["live_policy"] = _live_policy(patient_guard="accept-if-next-settlement-no-better")
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=advisor, bargaining_intervention_context=context)
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "reject"})
    assert action == {"decision": "accept"}
    assert safeguards == ["bargaining_live_policy_patient_no_better_next_settlement_guard"]


def test_live_policy_can_reject_a_materially_dominated_acceptance_without_rejecting_a_marginal_one() -> None:
    game = _decision_game(round_number=3)
    game["game_state"]["last_offer"].update({"player_1_gain": 65.0, "player_2_gain": 35.0})
    advisor = _decision_advisor()
    context = _context()
    context["live_policy"] = _live_policy(patient_guard="bidirectional-continuation-coherence")
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=advisor, bargaining_intervention_context=context)

    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.0448, "reject_value": 0.124499}
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "reject"}
    assert safeguards == ["bargaining_live_policy_materially_dominated_acceptance_guard"]

    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.35, "reject_value": 0.369075}
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "accept"})
    assert action == {"decision": "accept"}
    assert safeguards == []


def test_v219_continuation_margin_preserves_a_rejection_for_a_submaterial_override() -> None:
    game = _decision_game(round_number=10, max_rounds=12)
    game["game_state"]["last_offer"].update({"player_1_gain": 70.1199, "player_2_gain": 29.8801})
    advisor = _decision_advisor()
    advisor["behavioral_continuation"]["decision_comparison"] = {"accept_now_value": 0.298801, "reject_value": 0.279367}
    advisor["behavioral_continuation"]["consistency_check"] = {"modeled_next_offer_same_or_worse_for_us": True}
    context = _context()
    context["live_policy"] = _live_policy(patient_guard="bidirectional-continuation-coherence", continuation_tolerance=0.02)
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=advisor, bargaining_intervention_context=context)
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"decision": "reject"})
    assert action == {"decision": "reject"}
    assert safeguards == []


def test_v219_nonterminal_floor_rewrites_a_post_rejection_offer_below_25_percent() -> None:
    game = _offer_game(rejected_opponent_share=1 / 3)
    context = _context()
    context["live_policy"] = _live_policy(minimum_nonterminal_own_share=0.25)
    envelope = SimpleNamespace(game=game, bargaining_advisor_context=_advisor_context(), bargaining_intervention_context=context)
    action, safeguards = apply_bargaining_v217_intervention(envelope, {"alice_gain": 22.62190625, "bob_gain": 77.37809375})
    assert action == {"alice_gain": pytest.approx(60.0), "bob_gain": pytest.approx(40.0)}
    assert safeguards == ["bargaining_v217_post_rejection_capitulation_guard"]
