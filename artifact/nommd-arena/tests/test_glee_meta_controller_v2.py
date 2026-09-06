from __future__ import annotations

import copy

import pytest

from nommd_arena.glee_meta_controller_v2 import CONDITIONAL_MODEL_CARD_CONTRACT, CONDITIONAL_SELECTOR_PAYLOAD_CONTRACT, FAMILY_CANDIDATE_EVIDENCE_CONTRACT, GuardedSingleAction, INDEPENDENT_PLANNER_PAYLOAD_CONTRACT, POLICY_MARGINAL_FORECAST_CONTRACT, PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT, PUBLIC_SELF_MIRROR_AUTHORITY, PUBLIC_SELF_MIRROR_SURFACE_CONTRACT, SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT, build_conditional_surface, build_family_candidate_evidence, build_planner_payload, build_planner_payload_v15, build_public_self_mirror_admissibility, build_selector_payload, build_selector_payload_v15, conditional_model_card, freeze_planner_candidates, policy_marginal_forecast_from_shadow, select_frozen_candidate, selector_authority_contract, validate_public_self_mirror_surface
from nommd_arena.glee_nommd import nommd_candidate_plan_model, nommd_candidate_selection_model
from nommd_arena.glee_policy import action_model


def _game() -> dict[str, object]:
    return {
        "game_id": "two-call-test",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": {}, "bob_gain": {}, "message": {}}},
        "game_state": {"round": 3, "money_to_divide": 100.0, "current_player": "player_1", "complete_information": True, "horizon_known": False, "messages_allowed": True, "history": []},
    }


def _marginal() -> dict[str, object]:
    return {"contract": POLICY_MARGINAL_FORECAST_CONTRACT, "frontier": "authenticated-prefix-with-one-unobserved-self-action-bridge", "authority": "advisory-prior-only", "family": "bargaining", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.4, 0.55, 0.05]}


def _plan(game: dict[str, object]):
    model = nommd_candidate_plan_model(action_model(game))
    return model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 60.0, "bob_gain": 40.0, "message": "A clean settlement."}, "purpose": "supported settlement anchor"},
                {"action": {"alice_gain": 70.0, "bob_gain": 30.0, "message": "This is my firm split."}, "purpose": "bounded resolve test"},
            ]
        }
    )


def test_two_call_transport_freezes_candidates_and_selects_only_by_index() -> None:
    game = _game()
    worker = {"turn_receipt": {"turn_id": "turn-1"}, "game_family": "bargaining", "valid_actions": game["valid_actions"]}
    planner_payload = build_planner_payload(worker_payload=worker, policy_marginal_forecast=_marginal())
    assert planner_payload["stage"] == "candidate-planning"
    assert planner_payload["conditional_frontier"] == {"family": "bargaining", "action_type": "offer", "target": "direct opponent response after the candidate"}
    assert planner_payload["candidate_response_model_card"]["contract"] == CONDITIONAL_MODEL_CARD_CONTRACT
    assert planner_payload["candidate_response_model_card"]["authority"] == "live-advisory-candidate-response-evidence"
    assert "candidate_set" not in planner_payload
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    assert len(frozen.candidates) == 2
    assert frozen.candidates[0].action["alice_gain"] == 60.0
    with pytest.raises(TypeError):
        frozen.candidates[0].action["alice_gain"] = 0.0
    forecasts = [
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.8, 0.19, 0.01]},
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.49, 0.01]},
    ]
    surface = build_conditional_surface(candidate_set=frozen, forecasts=forecasts)
    selector_payload = build_selector_payload(worker_payload=worker, policy_marginal_forecast=_marginal(), candidate_set=frozen, conditional_surface=surface)
    assert selector_payload["candidate_set"]["candidate_set_sha256"] == frozen.candidate_set_sha256
    assert selector_payload["candidate_response_model_card"]["family"] == "bargaining"
    assert selector_payload["selector_authority_contract"]["contract"] == SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT
    selector = nommd_candidate_selection_model().model_validate({"candidate_index": 1})
    selected = select_frozen_candidate(candidate_set=frozen, parsed=selector)
    assert selected.candidate.action == frozen.candidates[1].action
    assert selected.candidate.action["alice_gain"] == 70.0


def test_one_and_half_round_planner_withholds_every_new_learned_model_artifact() -> None:
    game = _game()
    worker = {"turn_receipt": {"turn_id": "turn-1"}, "game_family": "bargaining", "valid_actions": game["valid_actions"], "opponent_statistical_decision_forecast": {"contract": "established-family-advisor-v1", "action": "reject"}}
    planner_payload = build_planner_payload_v15(worker_payload=worker)
    assert planner_payload["contract"] == INDEPENDENT_PLANNER_PAYLOAD_CONTRACT
    assert planner_payload["stage"] == "independent-candidate-planning"
    assert planner_payload["authenticated_turn"] == worker
    assert planner_payload["learned_model_boundary"]["new_learned_model_inputs"] == []
    assert "policy_marginal_opponent_forecast" not in planner_payload
    assert "candidate_response_model_card" not in planner_payload
    assert "conditional_opponent_response_surface" not in planner_payload
    leaked = dict(worker)
    leaked["policy_marginal_opponent_forecast"] = _marginal()
    with pytest.raises(ValueError, match="new learned-model artifact"):
        build_planner_payload_v15(worker_payload=leaked)


def test_one_and_half_round_selector_uses_only_committed_candidates_and_conditional_surface() -> None:
    game = _game()
    worker = {"turn_receipt": {"turn_id": "turn-1"}, "game_family": "bargaining", "valid_actions": game["valid_actions"]}
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    forecasts = [
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.8, 0.19, 0.01]},
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.49, 0.01]},
    ]
    surface = build_conditional_surface(candidate_set=frozen, forecasts=forecasts)
    selector_payload = build_selector_payload_v15(worker_payload=worker, candidate_set=frozen, conditional_surface=surface)
    assert selector_payload["contract"] == CONDITIONAL_SELECTOR_PAYLOAD_CONTRACT
    assert selector_payload["stage"] == "conditional-candidate-selection"
    assert selector_payload["candidate_set"]["candidate_set_sha256"] == frozen.candidate_set_sha256
    assert selector_payload["selector_authority_contract"]["contract"] == SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT
    assert selector_payload["selector_authority_contract"]["candidate_standing"] == "symmetric-after-hard-controls"
    assert selector_payload["selector_authority_contract"]["deterministic_candidate"].startswith("execution fallback if staged selection fails")
    assert selector_payload["learned_model_boundary"] == {"policy_marginal_forecast_visible": False, "candidate_set_committed_before_conditional_forecast": True, "selector_may_only_return_candidate_index": True}
    assert "policy_marginal_opponent_forecast" not in selector_payload


def test_one_and_half_round_selector_aligns_established_family_evidence_without_leaking_it_to_planner() -> None:
    game = _game()
    worker = {"turn_receipt": {"turn_id": "turn-1"}, "game_family": "bargaining", "valid_actions": game["valid_actions"]}
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    forecasts = [
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.8, 0.19, 0.01]},
        {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.49, 0.01]},
    ]
    conditional = build_conditional_surface(candidate_set=frozen, forecasts=forecasts)
    family = build_family_candidate_evidence(candidate_set=frozen, evidence=[{"bounded_expected_value": 0.48}, {"bounded_expected_value": 0.41}])
    selector = build_selector_payload_v15(worker_payload=worker, candidate_set=frozen, conditional_surface=conditional, family_candidate_evidence=family)
    assert selector["family_candidate_decision_evidence"]["contract"] == FAMILY_CANDIDATE_EVIDENCE_CONTRACT
    assert [row["candidate_index"] for row in selector["family_candidate_decision_evidence"]["rows"]] == [0, 1]
    leaked = dict(worker)
    leaked["family_candidate_decision_evidence"] = family
    with pytest.raises(ValueError, match="new learned-model artifact"):
        build_planner_payload_v15(worker_payload=leaked)


def test_public_self_mirror_is_selector_only_and_economically_bounded() -> None:
    game = _game()
    worker = {"turn_receipt": {"turn_id": "turn-mirror"}, "game_family": "bargaining", "valid_actions": game["valid_actions"]}
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    conditional = build_conditional_surface(candidate_set=frozen, forecasts=[{"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.6, 0.35, 0.05]} for _candidate in frozen.candidates])
    mirror = {
        "contract": PUBLIC_SELF_MIRROR_SURFACE_CONTRACT,
        "service_contract": "glee-public-self-mirror-live-service-v1",
        "authority": PUBLIC_SELF_MIRROR_AUTHORITY,
        "release_id": "self-v1",
        "turn_id": "turn-mirror",
        "game_id": "two-call-test",
        "family": "bargaining",
        "candidate_set_sha256": frozen.candidate_set_sha256,
        "rows": [
            {"candidate_index": frozen.candidates[0].index, "action_sha256": frozen.candidates[0].action_sha256, "forecast": {"ensemble_log_expectedness": -0.2, "relative_expectedness": 0.8, "expectedness_rank": 1, "expectedness_percentile": 1.0, "component_log_expectedness": {"1729": -0.1, "2718": -0.3}, "component_log_score_stddev": 0.1, "component_top_choice_disagreement": False, "population_prediction": True, "account_prediction": None, "message_wording_scored": False}},
            {"candidate_index": frozen.candidates[1].index, "action_sha256": frozen.candidates[1].action_sha256, "forecast": {"ensemble_log_expectedness": -1.6, "relative_expectedness": 0.2, "expectedness_rank": 2, "expectedness_percentile": 0.0, "component_log_expectedness": {"1729": -1.5, "2718": -1.7}, "component_log_score_stddev": 0.1, "component_top_choice_disagreement": False, "population_prediction": True, "account_prediction": None, "message_wording_scored": False}},
        ],
    }
    admissibility = build_public_self_mirror_admissibility(candidate_set=frozen, utility_values=[0.50, 0.495], absolute_regret_cap=0.01, value_units="normalized expected own payoff", evidence_source="test")
    assert admissibility["contract"] == PUBLIC_SELF_MIRROR_ADMISSIBILITY_CONTRACT
    assert admissibility["enabled"] is True
    selector = build_selector_payload_v15(worker_payload=worker, candidate_set=frozen, conditional_surface=conditional, public_self_mirror_surface=mirror, public_self_mirror_admissibility=admissibility)
    assert selector["public_self_mirror_candidate_surface"]["rows"][1]["forecast"]["expectedness_rank"] == 2
    assert selector["public_self_mirror_economic_admissibility"]["enabled"] is True
    assert selector["selector_authority_contract"]["public_self_mirror"]["outside_admissible_set"].startswith("Ignore")
    leaked = dict(worker)
    leaked["public_self_mirror_candidate_surface"] = mirror
    with pytest.raises(ValueError, match="new learned-model artifact"):
        build_planner_payload_v15(worker_payload=leaked)


def test_public_self_mirror_has_no_directional_authority_without_two_near_equivalent_candidates() -> None:
    game = _game()
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    admissibility = build_public_self_mirror_admissibility(candidate_set=frozen, utility_values=[0.50, 0.40], absolute_regret_cap=0.01, value_units="normalized expected own payoff", evidence_source="test")
    assert admissibility["enabled"] is False
    assert not any(row["eligible_for_public_expectedness"] for row in admissibility["rows"])


def test_public_self_mirror_tied_scores_cannot_encode_candidate_position() -> None:
    game = _game()
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    rows = [
        {"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"ensemble_log_expectedness": -0.5, "relative_expectedness": 0.5, "expectedness_rank": 1, "expectedness_percentile": 0.5, "component_log_expectedness": {"1729": -0.5, "2718": -0.5}, "component_log_score_stddev": 0.0, "component_top_choice_disagreement": False, "population_prediction": True, "account_prediction": None, "message_wording_scored": False}}
        for candidate in frozen.candidates
    ]
    mirror = {"contract": PUBLIC_SELF_MIRROR_SURFACE_CONTRACT, "authority": PUBLIC_SELF_MIRROR_AUTHORITY, "family": "bargaining", "candidate_set_sha256": frozen.candidate_set_sha256, "rows": rows}
    assert validate_public_self_mirror_surface(candidate_set=frozen, value=mirror)["rows"] == rows
    positional = copy.deepcopy(mirror)
    positional["rows"][1]["forecast"]["expectedness_rank"] = 2
    positional["rows"][1]["forecast"]["expectedness_percentile"] = 0.0
    with pytest.raises(ValueError, match="candidate position"):
        validate_public_self_mirror_surface(candidate_set=frozen, value=positional)


def test_candidate_guard_runs_before_hashing_and_conditional_alignment() -> None:
    game = _game()

    def guard(action: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        guarded = dict(action)
        if guarded["alice_gain"] == 70.0:
            guarded.update({"alice_gain": 65.0, "bob_gain": 35.0})
            return guarded, ["test-bound"]
        return guarded, []

    frozen = freeze_planner_candidates(game=game, parsed=_plan(game), guard=guard)
    assert frozen.candidates[1].action["alice_gain"] == 65.0
    assert frozen.candidates[1].safeguards == ("test-bound",)
    forecasts = [{"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.49, 0.01]} for _candidate in frozen.candidates]
    surface = build_conditional_surface(candidate_set=frozen, forecasts=forecasts)
    tampered = copy.deepcopy(surface)
    tampered["rows"][1]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="alignment"):
        build_selector_payload(worker_payload={"turn": 1, "game_family": "bargaining", "valid_actions": game["valid_actions"]}, policy_marginal_forecast=_marginal(), candidate_set=frozen, conditional_surface=tampered)


def test_duplicate_candidates_fail_when_fewer_than_two_unique_actions_survive() -> None:
    game = _game()
    model = nommd_candidate_plan_model(action_model(game))
    parsed = model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 50, "bob_gain": 50, "message": " same "}, "purpose": "first"},
                {"action": {"alice_gain": 50.0, "bob_gain": 50.0, "message": "same"}, "purpose": "duplicate"},
            ]
        }
    )
    with pytest.raises(ValueError, match="fewer than 2 unique"):
        freeze_planner_candidates(game=game, parsed=parsed)


def test_required_seed_duplicated_by_every_planner_candidate_becomes_a_deliberate_single_action() -> None:
    game = _game()
    model = nommd_candidate_plan_model(action_model(game))
    parsed = model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 50, "bob_gain": 50, "message": "same"}, "purpose": "planner duplicate one"},
                {"action": {"alice_gain": 50.0, "bob_gain": 50.0, "message": " same "}, "purpose": "planner duplicate two"},
            ]
        }
    )

    with pytest.raises(GuardedSingleAction) as raised:
        freeze_planner_candidates(game=game, parsed=parsed, guard=lambda action: (action, []), required_candidates=[{"action": {"alice_gain": 50.0, "bob_gain": 50.0, "message": "same"}, "purpose": "required deterministic seed"}])

    assert raised.value.receipt["pre_guard_candidate_count"] == 1
    assert [candidate.index for candidate in raised.value.candidate_set.candidates] == [0]
    assert raised.value.candidate_set.candidates[0].purpose == "required deterministic seed"


def test_duplicate_candidates_are_removed_and_survivors_are_reindexed() -> None:
    game = _game()
    model = nommd_candidate_plan_model(action_model(game))
    parsed = model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 50, "bob_gain": 50, "message": "same"}, "purpose": "first"},
                {"action": {"alice_gain": 50.0, "bob_gain": 50.0, "message": " same "}, "purpose": "duplicate"},
                {"action": {"alice_gain": 60, "bob_gain": 40, "message": "distinct"}, "purpose": "second survivor"},
            ]
        }
    )
    frozen = freeze_planner_candidates(game=game, parsed=parsed)
    assert [candidate.index for candidate in frozen.candidates] == [0, 1]
    assert [candidate.purpose for candidate in frozen.candidates] == ["first", "second survivor"]
    assert [candidate.action["alice_gain"] for candidate in frozen.candidates] == [50.0, 60.0]


def test_distinct_raw_candidates_collapsed_by_hard_guards_become_a_deliberate_single_action() -> None:
    game = _game()
    model = nommd_candidate_plan_model(action_model(game))
    parsed = model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 60, "bob_gain": 40, "message": "probe"}, "purpose": "probe"},
                {"action": {"alice_gain": 70, "bob_gain": 30, "message": "settle"}, "purpose": "settle"},
            ]
        }
    )

    def guard(_action: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        return {"alice_gain": 55.0, "bob_gain": 45.0, "message": "one defensible action"}, ["test-hard-boundary"]

    with pytest.raises(GuardedSingleAction) as captured:
        freeze_planner_candidates(game=game, parsed=parsed, guard=guard)
    assert captured.value.candidate_set.candidates[0].action == {"alice_gain": 55.0, "bob_gain": 45.0, "message": "one defensible action"}
    assert captured.value.candidate_set.candidates[0].safeguards == ("test-hard-boundary",)


def test_same_numeric_move_with_materially_different_wording_remains_distinct_after_guarding() -> None:
    game = _game()
    model = nommd_candidate_plan_model(action_model(game))
    parsed = model.model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 60, "bob_gain": 40, "message": "A reciprocal settlement."}, "purpose": "reciprocity frame"},
                {"action": {"alice_gain": 70, "bob_gain": 30, "message": "This closes the matter."}, "purpose": "finality frame"},
            ]
        }
    )

    def guard(action: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        return {**action, "alice_gain": 55.0, "bob_gain": 45.0}, ["test-numeric-control"]

    frozen = freeze_planner_candidates(game=game, parsed=parsed, guard=guard)
    assert len(frozen.candidates) == 2
    assert [candidate.action["alice_gain"] for candidate in frozen.candidates] == [55.0, 55.0]
    assert [candidate.action["message"] for candidate in frozen.candidates] == ["A reciprocal settlement.", "This closes the matter."]
    assert frozen.candidates[0].action_sha256 != frozen.candidates[1].action_sha256


def test_selector_cannot_address_an_unprovided_candidate() -> None:
    game = _game()
    frozen = freeze_planner_candidates(game=game, parsed=_plan(game))
    selector = nommd_candidate_selection_model().model_validate({"candidate_index": 4})
    with pytest.raises(ValueError, match="supplied candidate"):
        select_frozen_candidate(candidate_set=frozen, parsed=selector)


def test_missing_bridge_shadow_is_explicitly_recontracted_as_marginal_prior() -> None:
    prior = policy_marginal_forecast_from_shadow(
        {
            "candidate_id": "v2",
            "candidate_manifest_sha256": "a" * 64,
            "family": "bargaining",
            "target_kind": "response",
            "forecast_frontier": "authenticated-visible-prefix-before-terra",
            "causal_bridge_event_count": 1,
            "labels": ["accept", "proposal", "reject", "walkaway"],
            "action_probabilities": [0.3, 0.2, 0.45, 0.05],
        }
    )
    assert prior["authority"] == "advisory-prior-only"
    assert prior["labels"] == ["accept", "reject", "walkaway"]
    assert prior["response_probabilities"] == pytest.approx([0.375, 0.5625, 0.0625])
    with pytest.raises(ValueError, match="one-missing-self-action"):
        policy_marginal_forecast_from_shadow({"forecast_frontier": "after-action", "causal_bridge_event_count": 0})


def test_family_model_cards_preserve_calibration_and_weak_negotiation_authority() -> None:
    bargaining = conditional_model_card("bargaining")
    negotiation = conditional_model_card("negotiation")
    persuasion = conditional_model_card("persuasion")
    assert bargaining["direct_response_target_count"] == 3613
    assert negotiation["direct_response_target_count"] == 1185
    assert "calibration-discounted but substantive" in str(negotiation["selector_use"])
    assert persuasion["direct_response_target_count"] == 12359
    assert all(card["authority"] == "live-advisory-candidate-response-evidence" for card in (bargaining, negotiation, persuasion))


def test_selector_authority_is_symmetric_but_preserves_family_hard_boundaries() -> None:
    contracts = {family: selector_authority_contract(family) for family in ("bargaining", "negotiation", "persuasion")}
    assert all(value["contract"] == SYMMETRIC_SELECTOR_AUTHORITY_CONTRACT for value in contracts.values())
    assert all(value["candidate_standing"] == "symmetric-after-hard-controls" for value in contracts.values())
    assert all("deterministic origin" in value["candidate_provenance"] for value in contracts.values())
    assert all("unquantified future value" in value["unsupported_reasons"] for value in contracts.values())
    assert all("legality" in value["hard_boundaries"] for value in contracts.values())
    assert "accepted own allocation" in contracts["bargaining"]["objective"]["immediate_value"]
    assert "own feasible settlement surplus" in contracts["negotiation"]["objective"]["immediate_value"]
    assert "current sale payoff" in contracts["persuasion"]["objective"]["immediate_value"]
    with pytest.raises(ValueError, match="unsupported selector-authority family"):
        selector_authority_contract("unknown")


def test_planner_payload_rejects_family_mismatch_and_unsupported_turns() -> None:
    worker = {"game_family": "bargaining", "valid_actions": {"type": "offer", "fields": {}}}
    wrong = dict(_marginal())
    wrong["family"] = "negotiation"
    with pytest.raises(ValueError, match="family does not match"):
        build_planner_payload(worker_payload=worker, policy_marginal_forecast=wrong)
    wrong_labels = dict(_marginal())
    wrong_labels["labels"] = ["buy", "pass"]
    wrong_labels["response_probabilities"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="labels do not match"):
        build_planner_payload(worker_payload=worker, policy_marginal_forecast=wrong_labels)
    unsupported = {"game_family": "bargaining", "valid_actions": {"type": "decision", "fields": {}}}
    with pytest.raises(ValueError, match="outside the direct-response conditional frontier"):
        build_planner_payload(worker_payload=unsupported, policy_marginal_forecast=_marginal())


def test_one_and_half_round_planner_accepts_the_complete_negotiation_decision_frontier() -> None:
    compound = {"game_family": "negotiation", "valid_actions": {"type": "decision", "fields": {"decision": "choice", "product_price": "number", "message": "string"}}}
    payload = build_planner_payload_v15(worker_payload=compound)
    assert payload["conditional_frontier"] == {"family": "negotiation", "action_type": "decision", "target": "direct opponent response after the candidate"}
    terminal_only = {"game_family": "negotiation", "valid_actions": {"type": "decision", "fields": {"decision": "choice"}}}
    terminal_payload = build_planner_payload_v15(worker_payload=terminal_only)
    assert terminal_payload["conditional_frontier"]["action_type"] == "decision"
    final_round = {"game_family": "negotiation", "game_state": {"horizon_known": True, "round": 10, "max_rounds": 10}, "valid_actions": {"type": "decision", "fields": {"decision": "choice", "product_price": "number"}}}
    final_payload = build_planner_payload_v15(worker_payload=final_round)
    assert final_payload["conditional_frontier"]["action_type"] == "decision"
    bargaining_decision = {"game_family": "bargaining", "valid_actions": {"type": "decision", "fields": {"decision": "choice"}}}
    with pytest.raises(ValueError, match="outside the direct-response conditional frontier"):
        build_planner_payload_v15(worker_payload=bargaining_decision)
