from __future__ import annotations

from types import SimpleNamespace

from nommd_arena.glee_meta_controller_v2 import build_conditional_surface, build_family_candidate_evidence, build_selector_payload_v15, freeze_planner_candidates
from nommd_arena.glee_nommd import nommd_candidate_plan_model
from nommd_arena.glee_policy import action_model
from nommd_arena.glee_selector_backend import build_selector_backend_request, local_result

from glee_sequence_lab.selector_evaluation import MINIMUM_TURNS_PER_FAMILY, _evaluate_family


def _request():
    game = {
        "game_id": "prospective-selector",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}},
        "game_state": {"round": 3, "money_to_divide": 100.0, "current_player": "player_1", "history": []},
    }
    plan = nommd_candidate_plan_model(action_model(game)).model_validate(
        {
            "candidates": [
                {"action": {"alice_gain": 50.0, "bob_gain": 50.0, "message": "Fallback."}, "purpose": "fallback"},
                {"action": {"alice_gain": 60.0, "bob_gain": 40.0, "message": "Candidate."}, "purpose": "candidate"},
            ]
        }
    )
    candidates = freeze_planner_candidates(game=game, parsed=plan)
    surface = build_conditional_surface(
        candidate_set=candidates,
        forecasts=[
            {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.4, 0.6, 0.0]},
            {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.8, 0.2, 0.0]},
        ],
    )
    evidence = build_family_candidate_evidence(
        candidate_set=candidates,
        evidence=[
            {"behavioral_offer_evaluation": {"accepted_value": 0.5, "rejected_path_value": 0.1}},
            {"behavioral_offer_evaluation": {"accepted_value": 0.6, "rejected_path_value": 0.1}},
        ],
    )
    turn = {"turn_receipt": {"turn_id": "prospective-turn"}, "game_family": "bargaining", "valid_actions": game["valid_actions"]}
    payload = build_selector_payload_v15(worker_payload=turn, candidate_set=candidates, conditional_surface=surface, family_candidate_evidence=evidence)
    return build_selector_backend_request(candidate_set=candidates, selector_payload=payload, fallback_candidate_id=candidates.candidates[0].action_sha256, timeout_s=12.0)


def test_prospective_evaluator_requires_support_and_accepts_deterministic_admissible_choices() -> None:
    request = _request()
    selected_id = request.candidate_set.candidates[1].action_sha256

    class Policy:
        @staticmethod
        def __call__(wire):
            return local_result(request=wire, candidate_id=selected_id)

    example = SimpleNamespace(request=request, cloud_candidate_index=1, ts="2026-08-17T22:00:00Z")
    insufficient, _rows = _evaluate_family("bargaining", [example] * (MINIMUM_TURNS_PER_FAMILY - 1), Policy())
    sufficient, disagreements = _evaluate_family("bargaining", [example] * MINIMUM_TURNS_PER_FAMILY, Policy())
    assert insufficient["support_gate"] is False
    assert insufficient["safety_gate"] is False
    assert sufficient["support_gate"] is True
    assert sufficient["safety_gate"] is True
    assert sufficient["local_minus_fallback_proxy_minimum"] > 0
    assert disagreements == []
