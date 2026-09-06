from __future__ import annotations

import json

import pytest

from glee_sequence_lab.corpus import extract_events, object_sha256
from glee_sequence_lab.live_shadow import LIVE_SHADOW_CONTEXT_CONTRACT, PreTerraShadowRegistry, _warmup_opportunities, build_pre_terra_opportunity


def _bargaining_offer_turn() -> dict[str, object]:
    return {
        "game_id": "bargain-live",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "agent", "name": "Other"},
        "valid_actions": {"type": "offer", "fields": {}},
        "game_state": {"history": [], "round": 1, "money_to_divide": 100.0, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "delta_1": 0.9, "delta_2": 0.8},
    }


def _negotiation_offer_turn() -> dict[str, object]:
    return {
        "game_id": "negotiation-live",
        "game_family": "negotiation",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "hidden", "name": None},
        "valid_actions": {"type": "offer", "fields": {}},
        "game_state": {"history": [], "round": 1, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0},
    }


def _persuasion_seller_turn() -> dict[str, object]:
    return {
        "game_id": "persuasion-live",
        "game_family": "persuasion",
        "your_player": "player_1",
        "phase": "seller_message",
        "opponent": {"type": "agent", "name": "Buyer"},
        "valid_actions": {"type": "seller_message", "fields": {}},
        "game_state": {"history": [], "round": 1, "total_rounds": 4, "p": 0.6, "product_price": 10.0, "seller_message_type": "text", "is_seller_know_cv": True, "player_1_role": "seller", "player_2_role": "buyer"},
    }


@pytest.mark.parametrize("game_factory", [_bargaining_offer_turn, _negotiation_offer_turn, _persuasion_seller_turn])
def test_pre_terra_opportunity_targets_response_after_one_self_bridge(game_factory) -> None:
    opportunity = build_pre_terra_opportunity(game_factory(), turn_id="turn-1", synthetic_features={"contract": "features"})
    assert opportunity is not None
    assert opportunity.target_kind == "response"
    assert opportunity.target_event_index == len(opportunity.prefix_events) + 1
    assert opportunity.causal_bridge_event_count == 1
    assert opportunity.context["model_consumed_synthetic_features"] is False
    assert opportunity.context["synthetic_features_sha256"] == object_sha256({"contract": "features"})


def test_startup_warmup_covers_every_family_without_registry_evidence() -> None:
    opportunities = _warmup_opportunities()
    assert [opportunity.family for opportunity in opportunities] == ["bargaining", "negotiation", "persuasion"]
    assert all(opportunity.game_id.startswith("__shadow_warmup_") for opportunity in opportunities)


def test_projected_coordinate_matches_later_bargaining_history() -> None:
    game = _bargaining_offer_turn()
    opportunity = build_pre_terra_opportunity(game, turn_id="turn-1", synthetic_features={})
    assert opportunity is not None
    final = json.loads(json.dumps(game))
    final["game_state"]["history"] = [
        {"round": 1, "proposer": "player_1", "offer": {"round": 1, "proposer": "player_1", "player_1_gain": 60.0, "player_2_gain": 40.0, "message": "split"}, "decision": "reject", "response_time_ms": 1000}
    ]
    events = extract_events(final)
    assert events[opportunity.target_event_index]["actor"] == "opponent"
    assert events[opportunity.target_event_index]["kind"] == "response"
    assert events[opportunity.target_event_index]["action_label"] == "reject"


def test_projected_coordinate_matches_later_negotiation_history() -> None:
    game = _negotiation_offer_turn()
    opportunity = build_pre_terra_opportunity(game, turn_id="turn-1", synthetic_features={})
    assert opportunity is not None
    final = json.loads(json.dumps(game))
    final["game_state"]["history"] = [{"round": 1, "offer": {"round": 1, "from_player": "player_1", "price": 60.0, "message": "offer"}, "decided_by": "player_2", "decision": "RejectOffer", "response_time_ms": 1000}]
    events = extract_events(final)
    assert events[opportunity.target_event_index]["actor"] == "opponent"
    assert events[opportunity.target_event_index]["kind"] == "response"
    assert events[opportunity.target_event_index]["action_label"] == "reject"


def test_projected_coordinate_matches_later_persuasion_history() -> None:
    game = _persuasion_seller_turn()
    opportunity = build_pre_terra_opportunity(game, turn_id="turn-1", synthetic_features={})
    assert opportunity is not None
    final = json.loads(json.dumps(game))
    final["game_state"]["history"] = [{"round": 1, "seller_message": "yes", "bought": False, "buyer_decision": "no", "response_time_ms": 1000}]
    events = extract_events(final)
    assert events[opportunity.target_event_index]["actor"] == "opponent"
    assert events[opportunity.target_event_index]["kind"] == "response"
    assert events[opportunity.target_event_index]["action_label"] == "pass"


def test_partial_persuasion_history_does_not_invent_a_pass() -> None:
    game = _persuasion_seller_turn()
    game["game_state"]["history"] = [{"round": 1, "seller_message": "yes", "quality": "high", "bought": None}]
    events = extract_events(game)
    assert [event["kind"] for event in events] == ["signal"]


def test_registry_accepts_declared_bridge_and_never_overwrites(tmp_path) -> None:
    registry = PreTerraShadowRegistry(tmp_path / "shadow.sqlite3")
    prediction = {"candidate_id": "candidate", "game_id": "game", "target_event_index": 1, "target_kind": "response", "family": "bargaining", "authority": "prospective-shadow-only", "action_probabilities": [0.1, 0.8, 0.1]}
    context = {"contract": LIVE_SHADOW_CONTEXT_CONTRACT, "model_consumed_synthetic_features": False, "synthetic_features": {"x": 1}}
    first = registry.register(prediction, context=context, prefix_event_count=0, causal_bridge_event_count=1, prefix_sha256="a" * 64, source_turn_id="turn")
    second = registry.register(prediction, context=context, prefix_event_count=0, causal_bridge_event_count=1, prefix_sha256="a" * 64, source_turn_id="turn")
    assert first["status"] == "registered"
    assert second["status"] == "already-registered"
    with pytest.raises(ValueError, match="different immutable evidence"):
        registry.register({**prediction, "action_probabilities": [0.2, 0.7, 0.1]}, context=context, prefix_event_count=0, causal_bridge_event_count=1, prefix_sha256="a" * 64, source_turn_id="turn")
    outcome = {"actual_action": "reject", "event": {"actor": "opponent", "kind": "response"}}
    assert registry.record_outcome(candidate_id="candidate", game_id="game", target_event_index=1, target_kind="response", outcome=outcome)["status"] == "recorded"
    assert registry.record_outcome(candidate_id="candidate", game_id="game", target_event_index=1, target_kind="response", outcome=outcome)["status"] == "already-recorded"
    assert registry.summary()["pending"] == 0
    registry.close()


def test_registry_rejects_undeclared_gap(tmp_path) -> None:
    registry = PreTerraShadowRegistry(tmp_path / "shadow.sqlite3")
    prediction = {"candidate_id": "candidate", "game_id": "game", "target_event_index": 2, "target_kind": "response", "family": "bargaining", "authority": "prospective-shadow-only"}
    context = {"contract": LIVE_SHADOW_CONTEXT_CONTRACT, "model_consumed_synthetic_features": False}
    with pytest.raises(ValueError, match="declare every unobserved bridge"):
        registry.register(prediction, context=context, prefix_event_count=0, causal_bridge_event_count=1, prefix_sha256="a" * 64, source_turn_id="turn")
    registry.close()
