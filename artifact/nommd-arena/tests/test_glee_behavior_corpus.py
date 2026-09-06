from __future__ import annotations

import json

import pytest

from nommd_arena.glee_behavior_corpus import _lexical_signature, extract_behavior_game


def _envelope(game_id: str, family: str) -> dict[str, object]:
    return {"game_id": game_id, "family": family, "exact_public_player_id": "public-opponent", "disclosed_label": "Twin", "resolution_status": "unique-at-assignment-frontier", "assignment_frontier_sequence": 10, "aligned_completion_frontier_sequence": 11, "started_at": "2026-08-12T00:00:00+00:00", "completed_at": "2026-08-12T00:01:00+00:00", "source_archive_path": f"run/games/{family}-{game_id}.json", "source_archive_sha256": "source-sha"}


def test_language_signature_keeps_style_and_hashes_without_copying_prose() -> None:
    message = "Look, accept $50.25 now before value shrinks!"

    signature = _lexical_signature(message, family_act="urgency")

    assert signature["present"] is True
    assert signature["style"]["currency_marks"] == 1
    assert signature["style"]["decimal_numbers"] == 1
    assert "urgency" in signature["discourse_acts"]
    assert message not in json.dumps(signature)
    assert "shrinks" not in json.dumps(signature)


def test_bargaining_extracts_only_opponent_actions_and_attributes_response_time() -> None:
    game = {
        "game_id": "b",
        "game_family": "bargaining",
        "your_player": "player_1",
        "game_state": {
            "money_to_divide": 100,
            "messages_allowed": True,
            "complete_information": True,
            "horizon_known": False,
            "history": [
                {"round": 1, "proposer": "player_2", "offer": {"round": 1, "proposer": "player_2", "player_1_gain": 35, "player_2_gain": 65, "message": "Firm opening."}, "decision": "reject", "response_time_ms": 900},
                {"round": 2, "proposer": "player_1", "offer": {"round": 2, "proposer": "player_1", "player_1_gain": 55, "player_2_gain": 45, "message": "Ours"}, "decision": "accept", "response_time_ms": 12_000},
            ],
        },
    }

    record = extract_behavior_game(_envelope("b", "bargaining"), game)

    assert [move["kind"] for move in record["moves"]] == ["proposal", "response"]
    assert record["moves"][0]["action_value"] == pytest.approx(0.65)
    assert record["moves"][0]["response_time_ms"] is None
    assert record["moves"][1]["action_value"] == pytest.approx(0.45)
    assert record["moves"][1]["response_time_ms"] == 12_000
    assert record["channel_counts"] == {"moves": 2, "timed_moves": 1, "message_opportunities": 1, "nonempty_messages": 1}


def test_negotiation_normalizes_complete_information_offer_to_opponent_surplus() -> None:
    game = {
        "game_id": "n",
        "game_family": "negotiation",
        "your_player": "player_1",
        "game_state": {
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 80,
            "player_2_value": 120,
            "complete_information": True,
            "messages_allowed": True,
            "horizon_known": True,
            "max_rounds": 5,
            "history": [{"round": 1, "offer": {"round": 1, "from_player": "player_1", "price": 100, "message": "Our offer"}, "decided_by": "player_2", "decision": "AcceptOffer", "response_time_ms": 8_000}],
        },
    }

    record = extract_behavior_game(_envelope("n", "negotiation"), game)

    assert record["moves"][0]["kind"] == "response"
    assert record["moves"][0]["action_value"] == pytest.approx(0.5)
    assert record["moves"][0]["response_time_ms"] == 8_000


def test_persuasion_attributes_language_only_when_opponent_is_seller() -> None:
    seller_game = {
        "game_id": "p-seller",
        "game_family": "persuasion",
        "your_player": "player_2",
        "game_state": {"player_1_role": "seller", "player_2_role": "buyer", "seller_message_type": "binary", "total_rounds": 2, "history": [{"round": 1, "seller_message": "yes", "buyer_decision": "no", "bought": False, "response_time_ms": 2_000}]},
    }
    buyer_game = {**seller_game, "game_id": "p-buyer", "your_player": "player_1"}

    seller_record = extract_behavior_game(_envelope("p-seller", "persuasion"), seller_game)
    buyer_record = extract_behavior_game(_envelope("p-buyer", "persuasion"), buyer_game)

    assert seller_record["moves"][0]["kind"] == "signal"
    assert seller_record["moves"][0]["language"]["present"] is True
    assert seller_record["moves"][0]["response_time_ms"] is None
    assert buyer_record["moves"][0]["kind"] == "response"
    assert buyer_record["moves"][0]["language"] is None
    assert buyer_record["moves"][0]["response_time_ms"] == 2_000
