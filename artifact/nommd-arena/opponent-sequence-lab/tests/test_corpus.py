from __future__ import annotations

import json
from pathlib import Path

from glee_sequence_lab.corpus import _round_phase, extract_events, load_account_map
from glee_sequence_lab.data import SequenceCollator


def test_bargaining_extracts_both_actors_without_prefix_duplication() -> None:
    game = {
        "game_id": "b-1",
        "game_family": "bargaining",
        "your_player": "player_1",
        "game_state": {
            "money_to_divide": 100,
            "messages_allowed": True,
            "horizon_known": True,
            "max_rounds": 4,
            "history": [
                {"round": 1, "proposer": "player_1", "offer": {"proposer": "player_1", "round": 1, "player_1_gain": 60, "player_2_gain": 40, "message": "Fair split"}, "decision": "reject", "response_time_ms": 1200},
                {"round": 2, "proposer": "player_2", "offer": {"proposer": "player_2", "round": 2, "player_1_gain": 45, "player_2_gain": 55, "message": "Accept 55"}, "decision": "accept", "response_time_ms": 900},
            ],
        },
    }
    events = extract_events(game)
    assert [(event["actor"], event["action_label"]) for event in events] == [("self", "proposal"), ("opponent", "reject"), ("opponent", "proposal"), ("self", "accept")]
    assert events[2]["action_value"] == 0.55
    assert all(event["event_index"] == index for index, event in enumerate(events))


def test_negotiation_uses_role_invariant_opponent_demand() -> None:
    game = {
        "game_id": "n-1",
        "game_family": "negotiation",
        "your_player": "player_2",
        "game_state": {
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 80,
            "player_2_value": 150,
            "complete_information": True,
            "messages_allowed": True,
            "horizon_known": True,
            "max_rounds": 5,
            "history": [
                {"round": 1, "offer": {"from_player": "player_1", "round": 1, "price": 120, "message": "Fair price"}, "decided_by": "player_2", "decision": "RejectOffer", "response_time_ms": 500},
                {"round": 2, "offer": {"from_player": "player_2", "round": 2, "price": 110, "message": "My offer"}, "decided_by": "player_1", "decision": "AcceptOffer", "response_time_ms": 700},
            ],
        },
    }
    events = extract_events(game)
    assert [(event["actor"], event["action_label"]) for event in events] == [("opponent", "proposal"), ("self", "reject"), ("self", "proposal"), ("opponent", "accept")]
    assert events[0]["action_aux_value"] is not None


def test_persuasion_does_not_reveal_passed_quality() -> None:
    game = {
        "game_id": "p-1",
        "game_family": "persuasion",
        "your_player": "player_2",
        "game_state": {
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "is_seller_know_cv": True,
            "seller_message_type": "text",
            "total_rounds": 2,
            "history": [
                {"round": 1, "seller_message": "This is good", "buyer_decision": "no", "bought": False, "quality": "high", "response_time_ms": 300},
                {"round": 2, "seller_message": "This is poor", "buyer_decision": "yes", "bought": True, "quality": "low", "response_time_ms": 400},
            ],
        },
    }
    events = extract_events(game)
    assert [(event["actor"], event["kind"]) for event in events] == [("opponent", "signal"), ("self", "response"), ("opponent", "signal"), ("self", "response"), ("environment", "quality_reveal")]
    assert events[0]["visible_quality"] is None
    assert events[-1]["visible_quality"] == "low"


def test_account_map_excludes_ambiguous_and_medium_links(tmp_path: Path) -> None:
    path = tmp_path / "groups.json"
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {"key": "alpha", "confidence": "high", "members": {"One": "id-1"}},
                    {"key": "beta", "confidence": "very-high", "members": {"ONE": "id-2", "Two": "id-3"}},
                    {"key": "gamma", "confidence": "medium", "members": {"Three": "id-4"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    mapping, collisions = load_account_map(path)
    assert "one" not in mapping
    assert mapping["two"] == ("beta", "very-high")
    assert "three" not in mapping
    assert collisions["one"] == ["alpha", "beta"]


def test_timeout_is_censored_without_erasing_prior_actions() -> None:
    game = {
        "game_id": "b-timeout",
        "game_family": "bargaining",
        "your_player": "player_1",
        "game_state": {
            "money_to_divide": 100,
            "messages_allowed": True,
            "horizon_known": False,
            "history": [
                {"round": 1, "proposer": "player_2", "offer": {"proposer": "player_2", "round": 1, "player_1_gain": 40, "player_2_gain": 60, "message": ""}, "decision": "reject", "response_time_ms": 100},
                {"round": 2, "proposer": "player_2", "offer": None, "decision": "timeout", "response_time_ms": None, "timed_out_by": "player_2"},
            ],
        },
    }
    events = extract_events(game)
    assert [(event["actor"], event["action_label"]) for event in events] == [("opponent", "proposal"), ("self", "reject")]


def test_unknown_horizon_cannot_reveal_terminal_round_count() -> None:
    short = {"horizon_known": False, "max_rounds": 9, "total_rounds": 3}
    long = {"horizon_known": False, "max_rounds": 99, "total_rounds": 40}
    assert _round_phase(2, short) == _round_phase(2, long)
    static = {"horizon_known": False, "max_rounds": 99, "complete_information": False, "messages_allowed": False}
    assert SequenceCollator._static_numeric(static)[3] == 0.0
