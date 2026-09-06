from __future__ import annotations

import pytest

from nommd_arena.glee_joint_rating_analysis import ETA_SCHEDULES, _terminal_view, chronological_game_splits, display_to_raw, eta_for_game_count, implied_adjusted_percentile, predict_display_delta, raw_to_display


def test_display_rating_round_trip_and_structural_delta_round_trip() -> None:
    raw = 1875.0
    games = 900
    display = raw_to_display(raw, games)

    assert display_to_raw(display, games) == pytest.approx(raw)

    percentile = 0.72
    eta = 0.002
    delta = predict_display_delta(display, games, percentile, eta)
    recovered = implied_adjusted_percentile(display, display + delta, games, eta)

    assert recovered == pytest.approx(percentile)


def test_chronological_split_keeps_both_player_perspectives_together() -> None:
    samples = []
    for index in range(10):
        for scope in ("self", "opponent"):
            samples.append({"game_id": f"g{index}", "family": "bargaining", "completed_at_timestamp": float(index), "target_scope": scope})

    split = chronological_game_splits(samples, train_fraction=0.6, calibration_fraction=0.2)

    assert [split[f"g{index}"] for index in range(10)] == ["train"] * 6 + ["calibration"] * 2 + ["test"] * 2


def test_learning_rate_candidates_share_the_documented_high_count_floor() -> None:
    values_at_zero = {schedule["name"]: eta_for_game_count(schedule, 0) for schedule in ETA_SCHEDULES}
    values_at_large_count = {schedule["name"]: eta_for_game_count(schedule, 10_000) for schedule in ETA_SCHEDULES}

    assert values_at_zero["constant-0.002"] == pytest.approx(0.002)
    assert values_at_zero["piecewise-linear-120"] == pytest.approx(0.01)
    assert all(value == pytest.approx(0.002, abs=1e-5) for value in values_at_large_count.values())


def test_negotiation_terminal_view_recovers_missing_values_from_agreement() -> None:
    game = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "game_state": {
            "game_family": "negotiation",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 80.0,
            "complete_information": False,
            "horizon_known": False,
            "messages_allowed": True,
            "round": 3,
            "result": {"outcome": "agreement", "agreed_price": 118.0, "agreed_round": 3, "player_1_payoff": 38.0, "player_2_payoff": 2.0},
        },
        "result": {"outcome": "agreement", "agreed_price": 118.0, "agreed_round": 3, "player_1_payoff": 38.0, "player_2_payoff": 2.0},
    }

    seller = _terminal_view(game, "player_1")
    buyer = _terminal_view(game, "player_2")

    assert seller["base_features"]["negotiation_opponent_value_known"] == 1.0
    assert buyer["base_features"]["negotiation_own_value_ratio"] > 0.0
    assert seller["observed_configuration_sha256"] != buyer["observed_configuration_sha256"]


def test_terminal_view_preserves_persuasion_role_asymmetry() -> None:
    game = {
        "game_family": "persuasion",
        "your_player": "player_1",
        "game_state": {
            "game_family": "persuasion",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "product_price": 10.0,
            "p": 0.5,
            "u": 0.0,
            "v": 40.0,
            "total_rounds": 20,
            "is_seller_know_cv": True,
            "seller_message_type": "binary",
            "round": 20,
            "result": {"outcome": "completed", "rounds_played": 20, "rounds_total": 20, "player_1_payoff": 120.0, "player_2_payoff": 40.0},
        },
        "result": {"outcome": "completed", "rounds_played": 20, "rounds_total": 20, "player_1_payoff": 120.0, "player_2_payoff": 40.0},
    }

    seller = _terminal_view(game, "player_1")
    buyer = _terminal_view(game, "player_2")

    assert seller["role"] == "seller"
    assert buyer["role"] == "buyer"
    assert seller["base_features"]["target_seller"] == 1.0
    assert buyer["base_features"]["target_buyer"] == 1.0
    assert seller["own_payoff"] == 120.0
    assert buyer["own_payoff"] == 40.0
