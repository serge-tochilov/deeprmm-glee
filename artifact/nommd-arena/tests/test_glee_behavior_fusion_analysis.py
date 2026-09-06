from __future__ import annotations

import math

import pytest

from nommd_arena.glee_behavior_channel_analysis import UNKNOWN_ID
from nommd_arena.glee_behavior_fusion_analysis import FusionExample, assignment_parity, fit_conditional_stacker, reconstruct_assignment_probabilities


def test_reconstructed_independent_marginal_preserves_unknown_prior_and_player_mass() -> None:
    events = [
        {"event_id": "e-a", "public_player_id": "alpha", "games_delta": 1},
        {"event_id": "e-b", "public_player_id": "beta", "games_delta": 1},
    ]
    candidates = [
        {
            "game_id": "g1",
            "unknown_probability_prior": 0.1,
            "candidates": [
                {"event_id": "e-a", "activity_utility": 1.0, "rating_utility": 0.0},
                {"event_id": "e-b", "activity_utility": 0.0, "rating_utility": 0.0},
            ],
        }
    ]

    probabilities, summary = reconstruct_assignment_probabilities(candidates, events, model="activity-only", samples=24, temperature=1.0, frontier_sequence=1, auction_epsilon=0.02)

    assert summary["sampled_conflict_components"] == 0
    assert sum(probabilities["g1"].values()) == pytest.approx(1.0)
    assert probabilities["g1"][UNKNOWN_ID] == pytest.approx(0.1)
    assert probabilities["g1"]["alpha"] > probabilities["g1"]["beta"]


def test_assignment_parity_checks_compact_top_rows() -> None:
    reconstructed = {"g1": {"alpha": 0.6, "beta": 0.3, UNKNOWN_ID: 0.1}}
    expected = [{"game_id": "g1", "unknown_assignment_probability": 0.1, "top_public_players": [{"public_player_id": "alpha", "probability": 0.6}, {"public_player_id": "beta", "probability": 0.3}]}]

    parity = assignment_parity(reconstructed, expected)

    assert parity["rows"] == 1
    assert parity["mismatches"] == 0
    assert parity["label_mismatches"] == 0
    assert parity["probability_mismatches"] == 0
    assert parity["maximum_rounded_probability_error"] == 0.0


def test_assignment_parity_accepts_only_six_decimal_serialization_noise() -> None:
    expected = [{"game_id": "g1", "unknown_assignment_probability": 0.1, "top_public_players": [{"public_player_id": "alpha", "probability": 0.6}, {"public_player_id": "beta", "probability": 0.3}]}]

    compatible = assignment_parity({"g1": {"alpha": 0.600009, "beta": 0.299991, UNKNOWN_ID: 0.1}}, expected)
    changed = assignment_parity({"g1": {"alpha": 0.600011, "beta": 0.299989, UNKNOWN_ID: 0.1}}, expected)

    assert compatible["mismatches"] == 0
    assert compatible["strict_probability_differences"] == 1
    assert changed["probability_mismatches"] == 1
    assert changed["mismatches"] == 1


def _example(index: int, target: str) -> FusionExample:
    labels = ("alpha", "beta", UNKNOWN_ID)
    activity = {"alpha": 2.0 if target == "alpha" else -1.0, "beta": 2.0 if target == "beta" else -1.0, UNKNOWN_ID: 2.0 if target == UNKNOWN_ID else -1.0}
    features = {label: {"activity": activity[label], "rating": 0.0, "timing": 0.0, "action": 0.0, "lexical": 0.0, "discourse": 0.0} for label in labels}
    raw = {model: {label: 1.0 / 3.0 for label in labels} for model in ("activity-only", "rating-only", "joint")}
    return FusionExample(f"g{index}", "bargaining", "none", "calibration", f"2026-08-12T00:{index:02d}:00+00:00", labels, target, target if target != UNKNOWN_ID else "new-id", features, raw, {channel: 1 for channel in ("timing", "action", "lexical", "discourse")})


def test_nonnegative_conditional_stacker_learns_candidate_specific_signal() -> None:
    examples = [_example(index, "alpha" if index % 2 == 0 else "beta") for index in range(20)] + [_example(20, UNKNOWN_ID), _example(21, UNKNOWN_ID)]

    stacker = fit_conditional_stacker(examples, feature_names=("activity",), iterations=150)

    assert stacker.weights["activity"] > 0.0
    assert max(stacker.probabilities(_example(22, "alpha")), key=stacker.probabilities(_example(22, "alpha")).get) == "alpha"
    assert math.isfinite(stacker.unknown_bias)
