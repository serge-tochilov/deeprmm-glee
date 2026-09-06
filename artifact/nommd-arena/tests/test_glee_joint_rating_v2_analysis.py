from __future__ import annotations

import pytest

from nommd_arena.glee_joint_rating_v2_analysis import _apply_residual, _bootstrap_mae_difference, asymmetric_interval, cross_fit_fold, empirical_midrank, shrinkage_weight


def test_empirical_midrank_smooths_boundaries_and_ties() -> None:
    values = [1.0, 2.0, 2.0, 4.0]

    assert empirical_midrank(1.0, values) == pytest.approx((0.2, 4))
    assert empirical_midrank(2.0, values) == pytest.approx((0.5, 4))
    assert empirical_midrank(4.0, values) == pytest.approx((0.8, 4))


def test_empirical_midrank_leave_one_out_removes_one_tied_observation() -> None:
    assert empirical_midrank(2.0, [1.0, 2.0, 2.0, 4.0], remove_one=True) == pytest.approx((0.5, 3))
    assert empirical_midrank(2.0, [2.0], remove_one=True) == (None, 0)

    with pytest.raises(ValueError):
        empirical_midrank(3.0, [1.0, 2.0], remove_one=True)


def test_shrunken_residual_falls_back_for_an_unseen_configuration() -> None:
    table = {"known": {"mean": 6.0, "count": 2}}

    assert shrinkage_weight(2, 1.0) == pytest.approx(2.0 / 3.0)
    assert _apply_residual(1.0, "known", table, 1.0) == pytest.approx((5.0, 2, 4.0))
    assert _apply_residual(1.0, "unknown", table, 1.0) == (1.0, 0, 0.0)


def test_cross_fit_partition_keeps_both_perspectives_of_a_game_together() -> None:
    assert cross_fit_fold("game-17") == cross_fit_fold("game-17")
    assert 0 <= cross_fit_fold("game-17") < 5


def test_asymmetric_interval_uses_signed_residual_quantiles() -> None:
    interval = asymmetric_interval([-5.0, -2.0, 0.0, 1.0, 8.0])

    assert interval["lower_80"] < 0.0
    assert interval["upper_80"] > 0.0
    assert interval["lower_95"] <= interval["lower_80"]
    assert interval["upper_95"] >= interval["upper_80"]


def test_game_cluster_bootstrap_is_deterministic_and_preserves_row_weighting() -> None:
    rows = [
        {"game_id": "a", "actual_rating_delta": 1.0, "structural_prediction": 0.0, "hybrid_prediction": 1.0},
        {"game_id": "a", "actual_rating_delta": -1.0, "structural_prediction": 0.0, "hybrid_prediction": -1.0},
        {"game_id": "b", "actual_rating_delta": 2.0, "structural_prediction": 0.0, "hybrid_prediction": 1.0},
    ]

    first = _bootstrap_mae_difference(rows, replicates=100, seed=7)
    second = _bootstrap_mae_difference(rows, replicates=100, seed=7)

    assert first == second
    assert first["games"] == 2
    assert first["samples"] == 3
    assert first["point"] == pytest.approx(-1.0)
