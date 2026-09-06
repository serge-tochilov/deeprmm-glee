from __future__ import annotations

import polars as pl
import pytest

from glee_sequence_lab.analysis import _bootstrap_delta, _metrics


def test_metrics_exclude_constant_proposal_labels_from_action_score() -> None:
    frame = pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "account_key": [None, None],
            "action_scored": [False, True],
            "actual_action": [0, 1],
            "predicted_action": [None, 1],
            "action_negative_log_likelihood": [None, 0.2],
            "actual_value": [0.7, None],
            "predicted_value": [0.6, None],
            "value_negative_log_likelihood": [-0.5, None],
            "actual_delay": [None, 7.0],
            "predicted_delay": [None, 6.5],
            "delay_negative_log_likelihood": [None, 1.2],
        }
    )
    metrics = _metrics(frame)
    assert metrics["targets"] == 2
    assert metrics["action"]["rows"] == 1
    assert metrics["action"]["negative_log_likelihood"] == 0.2
    assert metrics["value"]["mean_absolute_error"] == pytest.approx(0.1)
    assert "message" not in metrics


def test_game_cluster_bootstrap_preserves_direction() -> None:
    frame = pl.DataFrame({"game_id": ["g1", "g1", "g2"], "delta": [-0.2, -0.1, -0.4]})
    result = _bootstrap_delta(frame, replicates=100, seed=9)
    assert result["micro_mean"] < 0
    assert result["micro_favorable_fraction"] == 1.0
    assert result["game_macro_favorable_fraction"] == 1.0
