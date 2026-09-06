from __future__ import annotations

import pytest

from nommd_arena.glee_account_identity_analysis import UNKNOWN_ACCOUNT, classification_metrics, collapse_account_posterior
from nommd_arena.glee_behavior_channel_analysis import UNKNOWN_ID


def test_account_collapse_sums_sibling_agents_and_unlinked_mass() -> None:
    collapsed = collapse_account_posterior(
        {"alpha-1": 0.20, "alpha-2": 0.25, "beta": 0.15, "unlinked": 0.10, UNKNOWN_ID: 0.30},
        {"alpha-1": "alpha", "alpha-2": "alpha", "beta": "beta"},
    )

    assert collapsed == pytest.approx({"alpha": 0.45, "beta": 0.15, UNKNOWN_ACCOUNT: 0.40})
    assert sum(collapsed.values()) == pytest.approx(1.0)


def test_account_metrics_reward_correct_coarse_identity_without_hiding_open_set() -> None:
    rows = [
        {"true_account": "alpha", "account_probabilities": {"alpha": 0.60, "beta": 0.25, UNKNOWN_ACCOUNT: 0.15}},
        {"true_account": "beta", "account_probabilities": {"alpha": 0.45, "beta": 0.40, UNKNOWN_ACCOUNT: 0.15}},
        {"true_account": UNKNOWN_ACCOUNT, "account_probabilities": {"alpha": 0.20, "beta": 0.10, UNKNOWN_ACCOUNT: 0.70}},
    ]

    metrics = classification_metrics(rows, probability_key="account_probabilities", target_key="true_account")

    assert metrics["games"] == 3
    assert metrics["top_one_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["named_prediction_coverage"] == pytest.approx(2 / 3)
    assert metrics["named_prediction_precision"] == pytest.approx(0.5)
