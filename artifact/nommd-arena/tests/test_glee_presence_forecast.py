from __future__ import annotations

import pytest

from nommd_arena.glee_presence_forecast import AdditiveHazardTrainer, BinaryAccumulator, EMPTY_SERIES, EventSeries, _normalized_identity, causal_state, fit_platt


def test_event_series_uses_strictly_future_half_open_horizon() -> None:
    series = EventSeries.build([(100.0, 1), (160.0, 2), (161.0, 1)])

    assert series.index_at(100.0) == 1
    assert series.future_event(100.0, 59) is False
    assert series.future_event(100.0, 60) is True
    assert series.additions_between(100.0, 160.0) == 2
    assert series.pulses_between(100.0, 161.0) == 2


def test_causal_state_excludes_later_events() -> None:
    series = EventSeries.build([(100.0, 2), (200.0, 3)])

    state = causal_state(player_id="p", stamp=150.0, first_seen_at=0.0, series=series, other_series=(EMPTY_SERIES,), family_traffic_60=4, global_traffic_60=7)

    assert state.prior_additions == 2
    assert state.regime == "canary-like"
    assert state.features["recent_60"] == "le-2"
    assert state.features["family_traffic"] == "le-5"


def test_additive_hazard_model_learns_positive_category_without_negative_weights() -> None:
    trainer = AdditiveHazardTrainer(("signal",))
    for _ in range(400):
        trainer.update({"signal": "hot"}, True)
        trainer.update({"signal": "cold"}, False)

    model = trainer.build()

    assert model.raw_logit({"signal": "hot"}) > model.raw_logit({"signal": "cold"})
    assert model.training_rows == 800


def test_platt_calibration_remains_monotone() -> None:
    scores = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] * 20
    targets = [0, 0, 0, 1, 1, 1] * 20

    calibrator = fit_platt(scores, targets)

    assert calibrator.slope >= 0.0
    assert calibrator.probability(-1.0) < calibrator.probability(1.0)


def test_platt_calibration_backtracks_instead_of_accepting_newton_overshoot() -> None:
    scores = [-0.08] * 700 + [-0.07] * 300
    targets = [0] * 700 + [1] * 300

    calibrator = fit_platt(scores, targets)

    assert 0.0 < calibrator.slope < 10_000.0
    assert abs(calibrator.intercept) < 1_000.0
    assert 0.0 < calibrator.probability(-0.08) < calibrator.probability(-0.07) < 1.0


def test_binary_metrics_and_identity_normalization() -> None:
    accumulator = BinaryAccumulator()
    accumulator.add(0.9, True)
    accumulator.add(0.1, False)
    metrics = accumulator.metrics()
    probabilities = _normalized_identity({"a": 2.0, "b": 1.0})

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["a"] == pytest.approx(2.0 * probabilities["b"])
    assert probabilities["__unknown__"] == pytest.approx(0.05)
