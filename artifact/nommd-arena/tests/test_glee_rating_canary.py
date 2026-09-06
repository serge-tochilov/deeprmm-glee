from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nommd_arena.glee_activity_eda import _file_digest
from nommd_arena.glee_rating_canary import BargainingRatingCanary, BargainingRatingCanaryRegistry, BargainingRatingPredictor, EPOCH_ORIGIN, PublicRatingReader, RATING_CANARY_CONTRACT, RATING_CANARY_SEED_KIND, _accepted_terminal, _sha


def _seed(protocol_path: Path) -> dict[str, object]:
    seed: dict[str, object] = {
        "contract": RATING_CANARY_CONTRACT,
        "schema_version": 1,
        "kind": RATING_CANARY_SEED_KIND,
        "rating_cutoff": "2026-08-12T18:31:08.569568+00:00",
        "epoch_origin": EPOCH_ORIGIN,
        "source": {"manifest_sha256": "a" * 64},
        "eta_schedule": {"kind": "constant", "name": "test", "floor": 0.002},
        "bargaining_model": {
            "structural_model": {"feature_names": ["bias"], "coefficients": [0.5], "ridge_lambda": 0.1, "training_count": 10},
            "references": {},
            "configuration_residuals": {},
            "rank_alpha": 2.0,
            "residual_alpha": 1.0,
            "training_samples": 10,
            "latest_completed_at": "2026-08-12T18:31:08.569568+00:00",
        },
        "intervals": {"lower_80": -1.0, "upper_80": 1.0, "lower_95": -2.0, "upper_95": 2.0},
        "implementation_sha256": {"glee_rating_canary.py": _file_digest(Path(__file__).resolve().parents[1] / "src" / "nommd_arena" / "glee_rating_canary.py"), protocol_path.name: _file_digest(protocol_path)},
        "boundary": "shadow-only",
    }
    seed["seed_sha256"] = _sha(seed)
    return seed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reporter(root: Path, *, player_id: str = "opponent-id", sequence: int = 7, completed_at: str = "2026-08-13T12:00:00+00:00") -> None:
    frontier: dict[str, object] = {
        "contract": "glee-arena-reporter-v1",
        "schema_version": 2,
        "kind": "glee-arena-family-frontier",
        "sequence": sequence,
        "started_at": completed_at,
        "completed_at": completed_at,
        "frontier_id": "frontier",
        "poll_interval_s": 10.0,
        "metrics": {},
        "families": {
            "bargaining": {
                "poll": {"status": "ok"},
                "rows": [
                    {
                        "first_seen_sequence": 1,
                        "last_changed_sequence": sequence,
                        "last_observed_sequence": sequence,
                        "position": 3,
                        "row": {"player_id": player_id, "player_name": "Opponent", "rating": 1900.0, "games_played": 200},
                    }
                ],
            }
        },
    }
    frontier["frontier_sha256"] = _sha(frontier)
    _write_json(root / "current.json", frontier)


def _game(*, hidden: bool = False) -> dict[str, object]:
    return {
        "game_id": "game-one",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "hidden" if hidden else "agent", "name": None if hidden else "Opponent"},
        "valid_actions": {"type": "offer", "fields": {}},
        "game_state": {
            "phase": "offer",
            "round": 1,
            "money_to_divide": 100.0,
            "delta_1": 0.95,
            "delta_2": 0.9,
            "complete_information": True,
            "horizon_known": False,
            "messages_allowed": False,
            "history": [],
        },
    }


def _package_context(*, hidden: bool = False) -> dict[str, object]:
    return {
        "contract": "glee-opponent-statistical-package-v1",
        "release": "test",
        "frontier_sequence": 1,
        "identity_resolution": {"status": "hidden-population" if hidden else "exact-current-label", "public_player_id": None if hidden else "opponent-id"},
        "live_overlay": {"contract": "glee-opponent-statistical-live-overlay-v1", "revision": 4, "state_sha256": "c" * 64},
    }


def _sensor() -> dict[str, object]:
    return {"contract": "glee-sensor-frontier-v1", "sequence": 9, "fetched_at": "2026-08-13T12:00:01+00:00", "frontier_sha256": "d" * 64, "stats": {"agent_id": "self", "scores": {"bargaining": {"rating": 1800.0, "games_played": 300}}}}


class _PackageReader:
    receipt = {"contract": "glee-opponent-statistical-package-v1", "release": "test", "manifest_sha256": "b" * 64, "live_overlay_contract": "glee-opponent-statistical-live-overlay-v1", "live_overlay_id": "test-live"}

    @staticmethod
    def status() -> dict[str, object]:
        return {"contract": "glee-opponent-statistical-live-overlay-v1", "revision": 4, "state_sha256": "c" * 64}


class _Rollout:
    @staticmethod
    def evaluate_offer(_context: object, opponent_share: float) -> dict[str, object]:
        return {"opponent_share": opponent_share, "opponent_accept_probability_conservative": opponent_share, "expected_value": opponent_share * (1.0 - opponent_share)}


class _AdvisorHandle:
    context = object()
    rollout = _Rollout()
    prompt_context = {"behavioral_continuation": {"modeled_offer_policy": {"opponent_share": 0.4}}}


def test_public_rating_reader_requires_fresh_current_exact_row(tmp_path: Path) -> None:
    _reporter(tmp_path)
    reader = PublicRatingReader(tmp_path, max_age_s=30.0)

    available = reader.player("bargaining", "opponent-id", now="2026-08-13T12:00:20+00:00")
    stale = reader.player("bargaining", "opponent-id", now="2026-08-13T12:01:00+00:00")
    missing = reader.player("bargaining", "other-id", now="2026-08-13T12:00:20+00:00")

    assert available["status"] == "available"
    assert available["rating"] == 1900.0
    assert stale["reason"] == "reporter-frontier-stale"
    assert missing["reason"] == "public-id-absent-or-ambiguous"


def test_predictor_produces_bounded_frozen_terminal_forecast(tmp_path: Path) -> None:
    protocol = Path(__file__).resolve().parents[1] / "protocols" / "glee-bargaining-rating-canary-v1.md"
    seed = _seed(protocol)
    predictor = BargainingRatingPredictor(seed)
    terminal = _accepted_terminal(_game(), opponent_share=0.4)

    result = predictor.predict(terminal, target_player="player_1", target_rating=1800.0, target_games=300, other_rating=1900.0, terminal_at="2026-08-13T12:00:05+00:00")

    assert result["status"] == "available"
    assert result["rank_support"] == 0
    assert result["residual_support"] == 0
    assert result["interval_80"] == pytest.approx([result["predicted_delta"] - 1.0, result["predicted_delta"] + 1.0])


def test_round_one_terminal_does_not_require_hidden_opponent_discount() -> None:
    game = _game(hidden=True)
    game["game_state"]["complete_information"] = False
    game["game_state"].pop("delta_2")

    terminal = _accepted_terminal(game, opponent_share=0.4)

    assert terminal["result"]["player_1_payoff"] == pytest.approx(60.0)
    assert terminal["result"]["player_2_payoff"] == pytest.approx(40.0)
    game["game_state"]["round"] = 2
    with pytest.raises(ValueError, match="hidden discount factor"):
        _accepted_terminal(game, opponent_share=0.4)


def test_canary_requires_live_packages_and_matures_only_after_registration(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    protocol = project_root / "protocols" / "glee-bargaining-rating-canary-v1.md"
    seed_path = tmp_path / "seed.json"
    _write_json(seed_path, _seed(protocol))
    reporter_root = tmp_path / "reporter"
    _reporter(reporter_root, completed_at="2026-08-13T12:00:00+00:00")
    history_path = tmp_path / "rating-deltas.json"
    bad_reader = _PackageReader()
    bad_reader.receipt = {key: value for key, value in bad_reader.receipt.items() if key != "live_overlay_contract"}
    with pytest.raises(RuntimeError, match="live overlay"):
        BargainingRatingCanary(seed_path=seed_path, protocol_path=protocol, registry_path=tmp_path / "bad.sqlite3", reporter_root=reporter_root, history_path=history_path, package_reader=bad_reader)
    canary = BargainingRatingCanary(seed_path=seed_path, protocol_path=protocol, registry_path=tmp_path / "canary.sqlite3", reporter_root=reporter_root, history_path=history_path, package_reader=_PackageReader(), reporter_max_age_s=3600.0)
    game = _game()
    action = {"player_1_gain": 60.0, "player_2_gain": 40.0, "message": ""}
    original = deepcopy(action)

    assert canary.capture_game(game, package_context=_package_context(), sensor_frontier=_sensor(), observed_at="2026-08-13T12:00:02+00:00") is True
    assert canary.register_turn(turn_id="game-one:round-1", stage="primary", game=game, action=action, advisor_handle=_AdvisorHandle(), package_context=_package_context(), causal_observation_at="2026-08-13T12:00:03+00:00") is True
    assert action == original
    terminal = _accepted_terminal(game, opponent_share=0.4)
    assert canary.register_terminal(terminal, terminal_at="2026-08-13T12:00:04+00:00") is True
    _write_json(history_path, {"contract": "glee-rating-history-v1", "schema_version": 1, "synchronized_at": "2099-01-01T00:00:00+00:00", "game_deltas_sha256": "e" * 64, "game_deltas": {"game-one": {"rating_delta": 1.25, "completed_at": "2026-08-13T12:00:04+00:00", "revision": 1, "record_sha256": "f" * 64}}})

    receipt = canary.reconcile(force=True)
    status = canary.registry.status()

    assert receipt["matured"] == 1
    assert status["game_contexts"] == 1
    assert status["turn_shadows"] == 1
    assert status["terminal_forecasts"] == 1
    assert status["self_maturations"] == 1
    assert status["self_metrics"]["count"] == 1
    assert status["recommendation_comparisons"] == 1


def test_registry_rejects_conflicting_append_only_turn(tmp_path: Path) -> None:
    registry = BargainingRatingCanaryRegistry(tmp_path / "registry.sqlite3", metadata={"seed_sha256": "a" * 64})
    registry.register_game("g", observed_at="2026-08-13T12:00:00+00:00", payload={"value": 1})
    registry.register_turn("g:t:primary", game_id="g", stage="primary", causal_observation_at="2026-08-13T12:00:01+00:00", payload={"action": 1})

    with pytest.raises(RuntimeError, match="conflicting immutable"):
        registry.register_turn("g:t:primary", game_id="g", stage="primary", causal_observation_at="2026-08-13T12:00:01+00:00", payload={"action": 2})
