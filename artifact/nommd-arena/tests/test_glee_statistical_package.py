import hashlib
import json
from pathlib import Path

import pytest

from nommd_arena.glee_statistical_package import STATISTICAL_DECISION_FORECAST_CONTRACT, OpponentStatisticalPackageCompiler, OpponentStatisticalPackageReader, _isotonic_nondecreasing, bargaining_submitted_offer_forecast, compact_bargaining_decision_forecast, compact_opponent_decision_forecast


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _manifest(directory: Path, contract: str, frontier: int, artifacts: tuple[str, ...]) -> None:
    _write_json(directory / "manifest.json", {"contract": contract, "schema_version": 1, "frontier_sequence": frontier, "artifacts": {name: {"sha256": _sha(directory / name)} for name in artifacts}})


def _move(kind: str, value: float, decision: str | None, *, role: str = "none") -> dict[str, object]:
    return {"kind": kind, "action_value": value, "decision": decision, "context": {"opponent_role": role, "complete_information": True, "horizon_known": True, "round_phase": 0.1, "messages_allowed": True}}


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    frontier = 10
    identity = tmp_path / "identity"
    behavior = tmp_path / "behavior"
    channels = tmp_path / "channels"
    activity = tmp_path / "activity.json"
    families = ("bargaining", "negotiation", "persuasion")
    rows = []
    for family in families:
        for player_id, label in (("self-id", "DeepRMM-01"), ("aster-id", "Aster"), ("reserve-old", "RESERVE"), ("reserve-new", "RESERVE")):
            rows.append({"contract": "glee-public-identity-registry-v1", "schema_version": 1, "family": family, "player_id": player_id, "current_label": label, "aliases": [label], "present_at_frontier": True, "latest_present_metadata": {"is_baseline": False, "is_benchmark": False}})
    _write_jsonl(identity / "identity-registry.jsonl", rows)
    _write_json(identity / "summary.json", {"contract": "glee-public-identity-registry-v1", "frontier_sequence": frontier})
    _manifest(identity, "glee-public-identity-registry-v1", frontier, ("identity-registry.jsonl", "summary.json"))
    games = [
        {"family": "bargaining", "game_id": "b-1", "public_player_id": "aster-id", "moves": [_move("proposal", 0.5, None), _move("response", 0.4, "accept")]},
        {"family": "negotiation", "game_id": "n-1", "public_player_id": "aster-id", "moves": [_move("proposal", 0.6, None, role="seller"), _move("response", 0.5, "rejectoffer", role="seller")]},
        {"family": "persuasion", "game_id": "p-1", "public_player_id": "aster-id", "moves": [_move("signal", 1.0, "positive", role="seller")]},
    ]
    _write_jsonl(behavior / "behavior-games.jsonl", games)
    _write_json(behavior / "summary.json", {"contract": "glee-collision-safe-behavior-corpus-v1", "inventory": {"games": 3}})
    _manifest(behavior, "glee-collision-safe-behavior-corpus-v1", frontier, ("behavior-games.jsonl", "summary.json"))
    channel_summary = {"contract": "glee-behavior-channel-evaluation-v1", "source": {"frontier_sequence": frontier}, "families": {family: {"action": {"selected_hyperparameters": {"alpha": 5.0}}} for family in families}}
    _write_json(channels / "summary.json", channel_summary)
    _manifest(channels, "glee-behavior-channel-evaluation-v1", frontier, ("summary.json",))
    _write_json(activity, {"source_frontier": {"frontier_sequence": frontier, "self_player_ids": {family: "self-id" for family in families}}})
    return identity, behavior, channels, activity


def _game(name: str | None, *, family: str = "bargaining", hidden: bool = False) -> dict[str, object]:
    return {"game_id": "game", "game_family": family, "your_player": "player_1", "opponent": {"type": "hidden" if hidden else "agent", "name": name}, "game_state": {"round": 1, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "player_2_role": "seller"}}


def test_compiler_builds_complete_cross_family_matrix_and_collision_safe_reader(tmp_path: Path) -> None:
    identity, behavior, channels, activity = _sources(tmp_path)
    output = tmp_path / "packages"
    summary = OpponentStatisticalPackageCompiler(identity_dir=identity, behavior_dir=behavior, channel_dir=channels, activity_summary=activity, output_root=output).run()
    assert summary["matrix"] == {"public_opponents": 3, "families": 3, "packages": 9, "complete_cross_product": True}
    assert all(summary["families"][family]["with_direct_evidence"] == 1 for family in ("bargaining", "negotiation", "persuasion"))
    assert all(summary["families"][family]["population_only"] == 2 for family in ("bargaining", "negotiation", "persuasion"))
    rows = [json.loads(line) for line in (output / "frontier-10-v1" / "packages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 9
    assert not any(row["public_player_id"] == "self-id" for row in rows)
    reader = OpponentStatisticalPackageReader(output)
    exact = reader.view(_game("Aster"))
    assert exact["identity_resolution"]["status"] == "exact-current-label"
    assert exact["identity_resolution"]["public_player_id"] == "aster-id"
    assert exact["evidence"]["tier"] == "direct-minimal"
    assert exact["action_model"]["contexts"]
    collision = reader.view(_game("RESERVE"))
    assert collision["identity_resolution"]["status"] == "current-label-collision"
    assert collision["identity_resolution"]["public_player_id"] is None
    assert collision["identity_resolution"]["candidate_public_player_ids"] == ["reserve-new", "reserve-old"]
    assert collision["evidence"]["tier"] == "population-only"
    hidden = reader.view(_game(None, hidden=True))
    assert hidden["identity_resolution"]["status"] == "hidden-population"
    assert hidden["evidence"]["tier"] == "population-only"
    assert len(hidden["action_model"]["contexts"]) <= 4


def test_reader_fails_closed_on_tampered_package(tmp_path: Path) -> None:
    identity, behavior, channels, activity = _sources(tmp_path)
    output = tmp_path / "packages"
    OpponentStatisticalPackageCompiler(identity_dir=identity, behavior_dir=behavior, channel_dir=channels, activity_summary=activity, output_root=output).run()
    packages = output / "frontier-10-v1" / "packages.jsonl"
    packages.write_text(packages.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        OpponentStatisticalPackageReader(output)


def test_bargaining_decision_forecast_separates_exact_nearby_and_population_support(tmp_path: Path) -> None:
    identity, behavior, channels, activity = _sources(tmp_path)
    output = tmp_path / "packages"
    OpponentStatisticalPackageCompiler(identity_dir=identity, behavior_dir=behavior, channel_dir=channels, activity_summary=activity, output_root=output).run()
    reader = OpponentStatisticalPackageReader(output)
    game = _game("Aster")
    game["valid_actions"] = {"type": "offer"}
    game["game_state"]["money_to_divide"] = 100.0
    exact = reader.view(game)
    exact_row = next(row for row in exact["decision_local_model"]["response_to_our_offer"] if row["offered_opponent_share_bin"] == 24)
    assert exact_row["evidence_scope"] == "exact-current-context"
    assert exact_row["exact_direct_support"] == 1
    game["game_state"]["messages_allowed"] = False
    nearby = reader.view(game)
    nearby_row = next(row for row in nearby["decision_local_model"]["response_to_our_offer"] if row["offered_opponent_share_bin"] == 24)
    assert nearby_row["evidence_scope"] == "nearby-context"
    assert nearby_row["exact_direct_support"] == 0
    assert nearby_row["nearby_direct_effective_support"] > 0
    advisor = {"response_to_our_numeric_offer": {"curve_by_opponent_share": [{"opponent_share": 0.4, "accept_probability_v2": 0.5}], "myopic_no_continuation_candidate": {"opponent_share": 0.4}}, "behavioral_continuation": {"modeled_offer_policy": {"opponent_share": 0.4}}}
    compact = compact_bargaining_decision_forecast(nearby, game, advisor)
    assert compact is not None and compact["contract"] == STATISTICAL_DECISION_FORECAST_CONTRACT
    assert compact["policy_gate"]["status"] == "bounded-authoritative"
    assert compact["response_forecasts"][0]["evidence_scope"] == "nearby-context"
    prediction = bargaining_submitted_offer_forecast(nearby, game, {"alice_gain": 60.0, "bob_gain": 40.0}, {"opponent_acceptance_probability_v2": 0.5})
    assert prediction is not None
    assert prediction["frontier"] == "registered-before-network-submission"
    assert prediction["action_changed"] is False


def test_bargaining_response_projection_is_monotone_and_retains_raw_values() -> None:
    projected = _isotonic_nondecreasing([0.2, 0.8, 0.4, 0.9], [1.0, 1.0, 1.0, 1.0])
    assert projected == pytest.approx([0.2, 0.6, 0.6, 0.9])
    assert all(left <= right for left, right in zip(projected, projected[1:]))


def test_other_family_compact_projection_selects_the_next_opponent_move() -> None:
    package = {
        "family": "negotiation",
        "identity_resolution": {"status": "exact-current-label", "public_player_id": "aster-id"},
        "evidence": {"games": 12, "observations": 30, "contexts": 4, "tier": "direct-high"},
        "action_model": {
            "contexts": [
                {"condition": "action|kind=proposal|role=seller|information=incomplete|horizon=known|phase=early|messages=disabled", "direct_support": 5, "population_support": 100, "distribution": {"outcomes": [{"outcome": "value=29", "probability": 0.8}], "omitted_probability": 0.2}},
                {"condition": "decision|kind=response|role=seller|information=incomplete|horizon=known|phase=early|messages=disabled|value=25", "direct_support": 3, "population_support": 40, "distribution": {"outcomes": [{"outcome": "decision=rejectoffer", "probability": 0.75}], "omitted_probability": 0.25}},
            ]
        },
    }
    game = _game("Aster", family="negotiation")
    game["valid_actions"] = {"type": "offer"}
    forecast = compact_opponent_decision_forecast(package, game)
    assert forecast is not None
    assert forecast["authority"] == "advisory-only"
    assert forecast["expected_next_opponent_move"] == {"prefix": "decision", "kind": "response"}
    assert len(forecast["current_context_projections"]) == 1
    assert forecast["current_context_projections"][0]["condition"].startswith("decision|kind=response")
