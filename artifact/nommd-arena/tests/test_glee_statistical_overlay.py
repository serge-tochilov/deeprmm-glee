import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nommd_arena.glee_statistical_overlay import LiveOpponentStatisticalPackageReader
from nommd_arena.glee_statistical_package import OpponentStatisticalPackageCompiler


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


def _compiled_package(tmp_path: Path) -> Path:
    frontier = 10
    identity = tmp_path / "identity"
    behavior = tmp_path / "behavior"
    channels = tmp_path / "channels"
    activity = tmp_path / "activity.json"
    families = ("bargaining", "negotiation", "persuasion")
    rows = []
    for family in families:
        for player_id, label in (("self-id", "DeepRMM-01"), ("aster-id", "Aster"), ("reserve-old", "RESERVE"), ("reserve-new", "RESERVE")):
            rows.append({"family": family, "player_id": player_id, "current_label": label, "aliases": [label], "present_at_frontier": True, "latest_present_metadata": {"is_baseline": False, "is_benchmark": False}})
    _write_jsonl(identity / "identity-registry.jsonl", rows)
    _write_json(identity / "summary.json", {"frontier_sequence": frontier})
    _manifest(identity, "glee-public-identity-registry-v1", frontier, ("identity-registry.jsonl", "summary.json"))
    moves = [{"kind": "proposal", "action_value": 0.5, "decision": None, "context": {"opponent_role": "none", "complete_information": True, "horizon_known": True, "round_phase": 0.1, "messages_allowed": True}}]
    games = [{"family": family, "game_id": f"base-{family}", "public_player_id": "aster-id", "moves": moves} for family in families]
    _write_jsonl(behavior / "behavior-games.jsonl", games)
    _write_json(behavior / "summary.json", {"inventory": {"games": 3}})
    _manifest(behavior, "glee-collision-safe-behavior-corpus-v1", frontier, ("behavior-games.jsonl", "summary.json"))
    _write_json(channels / "summary.json", {"source": {"frontier_sequence": frontier}, "families": {family: {"action": {"selected_hyperparameters": {"alpha": 5.0}}} for family in families}})
    _manifest(channels, "glee-behavior-channel-evaluation-v1", frontier, ("summary.json",))
    _write_json(activity, {"source_frontier": {"frontier_sequence": frontier, "self_player_ids": {family: "self-id" for family in families}}})
    output = tmp_path / "packages"
    OpponentStatisticalPackageCompiler(identity_dir=identity, behavior_dir=behavior, channel_dir=channels, activity_summary=activity, output_root=output).run()
    return output


def _terminal_bargaining(game_id: str, name: str | None, *, hidden: bool = False, opponent_gain: float = 70.0) -> dict[str, object]:
    return {
        "game_family": "bargaining",
        "game_id": game_id,
        "your_player": "player_1",
        "opponent": {"type": "hidden" if hidden else "agent", "name": name},
        "status": "completed",
        "result": {"outcome": "no_deal"},
        "game_state": {"round": 1, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "money_to_divide": 100.0, "history": [{"round": 1, "proposer": "player_2", "offer": {"round": 1, "proposer": "player_2", "player_1_gain": 100.0 - opponent_gain, "player_2_gain": opponent_gain}}]},
    }


def _terminal_negotiation(game_id: str) -> dict[str, object]:
    return {
        "game_family": "negotiation",
        "game_id": game_id,
        "your_player": "player_1",
        "opponent": {"type": "agent", "name": "Aster"},
        "status": "completed",
        "result": {"outcome": "no_deal"},
        "game_state": {"round": 1, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "player_1_role": "buyer", "player_2_role": "seller", "player_1_value": 100.0, "player_2_value": 20.0, "history": [{"round": 1, "offer": {"round": 1, "from_player": "player_2", "price": 70.0}}]},
    }


def _terminal_persuasion(game_id: str) -> dict[str, object]:
    return {
        "game_family": "persuasion",
        "game_id": game_id,
        "your_player": "player_1",
        "opponent": {"type": "agent", "name": "Aster"},
        "status": "completed",
        "result": {"outcome": "completed"},
        "game_state": {"round": 1, "total_rounds": 5, "seller_message_type": "binary", "player_1_role": "buyer", "player_2_role": "seller", "history": [{"round": 1, "seller_message": "high"}]},
    }


def test_live_overlay_updates_every_family_and_is_visible_across_readers(tmp_path: Path) -> None:
    root = _compiled_package(tmp_path)
    first = LiveOpponentStatisticalPackageReader(root)
    second = LiveOpponentStatisticalPackageReader(root)
    try:
        before = second.view(_terminal_bargaining("view", "Aster"))
        assert before["live_overlay"]["revision"] == 0
        bargaining = first.update_completed_game(_terminal_bargaining("b-1", "Aster"), completed_at="2026-08-13T12:00:00+00:00", completion_order=1)
        negotiation = first.update_completed_game(_terminal_negotiation("n-1"), completed_at="2026-08-13T12:00:01+00:00", completion_order=2)
        persuasion = first.update_completed_game(_terminal_persuasion("p-1"), completed_at="2026-08-13T12:00:02+00:00", completion_order=3)
        assert bargaining["direct_model_updated"] is True
        assert negotiation["direct_model_updated"] is True
        assert persuasion["direct_model_updated"] is True
        after = second.view(_terminal_bargaining("view", "Aster"))
        assert after["live_overlay"]["revision"] == 3
        assert after["live_overlay"]["family_completed_games"] == 1
        assert after["evidence"]["live_games"] == 1
        assert second.status()["games_by_family"] == {"bargaining": 1, "negotiation": 1, "persuasion": 1}
    finally:
        first.close()
        second.close()


def test_hidden_and_colliding_games_update_population_without_direct_assignment(tmp_path: Path) -> None:
    root = _compiled_package(tmp_path)
    reader = LiveOpponentStatisticalPackageReader(root)
    try:
        hidden = reader.update_completed_game(_terminal_bargaining("hidden-1", None, hidden=True), completed_at="now", completion_order=1)
        collision = reader.update_completed_game(_terminal_bargaining("collision-1", "RESERVE"), completed_at="now", completion_order=2)
        assert hidden["population_model_updated"] is True and hidden["direct_model_updated"] is False
        assert collision["population_model_updated"] is True and collision["direct_model_updated"] is False
        assert collision["identity_resolution"]["candidate_public_player_ids"] == ["reserve-new", "reserve-old"]
        view = reader.view(_terminal_bargaining("view", "RESERVE"))
        assert view["identity_resolution"]["status"] == "current-label-collision"
        assert view["evidence"]["tier"] == "population-only"
        assert view["live_overlay"]["family_completed_games"] == 2
    finally:
        reader.close()


def test_live_overlay_is_idempotent_and_rejects_changed_duplicate(tmp_path: Path) -> None:
    root = _compiled_package(tmp_path)
    reader = LiveOpponentStatisticalPackageReader(root)
    try:
        game = _terminal_bargaining("same", "Aster")
        assert reader.update_completed_game(game, completed_at="now", completion_order=1)["status"] == "updated"
        assert reader.update_completed_game(game, completed_at="later", completion_order=2)["status"] == "duplicate"
        assert reader.status()["revision"] == 1
        with pytest.raises(RuntimeError, match="differs from its applied"):
            reader.update_completed_game(_terminal_bargaining("same", "Aster", opponent_gain=60.0), completed_at="later", completion_order=3)
        assert reader.status()["revision"] == 1
    finally:
        reader.close()


def test_zero_observation_terminal_game_still_updates_model_evidence(tmp_path: Path) -> None:
    root = _compiled_package(tmp_path)
    reader = LiveOpponentStatisticalPackageReader(root)
    try:
        game = _terminal_bargaining("silent", "Aster")
        game["game_state"]["history"] = []
        receipt = reader.update_completed_game(game, completed_at="now", completion_order=1)
        assert receipt["observations"] == 0
        assert receipt["game_recorded"] is True
        assert receipt["population_model_updated"] is True
        assert receipt["population_counts_updated"] is False
        assert receipt["direct_model_updated"] is True
        assert receipt["direct_counts_updated"] is False
        view = reader.view(_terminal_bargaining("view", "Aster"))
        assert view["live_overlay"]["revision"] == 1
        assert view["evidence"]["live_games"] == 1
    finally:
        reader.close()


def test_live_overlay_serializes_concurrent_family_writers(tmp_path: Path) -> None:
    root = _compiled_package(tmp_path)
    first = LiveOpponentStatisticalPackageReader(root)
    second = LiveOpponentStatisticalPackageReader(root)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda item: item[0].update_completed_game(item[1], completed_at="now", completion_order=1), ((first, _terminal_bargaining("concurrent-b", "Aster")), (second, _terminal_negotiation("concurrent-n")))))
        assert [receipt["status"] for receipt in receipts] == ["updated", "updated"]
        assert first.status()["revision"] == 2
        assert second.status()["games_by_family"] == {"bargaining": 1, "negotiation": 1, "persuasion": 0}
    finally:
        first.close()
        second.close()
