import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nommd_arena.glee_bargaining_twin import BargainingTwin, BargainingTwinExperiment, TwinConfig, extract_bargaining_game, load_bargaining_corpus


def _hash(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _history_entry(*, proposer: str, opponent_share: float, decision: str, round_number: int, message: str = "") -> dict[str, Any]:
    return {
        "round": round_number,
        "proposer": proposer,
        "offer": {
            "round": round_number,
            "proposer": proposer,
            "player_1_gain": 100 * (1 - opponent_share),
            "player_2_gain": 100 * opponent_share,
            "message": message,
        },
        "decision": decision,
        "response_time_ms": 1200 + round_number,
    }


def _write_job(
    root: Path,
    *,
    opponent_id: str,
    opponent_name: str,
    game_index: int,
    completion_order: int,
    offered_share: float,
    accepted: bool,
    proposal_share: float,
    complete_information: bool = False,
) -> Path:
    game_id = f"{opponent_id}-game-{game_index:02d}"
    history = [_history_entry(proposer="player_1", opponent_share=offered_share, decision="accept" if accepted else "reject", round_number=1)]
    if not accepted:
        history.append(_history_entry(proposer="player_2", opponent_share=proposal_share, decision="reject", round_number=2, message="Fair final allocation"))
    state: dict[str, Any] = {
        "complete_information": complete_information,
        "current_player": "player_1",
        "delta_1": 0.95,
        "game_family": "bargaining",
        "history": history,
        "horizon_known": True,
        "max_rounds": 4,
        "messages_allowed": True,
        "money_to_divide": 100,
        "phase": "completed",
        "result": {"outcome": "agreement" if accepted else "no_agreement"},
        "round": len(history),
    }
    if complete_information:
        state["delta_2"] = 0.9
    final_game = {
        "game_family": "bargaining",
        "game_id": game_id,
        "game_state": state,
        "opponent": {"name": opponent_name, "type": "agent"},
        "result": state["result"],
        "status": "completed",
        "your_player": "player_1",
    }
    job_id = f"job-{opponent_id}-{game_index:02d}"
    job = {
        "completed_at": f"2026-08-09T00:00:{completion_order:02d}+00:00",
        "completion_order": completion_order,
        "final_game": final_game,
        "final_game_sha256": _hash(final_game),
        "game_family": "bargaining",
        "game_id": game_id,
        "job_id": job_id,
        "kind": "named-opponent-update-job",
        "opponent": {"id": opponent_id, "name": opponent_name},
        "schema_version": 3,
    }
    path = root / "jobs" / opponent_id / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_extractor_attributes_actions_and_preserves_information_mask(tmp_path: Path) -> None:
    path = _write_job(
        tmp_path,
        opponent_id="aster-id",
        opponent_name="Aster",
        game_index=1,
        completion_order=1,
        offered_share=0.3,
        accepted=False,
        proposal_share=0.7,
    )
    job = json.loads(path.read_text(encoding="utf-8"))
    job["final_game"]["game_state"]["history"][1]["offer"]["message"] = "Rubinstein subgame equilibrium 30/70"
    job["final_game_sha256"] = _hash(job["final_game"])
    game = extract_bargaining_game(job, job_path=path, job_sha256="sealed-job-sha")
    assert [row.action_type for row in game.rows] == ["response", "proposal"]
    response, proposal = game.rows
    assert response.offered_share == pytest.approx(0.3)
    assert response.accepted is False
    assert response.context.opponent_discount is None
    assert response.context.our_discount == pytest.approx(0.95)
    assert proposal.proposal_share == pytest.approx(0.7)
    assert proposal.message_act == "authority"
    assert proposal.context.previous_our_offer_to_opponent_share == pytest.approx(0.3)
    assert proposal.context.previous_opponent_response == "reject"
    assert all(row.job_sha256 == "sealed-job-sha" for row in game.rows)


def test_extractor_rejects_a_changed_final_game(tmp_path: Path) -> None:
    path = _write_job(
        tmp_path,
        opponent_id="aster-id",
        opponent_name="Aster",
        game_index=1,
        completion_order=1,
        offered_share=0.3,
        accepted=False,
        proposal_share=0.7,
    )
    job = json.loads(path.read_text(encoding="utf-8"))
    job["final_game"]["game_state"]["money_to_divide"] = 101
    with pytest.raises(ValueError, match="final game SHA-256 mismatch"):
        extract_bargaining_game(job, job_path=path, job_sha256="sealed-job-sha")


def test_shadow_experiment_fits_loadable_twins_without_future_population_leakage(tmp_path: Path) -> None:
    offered = (0.3, 0.75, 0.4, 0.8, 0.5, 0.7, 0.3, 0.8)
    for index, share in enumerate(offered):
        _write_job(
            tmp_path / "source",
            opponent_id="aster-id",
            opponent_name="Aster",
            game_index=index,
            completion_order=2 * index,
            offered_share=share,
            accepted=share >= 0.65,
            proposal_share=0.72,
            complete_information=index % 2 == 0,
        )
        _write_job(
            tmp_path / "source",
            opponent_id="beryl-id",
            opponent_name="Beryl",
            game_index=index,
            completion_order=2 * index + 1,
            offered_share=share,
            accepted=share >= 0.35,
            proposal_share=0.42,
            complete_information=index % 2 == 1,
        )
    output = tmp_path / "models"
    experiment = BargainingTwinExperiment(
        dossier_root=tmp_path / "source",
        output_dir=output,
        min_games=6,
        min_train_games=4,
        holdout_fraction=0.25,
        config=TwinConfig.small_test_config(),
    )
    result = experiment.run()
    assert result["manifest"]["status"] == "shadow-only"
    assert result["manifest"]["model_count"] == 2
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    aster_evaluation = next(target for target in evaluation["targets"] if target["opponent"]["name"] == "Aster")
    assert aster_evaluation["split"]["train_game_ids"] == [f"aster-id-game-{index:02d}" for index in range(6)]
    assert aster_evaluation["split"]["holdout_game_ids"] == ["aster-id-game-06", "aster-id-game-07"]
    assert aster_evaluation["split"]["population_prior_game_count"] == 6
    assert not set(aster_evaluation["split"]["train_game_ids"]) & set(aster_evaluation["split"]["holdout_game_ids"])
    artifact_path = output / "opponents" / "aster-id" / "bargaining" / "model.json"
    twin = BargainingTwin.from_path(artifact_path)
    games, rejected = load_bargaining_corpus(tmp_path / "source")
    assert len(games) == 16
    assert rejected == ()
    context = next(game.rows[0].context for game in games if game.opponent_id == "aster-id")
    assert twin.response_probability(context, 0.75) > twin.response_probability(context, 0.3)
    assert twin.proposal_distribution(context)["mean"] > 0.6
    assert twin.sample_response(context, 0.7, seed=19) == twin.sample_response(context, 0.7, seed=19)
    assert twin.sample_proposal(context, seed=23) == twin.sample_proposal(context, seed=23)
    projection = twin.compact_projection(context)
    assert projection["kind"] == "shadow-bargaining-twin-projection"
    assert projection["use_boundary"].startswith("Shadow prediction only")
    corrupted = json.loads(artifact_path.read_text(encoding="utf-8"))
    corrupted["training"]["game_count"] += 1
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        BargainingTwin(corrupted)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run()
