import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from nommd_arena.glee_bargaining_twin import TwinConfig
from nommd_arena.glee_bargaining_validation import BargainingRollingValidation, MODEL_NAMES, ValidationConfig


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
        "response_time_ms": 900 + round_number,
    }


def _write_job(
    root: Path,
    *,
    opponent_id: str,
    opponent_name: str,
    game_index: int,
    completion_order: int,
    threshold: float,
    proposal_share: float,
) -> Path:
    offered_share = (0.25, 0.7, 0.4, 0.8, 0.5, 0.75)[game_index % 6]
    accepted = offered_share >= threshold
    game_id = f"{opponent_id}-game-{game_index:02d}"
    history = [_history_entry(proposer="player_1", opponent_share=offered_share, decision="accept" if accepted else "reject", round_number=1)]
    if not accepted:
        history.append(_history_entry(proposer="player_2", opponent_share=proposal_share, decision="reject", round_number=2, message="Fair final split" if game_index % 2 else ""))
    state: dict[str, Any] = {
        "complete_information": game_index % 2 == 0,
        "current_player": "player_1",
        "delta_1": 0.95,
        "delta_2": 0.9,
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


def test_rolling_validation_preserves_game_frontiers_and_writes_figures(tmp_path: Path) -> None:
    specifications = (
        ("aster-id", "Aster", 0.65, 0.72),
        ("beryl-id", "Beryl", 0.35, 0.42),
        ("cygnus-id", "Cygnus", 0.5, 0.58),
    )
    for game_index in range(8):
        for opponent_index, (opponent_id, opponent_name, threshold, proposal_share) in enumerate(specifications):
            _write_job(
                tmp_path / "source",
                opponent_id=opponent_id,
                opponent_name=opponent_name,
                game_index=game_index,
                completion_order=3 * game_index + opponent_index,
                threshold=threshold,
                proposal_share=proposal_share,
            )
    for game_index in range(3):
        _write_job(
            tmp_path / "source",
            opponent_id="sparse-id",
            opponent_name="Sparse",
            game_index=game_index,
            completion_order=24 + game_index,
            threshold=0.5,
            proposal_share=0.5,
        )
    output = tmp_path / "validation"
    validation = BargainingRollingValidation(
        dossier_root=tmp_path / "source",
        output_dir=output,
        validation_config=ValidationConfig(min_games=6, warmup_games=3, surprise_window=4),
        twin_config=TwinConfig.small_test_config(),
    )
    result = validation.run()
    assert result["manifest"]["status"] == "offline-shadow-only"
    assert result["manifest"]["summary"]["eligible_opponent_count"] == 3
    assert result["manifest"]["summary"]["origin_count"] == 15
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert predictions
    assert set(predictions[0]["predictions"]) == set(MODEL_NAMES)
    assert all(math.isfinite(float(prediction["nll"])) for record in predictions for prediction in record["predictions"].values())
    by_game: dict[str, list[dict[str, Any]]] = {}
    for record in predictions:
        by_game.setdefault(record["game_id"], []).append(record)
    assert all(len({record["prior_target_game_count"] for record in records}) == 1 for records in by_game.values())
    first_aster = by_game["aster-id-game-03"]
    assert {record["prior_target_game_count"] for record in first_aster} == {3}
    assert {record["prior_population_game_count"] for record in first_aster} == {6}
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["origin_count"] == 15
    assert len(evaluation["per_opponent"]) == 3
    assert set(evaluation["micro"]) == set(MODEL_NAMES)
    assert set(evaluation["response_expected_calibration_error"]) == set(MODEL_NAMES)
    assert all(0 <= value <= 1 for value in evaluation["response_expected_calibration_error"].values())
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert len(diagnostics["opponents"]) == 3
    assert all("population_transfer" in opponent for opponent in diagnostics["opponents"])
    assert len(result["manifest"]["figures"]) == 6
    for figure in result["manifest"]["figures"]:
        svg_path = output / "figures" / figure["svg"]
        csv_path = output / "figures" / figure["data"]
        assert svg_path.is_file() and csv_path.is_file()
        assert ET.parse(svg_path).getroot().tag.endswith("svg")
        assert csv_path.read_text(encoding="utf-8").count("\n") >= 2
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.run()


def test_validation_rejects_an_impossible_selection() -> None:
    with pytest.raises(ValueError, match="min_games must exceed warmup_games"):
        ValidationConfig(min_games=4, warmup_games=4).validate()
