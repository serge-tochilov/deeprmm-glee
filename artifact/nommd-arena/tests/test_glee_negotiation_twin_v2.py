import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nommd_arena.glee_negotiation_twin_v2 import NegotiationModelConfig, NegotiationOpponentModelV2, PriorObservation, extract_negotiation_game, load_negotiation_corpus, opponent_demand, price_from_opponent_demand
from nommd_arena.glee_negotiation_validation_v2 import NegotiationRollingValidationV2, NegotiationValidationConfig


def _hash(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _entry(*, round_number: int, from_player: str, price: float, decision: str, message: str = "") -> dict[str, Any]:
    decided_by = "player_2" if from_player == "player_1" else "player_1"
    return {"round": round_number, "offer": {"round": round_number, "from_player": from_player, "price": price, "message": message}, "decided_by": decided_by, "decision": decision, "response_time_ms": 1000 + round_number}


def _write_job(
    root: Path,
    *,
    opponent_id: str,
    opponent_name: str,
    game_index: int,
    completion_order: int,
    our_player: str = "player_1",
    complete_information: bool = False,
    history: list[dict[str, Any]] | None = None,
    terminal: str = "completed",
    outcome: str = "no_deal",
) -> Path:
    game_id = f"{opponent_id}-game-{game_index:02d}"
    state: dict[str, Any] = {
        "complete_information": complete_information,
        "current_player": our_player,
        "game_family": "negotiation",
        "history": history or [_entry(round_number=1, from_player="player_1", price=100 + game_index, decision="RejectOffer", message="Price 100")],
        "horizon_known": True,
        "max_rounds": 10,
        "messages_allowed": True,
        "phase": terminal,
        "player_1_role": "seller",
        "player_1_value": 80.0,
        "player_2_role": "buyer",
        "round": len(history or [1]),
        "result": {"outcome": outcome, "player_1_payoff": 0, "player_2_payoff": 0},
    }
    if complete_information:
        state["player_2_value"] = 150.0
    elif our_player == "player_2":
        state["player_2_value"] = 150.0
    final_game = {"game_family": "negotiation", "game_id": game_id, "game_state": state, "opponent": {"name": opponent_name, "type": "agent"}, "result": state["result"], "status": terminal, "your_player": our_player}
    job = {
        "schema_version": 3,
        "kind": "named-opponent-update-job",
        "job_id": f"job-{opponent_id}-{game_index:02d}",
        "opponent": {"id": opponent_id, "name": opponent_name},
        "game_id": game_id,
        "game_family": "negotiation",
        "completed_at": f"2026-08-09T00:{completion_order // 60:02d}:{completion_order % 60:02d}+00:00",
        "completion_order": completion_order,
        "final_game_sha256": _hash(final_game),
        "final_game": final_game,
    }
    path = root / "jobs" / opponent_id / f"{job['job_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_opponent_demand_has_one_direction_across_roles_and_is_invertible() -> None:
    seller_boundary = opponent_demand(150.0, opponent_role="seller", our_value=150.0)
    seller_adverse = opponent_demand(180.0, opponent_role="seller", our_value=150.0)
    buyer_boundary = opponent_demand(80.0, opponent_role="buyer", our_value=80.0)
    buyer_adverse = opponent_demand(60.0, opponent_role="buyer", our_value=80.0)
    assert seller_boundary == pytest.approx(0.0)
    assert buyer_boundary == pytest.approx(0.0)
    assert seller_adverse > 0
    assert buyer_adverse > 0
    assert price_from_opponent_demand(seller_adverse, opponent_role="seller", our_value=150.0) == pytest.approx(180.0)
    assert price_from_opponent_demand(buyer_adverse, opponent_role="buyer", our_value=80.0) == pytest.approx(60.0)


def test_extractor_attributes_actions_masks_secrets_and_preserves_prefix_causality(tmp_path: Path) -> None:
    history = [
        _entry(round_number=1, from_player="player_1", price=110, decision="RejectOffer", message="I offer $110"),
        _entry(round_number=2, from_player="player_2", price=100, decision="RejectOffer", message="Final price is $100"),
        _entry(round_number=3, from_player="player_1", price=120, decision="AcceptOffer", message="I can settle at $120"),
    ]
    path = _write_job(tmp_path, opponent_id="aster-id", opponent_name="Aster", game_index=1, completion_order=1, history=history)
    job = json.loads(path.read_text(encoding="utf-8"))
    job["final_game"]["game_state"]["player_2_value"] = 999.0
    job["final_game_sha256"] = _hash(job["final_game"])
    game = extract_negotiation_game(job, job_path=path, job_sha256="sealed-job")
    assert [row.action_type for row in game.rows] == ["response", "proposal", "response"]
    first_response, proposal, final_response = game.rows
    assert first_response.context.opponent_value is None
    assert first_response.context.current_offer_message_act == "price"
    assert first_response.context.observed_opponent_actions == 0
    assert proposal.context.observed_opponent_actions == 1
    assert proposal.context.current_offer_demand is None
    assert proposal.context.current_offer_message_act == "none"
    assert proposal.message_act == "urgency"
    assert proposal.context.previous_our_demand == pytest.approx(first_response.offered_demand)
    assert proposal.context.previous_opponent_response == "RejectOffer"
    assert final_response.context.previous_opponent_demand == pytest.approx(proposal.proposal_demand)
    assert final_response.context.previous_our_response == "RejectOffer"
    assert all(row.job_sha256 == "sealed-job" for row in game.rows)


def test_extractor_rejects_tampering_and_loader_censors_timeouts(tmp_path: Path) -> None:
    valid = _write_job(tmp_path, opponent_id="aster-id", opponent_name="Aster", game_index=1, completion_order=1)
    job = json.loads(valid.read_text(encoding="utf-8"))
    job["final_game"]["game_state"]["history"][0]["offer"]["price"] = 999
    with pytest.raises(ValueError, match="final game SHA-256 mismatch"):
        extract_negotiation_game(job, job_path=valid)
    _write_job(tmp_path, opponent_id="aster-id", opponent_name="Aster", game_index=2, completion_order=2, terminal="timeout", outcome="timeout")
    games, rejected = load_negotiation_corpus(tmp_path)
    assert [game.game_id for game in games] == ["aster-id-game-01"]
    assert any(record["reason"] == "censored_terminal_state" for record in rejected)


def test_model_separates_population_target_and_visible_prefix() -> None:
    def extracted(opponent_id: str, game_id: int, price: float) -> Any:
        history = [_entry(round_number=1, from_player="player_2", price=price, decision="RejectOffer")]
        state = {"complete_information": False, "current_player": "player_1", "game_family": "negotiation", "history": history, "horizon_known": True, "max_rounds": 10, "messages_allowed": True, "phase": "completed", "player_1_role": "seller", "player_1_value": 80.0, "player_2_role": "buyer", "round": 1, "result": {"outcome": "no_deal"}}
        final = {"game_family": "negotiation", "game_id": f"{opponent_id}-{game_id}", "game_state": state, "opponent": {"name": opponent_id, "type": "agent"}, "result": state["result"], "status": "completed", "your_player": "player_1"}
        job = {"game_family": "negotiation", "game_id": final["game_id"], "completed_at": f"2026-08-09T00:00:0{game_id}+00:00", "completion_order": game_id, "job_id": f"job-{opponent_id}-{game_id}", "opponent": {"id": opponent_id, "name": opponent_id}, "final_game_sha256": _hash(final), "final_game": final}
        return extract_negotiation_game(job, job_path=Path(f"{opponent_id}-{game_id}.json"), job_sha256=f"sha-{opponent_id}-{game_id}")

    population = extracted("population", 1, 75.0)
    target = extracted("target", 2, 45.0)
    current = extracted("target", 3, 60.0).rows[0]
    prefix = [extracted("target", 4, 70.0).rows[0]]
    model = NegotiationOpponentModelV2(NegotiationModelConfig(proposal_floor_equivalent_rows=0.05))
    forecasts = model.proposal_forecast(
        current,
        population_prior=[PriorObservation(population.rows[0], 0, 0)],
        target_prior=[PriorObservation(target.rows[0], 1, 0)],
        prefix=prefix,
        current_global_game_index=3,
        current_target_game_index=2,
    )
    population_mean = forecasts["population_kernel"].mean
    target_mean = forecasts["target_kernel"].mean
    adaptive_mean = forecasts["adaptive_v2"].mean
    assert target_mean > population_mean
    assert adaptive_mean < target_mean


def test_rolling_validation_is_restart_safe_and_records_whole_game_origins(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for game_index in range(5):
        _write_job(source, opponent_id="aster-id", opponent_name="Aster", game_index=game_index, completion_order=2 * game_index, history=[_entry(round_number=1, from_player="player_1", price=95 + game_index, decision="AcceptOffer" if game_index % 2 else "RejectOffer"), _entry(round_number=2, from_player="player_2", price=70 - game_index, decision="RejectOffer")])
        _write_job(source, opponent_id="beryl-id", opponent_name="Beryl", game_index=game_index, completion_order=2 * game_index + 1, history=[_entry(round_number=1, from_player="player_1", price=110 + game_index, decision="RejectOffer"), _entry(round_number=2, from_player="player_2", price=60 + game_index, decision="RejectOffer")])
    output = tmp_path / "output"
    project_root = Path(__file__).resolve().parents[1]
    result = NegotiationRollingValidationV2(
        dossier_root=source,
        output_dir=output,
        project_root=project_root,
        validation_config=NegotiationValidationConfig(min_games=4, warmup_games=2),
        bootstrap_replicates=20,
        bootstrap_seed=17,
    ).run()
    assert result["corpus"]["game_count"] == 10
    assert result["evaluation"]["origin_count"] == 6
    assert result["evaluation"]["decision_count"] == 12
    origins = [json.loads(line) for line in (output / "origins.jsonl").read_text(encoding="utf-8").splitlines()]
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(origin["prior_target_game_count"] >= 2 for origin in origins)
    assert [record["prefix_action_count"] for record in predictions[:2]] == [0, 1]
    assert result["evaluation"]["complete_game_macro"]["response"]["adaptive_v2"]["group_count"] == 6
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        NegotiationRollingValidationV2(dossier_root=source, output_dir=output, project_root=project_root, validation_config=NegotiationValidationConfig(min_games=4, warmup_games=2), bootstrap_replicates=0).run()
