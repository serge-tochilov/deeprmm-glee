import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nommd_arena.glee_nommd import GLEE_MAIN_DESIRE, GleeNommdMemory, glee_action_model, nommd_action_model, validate_tetrad_transport
from nommd_arena.glee_policy import action_model


def _persuasion_turn() -> dict[str, Any]:
    return {
        "game_id": "game-memory-1",
        "game_family": "persuasion",
        "your_player": "player_2",
        "phase": "buyer_decision",
        "opponent": {"type": "agent", "name": "OpponentAlpha"},
        "prompt": "Official persuasion prompt",
        "game_state": {
            "current_player": "player_2",
            "history": [],
            "p": 0.8,
            "u": 0,
            "v": 125,
            "product_price": 100,
            "round": 1,
            "total_rounds": 20,
            "seller_message": "This unit is excellent.",
            "seller_message_type": "text",
            "buyer_total_payoff": 0,
            "seller_total_payoff": 0,
        },
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
    }


def _belief_trace(content: str) -> dict[str, object]:
    return {
        "kind": "belief",
        "disposition": "affirmed",
        "content": content,
        "strength": 70,
        "salience": 70,
        "mental_path": ["HIDDEN_OPPONENT"],
        "source_slots": [0],
        "tags": ["seller"],
    }


def test_nommd_action_schema_keeps_game_action_and_sparse_tetrad_separate() -> None:
    game = _persuasion_turn()
    model_cls = nommd_action_model(action_model(game))
    properties = model_cls.model_json_schema()["properties"]
    assert "update" in properties
    assert "tetrad_update" not in properties
    parsed = model_cls.model_validate({"action": {"decision": "no"}, "update": {"updates": []}})
    assert parsed.action.decision == "no"
    assert parsed.update.updates == []


def test_active_glee_action_schema_rejects_the_retired_tetrad_carrier() -> None:
    model_cls = glee_action_model(action_model(_persuasion_turn()))
    assert set(model_cls.model_json_schema()["properties"]) == {"action"}
    parsed = model_cls.model_validate({"action": {"decision": "no"}})
    assert parsed.action.decision == "no"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_cls.model_validate({"action": {"decision": "no"}, "update": None})


def test_nommd_action_schema_treats_three_updates_as_a_soft_cap() -> None:
    game = _persuasion_turn()
    model_cls = nommd_action_model(action_model(game))
    parsed = model_cls.model_validate({"action": {"decision": "no"}, "update": {"updates": [_belief_trace(f"belief {index}") for index in range(4)]}})
    update, issues = validate_tetrad_transport(parsed.update)
    assert update is not None and len(update.updates) == 4
    assert issues == []


def test_invalid_cognitive_semantics_do_not_invalidate_the_game_action() -> None:
    game = _persuasion_turn()
    model_cls = nommd_action_model(action_model(game))
    parsed = model_cls.model_validate(
        {
            "action": {"decision": "yes"},
            "update": {
                "updates": [
                    {
                        "kind": "belief",
                        "disposition": "active",
                        "content": "The seller appears confident.",
                        "strength": 70,
                        "salience": 70,
                        "mental_path": ["HIDDEN_OPPONENT"],
                        "source_slots": [0],
                        "tags": ["seller"],
                    }
                ]
            },
        }
    )
    update, issues = validate_tetrad_transport(parsed.update)
    assert parsed.action.decision == "yes"
    assert update is not None and update.updates == []
    assert issues == ["trace 1 discarded at trace: Value error, belief traces cannot use active disposition"]


def test_glee_memory_persists_named_opponent_and_processes_observations(tmp_path: Path) -> None:
    memory = GleeNommdMemory(root=tmp_path / "memory", agent_name="DeepRMM-01", retrieval_limit=12, decay=0.5)
    game = _persuasion_turn()
    opponent = memory.observe(game)
    context, metadata = memory.context(game, opponent)
    assert opponent == "OpponentAlpha"
    assert context["main_desire"] == GLEE_MAIN_DESIRE
    assert context["opponent_mind_label"] == "OpponentAlpha"
    assert len(context["new_observations"]) == 2
    receipt = memory.commit(None, metadata)
    assert receipt["submitted_updates"] == 0
    assert memory.ledger.summary()["participants"]["DeepRMM-01"]["unprocessed"] == 0
    reloaded = GleeNommdMemory(root=tmp_path / "memory", agent_name="DeepRMM-01", retrieval_limit=12, decay=0.5)
    assert "OpponentAlpha" in reloaded.ledger.summary()["known_minds"]


def test_glee_memory_records_accepted_action_for_next_turn(tmp_path: Path) -> None:
    memory = GleeNommdMemory(root=tmp_path / "memory", agent_name="DeepRMM-01", retrieval_limit=12, decay=0.5)
    game = _persuasion_turn()
    opponent = memory.observe(game)
    _context, metadata = memory.context(game, opponent)
    memory.publish_action(game, {"decision": "no"}, {"valid": True, "game_over": False, "result": None})
    memory.commit(None, metadata)
    pending = memory.ledger.pending("DeepRMM-01")
    assert len(pending) == 1
    assert "Accepted own action" in pending[0].content
    records = [json.loads(line) for line in (tmp_path / "memory" / "DeepRMM-01" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(record["kind"] == "desire" and "final ranking" in record["content"] for record in records)


def test_glee_memory_bounds_legal_long_messages_without_losing_receipt_identity(tmp_path: Path) -> None:
    memory = GleeNommdMemory(root=tmp_path / "memory", agent_name="DeepRMM-01", retrieval_limit=12, decay=0.5)
    game = _persuasion_turn()
    game["game_state"]["seller_message"] = "x" * 2000
    memory.observe(game)
    pending = memory.ledger.pending("DeepRMM-01")
    turn = next(record for record in pending if "Current visible persuasion turn" in record.content)
    assert len(turn.content) == 1200
    assert "ledger projection truncated; sha256=" in turn.content
