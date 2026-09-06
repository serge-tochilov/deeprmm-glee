import json
import math
from pathlib import Path

import polars as pl
import pytest

from glee_sequence_lab.corpus import EVENT_SCHEMA, GAME_SCHEMA, _artifact
from glee_sequence_lab.experiment import _selection_score, validate_model_config
from glee_sequence_lab.model import ModelConfig
from glee_sequence_lab.self_mirror import PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR, PublicSelfMirrorCorpusBuilder


def _game(game_id: str, family: str, split: str, *, identity_scope: str = "hidden") -> dict[str, object]:
    return {
        "game_id": game_id,
        "family": family,
        "source_type": "real",
        "generator_id": None,
        "started_at": f"2026-08-0{len(game_id)}T00:00:00Z",
        "completed_at": f"2026-08-0{len(game_id)}T00:01:00Z",
        "chronological_split": split,
        "identity_scope": identity_scope,
        "opponent_name_hash": "opponent-hash",
        "account_key": "participant-alpha",
        "account_confidence": "very-high",
        "account_fold": 2,
        "our_player": "player_1",
        "our_role": "buyer" if family == "negotiation" else "seller" if family == "persuasion" else "player_1",
        "opponent_role": "seller" if family == "negotiation" else "buyer" if family == "persuasion" else "player_2",
        "complete_information": False,
        "horizon_known": True,
        "messages_allowed": True,
        "max_rounds": 20,
        "static_scale_log": math.log1p(100.0),
        "static_self_value": 100.0,
        "static_visible_opponent_value": 80.0,
        "static_environment_probability": 0.5 if family == "persuasion" else None,
        "static_aux_value": math.log1p(10.0) if family == "persuasion" else None,
        "static_seller_knows_quality": family == "persuasion",
        "engine_version": "private-engine",
        "advisor_version": "private-advisor",
        "policy_revision": "private-policy",
        "archive_path": f"source/{game_id}.json",
        "archive_sha256": f"sha-{game_id}",
    }


def _event(game_id: str, event_index: int, *, actor: str, kind: str, action_label: str, action_value: float | None = None, action_aux_value: float | None = None, visible_quality: str | None = None, response_time_ms: float | None = None, message: str = "") -> dict[str, object]:
    return {
        "game_id": game_id,
        "event_index": event_index,
        "round_number": event_index + 1,
        "round_phase": event_index / 4,
        "actor": actor,
        "kind": kind,
        "action_label": action_label,
        "action_value": action_value,
        "action_aux_value": action_aux_value,
        "response_time_ms": response_time_ms,
        "visible_quality": visible_quality,
        "message_present": bool(message),
        "message_family_act": "recommend" if message else "none",
        "message_discourse_acts": ["recommend"] if message else ["silence"],
        "message_hash_bins": [7] if message else [],
        "message_chars": len(message),
        "message_words": len(message.split()),
        "message_uppercase_ratio": 0.0,
        "message_digit_ratio": 0.0,
        "message_question_marks": 0,
        "message_exclamation_marks": 0,
        "message_commas": 0,
        "message_semicolons": 0,
        "message_currency_marks": 0,
        "message_percent_marks": 0,
        "message_decimal_numbers": 0,
        "message_sha256": "message-hash" if message else "empty-hash",
    }


def _source_corpus(path: Path) -> None:
    path.mkdir()
    games = pl.DataFrame(
        [
            _game("b", "bargaining", "train"),
            _game("n", "negotiation", "validation"),
            _game("p", "persuasion", "test", identity_scope="known"),
        ],
        schema=GAME_SCHEMA,
        strict=False,
    )
    demand = math.log(1.2)
    events = pl.DataFrame(
        [
            _event("b", 0, actor="opponent", kind="proposal", action_label="proposal", action_value=0.6, action_aux_value=0.4, message="opening"),
            _event("b", 1, actor="self", kind="response", action_label="reject", action_value=0.6, action_aux_value=0.4, response_time_ms=900.0),
            _event("n", 0, actor="self", kind="proposal", action_label="proposal", action_value=math.tanh(demand / 3.0), action_aux_value=demand, message="offer"),
            _event("p", 0, actor="self", kind="signal", action_label="signal_positive", action_value=1.0, visible_quality="high", message="buy"),
            _event("p", 1, actor="opponent", kind="response", action_label="buy", action_value=1.0, response_time_ms=400.0),
            _event("p", 2, actor="environment", kind="quality_reveal", action_label="quality_high", action_value=1.0, visible_quality="high"),
            _event("p", 3, actor="self", kind="signal", action_label="signal_negative", action_value=-1.0, visible_quality="low", message="pass"),
        ],
        schema=EVENT_SCHEMA,
        strict=False,
    )
    artifacts = {
        "games.parquet": _artifact(games, path / "games.parquet"),
        "events.parquet": _artifact(events, path / "events.parquet"),
    }
    (path / "manifest.json").write_text(json.dumps({"contract": "glee-post-planner-action-conditional-corpus-v3", "status": "frozen-retrospective-core-corpus", "artifacts": artifacts}), encoding="utf-8")


def test_public_self_mirror_masks_private_inputs_and_targets_self(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "mirror"
    _source_corpus(source)
    result = PublicSelfMirrorCorpusBuilder(source_corpus=source, output_dir=output).run()
    assert result["inventory"]["targets"] == 4
    games = {row["game_id"]: row for row in pl.read_parquet(output / "games.parquet").to_dicts()}
    assert games["b"]["static_self_value"] is None
    assert games["b"]["static_visible_opponent_value"] is None
    assert games["b"]["account_key"] is None
    assert games["b"]["engine_version"] == ""
    assert games["p"]["account_key"] == "participant-alpha"
    events = pl.read_parquet(output / "events.parquet").sort(["game_id", "event_index"]).to_dicts()
    negotiation = next(row for row in events if row["game_id"] == "n")
    assert math.isclose(float(negotiation["action_value"]), math.log1p(120.0) / PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR, rel_tol=0.0, abs_tol=1e-10)
    assert negotiation["action_aux_value"] is None
    persuasion = [row for row in events if row["game_id"] == "p"]
    assert persuasion[0]["visible_quality"] is None
    assert persuasion[2]["visible_quality"] == "high"
    assert persuasion[3]["visible_quality"] is None
    targets = pl.read_parquet(output / "targets.parquet")
    joined = targets.join(pl.read_parquet(output / "events.parquet").select("game_id", "event_index", "actor"), left_on=["game_id", "target_event_index"], right_on=["game_id", "event_index"])
    assert set(joined["actor"].to_list()) == {"self"}
    assert not targets["target_message_present"].any()
    assert targets.filter(pl.col("prefix_length") != pl.col("target_event_index")).is_empty()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_information_masks"]["unrevealed_persuasion_quality"] is True
    assert manifest["promotion"].startswith("offline shadow")


def test_joint_self_mirror_selection_uses_action_and_offer_value() -> None:
    metrics = {
        "by_family": {
            "bargaining": {"action": {"negative_log_likelihood": 0.4}, "value": {"mean_absolute_error": 0.2}},
            "negotiation": {"action": {"negative_log_likelihood": 0.6}, "value": {"mean_absolute_error": 0.4}},
            "persuasion": {"action": {"negative_log_likelihood": 0.3}, "value": {"mean_absolute_error": None}},
        }
    }
    assert math.isclose(_selection_score(metrics, objective="joint-self-mirror"), (0.3 + 0.5 + 0.3) / 3)


def test_mamba2_rejects_an_incompatible_message_stream_width() -> None:
    with pytest.raises(ValueError, match="message_model_dim"):
        validate_model_config(ModelConfig(core="mamba2", event_streams="separate-head-gated", message_model_dim=24))


def test_mamba2_validates_the_expanded_width_used_by_the_kernel() -> None:
    validate_model_config(ModelConfig(core="mamba2", model_dim=80, mamba_head_dim=32))
    with pytest.raises(ValueError, match="model_dim"):
        validate_model_config(ModelConfig(core="mamba2", model_dim=72, mamba_head_dim=32))
