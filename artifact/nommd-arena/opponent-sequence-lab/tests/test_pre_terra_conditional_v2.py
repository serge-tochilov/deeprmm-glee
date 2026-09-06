from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import torch

from glee_sequence_lab.conditional_experiment import ConditionalFeatureModel, ConditionalTrainingConfig, _fit_stack
from glee_sequence_lab.conditional_release import ConditionalTwinRelease
from glee_sequence_lab.corpus import file_sha256, object_sha256
from glee_sequence_lab.pre_terra_conditional_v2 import CONDITIONAL_CORPUS_CONTRACT, CandidateAction, CandidateActionProjector, ConditionalCorpusBuilder, merge_sparse_vectors
from glee_sequence_lab.pre_terra_v3 import PRE_TERRA_V3_CORPUS_CONTRACT


def _write_source_corpus(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    games = pl.DataFrame([{"game_id": "g-1", "family": "bargaining"}])
    events = pl.DataFrame(
        [
            {"game_id": "g-1", "event_index": 0, "round_number": 1, "round_phase": 0.0, "actor": "opponent", "kind": "proposal", "action_label": "proposal", "action_value": 0.4, "action_aux_value": 0.6, "response_time_ms": 100.0, "visible_quality": None},
            {"game_id": "g-1", "event_index": 1, "round_number": 1, "round_phase": 0.0, "actor": "self", "kind": "proposal", "action_label": "proposal", "action_value": 0.6, "action_aux_value": 0.4, "response_time_ms": 2_000.0, "visible_quality": None},
            {"game_id": "g-1", "event_index": 2, "round_number": 1, "round_phase": 0.0, "actor": "opponent", "kind": "response", "action_label": "accept", "action_value": None, "action_aux_value": None, "response_time_ms": 500.0, "visible_quality": None},
        ]
    )
    targets = pl.DataFrame([{"sample_id": "s-1", "game_id": "g-1", "target_event_index": 2, "prefix_length": 1, "target_label": "accept", "chronological_split": "train"}])
    base_payload = {"indices": [3, 1_100], "values": [0.5, -0.25]}
    features = pl.DataFrame([{"sample_id": "s-1", "game_id": "g-1", "turn_id": "t-1", "family": "bargaining", "phase": "response", "round_number": 1, "feature_indices": base_payload["indices"], "feature_values": base_payload["values"], "feature_vector_sha256": object_sha256(base_payload)}])
    artifacts: dict[str, object] = {}
    for name, frame in (("games.parquet", games), ("events.parquet", events), ("targets.parquet", targets), ("features.parquet", features)):
        path = source / name
        frame.write_parquet(path)
        artifacts[name] = {"sha256": file_sha256(path)}
    (source / "manifest.json").write_text(json.dumps({"contract": PRE_TERRA_V3_CORPUS_CONTRACT, "status": "frozen-retrospective-core-corpus", "artifacts": artifacts}) + "\n", encoding="utf-8")
    return source


def test_candidate_action_projection_is_deterministic_and_action_sensitive() -> None:
    projector = CandidateActionProjector()
    first = CandidateAction(family="bargaining", phase="response", kind="proposal", action_label="proposal", action_value=0.55, action_aux_value=0.45, round_number=2, round_phase=0.25)
    second = CandidateAction(family="bargaining", phase="response", kind="proposal", action_label="proposal", action_value=0.65, action_aux_value=0.35, round_number=2, round_phase=0.25)
    projected = projector.project(first)
    assert projected == projector.project(first)
    assert projected.vector_sha256 != projector.project(second).vector_sha256
    merged = merge_sparse_vectors([3, 1_100], [0.5, -0.25], projected)
    assert merged.vector_sha256 == merge_sparse_vectors([3, 1_100], [0.5, -0.25], projected).vector_sha256


def test_live_actions_project_into_training_time_bridge_coordinates() -> None:
    bargaining = {
        "game_id": "b",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": {}, "bob_gain": {}, "message": {}}},
        "game_state": {"round": 2, "money_to_divide": 100.0, "current_player": "player_1", "horizon_known": True, "max_rounds": 5, "messages_allowed": True},
    }
    b_candidate = CandidateAction.from_live_action(game=bargaining, action={"alice_gain": 60, "bob_gain": 40, "message": "settle"})
    assert b_candidate.action_value == 0.4
    assert b_candidate.action_aux_value == 0.6
    assert b_candidate.message_present is True
    assert b_candidate.message_words == 1
    negotiation = {
        "game_id": "n",
        "game_family": "negotiation",
        "your_player": "player_1",
        "phase": "offer",
        "valid_actions": {"type": "offer", "fields": {"product_price": {}, "message": {}}},
        "game_state": {"round": 1, "current_player": "player_1", "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 40.0, "player_2_value": 100.0, "complete_information": True, "messages_allowed": True},
    }
    n_candidate = CandidateAction.from_live_action(game=negotiation, action={"product_price": 70.0, "message": "split"})
    assert n_candidate.kind == "proposal"
    assert n_candidate.action_value is not None
    assert n_candidate.action_aux_value == 0.5
    persuasion = {
        "game_id": "p",
        "game_family": "persuasion",
        "your_player": "player_1",
        "phase": "seller_recommendation",
        "valid_actions": {"type": "seller_recommendation", "fields": {"decision": ["yes", "no"]}},
        "game_state": {"round": 3, "total_rounds": 10, "current_player": "player_1", "current_quality": "low", "seller_message_type": "binary"},
    }
    p_candidate = CandidateAction.from_live_action(game=persuasion, action={"decision": "no"})
    assert p_candidate.action_label == "signal_negative"
    assert p_candidate.visible_quality == "low"


def test_same_polarity_persuasion_wording_changes_candidate_projection() -> None:
    game = {
        "game_id": "p-text",
        "game_family": "persuasion",
        "your_player": "player_1",
        "phase": "seller_message",
        "valid_actions": {"type": "seller_message", "fields": {"message": "string"}},
        "game_state": {"round": 4, "total_rounds": 20, "current_player": "player_1", "current_quality": "high", "seller_message_type": "text"},
    }
    concise = CandidateAction.from_live_action(game=game, action={"message": "This is high quality; buy."})
    contextual = CandidateAction.from_live_action(game=game, action={"message": "The product is high quality, consistent with the reliable offers you accepted earlier."})
    assert concise.action_label == contextual.action_label == "signal_positive"
    assert concise.message_sha256 != contextual.message_sha256
    projector = CandidateActionProjector()
    assert projector.project(concise).vector_sha256 != projector.project(contextual).vector_sha256


def test_conditional_corpus_adds_bridge_without_copying_large_source_tables(tmp_path: Path) -> None:
    source = _write_source_corpus(tmp_path)
    output = tmp_path / "conditional"
    result = ConditionalCorpusBuilder(source_corpus=source, output_dir=output).run()
    assert result["contract"] == CONDITIONAL_CORPUS_CONTRACT
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_labels"]["persuasion"] == ["buy", "pass"]
    assert os.stat(source / "games.parquet").st_ino == os.stat(output / "games.parquet").st_ino
    assert os.stat(source / "events.parquet").st_ino == os.stat(output / "events.parquet").st_ino
    target = pl.read_parquet(output / "targets.parquet").row(0, named=True)
    feature = pl.read_parquet(output / "features.parquet").row(0, named=True)
    assert target["pre_candidate_prefix_length"] == 1
    assert target["bridge_event_index"] == 1
    assert target["prefix_length"] == 2
    assert target["mask_last_prefix_future_fields"] is False
    assert target["mask_last_prefix_response_time"] is True
    assert target["mask_last_prefix_message_fields"] is False
    assert feature["candidate_action_label"] == "proposal"
    assert feature["candidate_action_value"] == 0.6
    candidate_payload = {"indices": feature["candidate_feature_indices"], "values": feature["candidate_feature_values"]}
    assert object_sha256(candidate_payload) == feature["candidate_feature_vector_sha256"]


def test_conditional_feature_model_and_convex_stack_are_family_specific() -> None:
    config = ConditionalTrainingConfig(hidden_dim=16, latent_dim=8, dropout=0.0, input_dropout=0.0, stack_grid_steps=10)
    model = ConditionalFeatureModel(config)
    logits = model(torch.randn(6, 2_048), torch.tensor([0, 0, 1, 1, 2, 2]))
    assert logits.shape == (6, 3)
    assert torch.all(logits[4:, 2] < -1e8)
    corpus = SimpleNamespace(
        indices_by_split={"validation": torch.arange(6)},
        families=torch.tensor([0, 0, 1, 1, 2, 2]),
        labels=torch.tensor([0, 0, 0, 0, 0, 0]),
    )
    sequence = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.8, 0.2, 0.0], [0.8, 0.2, 0.0]])
    engineered = torch.tensor([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.8, 0.2, 0.0], [0.8, 0.2, 0.0]])
    weights = _fit_stack(corpus, sequence, engineered, config)
    assert weights == {"bargaining": 1.0, "negotiation": 0.0, "persuasion": 0.0}


def test_candidate_substitution_preserves_authenticated_prefix_and_changes_only_bridge() -> None:
    release = ConditionalTwinRelease.__new__(ConditionalTwinRelease)
    release.projector = CandidateActionProjector()
    sample = {
        "game": {"game_id": "g", "family": "bargaining", "identity_scope": "hidden"},
        "events": [{"game_id": "g", "event_index": 0, "actor": "opponent", "kind": "proposal", "action_label": "proposal", "action_value": 0.4}],
        "target": {"sample_id": "s", "game_id": "g", "identity_scope": "hidden"},
    }
    before = json.loads(json.dumps(sample))
    base = {"indices": [4], "values": [0.5]}
    candidates = [
        {"kind": "proposal", "action_label": "proposal", "action_value": 0.5, "action_aux_value": 0.5, "round_number": 1, "round_phase": 0.0},
        {"kind": "proposal", "action_label": "proposal", "action_value": 0.6, "action_aux_value": 0.4, "round_number": 1, "round_phase": 0.0},
    ]
    prepared, features, parsed = release.prepare_candidates(sample=sample, family="bargaining", phase="response", base_feature_indices=base["indices"], base_feature_values=base["values"], base_feature_vector_sha256=object_sha256(base), candidates=candidates)
    assert sample == before
    assert prepared[0]["events"][:-1] == prepared[1]["events"][:-1] == sample["events"]
    assert prepared[0]["events"][-1]["action_value"] == 0.5
    assert prepared[1]["events"][-1]["action_value"] == 0.6
    assert not any(value["target"]["mask_last_prefix_future_fields"] for value in prepared)
    assert all(value["target"]["mask_last_prefix_response_time"] for value in prepared)
    assert not any(value["target"]["mask_last_prefix_message_fields"] for value in prepared)
    assert not torch.equal(features[0], features[1])
    assert [value.action_value for value in parsed] == [0.5, 0.6]
