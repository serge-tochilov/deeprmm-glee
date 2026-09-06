from __future__ import annotations

import hashlib
import json
from collections import Counter

import polars as pl
import torch
from torch.nn import functional as F

from glee_sequence_lab.corpus import file_sha256, object_sha256
from glee_sequence_lab.pre_terra_v3 import FEATURE_DIMENSION, EngineeredFeatureHasher, _read_prompt_snapshots, _source_run_receipt
from glee_sequence_lab.v3_fusion import V3ActionModel, V3Corpus, V3TrainingConfig, run_v3_fusion_suite


def _bundle() -> dict[str, object]:
    return {
        "contract": "glee-terra-synthetic-feature-bundle-v1",
        "frontier": "exact-worker-payload-before-terra",
        "game_family": "bargaining",
        "phase": "offer",
        "producer_keys": ["bargaining_opponent_model_v2"],
        "features": {
            "bargaining_opponent_model_v2": {
                "status": "available",
                "acceptance_probability": 0.625,
                "decision": {"action": "reject", "support": 7},
                "revision": 91,
                "model_version": "future-only-v99",
                "state_sha256": "a" * 64,
                "interpretation": "x" * 300,
            }
        },
    }


def test_archived_prompt_loader_accepts_integrity_checked_inline_payload(tmp_path) -> None:
    payload = {
        "game_family": "bargaining",
        "phase": "offer",
        "turn_receipt": {"turn_id": "turn-inline"},
        "bargaining_opponent_model_v2": {"status": "available", "acceptance_probability": 0.5},
    }
    user = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    run = tmp_path / "run"
    run.mkdir()
    (run / "llm_calls.jsonl").write_text(
        json.dumps(
            {
                "role": "glee_nommd_bargaining",
                "ts": "2026-08-09T00:00:00Z",
                "request": {"user": user, "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exclusions: Counter[str] = Counter()
    snapshots = _read_prompt_snapshots(run, exclusions)
    assert set(snapshots) == {"turn-inline"}
    assert snapshots["turn-inline"].family == "bargaining"
    assert snapshots["turn-inline"].bundle["phase"] == "offer"
    assert not any(exclusions.values())


def test_source_receipt_attests_explicit_unsealed_terminal_game_set(tmp_path) -> None:
    run = tmp_path / "run"
    games = run / "games"
    games.mkdir(parents=True)
    for name, body in {
        "manifest.json": {"run_id": "recovered"},
        "llm_calls.jsonl": {},
        "events.jsonl": {},
    }.items():
        (run / name).write_text(json.dumps(body) + "\n", encoding="utf-8")
    (games / "bargaining-game-1.json").write_text(json.dumps({"game_id": "game-1", "game_family": "bargaining", "status": "no_deal"}) + "\n", encoding="utf-8")
    receipt = _source_run_receipt(run, prompt_turns=3)
    assert receipt["source_state"] == "explicit-recovered-terminal-game-set"
    assert receipt["complete_sha256"] is None
    assert receipt["terminal_game_archives"] == 1
    assert len(str(receipt["terminal_game_archive_index_sha256"])) == 64


def test_engineered_feature_hash_is_order_invariant_and_excludes_provenance() -> None:
    hasher = EngineeredFeatureHasher()
    original = hasher.project(_bundle())
    reordered = _bundle()
    reordered["features"] = {"bargaining_opponent_model_v2": dict(reversed(list(reordered["features"]["bargaining_opponent_model_v2"].items())))}
    repeated = hasher.project(reordered)
    assert original.indices == repeated.indices
    assert original.values == repeated.values
    assert original.vector_sha256 == repeated.vector_sha256
    assert original.excluded_provenance_count == 3
    assert original.excluded_long_text_count == 1
    assert all(0 <= index < FEATURE_DIMENSION for index in original.indices)


def test_engineered_feature_hash_changes_for_strategic_numeric_and_categorical_values() -> None:
    hasher = EngineeredFeatureHasher()
    baseline = hasher.project(_bundle())
    changed = _bundle()
    changed["features"]["bargaining_opponent_model_v2"]["acceptance_probability"] = 0.125
    changed["features"]["bargaining_opponent_model_v2"]["decision"]["action"] = "accept"
    projected = hasher.project(changed)
    assert projected.vector_sha256 != baseline.vector_sha256


def test_late_fusion_is_bounded_and_backpropagates_into_feature_encoder() -> None:
    config = V3TrainingConfig(hidden_dim=16, latent_dim=8, dropout=0.0, input_dropout=0.0)
    model = V3ActionModel("late-fusion", config)
    features = torch.randn(6, FEATURE_DIMENSION)
    probabilities = torch.zeros(6, 5)
    mask = torch.zeros(6, 5, dtype=torch.bool)
    families = torch.tensor([0, 0, 1, 1, 2, 2])
    for index, family in enumerate(families.tolist()):
        classes = 5 if family == 2 else 4
        probabilities[index, :classes] = 1.0 / classes
        mask[index, :classes] = True
    labels = torch.tensor([0, 1, 2, 3, 4, 1])
    logits, gates = model(features, probabilities, families, mask)
    assert logits.shape == (6, 5)
    assert torch.all(gates > 0.0)
    assert torch.all(gates <= config.maximum_fusion_gate)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.feature_encoder.parameters())


def test_v3_suite_loads_verified_sparse_corpus_and_trains_all_arms(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    targets: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    sequence: list[dict[str, object]] = []
    games: list[dict[str, object]] = []
    labels_by_family = {
        "bargaining": ["proposal", "accept", "reject", "walkaway"],
        "negotiation": ["proposal", "accept", "reject", "walkaway"],
        "persuasion": ["signal_positive", "signal_negative", "signal_unknown", "buy", "pass"],
    }
    splits = ("train", "validation", "test")
    for family_index, (family, action_labels) in enumerate(labels_by_family.items()):
        for split_index, split in enumerate(splits):
            game_id = f"{family}-{split}"
            sample_id = f"sample-{family}-{split}"
            vector = {"indices": [family_index, 10 + split_index], "values": [1.0, 0.5]}
            games.append({"game_id": game_id, "family": family, "our_role": "seller" if family == "persuasion" else "player_1"})
            targets.append({"sample_id": sample_id, "game_id": game_id, "chronological_split": split, "identity_scope": "known" if split_index % 2 else "hidden", "target_label": action_labels[(family_index + split_index) % len(action_labels)]})
            features.append({"sample_id": sample_id, "game_id": game_id, "family": family, "phase": "offer", "round_number": 1, "source_call_ts": f"2026-08-1{split_index}T00:00:00Z", "feature_indices": vector["indices"], "feature_values": vector["values"], "feature_vector_sha256": object_sha256(vector)})
            canonical_probabilities = {label: (index + 1) / sum(range(1, len(action_labels) + 1)) for index, label in enumerate(action_labels)}
            frozen_labels = sorted(action_labels)
            sequence.append({"sample_id": sample_id, "game_id": game_id, "family": family, "labels": frozen_labels, "probabilities": [canonical_probabilities[label] for label in frozen_labels]})
    frames = {
        "games.parquet": pl.DataFrame(games),
        "events.parquet": pl.DataFrame({"game_id": ["unused"], "event_index": [0]}),
        "targets.parquet": pl.DataFrame(targets),
        "features.parquet": pl.DataFrame(features),
        "sequence-predictions.parquet": pl.DataFrame(sequence),
    }
    artifacts: dict[str, object] = {}
    for name, frame in frames.items():
        path = corpus_dir / name
        frame.write_parquet(path)
        artifacts[name] = {"sha256": file_sha256(path)}
    (corpus_dir / "manifest.json").write_text(json.dumps({"contract": "glee-pre-terra-feature-fusion-corpus-v3", "status": "frozen-retrospective-corpus", "artifacts": artifacts}) + "\n", encoding="utf-8")
    corpus = V3Corpus(corpus_dir)
    assert corpus.features.shape == (9, FEATURE_DIMENSION)
    for row_index, metadata in enumerate(corpus.metadata):
        action_labels = labels_by_family[metadata.family]
        denominator = sum(range(1, len(action_labels) + 1))
        assert torch.allclose(corpus.base_probabilities[row_index, : len(action_labels)], torch.tensor([(index + 1) / denominator for index in range(len(action_labels))]))
    output = tmp_path / "experiment"
    result = run_v3_fusion_suite(
        corpus_dir=corpus_dir,
        output_dir=output,
        config=V3TrainingConfig(seeds=(17,), epochs=2, patience=1, batch_size=3, evaluation_batch_size=3, hidden_dim=8, latent_dim=4, bootstrap_replicates=10),
    )
    assert result["status"] == "complete"
    assert set(result["runs"]) == {"sequence-raw", "sequence-calibrated-seed17", "engineered-only-seed17", "late-fusion-seed17", "late-fusion-ensemble"}
    assert (output / "test-predictions.parquet").is_file()
