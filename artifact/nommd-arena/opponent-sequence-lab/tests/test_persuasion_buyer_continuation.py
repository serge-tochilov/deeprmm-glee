from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from glee_sequence_lab.data import CorpusVocabs
from glee_sequence_lab.buyer_continuation_live import BUYER_CONTINUATION_AUTHORITY, BUYER_CONTINUATION_LIVE_CONTRACT, BuyerContinuationProspectiveRegistry
from glee_sequence_lab.model import HierarchicalSequenceTwin, ModelConfig
from glee_sequence_lab.persuasion_buyer_continuation import BUYER_CONTINUATION_LABELS, _paired_game_bootstrap, _reset_reversed_head, derive_buyer_continuation_targets


def _event(*, index: int, round_number: int, actor: str, kind: str, action: str, delay: float | None = None) -> dict[str, object]:
    return {"game_id": "g", "event_index": index, "round_number": round_number, "actor": actor, "kind": kind, "action_label": action, "response_time_ms": delay}


def _vocabs() -> CorpusVocabs:
    return CorpusVocabs.from_receipt(
        {
            "actor": ["<pad>", "<unk>", "<bos>", "self", "opponent", "environment"],
            "kind": ["<pad>", "<unk>", "<bos>", "signal", "response", "quality_reveal"],
            "event_action": ["<pad>", "<unk>", "<bos>", "signal_positive", "signal_negative", "signal_unknown", "buy", "pass", "quality_high", "quality_low"],
            "message_act": ["<pad>", "<unk>", "<bos>", "recommend", "discourage", "other"],
            "quality": ["<pad>", "<unk>", "<bos>", "high", "low"],
            "discourse": ["<unk>"],
            "our_role": ["<pad>", "<unk>", "<bos>", "buyer"],
            "opponent_role": ["<pad>", "<unk>", "<bos>", "seller"],
            "identity_scope": ["<pad>", "<unk>", "<bos>", "known", "hidden"],
            "account": ["<population>"],
            "target_labels": {
                "bargaining": ["proposal", "accept", "reject", "walkaway"],
                "negotiation": ["proposal", "accept", "reject", "walkaway"],
                "persuasion": ["signal_positive", "signal_negative", "signal_unknown", "buy", "pass"],
            },
            "target_messages": {"bargaining": ["none"], "negotiation": ["none"], "persuasion": ["none"]},
        }
    )


def test_buyer_continuation_skips_post_action_quality_and_terminal_response() -> None:
    game = {"game_id": "g", "family": "persuasion", "our_role": "buyer", "chronological_split": "train", "identity_scope": "known", "account_key": "participant-alpha", "account_confidence": "high", "account_fold": 0}
    events = [
        _event(index=0, round_number=1, actor="opponent", kind="signal", action="signal_positive"),
        _event(index=1, round_number=1, actor="self", kind="response", action="buy", delay=4_000),
        _event(index=2, round_number=1, actor="environment", kind="quality_reveal", action="quality_low"),
        _event(index=3, round_number=2, actor="opponent", kind="signal", action="signal_negative"),
        _event(index=4, round_number=2, actor="self", kind="response", action="pass", delay=2_000),
    ]
    targets, exclusions = derive_buyer_continuation_targets(game, events, source_type="deeprmm", source_agent="DeepRMM-01")
    assert len(targets) == 1
    assert targets[0]["prefix_length"] == 2
    assert targets[0]["target_event_index"] == 3
    assert targets[0]["buyer_action"] == "buy"
    assert targets[0]["preceding_signal"] == "signal_positive"
    assert targets[0]["target_label"] == "signal_negative"
    assert targets[0]["mask_last_prefix_response_time"] is True
    assert targets[0]["account_fold"] == 0
    assert exclusions == {"terminal-response": 1}


def test_reversed_head_reinitializes_only_persuasion_action_path() -> None:
    config = ModelConfig(core="gru", model_dim=32, hidden_dim=40, event_streams="separate-head-gated", message_model_dim=16, message_hidden_dim=20)
    model = HierarchicalSequenceTwin(_vocabs(), config)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    trainable = _reset_reversed_head(model, config, seed=17)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    assert model.heads["persuasion"].action.out_features == len(BUYER_CONTINUATION_LABELS)
    assert set(trainable) == {parameter for parameter in model.parameters() if parameter.requires_grad}
    assert all(not parameter.requires_grad for parameter in model.cores["persuasion"].parameters())
    assert all(parameter.requires_grad for parameter in model.heads["persuasion"].action_fusion.parameters())


def test_fieldglass_gate_bootstrap_is_paired_by_game() -> None:
    baseline = [
        {"sample_id": "a", "game_id": "g1", "actual": 0, "probabilities": [0.6, 0.3, 0.1]},
        {"sample_id": "b", "game_id": "g2", "actual": 1, "probabilities": [0.3, 0.6, 0.1]},
    ]
    improved = [
        {"sample_id": "a", "game_id": "g1", "actual": 0, "probabilities": [0.8, 0.1, 0.1]},
        {"sample_id": "b", "game_id": "g2", "actual": 1, "probabilities": [0.1, 0.8, 0.1]},
    ]
    result = _paired_game_bootstrap(baseline, improved, replicates=200, seed=23)
    assert result["arm_minus_baseline_game_macro_nll"] < 0
    assert result["bootstrap_95_percent_interval"][1] < 0


def test_buyer_continuation_registry_is_idempotent_but_immutable(tmp_path: Path) -> None:
    registry = BuyerContinuationProspectiveRegistry(tmp_path / "registry.sqlite3")
    prediction = {
        "contract": BUYER_CONTINUATION_LIVE_CONTRACT,
        "release_id": "continuation-v1",
        "authority": BUYER_CONTINUATION_AUTHORITY,
        "game_id": "game-1",
        "target_round": 4,
        "source_turn_id": "turn-3",
        "candidate_set_sha256": "a" * 64,
        "prefix_sha256": "b" * 64,
        "rows": [],
    }
    assert registry.register(prediction)["status"] == "registered"
    assert registry.register(prediction)["status"] == "already-registered"
    changed = {**prediction, "candidate_set_sha256": "c" * 64}
    with pytest.raises(ValueError, match="different immutable evidence"):
        registry.register(changed)
    assert registry.summary()["predictions"] == 1
    registry.close()
