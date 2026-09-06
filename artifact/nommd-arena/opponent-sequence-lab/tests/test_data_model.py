from __future__ import annotations

import pytest
import torch

from glee_sequence_lab.data import CorpusIndex, CorpusVocabs, SequenceCollator, Vocabulary
from glee_sequence_lab.model import HierarchicalSequenceTwin, ModelConfig, _bucket_mamba_sequence, sequence_twin_loss
from glee_sequence_lab.synthetic import SyntheticPolicyZooBuilder


def _vocabs() -> CorpusVocabs:
    event = lambda *values: Vocabulary.build(values)
    return CorpusVocabs(
        actor=event("self", "opponent", "environment"),
        kind=event("proposal", "response"),
        event_action=event("proposal", "accept", "reject"),
        message_act=event("none", "recommend"),
        quality=event("high", "low"),
        discourse=Vocabulary.build(("silence", "recommend"), specials=("<unk>",)),
        our_role=event("proposer"),
        opponent_role=event("responder"),
        identity_scope=event("known", "hidden"),
        account=Vocabulary.build(("participant-alpha",), specials=("<population>",)),
        target_labels={
            "bargaining": Vocabulary.build(("proposal", "accept", "reject", "walkaway"), specials=()),
            "negotiation": Vocabulary.build(("proposal", "accept", "reject", "walkaway"), specials=()),
            "persuasion": Vocabulary.build(("signal_positive", "signal_negative", "signal_unknown", "buy", "pass"), specials=()),
        },
        target_messages={family: Vocabulary.build(("none", "recommend"), specials=()) for family in ("bargaining", "negotiation", "persuasion")},
    )


def _batch(vocabs: CorpusVocabs) -> dict[str, object]:
    batch_size = 3
    sequence_length = 4
    message_hashes = 5
    return {
        "family": "bargaining",
        "event_numeric": torch.randn(batch_size, sequence_length, 19),
        "event_categorical": torch.zeros(batch_size, sequence_length, 5, dtype=torch.long),
        "discourse": torch.zeros(batch_size, sequence_length, len(vocabs.discourse)),
        "message_bins": torch.zeros(batch_size, sequence_length, message_hashes, dtype=torch.long),
        "message_bin_mask": torch.zeros(batch_size, sequence_length, message_hashes, dtype=torch.bool),
        "lengths": torch.tensor([4, 3, 2]),
        "static_numeric": torch.randn(batch_size, 13),
        "static_categorical": torch.zeros(batch_size, 3, dtype=torch.long),
        "accounts": torch.tensor([0, 1, 1]),
        "target_labels": torch.tensor([0, 1, 2]),
        "target_action_mask": torch.tensor([False, True, True]),
        "target_values": torch.tensor([0.0, 0.4, 0.6]),
        "target_value_mask": torch.tensor([False, True, True]),
        "target_delays": torch.tensor([0.0, 5.0, 6.0]),
        "target_delay_mask": torch.tensor([False, True, True]),
        "metadata": [],
    }


def test_gru_hierarchical_forward_and_population_ablation() -> None:
    vocabs = _vocabs()
    model = HierarchicalSequenceTwin(vocabs, ModelConfig(model_dim=32, hidden_dim=40, account_dim=8, message_hash_dim=6))
    batch = _batch(vocabs)
    outputs = model(batch, account_dropout=0.25)
    assert outputs["action_logits"].shape == (3, 4)
    assert "message_logits" not in outputs
    loss, parts = sequence_twin_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert set(parts) == {"total", "action", "value", "delay"}
    loss.backward()
    model.eval()
    population = model(batch, force_population=True)
    assert population["effective_accounts"].tolist() == [0, 0, 0]


def test_frozen_vocabulary_receipt_round_trip() -> None:
    vocabs = _vocabs()
    restored = CorpusVocabs.from_receipt(vocabs.receipt())
    assert restored.receipt() == vocabs.receipt()
    assert restored.target_labels["persuasion"].encode_strict("buy") == vocabs.target_labels["persuasion"].encode_strict("buy")


def test_causal_transformer_forward() -> None:
    vocabs = _vocabs()
    model = HierarchicalSequenceTwin(vocabs, ModelConfig(core="transformer", model_dim=32, hidden_dim=40, account_dim=8, message_hash_dim=6, transformer_heads=4))
    outputs = model(_batch(vocabs), force_population=True)
    assert outputs["action_logits"].shape == (3, 4)


def test_dual_stream_head_gates_are_bounded_and_delay_can_be_base_only() -> None:
    vocabs = _vocabs()
    config = ModelConfig(
        model_dim=32,
        hidden_dim=40,
        account_dim=8,
        message_hash_dim=6,
        dropout=0.0,
        event_streams="separate-head-gated",
        message_model_dim=16,
        message_hidden_dim=20,
        message_gate_max=0.4,
        delay_message_mode="base-only",
    )
    model = HierarchicalSequenceTwin(vocabs, config)
    batch = _batch(vocabs)
    batch["event_numeric"][..., 7:] = 0.0
    batch["event_numeric"][:, 1, 7] = 1.0
    batch["event_categorical"][:, 1, 3] = vocabs.message_act.encode("recommend")
    batch["discourse"][:, 1, vocabs.discourse.encode("recommend")] = 1.0
    batch["message_bins"][:, 1, 0] = 7
    batch["message_bin_mask"][:, 1, 0] = True
    outputs = model(batch, force_population=True)
    assert outputs["action_logits"].shape == (3, 4)
    assert torch.all(outputs["action_message_gate"] > 0.0)
    assert torch.all(outputs["action_message_gate"] <= config.message_gate_max)
    assert torch.all(outputs["value_message_gate"] > 0.0)
    assert torch.count_nonzero(outputs["delay_message_gate"]) == 0


def test_base_only_delay_is_invariant_to_message_features() -> None:
    vocabs = _vocabs()
    config = ModelConfig(
        model_dim=32,
        hidden_dim=40,
        account_dim=8,
        message_hash_dim=6,
        dropout=0.0,
        event_streams="separate-head-gated",
        message_model_dim=16,
        message_hidden_dim=20,
        delay_message_mode="base-only",
    )
    model = HierarchicalSequenceTwin(vocabs, config).eval()
    plain = _batch(vocabs)
    plain["event_numeric"][..., 7:] = 0.0
    plain["event_categorical"][..., 3] = 0
    plain["discourse"].zero_()
    plain["message_bins"].zero_()
    plain["message_bin_mask"].zero_()
    enriched = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in plain.items()}
    enriched["event_numeric"][:, 1, 7:] = torch.tensor([1.0, 0.4, 0.2, 0.1, 0.2, 0.25, 0.5, 0.125, 0.25, 0.25, 0.5, 0.25])
    enriched["event_categorical"][:, 1, 3] = vocabs.message_act.encode("recommend")
    enriched["discourse"][:, 1, vocabs.discourse.encode("recommend")] = 1.0
    enriched["message_bins"][:, 1, 0] = 11
    enriched["message_bin_mask"][:, 1, 0] = True
    with torch.inference_mode():
        plain_outputs = model(plain, force_population=True)
        enriched_outputs = model(enriched, force_population=True)
    assert torch.equal(plain_outputs["delay_location"], enriched_outputs["delay_location"])
    assert torch.equal(plain_outputs["delay_log_scale"], enriched_outputs["delay_log_scale"])
    assert torch.count_nonzero(plain_outputs["action_message_gate"]) == 0
    assert torch.all(enriched_outputs["action_message_gate"] > 0.0)


def test_mamba_shape_bucketing_preserves_prefix_and_zero_pads_tail() -> None:
    sequence = torch.randn(3, 65, 7)
    bucketed = _bucket_mamba_sequence(sequence)
    assert bucketed.shape == (3, 128, 7)
    assert torch.equal(bucketed[:, :65], sequence)
    assert torch.count_nonzero(bucketed[:, 65:]) == 0


def test_mamba2_core_selects_fused_memory_efficient_path_when_extra_is_installed() -> None:
    pytest.importorskip("causal_conv1d")
    from glee_sequence_lab.model import Mamba2SequenceCore

    core = Mamba2SequenceCore(ModelConfig(core="mamba2"))
    assert all(block.use_mem_eff_path for block in core.blocks)


def test_proposal_rows_do_not_contribute_trivial_action_loss() -> None:
    batch = _batch(_vocabs())
    outputs = {
        "action_logits": torch.tensor([[100.0, -100.0, -100.0, -100.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]], requires_grad=True),
        "value_location": torch.tensor([0.5, 0.4, 0.6], requires_grad=True),
        "value_log_scale": torch.zeros(3, requires_grad=True),
        "delay_location": torch.tensor([0.0, 5.0, 6.0], requires_grad=True),
        "delay_log_scale": torch.zeros(3, requires_grad=True),
        "effective_accounts": torch.zeros(3, dtype=torch.long),
    }
    first, _parts = sequence_twin_loss(outputs, batch)
    changed = {**outputs, "action_logits": outputs["action_logits"].clone()}
    changed["action_logits"][0] = torch.tensor([-100.0, -100.0, -100.0, 100.0])
    second, _parts = sequence_twin_loss(changed, batch)
    assert torch.allclose(first, second)


def test_synthetic_policy_zoo_is_train_only_and_account_free(tmp_path) -> None:
    output = tmp_path / "synthetic"
    receipt = SyntheticPolicyZooBuilder(output_dir=output, games_per_family=3, seed=17).run()
    assert receipt["inventory"]["games"] == 9
    corpus = CorpusIndex([output])
    assert {game["chronological_split"] for game in corpus.games.values()} == {"train"}
    assert {game["source_type"] for game in corpus.games.values()} == {"synthetic"}
    assert all(game["account_key"] is None for game in corpus.games.values())
    assert all(target["source_type"] == "synthetic" for target in corpus.targets)


def test_account_disjoint_fold_uses_population_embedding() -> None:
    vocabs = _vocabs()
    sample = {
        "game": {"account_key": "participant-alpha", "account_fold": 0},
        "events": [],
        "target": {"sample_id": "s", "game_id": "g", "prefix_length": 0, "target_kind": "response", "target_label": "accept", "target_value_present": False, "target_message_present": False, "target_delay_present": False, "identity_scope": "known", "account_key": "participant-alpha", "source_type": "real"},
    }
    batch = SequenceCollator(vocabs, "bargaining", excluded_account_fold=0)([sample])
    assert batch["accounts"].tolist() == [0]
    assert batch["metadata"][0]["account_known_to_model"] is False


def test_last_event_and_message_mask_ablation_preserve_action_state() -> None:
    vocabs = _vocabs()
    events = [
        {"actor": "self", "kind": "proposal", "action_label": "proposal", "message_family_act": "recommend", "message_present": True, "message_chars": 20, "message_words": 4, "message_discourse_acts": ["recommend"], "message_hash_bins": [7, 11]},
        {"actor": "opponent", "kind": "response", "action_label": "reject", "message_family_act": "recommend", "message_present": True, "message_chars": 30, "message_words": 5, "message_discourse_acts": ["recommend"], "message_hash_bins": [13]},
    ]
    sample = {
        "game": {"our_role": "proposer", "opponent_role": "responder", "identity_scope": "known", "account_key": "participant-alpha", "account_fold": 1, "horizon_known": False, "messages_allowed": True},
        "events": events,
        "target": {"sample_id": "s", "game_id": "g", "prefix_length": 2, "target_kind": "response", "target_label": "accept", "target_value_present": False, "target_message_present": False, "target_delay_present": False, "identity_scope": "known", "account_key": "participant-alpha", "source_type": "real"},
    }
    batch = SequenceCollator(vocabs, "bargaining", history_window=1, mask_message_inputs=True)([sample])
    assert batch["lengths"].tolist() == [2]
    assert batch["event_categorical"][0, 1, 2].item() == vocabs.event_action.encode("reject")
    assert batch["event_categorical"][0, 1, 3].item() == 0
    assert torch.count_nonzero(batch["event_numeric"][0, 1, 7:]) == 0
    assert torch.count_nonzero(batch["discourse"]) == 0
    assert not batch["message_bin_mask"].any()
    assert "target_messages" not in batch
    assert "target_message_mask" not in batch


def test_conditional_bridge_mask_hides_only_last_event_future_fields() -> None:
    vocabs = _vocabs()
    events = [
        {"actor": "opponent", "kind": "proposal", "action_label": "proposal", "action_value": 0.4, "response_time_ms": 900, "message_family_act": "recommend", "message_present": True, "message_chars": 20, "message_words": 4, "message_discourse_acts": ["recommend"], "message_hash_bins": [7]},
        {"actor": "self", "kind": "proposal", "action_label": "proposal", "action_value": 0.6, "action_aux_value": 0.4, "response_time_ms": 4_000, "message_family_act": "recommend", "message_present": True, "message_chars": 30, "message_words": 5, "message_discourse_acts": ["recommend"], "message_hash_bins": [11]},
    ]
    sample = {
        "game": {"our_role": "proposer", "opponent_role": "responder", "identity_scope": "known", "account_key": "participant-alpha", "account_fold": 1, "horizon_known": False, "messages_allowed": True},
        "events": events,
        "target": {"sample_id": "s", "game_id": "g", "prefix_length": 2, "target_event_index": 2, "target_kind": "response", "target_label": "accept", "target_value_present": False, "target_message_present": False, "target_delay_present": False, "identity_scope": "known", "account_key": "participant-alpha", "source_type": "real", "mask_last_prefix_future_fields": True},
    }
    batch = SequenceCollator(vocabs, "bargaining")([sample])
    assert batch["event_numeric"][0, 1, 5].item() > 0.0
    assert batch["event_numeric"][0, 1, 6].item() == 1.0
    assert batch["event_categorical"][0, 1, 3].item() == vocabs.message_act.encode("recommend")
    assert batch["message_bin_mask"][0, 1, 0]
    assert batch["event_numeric"][0, 2, 1].item() == pytest.approx(0.6)
    assert batch["event_numeric"][0, 2, 2].item() == 1.0
    assert batch["event_numeric"][0, 2, 3].item() == pytest.approx(0.4)
    assert torch.count_nonzero(batch["event_numeric"][0, 2, 5:]) == 0
    assert batch["event_categorical"][0, 2, 3].item() == 0
    assert torch.count_nonzero(batch["discourse"][0, 2]) == 0
    assert not batch["message_bin_mask"][0, 2].any()


def test_post_planner_bridge_masks_latency_but_keeps_candidate_wording() -> None:
    vocabs = _vocabs()
    events = [
        {"actor": "opponent", "kind": "proposal", "action_label": "proposal", "action_value": 0.4, "response_time_ms": 900, "message_family_act": "recommend", "message_present": True, "message_chars": 20, "message_words": 4, "message_discourse_acts": ["recommend"], "message_hash_bins": [7]},
        {"actor": "self", "kind": "proposal", "action_label": "proposal", "action_value": 0.6, "action_aux_value": 0.4, "response_time_ms": 4_000, "message_family_act": "recommend", "message_present": True, "message_chars": 30, "message_words": 5, "message_discourse_acts": ["recommend"], "message_hash_bins": [11]},
    ]
    sample = {
        "game": {"our_role": "proposer", "opponent_role": "responder", "identity_scope": "known", "account_key": "participant-alpha", "account_fold": 1, "horizon_known": False, "messages_allowed": True},
        "events": events,
        "target": {"sample_id": "s", "game_id": "g", "prefix_length": 2, "target_event_index": 2, "target_kind": "response", "target_label": "accept", "target_value_present": False, "target_message_present": False, "target_delay_present": False, "identity_scope": "known", "account_key": "participant-alpha", "source_type": "real", "mask_last_prefix_future_fields": False, "mask_last_prefix_response_time": True, "mask_last_prefix_message_fields": False},
    }
    batch = SequenceCollator(vocabs, "bargaining")([sample])
    assert torch.count_nonzero(batch["event_numeric"][0, 2, 5:7]) == 0
    assert batch["event_numeric"][0, 2, 7].item() == 1.0
    assert batch["event_numeric"][0, 2, 8].item() > 0.0
    assert batch["event_categorical"][0, 2, 3].item() == vocabs.message_act.encode("recommend")
    assert torch.count_nonzero(batch["discourse"][0, 2]) > 0
    assert batch["message_bin_mask"][0, 2, 0]
