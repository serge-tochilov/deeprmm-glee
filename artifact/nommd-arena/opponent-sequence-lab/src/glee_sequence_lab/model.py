"""Compact hierarchical sequence twins with shared encoders and family-specific cores."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .corpus import GLEE_FAMILIES, HASH_BINS
from .data import CorpusVocabs


@dataclass(frozen=True)
class ModelConfig:
    core: str = "gru"
    model_dim: int = 96
    hidden_dim: int = 128
    account_dim: int = 24
    message_hash_dim: int = 16
    dropout: float = 0.12
    transformer_layers: int = 1
    transformer_heads: int = 4
    maximum_sequence_length: int = 256
    mamba_layers: int = 1
    mamba_state_dim: int = 32
    mamba_head_dim: int = 32
    event_streams: str = "shared"
    message_model_dim: int = 32
    message_hidden_dim: int = 48
    message_gate_max: float = 0.5
    delay_message_mode: str = "gated"

    def receipt(self) -> dict[str, object]:
        return asdict(self)


def _embedding(size: int, dimension: int, *, padding_idx: int | None = None) -> nn.Embedding:
    layer = nn.Embedding(size, dimension, padding_idx=padding_idx)
    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    if padding_idx is not None:
        with torch.no_grad():
            layer.weight[padding_idx].zero_()
    return layer


class EventEncoder(nn.Module):
    def __init__(self, vocabs: CorpusVocabs, config: ModelConfig) -> None:
        super().__init__()
        self.actor = _embedding(len(vocabs.actor), 6, padding_idx=0)
        self.kind = _embedding(len(vocabs.kind), 8, padding_idx=0)
        self.action = _embedding(len(vocabs.event_action), 10, padding_idx=0)
        self.message_act = _embedding(len(vocabs.message_act), 12, padding_idx=0)
        self.quality = _embedding(len(vocabs.quality), 5, padding_idx=0)
        self.message_hash = _embedding(HASH_BINS, config.message_hash_dim)
        self.numeric = nn.Sequential(nn.Linear(19, 32), nn.SiLU(), nn.LayerNorm(32))
        self.discourse = nn.Sequential(nn.Linear(len(vocabs.discourse), 12), nn.SiLU())
        width = 6 + 8 + 10 + 12 + 5 + config.message_hash_dim + 32 + 12
        self.output = nn.Sequential(nn.Linear(width, config.model_dim), nn.SiLU(), nn.LayerNorm(config.model_dim), nn.Dropout(config.dropout))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        categorical = batch["event_categorical"]
        hashes = self.message_hash(batch["message_bins"])
        hash_mask = batch["message_bin_mask"].unsqueeze(-1)
        hash_sum = (hashes * hash_mask).sum(dim=-2)
        hash_count = hash_mask.sum(dim=-2).clamp_min(1)
        hash_mean = hash_sum / hash_count
        parts = (
            self.actor(categorical[..., 0]),
            self.kind(categorical[..., 1]),
            self.action(categorical[..., 2]),
            self.message_act(categorical[..., 3]),
            self.quality(categorical[..., 4]),
            hash_mean,
            self.numeric(batch["event_numeric"]),
            self.discourse(batch["discourse"]),
        )
        return self.output(torch.cat(parts, dim=-1))


class MechanicsEventEncoder(nn.Module):
    """Encode game mechanics without exposing any message-derived feature."""

    def __init__(self, vocabs: CorpusVocabs, config: ModelConfig) -> None:
        super().__init__()
        self.actor = _embedding(len(vocabs.actor), 6, padding_idx=0)
        self.kind = _embedding(len(vocabs.kind), 8, padding_idx=0)
        self.action = _embedding(len(vocabs.event_action), 10, padding_idx=0)
        self.quality = _embedding(len(vocabs.quality), 5, padding_idx=0)
        self.numeric = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.LayerNorm(32))
        width = 6 + 8 + 10 + 5 + 32
        self.output = nn.Sequential(nn.Linear(width, config.model_dim), nn.SiLU(), nn.LayerNorm(config.model_dim), nn.Dropout(config.dropout))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        categorical = batch["event_categorical"]
        parts = (
            self.actor(categorical[..., 0]),
            self.kind(categorical[..., 1]),
            self.action(categorical[..., 2]),
            self.quality(categorical[..., 4]),
            self.numeric(batch["event_numeric"][..., :7]),
        )
        return self.output(torch.cat(parts, dim=-1))


class MessageEventEncoder(nn.Module):
    """Encode message semantics and style with only relational action tags."""

    def __init__(self, vocabs: CorpusVocabs, config: ModelConfig) -> None:
        super().__init__()
        self.actor = _embedding(len(vocabs.actor), 4, padding_idx=0)
        self.kind = _embedding(len(vocabs.kind), 6, padding_idx=0)
        self.action = _embedding(len(vocabs.event_action), 8, padding_idx=0)
        self.message_act = _embedding(len(vocabs.message_act), 12, padding_idx=0)
        self.message_hash = _embedding(HASH_BINS, config.message_hash_dim)
        self.numeric = nn.Sequential(nn.Linear(12, 24), nn.SiLU(), nn.LayerNorm(24))
        self.discourse = nn.Sequential(nn.Linear(len(vocabs.discourse), 12), nn.SiLU())
        width = 4 + 6 + 8 + 12 + config.message_hash_dim + 24 + 12
        self.output = nn.Sequential(nn.Linear(width, config.message_model_dim), nn.SiLU(), nn.LayerNorm(config.message_model_dim), nn.Dropout(config.dropout))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        categorical = batch["event_categorical"]
        hashes = self.message_hash(batch["message_bins"])
        hash_mask = batch["message_bin_mask"].unsqueeze(-1)
        hash_sum = (hashes * hash_mask).sum(dim=-2)
        hash_count = hash_mask.sum(dim=-2).clamp_min(1)
        hash_mean = hash_sum / hash_count
        parts = (
            self.actor(categorical[..., 0]),
            self.kind(categorical[..., 1]),
            self.action(categorical[..., 2]),
            self.message_act(categorical[..., 3]),
            hash_mean,
            self.numeric(batch["event_numeric"][..., 7:]),
            self.discourse(batch["discourse"]),
        )
        return self.output(torch.cat(parts, dim=-1))


class StaticEncoder(nn.Module):
    def __init__(self, vocabs: CorpusVocabs, config: ModelConfig) -> None:
        super().__init__()
        self.our_role = _embedding(len(vocabs.our_role), 8, padding_idx=0)
        self.opponent_role = _embedding(len(vocabs.opponent_role), 8, padding_idx=0)
        self.identity_scope = _embedding(len(vocabs.identity_scope), 4, padding_idx=0)
        self.output = nn.Sequential(nn.Linear(13 + 8 + 8 + 4, config.model_dim), nn.SiLU(), nn.LayerNorm(config.model_dim))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        categorical = batch["static_categorical"]
        return self.output(
            torch.cat(
                (
                    batch["static_numeric"],
                    self.our_role(categorical[..., 0]),
                    self.opponent_role(categorical[..., 1]),
                    self.identity_scope(categorical[..., 2]),
                ),
                dim=-1,
            )
        )


class GRUSequenceCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.recurrent = nn.GRU(config.model_dim, config.hidden_dim, batch_first=True)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _output, hidden = self.recurrent(packed)
        return hidden[-1]


class CausalTransformerCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.position = _embedding(config.maximum_sequence_length, config.model_dim)
        layer = nn.TransformerEncoderLayer(d_model=config.model_dim, nhead=config.transformer_heads, dim_feedforward=config.hidden_dim, dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_layers, norm=nn.LayerNorm(config.model_dim))
        self.project = nn.Linear(config.model_dim, config.hidden_dim)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _width = sequence.shape
        if sequence_length > self.position.num_embeddings:
            raise ValueError(f"sequence length {sequence_length} exceeds configured maximum {self.position.num_embeddings}")
        positions = torch.arange(sequence_length, device=sequence.device)
        sequence = sequence + self.position(positions).unsqueeze(0)
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=sequence.device, dtype=torch.bool), diagonal=1)
        padding_mask = torch.arange(sequence_length, device=sequence.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(sequence, mask=causal_mask, src_key_padding_mask=padding_mask)
        final = encoded[torch.arange(batch_size, device=sequence.device), lengths - 1]
        return self.project(final)


def _bucket_mamba_sequence(sequence: torch.Tensor, *, minimum: int = 64, maximum: int = 256) -> torch.Tensor:
    """Pad only the causally later tail to a small power-of-two shape set for Triton cache reuse."""
    sequence_length = sequence.shape[1]
    bucket = max(minimum, 1 << max(0, sequence_length - 1).bit_length())
    if bucket > maximum:
        raise ValueError(f"sequence length {sequence_length} exceeds configured Mamba maximum {maximum}")
    if bucket == sequence_length:
        return sequence
    return F.pad(sequence, (0, 0, 0, bucket - sequence_length))


class Mamba2SequenceCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from .mamba_optional import load_mamba2

        mamba2 = load_mamba2()
        self.blocks = nn.ModuleList(
            [
                mamba2(
                    d_model=config.model_dim,
                    d_state=config.mamba_state_dim,
                    headdim=config.mamba_head_dim,
                    chunk_size=64,
                    use_mem_eff_path=True,
                )
                for _index in range(config.mamba_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.model_dim) for _index in range(config.mamba_layers)])
        self.project = nn.Linear(config.model_dim, config.hidden_dim)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = sequence
        for block, norm in zip(self.blocks, self.norms, strict=True):
            hidden = norm(hidden + block(hidden))
        final = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return self.project(final)


class Mamba3SISOSequenceCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from .mamba_optional import load_mamba3

        self.maximum_sequence_length = config.maximum_sequence_length
        mamba3 = load_mamba3()
        self.blocks = nn.ModuleList(
            [
                mamba3(
                    d_model=config.model_dim,
                    d_state=config.mamba_state_dim,
                    headdim=config.mamba_head_dim,
                    chunk_size=64,
                    is_mimo=False,
                )
                for _index in range(config.mamba_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.model_dim) for _index in range(config.mamba_layers)])
        self.project = nn.Linear(config.model_dim, config.hidden_dim)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = _bucket_mamba_sequence(sequence, maximum=self.maximum_sequence_length)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            hidden = norm(hidden + block(hidden))
        final = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return self.project(final)


class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int, action_classes: int) -> None:
        super().__init__()
        self.action = nn.Linear(hidden_dim, action_classes)
        self.value = nn.Linear(hidden_dim, 2)
        self.delay = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _location_scale(raw: torch.Tensor, *, bounded_location: bool) -> tuple[torch.Tensor, torch.Tensor]:
        location = torch.sigmoid(raw[..., 0]) if bounded_location else raw[..., 0]
        log_scale = raw[..., 1].clamp(min=-4.0, max=2.0)
        return location, log_scale

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        value_location, value_log_scale = self._location_scale(self.value(hidden), bounded_location=True)
        delay_location, delay_log_scale = self._location_scale(self.delay(hidden), bounded_location=False)
        return {
            "action_logits": self.action(hidden),
            "value_location": value_location,
            "value_log_scale": value_log_scale,
            "delay_location": delay_location,
            "delay_log_scale": delay_log_scale,
        }


class BoundedMessageFusion(nn.Module):
    """Add a target-specific message residual through a bounded, inspectable gate."""

    def __init__(self, mechanics_dim: int, message_dim: int, maximum_gate: float) -> None:
        super().__init__()
        if not 0.0 < maximum_gate <= 1.0:
            raise ValueError("maximum_gate must be in (0, 1]")
        self.maximum_gate = maximum_gate
        self.residual = nn.Linear(message_dim, mechanics_dim)
        self.gate = nn.Linear(mechanics_dim + message_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, mechanics: torch.Tensor, message: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.maximum_gate * torch.sigmoid(self.gate(torch.cat((mechanics, message), dim=-1)))
        gate = gate * available.to(dtype=gate.dtype).unsqueeze(-1)
        fused = mechanics + gate * torch.tanh(self.residual(message))
        return fused, gate.squeeze(-1)


class DualStreamPredictionHeads(nn.Module):
    """Predict each target from mechanics plus its own bounded message residual."""

    def __init__(self, config: ModelConfig, action_classes: int) -> None:
        super().__init__()
        self.action_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
        self.value_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
        self.delay_message_mode = config.delay_message_mode
        if self.delay_message_mode not in {"gated", "base-only"}:
            raise ValueError(f"unsupported delay message mode: {self.delay_message_mode}")
        self.delay_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
        self.action = nn.Linear(config.hidden_dim, action_classes)
        self.value = nn.Linear(config.hidden_dim, 2)
        self.delay = nn.Linear(config.hidden_dim, 2)

    def forward(self, mechanics: torch.Tensor, message: torch.Tensor, available: torch.Tensor) -> dict[str, torch.Tensor]:
        action_hidden, action_gate = self.action_fusion(mechanics, message, available)
        value_hidden, value_gate = self.value_fusion(mechanics, message, available)
        if self.delay_message_mode == "base-only":
            delay_hidden = mechanics
            delay_gate = mechanics.new_zeros(mechanics.shape[0])
        else:
            delay_hidden, delay_gate = self.delay_fusion(mechanics, message, available)
        value_location, value_log_scale = PredictionHeads._location_scale(self.value(value_hidden), bounded_location=True)
        delay_location, delay_log_scale = PredictionHeads._location_scale(self.delay(delay_hidden), bounded_location=False)
        return {
            "action_logits": self.action(action_hidden),
            "value_location": value_location,
            "value_log_scale": value_log_scale,
            "delay_location": delay_location,
            "delay_log_scale": delay_log_scale,
            "action_message_gate": action_gate,
            "value_message_gate": value_gate,
            "delay_message_gate": delay_gate,
        }


def _core_class(name: str) -> type[nn.Module]:
    return {
        "gru": GRUSequenceCore,
        "transformer": CausalTransformerCore,
        "mamba2": Mamba2SequenceCore,
        "mamba3-siso": Mamba3SISOSequenceCore,
    }[name]


def _compact_message_sequence(sequence: torch.Tensor, present: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep BOS and causally prior message-bearing events in their original order."""
    positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    selected = valid & present
    selected[:, 0] = True
    rows = [sequence[index][selected[index]] for index in range(sequence.shape[0])]
    compact = nn.utils.rnn.pad_sequence(rows, batch_first=True)
    compact_lengths = selected.sum(dim=1)
    available = (selected.sum(dim=1) > 1)
    return compact, compact_lengths, available


class HierarchicalSequenceTwin(nn.Module):
    """Shared event representation with family cores and shrinkage-style account residuals."""

    def __init__(self, vocabs: CorpusVocabs, config: ModelConfig) -> None:
        super().__init__()
        if config.core not in {"gru", "transformer", "mamba2", "mamba3-siso"}:
            raise ValueError(f"unsupported sequence core: {config.core}")
        if config.event_streams not in {"shared", "separate-head-gated"}:
            raise ValueError(f"unsupported event stream mode: {config.event_streams}")
        if config.message_model_dim < 1 or config.message_hidden_dim < 1:
            raise ValueError("message stream dimensions must be positive")
        self.config = config
        self.event_encoder = EventEncoder(vocabs, config) if config.event_streams == "shared" else MechanicsEventEncoder(vocabs, config)
        self.static_encoder = StaticEncoder(vocabs, config)
        self.account_global = _embedding(len(vocabs.account), config.account_dim, padding_idx=0)
        self.account_family = nn.ModuleDict({family: _embedding(len(vocabs.account), config.account_dim, padding_idx=0) for family in GLEE_FAMILIES})
        self.account_project = nn.Sequential(nn.Linear(config.account_dim, config.model_dim), nn.Tanh())
        self.context_mix = nn.Sequential(nn.Linear(config.model_dim * 3, config.model_dim), nn.SiLU(), nn.LayerNorm(config.model_dim))
        core_class = _core_class(config.core)
        self.cores = nn.ModuleDict({family: core_class(config) for family in GLEE_FAMILIES})
        if config.event_streams == "shared":
            self.heads = nn.ModuleDict({family: PredictionHeads(config.hidden_dim, len(vocabs.target_labels[family])) for family in GLEE_FAMILIES})
            self.message_encoder = None
            self.message_context_mix = None
            self.message_cores = None
        else:
            message_config = replace(config, model_dim=config.message_model_dim, hidden_dim=config.message_hidden_dim)
            message_core_class = _core_class(message_config.core)
            self.message_encoder = MessageEventEncoder(vocabs, config)
            self.message_context_mix = nn.Sequential(nn.Linear(config.model_dim * 3, config.message_model_dim), nn.SiLU(), nn.LayerNorm(config.message_model_dim))
            self.message_cores = nn.ModuleDict({family: message_core_class(message_config) for family in GLEE_FAMILIES})
            self.heads = nn.ModuleDict({family: DualStreamPredictionHeads(config, len(vocabs.target_labels[family])) for family in GLEE_FAMILIES})

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def account_context(self, family: str, accounts: torch.Tensor, *, force_population: bool = False, account_dropout: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        effective = accounts
        if force_population:
            effective = torch.zeros_like(accounts)
        elif self.training and account_dropout > 0.0:
            keep = torch.rand(accounts.shape, device=accounts.device) >= account_dropout
            effective = torch.where(keep, accounts, torch.zeros_like(accounts))
        residual = self.account_global(effective) + self.account_family[family](effective)
        return self.account_project(residual), effective

    def forward(self, batch: Mapping[str, object], *, force_population: bool = False, account_dropout: float = 0.0) -> dict[str, torch.Tensor]:
        family = str(batch["family"])
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        tensor_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        events = self.event_encoder(tensor_batch)
        static = self.static_encoder(tensor_batch)
        account, effective_accounts = self.account_context(family, tensor_batch["accounts"], force_population=force_population, account_dropout=account_dropout)
        context = self.context_mix(torch.cat((static, account, static * account), dim=-1))
        sequence = events + context.unsqueeze(1)
        hidden = self.cores[family](sequence, tensor_batch["lengths"])
        if self.config.event_streams == "shared":
            outputs = self.heads[family](hidden)
        else:
            if self.message_encoder is None or self.message_context_mix is None or self.message_cores is None:
                raise AssertionError("separate message stream was not initialized")
            message_context = self.message_context_mix(torch.cat((static, account, static * account), dim=-1))
            message_events = self.message_encoder(tensor_batch) + message_context.unsqueeze(1)
            message_present = tensor_batch["event_numeric"][..., 7] > 0.5
            message_sequence, message_lengths, message_available = _compact_message_sequence(message_events, message_present, tensor_batch["lengths"])
            message_hidden = self.message_cores[family](message_sequence, message_lengths)
            outputs = self.heads[family](hidden, message_hidden, message_available)
        outputs["effective_accounts"] = effective_accounts
        return outputs


def gaussian_negative_log_likelihood(target: torch.Tensor, location: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    inverse_variance = torch.exp(-2.0 * log_scale)
    return 0.5 * (target - location).square() * inverse_variance + log_scale + 0.5 * math.log(2.0 * math.pi)


def sequence_twin_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, object], *, value_weight: float = 0.45, delay_weight: float = 0.10) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action_mask = batch["target_action_mask"]
    action_rows = F.cross_entropy(outputs["action_logits"], batch["target_labels"], reduction="none")
    action = action_rows[action_mask].mean() if action_mask.any() else outputs["action_logits"].sum() * 0.0
    value_mask = batch["target_value_mask"]
    value_rows = gaussian_negative_log_likelihood(batch["target_values"], outputs["value_location"], outputs["value_log_scale"])
    value = value_rows[value_mask].mean() if value_mask.any() else action.new_zeros(())
    delay_mask = batch["target_delay_mask"]
    delay_rows = gaussian_negative_log_likelihood(batch["target_delays"], outputs["delay_location"], outputs["delay_log_scale"])
    delay = delay_rows[delay_mask].mean() if delay_mask.any() else action.new_zeros(())
    total = action + value_weight * value + delay_weight * delay
    return total, {"total": total.detach(), "action": action.detach(), "value": value.detach(), "delay": delay.detach()}
