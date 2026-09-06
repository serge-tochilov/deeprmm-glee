"""Train and compare compact engineered-feature fusion at the exact pre-Terra frontier."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from .corpus import GLEE_FAMILIES, file_sha256, object_sha256
from .data import TARGET_LABELS
from .pre_terra_v3 import FEATURE_DIMENSION, PRE_TERRA_V3_CORPUS_CONTRACT


V3_EXPERIMENT_CONTRACT = "glee-pre-terra-feature-fusion-experiment-v3"
V3_ARMS = ("sequence-calibrated", "engineered-only", "late-fusion")
V3_SEEDS = (1_729, 2_718)
FAMILY_INDEX = {family: index for index, family in enumerate(GLEE_FAMILIES)}
MAX_ACTION_CLASSES = max(len(TARGET_LABELS[family]) for family in GLEE_FAMILIES)


@dataclass(frozen=True)
class V3TrainingConfig:
    seeds: tuple[int, ...] = V3_SEEDS
    epochs: int = 80
    patience: int = 8
    batch_size: int = 128
    evaluation_batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 1e-3
    dropout: float = 0.15
    input_dropout: float = 0.08
    hidden_dim: int = 128
    latent_dim: int = 64
    maximum_fusion_gate: float = 0.75
    bootstrap_replicates: int = 2_000
    bootstrap_seed: int = 81_077


@dataclass(frozen=True)
class V3RowMetadata:
    sample_id: str
    game_id: str
    family: str
    chronological_split: str
    identity_scope: str
    our_role: str
    phase: str
    round_number: int
    target_label: str


class V3Corpus:
    """Load one verified sparse-feature corpus into compact CPU tensors and immutable metadata."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir.resolve()
        self.manifest = json.loads((self.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("contract") != PRE_TERRA_V3_CORPUS_CONTRACT or self.manifest.get("status") != "frozen-retrospective-corpus":
            raise ValueError("unsupported or unfrozen pre-Terra v3 corpus")
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("pre-Terra v3 corpus has no artifact receipts")
        for name in ("games.parquet", "events.parquet", "targets.parquet", "features.parquet", "sequence-predictions.parquet"):
            receipt = artifacts.get(name)
            path = self.corpus_dir / name
            if not isinstance(receipt, Mapping) or not path.is_file() or file_sha256(path) != receipt.get("sha256"):
                raise RuntimeError(f"pre-Terra v3 corpus artifact hash mismatch: {name}")
        targets = pl.read_parquet(self.corpus_dir / "targets.parquet")
        features = pl.read_parquet(self.corpus_dir / "features.parquet")
        sequence = pl.read_parquet(self.corpus_dir / "sequence-predictions.parquet")
        games = pl.read_parquet(self.corpus_dir / "games.parquet").select("game_id", "family", "our_role")
        joined = targets.join(features, on=["sample_id", "game_id"], how="inner").join(sequence, on=["sample_id", "game_id", "family"], how="inner").join(games, on=["game_id", "family"], how="inner")
        if joined.height != targets.height or joined["sample_id"].n_unique() != joined.height:
            raise RuntimeError("v3 corpus feature, sequence, and target rows are not one-to-one")
        rows = joined.sort(["family", "source_call_ts", "game_id", "round_number"]).to_dicts()
        feature_tensor = torch.zeros(len(rows), FEATURE_DIMENSION, dtype=torch.float32)
        base_probabilities = torch.zeros(len(rows), MAX_ACTION_CLASSES, dtype=torch.float32)
        action_mask = torch.zeros(len(rows), MAX_ACTION_CLASSES, dtype=torch.bool)
        labels = torch.zeros(len(rows), dtype=torch.long)
        families = torch.zeros(len(rows), dtype=torch.long)
        metadata: list[V3RowMetadata] = []
        for row_index, row in enumerate(rows):
            family = str(row["family"])
            expected_labels = list(TARGET_LABELS[family])
            frozen_labels = [str(value) for value in row["labels"]]
            if len(frozen_labels) != len(expected_labels) or len(set(frozen_labels)) != len(frozen_labels) or set(frozen_labels) != set(expected_labels):
                raise RuntimeError(f"frozen sequence labels disagree for {family}")
            indices = [int(value) for value in row["feature_indices"]]
            values = [float(value) for value in row["feature_values"]]
            if len(indices) != len(values) or len(set(indices)) != len(indices) or any(index < 0 or index >= FEATURE_DIMENSION for index in indices):
                raise RuntimeError("invalid sparse engineered-feature vector")
            if object_sha256({"indices": indices, "values": values}) != row["feature_vector_sha256"]:
                raise RuntimeError("engineered-feature vector hash mismatch")
            if indices:
                feature_tensor[row_index, torch.tensor(indices, dtype=torch.long)] = torch.tensor(values, dtype=torch.float32)
            frozen_probabilities = [float(value) for value in row["probabilities"]]
            if len(frozen_probabilities) != len(frozen_labels):
                raise RuntimeError("frozen sequence labels and probabilities have different lengths")
            probability_by_label = dict(zip(frozen_labels, frozen_probabilities, strict=True))
            probabilities = torch.tensor([probability_by_label[label] for label in expected_labels], dtype=torch.float32)
            if probabilities.shape != (len(expected_labels),) or not torch.isfinite(probabilities).all() or float(probabilities.min()) < 0 or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError("invalid frozen sequence probabilities")
            base_probabilities[row_index, : len(expected_labels)] = probabilities
            action_mask[row_index, : len(expected_labels)] = True
            try:
                labels[row_index] = expected_labels.index(str(row["target_label"]))
            except ValueError as error:
                raise RuntimeError(f"unknown target label for {family}: {row['target_label']!r}") from error
            families[row_index] = FAMILY_INDEX[family]
            metadata.append(
                V3RowMetadata(
                    sample_id=str(row["sample_id"]),
                    game_id=str(row["game_id"]),
                    family=family,
                    chronological_split=str(row["chronological_split"]),
                    identity_scope=str(row["identity_scope"]),
                    our_role=str(row["our_role"]),
                    phase=str(row["phase"]),
                    round_number=int(row["round_number"]),
                    target_label=str(row["target_label"]),
                )
            )
        self.features = feature_tensor
        self.base_probabilities = base_probabilities
        self.action_mask = action_mask
        self.labels = labels
        self.families = families
        self.metadata = tuple(metadata)
        self.indices_by_split = {split: torch.tensor([index for index, row in enumerate(self.metadata) if row.chronological_split == split], dtype=torch.long) for split in ("train", "validation", "test")}
        if any(len(indices) == 0 for indices in self.indices_by_split.values()):
            raise RuntimeError("v3 corpus has an empty chronological split")

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(self.corpus_dir),
            "manifest_sha256": file_sha256(self.corpus_dir / "manifest.json"),
            "rows": len(self.metadata),
            "split_rows": {split: len(indices) for split, indices in self.indices_by_split.items()},
            "family_rows": dict(sorted(Counter(row.family for row in self.metadata).items())),
        }


class FeatureEncoder(nn.Module):
    def __init__(self, config: V3TrainingConfig) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(config.input_dropout)
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIMENSION, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(config.latent_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(self.input_dropout(features))


class V3ActionModel(nn.Module):
    """Calibrate the frozen sequence distribution, predict from features, or add a bounded feature residual."""

    def __init__(self, arm: str, config: V3TrainingConfig) -> None:
        super().__init__()
        if arm not in V3_ARMS:
            raise ValueError(f"unsupported v3 arm: {arm}")
        self.arm = arm
        self.config = config
        self.log_temperature = nn.Parameter(torch.zeros(len(GLEE_FAMILIES)))
        self.calibration_bias = nn.Parameter(torch.zeros(len(GLEE_FAMILIES), MAX_ACTION_CLASSES))
        if arm == "sequence-calibrated":
            self.feature_encoder = None
            self.feature_heads = None
            self.gate_heads = None
        else:
            self.feature_encoder = FeatureEncoder(config)
            self.feature_heads = nn.ModuleList([nn.Linear(config.latent_dim, len(TARGET_LABELS[family])) for family in GLEE_FAMILIES])
            self.gate_heads = nn.ModuleList([nn.Linear(config.latent_dim + 2, 1) for _family in GLEE_FAMILIES]) if arm == "late-fusion" else None
            if self.gate_heads is not None:
                for gate in self.gate_heads:
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, -2.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor, base_probabilities: torch.Tensor, families: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_log = torch.log(base_probabilities.clamp_min(1e-8))
        temperatures = self.log_temperature.exp().clamp(0.25, 4.0)[families].unsqueeze(-1)
        calibrated = base_log / temperatures + self.calibration_bias[families]
        calibrated = calibrated.masked_fill(~action_mask, -1e9)
        gates = calibrated.new_zeros(calibrated.shape[0])
        if self.arm == "sequence-calibrated":
            return calibrated, gates
        if self.feature_encoder is None or self.feature_heads is None:
            raise AssertionError("feature arm has no feature encoder")
        hidden = self.feature_encoder(features)
        logits = calibrated.new_full(calibrated.shape, -1e9)
        for family_index, family in enumerate(GLEE_FAMILIES):
            rows = torch.nonzero(families == family_index, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            feature_logits = self.feature_heads[family_index](hidden.index_select(0, rows))
            classes = len(TARGET_LABELS[family])
            if self.arm == "engineered-only":
                logits[rows, :classes] = feature_logits
                continue
            if self.gate_heads is None:
                raise AssertionError("late-fusion arm has no gates")
            base = base_probabilities.index_select(0, rows)[:, :classes].clamp_min(1e-8)
            entropy = -(base * base.log()).sum(dim=-1, keepdim=True) / math.log(classes)
            confidence = base.max(dim=-1, keepdim=True).values
            gate = self.config.maximum_fusion_gate * torch.sigmoid(self.gate_heads[family_index](torch.cat((hidden.index_select(0, rows), entropy, confidence), dim=-1))).squeeze(-1)
            logits[rows, :classes] = calibrated.index_select(0, rows)[:, :classes] + gate.unsqueeze(-1) * feature_logits
            gates[rows] = gate
        return logits, gates


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(indices: torch.Tensor, *, batch_size: int, seed: int | None = None) -> Sequence[torch.Tensor]:
    selected = indices
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        selected = selected[torch.randperm(len(selected), generator=generator)]
    return tuple(selected[start : start + batch_size] for start in range(0, len(selected), batch_size))


def _family_weights(corpus: V3Corpus, indices: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(corpus.families.index_select(0, indices), minlength=len(GLEE_FAMILIES)).float()
    if bool((counts == 0).any()):
        raise RuntimeError("training split lacks one family")
    weights = len(indices) / (len(GLEE_FAMILIES) * counts)
    return weights


def _model_probabilities(model: V3ActionModel, corpus: V3Corpus, indices: torch.Tensor, device: torch.device, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities = torch.zeros(len(indices), MAX_ACTION_CLASSES, dtype=torch.float32)
    gates = torch.zeros(len(indices), dtype=torch.float32)
    cursor = 0
    with torch.inference_mode():
        for batch_indices in _batches(indices, batch_size=batch_size):
            features = corpus.features.index_select(0, batch_indices).to(device)
            base = corpus.base_probabilities.index_select(0, batch_indices).to(device)
            families = corpus.families.index_select(0, batch_indices).to(device)
            mask = corpus.action_mask.index_select(0, batch_indices).to(device)
            logits, batch_gates = model(features, base, families, mask)
            batch_probabilities = F.softmax(logits, dim=-1).cpu()
            probabilities[cursor : cursor + len(batch_indices)] = batch_probabilities
            gates[cursor : cursor + len(batch_indices)] = batch_gates.cpu()
            cursor += len(batch_indices)
    return probabilities, gates


def _rows_for_predictions(corpus: V3Corpus, indices: torch.Tensor, probabilities: torch.Tensor, gates: torch.Tensor, *, arm: str, seed: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, row_index in enumerate(indices.tolist()):
        metadata = corpus.metadata[row_index]
        classes = len(TARGET_LABELS[metadata.family])
        values = [float(value) for value in probabilities[offset, :classes].tolist()]
        actual = int(corpus.labels[row_index])
        rows.append(
            {
                **asdict(metadata),
                "arm": arm,
                "seed": seed,
                "actual_action": actual,
                "predicted_action": max(range(classes), key=values.__getitem__),
                "probabilities": values,
                "negative_log_likelihood": -math.log(max(values[actual], 1e-12)),
                "brier_score": sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(values)),
                "confidence": max(values),
                "correct": int(max(range(classes), key=values.__getitem__) == actual),
                "fusion_gate": float(gates[offset]),
            }
        )
    return rows


def _expected_calibration_error(rows: Sequence[Mapping[str, object]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    error = 0.0
    for lower_index in range(bins):
        lower = lower_index / bins
        upper = (lower_index + 1) / bins
        if lower_index == bins - 1:
            selected = [row for row in rows if lower <= float(row["confidence"]) <= upper]
        else:
            selected = [row for row in rows if lower <= float(row["confidence"]) < upper]
        if not selected:
            continue
        confidence = sum(float(row["confidence"]) for row in selected) / len(selected)
        accuracy = sum(int(row["correct"]) for row in selected) / len(selected)
        error += len(selected) / total * abs(confidence - accuracy)
    return error


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "games": 0, "negative_log_likelihood": None, "brier_score": None, "accuracy": None, "game_macro_negative_log_likelihood": None, "expected_calibration_error_10": None, "mean_fusion_gate": None}
    game_losses: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        game_losses[str(row["game_id"])].append(float(row["negative_log_likelihood"]))
    game_means = [sum(values) / len(values) for values in game_losses.values()]
    return {
        "rows": len(rows),
        "games": len(game_means),
        "negative_log_likelihood": sum(float(row["negative_log_likelihood"]) for row in rows) / len(rows),
        "brier_score": sum(float(row["brier_score"]) for row in rows) / len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "game_macro_negative_log_likelihood": sum(game_means) / len(game_means),
        "expected_calibration_error_10": _expected_calibration_error(rows),
        "mean_fusion_gate": sum(float(row["fusion_gate"]) for row in rows) / len(rows),
    }


def _evaluation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family = {family: _metrics([row for row in rows if row["family"] == family]) for family in GLEE_FAMILIES}
    return {
        "all": _metrics(rows),
        "equal_family_negative_log_likelihood": sum(float(by_family[family]["negative_log_likelihood"]) for family in GLEE_FAMILIES) / len(GLEE_FAMILIES),
        "by_family": by_family,
        "by_identity_scope": {scope: _metrics([row for row in rows if row["identity_scope"] == scope]) for scope in ("known", "hidden")},
        "by_role": {role: _metrics([row for row in rows if row["our_role"] == role]) for role in sorted({str(row["our_role"]) for row in rows})},
        "by_phase": {phase: _metrics([row for row in rows if row["phase"] == phase]) for phase in sorted({str(row["phase"]) for row in rows})},
    }


def _validation_score(model: V3ActionModel, corpus: V3Corpus, device: torch.device, config: V3TrainingConfig) -> float:
    indices = corpus.indices_by_split["validation"]
    probabilities, gates = _model_probabilities(model, corpus, indices, device, batch_size=config.evaluation_batch_size)
    rows = _rows_for_predictions(corpus, indices, probabilities, gates, arm=model.arm, seed=None)
    return float(_evaluation(rows)["equal_family_negative_log_likelihood"])


def _train_model(arm: str, seed: int, corpus: V3Corpus, device: torch.device, config: V3TrainingConfig) -> tuple[V3ActionModel, dict[str, object]]:
    _set_seed(seed)
    model = V3ActionModel(arm, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_indices = corpus.indices_by_split["train"]
    family_weights = _family_weights(corpus, train_indices).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for batch_indices in _batches(train_indices, batch_size=config.batch_size, seed=seed + epoch * 1_009):
            features = corpus.features.index_select(0, batch_indices).to(device)
            base = corpus.base_probabilities.index_select(0, batch_indices).to(device)
            families = corpus.families.index_select(0, batch_indices).to(device)
            mask = corpus.action_mask.index_select(0, batch_indices).to(device)
            labels = corpus.labels.index_select(0, batch_indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _gates = model(features, base, families, mask)
            row_loss = F.cross_entropy(logits, labels, reduction="none")
            row_weights = family_weights[families]
            loss = (row_loss * row_weights).sum() / row_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float((row_loss.detach() * row_weights).sum().cpu())
            total_weight += float(row_weights.sum().cpu())
        validation_score = _validation_score(model, corpus, device, config)
        improved = validation_score < best_score - 1e-5
        history.append({"epoch": epoch, "train_equal_family_weighted_nll": total_loss / max(total_weight, 1e-12), "validation_equal_family_nll": validation_score, "improved": improved})
        if improved:
            best_score = validation_score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("v3 training never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_equal_family_nll": best_score, "history": history}


def _bootstrap_difference(arm_rows: Sequence[Mapping[str, object]], reference_rows: Sequence[Mapping[str, object]], *, replicates: int, seed: int) -> dict[str, object]:
    arm = {str(row["sample_id"]): row for row in arm_rows}
    reference = {str(row["sample_id"]): row for row in reference_rows}
    if arm.keys() != reference.keys():
        raise ValueError("paired v3 predictions are not aligned")
    game_rows: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for sample_id in sorted(arm):
        family = str(arm[sample_id]["family"])
        if family != reference[sample_id]["family"] or arm[sample_id]["game_id"] != reference[sample_id]["game_id"]:
            raise ValueError("paired v3 prediction coordinates disagree")
        game_rows[str(arm[sample_id]["game_id"])].append((family, float(arm[sample_id]["negative_log_likelihood"]) - float(reference[sample_id]["negative_log_likelihood"])))
    games_by_family = {family: [game_id for game_id, rows in game_rows.items() if rows[0][0] == family] for family in GLEE_FAMILIES}
    observed_family = {family: sum(value for game_id in games_by_family[family] for _row_family, value in game_rows[game_id]) / sum(len(game_rows[game_id]) for game_id in games_by_family[family]) for family in GLEE_FAMILIES}
    observed_pooled = sum(value for rows in game_rows.values() for _family, value in rows) / sum(len(rows) for rows in game_rows.values())
    observed_equal = sum(observed_family.values()) / len(GLEE_FAMILIES)
    rng = np.random.default_rng(seed)
    pooled_samples: list[float] = []
    equal_samples: list[float] = []
    for _replicate in range(replicates):
        family_values: dict[str, list[float]] = {}
        all_values: list[float] = []
        for family in GLEE_FAMILIES:
            games = games_by_family[family]
            drawn = rng.choice(games, size=len(games), replace=True)
            values = [value for game_id in drawn.tolist() for _row_family, value in game_rows[str(game_id)]]
            family_values[family] = values
            all_values.extend(values)
        pooled_samples.append(sum(all_values) / len(all_values))
        equal_samples.append(sum(sum(family_values[family]) / len(family_values[family]) for family in GLEE_FAMILIES) / len(GLEE_FAMILIES))

    def interval(values: Sequence[float]) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "difference": "arm minus reference; negative favors arm",
        "observed_pooled_nll": observed_pooled,
        "pooled_95_interval": interval(pooled_samples),
        "observed_equal_family_nll": observed_equal,
        "equal_family_95_interval": interval(equal_samples),
        "observed_by_family": observed_family,
        "replicates": replicates,
        "complete_game_clusters": len(game_rows),
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_v3_fusion_suite(*, corpus_dir: Path, output_dir: Path, config: V3TrainingConfig | None = None) -> dict[str, object]:
    selected = config or V3TrainingConfig()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"v3 experiment output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    models_dir = staging / "models"
    models_dir.mkdir()
    corpus = V3Corpus(corpus_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_indices = corpus.indices_by_split["test"]
    raw_probabilities = corpus.base_probabilities.index_select(0, test_indices)
    raw_gates = torch.zeros(len(test_indices), dtype=torch.float32)
    raw_rows = _rows_for_predictions(corpus, test_indices, raw_probabilities, raw_gates, arm="sequence-raw", seed=None)
    all_prediction_rows = list(raw_rows)
    runs: dict[str, dict[str, object]] = {
        "sequence-raw": {"arm": "sequence-raw", "seed": None, "evaluation": _evaluation(raw_rows), "model_parameters": 0, "training": None}
    }
    rows_by_key: dict[str, list[dict[str, object]]] = {"sequence-raw": raw_rows}
    for arm in V3_ARMS:
        for seed in selected.seeds:
            model, training = _train_model(arm, seed, corpus, device, selected)
            probabilities, gates = _model_probabilities(model, corpus, test_indices, device, batch_size=selected.evaluation_batch_size)
            rows = _rows_for_predictions(corpus, test_indices, probabilities, gates, arm=arm, seed=seed)
            key = f"{arm}-seed{seed}"
            checkpoint_path = models_dir / f"{key}.pt"
            torch.save({"contract": V3_EXPERIMENT_CONTRACT, "arm": arm, "seed": seed, "config": asdict(selected), "model": model.state_dict(), "feature_dimension": FEATURE_DIMENSION}, checkpoint_path)
            runs[key] = {"arm": arm, "seed": seed, "evaluation": _evaluation(rows), "model_parameters": model.parameter_count(), "training": training, "checkpoint": {"path": str(checkpoint_path.relative_to(staging)), "sha256": file_sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}}
            rows_by_key[key] = rows
            all_prediction_rows.extend(rows)
    comparisons: dict[str, object] = {}
    promotion_checks: dict[str, object] = {}
    for seed in selected.seeds:
        late_key = f"late-fusion-seed{seed}"
        calibrated_key = f"sequence-calibrated-seed{seed}"
        engineered_key = f"engineered-only-seed{seed}"
        comparisons[f"{late_key}_vs_{calibrated_key}"] = _bootstrap_difference(rows_by_key[late_key], rows_by_key[calibrated_key], replicates=selected.bootstrap_replicates, seed=selected.bootstrap_seed + seed)
        comparisons[f"{late_key}_vs_{engineered_key}"] = _bootstrap_difference(rows_by_key[late_key], rows_by_key[engineered_key], replicates=selected.bootstrap_replicates, seed=selected.bootstrap_seed + seed * 3)
        late_eval = runs[late_key]["evaluation"]
        calibrated_eval = runs[calibrated_key]["evaluation"]
        engineered_eval = runs[engineered_key]["evaluation"]
        family_regressions = {
            reference: {family: float(late_eval["by_family"][family]["negative_log_likelihood"]) - float(runs[f"{reference}-seed{seed}"]["evaluation"]["by_family"][family]["negative_log_likelihood"]) for family in GLEE_FAMILIES}
            for reference in ("sequence-calibrated", "engineered-only")
        }
        checks = {
            "pooled_better_than_sequence_calibrated": float(late_eval["all"]["negative_log_likelihood"]) < float(calibrated_eval["all"]["negative_log_likelihood"]),
            "pooled_better_than_engineered_only": float(late_eval["all"]["negative_log_likelihood"]) < float(engineered_eval["all"]["negative_log_likelihood"]),
            "equal_family_better_than_sequence_calibrated": float(late_eval["equal_family_negative_log_likelihood"]) < float(calibrated_eval["equal_family_negative_log_likelihood"]),
            "equal_family_better_than_engineered_only": float(late_eval["equal_family_negative_log_likelihood"]) < float(engineered_eval["equal_family_negative_log_likelihood"]),
            "no_family_regression_over_0_02": all(value <= 0.02 for values in family_regressions.values() for value in values.values()),
            "family_regressions": family_regressions,
        }
        checks["seed_gate_passed"] = all(value for key, value in checks.items() if key != "family_regressions")
        promotion_checks[str(seed)] = checks
    late_rows_by_sample: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for seed in selected.seeds:
        for row in rows_by_key[f"late-fusion-seed{seed}"]:
            late_rows_by_sample[str(row["sample_id"])].append(row)
    ensemble_rows: list[dict[str, object]] = []
    for sample_id in sorted(late_rows_by_sample):
        components = late_rows_by_sample[sample_id]
        reference = components[0]
        probabilities = [sum(float(row["probabilities"][index]) for row in components) / len(components) for index in range(len(reference["probabilities"]))]
        actual = int(reference["actual_action"])
        ensemble_rows.append(
            {
                **{key: reference[key] for key in asdict(corpus.metadata[0]) if key in reference},
                "arm": "late-fusion-ensemble",
                "seed": None,
                "actual_action": actual,
                "predicted_action": max(range(len(probabilities)), key=probabilities.__getitem__),
                "probabilities": probabilities,
                "negative_log_likelihood": -math.log(max(probabilities[actual], 1e-12)),
                "brier_score": sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities)),
                "confidence": max(probabilities),
                "correct": int(max(range(len(probabilities)), key=probabilities.__getitem__) == actual),
                "fusion_gate": sum(float(row["fusion_gate"]) for row in components) / len(components),
            }
        )
    runs["late-fusion-ensemble"] = {"arm": "late-fusion-ensemble", "seed": None, "evaluation": _evaluation(ensemble_rows), "model_parameters": None, "training": None}
    rows_by_key["late-fusion-ensemble"] = ensemble_rows
    all_prediction_rows.extend(ensemble_rows)
    gate_passed = all(bool(promotion_checks[str(seed)]["seed_gate_passed"]) for seed in selected.seeds)
    predictions_path = staging / "test-predictions.parquet"
    pl.DataFrame(all_prediction_rows, infer_schema_length=None).write_parquet(predictions_path, compression="zstd", compression_level=7, statistics=True)
    result = {
        "schema_version": 1,
        "contract": V3_EXPERIMENT_CONTRACT,
        "status": "complete",
        "corpus": corpus.receipt(),
        "configuration": asdict(selected),
        "device": {"type": device.type, "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__, "cuda": torch.version.cuda},
        "runs": runs,
        "paired_complete_game_bootstrap": comparisons,
        "promotion_gate": {"passed": gate_passed, "seed_checks": promotion_checks, "decision": "eligible-for-behaviorally-inert-v3-shadow-release" if gate_passed else "retain-v2-sequence-shadow-candidate"},
        "test_predictions": {"path": predictions_path.name, "rows": len(all_prediction_rows), "sha256": file_sha256(predictions_path)},
        "implementation_sha256": file_sha256(Path(__file__)),
        "authority": "retrospective predictive evidence only; no live policy authority",
    }
    _atomic_json(staging / "result.json", result)
    os.replace(staging, output)
    result["output_dir"] = str(output)
    result["result_sha256"] = file_sha256(output / "result.json")
    return result
