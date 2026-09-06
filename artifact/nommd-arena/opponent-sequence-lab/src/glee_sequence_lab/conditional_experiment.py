"""Train and evaluate the action-conditioned pre-Terra response twin."""

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
from .data import CorpusIndex
from .pre_terra_conditional_v2 import CONDITIONAL_CORPUS_CONTRACT, CONDITIONAL_TARGET_LABELS, FEATURE_DIMENSION, CandidateAction, CandidateActionProjector, merge_sparse_vectors
from .shadow import ShadowCandidate


CONDITIONAL_EXPERIMENT_CONTRACT = "glee-post-planner-action-conditional-experiment-v3"
CONDITIONAL_SEEDS = (1_729, 2_718)
FAMILY_INDEX = {family: index for index, family in enumerate(GLEE_FAMILIES)}
MAX_RESPONSE_CLASSES = max(len(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES)


@dataclass(frozen=True)
class ConditionalTrainingConfig:
    seeds: tuple[int, ...] = CONDITIONAL_SEEDS
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
    stack_grid_steps: int = 1_000


@dataclass(frozen=True)
class ConditionalRowMetadata:
    sample_id: str
    game_id: str
    family: str
    chronological_split: str
    identity_scope: str
    our_role: str
    phase: str
    round_number: int
    target_label: str
    candidate_kind: str
    candidate_action_label: str
    candidate_action_value: float | None
    candidate_action_aux_value: float | None
    candidate_round_phase: float
    candidate_message_family_act: str
    candidate_message_present: bool
    candidate_message_sha256: str | None


class ConditionalCorpus:
    """Load the verified action-conditioned sparse corpus into compact tensors."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir.resolve()
        manifest_path = self.corpus_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != CONDITIONAL_CORPUS_CONTRACT or self.manifest.get("status") != "frozen-retrospective-core-corpus":
            raise ValueError("unsupported or unfrozen action-conditioned corpus")
        expected_spaces = {family: list(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES}
        if self.manifest.get("target_labels") != expected_spaces:
            raise ValueError("conditional corpus target-label space changed")
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("conditional corpus has no artifact receipts")
        for name in ("games.parquet", "events.parquet", "targets.parquet", "features.parquet"):
            receipt = artifacts.get(name)
            path = self.corpus_dir / name
            if not isinstance(receipt, Mapping) or not path.is_file() or file_sha256(path) != receipt.get("sha256"):
                raise RuntimeError(f"conditional corpus artifact hash mismatch: {name}")
        targets = pl.read_parquet(self.corpus_dir / "targets.parquet")
        features = pl.read_parquet(self.corpus_dir / "features.parquet")
        games = pl.read_parquet(self.corpus_dir / "games.parquet").select("game_id", "family", "our_role")
        joined = targets.join(features, on=["sample_id", "game_id"], how="inner").join(games, on=["game_id"], how="inner", suffix="_game")
        if joined.height != targets.height or joined["sample_id"].n_unique() != joined.height:
            raise RuntimeError("conditional feature and target rows are not one-to-one")
        rows = joined.sort(["family", "source_call_ts", "game_id", "round_number"]).to_dicts()
        feature_tensor = torch.zeros(len(rows), FEATURE_DIMENSION, dtype=torch.float32)
        action_mask = torch.zeros(len(rows), MAX_RESPONSE_CLASSES, dtype=torch.bool)
        labels = torch.zeros(len(rows), dtype=torch.long)
        families = torch.zeros(len(rows), dtype=torch.long)
        metadata: list[ConditionalRowMetadata] = []
        projector = CandidateActionProjector()
        for row_index, row in enumerate(rows):
            family = str(row.get("family_game") or row["family"])
            if family != str(row["family"]):
                raise RuntimeError("conditional feature and game families disagree")
            base_indices = [int(value) for value in row["feature_indices"]]
            base_values = [float(value) for value in row["feature_values"]]
            if object_sha256({"indices": base_indices, "values": base_values}) != row["feature_vector_sha256"]:
                raise RuntimeError("base engineered-feature vector hash mismatch")
            candidate = CandidateAction.from_feature_row(row, family=family, phase=str(row["phase"]))
            candidate_vector = projector.project(candidate)
            if list(candidate_vector.indices) != [int(value) for value in row["candidate_feature_indices"]] or list(candidate_vector.values) != [float(value) for value in row["candidate_feature_values"]] or candidate_vector.vector_sha256 != row["candidate_feature_vector_sha256"]:
                raise RuntimeError("candidate-action projection does not reproduce")
            conditioned = merge_sparse_vectors(base_indices, base_values, candidate_vector)
            if conditioned.vector_sha256 != row["conditioned_feature_vector_sha256"]:
                raise RuntimeError("conditioned engineered-feature vector hash mismatch")
            if conditioned.indices:
                feature_tensor[row_index, torch.tensor(conditioned.indices, dtype=torch.long)] = torch.tensor(conditioned.values, dtype=torch.float32)
            response_labels = list(CONDITIONAL_TARGET_LABELS[family])
            try:
                labels[row_index] = response_labels.index(str(row["target_label"]))
            except ValueError as error:
                raise RuntimeError(f"unknown conditional response label for {family}: {row['target_label']!r}") from error
            action_mask[row_index, : len(response_labels)] = True
            families[row_index] = FAMILY_INDEX[family]
            metadata.append(
                ConditionalRowMetadata(
                    sample_id=str(row["sample_id"]),
                    game_id=str(row["game_id"]),
                    family=family,
                    chronological_split=str(row["chronological_split"]),
                    identity_scope=str(row["identity_scope"]),
                    our_role=str(row["our_role"]),
                    phase=str(row["phase"]),
                    round_number=int(row["round_number"]),
                    target_label=str(row["target_label"]),
                    candidate_kind=candidate.kind,
                    candidate_action_label=candidate.action_label,
                    candidate_action_value=candidate.action_value,
                    candidate_action_aux_value=candidate.action_aux_value,
                    candidate_round_phase=candidate.round_phase,
                    candidate_message_family_act=candidate.message_family_act,
                    candidate_message_present=candidate.message_present,
                    candidate_message_sha256=candidate.message_sha256,
                )
            )
        self.features = feature_tensor
        self.action_mask = action_mask
        self.labels = labels
        self.families = families
        self.metadata = tuple(metadata)
        self.row_by_sample_id = {row.sample_id: index for index, row in enumerate(self.metadata)}
        if len(self.row_by_sample_id) != len(self.metadata):
            raise RuntimeError("conditional corpus contains duplicate sample IDs")
        self.indices_by_split = {split: torch.tensor([index for index, row in enumerate(self.metadata) if row.chronological_split == split], dtype=torch.long) for split in ("train", "validation", "test")}
        if any(len(indices) == 0 for indices in self.indices_by_split.values()):
            raise RuntimeError("conditional corpus has an empty chronological split")

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(self.corpus_dir),
            "manifest_sha256": file_sha256(self.corpus_dir / "manifest.json"),
            "rows": len(self.metadata),
            "split_rows": {split: len(indices) for split, indices in self.indices_by_split.items()},
            "family_rows": dict(sorted(Counter(row.family for row in self.metadata).items())),
        }


class ConditionalFeatureModel(nn.Module):
    """Predict the direct response from exact synthetic fields plus one candidate action."""

    def __init__(self, config: ConditionalTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Dropout(config.input_dropout),
            nn.Linear(FEATURE_DIMENSION, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(config.latent_dim),
            nn.Dropout(config.dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(config.latent_dim, len(CONDITIONAL_TARGET_LABELS[family])) for family in GLEE_FAMILIES])

    def forward(self, features: torch.Tensor, families: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        logits = hidden.new_full((features.shape[0], MAX_RESPONSE_CLASSES), -1e9)
        for family_index, family in enumerate(GLEE_FAMILIES):
            rows = torch.nonzero(families == family_index, as_tuple=False).squeeze(-1)
            if rows.numel():
                logits[rows, : len(CONDITIONAL_TARGET_LABELS[family])] = self.heads[family_index](hidden.index_select(0, rows))
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


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


def _family_weights(corpus: ConditionalCorpus, indices: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(corpus.families.index_select(0, indices), minlength=len(GLEE_FAMILIES)).float()
    if bool((counts == 0).any()):
        raise RuntimeError("conditional training split lacks one family")
    return len(indices) / (len(GLEE_FAMILIES) * counts)


@torch.inference_mode()
def _feature_probabilities(model: ConditionalFeatureModel, corpus: ConditionalCorpus, indices: torch.Tensor, device: torch.device, *, batch_size: int) -> torch.Tensor:
    model.eval()
    probabilities = torch.zeros(len(indices), MAX_RESPONSE_CLASSES, dtype=torch.float32)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        logits = model(corpus.features.index_select(0, selected).to(device), corpus.families.index_select(0, selected).to(device))
        probabilities[start : start + len(selected)] = F.softmax(logits, dim=-1).cpu()
    return probabilities


def _validation_score(model: ConditionalFeatureModel, corpus: ConditionalCorpus, device: torch.device, config: ConditionalTrainingConfig) -> float:
    indices = corpus.indices_by_split["validation"]
    probabilities = _feature_probabilities(model, corpus, indices, device, batch_size=config.evaluation_batch_size)
    family_scores: list[float] = []
    for family_index, _family in enumerate(GLEE_FAMILIES):
        mask = corpus.families.index_select(0, indices) == family_index
        labels = corpus.labels.index_select(0, indices)[mask]
        family_scores.append(float(-torch.log(probabilities[mask].gather(1, labels.unsqueeze(1)).clamp_min(1e-12)).mean()))
    return sum(family_scores) / len(family_scores)


def _train_feature_model(seed: int, corpus: ConditionalCorpus, device: torch.device, config: ConditionalTrainingConfig) -> tuple[ConditionalFeatureModel, dict[str, object]]:
    _set_seed(seed)
    model = ConditionalFeatureModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_indices = corpus.indices_by_split["train"]
    weights = _family_weights(corpus, train_indices).to(device)
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
            families = corpus.families.index_select(0, batch_indices).to(device)
            labels = corpus.labels.index_select(0, batch_indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            row_loss = F.cross_entropy(model(features, families), labels, reduction="none")
            row_weights = weights[families]
            loss = (row_loss * row_weights).sum() / row_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float((row_loss.detach() * row_weights).sum().cpu())
            total_weight += float(row_weights.sum().cpu())
        score = _validation_score(model, corpus, device, config)
        improved = score < best_score - 1e-5
        history.append({"epoch": epoch, "train_equal_family_weighted_nll": total_loss / max(total_weight, 1e-12), "validation_equal_family_nll": score, "improved": improved})
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("conditional feature training never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_equal_family_nll": best_score, "history": history}


def _sequence_probabilities(corpus: ConditionalCorpus, release_dir: Path, *, batch_size: int) -> tuple[dict[int, torch.Tensor], list[dict[str, object]], dict[str, object]]:
    candidate = ShadowCandidate(release_dir)
    index = CorpusIndex([corpus.corpus_dir])
    if {family: index.vocabs.target_labels[family].values for family in GLEE_FAMILIES} != CONDITIONAL_TARGET_LABELS:
        raise RuntimeError("conditional sequence release target labels changed")
    seeds = [seed for seed, _model in candidate.models]
    component_tensors = {seed: torch.zeros(len(corpus.metadata), MAX_RESPONSE_CLASSES, dtype=torch.float32) for seed in seeds}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for family in GLEE_FAMILIES:
        targets = [target for target in index.targets if index.games[str(target["game_id"])]["family"] == family]
        samples = [{"game": index.games[str(target["game_id"])], "events": index.events[str(target["game_id"])][: int(target["prefix_length"])], "target": target} for target in targets]
        for start in range(0, len(samples), batch_size):
            predictions = candidate.predict(samples[start : start + batch_size], family=family)
            for target, prediction in zip(targets[start : start + batch_size], predictions, strict=True):
                sample_id = str(target["sample_id"])
                row_index = corpus.row_by_sample_id[sample_id]
                expected_labels = list(CONDITIONAL_TARGET_LABELS[family])
                if prediction["labels"] != expected_labels:
                    raise RuntimeError("conditional sequence release label order changed")
                for seed in seeds:
                    values = [float(value) for value in prediction["component_probabilities"][str(seed)]]
                    component_tensors[seed][row_index, : len(values)] = torch.tensor(values)
                rows.append({"sample_id": sample_id, "game_id": str(target["game_id"]), "family": family, "labels": expected_labels, "component_probabilities": prediction["component_probabilities"], "ensemble_probabilities": prediction["action_probabilities"]})
                seen.add(sample_id)
    if len(seen) != len(corpus.metadata):
        raise RuntimeError("conditional sequence release did not cover every sample")
    return component_tensors, rows, {"path": str(Path(release_dir).resolve()), "manifest_sha256": file_sha256(Path(release_dir).resolve() / "manifest.json"), "candidate_id": candidate.candidate_id, "seeds": seeds}


def _prediction_rows(corpus: ConditionalCorpus, indices: torch.Tensor, probabilities: torch.Tensor, *, arm: str, seed: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, row_index in enumerate(indices.tolist()):
        metadata = corpus.metadata[row_index]
        classes = len(CONDITIONAL_TARGET_LABELS[metadata.family])
        values = [float(value) for value in probabilities[offset, :classes].tolist()]
        actual = int(corpus.labels[row_index])
        predicted = max(range(classes), key=values.__getitem__)
        rows.append(
            {
                **asdict(metadata),
                "arm": arm,
                "seed": seed,
                "actual_action": actual,
                "predicted_action": predicted,
                "probabilities": values,
                "negative_log_likelihood": -math.log(max(values[actual], 1e-12)),
                "brier_score": sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(values)),
                "confidence": max(values),
                "correct": int(predicted == actual),
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "games": 0, "negative_log_likelihood": None, "brier_score": None, "accuracy": None, "game_macro_negative_log_likelihood": None}
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
    }


def _evaluation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family = {family: _metrics([row for row in rows if row["family"] == family]) for family in GLEE_FAMILIES}
    return {
        "all": _metrics(rows),
        "equal_family_negative_log_likelihood": sum(float(by_family[family]["negative_log_likelihood"]) for family in GLEE_FAMILIES) / len(GLEE_FAMILIES),
        "by_family": by_family,
        "by_identity_scope": {scope: _metrics([row for row in rows if row["identity_scope"] == scope]) for scope in ("known", "hidden")},
    }


def _fit_stack(corpus: ConditionalCorpus, sequence: torch.Tensor, engineered: torch.Tensor, config: ConditionalTrainingConfig) -> dict[str, float]:
    validation = corpus.indices_by_split["validation"]
    weights: dict[str, float] = {}
    for family_index, family in enumerate(GLEE_FAMILIES):
        mask = corpus.families.index_select(0, validation) == family_index
        selected = validation[mask]
        labels = corpus.labels.index_select(0, selected)
        best_weight = 0.0
        best_nll = math.inf
        for step in range(config.stack_grid_steps + 1):
            weight = step / config.stack_grid_steps
            probabilities = weight * sequence.index_select(0, selected) + (1.0 - weight) * engineered.index_select(0, selected)
            nll = float(-torch.log(probabilities.gather(1, labels.unsqueeze(1)).clamp_min(1e-12)).mean())
            if nll < best_nll - 1e-6:
                best_nll = nll
                best_weight = weight
        weights[family] = best_weight
    return weights


def _stack_probabilities(corpus: ConditionalCorpus, sequence: torch.Tensor, engineered: torch.Tensor, weights: Mapping[str, float]) -> torch.Tensor:
    result = torch.zeros_like(sequence)
    for family_index, family in enumerate(GLEE_FAMILIES):
        rows = torch.nonzero(corpus.families == family_index, as_tuple=False).squeeze(-1)
        weight = float(weights[family])
        result[rows] = weight * sequence.index_select(0, rows) + (1.0 - weight) * engineered.index_select(0, rows)
    return result


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_conditional_experiment(*, corpus_dir: Path, sequence_release: Path, output_dir: Path, config: ConditionalTrainingConfig | None = None) -> dict[str, object]:
    selected = config or ConditionalTrainingConfig()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"conditional experiment output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    models_dir = staging / "models"
    models_dir.mkdir()
    corpus = ConditionalCorpus(corpus_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_components, sequence_rows, sequence_receipt = _sequence_probabilities(corpus, sequence_release, batch_size=selected.evaluation_batch_size)
    if set(sequence_components) != set(selected.seeds):
        raise RuntimeError("sequence and engineered expert seeds differ")
    sequence_ensemble = torch.stack([sequence_components[seed] for seed in selected.seeds]).mean(dim=0)
    engineered_components: dict[int, torch.Tensor] = {}
    training_receipts: dict[str, object] = {}
    all_prediction_rows: list[dict[str, object]] = []
    evaluations: dict[str, object] = {}
    for seed in selected.seeds:
        model, training = _train_feature_model(seed, corpus, device, selected)
        all_indices = torch.arange(len(corpus.metadata), dtype=torch.long)
        engineered_components[seed] = _feature_probabilities(model, corpus, all_indices, device, batch_size=selected.evaluation_batch_size)
        checkpoint_path = models_dir / f"engineered-seed{seed}.pt"
        torch.save({"contract": CONDITIONAL_EXPERIMENT_CONTRACT, "seed": seed, "config": asdict(selected), "model": model.state_dict(), "feature_dimension": FEATURE_DIMENSION, "target_labels": {family: list(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES}}, checkpoint_path)
        training_receipts[str(seed)] = {"training": training, "parameters": model.parameter_count(), "checkpoint": {"path": str(checkpoint_path.relative_to(staging)), "sha256": file_sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}}
    engineered_ensemble = torch.stack([engineered_components[seed] for seed in selected.seeds]).mean(dim=0)
    stack_weights = _fit_stack(corpus, sequence_ensemble, engineered_ensemble, selected)
    stack = _stack_probabilities(corpus, sequence_ensemble, engineered_ensemble, stack_weights)
    arms: dict[str, torch.Tensor] = {"sequence-ensemble": sequence_ensemble, "engineered-ensemble": engineered_ensemble, "convex-stack": stack}
    for seed in selected.seeds:
        arms[f"sequence-seed{seed}"] = sequence_components[seed]
        arms[f"engineered-seed{seed}"] = engineered_components[seed]
    for arm, probabilities in arms.items():
        evaluations[arm] = {}
        for split, indices in corpus.indices_by_split.items():
            rows = _prediction_rows(corpus, indices, probabilities.index_select(0, indices), arm=arm, seed=None)
            evaluations[arm][split] = _evaluation(rows)
            all_prediction_rows.extend(rows)
    sequence_predictions_path = staging / "sequence-predictions.parquet"
    pl.DataFrame(sequence_rows, infer_schema_length=None).sort("sample_id").write_parquet(sequence_predictions_path, compression="zstd", compression_level=7, statistics=True)
    predictions_path = staging / "predictions.parquet"
    pl.DataFrame(all_prediction_rows, infer_schema_length=None).write_parquet(predictions_path, compression="zstd", compression_level=7, statistics=True)
    validation_winner = min(("sequence-ensemble", "engineered-ensemble", "convex-stack"), key=lambda arm: float(evaluations[arm]["validation"]["equal_family_negative_log_likelihood"]))
    result = {
        "schema_version": 1,
        "contract": CONDITIONAL_EXPERIMENT_CONTRACT,
        "status": "complete",
        "corpus": corpus.receipt(),
        "sequence_release": sequence_receipt,
        "configuration": asdict(selected),
        "device": {"type": device.type, "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__, "cuda": torch.version.cuda},
        "engineered_models": training_receipts,
        "stack": {"rule": "per-family convex probability mixture: weight * sequence + (1 - weight) * engineered", "sequence_weights": stack_weights, "fit_split": "validation", "grid_steps": selected.stack_grid_steps},
        "evaluation": evaluations,
        "validation_winner": validation_winner,
        "artifacts": {
            "sequence_predictions": {"path": sequence_predictions_path.name, "sha256": file_sha256(sequence_predictions_path), "rows": len(sequence_rows)},
            "predictions": {"path": predictions_path.name, "sha256": file_sha256(predictions_path), "rows": len(all_prediction_rows)},
        },
        "implementation_sha256": file_sha256(Path(__file__)),
        "authority": "Adaptive retrospective diagnostic only. Any frozen derivative remains behaviorally inert prospective shadow until a clean post-freeze stream validates it.",
    }
    _atomic_json(staging / "result.json", result)
    os.replace(staging, output)
    return {**result, "output_dir": str(output), "result_sha256": file_sha256(output / "result.json")}
