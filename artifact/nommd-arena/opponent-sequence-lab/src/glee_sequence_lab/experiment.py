"""Train and evaluate frozen hierarchical sequence-twin experiment arms."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .corpus import GLEE_FAMILIES, canonical_json, file_sha256
from . import data as data_module
from . import model as model_module
from .data import CorpusIndex, SequenceCollator, move_batch
from .model import HierarchicalSequenceTwin, ModelConfig, gaussian_negative_log_likelihood, sequence_twin_loss


EXPERIMENT_CONTRACT = "glee-hierarchical-sequence-twin-experiment-v3"
OBJECTIVE_CONTRACT = "strategic-action-value-delay-v2"
MAMBA2_DEFAULT_EXPANSION = 2


def validate_model_config(config: ModelConfig) -> None:
    """Reject Mamba-2 widths that its default expansion cannot partition into heads."""
    if config.core != "mamba2":
        return
    if config.mamba_head_dim < 1 or (MAMBA2_DEFAULT_EXPANSION * config.model_dim) % config.mamba_head_dim:
        raise ValueError("Mamba-2 expanded model_dim must be divisible by mamba_head_dim")
    if config.event_streams == "separate-head-gated" and (MAMBA2_DEFAULT_EXPANSION * config.message_model_dim) % config.mamba_head_dim:
        raise ValueError("Mamba-2 expanded message_model_dim must be divisible by mamba_head_dim")


@dataclass(frozen=True)
class TrainingConfig:
    arm: str
    label: str | None = None
    seed: int = 1729
    epochs: int = 12
    patience: int = 3
    batch_size: int = 256
    evaluation_batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 1e-3
    account_weight_decay: float = 2e-2
    account_dropout: float = 0.35
    gradient_clip: float = 1.0
    workers: int = 2
    mixed_precision: bool = True
    source_types: tuple[str, ...] = ("real",)
    account_disjoint_fold: int | None = None
    history_window: int | None = None
    mask_message_inputs: bool = False
    equal_family_weighting: bool = False
    selection_objective: str = "action"
    cuda_memory_fraction: float | None = None

    @property
    def force_population(self) -> bool:
        return self.arm.startswith("population-")

    @property
    def core(self) -> str:
        if self.arm.endswith("-gru"):
            return "gru"
        if self.arm.endswith("-transformer"):
            return "transformer"
        if self.arm.endswith("-mamba2"):
            return "mamba2"
        if self.arm.endswith("-mamba3-siso"):
            return "mamba3-siso"
        raise ValueError(f"unsupported experiment arm: {self.arm}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sha_object(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _loaders(index: CorpusIndex, *, split: str, batch_size: int, workers: int, source_types: set[str], seed: int, account_disjoint_fold: int | None = None, history_window: int | None = None, mask_message_inputs: bool = False) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for family_index, family in enumerate(GLEE_FAMILIES):
        dataset = index.subset(family=family, split=split, source_types=source_types, account_disjoint_fold=account_disjoint_fold)
        generator = torch.Generator().manual_seed(seed + family_index * 101 + {"train": 0, "validation": 1, "test": 2}[split])
        loaders[family] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=SequenceCollator(index.vocabs, family, excluded_account_fold=account_disjoint_fold, history_window=history_window, mask_message_inputs=mask_message_inputs),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            generator=generator,
        )
    return loaders


def _interleaved(loaders: Mapping[str, DataLoader], *, seed: int) -> Iterator[dict[str, object]]:
    iterators = {family: iter(loader) for family, loader in loaders.items()}
    remaining = {family: len(loader) for family, loader in loaders.items()}
    rng = random.Random(seed)
    while sum(remaining.values()) > 0:
        population = [family for family, count in remaining.items() if count > 0]
        weights = [remaining[family] for family in population]
        family = rng.choices(population, weights=weights, k=1)[0]
        try:
            yield next(iterators[family])
        except StopIteration as error:
            raise RuntimeError(f"loader length contract failed for {family}") from error
        remaining[family] -= 1


def _optimizer(model: HierarchicalSequenceTwin, config: TrainingConfig) -> torch.optim.Optimizer:
    account_parameters: list[torch.nn.Parameter] = []
    ordinary_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("account_global.") or name.startswith("account_family."):
            account_parameters.append(parameter)
        else:
            ordinary_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": ordinary_parameters, "weight_decay": config.weight_decay},
            {"params": account_parameters, "weight_decay": config.account_weight_decay},
        ],
        lr=config.learning_rate,
    )


def _train_epoch(model: HierarchicalSequenceTwin, loaders: Mapping[str, DataLoader], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device, config: TrainingConfig, *, epoch: int) -> dict[str, float]:
    model.train()
    sums: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    family_step_weights = {family: sum(len(loader) for loader in loaders.values()) / (len(loaders) * len(loaders[family])) for family in loaders} if config.equal_family_weighting else {family: 1.0 for family in loaders}
    for raw_batch in _interleaved(loaders, seed=config.seed + epoch * 1009):
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.mixed_precision and device.type == "cuda"):
            outputs = model(batch, force_population=config.force_population, account_dropout=0.0 if config.force_population else config.account_dropout)
            unweighted_loss, parts = sequence_twin_loss(outputs, batch)
            loss = unweighted_loss * family_step_weights[str(batch["family"])]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        count = int(batch["target_labels"].shape[0])
        sample_count += count
        for key, value in parts.items():
            sums[key] += float(value.cpu()) * count
    return {key: value / max(1, sample_count) for key, value in sorted(sums.items())} | {"samples": float(sample_count)}


class MetricAccumulator:
    def __init__(self) -> None:
        self.targets = 0
        self.action_rows = 0
        self.action_nll = 0.0
        self.action_brier = 0.0
        self.action_correct = 0
        self.value_rows = 0
        self.value_nll = 0.0
        self.value_absolute_error = 0.0
        self.delay_rows = 0
        self.delay_nll = 0.0
        self.delay_absolute_error = 0.0
        self.gate_sums: defaultdict[str, float] = defaultdict(float)
        self.gate_rows: defaultdict[str, int] = defaultdict(int)
        self.game_action_nll: defaultdict[str, list[float]] = defaultdict(list)

    def add(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, object]) -> None:
        labels = batch["target_labels"]
        action_mask = batch["target_action_mask"]
        log_probabilities = F.log_softmax(outputs["action_logits"], dim=-1)
        probabilities = log_probabilities.exp()
        row_nll = -log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        one_hot = F.one_hot(labels, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
        row_brier = (probabilities - one_hot).square().sum(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        self.targets += int(labels.numel())
        self.action_rows += int(action_mask.sum().cpu())
        self.action_nll += float(row_nll[action_mask].sum().cpu())
        self.action_brier += float(row_brier[action_mask].sum().cpu())
        self.action_correct += int((predictions[action_mask] == labels[action_mask]).sum().cpu())
        for metadata, loss, included in zip(batch["metadata"], row_nll.detach().cpu().tolist(), action_mask.detach().cpu().tolist(), strict=True):
            if included:
                self.game_action_nll[str(metadata["game_id"])].append(float(loss))
        value_mask = batch["target_value_mask"]
        if value_mask.any():
            value_loss = gaussian_negative_log_likelihood(batch["target_values"], outputs["value_location"], outputs["value_log_scale"])
            self.value_rows += int(value_mask.sum().cpu())
            self.value_nll += float(value_loss[value_mask].sum().cpu())
            self.value_absolute_error += float((batch["target_values"] - outputs["value_location"]).abs()[value_mask].sum().cpu())
        delay_mask = batch["target_delay_mask"]
        if delay_mask.any():
            delay_loss = gaussian_negative_log_likelihood(batch["target_delays"], outputs["delay_location"], outputs["delay_log_scale"])
            self.delay_rows += int(delay_mask.sum().cpu())
            self.delay_nll += float(delay_loss[delay_mask].sum().cpu())
            self.delay_absolute_error += float((batch["target_delays"] - outputs["delay_location"]).abs()[delay_mask].sum().cpu())
        for target, mask in (("action", action_mask), ("value", value_mask), ("delay", delay_mask)):
            key = f"{target}_message_gate"
            if key in outputs and mask.any():
                self.gate_sums[target] += float(outputs[key][mask].sum().cpu())
                self.gate_rows[target] += int(mask.sum().cpu())

    def receipt(self) -> dict[str, object]:
        game_means = [sum(values) / len(values) for values in self.game_action_nll.values() if values]
        return {
            "targets": self.targets,
            "action_rows": self.action_rows,
            "games": len(game_means),
            "action": {
                "negative_log_likelihood": self.action_nll / self.action_rows if self.action_rows else None,
                "brier_score": self.action_brier / self.action_rows if self.action_rows else None,
                "accuracy": self.action_correct / self.action_rows if self.action_rows else None,
                "game_macro_negative_log_likelihood": sum(game_means) / len(game_means) if game_means else None,
            },
            "value": {"rows": self.value_rows, "negative_log_likelihood": self.value_nll / self.value_rows if self.value_rows else None, "mean_absolute_error": self.value_absolute_error / self.value_rows if self.value_rows else None},
            "delay": {"rows": self.delay_rows, "negative_log_likelihood": self.delay_nll / self.delay_rows if self.delay_rows else None, "log_mean_absolute_error": self.delay_absolute_error / self.delay_rows if self.delay_rows else None},
            "message_fusion": {target: {"rows": self.gate_rows[target], "mean_gate": self.gate_sums[target] / self.gate_rows[target] if self.gate_rows[target] else None} for target in ("action", "value", "delay")},
        }


@torch.inference_mode()
def evaluate(model: HierarchicalSequenceTwin, loaders: Mapping[str, DataLoader], device: torch.device, *, force_population: bool, prediction_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    model.eval()
    aggregate = MetricAccumulator()
    per_family = {family: MetricAccumulator() for family in GLEE_FAMILIES}
    per_identity = {scope: MetricAccumulator() for scope in ("known", "hidden")}
    for raw_batch in _interleaved(loaders, seed=0):
        batch = move_batch(raw_batch, device)
        outputs = model(batch, force_population=force_population)
        if prediction_rows is not None:
            action_log_probabilities = F.log_softmax(outputs["action_logits"], dim=-1)
            action_probabilities = action_log_probabilities.exp()
            value_losses = gaussian_negative_log_likelihood(batch["target_values"], outputs["value_location"], outputs["value_log_scale"])
            delay_losses = gaussian_negative_log_likelihood(batch["target_delays"], outputs["delay_location"], outputs["delay_log_scale"])
            for index, metadata in enumerate(batch["metadata"]):
                actual_action = int(batch["target_labels"][index].cpu())
                action_scored = bool(batch["target_action_mask"][index].cpu())
                prediction_rows.append(
                    {
                        **metadata,
                        "family": str(batch["family"]),
                        "force_population": force_population,
                        "action_scored": action_scored,
                        "actual_action": actual_action,
                        "predicted_action": int(action_probabilities[index].argmax().cpu()) if action_scored else None,
                        "action_negative_log_likelihood": float(-action_log_probabilities[index, actual_action].cpu()) if action_scored else None,
                        "action_probabilities": [float(value) for value in action_probabilities[index].cpu().tolist()] if action_scored else None,
                        "actual_value": float(batch["target_values"][index].cpu()) if bool(batch["target_value_mask"][index].cpu()) else None,
                        "predicted_value": float(outputs["value_location"][index].cpu()) if bool(batch["target_value_mask"][index].cpu()) else None,
                        "value_negative_log_likelihood": float(value_losses[index].cpu()) if bool(batch["target_value_mask"][index].cpu()) else None,
                        "actual_delay": float(batch["target_delays"][index].cpu()) if bool(batch["target_delay_mask"][index].cpu()) else None,
                        "predicted_delay": float(outputs["delay_location"][index].cpu()) if bool(batch["target_delay_mask"][index].cpu()) else None,
                        "delay_negative_log_likelihood": float(delay_losses[index].cpu()) if bool(batch["target_delay_mask"][index].cpu()) else None,
                        "action_message_gate": float(outputs["action_message_gate"][index].cpu()) if "action_message_gate" in outputs and action_scored else None,
                        "value_message_gate": float(outputs["value_message_gate"][index].cpu()) if "value_message_gate" in outputs and bool(batch["target_value_mask"][index].cpu()) else None,
                        "delay_message_gate": float(outputs["delay_message_gate"][index].cpu()) if "delay_message_gate" in outputs and bool(batch["target_delay_mask"][index].cpu()) else None,
                    }
                )
        aggregate.add(outputs, batch)
        family = str(batch["family"])
        per_family[family].add(outputs, batch)
        for scope in ("known", "hidden"):
            indices = [index for index, metadata in enumerate(batch["metadata"]) if metadata["identity_scope"] == scope]
            if not indices:
                continue
            selection = torch.tensor(indices, device=device)
            subset_batch = {key: value.index_select(0, selection) if isinstance(value, torch.Tensor) and value.shape[:1] == batch["target_labels"].shape[:1] else value for key, value in batch.items()}
            subset_batch["metadata"] = [batch["metadata"][index] for index in indices]
            subset_outputs = {key: value.index_select(0, selection) if isinstance(value, torch.Tensor) and value.shape[:1] == batch["target_labels"].shape[:1] else value for key, value in outputs.items()}
            per_identity[scope].add(subset_outputs, subset_batch)
    return {"all": aggregate.receipt(), "by_family": {family: accumulator.receipt() for family, accumulator in per_family.items()}, "by_identity_scope": {scope: accumulator.receipt() for scope, accumulator in per_identity.items()}}


def _selection_score(metrics: Mapping[str, object], *, objective: str = "action") -> float:
    by_family = metrics["by_family"]
    if objective == "action":
        values = [float(by_family[family]["action"]["negative_log_likelihood"]) for family in GLEE_FAMILIES]
    elif objective == "joint-self-mirror":
        values = []
        for family in GLEE_FAMILIES:
            action = by_family[family]["action"]["negative_log_likelihood"]
            value = by_family[family]["value"]["mean_absolute_error"]
            components = [float(component) for component in (action, value) if component is not None]
            if not components:
                raise RuntimeError(f"self-mirror validation has no strategic target for {family}")
            values.append(sum(components) / len(components))
    else:
        raise ValueError(f"unsupported validation selection objective: {objective}")
    return sum(values) / len(values)


def _save_checkpoint(path: Path, model: HierarchicalSequenceTwin, optimizer: torch.optim.Optimizer, *, epoch: int, score: float) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "score": score}, temporary)
    os.replace(temporary, path)


def run_experiment(*, corpus_dirs: Sequence[Path], output_dir: Path, training: TrainingConfig, model_config: ModelConfig | None = None, initial_checkpoint: Path | None = None) -> dict[str, object]:
    started = time.time()
    selected_model = model_config or ModelConfig(core=training.core)
    validate_model_config(selected_model)
    if selected_model.core != training.core:
        raise ValueError("training arm and model core disagree")
    if output_dir.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    _set_seed(training.seed)
    if training.selection_objective not in {"action", "joint-self-mirror"}:
        raise ValueError(f"unsupported validation selection objective: {training.selection_objective}")
    if training.cuda_memory_fraction is not None and not 0.05 <= training.cuda_memory_fraction <= 1.0:
        raise ValueError("cuda_memory_fraction must be between 0.05 and 1.0")
    index = CorpusIndex(corpus_dirs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if training.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(training.cuda_memory_fraction, device=0)
        torch.cuda.reset_peak_memory_stats(device)
    model = HierarchicalSequenceTwin(index.vocabs, selected_model).to(device)
    initialization: dict[str, object] | None = None
    if initial_checkpoint is not None:
        initial_path = initial_checkpoint.resolve()
        initial = torch.load(initial_path, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
        initialization = {"path": str(initial_path), "sha256": file_sha256(initial_path), "source_epoch": initial.get("epoch"), "source_score": initial.get("score")}
    optimizer = _optimizer(model, training)
    scaler = torch.amp.GradScaler(device.type, enabled=training.mixed_precision and device.type == "cuda")
    source_types = set(training.source_types)
    train_loaders = _loaders(index, split="train", batch_size=training.batch_size, workers=training.workers, source_types=source_types, seed=training.seed, account_disjoint_fold=training.account_disjoint_fold, history_window=training.history_window, mask_message_inputs=training.mask_message_inputs)
    validation_loaders = _loaders(index, split="validation", batch_size=training.evaluation_batch_size, workers=training.workers, source_types={"real"}, seed=training.seed, account_disjoint_fold=training.account_disjoint_fold, history_window=training.history_window, mask_message_inputs=training.mask_message_inputs)
    history: list[dict[str, object]] = []
    best_score = math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint = output_dir / "best.pt"
    if training.epochs == 0:
        if initial_checkpoint is None or initialization is None:
            raise ValueError("zero-epoch evaluation requires an initial checkpoint")
        best_score = float(initialization["source_score"])
        best_epoch = int(initialization["source_epoch"])
        _save_checkpoint(checkpoint, model, optimizer, epoch=best_epoch, score=best_score)
    for epoch in range(1, training.epochs + 1):
        train_metrics = _train_epoch(model, train_loaders, optimizer, scaler, device, training, epoch=epoch)
        validation = evaluate(model, validation_loaders, device, force_population=training.force_population)
        score = _selection_score(validation, objective=training.selection_objective)
        improved = score < best_score - 1e-4
        history.append({"epoch": epoch, "train": train_metrics, "validation_selection_score": score, "validation": validation, "improved": improved})
        _atomic_json(output_dir / "progress.json", {"contract": EXPERIMENT_CONTRACT, "arm": training.arm, "history": history})
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(checkpoint, model, optimizer, epoch=epoch, score=score)
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    test_loaders = _loaders(index, split="test", batch_size=training.evaluation_batch_size, workers=training.workers, source_types={"real"}, seed=training.seed, account_disjoint_fold=training.account_disjoint_fold, history_window=training.history_window, mask_message_inputs=training.mask_message_inputs)
    validation_primary = evaluate(model, validation_loaders, device, force_population=training.force_population)
    test_prediction_rows: list[dict[str, object]] = []
    test_primary = evaluate(model, test_loaders, device, force_population=training.force_population, prediction_rows=test_prediction_rows)
    account_ablation = None
    ablation_prediction_rows: list[dict[str, object]] = []
    if not training.force_population:
        account_ablation = {"validation": evaluate(model, validation_loaders, device, force_population=True), "test": evaluate(model, test_loaders, device, force_population=True, prediction_rows=ablation_prediction_rows)}
    prediction_path = output_dir / "test-predictions.parquet"
    pl.DataFrame(test_prediction_rows, infer_schema_length=None).write_parquet(prediction_path, compression="zstd", compression_level=7, statistics=True)
    ablation_prediction = None
    if ablation_prediction_rows:
        ablation_path = output_dir / "test-predictions-population-ablation.parquet"
        pl.DataFrame(ablation_prediction_rows, infer_schema_length=None).write_parquet(ablation_path, compression="zstd", compression_level=7, statistics=True)
        ablation_prediction = {"path": ablation_path.name, "rows": len(ablation_prediction_rows), "sha256": file_sha256(ablation_path)}
    receipt = {
        "contract": EXPERIMENT_CONTRACT,
        "objective": {
            "contract": OBJECTIVE_CONTRACT,
            "predicted": ["categorical strategic action", "numeric move value", "log response delay"],
            "message_role": "causally prior input evidence only; a move expressed in a message is represented by the action and numeric-value targets",
            "message_reconstruction": False,
        },
        "arm": training.arm,
        "label": training.label or training.arm,
        "status": "complete",
        "training": asdict(training),
        "model": selected_model.receipt(),
        "model_parameters": model.parameter_count(),
        "device": {"type": device.type, "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_memory_fraction": training.cuda_memory_fraction, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
        "corpora": [{"path": str(path.resolve()), "manifest_sha256": file_sha256(path.resolve() / "manifest.json")} for path in corpus_dirs],
        "implementation": {"experiment_sha256": file_sha256(Path(__file__)), "data_sha256": file_sha256(Path(data_module.__file__)), "model_sha256": file_sha256(Path(model_module.__file__))},
        "vocabulary_sha256": _sha_object(index.vocabs.receipt()),
        "initialization": initialization,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "history": history,
        "evaluation": {"validation": validation_primary, "test": test_primary, "account_ablation": account_ablation},
        "test_predictions": {"path": prediction_path.name, "rows": len(test_prediction_rows), "sha256": file_sha256(prediction_path)},
        "account_ablation_test_predictions": ablation_prediction,
        "elapsed_seconds": time.time() - started,
        "promotion": "offline predictive evidence only; no live authority",
    }
    _atomic_json(output_dir / "result.json", receipt)
    return receipt
