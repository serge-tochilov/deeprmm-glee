"""Frozen action-only sequence candidates and append-only prospective shadow receipts."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl
import torch
from torch.nn import functional as F

from . import data as data_module
from . import model as model_module
from .corpus import GLEE_FAMILIES, canonical_json, file_sha256, object_sha256
from .data import CorpusIndex, CorpusVocabs, SequenceCollator, move_batch
from .model import HierarchicalSequenceTwin, ModelConfig


SHADOW_CANDIDATE_CONTRACT = "glee-sequence-shadow-candidate-v1"
SHADOW_COMPONENT_CONTRACT = "glee-sequence-shadow-component-v1"
SHADOW_REGISTRY_CONTRACT = "glee-sequence-prospective-shadow-registry-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_completed_component(directory: Path) -> tuple[dict[str, object], Path]:
    resolved = directory.resolve()
    result_path = resolved / "result.json"
    checkpoint_path = resolved / "best.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("contract") != "glee-hierarchical-sequence-twin-experiment-v3":
        raise ValueError(f"component is not a completed v3 experiment: {resolved}")
    model = _require_mapping(result.get("model"), name="model")
    training = _require_mapping(result.get("training"), name="training")
    if result.get("arm") != "population-mamba2" or model.get("core") != "mamba2":
        raise ValueError(f"component is not a population Mamba-2 arm: {resolved}")
    if model.get("event_streams") != "separate-head-gated" or model.get("delay_message_mode") != "gated":
        raise ValueError(f"component is not the all-gated dual-stream candidate: {resolved}")
    if not isinstance(training.get("seed"), int) or isinstance(training.get("seed"), bool):
        raise ValueError(f"component has no integer seed: {resolved}")
    if not checkpoint_path.is_file():
        raise ValueError(f"component checkpoint is absent: {checkpoint_path}")
    return result, checkpoint_path


def _action_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "games": 0, "negative_log_likelihood": None, "brier_score": None, "accuracy": None, "game_macro_negative_log_likelihood": None}
    game_losses: dict[str, list[float]] = {}
    nll_sum = 0.0
    brier_sum = 0.0
    correct = 0
    for row in rows:
        probabilities = row["ensemble_probabilities"]
        actual = int(row["actual_action"])
        nll = -math.log(max(float(probabilities[actual]), 1e-12))
        nll_sum += nll
        brier_sum += sum((float(probability) - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities))
        correct += int(max(range(len(probabilities)), key=probabilities.__getitem__) == actual)
        game_losses.setdefault(str(row["game_id"]), []).append(nll)
    game_means = [sum(values) / len(values) for values in game_losses.values()]
    return {
        "rows": len(rows),
        "games": len(game_means),
        "negative_log_likelihood": nll_sum / len(rows),
        "brier_score": brier_sum / len(rows),
        "accuracy": correct / len(rows),
        "game_macro_negative_log_likelihood": sum(game_means) / len(game_means),
    }


def _retrospective_ensemble_evidence(experiment_dirs: Sequence[Path], results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    frames = [pl.read_parquet(directory / "test-predictions.parquet").filter(pl.col("action_scored")).sort("sample_id") for directory in experiment_dirs]
    reference_ids = frames[0]["sample_id"].to_list()
    if any(frame["sample_id"].to_list() != reference_ids for frame in frames[1:]):
        raise ValueError("shadow candidate component prediction rows are not aligned")
    component_rows = [frame.to_dicts() for frame in frames]
    rows: list[dict[str, object]] = []
    disagreements = 0
    for aligned in zip(*component_rows, strict=True):
        reference = aligned[0]
        coordinates = (reference["sample_id"], reference["game_id"], reference["family"], reference["target_kind"], reference["actual_action"])
        if any((row["sample_id"], row["game_id"], row["family"], row["target_kind"], row["actual_action"]) != coordinates for row in aligned[1:]):
            raise ValueError("shadow candidate component targets disagree")
        if any(len(row["action_probabilities"]) != len(reference["action_probabilities"]) for row in aligned[1:]):
            raise ValueError("shadow candidate component action vocabularies disagree")
        probabilities = [sum(float(row["action_probabilities"][index]) for row in aligned) / len(aligned) for index in range(len(reference["action_probabilities"]))]
        component_choices = [max(range(len(row["action_probabilities"])), key=row["action_probabilities"].__getitem__) for row in aligned]
        disagreements += int(len(set(component_choices)) > 1)
        rows.append({"sample_id": reference["sample_id"], "game_id": reference["game_id"], "family": reference["family"], "identity_scope": reference["identity_scope"], "target_kind": reference["target_kind"], "actual_action": reference["actual_action"], "ensemble_probabilities": probabilities})
    all_metrics = _action_metrics(rows)
    by_family = {family: _action_metrics([row for row in rows if row["family"] == family]) for family in GLEE_FAMILIES}
    by_identity = {scope: _action_metrics([row for row in rows if row["identity_scope"] == scope]) for scope in ("known", "hidden")}
    by_target_kind = {kind: _action_metrics([row for row in rows if row["target_kind"] == kind]) for kind in sorted({str(row["target_kind"]) for row in rows})}
    component_comparison: dict[str, object] = {}
    for result in results:
        seed = int(_require_mapping(result["training"], name="training")["seed"])
        action = _require_mapping(_require_mapping(_require_mapping(result["evaluation"], name="evaluation")["test"], name="test")["all"], name="all")["action"]
        action_metrics = _require_mapping(action, name="action")
        component_comparison[str(seed)] = {
            "ensemble_minus_component_negative_log_likelihood": float(all_metrics["negative_log_likelihood"]) - float(action_metrics["negative_log_likelihood"]),
            "ensemble_minus_component_accuracy": float(all_metrics["accuracy"]) - float(action_metrics["accuracy"]),
        }
    return {
        "status": "retrospective-held-out-evidence-only",
        "all": all_metrics,
        "by_family": by_family,
        "by_identity_scope": by_identity,
        "by_target_kind": by_target_kind,
        "component_argmax_disagreement_fraction": disagreements / len(rows),
        "relative_to_components": component_comparison,
    }


class ShadowCandidateBuilder:
    """Seal matched trained components into one portable, action-only shadow ensemble."""

    def __init__(self, *, experiment_dirs: Sequence[Path], output_dir: Path, candidate_id: str) -> None:
        self.experiment_dirs = tuple(path.resolve() for path in experiment_dirs)
        self.output_dir = output_dir.resolve()
        self.candidate_id = candidate_id

    def run(self) -> dict[str, object]:
        if len(self.experiment_dirs) < 2:
            raise ValueError("a shadow ensemble requires at least 2 independently seeded components")
        if not self.candidate_id.strip():
            raise ValueError("candidate ID cannot be empty")
        if self.output_dir.exists():
            raise FileExistsError(f"shadow candidate output already exists: {self.output_dir}")
        loaded = [_load_completed_component(directory) for directory in self.experiment_dirs]
        results = [result for result, _checkpoint in loaded]
        reference = results[0]
        reference_model = _require_mapping(reference["model"], name="model")
        reference_corpora = reference.get("corpora")
        reference_vocab = reference.get("vocabulary_sha256")
        seeds = [int(_require_mapping(result["training"], name="training")["seed"]) for result in results]
        if len(set(seeds)) != len(seeds):
            raise ValueError("shadow candidate component seeds must be distinct")
        for result in results[1:]:
            if _require_mapping(result["model"], name="model") != reference_model:
                raise ValueError("shadow candidate components have different model configurations")
            if result.get("corpora") != reference_corpora or result.get("vocabulary_sha256") != reference_vocab:
                raise ValueError("shadow candidate components have different corpora or vocabularies")
            if result.get("implementation") != reference.get("implementation"):
                raise ValueError("shadow candidate components have different implementation hashes")
        if not isinstance(reference_corpora, list) or len(reference_corpora) != 1 or not isinstance(reference_corpora[0], Mapping):
            raise ValueError("shadow candidate requires exactly one frozen corpus")
        corpus_dir = Path(str(reference_corpora[0]["path"]))
        index = CorpusIndex([corpus_dir])
        vocabularies = index.vocabs.receipt()
        if object_sha256(vocabularies) != reference_vocab:
            raise ValueError("reconstructed vocabulary does not match the training receipt")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            vocabulary_path = staging / "vocabulary.json"
            _write_json(vocabulary_path, vocabularies)
            ensemble_evidence = _retrospective_ensemble_evidence(self.experiment_dirs, results)
            components: list[dict[str, object]] = []
            for result, checkpoint_path in sorted(loaded, key=lambda item: int(_require_mapping(item[0]["training"], name="training")["seed"])):
                training = _require_mapping(result["training"], name="training")
                seed = int(training["seed"])
                source_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                component_name = f"component-seed{seed}.pt"
                component_path = staging / component_name
                torch.save(
                    {
                        "contract": SHADOW_COMPONENT_CONTRACT,
                        "candidate_id": self.candidate_id,
                        "seed": seed,
                        "model_config": dict(reference_model),
                        "model": source_checkpoint["model"],
                        "source_epoch": source_checkpoint.get("epoch"),
                        "source_score": source_checkpoint.get("score"),
                    },
                    component_path,
                )
                component_path.chmod(0o644)
                evaluation = _require_mapping(_require_mapping(result["evaluation"], name="evaluation")["test"], name="test evaluation")
                components.append(
                    {
                        "seed": seed,
                        "path": component_name,
                        "sha256": file_sha256(component_path),
                        "bytes": component_path.stat().st_size,
                        "source_experiment": str(self.experiment_dirs[seeds.index(seed)]),
                        "source_result_sha256": file_sha256(self.experiment_dirs[seeds.index(seed)] / "result.json"),
                        "source_checkpoint_sha256": file_sha256(checkpoint_path),
                        "source_epoch": result.get("best_epoch"),
                        "source_validation_score": result.get("best_validation_selection_score"),
                        "test": _require_mapping(evaluation["all"], name="all test metrics"),
                    }
                )
            manifest = {
                "schema_version": 1,
                "contract": SHADOW_CANDIDATE_CONTRACT,
                "candidate_id": self.candidate_id,
                "status": "frozen-prospective-shadow-candidate",
                "frozen_at": _utc_now(),
                "scope": "categorical strategic-action prediction only",
                "ensemble": {"components": len(components), "rule": "unweighted arithmetic mean of component categorical probabilities", "component_seeds": sorted(seeds), "minimum_cuda_batch": 8, "cuda_padding_rule": "repeat the final causal prefix to 8 rows and discard padded outputs; evaluation mode makes duplicates prediction-invariant"},
                "model": dict(reference_model),
                "model_parameters_per_component": reference.get("model_parameters"),
                "components": components,
                "retrospective_action_evidence": ensemble_evidence,
                "vocabulary": {"path": vocabulary_path.name, "sha256": file_sha256(vocabulary_path), "semantic_sha256": reference_vocab},
                "corpus": {"path_at_freeze": str(corpus_dir.resolve()), "manifest_sha256": file_sha256(corpus_dir / "manifest.json")},
                "training_implementation": reference.get("implementation"),
                "release_implementation": {
                    "shadow_sha256": file_sha256(Path(__file__)),
                    "data_sha256": file_sha256(Path(data_module.__file__)),
                    "model_sha256": file_sha256(Path(model_module.__file__)),
                },
                "boundary": "Prospective shadow only. The candidate cannot alter prompts, actions, timing, matchmaking, identity claims, ratings, deterministic safeguards, or model training. Numeric value and response delay remain diagnostics outside this release's candidate scope.",
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
            return {**manifest, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json")}
        except BaseException:
            if staging.exists():
                for path in staging.iterdir():
                    path.unlink()
                staging.rmdir()
            raise


class ShadowCandidate:
    """Fail-closed loader and predictor for a frozen action-only ensemble."""

    def __init__(self, release_dir: Path, *, device: str | None = None) -> None:
        self.release_dir = release_dir.resolve()
        self.manifest_path = self.release_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != SHADOW_CANDIDATE_CONTRACT or self.manifest.get("status") != "frozen-prospective-shadow-candidate":
            raise ValueError("unsupported or unfrozen shadow candidate")
        implementation = _require_mapping(self.manifest.get("release_implementation"), name="release implementation")
        current_implementation = {"shadow_sha256": file_sha256(Path(__file__)), "data_sha256": file_sha256(Path(data_module.__file__)), "model_sha256": file_sha256(Path(model_module.__file__))}
        if implementation != current_implementation:
            raise ValueError("shadow candidate release implementation hash mismatch")
        vocabulary_receipt = _require_mapping(self.manifest.get("vocabulary"), name="vocabulary receipt")
        vocabulary_path = self.release_dir / str(vocabulary_receipt["path"])
        if file_sha256(vocabulary_path) != vocabulary_receipt.get("sha256"):
            raise ValueError("shadow candidate vocabulary hash mismatch")
        vocabulary_payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if object_sha256(vocabulary_payload) != vocabulary_receipt.get("semantic_sha256"):
            raise ValueError("shadow candidate vocabulary semantic hash mismatch")
        self.vocabs = CorpusVocabs.from_receipt(vocabulary_payload)
        self.config = ModelConfig(**dict(_require_mapping(self.manifest.get("model"), name="model configuration")))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: list[tuple[int, HierarchicalSequenceTwin]] = []
        components = self.manifest.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("shadow candidate has fewer than 2 components")
        for component in components:
            receipt = _require_mapping(component, name="component receipt")
            path = self.release_dir / str(receipt["path"])
            if file_sha256(path) != receipt.get("sha256"):
                raise ValueError(f"shadow component hash mismatch: {path}")
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if payload.get("contract") != SHADOW_COMPONENT_CONTRACT or payload.get("candidate_id") != self.manifest.get("candidate_id"):
                raise ValueError(f"shadow component contract mismatch: {path}")
            if payload.get("model_config") != self.config.receipt():
                raise ValueError(f"shadow component configuration mismatch: {path}")
            model = HierarchicalSequenceTwin(self.vocabs, self.config).to(self.device)
            model.load_state_dict(payload["model"])
            model.eval()
            self.models.append((int(payload["seed"]), model))

    @property
    def candidate_id(self) -> str:
        return str(self.manifest["candidate_id"])

    @torch.inference_mode()
    def predict(self, samples: Sequence[Mapping[str, object]], *, family: str) -> list[dict[str, object]]:
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        if not samples:
            return []
        requested_rows = len(samples)
        padded_samples = list(samples)
        if self.device.type == "cuda" and len(padded_samples) < 8:
            padded_samples.extend([padded_samples[-1]] * (8 - len(padded_samples)))
        raw_batch = SequenceCollator(self.vocabs, family)(padded_samples)
        batch = move_batch(raw_batch, self.device)
        if not bool(batch["target_action_mask"].all()):
            raise ValueError("action-only shadow candidate received a non-action target")
        component_probabilities: list[torch.Tensor] = []
        component_gates: list[torch.Tensor] = []
        for _seed, model in self.models:
            outputs = model(batch, force_population=True)
            component_probabilities.append(F.softmax(outputs["action_logits"], dim=-1))
            component_gates.append(outputs["action_message_gate"])
        ensemble = torch.stack(component_probabilities).mean(dim=0)
        labels = self.vocabs.target_labels[family].values
        rows: list[dict[str, object]] = []
        for index, metadata in enumerate(batch["metadata"][:requested_rows]):
            probabilities = [float(value) for value in ensemble[index].cpu().tolist()]
            rows.append(
                {
                    "contract": SHADOW_CANDIDATE_CONTRACT,
                    "candidate_id": self.candidate_id,
                    "candidate_manifest_sha256": file_sha256(self.manifest_path),
                    "family": family,
                    "game_id": str(metadata["game_id"]),
                    "target_event_index": int(metadata["target_event_index"]),
                    "target_kind": str(metadata["target_kind"]),
                    "labels": list(labels),
                    "action_probabilities": probabilities,
                    "predicted_action": labels[max(range(len(probabilities)), key=probabilities.__getitem__)],
                    "component_probabilities": {str(seed): [float(value) for value in values[index].cpu().tolist()] for (seed, _model), values in zip(self.models, component_probabilities, strict=True)},
                    "action_message_gate": float(torch.stack(component_gates)[:, index].mean().cpu()),
                    "authority": "prospective-shadow-only",
                }
            )
        return rows

    def benchmark(self, samples: Sequence[Mapping[str, object]], *, family: str, warmup: int = 3, repetitions: int = 20) -> dict[str, object]:
        if warmup < 1 or repetitions < 1:
            raise ValueError("benchmark requires positive warmup and repetition counts")
        for _index in range(warmup):
            self.predict(samples, family=family)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        durations: list[float] = []
        for _index in range(repetitions):
            started = time.perf_counter()
            self.predict(samples, family=family)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            durations.append((time.perf_counter() - started) * 1_000)
        ordered = sorted(durations)
        return {
            "device": str(self.device),
            "samples_per_call": len(samples),
            "repetitions": repetitions,
            "minimum_ms": ordered[0],
            "median_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "maximum_ms": ordered[-1],
        }


def run_shadow_smoke(*, release_dir: Path, corpus_dir: Path, output_path: Path, warmup: int = 3, repetitions: int = 20) -> dict[str, object]:
    """Exercise loading and single-prefix inference without registering historical outcomes as prospective."""
    if output_path.exists():
        raise FileExistsError(f"shadow smoke output already exists: {output_path}")
    index = CorpusIndex([corpus_dir.resolve()])
    candidate = ShadowCandidate(release_dir)
    families: dict[str, object] = {}
    for family in GLEE_FAMILIES:
        dataset = index.subset(family=family, split="test", source_types={"real"})
        sample = next((dataset[index] for index in range(len(dataset)) if dataset[index]["target"]["target_kind"] != "proposal"), None)
        if sample is None:
            raise ValueError(f"test corpus has no action target for {family}")
        families[family] = {
            "prediction": candidate.predict([sample], family=family)[0],
            "latency": candidate.benchmark([sample], family=family, warmup=warmup, repetitions=repetitions),
        }
    result = {
        "contract": "glee-sequence-shadow-transport-smoke-v1",
        "status": "passed-retrospective-transport-only",
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_sha256": file_sha256(candidate.manifest_path),
        "corpus_manifest_sha256": file_sha256(corpus_dir.resolve() / "manifest.json"),
        "families": families,
        "prospective_predictions_registered": 0,
        "authority": "none",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    return result


class ProspectiveShadowRegistry:
    """SQLite WAL registry that never overwrites a prediction or an observed outcome."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                candidate_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_event_index INTEGER NOT NULL,
                target_kind TEXT NOT NULL,
                family TEXT NOT NULL,
                prefix_event_count INTEGER NOT NULL,
                prefix_sha256 TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                prediction_sha256 TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                PRIMARY KEY (candidate_id, game_id, target_event_index, target_kind)
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                candidate_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_event_index INTEGER NOT NULL,
                target_kind TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                outcome_sha256 TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                PRIMARY KEY (candidate_id, game_id, target_event_index, target_kind),
                FOREIGN KEY (candidate_id, game_id, target_event_index, target_kind) REFERENCES predictions(candidate_id, game_id, target_event_index, target_kind)
            );
            """
        )
        self.connection.execute("INSERT OR IGNORE INTO registry_metadata(key, value) VALUES (?, ?)", ("contract", SHADOW_REGISTRY_CONTRACT))
        contract = self.connection.execute("SELECT value FROM registry_metadata WHERE key = ?", ("contract",)).fetchone()
        if contract != (SHADOW_REGISTRY_CONTRACT,):
            raise ValueError("prospective shadow registry contract mismatch")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ProspectiveShadowRegistry":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def register(self, prediction: Mapping[str, object], *, prefix_event_count: int, prefix_sha256: str, registered_at: str | None = None) -> dict[str, object]:
        if prefix_event_count != prediction.get("target_event_index"):
            raise ValueError("prospective prediction must end immediately before its target event")
        if len(prefix_sha256) != 64 or any(character not in "0123456789abcdef" for character in prefix_sha256):
            raise ValueError("prefix SHA-256 must be lowercase hexadecimal")
        if any(str(key).startswith("actual_") or str(key).startswith("outcome") for key in prediction):
            raise ValueError("prospective prediction cannot contain an observed outcome")
        required = ("candidate_id", "game_id", "target_event_index", "target_kind", "family", "authority")
        if any(key not in prediction for key in required) or prediction.get("authority") != "prospective-shadow-only":
            raise ValueError("prediction is missing its prospective shadow identity or authority")
        payload = canonical_json(dict(prediction))
        payload_sha256 = object_sha256(dict(prediction))
        key = (str(prediction["candidate_id"]), str(prediction["game_id"]), int(prediction["target_event_index"]), str(prediction["target_kind"]))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT prefix_sha256, prediction_sha256 FROM predictions WHERE candidate_id = ? AND game_id = ? AND target_event_index = ? AND target_kind = ?", key).fetchone()
            if existing is not None:
                if existing != (prefix_sha256, payload_sha256):
                    raise ValueError("prospective prediction key already has different immutable evidence")
                self.connection.execute("COMMIT")
                return {"status": "already-registered", "prediction_sha256": payload_sha256, "key": key}
            self.connection.execute(
                "INSERT INTO predictions(candidate_id, game_id, target_event_index, target_kind, family, prefix_event_count, prefix_sha256, registered_at, prediction_sha256, prediction_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, str(prediction["family"]), prefix_event_count, prefix_sha256, registered_at or _utc_now(), payload_sha256, payload),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "registered", "prediction_sha256": payload_sha256, "key": key}

    def record_outcome(self, *, candidate_id: str, game_id: str, target_event_index: int, target_kind: str, outcome: Mapping[str, object], observed_at: str | None = None) -> dict[str, object]:
        key = (candidate_id, game_id, target_event_index, target_kind)
        payload = canonical_json(dict(outcome))
        payload_sha256 = object_sha256(dict(outcome))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.connection.execute("SELECT 1 FROM predictions WHERE candidate_id = ? AND game_id = ? AND target_event_index = ? AND target_kind = ?", key).fetchone() is None:
                raise ValueError("cannot mature an outcome without a preregistered prediction")
            existing = self.connection.execute("SELECT outcome_sha256 FROM outcomes WHERE candidate_id = ? AND game_id = ? AND target_event_index = ? AND target_kind = ?", key).fetchone()
            if existing is not None:
                if existing != (payload_sha256,):
                    raise ValueError("prospective outcome key already has a different immutable outcome")
                self.connection.execute("COMMIT")
                return {"status": "already-recorded", "outcome_sha256": payload_sha256, "key": key}
            self.connection.execute(
                "INSERT INTO outcomes(candidate_id, game_id, target_event_index, target_kind, observed_at, outcome_sha256, outcome_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (*key, observed_at or _utc_now(), payload_sha256, payload),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "recorded", "outcome_sha256": payload_sha256, "key": key}
