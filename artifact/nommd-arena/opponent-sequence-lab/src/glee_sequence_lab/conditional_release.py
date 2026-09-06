"""Portable behaviorally inert release for action-conditioned pre-Terra inference."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl
import torch
from torch.nn import functional as F

from . import conditional_experiment as experiment_module
from . import data as data_module
from . import model as model_module
from . import pre_terra_conditional_v2 as corpus_module
from . import shadow as shadow_module
from .conditional_experiment import CONDITIONAL_EXPERIMENT_CONTRACT, ConditionalFeatureModel, ConditionalTrainingConfig, MAX_RESPONSE_CLASSES
from .corpus import GLEE_FAMILIES, file_sha256, object_sha256
from .data import CorpusIndex
from .pre_terra_conditional_v2 import CONDITIONAL_CORPUS_CONTRACT, CONDITIONAL_TARGET_LABELS, FEATURE_DIMENSION, CandidateAction, CandidateActionProjector, merge_sparse_vectors
from .shadow import SHADOW_CANDIDATE_CONTRACT, ShadowCandidate


CONDITIONAL_RELEASE_CONTRACT = "glee-post-planner-action-conditional-release-v3"
CONDITIONAL_FEATURE_COMPONENT_CONTRACT = "glee-post-planner-action-conditional-feature-component-v3"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _link_file(source: Path, destination: Path) -> None:
    os.link(source, destination)
    if file_sha256(source) != file_sha256(destination):
        raise RuntimeError(f"hard-linked release artifact differs from source: {source}")


def _implementation_receipt() -> dict[str, str]:
    return {
        "conditional_release_sha256": file_sha256(Path(__file__)),
        "conditional_experiment_sha256": file_sha256(Path(experiment_module.__file__)),
        "conditional_corpus_sha256": file_sha256(Path(corpus_module.__file__)),
        "shadow_sha256": file_sha256(Path(shadow_module.__file__)),
        "data_sha256": file_sha256(Path(data_module.__file__)),
        "model_sha256": file_sha256(Path(model_module.__file__)),
    }


class ConditionalReleaseBuilder:
    """Seal the sequence ensemble, engineered ensemble, and validation-fitted stack without policy authority."""

    def __init__(self, *, experiment_dir: Path, sequence_release: Path, corpus_dir: Path, output_dir: Path, candidate_id: str) -> None:
        self.experiment_dir = experiment_dir.resolve()
        self.sequence_release = sequence_release.resolve()
        self.corpus_dir = corpus_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.candidate_id = candidate_id

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"conditional release output already exists: {self.output_dir}")
        if not self.candidate_id.strip():
            raise ValueError("conditional release candidate ID cannot be empty")
        result_path = self.experiment_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("contract") != CONDITIONAL_EXPERIMENT_CONTRACT or result.get("status") != "complete":
            raise ValueError("conditional release requires a completed conditional experiment")
        corpus_manifest_path = self.corpus_dir / "manifest.json"
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest.get("contract") != CONDITIONAL_CORPUS_CONTRACT or corpus_manifest.get("status") != "frozen-retrospective-core-corpus":
            raise ValueError("conditional release requires a frozen conditional corpus")
        if _require_mapping(result.get("corpus"), name="experiment corpus").get("manifest_sha256") != file_sha256(corpus_manifest_path):
            raise ValueError("conditional experiment and release corpus differ")
        sequence_manifest_path = self.sequence_release / "manifest.json"
        sequence_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
        if sequence_manifest.get("contract") != SHADOW_CANDIDATE_CONTRACT or sequence_manifest.get("status") != "frozen-prospective-shadow-candidate":
            raise ValueError("conditional release requires a frozen sequence ensemble")
        sequence_receipt = _require_mapping(result.get("sequence_release"), name="experiment sequence release")
        if sequence_receipt.get("manifest_sha256") != file_sha256(sequence_manifest_path):
            raise ValueError("conditional experiment and release sequence ensemble differ")
        configuration = dict(_require_mapping(result.get("configuration"), name="conditional configuration"))
        expected_seeds = [int(value) for value in configuration.get("seeds", [])]
        engineered = _require_mapping(result.get("engineered_models"), name="engineered models")
        if set(engineered) != {str(seed) for seed in expected_seeds}:
            raise ValueError("conditional experiment has an incomplete engineered ensemble")
        stack = _require_mapping(result.get("stack"), name="conditional stack")
        stack_weights = _require_mapping(stack.get("sequence_weights"), name="stack weights")
        if set(stack_weights) != set(GLEE_FAMILIES) or any(not 0.0 <= float(value) <= 1.0 for value in stack_weights.values()):
            raise ValueError("conditional stack weights are invalid")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            sequence_dir = staging / "sequence"
            sequence_dir.mkdir()
            sequence_files = [sequence_manifest_path, self.sequence_release / str(_require_mapping(sequence_manifest["vocabulary"], name="sequence vocabulary")["path"])]
            for component in sequence_manifest["components"]:
                sequence_files.append(self.sequence_release / str(_require_mapping(component, name="sequence component")["path"]))
            for source in sequence_files:
                _link_file(source, sequence_dir / source.name)
            feature_components: list[dict[str, object]] = []
            for seed in sorted(expected_seeds):
                receipt = _require_mapping(_require_mapping(engineered[str(seed)], name=f"engineered seed {seed}").get("checkpoint"), name=f"engineered checkpoint {seed}")
                source = self.experiment_dir / str(receipt["path"])
                if file_sha256(source) != receipt.get("sha256"):
                    raise RuntimeError(f"engineered checkpoint hash mismatch: {seed}")
                payload = torch.load(source, map_location="cpu", weights_only=False)
                target = staging / f"engineered-seed{seed}.pt"
                torch.save(
                    {
                        "contract": CONDITIONAL_FEATURE_COMPONENT_CONTRACT,
                        "candidate_id": self.candidate_id,
                        "seed": seed,
                        "config": configuration,
                        "model": payload["model"],
                        "feature_dimension": payload["feature_dimension"],
                        "target_labels": payload["target_labels"],
                    },
                    target,
                )
                feature_components.append({"seed": seed, "path": target.name, "sha256": file_sha256(target), "bytes": target.stat().st_size, "source_sha256": receipt["sha256"]})
            manifest = {
                "schema_version": 1,
                "contract": CONDITIONAL_RELEASE_CONTRACT,
                "candidate_id": self.candidate_id,
                "status": "frozen-behaviorally-inert-prospective-shadow",
                "scope": "direct opponent-response distributions conditional on bounded candidate DeepRMM-01 actions",
                "target_labels": {family: list(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES},
                "sequence": {"path": sequence_dir.name, "manifest_sha256": file_sha256(sequence_dir / "manifest.json"), "candidate_id": sequence_manifest["candidate_id"], "components": len(sequence_manifest["components"])},
                "engineered": {"configuration": configuration, "components": feature_components, "projection": corpus_manifest["candidate_action_projection"]},
                "stack": dict(stack),
                "corpus": {"path_at_freeze": str(self.corpus_dir), "manifest_sha256": file_sha256(corpus_manifest_path)},
                "experiment": {"path_at_freeze": str(self.experiment_dir), "result_sha256": file_sha256(result_path)},
                "implementation": _implementation_receipt(),
                "boundary": "Behaviorally inert prospective shadow only. This release cannot alter prompts, candidates, actions, messages, timing, matchmaking, identity claims, ratings, or deterministic safeguards.",
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
            return {**manifest, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json")}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


class ConditionalTwinRelease:
    """Fail-closed batched candidate-action predictor for the frozen conditional ensemble."""

    def __init__(self, release_dir: Path, *, device: str | None = None) -> None:
        self.release_dir = release_dir.resolve()
        self.manifest_path = self.release_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != CONDITIONAL_RELEASE_CONTRACT or self.manifest.get("status") != "frozen-behaviorally-inert-prospective-shadow":
            raise ValueError("unsupported or unfrozen conditional release")
        if self.manifest.get("target_labels") != {family: list(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES}:
            raise ValueError("conditional release target labels changed")
        if self.manifest.get("implementation") != _implementation_receipt():
            raise ValueError("conditional release implementation hash mismatch")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sequence = _require_mapping(self.manifest.get("sequence"), name="sequence receipt")
        sequence_dir = self.release_dir / str(sequence["path"])
        if file_sha256(sequence_dir / "manifest.json") != sequence.get("manifest_sha256"):
            raise ValueError("conditional sequence manifest hash mismatch")
        self.sequence = ShadowCandidate(sequence_dir, device=str(self.device))
        engineered = _require_mapping(self.manifest.get("engineered"), name="engineered receipt")
        self.config = ConditionalTrainingConfig(**dict(_require_mapping(engineered.get("configuration"), name="engineered configuration")))
        self.feature_models: list[tuple[int, ConditionalFeatureModel]] = []
        components = engineered.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("conditional release has fewer than 2 engineered components")
        for component in components:
            receipt = _require_mapping(component, name="engineered component")
            path = self.release_dir / str(receipt["path"])
            if file_sha256(path) != receipt.get("sha256"):
                raise ValueError(f"conditional engineered component hash mismatch: {path}")
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if payload.get("contract") != CONDITIONAL_FEATURE_COMPONENT_CONTRACT or payload.get("candidate_id") != self.manifest.get("candidate_id") or payload.get("config") != dict(_require_mapping(engineered.get("configuration"), name="engineered configuration")):
                raise ValueError(f"conditional engineered component contract mismatch: {path}")
            model = ConditionalFeatureModel(self.config).to(self.device)
            model.load_state_dict(payload["model"])
            model.eval()
            self.feature_models.append((int(payload["seed"]), model))
        stack = _require_mapping(self.manifest.get("stack"), name="stack")
        self.sequence_weights = {family: float(value) for family, value in _require_mapping(stack.get("sequence_weights"), name="sequence weights").items()}
        self.projector = CandidateActionProjector()

    @property
    def candidate_id(self) -> str:
        return str(self.manifest["candidate_id"])

    def prepare_candidates(self, *, sample: Mapping[str, object], family: str, phase: str, base_feature_indices: Sequence[int], base_feature_values: Sequence[float], base_feature_vector_sha256: str, candidates: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], torch.Tensor, list[CandidateAction]]:
        if family not in GLEE_FAMILIES or not candidates or len(candidates) > 32:
            raise ValueError("conditional inference requires 1 to 32 candidates from one supported family")
        game = _require_mapping(sample.get("game"), name="sample game")
        raw_events = sample.get("events")
        raw_target = _require_mapping(sample.get("target"), name="sample target")
        if not isinstance(raw_events, list) or str(game.get("family")) != family:
            raise ValueError("conditional sample has invalid events or family")
        base_indices = [int(value) for value in base_feature_indices]
        base_values = [float(value) for value in base_feature_values]
        if object_sha256({"indices": base_indices, "values": base_values}) != base_feature_vector_sha256:
            raise ValueError("conditional base feature vector hash mismatch")
        parsed = [CandidateAction.from_mapping(value, family=family, phase=phase) for value in candidates]
        receipts = [object_sha256(value.receipt()) for value in parsed]
        if len(set(receipts)) != len(receipts):
            raise ValueError("conditional candidates are not unique after legal normalization")
        samples: list[dict[str, object]] = []
        features = torch.zeros(len(parsed), FEATURE_DIMENSION, dtype=torch.float32)
        for index, candidate in enumerate(parsed):
            candidate_vector = self.projector.project(candidate)
            conditioned = merge_sparse_vectors(base_indices, base_values, candidate_vector)
            if conditioned.indices:
                features[index, torch.tensor(conditioned.indices, dtype=torch.long)] = torch.tensor(conditioned.values, dtype=torch.float32)
            events = [dict(event) for event in raw_events]
            events.append(candidate.event(game_id=str(game["game_id"]), event_index=len(events)))
            target = dict(raw_target)
            target.update(
                {
                    "sample_id": f"{raw_target.get('sample_id', game['game_id'])}:candidate:{receipts[index][:16]}",
                    "game_id": str(game["game_id"]),
                    "prefix_length": len(events),
                    "target_event_index": len(events),
                    "target_kind": "response",
                    "target_label": CONDITIONAL_TARGET_LABELS[family][0],
                    "target_value_present": False,
                    "target_delay_present": False,
                    "identity_scope": str(game.get("identity_scope") or raw_target.get("identity_scope") or "hidden"),
                    "source_type": "prospective-shadow",
                    "mask_last_prefix_future_fields": False,
                    "mask_last_prefix_response_time": True,
                    "mask_last_prefix_message_fields": False,
                }
            )
            samples.append({"game": dict(game), "events": events, "target": target})
        return samples, features, parsed

    @torch.inference_mode()
    def predict_candidates(self, *, sample: Mapping[str, object], family: str, phase: str, base_feature_indices: Sequence[int], base_feature_values: Sequence[float], base_feature_vector_sha256: str, candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        samples, features, parsed = self.prepare_candidates(sample=sample, family=family, phase=phase, base_feature_indices=base_feature_indices, base_feature_values=base_feature_values, base_feature_vector_sha256=base_feature_vector_sha256, candidates=candidates)
        sequence_rows = self.sequence.predict(samples, family=family)
        family_tensor = torch.full((len(parsed),), GLEE_FAMILIES.index(family), dtype=torch.long, device=self.device)
        feature_probabilities = []
        for _seed, model in self.feature_models:
            feature_probabilities.append(F.softmax(model(features.to(self.device), family_tensor), dim=-1))
        engineered = torch.stack(feature_probabilities).mean(dim=0).cpu()
        labels = list(CONDITIONAL_TARGET_LABELS[family])
        weight = self.sequence_weights[family]
        results: list[dict[str, object]] = []
        for index, (candidate, sequence_row) in enumerate(zip(parsed, sequence_rows, strict=True)):
            sequence_probabilities = torch.tensor(sequence_row["action_probabilities"], dtype=torch.float32)
            conditioned = weight * sequence_probabilities + (1.0 - weight) * engineered[index, : len(labels)]
            if not torch.isfinite(conditioned).all() or float(conditioned.min()) < 0.0 or not math.isclose(float(conditioned.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError("conditional response distribution is invalid")
            probabilities = [float(value) for value in conditioned.tolist()]
            results.append(
                {
                    "contract": CONDITIONAL_RELEASE_CONTRACT,
                    "candidate_id": self.candidate_id,
                    "candidate_manifest_sha256": file_sha256(self.manifest_path),
                    "family": family,
                    "game_id": str(_require_mapping(sample["game"], name="sample game")["game_id"]),
                    "candidate": candidate.public_receipt(),
                    "labels": labels,
                    "response_probabilities": probabilities,
                    "predicted_response": labels[max(range(len(labels)), key=probabilities.__getitem__)],
                    "sequence_probabilities": [float(value) for value in sequence_probabilities.tolist()],
                    "engineered_probabilities": [float(value) for value in engineered[index, : len(labels)].tolist()],
                    "sequence_weight": weight,
                    "authority": "prospective-shadow-only",
                }
            )
        return results


def run_conditional_smoke(*, release_dir: Path, corpus_dir: Path, output_path: Path) -> dict[str, object]:
    """Exercise historical candidate substitution without registering a prospective prediction."""

    release = ConditionalTwinRelease(release_dir)
    index = CorpusIndex([corpus_dir.resolve()])
    feature_rows = {str(row["sample_id"]): row for row in pl.read_parquet(corpus_dir.resolve() / "features.parquet").to_dicts()}
    results: list[dict[str, object]] = []
    for family in GLEE_FAMILIES:
        target = next(value for value in index.targets if index.games[str(value["game_id"])]["family"] == family)
        feature = feature_rows[str(target["sample_id"])]
        prefix_length = int(target["pre_candidate_prefix_length"])
        sample = {"game": index.games[str(target["game_id"])], "events": index.events[str(target["game_id"])][:prefix_length], "target": target}
        actual = CandidateAction.from_feature_row(feature, family=family, phase=str(feature["phase"])).receipt()
        alternative = dict(actual)
        if family == "persuasion":
            alternative["action_label"] = "signal_negative" if actual["action_label"] == "signal_positive" else "signal_positive"
            alternative["action_value"] = -float(actual["action_value"])
        else:
            alternative["action_value"] = float(actual["action_value"]) + 0.05
            if actual["action_aux_value"] is not None:
                alternative["action_aux_value"] = float(actual["action_aux_value"]) - 0.05
        before = object_sha256(sample)
        started = time.perf_counter()
        predictions = release.predict_candidates(
            sample=sample,
            family=family,
            phase=str(feature["phase"]),
            base_feature_indices=feature["feature_indices"],
            base_feature_values=feature["feature_values"],
            base_feature_vector_sha256=str(feature["feature_vector_sha256"]),
            candidates=[actual, alternative],
        )
        elapsed = time.perf_counter() - started
        if object_sha256(sample) != before:
            raise RuntimeError("conditional candidate inference mutated the authenticated sample")
        results.append({"family": family, "sample_id": target["sample_id"], "elapsed_seconds": elapsed, "predictions": predictions})
    receipt = {
        "schema_version": 1,
        "contract": "glee-post-planner-action-conditional-smoke-v2",
        "status": "passed",
        "release": {"path": str(release.release_dir), "manifest_sha256": file_sha256(release.manifest_path)},
        "corpus": {"path": str(corpus_dir.resolve()), "manifest_sha256": file_sha256(corpus_dir.resolve() / "manifest.json")},
        "families": results,
        "boundary": "Historical transport and substitution smoke only; no prospective registry write and no policy authority.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    _write_json(temporary, receipt)
    os.replace(temporary, output_path)
    return receipt
