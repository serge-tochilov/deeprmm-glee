"""Frozen public self-mirror ensembles with joint categorical and proposal scoring."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from . import data as data_module
from . import model as model_module
from .corpus import GLEE_FAMILIES, file_sha256, object_sha256
from .data import CorpusIndex, CorpusVocabs, SequenceCollator, move_batch
from .model import HierarchicalSequenceTwin, ModelConfig
from .self_mirror import SELF_MIRROR_CORPUS_CONTRACT


SELF_MIRROR_RELEASE_CONTRACT = "glee-public-self-mirror-release-v1"
SELF_MIRROR_COMPONENT_CONTRACT = "glee-public-self-mirror-component-v1"
SELF_MIRROR_AUTHORITY = "bounded-public-expectedness-selector-evidence-only"
SELF_MIRROR_RECURSIVE_MODEL = {
    "rmm_depth": "order-2",
    "tsr_awareness_tier": "population-level self-awareness",
    "referent": "how a generic bounded outside observer can model DeepRMM from its public trajectory",
    "excluded_tier": "opponent-specific peer-awareness",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _completed_component(directory: Path) -> tuple[dict[str, object], Path]:
    resolved = directory.resolve()
    result_path = resolved / "result.json"
    checkpoint_path = resolved / "best.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    training = _mapping(result.get("training"), name="training")
    model = _mapping(result.get("model"), name="model")
    if result.get("contract") != "glee-hierarchical-sequence-twin-experiment-v3" or result.get("status") != "complete":
        raise ValueError(f"self-mirror component is not a completed v3 experiment: {resolved}")
    if result.get("arm") != "hierarchical-mamba2" or model.get("core") != "mamba2":
        raise ValueError(f"self-mirror component must be a hierarchical Mamba-2 arm: {resolved}")
    if training.get("selection_objective") != "joint-self-mirror":
        raise ValueError(f"self-mirror component was not selected on the joint objective: {resolved}")
    if not isinstance(training.get("seed"), int) or isinstance(training.get("seed"), bool):
        raise ValueError(f"self-mirror component has no integer seed: {resolved}")
    if not checkpoint_path.is_file():
        raise ValueError(f"self-mirror checkpoint is absent: {checkpoint_path}")
    return result, checkpoint_path


class PublicSelfMirrorReleaseBuilder:
    """Seal independently seeded public-information self-mirror arms into one portable ensemble."""

    def __init__(self, *, experiment_dirs: Sequence[Path], output_dir: Path, release_id: str) -> None:
        self.experiment_dirs = tuple(path.resolve() for path in experiment_dirs)
        self.output_dir = output_dir.resolve()
        self.release_id = release_id.strip()

    def run(self) -> dict[str, object]:
        if len(self.experiment_dirs) < 2:
            raise ValueError("a public self-mirror release requires at least 2 independently seeded components")
        if not self.release_id:
            raise ValueError("self-mirror release ID cannot be empty")
        if self.output_dir.exists():
            raise FileExistsError(f"self-mirror release already exists: {self.output_dir}")
        loaded = [_completed_component(directory) for directory in self.experiment_dirs]
        results = [item[0] for item in loaded]
        reference = results[0]
        model_config = dict(_mapping(reference.get("model"), name="model"))
        corpora = reference.get("corpora")
        vocabulary_sha256 = reference.get("vocabulary_sha256")
        implementation = reference.get("implementation")
        seeds = [int(_mapping(result.get("training"), name="training")["seed"]) for result in results]
        if len(set(seeds)) != len(seeds):
            raise ValueError("self-mirror component seeds must be distinct")
        for result in results[1:]:
            if dict(_mapping(result.get("model"), name="model")) != model_config:
                raise ValueError("self-mirror components use different model configurations")
            if result.get("corpora") != corpora or result.get("vocabulary_sha256") != vocabulary_sha256:
                raise ValueError("self-mirror components use different corpora or vocabularies")
            if result.get("implementation") != implementation:
                raise ValueError("self-mirror components use different training implementations")
        if not isinstance(corpora, list) or len(corpora) != 1 or not isinstance(corpora[0], Mapping):
            raise ValueError("self-mirror release requires exactly one frozen corpus")
        corpus_dir = Path(str(corpora[0]["path"])).resolve()
        corpus_manifest_path = corpus_dir / "manifest.json"
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest.get("contract") != SELF_MIRROR_CORPUS_CONTRACT:
            raise ValueError("self-mirror experiment does not reference a public self-mirror corpus")
        index = CorpusIndex([corpus_dir])
        vocabulary = index.vocabs.receipt()
        if object_sha256(vocabulary) != vocabulary_sha256:
            raise ValueError("reconstructed self-mirror vocabulary does not match training")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            vocabulary_path = staging / "vocabulary.json"
            _write_json(vocabulary_path, vocabulary)
            components: list[dict[str, object]] = []
            for directory, (result, checkpoint_path) in sorted(zip(self.experiment_dirs, loaded, strict=True), key=lambda item: int(_mapping(item[1][0].get("training"), name="training")["seed"])):
                seed = int(_mapping(result.get("training"), name="training")["seed"])
                source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                component_name = f"component-seed{seed}.pt"
                component_path = staging / component_name
                torch.save(
                    {
                        "contract": SELF_MIRROR_COMPONENT_CONTRACT,
                        "release_id": self.release_id,
                        "seed": seed,
                        "model_config": model_config,
                        "model": source["model"],
                        "source_epoch": source.get("epoch"),
                        "source_score": source.get("score"),
                    },
                    component_path,
                )
                component_path.chmod(0o644)
                test = _mapping(_mapping(result.get("evaluation"), name="evaluation").get("test"), name="test evaluation")
                components.append(
                    {
                        "seed": seed,
                        "path": component_name,
                        "sha256": file_sha256(component_path),
                        "bytes": component_path.stat().st_size,
                        "source_experiment": str(directory),
                        "source_result_sha256": file_sha256(directory / "result.json"),
                        "source_checkpoint_sha256": file_sha256(checkpoint_path),
                        "source_epoch": result.get("best_epoch"),
                        "source_validation_score": result.get("best_validation_selection_score"),
                        "test": dict(_mapping(test.get("all"), name="all test metrics")),
                    }
                )
            manifest = {
                "schema_version": 1,
                "contract": SELF_MIRROR_RELEASE_CONTRACT,
                "release_id": self.release_id,
                "status": "frozen-public-self-mirror",
                "frozen_at": _utc_now(),
                "authority": SELF_MIRROR_AUTHORITY,
                "scope": "public-information prediction of DeepRMM's own next categorical action or public proposal coordinate",
                "recursive_model": dict(SELF_MIRROR_RECURSIVE_MODEL),
                "ensemble": {
                    "components": len(components),
                    "component_seeds": sorted(seeds),
                    "rule": "unweighted log-mean-exp of component action log probability or Gaussian public-coordinate log density",
                    "minimum_cuda_batch": 8,
                    "account_path": "population inference is authoritative; account-conditioned outputs are not exposed live",
                },
                "model": model_config,
                "model_parameters_per_component": reference.get("model_parameters"),
                "components": components,
                "vocabulary": {"path": vocabulary_path.name, "sha256": file_sha256(vocabulary_path), "semantic_sha256": vocabulary_sha256},
                "corpus": {"path_at_freeze": str(corpus_dir), "manifest_sha256": file_sha256(corpus_manifest_path), "inventory": corpus_manifest.get("inventory")},
                "training_implementation": implementation,
                "release_implementation": {
                    "self_mirror_release_sha256": file_sha256(Path(__file__)),
                    "self_mirror_sha256": file_sha256(Path(__file__).with_name("self_mirror.py")),
                    "self_mirror_live_sha256": file_sha256(Path(__file__).with_name("self_mirror_live.py")),
                    "data_sha256": file_sha256(Path(data_module.__file__)),
                    "model_sha256": file_sha256(Path(model_module.__file__)),
                },
                "boundary": "The release receives opponent-observable game state and history only. It cannot generate candidates, alter the planner, reconstruct exact message text, claim hidden identity, or override legality, arithmetic, terminal, dominant-action, and catastrophic-loss controls.",
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
            return {**manifest, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json")}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


class PublicSelfMirrorRelease:
    """Load a frozen public self-mirror and expose per-component predictive distributions."""

    def __init__(self, release_dir: Path, *, device: str | None = None) -> None:
        self.release_dir = release_dir.resolve()
        self.manifest_path = self.release_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != SELF_MIRROR_RELEASE_CONTRACT or self.manifest.get("status") != "frozen-public-self-mirror" or self.manifest.get("authority") != SELF_MIRROR_AUTHORITY:
            raise ValueError("unsupported or unfrozen public self-mirror release")
        if self.manifest.get("recursive_model") != SELF_MIRROR_RECURSIVE_MODEL:
            raise ValueError("public self-mirror recursive-model contract mismatch")
        expected_implementation = _mapping(self.manifest.get("release_implementation"), name="release implementation")
        current_implementation = {
            "self_mirror_release_sha256": file_sha256(Path(__file__)),
            "self_mirror_sha256": file_sha256(Path(__file__).with_name("self_mirror.py")),
            "self_mirror_live_sha256": file_sha256(Path(__file__).with_name("self_mirror_live.py")),
            "data_sha256": file_sha256(Path(data_module.__file__)),
            "model_sha256": file_sha256(Path(model_module.__file__)),
        }
        if dict(expected_implementation) != current_implementation:
            raise ValueError("public self-mirror release implementation hash mismatch")
        vocabulary_receipt = _mapping(self.manifest.get("vocabulary"), name="vocabulary")
        vocabulary_path = self.release_dir / str(vocabulary_receipt["path"])
        if file_sha256(vocabulary_path) != vocabulary_receipt.get("sha256"):
            raise ValueError("public self-mirror vocabulary hash mismatch")
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if object_sha256(vocabulary) != vocabulary_receipt.get("semantic_sha256"):
            raise ValueError("public self-mirror vocabulary semantic hash mismatch")
        self.vocabs = CorpusVocabs.from_receipt(vocabulary)
        self.config = ModelConfig(**dict(_mapping(self.manifest.get("model"), name="model")))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: list[tuple[int, HierarchicalSequenceTwin]] = []
        components = self.manifest.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("public self-mirror release has fewer than 2 components")
        for raw in components:
            receipt = _mapping(raw, name="component")
            path = self.release_dir / str(receipt["path"])
            if file_sha256(path) != receipt.get("sha256"):
                raise ValueError(f"public self-mirror component hash mismatch: {path}")
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if payload.get("contract") != SELF_MIRROR_COMPONENT_CONTRACT or payload.get("release_id") != self.manifest.get("release_id") or payload.get("model_config") != self.config.receipt():
                raise ValueError(f"public self-mirror component contract mismatch: {path}")
            model = HierarchicalSequenceTwin(self.vocabs, self.config).to(self.device)
            model.load_state_dict(payload["model"])
            model.eval()
            self.models.append((int(payload["seed"]), model))

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @torch.inference_mode()
    def predict_components(self, samples: Sequence[Mapping[str, object]], *, family: str) -> list[dict[str, object]]:
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported self-mirror family: {family}")
        if not samples:
            return []
        requested = len(samples)
        padded = list(samples)
        if self.device.type == "cuda" and len(padded) < 8:
            padded.extend([padded[-1]] * (8 - len(padded)))
        batch = move_batch(SequenceCollator(self.vocabs, family)(padded), self.device)
        component_outputs: list[tuple[int, Mapping[str, torch.Tensor]]] = []
        for seed, model in self.models:
            component_outputs.append((seed, model(batch, force_population=True)))
        labels = list(self.vocabs.target_labels[family].values)
        rows: list[dict[str, object]] = []
        for index, metadata in enumerate(batch["metadata"][:requested]):
            components: list[dict[str, object]] = []
            for seed, output in component_outputs:
                probabilities = F.softmax(output["action_logits"][index], dim=-1)
                components.append(
                    {
                        "seed": seed,
                        "action_probabilities": [float(value) for value in probabilities.cpu().tolist()],
                        "value_location": float(output["value_location"][index].cpu()),
                        "value_log_scale": float(output["value_log_scale"][index].cpu()),
                        "delay_location": float(output["delay_location"][index].cpu()),
                        "delay_log_scale": float(output["delay_log_scale"][index].cpu()),
                    }
                )
            rows.append(
                {
                    "contract": SELF_MIRROR_RELEASE_CONTRACT,
                    "release_id": self.release_id,
                    "release_manifest_sha256": file_sha256(self.manifest_path),
                    "family": family,
                    "game_id": str(metadata["game_id"]),
                    "target_event_index": int(metadata["target_event_index"]),
                    "target_kind": str(metadata["target_kind"]),
                    "labels": labels,
                    "components": components,
                    "authority": SELF_MIRROR_AUTHORITY,
                }
            )
        return rows

    def benchmark(self, samples: Sequence[Mapping[str, object]], *, family: str, warmup: int = 2, repetitions: int = 10) -> dict[str, object]:
        if warmup < 1 or repetitions < 1:
            raise ValueError("self-mirror benchmark counts must be positive")
        for _index in range(warmup):
            self.predict_components(samples, family=family)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        durations: list[float] = []
        for _index in range(repetitions):
            started = time.perf_counter()
            self.predict_components(samples, family=family)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            durations.append((time.perf_counter() - started) * 1_000.0)
        ordered = sorted(durations)
        return {"device": str(self.device), "samples_per_call": len(samples), "repetitions": repetitions, "minimum_ms": ordered[0], "median_ms": ordered[len(ordered) // 2], "maximum_ms": ordered[-1]}
