"""Train a buyer-action-to-next-seller-signal Persuasion head on frozen sequence backbones."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .corpus import EVENT_SCHEMA, GAME_SCHEMA, TARGET_SCHEMA, _artifact, _static_game, extract_events, file_sha256, load_account_map, object_sha256
from .data import CorpusVocabs, SequenceCollator, Vocabulary, move_batch
from .model import BoundedMessageFusion, HierarchicalSequenceTwin, ModelConfig, _compact_message_sequence
from .shadow import SHADOW_CANDIDATE_CONTRACT, SHADOW_COMPONENT_CONTRACT


BUYER_CONTINUATION_CORPUS_CONTRACT = "glee-persuasion-buyer-continuation-corpus-v1"
BUYER_CONTINUATION_EXPERIMENT_CONTRACT = "glee-persuasion-buyer-continuation-experiment-v1"
BUYER_CONTINUATION_RELEASE_CONTRACT = "glee-persuasion-buyer-continuation-release-v1"
BUYER_CONTINUATION_LABELS = ("signal_positive", "signal_negative", "signal_unknown")
BUYER_ACTIONS = ("buy", "pass")
TARGET_SCHEMA_V1 = {
    **TARGET_SCHEMA,
    "mask_last_prefix_response_time": pl.Boolean,
    "buyer_action": pl.String,
    "preceding_signal": pl.String,
    "source_agent": pl.String,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _distribution(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    labels = Counter(str(row["target_label"]) for row in rows)
    actions = Counter(str(row["buyer_action"]) for row in rows)
    transitions = Counter(f"{row['preceding_signal']}|{row['buyer_action']}->{row['target_label']}" for row in rows)
    return {
        "rows": len(rows),
        "games": len({str(row["game_id"]) for row in rows}),
        "labels": dict(sorted(labels.items())),
        "buyer_actions": dict(sorted(actions.items())),
        "transitions": dict(sorted(transitions.items())),
    }


def derive_buyer_continuation_targets(game: Mapping[str, object], events: Sequence[Mapping[str, object]], *, source_type: str, source_agent: str) -> tuple[list[dict[str, object]], Counter[str]]:
    """Append an observed buyer action and predict the opponent seller's next-round signal without exposing the post-action quality reveal."""
    exclusions: Counter[str] = Counter()
    if game.get("family") != "persuasion" or game.get("our_role") != "buyer":
        exclusions["not-persuasion-buyer"] += 1
        return [], exclusions
    rows: list[dict[str, object]] = []
    for index, event in enumerate(events):
        if event.get("actor") != "self" or event.get("kind") != "response" or event.get("action_label") not in BUYER_ACTIONS:
            continue
        round_number = int(event["round_number"])
        preceding = next((prior for prior in reversed(events[:index]) if prior.get("round_number") == round_number and prior.get("actor") == "opponent" and prior.get("kind") == "signal"), None)
        if preceding is None:
            exclusions["response-without-preceding-signal"] += 1
            continue
        target_index = index + 1
        while target_index < len(events) and events[target_index].get("actor") == "environment":
            target_index += 1
        if target_index >= len(events):
            exclusions["terminal-response"] += 1
            continue
        target = events[target_index]
        if target.get("actor") != "opponent" or target.get("kind") != "signal" or int(target.get("round_number") or -1) != round_number + 1:
            exclusions["noncanonical-next-event"] += 1
            continue
        label = str(target.get("action_label") or "")
        if label not in BUYER_CONTINUATION_LABELS:
            exclusions["unsupported-next-signal"] += 1
            continue
        sample_id = hashlib.sha256(f"buyer-continuation:{game['game_id']}:{index}:{target_index}".encode("utf-8")).hexdigest()
        raw_fold = game.get("account_fold")
        rows.append(
            {
                "sample_id": sample_id,
                "game_id": str(game["game_id"]),
                "source_type": source_type,
                "target_event_index": target_index,
                "prefix_length": index + 1,
                "target_kind": "signal",
                "target_label": label,
                "target_value": None,
                "target_value_present": False,
                "target_message_act": None,
                "target_message_present": False,
                "target_delay_log_ms": None,
                "target_delay_present": False,
                "chronological_split": str(game["chronological_split"]),
                "identity_scope": str(game["identity_scope"]),
                "account_key": game.get("account_key"),
                "account_confidence": game.get("account_confidence"),
                "account_fold": int(raw_fold) if isinstance(raw_fold, int) and not isinstance(raw_fold, bool) else -1,
                "mask_last_prefix_response_time": True,
                "buyer_action": str(event["action_label"]),
                "preceding_signal": str(preceding["action_label"]),
                "source_agent": source_agent,
            }
        )
    return rows, exclusions


class PersuasionBuyerContinuationCorpusBuilder:
    """Derive DeepRMM targets and a separately identified train-only Fieldglass augmentation set."""

    def __init__(self, *, source_corpus: Path, output_dir: Path, account_groups: Path, fieldglass_roots: Sequence[Path] = ()) -> None:
        self.source_corpus = source_corpus.resolve()
        self.output_dir = output_dir.resolve()
        self.account_groups = account_groups.resolve()
        self.fieldglass_roots = tuple(path.resolve() for path in fieldglass_roots)

    def _deep_rows(self) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], Counter[str]]:
        games_frame = pl.read_parquet(self.source_corpus / "games.parquet").filter((pl.col("family") == "persuasion") & (pl.col("our_role") == "buyer"))
        selected_ids = games_frame["game_id"].to_list()
        events_frame = pl.read_parquet(self.source_corpus / "events.parquet").filter(pl.col("game_id").is_in(selected_ids)).sort(["game_id", "event_index"])
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for event in events_frame.to_dicts():
            grouped[str(event["game_id"])].append(event)
        targets: list[dict[str, object]] = []
        exclusions: Counter[str] = Counter()
        games = games_frame.to_dicts()
        for game in games:
            rows, omitted = derive_buyer_continuation_targets(game, grouped[str(game["game_id"])], source_type="deeprmm", source_agent="DeepRMM-01")
            targets.extend(rows)
            exclusions.update(omitted)
            game["source_type"] = "deeprmm"
            game["generator_id"] = "DeepRMM-01"
        return games, events_frame.to_dicts(), targets, exclusions

    def _fieldglass_rows(self) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], Counter[str], list[dict[str, object]]]:
        account_map, _collisions = load_account_map(self.account_groups)
        selected: dict[str, tuple[Path, Mapping[str, Any], str]] = {}
        exclusions: Counter[str] = Counter()
        receipts: list[dict[str, object]] = []
        for root in self.fieldglass_roots:
            if not root.is_dir():
                exclusions["fieldglass-root-missing"] += 1
                continue
            for path in sorted(root.rglob("games/persuasion-*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    exclusions["fieldglass-unreadable"] += 1
                    continue
                if not isinstance(payload, Mapping) or payload.get("game_family") != "persuasion" or payload.get("status") != "completed":
                    exclusions["fieldglass-not-completed-persuasion"] += 1
                    continue
                game_id = str(payload.get("game_id") or "")
                digest = object_sha256(payload)
                if not game_id:
                    exclusions["fieldglass-missing-game-id"] += 1
                    continue
                existing = selected.get(game_id)
                if existing is not None:
                    if existing[2] != digest:
                        raise ValueError(f"Fieldglass game has conflicting terminal archives: {game_id}")
                    exclusions["fieldglass-duplicate-archive"] += 1
                    continue
                selected[game_id] = (path, payload, digest)
        games: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        targets: list[dict[str, object]] = []
        for game_id, (path, payload, digest) in sorted(selected.items()):
            try:
                extracted = extract_events(payload)
            except (KeyError, TypeError, ValueError, OverflowError):
                exclusions["fieldglass-feature-extraction-invalid"] += 1
                continue
            state = payload.get("game_state")
            our_player = str(payload.get("your_player") or "")
            if not isinstance(state, Mapping) or state.get(f"{our_player}_role") != "buyer":
                exclusions["fieldglass-not-buyer"] += 1
                continue
            opponent = payload.get("opponent")
            opponent_name = str(opponent.get("name") or "") if isinstance(opponent, Mapping) else ""
            identity_scope = "known" if opponent_name and opponent_name.casefold() not in {"anonymous", "hidden", "unknown"} else "hidden"
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            source = {
                "game_id": game_id,
                "family": "persuasion",
                "identity_scope": identity_scope,
                "opponent_name": opponent_name,
                "started_at": timestamp,
                "completed_at": timestamp,
                "engine_version": f"fieldglass:{path.parents[1].name}",
                "advisor_version": "local-collector",
                "policy_revision": path.parents[1].name,
                "archive_path": str(path),
                "archive_sha256": digest,
            }
            game = _static_game(source, payload, split="train", account_map=account_map)
            game["source_type"] = "fieldglass"
            game["generator_id"] = "fieldglass"
            rows, omitted = derive_buyer_continuation_targets(game, extracted, source_type="fieldglass", source_agent="fieldglass")
            exclusions.update(omitted)
            if not rows:
                exclusions["fieldglass-no-continuation-target"] += 1
                continue
            games.append(game)
            events.extend(extracted)
            targets.extend(rows)
            receipts.append({"game_id": game_id, "path": str(path), "sha256": file_sha256(path), "semantic_sha256": digest})
        return games, events, targets, exclusions, receipts

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"buyer-continuation corpus output already exists: {self.output_dir}")
        source_manifest = json.loads((self.source_corpus / "manifest.json").read_text(encoding="utf-8"))
        deep_games, deep_events, deep_targets, deep_exclusions = self._deep_rows()
        fieldglass_games, fieldglass_events, fieldglass_targets, fieldglass_exclusions, fieldglass_receipts = self._fieldglass_rows()
        games = pl.DataFrame([*deep_games, *fieldglass_games], schema=GAME_SCHEMA, strict=False).sort(["completed_at", "game_id"])
        events = pl.DataFrame([*deep_events, *fieldglass_events], schema=EVENT_SCHEMA, strict=False).sort(["game_id", "event_index"])
        targets = pl.DataFrame([*deep_targets, *fieldglass_targets], schema=TARGET_SCHEMA_V1, strict=False).sort(["game_id", "target_event_index"])
        if games["game_id"].n_unique() != games.height:
            raise RuntimeError("buyer-continuation corpus contains duplicate games")
        if events.select(pl.struct("game_id", "event_index").n_unique()).item(0, 0) != events.height:
            raise RuntimeError("buyer-continuation corpus contains duplicate event coordinates")
        if targets["sample_id"].n_unique() != targets.height:
            raise RuntimeError("buyer-continuation corpus contains duplicate targets")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            artifacts = {
                "games.parquet": _artifact(games, staging / "games.parquet"),
                "events.parquet": _artifact(events, staging / "events.parquet"),
                "targets.parquet": _artifact(targets, staging / "targets.parquet"),
            }
            inventory = {
                "games": games.height,
                "events": events.height,
                "targets": targets.height,
                "deeprmm": _distribution(deep_targets),
                "fieldglass": _distribution(fieldglass_targets),
                "deeprmm_by_split": {split: _distribution([row for row in deep_targets if row["chronological_split"] == split]) for split in ("train", "validation", "test")},
                "exclusions": {"deeprmm": dict(sorted(deep_exclusions.items())), "fieldglass": dict(sorted(fieldglass_exclusions.items()))},
            }
            manifest = {
                "schema_version": 1,
                "contract": BUYER_CONTINUATION_CORPUS_CONTRACT,
                "status": "frozen-retrospective-causal-corpus",
                "created_at": _utc_now(),
                "causal_contract": "The prefix ends with the observed buyer buy/pass action; its response time is masked; any quality reveal caused by buying is excluded; the target is the opponent seller's signal in the next round.",
                "target_labels": list(BUYER_CONTINUATION_LABELS),
                "source": {
                    "deeprmm_corpus": str(self.source_corpus),
                    "deeprmm_manifest_sha256": file_sha256(self.source_corpus / "manifest.json"),
                    "deeprmm_selection": source_manifest.get("source", {}).get("selection"),
                    "fieldglass_roots": [str(path) for path in self.fieldglass_roots],
                    "fieldglass_archives": fieldglass_receipts,
                    "fieldglass_boundary": "Fieldglass rows are train-only augmentation candidates; DeepRMM validation and test rows remain the sole selection and evaluation domains.",
                    "account_groups": str(self.account_groups),
                    "account_groups_sha256": file_sha256(self.account_groups),
                },
                "inventory": inventory,
                "artifacts": artifacts,
                "implementation_sha256": file_sha256(Path(__file__)),
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {"contract": BUYER_CONTINUATION_CORPUS_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json"), "inventory": inventory}


class _ContinuationDataset(Dataset[dict[str, object]]):
    def __init__(self, *, games: Mapping[str, Mapping[str, object]], events: Mapping[str, Sequence[Mapping[str, object]]], targets: Sequence[Mapping[str, object]]) -> None:
        self.games = games
        self.events = events
        self.targets = tuple(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        game_id = str(target["game_id"])
        return {"game": self.games[game_id], "events": list(self.events[game_id][: int(target["prefix_length"])]), "target": target}


class _ContinuationIndex:
    def __init__(self, corpus_dir: Path, vocabs: CorpusVocabs) -> None:
        self.corpus_dir = corpus_dir.resolve()
        manifest = json.loads((self.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("contract") != BUYER_CONTINUATION_CORPUS_CONTRACT:
            raise ValueError("unsupported buyer-continuation corpus")
        if tuple(manifest.get("target_labels") or ()) != BUYER_CONTINUATION_LABELS:
            raise ValueError("buyer-continuation target labels differ from the frozen contract")
        games_frame = pl.read_parquet(self.corpus_dir / "games.parquet")
        events_frame = pl.read_parquet(self.corpus_dir / "events.parquet").sort(["game_id", "event_index"])
        targets_frame = pl.read_parquet(self.corpus_dir / "targets.parquet").sort(["game_id", "target_event_index"])
        self.games = {str(row["game_id"]): row for row in games_frame.to_dicts()}
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in events_frame.to_dicts():
            grouped[str(row["game_id"])].append(row)
        self.events = grouped
        self.targets = targets_frame.to_dicts()
        self.vocabs = vocabs

    def dataset(self, *, split: str, source_types: set[str]) -> _ContinuationDataset:
        targets = [row for row in self.targets if str(row["chronological_split"]) == split and str(row["source_type"]) in source_types]
        return _ContinuationDataset(games=self.games, events=self.events, targets=targets)


@dataclass(frozen=True)
class BuyerContinuationTrainingConfig:
    epochs: int = 24
    patience: int = 4
    batch_size: int = 256
    evaluation_batch_size: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    bootstrap_replicates: int = 2_000
    mixed_precision: bool = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_loader(dataset: _ContinuationDataset, *, vocabs: CorpusVocabs, batch_size: int, shuffle: bool, seed: int, minimum_batch: int) -> DataLoader:
    collator = SequenceCollator(vocabs, "persuasion")

    def collate(samples: list[dict[str, object]]) -> dict[str, object]:
        valid_rows = len(samples)
        if samples and len(samples) < minimum_batch:
            samples.extend([samples[-1]] * (minimum_batch - len(samples)))
        batch = collator(samples)
        batch["valid_rows"] = valid_rows
        return batch

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0, generator=torch.Generator().manual_seed(seed))


def _reset_reversed_head(model: HierarchicalSequenceTwin, config: ModelConfig, *, seed: int) -> list[nn.Parameter]:
    _set_seed(seed)
    head = model.heads["persuasion"]
    if not hasattr(head, "action_fusion"):
        raise ValueError("reversed Persuasion head requires the separate-head-gated backbone")
    head.action_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
    head.action = nn.Linear(config.hidden_dim, len(BUYER_CONTINUATION_LABELS))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [*head.action_fusion.parameters(), *head.action.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _rows_metrics(rows: Sequence[Mapping[str, object]], *, bins: int = 10) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "games": 0, "negative_log_likelihood": None, "brier_score": None, "accuracy": None, "balanced_accuracy": None, "game_macro_negative_log_likelihood": None, "expected_calibration_error": None}
    losses: defaultdict[str, list[float]] = defaultdict(list)
    recalls: defaultdict[int, list[int]] = defaultdict(list)
    nll = 0.0
    brier = 0.0
    correct = 0
    calibration = [0.0] * bins
    calibration_accuracy = [0.0] * bins
    calibration_count = [0] * bins
    confusion = [[0 for _predicted in BUYER_CONTINUATION_LABELS] for _actual in BUYER_CONTINUATION_LABELS]
    predicted_probability = [0.0 for _label in BUYER_CONTINUATION_LABELS]
    for row in rows:
        probabilities = [float(value) for value in row["probabilities"]]
        actual = int(row["actual"])
        chosen = max(range(len(probabilities)), key=probabilities.__getitem__)
        loss = -math.log(max(probabilities[actual], 1e-12))
        nll += loss
        brier += sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities))
        correct += int(chosen == actual)
        recalls[actual].append(int(chosen == actual))
        confusion[actual][chosen] += 1
        for index, probability in enumerate(probabilities):
            predicted_probability[index] += probability
        losses[str(row["game_id"])].append(loss)
        confidence = probabilities[chosen]
        bucket = min(bins - 1, int(confidence * bins))
        calibration[bucket] += confidence
        calibration_accuracy[bucket] += int(chosen == actual)
        calibration_count[bucket] += 1
    game_means = [sum(values) / len(values) for values in losses.values()]
    ece = sum((count / len(rows)) * abs(calibration[index] / count - calibration_accuracy[index] / count) for index, count in enumerate(calibration_count) if count)
    return {
        "rows": len(rows),
        "games": len(game_means),
        "negative_log_likelihood": nll / len(rows),
        "brier_score": brier / len(rows),
        "accuracy": correct / len(rows),
        "balanced_accuracy": sum(sum(values) / len(values) for values in recalls.values()) / len(recalls),
        "game_macro_negative_log_likelihood": sum(game_means) / len(game_means),
        "expected_calibration_error": ece,
        "per_label": {
            label: {
                "support": sum(confusion[index]),
                "recall": confusion[index][index] / sum(confusion[index]) if sum(confusion[index]) else None,
                "mean_predicted_probability": predicted_probability[index] / len(rows),
                "confusion_counts": {predicted: confusion[index][predicted_index] for predicted_index, predicted in enumerate(BUYER_CONTINUATION_LABELS)},
            }
            for index, label in enumerate(BUYER_CONTINUATION_LABELS)
        },
    }


def _sliced_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "all": _rows_metrics(rows),
        "by_buyer_action": {action: _rows_metrics([row for row in rows if row.get("buyer_action") == action]) for action in BUYER_ACTIONS},
        "by_preceding_signal": {label: _rows_metrics([row for row in rows if row.get("preceding_signal") == label]) for label in BUYER_CONTINUATION_LABELS},
        "by_identity_scope": {scope: _rows_metrics([row for row in rows if row.get("identity_scope") == scope]) for scope in ("known", "hidden")},
    }


@dataclass(frozen=True)
class _EncodedPartition:
    mechanics: torch.Tensor
    message: torch.Tensor
    message_available: torch.Tensor
    labels: torch.Tensor
    metadata: tuple[Mapping[str, object], ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


class _ReversedPersuasionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.action_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
        self.action = nn.Linear(config.hidden_dim, len(BUYER_CONTINUATION_LABELS))

    def forward(self, mechanics: torch.Tensor, message: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        hidden, _gate = self.action_fusion(mechanics, message, available)
        return self.action(hidden)


def _encode_partition(model: HierarchicalSequenceTwin, dataset: _ContinuationDataset, *, vocabs: CorpusVocabs, batch_size: int, seed: int, device: torch.device, target_lookup: Mapping[str, Mapping[str, object]], mixed_precision: bool) -> _EncodedPartition:
    if not len(dataset):
        return _EncodedPartition(
            mechanics=torch.empty((0, model.config.hidden_dim), dtype=torch.float32),
            message=torch.empty((0, model.config.message_hidden_dim), dtype=torch.float32),
            message_available=torch.empty((0,), dtype=torch.bool),
            labels=torch.empty((0,), dtype=torch.long),
            metadata=(),
        )
    minimum_batch = 8 if device.type == "cuda" else 1
    loader = _make_loader(dataset, vocabs=vocabs, batch_size=batch_size, shuffle=False, seed=seed, minimum_batch=minimum_batch)
    mechanics_rows: list[torch.Tensor] = []
    message_rows: list[torch.Tensor] = []
    available_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    metadata_rows: list[Mapping[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            valid_rows = int(raw_batch.pop("valid_rows"))
            batch = move_batch(raw_batch, device)
            tensor_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision and device.type == "cuda"):
                events = model.event_encoder(tensor_batch)
                static = model.static_encoder(tensor_batch)
                account, _effective = model.account_context("persuasion", tensor_batch["accounts"], force_population=True)
                context = model.context_mix(torch.cat((static, account, static * account), dim=-1))
                mechanics = model.cores["persuasion"](events + context.unsqueeze(1), tensor_batch["lengths"])
                if model.message_encoder is None or model.message_context_mix is None or model.message_cores is None:
                    raise ValueError("buyer-continuation encoding requires the separate message stream")
                message_context = model.message_context_mix(torch.cat((static, account, static * account), dim=-1))
                message_events = model.message_encoder(tensor_batch) + message_context.unsqueeze(1)
                message_present = tensor_batch["event_numeric"][..., 7] > 0.5
                message_sequence, message_lengths, message_available = _compact_message_sequence(message_events, message_present, tensor_batch["lengths"])
                message = model.message_cores["persuasion"](message_sequence, message_lengths)
            mechanics_rows.append(mechanics[:valid_rows].float().cpu())
            message_rows.append(message[:valid_rows].float().cpu())
            available_rows.append(message_available[:valid_rows].cpu())
            label_rows.append(tensor_batch["target_labels"][:valid_rows].cpu())
            for metadata in batch["metadata"][:valid_rows]:
                target = target_lookup[str(metadata["sample_id"])]
                metadata_rows.append({"sample_id": str(metadata["sample_id"]), "game_id": str(metadata["game_id"]), "identity_scope": str(metadata["identity_scope"]), "buyer_action": str(target["buyer_action"]), "preceding_signal": str(target["preceding_signal"])})
    return _EncodedPartition(mechanics=torch.cat(mechanics_rows), message=torch.cat(message_rows), message_available=torch.cat(available_rows), labels=torch.cat(label_rows), metadata=tuple(metadata_rows))


def _combine_encoded(parts: Sequence[_EncodedPartition]) -> _EncodedPartition:
    selected = [part for part in parts if len(part)]
    if not selected:
        raise ValueError("cannot combine empty encoded partitions")
    return _EncodedPartition(mechanics=torch.cat([part.mechanics for part in selected]), message=torch.cat([part.message for part in selected]), message_available=torch.cat([part.message_available for part in selected]), labels=torch.cat([part.labels for part in selected]), metadata=tuple(row for part in selected for row in part.metadata))


def _encode_component(*, component_path: Path, source_vocabs: CorpusVocabs, continuation_vocabs: CorpusVocabs, model_config: ModelConfig, index: _ContinuationIndex, config: BuyerContinuationTrainingConfig, device: torch.device, seed: int) -> dict[str, _EncodedPartition]:
    payload = torch.load(component_path, map_location="cpu", weights_only=False)
    if payload.get("contract") != SHADOW_COMPONENT_CONTRACT:
        raise ValueError(f"unsupported frozen sequence component: {component_path}")
    model = HierarchicalSequenceTwin(source_vocabs, model_config)
    model.load_state_dict(payload["model"])
    model.to(device)
    lookup = {str(row["sample_id"]): row for row in index.targets}
    partitions = {
        "train-deeprmm": index.dataset(split="train", source_types={"deeprmm"}),
        "train-fieldglass": index.dataset(split="train", source_types={"fieldglass"}),
        "validation": index.dataset(split="validation", source_types={"deeprmm"}),
        "test": index.dataset(split="test", source_types={"deeprmm"}),
    }
    encoded = {name: _encode_partition(model, dataset, vocabs=continuation_vocabs, batch_size=config.evaluation_batch_size, seed=seed + offset, device=device, target_lookup=lookup, mixed_precision=config.mixed_precision) for offset, (name, dataset) in enumerate(partitions.items())}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded


def _predict_head(head: _ReversedPersuasionHead, encoded: _EncodedPartition, *, batch_size: int, device: torch.device) -> list[dict[str, object]]:
    head.eval()
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            stop = min(len(encoded), start + batch_size)
            logits = head(encoded.mechanics[start:stop].to(device), encoded.message[start:stop].to(device), encoded.message_available[start:stop].to(device))
            probabilities.extend(F.softmax(logits, dim=-1).cpu().tolist())
    return [{**metadata, "actual": int(label), "probabilities": [float(value) for value in probability]} for metadata, label, probability in zip(encoded.metadata, encoded.labels.tolist(), probabilities, strict=True)]


def _train_component(*, component_path: Path, component_seed: int, encoded: Mapping[str, _EncodedPartition], train_sources: set[str], config: BuyerContinuationTrainingConfig, model_config: ModelConfig, output_path: Path, device: torch.device) -> dict[str, object]:
    train_parts = [encoded["train-deeprmm"]]
    if "fieldglass" in train_sources:
        train_parts.append(encoded["train-fieldglass"])
    train = _combine_encoded(train_parts)
    validation = encoded["validation"]
    test = encoded["test"]
    _set_seed(component_seed + 930_001)
    head = _ReversedPersuasionHead(model_config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
    train_loader = DataLoader(TensorDataset(train.mechanics, train.message, train.message_available, train.labels), batch_size=config.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(component_seed))
    best_nll = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        head.train()
        losses: list[float] = []
        for mechanics, message, available, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.mixed_precision and device.type == "cuda"):
                logits = head(mechanics.to(device), message.to(device), available.to(device))
                loss = F.cross_entropy(logits, labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_rows = _predict_head(head, validation, batch_size=config.evaluation_batch_size, device=device)
        validation_metrics = _rows_metrics(validation_rows)
        current_nll = float(validation_metrics["negative_log_likelihood"])
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation": validation_metrics})
        if current_nll < best_nll - 1e-5:
            best_nll = current_nll
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("reversed Persuasion head did not produce a validation checkpoint")
    head.load_state_dict(best_state)
    validation_rows = _predict_head(head, validation, batch_size=config.evaluation_batch_size, device=device)
    test_rows = _predict_head(head, test, batch_size=config.evaluation_batch_size, device=device)
    checkpoint = {
        "contract": BUYER_CONTINUATION_EXPERIMENT_CONTRACT,
        "component_seed": component_seed,
        "source_component_sha256": file_sha256(component_path),
        "target_labels": list(BUYER_CONTINUATION_LABELS),
        "train_sources": sorted(train_sources),
        "best_epoch": best_epoch,
        "head": best_state,
    }
    torch.save(checkpoint, output_path)
    return {
        "component_seed": component_seed,
        "checkpoint": output_path.name,
        "checkpoint_sha256": file_sha256(output_path),
        "train_rows": len(train),
        "validation": _rows_metrics(validation_rows),
        "test": _rows_metrics(test_rows),
        "best_epoch": best_epoch,
        "history": history,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
    }


def _ensemble(component_results: Sequence[Mapping[str, object]], *, split: str) -> list[dict[str, object]]:
    key = f"{split}_rows"
    aligned = [sorted(result[key], key=lambda row: str(row["sample_id"])) for result in component_results]
    if not aligned:
        return []
    sample_ids = [str(row["sample_id"]) for row in aligned[0]]
    if any([str(row["sample_id"]) for row in rows] != sample_ids for rows in aligned[1:]):
        raise ValueError("reversed-head component predictions are not aligned")
    result: list[dict[str, object]] = []
    for rows in zip(*aligned, strict=True):
        reference = rows[0]
        probabilities = [sum(float(row["probabilities"][index]) for row in rows) / len(rows) for index in range(len(BUYER_CONTINUATION_LABELS))]
        result.append({**reference, "probabilities": probabilities})
    return result


def _paired_game_bootstrap(baseline: Sequence[Mapping[str, object]], arm: Sequence[Mapping[str, object]], *, replicates: int, seed: int) -> dict[str, object]:
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    arm_by_id = {str(row["sample_id"]): row for row in arm}
    if baseline_by_id.keys() != arm_by_id.keys():
        raise ValueError("paired bootstrap prediction sets differ")
    by_game: defaultdict[str, list[float]] = defaultdict(list)
    for sample_id, base in baseline_by_id.items():
        other = arm_by_id[sample_id]
        actual = int(base["actual"])
        delta = -math.log(max(float(other["probabilities"][actual]), 1e-12)) + math.log(max(float(base["probabilities"][actual]), 1e-12))
        by_game[str(base["game_id"])].append(delta)
    game_deltas = [sum(values) / len(values) for values in by_game.values()]
    observed = sum(game_deltas) / len(game_deltas)
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(game_deltas) for _index in game_deltas) / len(game_deltas) for _replicate in range(replicates))
    return {"games": len(game_deltas), "arm_minus_baseline_game_macro_nll": observed, "bootstrap_95_percent_interval": [draws[int(0.025 * replicates)], draws[min(replicates - 1, int(0.975 * replicates))]]}


def _markov_baseline(index: _ContinuationIndex) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    train = index.dataset(split="train", source_types={"deeprmm"})
    counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for target in train.targets:
        key = (str(target["preceding_signal"]), str(target["buyer_action"]))
        counts[key][str(target["target_label"])] += 1
        overall[str(target["target_label"])] += 1
    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("validation", "test"):
        dataset = index.dataset(split=split, source_types={"deeprmm"})
        rows: list[dict[str, object]] = []
        for target in dataset.targets:
            key = (str(target["preceding_signal"]), str(target["buyer_action"]))
            selected = counts.get(key) or overall
            denominator = sum(selected.values()) + len(BUYER_CONTINUATION_LABELS)
            probabilities = [(selected[label] + 1) / denominator for label in BUYER_CONTINUATION_LABELS]
            rows.append({"sample_id": str(target["sample_id"]), "game_id": str(target["game_id"]), "actual": BUYER_CONTINUATION_LABELS.index(str(target["target_label"])), "probabilities": probabilities})
        rows_by_split[split] = rows
    return {"contract": "preceding-signal-and-buyer-action-laplace-markov-baseline-v1", "validation": _rows_metrics(rows_by_split["validation"]), "test": _rows_metrics(rows_by_split["test"])}, rows_by_split


def run_persuasion_buyer_continuation_experiment(*, corpus_dir: Path, sequence_release: Path, output_dir: Path, config: BuyerContinuationTrainingConfig = BuyerContinuationTrainingConfig()) -> dict[str, object]:
    started = time.time()
    corpus_dir = corpus_dir.resolve()
    sequence_release = sequence_release.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"buyer-continuation experiment output already exists: {output_dir}")
    release_manifest = json.loads((sequence_release / "manifest.json").read_text(encoding="utf-8"))
    if release_manifest.get("contract") != SHADOW_CANDIDATE_CONTRACT:
        raise ValueError("buyer-continuation experiment requires a frozen sequence release")
    vocabulary_receipt = release_manifest["vocabulary"]
    vocabulary_path = sequence_release / str(vocabulary_receipt["path"])
    if file_sha256(vocabulary_path) != vocabulary_receipt["sha256"]:
        raise ValueError("frozen sequence vocabulary hash mismatch")
    source_vocabs = CorpusVocabs.from_receipt(json.loads(vocabulary_path.read_text(encoding="utf-8")))
    continuation_vocabs = replace(source_vocabs, target_labels={**source_vocabs.target_labels, "persuasion": Vocabulary(BUYER_CONTINUATION_LABELS)})
    model_config = ModelConfig(**release_manifest["model"])
    index = _ContinuationIndex(corpus_dir, continuation_vocabs)
    has_fieldglass_targets = any(str(target.get("source_type") or "") == "fieldglass" for target in index.targets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        encoded_components: dict[int, dict[str, _EncodedPartition]] = {}
        component_paths: dict[int, Path] = {}
        for component in release_manifest["components"]:
            seed = int(component["seed"])
            direct_path = sequence_release / str(component["path"])
            component_path = direct_path if direct_path.is_file() else sequence_release / "sequence" / str(component["path"])
            component_paths[seed] = component_path
            encoded_components[seed] = _encode_component(component_path=component_path, source_vocabs=source_vocabs, continuation_vocabs=continuation_vocabs, model_config=model_config, index=index, config=config, device=device, seed=seed)
        arms: dict[str, dict[str, object]] = {}
        arm_definitions: list[tuple[str, set[str]]] = [("deeprmm-only", {"deeprmm"})]
        if has_fieldglass_targets:
            arm_definitions.append(("deeprmm-plus-fieldglass", {"deeprmm", "fieldglass"}))
        for arm_name, train_sources in arm_definitions:
            component_results: list[dict[str, object]] = []
            for component in release_manifest["components"]:
                seed = int(component["seed"])
                component_results.append(
                    _train_component(
                        component_path=component_paths[seed],
                        component_seed=seed,
                        model_config=model_config,
                        encoded=encoded_components[seed],
                        train_sources=train_sources,
                        config=config,
                        output_path=staging / f"{arm_name}-seed{seed}.pt",
                        device=device,
                    )
                )
            validation_rows = _ensemble(component_results, split="validation")
            test_rows = _ensemble(component_results, split="test")
            arms[arm_name] = {
                "train_sources": sorted(train_sources),
                "components": [{key: value for key, value in result.items() if key not in {"validation_rows", "test_rows"}} for result in component_results],
                "ensemble": {"validation": _rows_metrics(validation_rows), "test": _rows_metrics(test_rows), "validation_slices": _sliced_metrics(validation_rows), "test_slices": _sliced_metrics(test_rows)},
                "validation_rows": validation_rows,
                "test_rows": test_rows,
            }
        selected_arm = "deeprmm-only"
        validation_comparison: dict[str, object] | None = None
        test_comparison: dict[str, object] | None = None
        if "deeprmm-plus-fieldglass" in arms:
            validation_comparison = _paired_game_bootstrap(arms["deeprmm-only"]["validation_rows"], arms["deeprmm-plus-fieldglass"]["validation_rows"], replicates=config.bootstrap_replicates, seed=72_011)
            test_comparison = _paired_game_bootstrap(arms["deeprmm-only"]["test_rows"], arms["deeprmm-plus-fieldglass"]["test_rows"], replicates=config.bootstrap_replicates, seed=72_012)
            upper = float(validation_comparison["bootstrap_95_percent_interval"][1])
            selected_arm = "deeprmm-plus-fieldglass" if upper < 0.0 else "deeprmm-only"
        baseline, baseline_rows = _markov_baseline(index)
        predictive_evidence = {
            "validation": _paired_game_bootstrap(baseline_rows["validation"], arms[selected_arm]["validation_rows"], replicates=config.bootstrap_replicates, seed=72_013),
            "test": _paired_game_bootstrap(baseline_rows["test"], arms[selected_arm]["test_rows"], replicates=config.bootstrap_replicates, seed=72_014),
            "interpretation": "Negative intervals favor the selected reversed head over the preceding-signal-and-buyer-action Markov baseline; the test interval is confirmatory only for this frozen retrospective cut.",
        }
        fieldglass_decision = {
            "evaluated": validation_comparison is not None,
            "selected_arm": selected_arm,
            "include_fieldglass": selected_arm == "deeprmm-plus-fieldglass",
            "rule": "When Fieldglass rows are supplied, include them only when the upper bound of the paired game-bootstrap 95% interval for augmented-minus-DeepRMM validation game-macro NLL is below 0; test data never select the arm. When no Fieldglass root is supplied, do not train a redundant augmentation arm.",
            "validation": validation_comparison,
            "test_diagnostic_not_used_for_selection": test_comparison,
        }
        for arm in arms.values():
            arm.pop("validation_rows")
            arm.pop("test_rows")
        corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        result = {
            "schema_version": 1,
            "contract": BUYER_CONTINUATION_EXPERIMENT_CONTRACT,
            "status": "complete-offline-candidate",
            "created_at": _utc_now(),
            "objective": "Given the causal Persuasion prefix plus a buyer buy/pass action, predict the opponent seller's next-round positive, negative, or unknown signal.",
            "causal_boundary": "The candidate buyer action is present, its eventual response time is masked, and a buy-caused quality reveal is absent because it is unavailable before action selection.",
            "target_labels": list(BUYER_CONTINUATION_LABELS),
            "training": asdict(config),
            "device": {"type": device.type, "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__, "cuda": torch.version.cuda},
            "corpus": {"path": str(corpus_dir), "manifest_sha256": file_sha256(corpus_dir / "manifest.json"), "inventory": corpus_manifest["inventory"]},
            "source_sequence_release": {"path": str(sequence_release), "manifest_sha256": file_sha256(sequence_release / "manifest.json"), "candidate_id": release_manifest["candidate_id"]},
            "model": model_config.receipt(),
            "trainable_scope": "A newly initialized Persuasion action fusion and 3-class action layer on each frozen population Mamba-2 backbone; all shared encoders and sequence cores remain frozen.",
            "baseline": baseline,
            "predictive_evidence_against_baseline": predictive_evidence,
            "arms": arms,
            "fieldglass_assessment": fieldglass_decision,
            "selected_arm": selected_arm,
            "selected_test": arms[selected_arm]["ensemble"]["test"],
            "authority": "offline evidence only; no live policy or selector activation",
            "elapsed_seconds": time.time() - started,
            "implementation_sha256": file_sha256(Path(__file__)),
        }
        _write_json(staging / "result.json", result)
        _write_json(staging / "manifest.json", {key: result[key] for key in ("schema_version", "contract", "status", "created_at", "objective", "causal_boundary", "target_labels", "corpus", "source_sequence_release", "trainable_scope", "fieldglass_assessment", "selected_arm", "selected_test", "authority", "implementation_sha256")})
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**result, "output_dir": str(output_dir), "result_sha256": file_sha256(output_dir / "result.json")}


def freeze_persuasion_buyer_continuation_release(*, experiment_dir: Path, output_dir: Path, release_id: str) -> dict[str, object]:
    """Freeze only the validation-selected reversed heads and their evidence without activating them live."""
    experiment_dir = experiment_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"buyer-continuation release output already exists: {output_dir}")
    result_path = experiment_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("contract") != BUYER_CONTINUATION_EXPERIMENT_CONTRACT or result.get("status") != "complete-offline-candidate":
        raise ValueError("buyer-continuation experiment is not complete")
    selected_arm = str(result.get("selected_arm") or "")
    selected = result.get("arms", {}).get(selected_arm)
    if not isinstance(selected, Mapping):
        raise ValueError("buyer-continuation experiment has no selected arm")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        components: list[dict[str, object]] = []
        raw_components = selected.get("components")
        if not isinstance(raw_components, list) or len(raw_components) < 2:
            raise ValueError("selected buyer-continuation arm has fewer than 2 components")
        for raw in raw_components:
            if not isinstance(raw, Mapping):
                raise ValueError("malformed buyer-continuation component receipt")
            source = experiment_dir / str(raw["checkpoint"])
            if file_sha256(source) != raw.get("checkpoint_sha256"):
                raise ValueError(f"buyer-continuation checkpoint hash mismatch: {source}")
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if payload.get("contract") != BUYER_CONTINUATION_EXPERIMENT_CONTRACT or payload.get("train_sources") != selected.get("train_sources"):
                raise ValueError(f"buyer-continuation checkpoint contract mismatch: {source}")
            destination = staging / source.name
            shutil.copy2(source, destination)
            destination.chmod(0o644)
            components.append({"seed": int(raw["component_seed"]), "path": destination.name, "bytes": destination.stat().st_size, "sha256": file_sha256(destination), "best_epoch": int(raw["best_epoch"]), "validation": raw["validation"], "test": raw["test"]})
        manifest = {
            "schema_version": 1,
            "contract": BUYER_CONTINUATION_RELEASE_CONTRACT,
            "release_id": release_id,
            "status": "frozen-offline-candidate",
            "frozen_at": _utc_now(),
            "objective": result["objective"],
            "causal_boundary": result["causal_boundary"],
            "target_labels": result["target_labels"],
            "selected_arm": selected_arm,
            "train_sources": selected["train_sources"],
            "ensemble": {"components": len(components), "rule": "unweighted arithmetic mean of component categorical probabilities"},
            "components": components,
            "retrospective_evidence": {"baseline": result["baseline"], "selected": selected["ensemble"], "against_baseline": result["predictive_evidence_against_baseline"], "fieldglass_assessment": result["fieldglass_assessment"]},
            "source_experiment": {"path_at_freeze": str(experiment_dir), "result_sha256": file_sha256(result_path), "implementation_sha256": result["implementation_sha256"]},
            "source_corpus": result["corpus"],
            "source_sequence_release": result["source_sequence_release"],
            "model": result["model"],
            "trainable_scope": result["trainable_scope"],
            "authority": "none; frozen offline candidate only",
        }
        _write_json(staging / "manifest.json", manifest)
        fieldglass_summary = "Fieldglass was not supplied or trained." if not result["fieldglass_assessment"].get("evaluated") else f"Fieldglass was evaluated as train-only augmentation against DeepRMM-only validation and was {'included' if result['fieldglass_assessment']['include_fieldglass'] else 'rejected'}."
        readme = f"# Persuasion buyer-continuation head {release_id}\n\nThis frozen offline candidate predicts the opponent seller's next-round positive, negative, or unknown signal after a candidate buyer `buy` or `pass` action. It does not have live policy authority.\n\nThe selected arm is `{selected_arm}`. {fieldglass_summary} See `manifest.json` for the causal boundary, artifact hashes, proper-score evidence, and component receipts.\n"
        (staging / "README.md").write_text(readme, encoding="utf-8")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**manifest, "output_dir": str(output_dir), "manifest_sha256": file_sha256(output_dir / "manifest.json")}
