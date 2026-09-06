"""In-memory indices and PyTorch collation for normalized sequence corpora."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl
import torch
from torch.utils.data import Dataset

from .corpus import GLEE_FAMILIES, HASH_BINS


SPECIALS = ("<pad>", "<unk>", "<bos>")
TARGET_LABELS = {
    "bargaining": ("proposal", "accept", "reject", "walkaway"),
    "negotiation": ("proposal", "accept", "reject", "walkaway"),
    "persuasion": ("signal_positive", "signal_negative", "signal_unknown", "buy", "pass"),
}
MAX_BATCH_MESSAGE_HASHES = 64


@dataclass(frozen=True)
class Vocabulary:
    values: tuple[str, ...]
    index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {value: index for index, value in enumerate(self.values)})

    @classmethod
    def build(cls, values: Iterable[object], *, specials: Sequence[str] = SPECIALS) -> "Vocabulary":
        ordered = tuple(dict.fromkeys((*specials, *(sorted({str(value) for value in values if value is not None})))))
        return cls(ordered)

    def encode(self, value: object) -> int:
        return self.index.get(str(value), self.index.get("<unk>", 0))

    def encode_strict(self, value: object) -> int:
        try:
            return self.index[str(value)]
        except KeyError as error:
            raise ValueError(f"value is absent from frozen vocabulary: {value!r}") from error

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class CorpusVocabs:
    actor: Vocabulary
    kind: Vocabulary
    event_action: Vocabulary
    message_act: Vocabulary
    quality: Vocabulary
    discourse: Vocabulary
    our_role: Vocabulary
    opponent_role: Vocabulary
    identity_scope: Vocabulary
    account: Vocabulary
    target_labels: Mapping[str, Vocabulary]
    target_messages: Mapping[str, Vocabulary]

    def receipt(self) -> dict[str, object]:
        return {
            "actor": list(self.actor.values),
            "kind": list(self.kind.values),
            "event_action": list(self.event_action.values),
            "message_act": list(self.message_act.values),
            "quality": list(self.quality.values),
            "discourse": list(self.discourse.values),
            "our_role": list(self.our_role.values),
            "opponent_role": list(self.opponent_role.values),
            "identity_scope": list(self.identity_scope.values),
            "account": list(self.account.values),
            "target_labels": {family: list(vocab.values) for family, vocab in self.target_labels.items()},
            "target_messages": {family: list(vocab.values) for family, vocab in self.target_messages.items()},
        }

    @classmethod
    def from_receipt(cls, value: Mapping[str, object]) -> "CorpusVocabs":
        def vocabulary(name: str) -> Vocabulary:
            raw = value.get(name)
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise ValueError(f"invalid frozen vocabulary: {name}")
            return Vocabulary(tuple(raw))

        def family_vocabularies(name: str) -> dict[str, Vocabulary]:
            raw = value.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"invalid frozen family vocabulary: {name}")
            result: dict[str, Vocabulary] = {}
            for family in GLEE_FAMILIES:
                items = raw.get(family)
                if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                    raise ValueError(f"invalid frozen family vocabulary: {name}/{family}")
                result[family] = Vocabulary(tuple(items))
            return result

        return cls(
            actor=vocabulary("actor"),
            kind=vocabulary("kind"),
            event_action=vocabulary("event_action"),
            message_act=vocabulary("message_act"),
            quality=vocabulary("quality"),
            discourse=vocabulary("discourse"),
            our_role=vocabulary("our_role"),
            opponent_role=vocabulary("opponent_role"),
            identity_scope=vocabulary("identity_scope"),
            account=vocabulary("account"),
            target_labels=family_vocabularies("target_labels"),
            target_messages=family_vocabularies("target_messages"),
        )


class CorpusIndex:
    """Load each normalized event once and expose prefix references as training samples."""

    def __init__(self, corpus_dirs: Sequence[Path]) -> None:
        if not corpus_dirs:
            raise ValueError("at least one corpus directory is required")
        game_frames: list[pl.DataFrame] = []
        event_frames: list[pl.DataFrame] = []
        target_frames: list[pl.DataFrame] = []
        self.manifests: list[dict[str, object]] = []
        import json

        for raw in corpus_dirs:
            directory = raw.resolve()
            self.manifests.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
            game_frames.append(pl.read_parquet(directory / "games.parquet"))
            event_frames.append(pl.read_parquet(directory / "events.parquet"))
            target_frames.append(pl.read_parquet(directory / "targets.parquet"))
        target_label_spaces = [manifest.get("target_labels") for manifest in self.manifests]
        supplied_target_label_spaces = [value for value in target_label_spaces if value is not None]
        if supplied_target_label_spaces and len(supplied_target_label_spaces) != len(target_label_spaces):
            raise ValueError("combined corpora mix explicit and default target-label spaces")
        target_labels: Mapping[str, Sequence[str]] = TARGET_LABELS
        if supplied_target_label_spaces:
            reference = supplied_target_label_spaces[0]
            if any(value != reference for value in supplied_target_label_spaces[1:]) or not isinstance(reference, Mapping):
                raise ValueError("combined corpora have different target-label spaces")
            parsed: dict[str, tuple[str, ...]] = {}
            for family in GLEE_FAMILIES:
                values = reference.get(family)
                if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values) or len(set(values)) != len(values):
                    raise ValueError(f"invalid explicit target-label space for {family}")
                parsed[family] = tuple(values)
            target_labels = parsed
        games = pl.concat(game_frames, how="vertical_relaxed")
        events = pl.concat(event_frames, how="vertical_relaxed")
        targets = pl.concat(target_frames, how="vertical_relaxed")
        if games["game_id"].n_unique() != games.height:
            raise ValueError("combined corpus has duplicate game IDs")
        self.games = {str(row["game_id"]): row for row in games.to_dicts()}
        grouped: dict[str, list[dict[str, object]]] = {game_id: [] for game_id in self.games}
        for row in events.sort(["game_id", "event_index"]).to_dicts():
            grouped[str(row["game_id"])].append(row)
        self.events = grouped
        self.targets = targets.sort(["game_id", "target_event_index"]).to_dicts()
        self.vocabs = self._build_vocabs(games, events, targets, target_labels=target_labels)

    @staticmethod
    def _build_vocabs(games: pl.DataFrame, events: pl.DataFrame, targets: pl.DataFrame, *, target_labels: Mapping[str, Sequence[str]] = TARGET_LABELS) -> CorpusVocabs:
        training_accounts = games.filter((pl.col("chronological_split") == "train") & pl.col("account_key").is_not_null())["account_key"].to_list()
        discourse_values = [value for values in events["message_discourse_acts"].to_list() for value in (values or [])]
        frozen_target_labels = {family: Vocabulary.build(target_labels[family], specials=()) for family in GLEE_FAMILIES}
        target_messages: dict[str, Vocabulary] = {}
        for family in GLEE_FAMILIES:
            family_game_ids = set(games.filter(pl.col("family") == family)["game_id"].to_list())
            values = targets.filter(pl.col("game_id").is_in(list(family_game_ids)) & pl.col("target_message_present"))["target_message_act"].drop_nulls().to_list()
            target_messages[family] = Vocabulary.build(values, specials=("none",))
        return CorpusVocabs(
            actor=Vocabulary.build(events["actor"].to_list()),
            kind=Vocabulary.build(events["kind"].to_list()),
            event_action=Vocabulary.build(events["action_label"].to_list()),
            message_act=Vocabulary.build(events["message_family_act"].to_list()),
            quality=Vocabulary.build(events["visible_quality"].drop_nulls().to_list()),
            discourse=Vocabulary.build(discourse_values, specials=("<unk>",)),
            our_role=Vocabulary.build(games["our_role"].to_list()),
            opponent_role=Vocabulary.build(games["opponent_role"].to_list()),
            identity_scope=Vocabulary.build(games["identity_scope"].to_list()),
            account=Vocabulary.build(training_accounts, specials=("<population>",)),
            target_labels=frozen_target_labels,
            target_messages=target_messages,
        )

    def subset(self, *, family: str, split: str, account_disjoint_fold: int | None = None, source_types: set[str] | None = None) -> "SequenceTargetDataset":
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        selected: list[dict[str, object]] = []
        for target in self.targets:
            game = self.games[str(target["game_id"])]
            if game["family"] != family or target["chronological_split"] != split:
                continue
            if source_types is not None and str(target["source_type"]) not in source_types:
                continue
            fold = int(target.get("account_fold") if target.get("account_fold") is not None else -1)
            if account_disjoint_fold is not None:
                if split == "train" and fold == account_disjoint_fold:
                    continue
                if split != "train" and fold != account_disjoint_fold:
                    continue
            selected.append(target)
        return SequenceTargetDataset(self, selected)


class SequenceTargetDataset(Dataset[dict[str, object]]):
    def __init__(self, corpus: CorpusIndex, targets: Sequence[dict[str, object]]) -> None:
        self.corpus = corpus
        self.targets = tuple(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        game_id = str(target["game_id"])
        prefix_length = int(target["prefix_length"])
        return {"game": self.corpus.games[game_id], "events": self.corpus.events[game_id][:prefix_length], "target": target}


def _scaled_signed(value: object, divisor: float = 10.0) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0, 0.0
    number = float(value)
    return math.copysign(math.log1p(abs(number)) / divisor, number), 1.0


class SequenceCollator:
    """Convert reference samples into padded causal prefixes without materializing prefixes on disk."""

    def __init__(self, vocabs: CorpusVocabs, family: str, *, excluded_account_fold: int | None = None, history_window: int | None = None, mask_message_inputs: bool = False) -> None:
        if history_window is not None and history_window < 1:
            raise ValueError("history_window must be positive when supplied")
        self.vocabs = vocabs
        self.family = family
        self.excluded_account_fold = excluded_account_fold
        self.history_window = history_window
        self.mask_message_inputs = mask_message_inputs
        self.discourse_mapping = vocabs.discourse.index

    def _events(self, sample: Mapping[str, object]) -> list[Mapping[str, object]]:
        events = list(sample["events"])
        return events if self.history_window is None else events[-self.history_window :]

    @staticmethod
    def _static_numeric(game: Mapping[str, object]) -> list[float]:
        self_value, self_present = _scaled_signed(game.get("static_self_value"))
        opponent_value, opponent_present = _scaled_signed(game.get("static_visible_opponent_value"))
        probability = game.get("static_environment_probability")
        probability_present = isinstance(probability, (int, float)) and not isinstance(probability, bool) and math.isfinite(float(probability))
        maximum = game.get("max_rounds") if game.get("horizon_known") is True else None
        return [
            float(game.get("complete_information") is True),
            float(game.get("horizon_known") is True),
            float(game.get("messages_allowed") is True),
            math.log1p(int(maximum)) / 6.0 if isinstance(maximum, int) and maximum > 0 else 0.0,
            float(game.get("static_scale_log") or 0.0) / 20.0,
            self_value,
            self_present,
            opponent_value,
            opponent_present,
            float(probability) if probability_present else 0.0,
            float(probability_present),
            float(game.get("static_aux_value") or 0.0) / 20.0,
            float(game.get("static_seller_knows_quality") is True),
        ]

    @staticmethod
    def _event_numeric(event: Mapping[str, object]) -> list[float]:
        value = event.get("action_value")
        value_present = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        auxiliary = event.get("action_aux_value")
        auxiliary_present = isinstance(auxiliary, (int, float)) and not isinstance(auxiliary, bool) and math.isfinite(float(auxiliary))
        delay = event.get("response_time_ms")
        delay_present = isinstance(delay, (int, float)) and not isinstance(delay, bool) and float(delay) >= 0 and math.isfinite(float(delay))
        return [
            float(event.get("round_phase") or 0.0),
            float(value) if value_present else 0.0,
            float(value_present),
            float(auxiliary) if auxiliary_present else 0.0,
            float(auxiliary_present),
            math.log1p(float(delay)) / 12.0 if delay_present else 0.0,
            float(delay_present),
            float(event.get("message_present") is True),
            min(2.0, float(event.get("message_chars") or 0) / 256.0),
            min(2.0, float(event.get("message_words") or 0) / 64.0),
            float(event.get("message_uppercase_ratio") or 0.0),
            float(event.get("message_digit_ratio") or 0.0),
            min(2.0, float(event.get("message_question_marks") or 0) / 4.0),
            min(2.0, float(event.get("message_exclamation_marks") or 0) / 4.0),
            min(2.0, float(event.get("message_commas") or 0) / 8.0),
            min(2.0, float(event.get("message_semicolons") or 0) / 4.0),
            min(2.0, float(event.get("message_currency_marks") or 0) / 4.0),
            min(2.0, float(event.get("message_percent_marks") or 0) / 4.0),
            min(2.0, float(event.get("message_decimal_numbers") or 0) / 4.0),
        ]

    def __call__(self, samples: Sequence[dict[str, object]]) -> dict[str, object]:
        batch_size = len(samples)
        event_lists = [self._events(sample) for sample in samples]
        lengths = torch.tensor([len(events) + 1 for events in event_lists], dtype=torch.long)
        maximum = int(lengths.max().item())
        event_numeric = torch.zeros(batch_size, maximum, 19, dtype=torch.float32)
        event_categorical = torch.zeros(batch_size, maximum, 5, dtype=torch.long)
        discourse = torch.zeros(batch_size, maximum, len(self.vocabs.discourse), dtype=torch.float32)
        message_bins = torch.zeros(batch_size, maximum, MAX_BATCH_MESSAGE_HASHES, dtype=torch.long)
        message_bin_mask = torch.zeros(batch_size, maximum, MAX_BATCH_MESSAGE_HASHES, dtype=torch.bool)
        static_numeric = torch.zeros(batch_size, 13, dtype=torch.float32)
        static_categorical = torch.zeros(batch_size, 3, dtype=torch.long)
        accounts = torch.zeros(batch_size, dtype=torch.long)
        target_labels = torch.zeros(batch_size, dtype=torch.long)
        target_action_mask = torch.zeros(batch_size, dtype=torch.bool)
        target_values = torch.zeros(batch_size, dtype=torch.float32)
        target_value_mask = torch.zeros(batch_size, dtype=torch.bool)
        target_delays = torch.zeros(batch_size, dtype=torch.float32)
        target_delay_mask = torch.zeros(batch_size, dtype=torch.bool)
        metadata: list[dict[str, object]] = []
        for batch_index, (sample, events) in enumerate(zip(samples, event_lists, strict=True)):
            game = sample["game"]
            target = sample["target"]
            static_numeric[batch_index] = torch.tensor(self._static_numeric(game), dtype=torch.float32)
            static_categorical[batch_index] = torch.tensor([self.vocabs.our_role.encode(game.get("our_role")), self.vocabs.opponent_role.encode(game.get("opponent_role")), self.vocabs.identity_scope.encode(game.get("identity_scope"))], dtype=torch.long)
            raw_fold = game.get("account_fold")
            excluded = self.excluded_account_fold is not None and isinstance(raw_fold, int) and raw_fold == self.excluded_account_fold
            accounts[batch_index] = 0 if excluded else self.vocabs.account.encode(game.get("account_key") if game.get("account_key") is not None else "<population>")
            event_categorical[batch_index, 0] = torch.tensor([self.vocabs.actor.encode("<bos>"), self.vocabs.kind.encode("<bos>"), self.vocabs.event_action.encode("<bos>"), self.vocabs.message_act.encode("<bos>"), self.vocabs.quality.encode("<bos>")], dtype=torch.long)
            for sequence_index, event in enumerate(events, start=1):
                numeric = self._event_numeric(event)
                last_prefix_event = sequence_index == len(events)
                legacy_future_mask = bool(target.get("mask_last_prefix_future_fields")) and last_prefix_event
                delay_masked = last_prefix_event and (legacy_future_mask or bool(target.get("mask_last_prefix_response_time")))
                message_masked = last_prefix_event and (legacy_future_mask or bool(target.get("mask_last_prefix_message_fields")))
                if delay_masked:
                    numeric[5:7] = [0.0, 0.0]
                if self.mask_message_inputs or message_masked:
                    numeric[7:] = [0.0] * (len(numeric) - 7)
                event_numeric[batch_index, sequence_index] = torch.tensor(numeric, dtype=torch.float32)
                event_categorical[batch_index, sequence_index] = torch.tensor(
                    [
                        self.vocabs.actor.encode(event.get("actor")),
                        self.vocabs.kind.encode(event.get("kind")),
                        self.vocabs.event_action.encode(event.get("action_label")),
                        0 if self.mask_message_inputs or message_masked else self.vocabs.message_act.encode(event.get("message_family_act")),
                        self.vocabs.quality.encode(event.get("visible_quality") if event.get("visible_quality") is not None else "<unk>"),
                    ],
                    dtype=torch.long,
                )
                if not self.mask_message_inputs and not message_masked:
                    for act in event.get("message_discourse_acts") or []:
                        discourse[batch_index, sequence_index, self.discourse_mapping.get(str(act), self.discourse_mapping.get("<unk>", 0))] = 1.0
                    bins = [int(value) % HASH_BINS for value in (event.get("message_hash_bins") or [])[:MAX_BATCH_MESSAGE_HASHES]]
                    if bins:
                        message_bins[batch_index, sequence_index, : len(bins)] = torch.tensor(bins, dtype=torch.long)
                        message_bin_mask[batch_index, sequence_index, : len(bins)] = True
            target_labels[batch_index] = self.vocabs.target_labels[self.family].encode_strict(target.get("target_label"))
            target_action_mask[batch_index] = target.get("target_kind") != "proposal"
            if target.get("target_value_present") is True and isinstance(target.get("target_value"), (int, float)):
                target_values[batch_index] = float(target["target_value"])
                target_value_mask[batch_index] = True
            if target.get("target_delay_present") is True and isinstance(target.get("target_delay_log_ms"), (int, float)):
                target_delays[batch_index] = float(target["target_delay_log_ms"])
                target_delay_mask[batch_index] = True
            metadata.append({"sample_id": target["sample_id"], "game_id": target["game_id"], "prefix_length": target["prefix_length"], "target_event_index": target.get("target_event_index", target["prefix_length"]), "target_kind": target["target_kind"], "target_label": target["target_label"], "identity_scope": target["identity_scope"], "account_key": target.get("account_key"), "account_known_to_model": int(accounts[batch_index]) > 0, "source_type": target["source_type"]})
        return {
            "family": self.family,
            "event_numeric": event_numeric,
            "event_categorical": event_categorical,
            "discourse": discourse,
            "message_bins": message_bins,
            "message_bin_mask": message_bin_mask,
            "lengths": lengths,
            "static_numeric": static_numeric,
            "static_categorical": static_categorical,
            "accounts": accounts,
            "target_labels": target_labels,
            "target_action_mask": target_action_mask,
            "target_values": target_values,
            "target_value_mask": target_value_mask,
            "target_delays": target_delays,
            "target_delay_mask": target_delay_mask,
            "metadata": metadata,
        }


def move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
