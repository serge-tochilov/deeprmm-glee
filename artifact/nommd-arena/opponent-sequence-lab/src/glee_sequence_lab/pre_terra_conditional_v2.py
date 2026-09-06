"""Build and validate action-conditioned pre-Terra opponent-response corpora."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from nommd_arena.glee_bargaining_twin import classify_message_act
from nommd_arena.glee_negotiation_twin_v2 import classify_negotiation_message, opponent_demand, opponent_surplus_share
from nommd_arena.glee_persuasion_twin_v2 import classify_persuasion_signal
from nommd_arena.glee_policy import normalize_action

from .corpus import GLEE_FAMILIES, HASH_BINS, MAX_MESSAGE_HASHES, _message_features, _round_phase, file_sha256, object_sha256
from .pre_terra_v3 import CATEGORICAL_BINS, FEATURE_DIMENSION, NUMERIC_BINS, PRESENCE_BINS, PRE_TERRA_V3_CORPUS_CONTRACT, _path_segment, _stable_bin


CONDITIONAL_CORPUS_CONTRACT = "glee-post-planner-action-conditional-corpus-v3"
CANDIDATE_ACTION_PROJECTION_CONTRACT = "glee-post-planner-candidate-action-projection-v2"
CONDITIONAL_TARGET_LABELS = {
    "bargaining": ("accept", "reject", "walkaway"),
    "negotiation": ("accept", "reject", "walkaway"),
    "persuasion": ("buy", "pass"),
}


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class CandidateAction:
    family: str
    phase: str
    kind: str
    action_label: str
    action_value: float | None
    action_aux_value: float | None
    round_number: int
    round_phase: float
    visible_quality: str | None = None
    message_present: bool = False
    message_family_act: str = "none"
    message_discourse_acts: tuple[str, ...] = ()
    message_hash_bins: tuple[int, ...] = ()
    message_chars: int = 0
    message_words: int = 0
    message_uppercase_ratio: float = 0.0
    message_digit_ratio: float = 0.0
    message_question_marks: int = 0
    message_exclamation_marks: int = 0
    message_commas: int = 0
    message_semicolons: int = 0
    message_currency_marks: int = 0
    message_percent_marks: int = 0
    message_decimal_numbers: int = 0
    message_sha256: str | None = None

    @staticmethod
    def _message_fields(value: Mapping[str, object]) -> dict[str, object]:
        return {
            "message_present": value.get("message_present") is True,
            "message_family_act": str(value.get("message_family_act") or "none"),
            "message_discourse_acts": tuple(str(item) for item in (value.get("message_discourse_acts") or ())),
            "message_hash_bins": tuple(int(item) for item in (value.get("message_hash_bins") or ()))[:MAX_MESSAGE_HASHES],
            "message_chars": int(value.get("message_chars") or 0),
            "message_words": int(value.get("message_words") or 0),
            "message_uppercase_ratio": float(value.get("message_uppercase_ratio") or 0.0),
            "message_digit_ratio": float(value.get("message_digit_ratio") or 0.0),
            "message_question_marks": int(value.get("message_question_marks") or 0),
            "message_exclamation_marks": int(value.get("message_exclamation_marks") or 0),
            "message_commas": int(value.get("message_commas") or 0),
            "message_semicolons": int(value.get("message_semicolons") or 0),
            "message_currency_marks": int(value.get("message_currency_marks") or 0),
            "message_percent_marks": int(value.get("message_percent_marks") or 0),
            "message_decimal_numbers": int(value.get("message_decimal_numbers") or 0),
            "message_sha256": str(value["message_sha256"]) if value.get("message_sha256") is not None else None,
        }

    @classmethod
    def from_event(cls, *, family: str, phase: str, event: Mapping[str, object]) -> "CandidateAction":
        value = cls(
            family=family,
            phase=phase,
            kind=str(event.get("kind") or ""),
            action_label=str(event.get("action_label") or ""),
            action_value=_finite_number(event.get("action_value")),
            action_aux_value=_finite_number(event.get("action_aux_value")),
            round_number=int(event.get("round_number") or 0),
            round_phase=float(_finite_number(event.get("round_phase")) or 0.0),
            visible_quality=str(event["visible_quality"]) if event.get("visible_quality") is not None else None,
            **cls._message_fields(event),
        )
        value.validate()
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, family: str, phase: str) -> "CandidateAction":
        candidate = cls(
            family=family,
            phase=phase,
            kind=str(value.get("kind") or ""),
            action_label=str(value.get("action_label") or ""),
            action_value=_finite_number(value.get("action_value")),
            action_aux_value=_finite_number(value.get("action_aux_value")),
            round_number=int(value.get("round_number") or 0),
            round_phase=float(_finite_number(value.get("round_phase")) or 0.0),
            visible_quality=str(value["visible_quality"]) if value.get("visible_quality") is not None else None,
            **cls._message_fields(value),
        )
        candidate.validate()
        return candidate

    @classmethod
    def from_feature_row(cls, value: Mapping[str, object], *, family: str, phase: str) -> "CandidateAction":
        return cls.from_mapping(
            {
                "kind": value.get("candidate_kind"),
                "action_label": value.get("candidate_action_label"),
                "action_value": value.get("candidate_action_value"),
                "action_aux_value": value.get("candidate_action_aux_value"),
                "round_number": value.get("candidate_round_number"),
                "round_phase": value.get("candidate_round_phase"),
                "visible_quality": value.get("candidate_visible_quality"),
                "message_present": value.get("candidate_message_present"),
                "message_family_act": value.get("candidate_message_family_act"),
                "message_discourse_acts": value.get("candidate_message_discourse_acts"),
                "message_hash_bins": value.get("candidate_message_hash_bins"),
                "message_chars": value.get("candidate_message_chars"),
                "message_words": value.get("candidate_message_words"),
                "message_uppercase_ratio": value.get("candidate_message_uppercase_ratio"),
                "message_digit_ratio": value.get("candidate_message_digit_ratio"),
                "message_question_marks": value.get("candidate_message_question_marks"),
                "message_exclamation_marks": value.get("candidate_message_exclamation_marks"),
                "message_commas": value.get("candidate_message_commas"),
                "message_semicolons": value.get("candidate_message_semicolons"),
                "message_currency_marks": value.get("candidate_message_currency_marks"),
                "message_percent_marks": value.get("candidate_message_percent_marks"),
                "message_decimal_numbers": value.get("candidate_message_decimal_numbers"),
                "message_sha256": value.get("candidate_message_sha256"),
            },
            family=family,
            phase=phase,
        )

    @classmethod
    def from_live_action(cls, *, game: Mapping[str, object], action: Mapping[str, object]) -> "CandidateAction":
        """Project one exact normalized planner action into the training-time bridge coordinates."""
        mutable_game = dict(game)
        family = str(mutable_game.get("game_family") or "")
        state = mutable_game.get("game_state")
        valid_actions = mutable_game.get("valid_actions")
        if not isinstance(state, Mapping) or not isinstance(valid_actions, Mapping):
            raise ValueError("live candidate has no valid game state or action contract")
        action_type = str(valid_actions.get("type") or "")
        phase = str(mutable_game.get("phase") or action_type)
        round_number = int(state.get("round") or 0)
        if round_number < 1:
            raise ValueError("live candidate has no positive round number")
        normalized = normalize_action(mutable_game, dict(action))
        round_phase = _round_phase(round_number, state)
        message = ""
        family_act = "none"
        if family == "bargaining" and action_type == "offer":
            pool = _finite_number(state.get("money_to_divide"))
            our_player = str(mutable_game.get("your_player") or state.get("current_player") or "")
            self_key, opponent_key = ("alice_gain", "bob_gain") if our_player == "player_1" else ("bob_gain", "alice_gain") if our_player == "player_2" else ("", "")
            if pool is None or pool <= 0 or not self_key:
                raise ValueError("live Bargaining candidate has invalid player or pool coordinates")
            message = str(normalized.get("message") or "")
            family_act = classify_message_act(message, messages_allowed=state.get("messages_allowed") is True)
            candidate_coordinates = {"family": family, "phase": phase, "kind": "proposal", "action_label": "proposal", "action_value": float(normalized[opponent_key]) / pool, "action_aux_value": float(normalized[self_key]) / pool, "round_number": round_number, "round_phase": round_phase}
        elif family == "negotiation" and action_type == "offer":
            our_player = str(mutable_game.get("your_player") or state.get("current_player") or "")
            opponent_player = "player_2" if our_player == "player_1" else "player_1" if our_player == "player_2" else ""
            our_role = str(state.get(f"{our_player}_role") or "")
            opponent_role = str(state.get(f"{opponent_player}_role") or "")
            our_value = _finite_number(state.get(f"{our_player}_value"))
            opponent_value = _finite_number(state.get(f"{opponent_player}_value")) if state.get("complete_information") is True else None
            if not opponent_player or our_value is None or our_value <= 0 or {our_role, opponent_role} != {"buyer", "seller"}:
                raise ValueError("live Negotiation candidate has invalid role or value coordinates")
            price = float(normalized["product_price"])
            demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
            share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
            message = str(normalized.get("message") or "")
            family_act = classify_negotiation_message(message, messages_allowed=state.get("messages_allowed") is not False)
            candidate_coordinates = {"family": family, "phase": phase, "kind": "proposal", "action_label": "proposal", "action_value": math.tanh(demand / 3.0), "action_aux_value": share if share is not None else demand, "round_number": round_number, "round_phase": round_phase}
        elif family == "persuasion" and action_type in {"seller_message", "seller_recommendation"}:
            if action_type == "seller_recommendation":
                polarity = "positive" if normalized.get("decision") == "yes" else "negative"
                family_act = f"recommend_{polarity}"
            else:
                channel = str(state.get("seller_message_type") or "text").casefold()
                message = str(normalized.get("message") or "")
                polarity, family_act, _fingerprint = classify_persuasion_signal(message, channel=channel)
            if polarity not in {"positive", "negative"}:
                raise ValueError("live Persuasion candidate has no supported positive or negative signal polarity")
            quality = str(state.get("current_quality") or "").casefold()
            candidate_coordinates = {"family": family, "phase": phase, "kind": "signal", "action_label": f"signal_{polarity}", "action_value": 1.0 if polarity == "positive" else -1.0, "action_aux_value": None, "round_number": round_number, "round_phase": round_phase, "visible_quality": quality if quality in {"high", "low"} else None}
        else:
            raise ValueError("live candidate is outside the direct-response conditional frontier")
        candidate = cls(**candidate_coordinates, **cls._message_fields(_message_features(message, family_act=family_act)))
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if self.family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported candidate family: {self.family}")
        if not self.phase:
            raise ValueError("candidate phase cannot be empty")
        if self.round_number < 1 or not 0.0 <= self.round_phase <= 1.0:
            raise ValueError("candidate round coordinates are invalid")
        if self.family in {"bargaining", "negotiation"}:
            if self.kind != "proposal" or self.action_label != "proposal" or self.action_value is None:
                raise ValueError(f"{self.family} candidate must be a numeric proposal")
        elif self.kind != "signal" or self.action_label not in {"signal_positive", "signal_negative"}:
            raise ValueError("persuasion candidate must be a supported signal")
        if self.family == "persuasion":
            expected = 1.0 if self.action_label == "signal_positive" else -1.0
            if self.action_value is None or not math.isclose(self.action_value, expected, rel_tol=0.0, abs_tol=1e-9) or self.action_aux_value is not None:
                raise ValueError("persuasion signal value disagrees with its action label")
        if any(value < 0 or value >= HASH_BINS for value in self.message_hash_bins):
            raise ValueError("candidate message hash bin is outside the frozen vocabulary")
        integer_fields = (self.message_chars, self.message_words, self.message_question_marks, self.message_exclamation_marks, self.message_commas, self.message_semicolons, self.message_currency_marks, self.message_percent_marks, self.message_decimal_numbers)
        if any(value < 0 for value in integer_fields) or not all(math.isfinite(value) and value >= 0.0 for value in (self.message_uppercase_ratio, self.message_digit_ratio)):
            raise ValueError("candidate message statistics are invalid")
        if self.message_present and not self.message_sha256:
            raise ValueError("present candidate message has no content fingerprint")

    def message_receipt(self) -> dict[str, object]:
        return {
            "message_present": self.message_present,
            "message_family_act": self.message_family_act,
            "message_discourse_acts": list(self.message_discourse_acts),
            "message_hash_bins": list(self.message_hash_bins),
            "message_chars": self.message_chars,
            "message_words": self.message_words,
            "message_uppercase_ratio": self.message_uppercase_ratio,
            "message_digit_ratio": self.message_digit_ratio,
            "message_question_marks": self.message_question_marks,
            "message_exclamation_marks": self.message_exclamation_marks,
            "message_commas": self.message_commas,
            "message_semicolons": self.message_semicolons,
            "message_currency_marks": self.message_currency_marks,
            "message_percent_marks": self.message_percent_marks,
            "message_decimal_numbers": self.message_decimal_numbers,
            "message_sha256": self.message_sha256,
        }

    def event(self, *, game_id: str, event_index: int) -> dict[str, object]:
        self.validate()
        return {
            "game_id": game_id,
            "event_index": event_index,
            "round_number": self.round_number,
            "round_phase": self.round_phase,
            "actor": "self",
            "kind": self.kind,
            "action_label": self.action_label,
            "action_value": self.action_value,
            "action_aux_value": self.action_aux_value,
            "response_time_ms": None,
            "visible_quality": self.visible_quality,
            **self.message_receipt(),
        }

    def receipt(self) -> dict[str, object]:
        return {
            "family": self.family,
            "phase": self.phase,
            "kind": self.kind,
            "action_label": self.action_label,
            "action_value": self.action_value,
            "action_aux_value": self.action_aux_value,
            "round_number": self.round_number,
            "round_phase": self.round_phase,
            "visible_quality": self.visible_quality,
            **self.message_receipt(),
        }

    def public_receipt(self) -> dict[str, object]:
        return {
            "family": self.family,
            "phase": self.phase,
            "kind": self.kind,
            "action_label": self.action_label,
            "action_value": self.action_value,
            "action_aux_value": self.action_aux_value,
            "round_number": self.round_number,
            "round_phase": self.round_phase,
            "visible_quality": self.visible_quality,
            "message_present": self.message_present,
            "message_family_act": self.message_family_act,
            "message_discourse_acts": list(self.message_discourse_acts),
            "message_chars": self.message_chars,
            "message_words": self.message_words,
            "message_sha256": self.message_sha256,
        }


@dataclass(frozen=True)
class CandidateFeatureVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]
    vector_sha256: str


class CandidateActionProjector:
    """Hash one legal candidate action into the frozen 2,048-dimensional feature space."""

    dimension = FEATURE_DIMENSION

    @staticmethod
    def _numeric_transform(value: float) -> float:
        return math.copysign(min(4.0, math.log1p(abs(value)) / 8.0), value)

    def project(self, candidate: CandidateAction) -> CandidateFeatureVector:
        candidate.validate()
        merged: defaultdict[int, float] = defaultdict(float)
        categorical = {
            "family": candidate.family,
            "phase": candidate.phase,
            "kind": candidate.kind,
            "action_label": candidate.action_label,
            "visible_quality": candidate.visible_quality if candidate.visible_quality is not None else "<null>",
            "action_value_present": str(candidate.action_value is not None).casefold(),
            "action_aux_value_present": str(candidate.action_aux_value is not None).casefold(),
            "message_present": str(candidate.message_present).casefold(),
            "message_family_act": candidate.message_family_act,
        }
        for name, value in categorical.items():
            path = f"candidate_action.{_path_segment(name)}"
            merged[_stable_bin(f"{path}={_path_segment(value)}", CATEGORICAL_BINS)] += math.log(2.0)
        discourse_counts = Counter(candidate.message_discourse_acts)
        for value, count in discourse_counts.items():
            merged[_stable_bin(f"candidate_message.discourse={_path_segment(value)}", CATEGORICAL_BINS)] += math.log1p(count)
        lexeme_counts = Counter(candidate.message_hash_bins)
        for value, count in lexeme_counts.items():
            merged[_stable_bin(f"candidate_message.lexeme_bin={value}", CATEGORICAL_BINS)] += math.log1p(count)
        numeric = {
            "action_value": candidate.action_value,
            "action_aux_value": candidate.action_aux_value,
            "round_number": float(candidate.round_number),
            "round_phase": candidate.round_phase,
            "message_chars": float(candidate.message_chars),
            "message_words": float(candidate.message_words),
            "message_uppercase_ratio": candidate.message_uppercase_ratio,
            "message_digit_ratio": candidate.message_digit_ratio,
            "message_question_marks": float(candidate.message_question_marks),
            "message_exclamation_marks": float(candidate.message_exclamation_marks),
            "message_commas": float(candidate.message_commas),
            "message_semicolons": float(candidate.message_semicolons),
            "message_currency_marks": float(candidate.message_currency_marks),
            "message_percent_marks": float(candidate.message_percent_marks),
            "message_decimal_numbers": float(candidate.message_decimal_numbers),
        }
        for name, value in numeric.items():
            if value is None:
                continue
            path = f"candidate_action.{_path_segment(name)}"
            numeric_index = _stable_bin(path, NUMERIC_BINS)
            merged[CATEGORICAL_BINS + numeric_index] += self._numeric_transform(value)
            merged[CATEGORICAL_BINS + NUMERIC_BINS + _stable_bin(path, PRESENCE_BINS)] += math.log(2.0)
        ordered = tuple(sorted((index, float(value)) for index, value in merged.items() if value != 0.0))
        payload = {"indices": [index for index, _value in ordered], "values": [value for _index, value in ordered]}
        return CandidateFeatureVector(indices=tuple(index for index, _value in ordered), values=tuple(value for _index, value in ordered), vector_sha256=object_sha256(payload))

    def receipt(self) -> dict[str, object]:
        return {
            "contract": CANDIDATE_ACTION_PROJECTION_CONTRACT,
            "dimension": self.dimension,
            "partitions": {"categorical": CATEGORICAL_BINS, "numeric": NUMERIC_BINS, "numeric_presence": PRESENCE_BINS},
            "categorical_fields": ["family", "phase", "kind", "action_label", "visible_quality", "action_value_present", "action_aux_value_present", "message_present", "message_family_act", "message_discourse_acts", "message_hash_bins"],
            "numeric_fields": ["action_value", "action_aux_value", "round_number", "round_phase", "message_chars", "message_words", "message_uppercase_ratio", "message_digit_ratio", "message_question_marks", "message_exclamation_marks", "message_commas", "message_semicolons", "message_currency_marks", "message_percent_marks", "message_decimal_numbers"],
            "numeric_transform": "sign(x) * min(4, log1p(abs(x)) / 8)",
        }


def merge_sparse_vectors(base_indices: Sequence[int], base_values: Sequence[float], candidate: CandidateFeatureVector) -> CandidateFeatureVector:
    if len(base_indices) != len(base_values) or len(set(int(index) for index in base_indices)) != len(base_indices):
        raise ValueError("invalid base engineered-feature vector")
    merged: defaultdict[int, float] = defaultdict(float)
    for raw_index, raw_value in zip(base_indices, base_values, strict=True):
        index = int(raw_index)
        value = float(raw_value)
        if index < 0 or index >= FEATURE_DIMENSION or not math.isfinite(value):
            raise ValueError("base engineered-feature vector is out of bounds or nonfinite")
        merged[index] += value
    for index, value in zip(candidate.indices, candidate.values, strict=True):
        merged[index] += value
    ordered = tuple(sorted((index, float(value)) for index, value in merged.items() if value != 0.0))
    payload = {"indices": [index for index, _value in ordered], "values": [value for _index, value in ordered]}
    return CandidateFeatureVector(indices=tuple(index for index, _value in ordered), values=tuple(value for _index, value in ordered), vector_sha256=object_sha256(payload))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, object]:
    frame = pl.read_parquet(path)
    return {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size, "rows": frame.height, "columns": frame.width}


def _hardlink(source: Path, destination: Path) -> None:
    os.link(source, destination)
    if file_sha256(source) != file_sha256(destination):
        raise RuntimeError(f"hard-linked artifact differs from its source: {source}")


class ConditionalCorpusBuilder:
    """Add the observed self-action bridge while masking fields unavailable before Terra."""

    def __init__(self, *, source_corpus: Path, output_dir: Path) -> None:
        self.source_corpus = source_corpus.resolve()
        self.output_dir = output_dir.resolve()

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"conditional corpus output already exists: {self.output_dir}")
        source_manifest_path = self.source_corpus / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("contract") != PRE_TERRA_V3_CORPUS_CONTRACT or source_manifest.get("status") != "frozen-retrospective-core-corpus":
            raise ValueError("conditional corpus requires the frozen pre-Terra v3 core corpus")
        source_artifacts = source_manifest.get("artifacts")
        if not isinstance(source_artifacts, Mapping):
            raise ValueError("source core corpus has no artifact receipts")
        for name in ("games.parquet", "events.parquet", "targets.parquet", "features.parquet"):
            path = self.source_corpus / name
            receipt = source_artifacts.get(name)
            if not path.is_file() or not isinstance(receipt, Mapping) or file_sha256(path) != receipt.get("sha256"):
                raise RuntimeError(f"source core corpus artifact hash mismatch: {name}")
        games = pl.read_parquet(self.source_corpus / "games.parquet")
        events = pl.read_parquet(self.source_corpus / "events.parquet")
        targets = pl.read_parquet(self.source_corpus / "targets.parquet")
        features = pl.read_parquet(self.source_corpus / "features.parquet")
        game_by_id = {str(row["game_id"]): row for row in games.to_dicts()}
        event_by_coordinate = {(str(row["game_id"]), int(row["event_index"])): row for row in events.to_dicts()}
        feature_by_sample = {str(row["sample_id"]): row for row in features.to_dicts()}
        if len(game_by_id) != games.height or len(event_by_coordinate) != events.height or len(feature_by_sample) != features.height or targets["sample_id"].n_unique() != targets.height:
            raise RuntimeError("source core corpus has duplicate coordinates")
        projector = CandidateActionProjector()
        conditioned_targets: list[dict[str, object]] = []
        conditioned_features: list[dict[str, object]] = []
        by_family: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        exclusions: Counter[str] = Counter()
        for target in targets.to_dicts():
            sample_id = str(target["sample_id"])
            game_id = str(target["game_id"])
            game = game_by_id.get(game_id)
            feature = feature_by_sample.get(sample_id)
            if game is None or feature is None or str(feature.get("game_id")) != game_id:
                raise RuntimeError("source target has no unique game or feature row")
            family = str(game["family"])
            target_index = int(target["target_event_index"])
            original_prefix_length = int(target["prefix_length"])
            bridge_index = target_index - 1
            bridge = event_by_coordinate.get((game_id, bridge_index))
            response = event_by_coordinate.get((game_id, target_index))
            if bridge_index != original_prefix_length or bridge is None or response is None:
                raise RuntimeError("source target does not have one withheld bridge event")
            if bridge.get("actor") != "self" or response.get("actor") != "opponent" or response.get("kind") != "response" or response.get("action_label") != target.get("target_label"):
                raise RuntimeError("source bridge-response coordinates violate the conditional contract")
            if str(target.get("target_label")) not in CONDITIONAL_TARGET_LABELS[family]:
                raise RuntimeError(f"unsupported direct response label for {family}: {target.get('target_label')!r}")
            if family == "persuasion" and bridge.get("action_label") not in {"signal_positive", "signal_negative"}:
                exclusions[f"unsupported-candidate:{family}:{bridge.get('action_label')}"] += 1
                continue
            candidate = CandidateAction.from_event(family=family, phase=str(feature.get("phase") or ""), event=bridge)
            candidate_vector = projector.project(candidate)
            base_indices = [int(value) for value in feature["feature_indices"]]
            base_values = [float(value) for value in feature["feature_values"]]
            if object_sha256({"indices": base_indices, "values": base_values}) != feature.get("feature_vector_sha256"):
                raise RuntimeError("source engineered-feature vector hash mismatch")
            conditioned_vector = merge_sparse_vectors(base_indices, base_values, candidate_vector)
            conditioned_targets.append(
                {
                    **target,
                    "pre_candidate_prefix_length": original_prefix_length,
                    "bridge_event_index": bridge_index,
                    "prefix_length": target_index,
                    "mask_last_prefix_future_fields": False,
                    "mask_last_prefix_response_time": True,
                    "mask_last_prefix_message_fields": False,
                }
            )
            conditioned_features.append(
                {
                    **feature,
                    "candidate_kind": candidate.kind,
                    "candidate_action_label": candidate.action_label,
                    "candidate_action_value": candidate.action_value,
                    "candidate_action_aux_value": candidate.action_aux_value,
                    "candidate_round_number": candidate.round_number,
                    "candidate_round_phase": candidate.round_phase,
                    "candidate_visible_quality": candidate.visible_quality,
                    **{f"candidate_{name}": value for name, value in candidate.message_receipt().items()},
                    "candidate_feature_indices": list(candidate_vector.indices),
                    "candidate_feature_values": list(candidate_vector.values),
                    "candidate_feature_vector_sha256": candidate_vector.vector_sha256,
                    "conditioned_feature_vector_sha256": conditioned_vector.vector_sha256,
                }
            )
            by_family[family] += 1
            action_counts[f"{family}:{candidate.action_label}"] += 1
        if not conditioned_targets or len(conditioned_targets) != len(conditioned_features) or len(conditioned_targets) + sum(exclusions.values()) != targets.height:
            raise RuntimeError("conditional construction did not account for every source row")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            _hardlink(self.source_corpus / "games.parquet", staging / "games.parquet")
            _hardlink(self.source_corpus / "events.parquet", staging / "events.parquet")
            pl.DataFrame(conditioned_targets, infer_schema_length=None).sort(["game_id", "target_event_index"]).write_parquet(staging / "targets.parquet", compression="zstd", compression_level=7, statistics=True)
            pl.DataFrame(conditioned_features, infer_schema_length=None).sort(["game_id", "round_number", "turn_id"]).write_parquet(staging / "features.parquet", compression="zstd", compression_level=7, statistics=True)
            artifacts = {name: _artifact(staging / name) for name in ("games.parquet", "events.parquet", "targets.parquet", "features.parquet")}
            manifest = {
                "schema_version": 1,
                "contract": CONDITIONAL_CORPUS_CONTRACT,
                "status": "frozen-retrospective-core-corpus",
                "source_corpus": {"path": str(self.source_corpus), "manifest_sha256": file_sha256(source_manifest_path), "artifacts": source_artifacts},
                "target_labels": {family: list(CONDITIONAL_TARGET_LABELS[family]) for family in GLEE_FAMILIES},
                "candidate_action_projection": projector.receipt(),
                "inventory": {
                    "games": games.height,
                    "events": events.height,
                    "targets": len(conditioned_targets),
                    "features": len(conditioned_features),
                    "by_family": dict(sorted(by_family.items())),
                    "by_candidate_action": dict(sorted(action_counts.items())),
                    "by_split": dict(sorted(Counter(str(row["chronological_split"]) for row in conditioned_targets).items())),
                    "source_targets": targets.height,
                    "exclusions": dict(sorted(exclusions.items())),
                },
                "invariants": {
                    "historical_bridge": "target_event_index - 1",
                    "prefix_extension_events": 1,
                    "bridge_actor": "self",
                    "target_actor": "opponent",
                    "target_kind": "response",
                    "masked_bridge_fields": ["response_time_ms"],
                    "visible_bridge_fields": ["message_*"],
                    "earlier_authenticated_messages_and_timings_visible": True,
                    "candidate_wording_available_after_planner": True,
                    "whole_game_splits_preserved": True,
                },
                "artifacts": artifacts,
                "implementation_sha256": file_sha256(Path(__file__)),
                "boundary": "Retrospective observed-action support only; each input includes the actual post-planner self bridge with candidate polarity and wording visible, response latency masked, and the direct authenticated opponent response as label.",
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
            return {"contract": CONDITIONAL_CORPUS_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json"), "inventory": manifest["inventory"]}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
