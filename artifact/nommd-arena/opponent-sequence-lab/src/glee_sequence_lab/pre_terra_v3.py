"""Build exact historical pre-Terra feature-fusion corpora without copying prompt bodies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from nommd_arena.glee_sequence_shadow import SEQUENCE_SHADOW_FEATURE_CONTRACT, terra_synthetic_feature_bundle
from nommd_arena.immutable_blob import load_referenced_text_blob

from .corpus import EVENT_SCHEMA, GAME_SCHEMA, GLEE_FAMILIES, TARGET_SCHEMA, extract_events, file_sha256, load_account_map, normalized_label, object_sha256
from .live_shadow import _static_live_game, build_pre_terra_opportunity
from .shadow import ShadowCandidate


PRE_TERRA_V3_CORPUS_CONTRACT = "glee-pre-terra-feature-fusion-corpus-v3"
FEATURE_PROJECTION_CONTRACT = "glee-terra-engineered-feature-hash-v1"
FEATURE_DIMENSION = 2_048
CATEGORICAL_BINS = 1_024
NUMERIC_BINS = 512
PRESENCE_BINS = 512
MAX_FEATURE_LEAVES = 8_192
MAX_CATEGORICAL_CHARS = 160

_PROVENANCE_NAMES = frozenset(
    {
        "contract",
        "frontier",
        "revision",
        "rowid",
        "path",
        "receipt",
        "receipts",
        "timestamp",
        "ts",
        "generated_at",
        "updated_at",
        "created_at",
        "observed_at",
        "release",
        "implementation",
        "request_id",
    }
)
_PROVENANCE_FRAGMENTS = ("sha256", "checksum", "manifest_hash", "state_hash", "semantic_hash", "revision", "implementation", "version")


def _parse_time(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is absent")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _stable_bin(value: str, width: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % width


def _path_segment(value: object) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(value).casefold()).strip("_") or "empty"


def _skip_field(name: object) -> bool:
    normalized = _path_segment(name)
    return normalized in _PROVENANCE_NAMES or normalized.endswith(("_timestamp", "_ts", "_at")) or any(fragment in normalized for fragment in _PROVENANCE_FRAGMENTS)


@dataclass(frozen=True)
class HashedFeatureVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]
    leaf_count: int
    dropped_leaf_count: int
    excluded_provenance_count: int
    excluded_long_text_count: int
    vector_sha256: str


class EngineeredFeatureHasher:
    """Project evolving structured advisor payloads into one bounded deterministic sparse vector."""

    def __init__(self, *, dimension: int = FEATURE_DIMENSION, maximum_leaves: int = MAX_FEATURE_LEAVES) -> None:
        if dimension != CATEGORICAL_BINS + NUMERIC_BINS + PRESENCE_BINS:
            raise ValueError("feature dimension disagrees with the frozen hash partition")
        if maximum_leaves < 1:
            raise ValueError("maximum feature leaves must be positive")
        self.dimension = dimension
        self.maximum_leaves = maximum_leaves

    def project(self, bundle: Mapping[str, object]) -> HashedFeatureVector:
        if bundle.get("contract") != SEQUENCE_SHADOW_FEATURE_CONTRACT:
            raise ValueError("unsupported engineered-feature bundle")
        categorical: defaultdict[int, int] = defaultdict(int)
        numeric_sum: defaultdict[int, float] = defaultdict(float)
        numeric_count: defaultdict[int, int] = defaultdict(int)
        leaf_count = 0
        dropped = 0
        excluded_provenance = 0
        excluded_long_text = 0

        def categorical_leaf(path: str, value: str) -> None:
            nonlocal leaf_count, dropped
            if leaf_count >= self.maximum_leaves:
                dropped += 1
                return
            leaf_count += 1
            categorical[_stable_bin(f"{path}={value}", CATEGORICAL_BINS)] += 1

        def numeric_leaf(path: str, value: float) -> None:
            nonlocal leaf_count, dropped
            if leaf_count >= self.maximum_leaves:
                dropped += 1
                return
            leaf_count += 1
            transformed = math.copysign(min(4.0, math.log1p(abs(value)) / 8.0), value)
            index = _stable_bin(path, NUMERIC_BINS)
            numeric_sum[index] += transformed
            numeric_count[index] += 1

        def visit(value: object, path: str) -> None:
            nonlocal excluded_provenance, excluded_long_text, dropped
            if isinstance(value, Mapping):
                categorical_leaf(path, "<object>")
                for raw_key in sorted(value, key=lambda item: str(item)):
                    if _skip_field(raw_key):
                        excluded_provenance += 1
                        continue
                    visit(value[raw_key], f"{path}.{_path_segment(raw_key)}")
                return
            if isinstance(value, list | tuple):
                numeric_leaf(f"{path}.length", float(len(value)))
                for index, item in enumerate(value):
                    if leaf_count >= self.maximum_leaves:
                        dropped += len(value) - index
                        break
                    bucket = str(index) if index < 32 else "32_plus"
                    visit(item, f"{path}[{bucket}]")
                return
            if value is None:
                categorical_leaf(path, "<null>")
                return
            if isinstance(value, bool):
                categorical_leaf(path, "true" if value else "false")
                return
            if isinstance(value, int | float) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    numeric_leaf(path, number)
                else:
                    categorical_leaf(path, "<nonfinite>")
                return
            text = re.sub(r"\s+", " ", str(value)).strip().casefold()
            if len(text) > MAX_CATEGORICAL_CHARS:
                excluded_long_text += 1
                return
            categorical_leaf(path, text or "<empty>")

        features = bundle.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("engineered-feature bundle has no features object")
        visit(bundle.get("game_family"), "game_family")
        visit(bundle.get("phase"), "phase")
        visit(features, "features")
        merged: defaultdict[int, float] = defaultdict(float)
        for index, count in categorical.items():
            merged[index] += math.log1p(count)
        for index, total in numeric_sum.items():
            merged[CATEGORICAL_BINS + index] += total / numeric_count[index]
            presence = _stable_bin(str(index), PRESENCE_BINS)
            merged[CATEGORICAL_BINS + NUMERIC_BINS + presence] += math.log1p(numeric_count[index])
        ordered = tuple(sorted((index, value) for index, value in merged.items() if value != 0.0))
        vector_payload = {"indices": [index for index, _value in ordered], "values": [value for _index, value in ordered]}
        return HashedFeatureVector(
            indices=tuple(index for index, _value in ordered),
            values=tuple(float(value) for _index, value in ordered),
            leaf_count=leaf_count,
            dropped_leaf_count=dropped,
            excluded_provenance_count=excluded_provenance,
            excluded_long_text_count=excluded_long_text,
            vector_sha256=object_sha256(vector_payload),
        )

    def receipt(self) -> dict[str, object]:
        return {
            "contract": FEATURE_PROJECTION_CONTRACT,
            "dimension": self.dimension,
            "partitions": {"categorical": CATEGORICAL_BINS, "numeric": NUMERIC_BINS, "numeric_presence": PRESENCE_BINS},
            "maximum_leaves": self.maximum_leaves,
            "maximum_categorical_chars": MAX_CATEGORICAL_CHARS,
            "provenance_names": sorted(_PROVENANCE_NAMES),
            "provenance_fragments": list(_PROVENANCE_FRAGMENTS),
            "numeric_transform": "sign(x) * min(4, log1p(abs(x)) / 8)",
        }


@dataclass(frozen=True)
class ArchivedPromptSnapshot:
    source_run: Path
    line_number: int
    role: str
    call_ts: str
    turn_id: str
    family: str
    user_sha256: str
    bundle: dict[str, object]
    bundle_sha256: str


def _load_call_user_text(*, calls_path: Path, request: Mapping[str, object]) -> str:
    reference = request.get("user_ref")
    if isinstance(reference, Mapping):
        return load_referenced_text_blob(receipt_dir=calls_path.parent, reference=dict(reference))
    inline = request.get("user")
    if isinstance(inline, str):
        return inline
    raise ValueError("cloud call has neither a referenced nor inline user payload")


def _source_run_receipt(source_run: Path, *, prompt_turns: int) -> dict[str, object]:
    required = (source_run / "manifest.json", source_run / "llm_calls.jsonl", source_run / "events.jsonl", source_run / "games")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"source run lacks required historical evidence: {source_run}")
    receipt: dict[str, object] = {
        "path": str(source_run),
        "manifest_sha256": file_sha256(source_run / "manifest.json"),
        "llm_calls_sha256": file_sha256(source_run / "llm_calls.jsonl"),
        "events_sha256": file_sha256(source_run / "events.jsonl"),
        "prompt_turns": prompt_turns,
    }
    complete_path = source_run / "complete.json"
    if complete_path.is_file():
        receipt.update({"source_state": "sealed-complete-receipt", "complete_sha256": file_sha256(complete_path)})
        return receipt
    archive_index: list[dict[str, object]] = []
    for archive_path in sorted((source_run / "games").glob("*.json")):
        try:
            game = json.loads(archive_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"unsealed source has an unreadable game archive: {archive_path}") from exc
        if game.get("status") not in {"completed", "no_deal"}:
            raise ValueError(f"unsealed source contains a nonterminal game archive: {archive_path}")
        archive_index.append({"path": archive_path.name, "sha256": file_sha256(archive_path), "status": game.get("status")})
    if not archive_index:
        raise ValueError(f"unsealed source has no terminal game archives: {source_run}")
    receipt.update(
        {
            "source_state": "explicit-recovered-terminal-game-set",
            "complete_sha256": None,
            "terminal_game_archives": len(archive_index),
            "terminal_game_archive_index_sha256": object_sha256(archive_index),
        }
    )
    return receipt


def _read_prompt_snapshots(source_run: Path, exclusions: Counter[str]) -> dict[str, ArchivedPromptSnapshot]:
    calls_path = source_run / "llm_calls.jsonl"
    grouped: defaultdict[str, list[ArchivedPromptSnapshot]] = defaultdict(list)
    with calls_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                role = str(row.get("role") or "")
                if role not in {"glee_nommd_bargaining", "glee_nommd_negotiation", "glee_nommd_persuasion"}:
                    continue
                request = row.get("request")
                if not isinstance(request, Mapping):
                    exclusions["call-user-reference-missing"] += 1
                    continue
                try:
                    user_text = _load_call_user_text(calls_path=calls_path, request=request)
                except (OSError, TypeError, ValueError, KeyError):
                    exclusions["call-user-reference-missing"] += 1
                    continue
                user_sha256 = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
                if user_sha256 != request.get("user_sha256"):
                    exclusions["call-user-hash-mismatch"] += 1
                    continue
                payload = json.loads(user_text)
                if not isinstance(payload, Mapping):
                    exclusions["call-user-not-object"] += 1
                    continue
                receipt = payload.get("turn_receipt")
                turn_id = str(receipt.get("turn_id") or "") if isinstance(receipt, Mapping) else ""
                family = str(payload.get("game_family") or "")
                if not turn_id or family not in GLEE_FAMILIES or role != f"glee_nommd_{family}":
                    exclusions["call-turn-identity-invalid"] += 1
                    continue
                bundle = terra_synthetic_feature_bundle(payload)
                snapshot = ArchivedPromptSnapshot(
                    source_run=source_run,
                    line_number=line_number,
                    role=role,
                    call_ts=str(row.get("ts") or ""),
                    turn_id=turn_id,
                    family=family,
                    user_sha256=user_sha256,
                    bundle=bundle,
                    bundle_sha256=object_sha256(bundle),
                )
                grouped[turn_id].append(snapshot)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                exclusions["call-unreadable"] += 1
    selected: dict[str, ArchivedPromptSnapshot] = {}
    for turn_id, snapshots in grouped.items():
        hashes = {snapshot.bundle_sha256 for snapshot in snapshots}
        if len(hashes) != 1:
            exclusions["turn-feature-conflict"] += 1
            continue
        selected[turn_id] = min(snapshots, key=lambda snapshot: (snapshot.line_number, snapshot.call_ts))
        exclusions["duplicate-call-collapsed"] += len(snapshots) - 1
    return selected


@dataclass
class RunEvidence:
    observed_games: dict[str, dict[str, Any]]
    observed_ts: dict[str, str]
    submitted_ts: dict[str, str]
    completed_ts: dict[str, str]
    started_ts: dict[str, str]


def _read_run_evidence(source_run: Path, turn_ids: set[str], exclusions: Counter[str]) -> RunEvidence:
    observed_games: dict[str, dict[str, Any]] = {}
    observed_ts: dict[str, str] = {}
    submitted_ts: dict[str, str] = {}
    completed_ts: dict[str, str] = {}
    started_ts: dict[str, str] = {}
    with (source_run / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                exclusions["event-unreadable"] += 1
                continue
            kind = row.get("kind")
            turn_id = str(row.get("turn_id") or "")
            game_id = str(row.get("game_id") or "")
            if kind == "turn_observed" and turn_id in turn_ids:
                game = row.get("game")
                if not isinstance(game, dict):
                    exclusions["turn-observation-game-missing"] += 1
                    continue
                prior = observed_games.get(turn_id)
                if prior is not None and object_sha256(prior) != object_sha256(game):
                    exclusions["turn-observation-conflict"] += 1
                    observed_games.pop(turn_id, None)
                    observed_ts.pop(turn_id, None)
                    continue
                observed_games[turn_id] = game
                observed_ts[turn_id] = str(row.get("ts") or "")
            elif kind == "move_submitted" and turn_id in turn_ids:
                submitted_ts[turn_id] = str(row.get("ts") or "")
            elif kind in {"game_completed", "game_completed_during_opponent_turn"} and game_id:
                completed_ts[game_id] = str(row.get("ts") or "")
            elif kind == "activity_game_started" and game_id:
                started_ts.setdefault(game_id, str(row.get("ts") or ""))
    return RunEvidence(observed_games=observed_games, observed_ts=observed_ts, submitted_ts=submitted_ts, completed_ts=completed_ts, started_ts=started_ts)


def _artifact(frame: pl.DataFrame, path: Path) -> dict[str, object]:
    frame.write_parquet(path, compression="zstd", compression_level=7, statistics=True, row_group_size=4_096)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item(0, 0)
    if rows != frame.height:
        raise RuntimeError(f"Parquet row-count mismatch: {path}")
    return {"path": path.name, "rows": frame.height, "columns": frame.width, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chronological_splits(game_rows: Sequence[Mapping[str, object]], *, train_fraction: float, validation_fraction: float) -> dict[str, str]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("invalid chronological split fractions")
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in game_rows:
        grouped[str(row["family"])].append(row)
    assignments: dict[str, str] = {}
    for family in GLEE_FAMILIES:
        ordered = sorted(grouped[family], key=lambda row: (str(row["completed_at"]), str(row["game_id"])))
        train_end = int(len(ordered) * train_fraction)
        validation_end = int(len(ordered) * (train_fraction + validation_fraction))
        for index, row in enumerate(ordered):
            assignments[str(row["game_id"])] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return assignments


class PreTerraV3CorpusBuilder:
    """Freeze exact pre-Terra snapshots and optionally attach an aligned sequence release."""

    def __init__(self, *, source_runs: Sequence[Path], sequence_release: Path | None, reference_core_corpus: Path | None, account_groups: Path, output_dir: Path, train_fraction: float = 0.70, validation_fraction: float = 0.15, prediction_batch_size: int = 256) -> None:
        self.source_runs = tuple(sorted({path.resolve() for path in source_runs}, key=str))
        self.sequence_release = sequence_release.resolve() if sequence_release is not None else None
        self.reference_core_corpus = reference_core_corpus.resolve() if reference_core_corpus is not None else None
        self.account_groups = account_groups.resolve()
        self.output_dir = output_dir.resolve()
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.prediction_batch_size = prediction_batch_size

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"corpus output already exists: {self.output_dir}")
        if not self.source_runs:
            raise ValueError("v3 protocol requires at least one frozen source run")
        if (self.sequence_release is None) != (self.reference_core_corpus is None):
            raise ValueError("an aligned sequence release and reference core corpus must be supplied together")
        if self.sequence_release is not None and self.prediction_batch_size < 8:
            raise ValueError("sequence prediction batch size must be at least 8")
        exclusions: Counter[str] = Counter()
        hasher = EngineeredFeatureHasher()
        account_map, account_collisions = load_account_map(self.account_groups)
        game_rows_by_id: dict[str, dict[str, object]] = {}
        event_rows_by_game: dict[str, list[dict[str, object]]] = {}
        target_rows: list[dict[str, object]] = []
        feature_rows: list[dict[str, object]] = []
        source_receipts: list[dict[str, object]] = []
        samples_by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        sample_ids_by_family: defaultdict[str, list[str]] = defaultdict(list)
        seen_turn_ids: set[str] = set()
        for source_run in self.source_runs:
            prompts = _read_prompt_snapshots(source_run, exclusions)
            evidence = _read_run_evidence(source_run, set(prompts), exclusions)
            source_receipts.append(_source_run_receipt(source_run, prompt_turns=len(prompts)))
            for turn_id, prompt in sorted(prompts.items(), key=lambda item: (item[1].call_ts, item[0])):
                if turn_id in seen_turn_ids:
                    exclusions["cross-run-turn-duplicate"] += 1
                    continue
                observed_game = evidence.observed_games.get(turn_id)
                observed_at = evidence.observed_ts.get(turn_id)
                submitted_at = evidence.submitted_ts.get(turn_id)
                if observed_game is None or not observed_at or not submitted_at:
                    exclusions["turn-lifecycle-incomplete"] += 1
                    continue
                try:
                    if not (_parse_time(observed_at) <= _parse_time(prompt.call_ts) <= _parse_time(submitted_at)):
                        exclusions["call-outside-observe-submit-frontier"] += 1
                        continue
                    opportunity = build_pre_terra_opportunity(observed_game, turn_id=turn_id, synthetic_features=prompt.bundle)
                except (TypeError, ValueError, KeyError, OverflowError):
                    exclusions["pre-terra-opportunity-invalid"] += 1
                    continue
                if opportunity is None:
                    exclusions["turn-not-direct-response-opportunity"] += 1
                    continue
                game_id = opportunity.game_id
                completed_at = evidence.completed_ts.get(game_id)
                archive_path = source_run / "games" / f"{prompt.family}-{game_id}.json"
                if not completed_at or not archive_path.is_file():
                    exclusions["terminal-game-missing"] += 1
                    continue
                try:
                    final_game = json.loads(archive_path.read_text(encoding="utf-8"))
                    if final_game.get("status") not in {"completed", "no_deal"} or final_game.get("game_id") != game_id or final_game.get("game_family") != prompt.family:
                        exclusions["terminal-game-invalid"] += 1
                        continue
                    final_events = extract_events(final_game)
                    prefix = list(opportunity.prefix_events)
                    bridge_index = len(prefix)
                    target_index = opportunity.target_event_index
                    if final_events[:bridge_index] != prefix:
                        exclusions["terminal-prefix-mismatch"] += 1
                        continue
                    if target_index != bridge_index + 1 or target_index >= len(final_events):
                        exclusions["terminal-target-absent"] += 1
                        continue
                    bridge = final_events[bridge_index]
                    target_event = final_events[target_index]
                    if bridge.get("actor") != "self" or bridge.get("kind") not in {"proposal", "signal"}:
                        exclusions["causal-bridge-invalid"] += 1
                        continue
                    if target_event.get("actor") != "opponent" or target_event.get("kind") != "response":
                        exclusions["opponent-response-invalid"] += 1
                        continue
                    vector = hasher.project(prompt.bundle)
                except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError, OverflowError):
                    exclusions["sample-construction-invalid"] += 1
                    continue
                sample_id = hashlib.sha256(f"pre-terra-v3:{turn_id}".encode("utf-8")).hexdigest()
                static_game = dict(opportunity.sample["game"])
                opponent = final_game.get("opponent") if isinstance(final_game.get("opponent"), Mapping) else {}
                opponent_name = normalized_label(opponent.get("name")) if static_game["identity_scope"] == "known" else ""
                account = account_map.get(opponent_name)
                account_key = account[0] if account else None
                account_confidence = account[1] if account else None
                account_fold = int(hashlib.sha256(account_key.encode("utf-8")).hexdigest()[:8], 16) % 5 if account_key else -1
                static_game.update(
                    {
                            "source_type": "real",
                        "started_at": evidence.started_ts.get(game_id) or observed_at,
                        "completed_at": completed_at,
                        "chronological_split": "pending",
                        "account_key": account_key,
                        "account_confidence": account_confidence,
                        "account_fold": account_fold,
                        "engine_version": "pre-terra-rating-v3",
                        "archive_path": str(archive_path.relative_to(source_run.parent)),
                        "archive_sha256": file_sha256(archive_path),
                    }
                )
                if game_id not in game_rows_by_id:
                    game_rows_by_id[game_id] = static_game
                    event_rows_by_game[game_id] = final_events
                elif event_rows_by_game[game_id] != final_events or game_rows_by_id[game_id]["family"] != static_game["family"]:
                    raise RuntimeError(f"historical game identity conflicts across source runs: {game_id}")
                target_row = {
                    "sample_id": sample_id,
                    "game_id": game_id,
                    "source_type": "real",
                    "target_event_index": target_index,
                    "prefix_length": bridge_index,
                    "target_kind": "response",
                    "target_label": str(target_event["action_label"]),
                    "target_value": None,
                    "target_value_present": False,
                    "target_message_act": None,
                    "target_message_present": False,
                    "target_delay_log_ms": math.log1p(float(target_event["response_time_ms"])) if isinstance(target_event.get("response_time_ms"), int | float) and not isinstance(target_event.get("response_time_ms"), bool) and float(target_event["response_time_ms"]) >= 0 else None,
                    "target_delay_present": isinstance(target_event.get("response_time_ms"), int | float) and not isinstance(target_event.get("response_time_ms"), bool) and float(target_event["response_time_ms"]) >= 0,
                    "chronological_split": "pending",
                    "identity_scope": static_game["identity_scope"],
                    "account_key": account_key,
                    "account_confidence": account_confidence,
                    "account_fold": account_fold,
                }
                feature_row = {
                    "sample_id": sample_id,
                    "game_id": game_id,
                    "turn_id": turn_id,
                    "family": prompt.family,
                    "phase": str(prompt.bundle.get("phase") or ""),
                    "round_number": int(target_event.get("round_number") or 0),
                    "source_run": str(source_run),
                    "source_llm_call_line": prompt.line_number,
                    "source_call_ts": prompt.call_ts,
                    "source_user_sha256": prompt.user_sha256,
                    "synthetic_feature_sha256": prompt.bundle_sha256,
                    "producer_keys": list(prompt.bundle.get("producer_keys") or []),
                    "feature_indices": list(vector.indices),
                    "feature_values": list(vector.values),
                    "feature_vector_sha256": vector.vector_sha256,
                    "leaf_count": vector.leaf_count,
                    "dropped_leaf_count": vector.dropped_leaf_count,
                    "excluded_provenance_count": vector.excluded_provenance_count,
                    "excluded_long_text_count": vector.excluded_long_text_count,
                }
                target_rows.append(target_row)
                feature_rows.append(feature_row)
                prediction_target = dict(target_row)
                prediction_target["target_label"] = "pass" if prompt.family == "persuasion" else "reject"
                prediction_target["target_delay_log_ms"] = None
                prediction_target["target_delay_present"] = False
                samples_by_family[prompt.family].append({"game": static_game, "events": prefix, "target": prediction_target})
                sample_ids_by_family[prompt.family].append(sample_id)
                seen_turn_ids.add(turn_id)
        if not target_rows:
            raise RuntimeError("pre-Terra corpus has no causally valid direct-response targets")
        target_families = {str(row["family"]) for row in feature_rows}
        missing_families = set(GLEE_FAMILIES) - target_families
        if missing_families:
            raise RuntimeError(f"pre-Terra corpus has no targets for families: {sorted(missing_families)}")
        game_rows = list(game_rows_by_id.values())
        assignments = _chronological_splits(game_rows, train_fraction=self.train_fraction, validation_fraction=self.validation_fraction)
        for row in game_rows:
            row["chronological_split"] = assignments[str(row["game_id"])]
        for row in target_rows:
            row["chronological_split"] = assignments[str(row["game_id"])]
        for family, samples in samples_by_family.items():
            for sample in samples:
                sample["game"]["chronological_split"] = assignments[str(sample["game"]["game_id"])]
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            games_frame = pl.DataFrame(game_rows, schema=GAME_SCHEMA, strict=False).sort(["completed_at", "game_id"])
            events_frame = pl.DataFrame([event for game_id in sorted(event_rows_by_game) for event in event_rows_by_game[game_id]], schema=EVENT_SCHEMA, strict=False).sort(["game_id", "event_index"])
            targets_frame = pl.DataFrame(target_rows, schema=TARGET_SCHEMA, strict=False).sort(["game_id", "target_event_index"])
            features_frame = pl.DataFrame(feature_rows, infer_schema_length=None).sort(["game_id", "round_number", "turn_id"])
            if games_frame["game_id"].n_unique() != games_frame.height or targets_frame["sample_id"].n_unique() != targets_frame.height or features_frame["sample_id"].n_unique() != features_frame.height:
                raise RuntimeError("pre-Terra corpus contains duplicate coordinates")
            prediction_rows: list[dict[str, object]] = []
            sequence_candidate = ShadowCandidate(self.sequence_release) if self.sequence_release is not None else None
            if sequence_candidate is not None:
                for family in GLEE_FAMILIES:
                    samples = samples_by_family[family]
                    sample_ids = sample_ids_by_family[family]
                    for start in range(0, len(samples), self.prediction_batch_size):
                        batch_samples = samples[start : start + self.prediction_batch_size]
                        batch_ids = sample_ids[start : start + self.prediction_batch_size]
                        predictions = sequence_candidate.predict(batch_samples, family=family)
                        for sample_id, prediction in zip(batch_ids, predictions, strict=True):
                            prediction_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "game_id": prediction["game_id"],
                                    "family": family,
                                    "candidate_id": prediction["candidate_id"],
                                    "candidate_manifest_sha256": prediction["candidate_manifest_sha256"],
                                    "labels": prediction["labels"],
                                    "probabilities": prediction["action_probabilities"],
                                    "predicted_action": prediction["predicted_action"],
                                    "action_message_gate": prediction["action_message_gate"],
                                }
                            )
            artifacts = {
                "games.parquet": _artifact(games_frame, staging / "games.parquet"),
                "events.parquet": _artifact(events_frame, staging / "events.parquet"),
                "targets.parquet": _artifact(targets_frame, staging / "targets.parquet"),
                "features.parquet": _artifact(features_frame, staging / "features.parquet"),
            }
            if sequence_candidate is not None:
                predictions_frame = pl.DataFrame(prediction_rows, infer_schema_length=None).sort("sample_id")
                if predictions_frame["sample_id"].n_unique() != targets_frame.height:
                    raise RuntimeError("aligned sequence predictions do not cover every v3 target")
                artifacts["sequence-predictions.parquet"] = _artifact(predictions_frame, staging / "sequence-predictions.parquet")
            reference_core = None
            if self.reference_core_corpus is not None:
                reference_core = json.loads((self.reference_core_corpus / "manifest.json").read_text(encoding="utf-8"))
                if reference_core.get("contract") != PRE_TERRA_V3_CORPUS_CONTRACT or reference_core.get("status") != "frozen-retrospective-core-corpus":
                    raise ValueError("reference pre-Terra core corpus is unsupported or unfrozen")
                reference_artifacts = reference_core.get("artifacts")
                if not isinstance(reference_artifacts, Mapping):
                    raise ValueError("reference pre-Terra core corpus has no artifact receipts")
                for name in ("games.parquet", "events.parquet", "targets.parquet", "features.parquet"):
                    receipt = reference_artifacts.get(name)
                    if not isinstance(receipt, Mapping) or artifacts[name]["sha256"] != receipt.get("sha256"):
                        raise RuntimeError(f"final v3 corpus does not reproduce the reference core artifact: {name}")
            joined = targets_frame.join(games_frame.select("game_id", "family"), on="game_id")
            inventory = {
                "games": games_frame.height,
                "events": events_frame.height,
                "targets": targets_frame.height,
                "feature_rows": features_frame.height,
                "sequence_prediction_rows": len(prediction_rows),
                "by_family": {family: {"games": games_frame.filter(pl.col("family") == family).height, "targets": joined.filter(pl.col("family") == family).height} for family in GLEE_FAMILIES},
                "by_split": dict(sorted(Counter(targets_frame["chronological_split"].to_list()).items())),
                "known_identity_targets": targets_frame.filter(pl.col("identity_scope") == "known").height,
                "account_labeled_targets": targets_frame.filter(pl.col("account_key").is_not_null()).height,
                "feature_nonzero_mean": features_frame.select(pl.col("feature_indices").list.len().mean()).item(),
                "feature_leaf_maximum": features_frame["leaf_count"].max(),
                "feature_dropped_rows": features_frame.filter(pl.col("dropped_leaf_count") > 0).height,
                "exclusions": dict(sorted(exclusions.items())),
                "account_label_collisions": account_collisions,
            }
            manifest = {
                "schema_version": 1,
                "contract": PRE_TERRA_V3_CORPUS_CONTRACT,
                "status": "frozen-retrospective-corpus" if sequence_candidate is not None else "frozen-retrospective-core-corpus",
                "source_runs": source_receipts,
                "sequence_reference": {"path": str(self.sequence_release), "manifest_sha256": file_sha256(self.sequence_release / "manifest.json"), "candidate_id": sequence_candidate.candidate_id} if sequence_candidate is not None and self.sequence_release is not None else None,
                "reference_core_corpus": {"path": str(self.reference_core_corpus), "manifest_sha256": file_sha256(self.reference_core_corpus / "manifest.json")} if self.reference_core_corpus is not None else None,
                "account_groups": {"path": str(self.account_groups), "sha256": file_sha256(self.account_groups)},
                "parameters": {"train_fraction": self.train_fraction, "validation_fraction": self.validation_fraction, "test_fraction": 1.0 - self.train_fraction - self.validation_fraction, "whole_game_split": True, "causal_bridge_events": 1, "source_run_count": len(self.source_runs), "feature_projection": hasher.receipt(), "prediction_batch_size": self.prediction_batch_size},
                "inventory": inventory,
                "artifacts": artifacts,
                "implementation_sha256": file_sha256(Path(__file__)),
                "boundary": "Exact pre-Terra payload features and authenticated prefixes only; realized self bridge actions and target outcomes are labels, never inputs.",
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, self.output_dir)
            return {"contract": PRE_TERRA_V3_CORPUS_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": file_sha256(self.output_dir / "manifest.json"), "inventory": inventory}
        except BaseException:
            if staging.exists():
                for path in staging.iterdir():
                    path.unlink()
                staging.rmdir()
            raise
