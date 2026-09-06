"""Temporal, collision-safe public identities and causal local-game envelopes for GLEE."""

from __future__ import annotations

import bisect
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .glee_activity_eda import GLEE_FAMILIES, _atomic_text, _file_digest, _read_only_database, _write_json, _write_jsonl
from .glee_effective_events import HighWaterDeriver


IDENTITY_REGISTRY_CONTRACT = "glee-public-identity-registry-v1"
IDENTITY_ENVELOPE_CONTRACT = "glee-game-identity-envelope-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _normalize_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized.casefold() if normalized else None


def _display_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _compact_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    keys = ("player_name", "player_type", "rank", "rating", "games_played", "is_baseline", "is_benchmark", "is_owner_best")
    return {key: payload.get(key) for key in keys if key in payload}


def _classification(payload: Mapping[str, object]) -> tuple[bool, bool, bool | None]:
    owner_best = payload.get("is_owner_best")
    return bool(payload.get("is_baseline")), bool(payload.get("is_benchmark")), bool(owner_best) if owner_best is not None else None


@dataclass(frozen=True)
class LabelInterval:
    family: str
    player_id: str
    display_label: str
    normalized_label: str
    start_sequence: int
    end_sequence_exclusive: int | None

    def contains(self, sequence: int) -> bool:
        return self.start_sequence <= sequence and (self.end_sequence_exclusive is None or sequence < self.end_sequence_exclusive)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract": IDENTITY_REGISTRY_CONTRACT,
            "schema_version": 1,
            "family": self.family,
            "player_id": self.player_id,
            "display_label": self.display_label,
            "normalized_label": self.normalized_label,
            "start_sequence": self.start_sequence,
            "end_sequence_exclusive": self.end_sequence_exclusive,
        }


@dataclass(frozen=True)
class ClassificationInterval:
    family: str
    player_id: str
    is_baseline: bool
    is_benchmark: bool
    is_owner_best: bool | None
    start_sequence: int
    end_sequence_exclusive: int | None

    def contains(self, sequence: int) -> bool:
        return self.start_sequence <= sequence and (self.end_sequence_exclusive is None or sequence < self.end_sequence_exclusive)


@dataclass
class _IdentityState:
    family: str
    player_id: str
    active: bool = False
    first_seen_sequence: int | None = None
    last_event_sequence: int | None = None
    row_revision_count: int = 0
    current_label: str | None = None
    current_normalized_label: str | None = None
    label_start_sequence: int | None = None
    presence_start_sequence: int | None = None
    presence_last_event_sequence: int | None = None
    current_classification: tuple[bool, bool, bool | None] | None = None
    classification_start_sequence: int | None = None
    latest_metadata: dict[str, object] | None = None
    aliases: set[str] = field(default_factory=set)
    normalized_aliases: set[str] = field(default_factory=set)
    presence_intervals: list[tuple[int, int | None, int]] = field(default_factory=list)
    label_intervals: list[LabelInterval] = field(default_factory=list)
    classification_intervals: list[ClassificationInterval] = field(default_factory=list)


class TemporalIdentityRegistry:
    """Resolve public labels at a pinned frontier without treating a display name as an identifier."""

    def __init__(self, *, frontier_sequence: int) -> None:
        self.frontier_sequence = frontier_sequence
        self.states: dict[tuple[str, str], _IdentityState] = {}
        self.label_index: dict[tuple[str, str], list[LabelInterval]] = defaultdict(list)
        self.classification_index: dict[tuple[str, str], list[ClassificationInterval]] = defaultdict(list)
        self.source_digest = hashlib.sha256()
        self.source_rows = 0
        self.observation_frontiers: dict[str, list[int]] = {}
        self._finished = False

    def _close_label(self, state: _IdentityState, end_sequence: int | None) -> None:
        if state.current_label is None or state.current_normalized_label is None or state.label_start_sequence is None:
            return
        interval = LabelInterval(state.family, state.player_id, state.current_label, state.current_normalized_label, state.label_start_sequence, end_sequence)
        state.label_intervals.append(interval)
        self.label_index[(state.family, state.current_normalized_label)].append(interval)
        state.current_label = None
        state.current_normalized_label = None
        state.label_start_sequence = None

    def _close_classification(self, state: _IdentityState, end_sequence: int | None) -> None:
        if state.current_classification is None or state.classification_start_sequence is None:
            return
        baseline, benchmark, owner_best = state.current_classification
        interval = ClassificationInterval(state.family, state.player_id, baseline, benchmark, owner_best, state.classification_start_sequence, end_sequence)
        state.classification_intervals.append(interval)
        self.classification_index[(state.family, state.player_id)].append(interval)
        state.current_classification = None
        state.classification_start_sequence = None

    def consume(self, row: Mapping[str, object]) -> None:
        if self._finished:
            raise RuntimeError("cannot consume identity rows after registry finalization")
        sequence = int(row["frontier_sequence"])
        family = str(row["family"])
        player_id = str(row["player_id"])
        row_json = row["row_json"]
        source = {"sequence": sequence, "family": family, "player_id": player_id, "change_kind": row["change_kind"], "row_sha256": row["row_sha256"]}
        self.source_digest.update(_canonical(source).encode("utf-8") + b"\n")
        self.source_rows += 1
        state = self.states.setdefault((family, player_id), _IdentityState(family, player_id))
        state.row_revision_count += 1
        state.last_event_sequence = sequence
        if row_json is None:
            if state.active:
                self._close_label(state, sequence)
                self._close_classification(state, sequence)
                if state.presence_start_sequence is not None and state.presence_last_event_sequence is not None:
                    state.presence_intervals.append((state.presence_start_sequence, sequence, state.presence_last_event_sequence))
            state.active = False
            state.presence_start_sequence = None
            state.presence_last_event_sequence = None
            state.latest_metadata = None
            return
        payload = json.loads(str(row_json))
        if not isinstance(payload, dict):
            raise ValueError(f"public row is not an object at {family}/{player_id}/{sequence}")
        label = _display_label(payload.get("player_name")) or player_id
        normalized = _normalize_label(label)
        if normalized is None:
            raise ValueError(f"public row has no usable label at {family}/{player_id}/{sequence}")
        classification = _classification(payload)
        if not state.active:
            state.active = True
            state.presence_start_sequence = sequence
            state.presence_last_event_sequence = sequence
            state.first_seen_sequence = sequence if state.first_seen_sequence is None else state.first_seen_sequence
            state.current_label = label
            state.current_normalized_label = normalized
            state.label_start_sequence = sequence
            state.current_classification = classification
            state.classification_start_sequence = sequence
        else:
            if label != state.current_label:
                self._close_label(state, sequence)
                state.current_label = label
                state.current_normalized_label = normalized
                state.label_start_sequence = sequence
            if classification != state.current_classification:
                self._close_classification(state, sequence)
                state.current_classification = classification
                state.classification_start_sequence = sequence
        state.aliases.add(label)
        state.normalized_aliases.add(normalized)
        state.latest_metadata = _compact_metadata(payload)
        state.presence_last_event_sequence = sequence

    def finish(self) -> None:
        if self._finished:
            return
        for state in self.states.values():
            if state.active:
                self._close_label(state, None)
                self._close_classification(state, None)
                if state.presence_start_sequence is not None and state.presence_last_event_sequence is not None:
                    state.presence_intervals.append((state.presence_start_sequence, None, state.presence_last_event_sequence))
                    state.presence_start_sequence = None
                    state.presence_last_event_sequence = None
        for intervals in self.label_index.values():
            intervals.sort(key=lambda row: (row.start_sequence, row.end_sequence_exclusive or self.frontier_sequence + 1, row.player_id))
        for intervals in self.classification_index.values():
            intervals.sort(key=lambda row: row.start_sequence)
        self._finished = True

    def set_observation_frontiers(self, frontiers: Mapping[str, Iterable[int]]) -> None:
        self.observation_frontiers = {family: sorted({int(sequence) for sequence in sequences if int(sequence) <= self.frontier_sequence}) for family, sequences in frontiers.items()}

    def _last_observed(self, family: str, start_sequence: int, end_sequence_exclusive: int | None, last_event_sequence: int) -> int:
        sequences = self.observation_frontiers.get(family, [])
        boundary = end_sequence_exclusive if end_sequence_exclusive is not None else self.frontier_sequence + 1
        index = bisect.bisect_left(sequences, boundary) - 1
        complete_frontier = sequences[index] if index >= 0 and sequences[index] >= start_sequence else start_sequence
        return max(last_event_sequence, complete_frontier)

    def ids_for_label(self, family: str, label: object, sequence: int) -> list[str]:
        normalized = _normalize_label(label)
        if normalized is None:
            return []
        return sorted({row.player_id for row in self.label_index.get((family, normalized), ()) if row.contains(sequence)})

    def ever_ids_for_label(self, family: str, label: object) -> list[str]:
        normalized = _normalize_label(label)
        if normalized is None:
            return []
        return sorted({row.player_id for row in self.label_index.get((family, normalized), ())})

    def classification_at(self, family: str, player_id: str, sequence: int) -> ClassificationInterval | None:
        return next((row for row in self.classification_index.get((family, player_id), ()) if row.contains(sequence)), None)

    def competitive_at(self, family: str, player_id: str, sequence: int) -> bool:
        row = self.classification_at(family, player_id, sequence)
        return bool(row is not None and not row.is_baseline and not row.is_benchmark)

    def registry_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key, state in sorted(self.states.items()):
            presence = [
                {"start_sequence": start, "last_observed_sequence": self._last_observed(key[0], start, end, last_event), "end_sequence_exclusive": end}
                for start, end, last_event in state.presence_intervals
            ]
            rows.append(
                {
                    "contract": IDENTITY_REGISTRY_CONTRACT,
                    "schema_version": 1,
                    "family": key[0],
                    "player_id": key[1],
                    "first_seen_sequence": state.first_seen_sequence,
                    "last_observed_sequence": presence[-1]["last_observed_sequence"] if presence else None,
                    "last_event_sequence": state.last_event_sequence,
                    "present_at_frontier": state.active,
                    "current_label": state.latest_metadata.get("player_name") if state.latest_metadata is not None else None,
                    "aliases": sorted(state.aliases, key=lambda value: (value.casefold(), value)),
                    "normalized_aliases": sorted(state.normalized_aliases),
                    "row_revision_count": state.row_revision_count,
                    "presence_intervals": presence,
                    "classification_intervals": [
                        {
                            "start_sequence": row.start_sequence,
                            "end_sequence_exclusive": row.end_sequence_exclusive,
                            "is_baseline": row.is_baseline,
                            "is_benchmark": row.is_benchmark,
                            "is_owner_best": row.is_owner_best,
                        }
                        for row in state.classification_intervals
                    ],
                    "latest_present_metadata": state.latest_metadata,
                }
            )
        return rows

    def label_rows(self) -> list[dict[str, object]]:
        return [row.as_dict() for row in sorted((item for values in self.label_index.values() for item in values), key=lambda item: (item.family, item.normalized_label, item.start_sequence, item.player_id))]

    def collision_rows(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        sentinel = self.frontier_sequence + 1
        for (family, normalized), intervals in sorted(self.label_index.items()):
            changes: dict[int, dict[str, set[str]]] = defaultdict(lambda: {"add": set(), "remove": set()})
            labels_by_player: dict[str, set[str]] = defaultdict(set)
            for interval in intervals:
                changes[interval.start_sequence]["add"].add(interval.player_id)
                changes[interval.end_sequence_exclusive or sentinel]["remove"].add(interval.player_id)
                labels_by_player[interval.player_id].add(interval.display_label)
            active: set[str] = set()
            prior: int | None = None
            for sequence in sorted(changes):
                if prior is not None and prior < sequence and len(active) > 1:
                    result.append(
                        {
                            "contract": IDENTITY_REGISTRY_CONTRACT,
                            "schema_version": 1,
                            "family": family,
                            "normalized_label": normalized,
                            "display_labels": sorted({label for player_id in active for label in labels_by_player[player_id]}, key=lambda value: (value.casefold(), value)),
                            "player_ids": sorted(active),
                            "start_sequence": prior,
                            "end_sequence_exclusive": None if sequence == sentinel else sequence,
                        }
                    )
                active.difference_update(changes[sequence]["remove"])
                active.update(changes[sequence]["add"])
                prior = sequence
        return result


def _observation_rows(connection: sqlite3.Connection, frontier_sequence: int) -> Iterable[sqlite3.Row]:
    return connection.execute(
        """
        SELECT ch.change_sequence, ch.frontier_sequence, ch.family, ch.player_id, ch.change_kind,
               ch.observed_after, ch.observed_by, ch.games_delta, ch.rating_delta,
               rv.row_sha256, rv.row_json,
               json_extract(rv.row_json, '$.games_played') AS observed_count,
               json_extract(rv.row_json, '$.rating') AS observed_rating
        FROM changes AS ch
        JOIN row_versions AS rv
          ON rv.sequence = ch.frontier_sequence
         AND rv.family = ch.family
         AND rv.player_id = ch.player_id
        WHERE ch.frontier_sequence <= ?
        ORDER BY ch.frontier_sequence, ch.family, ch.player_id, ch.change_sequence
        """,
        (frontier_sequence,),
    )


def _frontiers(connection: sqlite3.Connection, frontier_sequence: int) -> tuple[list[int], list[float]]:
    sequences: list[int] = []
    completed: list[float] = []
    for row in connection.execute("SELECT sequence, completed_at FROM frontiers WHERE sequence <= ? ORDER BY sequence", (frontier_sequence,)):
        stamp = _timestamp(row["completed_at"])
        if stamp is not None:
            sequences.append(int(row["sequence"]))
            completed.append(stamp)
    return sequences, completed


def _observation_frontiers(connection: sqlite3.Connection, frontier_sequence: int) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for row in connection.execute("SELECT family, sequence FROM family_polls WHERE sequence <= ? AND status = 'ok' AND truncated = 0 ORDER BY family, sequence", (frontier_sequence,)):
        result[str(row["family"])].append(int(row["sequence"]))
    return result


def _assignment_frontier(sequences: list[int], completed: list[float], started_at: object) -> int | None:
    stamp = _timestamp(started_at)
    if stamp is None:
        return None
    index = bisect.bisect_right(completed, stamp) - 1
    return sequences[index] if index >= 0 else None


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _profile_rows(dossier_index: Path | None, incremental_root: Path | None, timing_database: Path | None) -> tuple[list[dict[str, object]], dict[str, object]]:
    profiles: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    if dossier_index is not None and dossier_index.is_file():
        dossier_bytes = dossier_index.read_bytes()
        payload = json.loads(dossier_bytes)
        opponents = payload.get("opponents") if isinstance(payload, dict) else None
        if isinstance(opponents, dict):
            for opponent_key, record in opponents.items():
                if not isinstance(record, dict):
                    continue
                names = {name for name in [record.get("name"), *(record.get("aliases") if isinstance(record.get("aliases"), list) else [])] if isinstance(name, str) and _display_label(name)}
                for name in sorted(names, key=lambda value: (value.casefold(), value)):
                    profiles.append({"source_kind": "dossier-index", "source_ref": str(dossier_index), "profile_key": str(opponent_key), "family": None, "name": _display_label(name)})
        sources["dossier_index"] = {"path": str(dossier_index), "sha256": hashlib.sha256(dossier_bytes).hexdigest()}
    if incremental_root is not None and incremental_root.is_dir():
        paths = sorted(incremental_root.glob("opponents/*/*/current.json"))
        pointer_receipts: list[tuple[str, str]] = []
        for path in paths:
            pointer_bytes = path.read_bytes()
            payload = json.loads(pointer_bytes)
            pointer_receipts.append((str(path.relative_to(incremental_root)), hashlib.sha256(pointer_bytes).hexdigest()))
            name = _display_label(payload.get("opponent_name")) if isinstance(payload, dict) else None
            if name:
                profiles.append({"source_kind": "incremental-current", "source_ref": str(path), "profile_key": str(payload.get("opponent_id") or path.parent.parent.name), "family": str(payload.get("game_family") or path.parent.name), "name": name})
        digest = hashlib.sha256()
        for relative_path, pointer_sha256 in pointer_receipts:
            digest.update(relative_path.encode("utf-8") + b"\0" + pointer_sha256.encode("ascii") + b"\n")
        sources["incremental_current"] = {"root": str(incremental_root), "files": len(paths), "logical_sha256": digest.hexdigest()}
    if timing_database is not None and timing_database.is_file():
        connection = _read_only_database(timing_database)
        try:
            rows = connection.execute("SELECT opponent_id, opponent_name, family, COUNT(*) AS observations, COUNT(DISTINCT game_id) AS games FROM observations WHERE identity_kind = 'named' AND opponent_name IS NOT NULL GROUP BY opponent_id, opponent_name, family ORDER BY opponent_id, family").fetchall()
            for row in rows:
                name = _display_label(row["opponent_name"])
                if name:
                    profiles.append({"source_kind": "timing-profile", "source_ref": str(timing_database), "profile_key": str(row["opponent_id"]), "family": str(row["family"]), "name": name, "observations": int(row["observations"]), "games": int(row["games"])})
            logical = hashlib.sha256()
            for row in profiles:
                if row["source_kind"] == "timing-profile":
                    logical.update(_canonical(row).encode("utf-8") + b"\n")
            sources["timing_profiles"] = {"path": str(timing_database), "profiles": len(rows), "logical_sha256": logical.hexdigest()}
        finally:
            connection.close()
    return profiles, sources


class GleeIdentityRegistryAnalysis:
    """Build the Stage 1 registry and collision audit without publishing identity claims to live code."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        activity_summary: Path,
        attribution_path: Path,
        output_dir: Path,
        reporter_frontier: int | None = None,
        dossier_index: Path | None = None,
        incremental_dossier_root: Path | None = None,
        timing_database: Path | None = None,
    ) -> None:
        self.reporter_database = reporter_database.resolve()
        self.activity_summary = activity_summary.resolve()
        self.attribution_path = attribution_path.resolve()
        self.output_dir = output_dir.resolve()
        self.reporter_frontier = reporter_frontier
        self.dossier_index = dossier_index.resolve() if dossier_index is not None else None
        self.incremental_dossier_root = incremental_dossier_root.resolve() if incremental_dossier_root is not None else None
        self.timing_database = timing_database.resolve() if timing_database is not None else None

    @staticmethod
    def _envelope(
        game: dict[str, object],
        *,
        registry: TemporalIdentityRegistry,
        assignment_sequence: int | None,
        activity: dict[str, int],
    ) -> dict[str, object]:
        family = str(game["family"])
        completion_sequence = int(game["public_frontier_sequence"])
        identity_scope = str(game.get("identity_scope") or "hidden")
        opponent_name = game.get("opponent_name")
        known = identity_scope == "known" and isinstance(opponent_name, str) and bool(_normalize_label(opponent_name))
        assignment_ids = registry.ids_for_label(family, opponent_name, assignment_sequence) if known and assignment_sequence is not None else []
        completion_ids = registry.ids_for_label(family, opponent_name, completion_sequence) if known else []
        temporal_ids = sorted(set(assignment_ids).union(completion_ids))
        activity_ids = sorted(activity)
        joint_ids = sorted(set(temporal_ids).intersection(activity_ids)) if known else activity_ids
        excluded_label_ids: list[dict[str, object]] = []
        for player_id in sorted(set(temporal_ids) - set(activity_ids)):
            excluded_label_ids.append({"player_id": player_id, "reason": "no-effective-increment-at-aligned-completion-frontier"})
        exact_id = assignment_ids[0] if known and len(assignment_ids) == 1 else None
        if not known:
            status = "hidden-no-disclosed-label"
        elif len(assignment_ids) == 1:
            status = "unique-at-assignment-frontier"
        elif len(assignment_ids) > 1:
            status = "collision-set-at-assignment-frontier"
        elif len(completion_ids) == 1:
            status = "unique-at-completion-frontier-only"
        elif len(completion_ids) > 1:
            status = "collision-set-at-completion-frontier"
        else:
            status = "no-observed-public-label-match"
        return {
            "contract": IDENTITY_ENVELOPE_CONTRACT,
            "schema_version": 1,
            "game_id": game["game_id"],
            "family": family,
            "started_at": game.get("started_at"),
            "completed_at": game.get("completed_at"),
            "identity_scope": identity_scope,
            "disclosed_label": opponent_name,
            "assignment_frontier_sequence": assignment_sequence,
            "aligned_completion_frontier_sequence": completion_sequence,
            "assignment_label_ids": assignment_ids,
            "completion_label_ids": completion_ids,
            "temporal_label_ids": temporal_ids,
            "activity_frontier_ref": {"family": family, "frontier_sequence": completion_sequence},
            "activity_compatible_count": len(activity_ids),
            "joint_activity_label_ids": joint_ids if known else None,
            "excluded_label_ids": excluded_label_ids,
            "resolution_status": status,
            "exact_public_player_id": exact_id,
            "unknown_mass": None,
            "unknown_status": "unquantified-until-calibrated-joint-assignment",
            "source_attribution_contract": game.get("contract"),
            "source_archive_path": game.get("archive_path"),
            "source_archive_sha256": game.get("archive_sha256"),
        }

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(not path.name.startswith(".") for path in self.output_dir.iterdir()):
            raise FileExistsError(f"identity registry output directory is not empty: {self.output_dir}")
        activity_summary = json.loads(self.activity_summary.read_text(encoding="utf-8"))
        if activity_summary.get("contract") != "glee-activity-eda-v2":
            raise ValueError("identity registry requires a glee-activity-eda-v2 summary")
        frontier = int(self.reporter_frontier or activity_summary["source_frontier"]["frontier_sequence"])
        if frontier != int(activity_summary["source_frontier"]["frontier_sequence"]):
            raise ValueError("reporter frontier and activity summary frontier differ")
        games = _load_jsonl(self.attribution_path)
        target_frontiers = {(int(game["public_frontier_sequence"]), str(game["family"])) for game in games}
        activity: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
        reporter = _read_only_database(self.reporter_database)
        registry = TemporalIdentityRegistry(frontier_sequence=frontier)
        high_water = HighWaterDeriver()
        try:
            reporter.execute("BEGIN")
            frontier_sequences, frontier_completed = _frontiers(reporter, frontier)
            registry.set_observation_frontiers(_observation_frontiers(reporter, frontier))
            for row in _observation_rows(reporter, frontier):
                registry.consume(row)
                event, _observation = high_water.consume(row)
                if event is not None and (event.frontier_sequence, event.family) in target_frontiers:
                    activity[(event.frontier_sequence, event.family)][event.player_id] = event.games_delta
            registry.finish()
            effective_summary = high_water.finish(frontier_sequence=frontier)
            reporter.execute("ROLLBACK")
        finally:
            reporter.close()
        expected_source_hash = activity_summary["effective_event_reconstruction"]["source_rows_sha256"]
        if effective_summary["source_rows_sha256"] != expected_source_hash:
            raise RuntimeError("identity scan does not reproduce the frozen activity source receipt")
        self_ids = activity_summary["source_frontier"].get("self_player_ids") or {}
        activity_rows: list[dict[str, object]] = []
        competitive_activity: dict[tuple[int, str], dict[str, int]] = {}
        for key in sorted(target_frontiers):
            sequence, family = key
            self_id = str(self_ids.get(family)) if self_ids.get(family) else None
            accepted: dict[str, int] = {}
            excluded: list[dict[str, object]] = []
            for player_id, capacity in sorted(activity.get(key, {}).items()):
                if player_id == self_id:
                    continue
                if registry.competitive_at(family, player_id, sequence):
                    accepted[player_id] = capacity
                else:
                    excluded.append({"player_id": player_id, "reason": "baseline-benchmark-or-not-present-at-frontier"})
            competitive_activity[key] = accepted
            activity_rows.append(
                {
                    "contract": IDENTITY_ENVELOPE_CONTRACT,
                    "schema_version": 1,
                    "family": family,
                    "frontier_sequence": sequence,
                    "candidate_capacity_by_id": accepted,
                    "excluded_noncompetitive_events": excluded,
                }
            )
        envelopes = [
            self._envelope(
                game,
                registry=registry,
                assignment_sequence=_assignment_frontier(frontier_sequences, frontier_completed, game.get("started_at")),
                activity=competitive_activity.get((int(game["public_frontier_sequence"]), str(game["family"])), {}),
            )
            for game in games
        ]
        collision_rows = registry.collision_rows()
        profile_rows, profile_sources = _profile_rows(self.dossier_index, self.incremental_dossier_root, self.timing_database)
        collision_profiles: list[dict[str, object]] = []
        for profile in profile_rows:
            families = [str(profile["family"])] if profile.get("family") in GLEE_FAMILIES else list(GLEE_FAMILIES)
            ever = {family: registry.ever_ids_for_label(family, profile["name"]) for family in families}
            current = {family: registry.ids_for_label(family, profile["name"], frontier) for family in families}
            if any(len(ids) > 1 for ids in ever.values()):
                collision_profiles.append({**profile, "ever_public_ids_by_family": ever, "frontier_public_ids_by_family": current, "status": "current-collision" if any(len(ids) > 1 for ids in current.values()) else "historically-ambiguous"})
        envelope_statuses = Counter(str(row["resolution_status"]) for row in envelopes)
        known = [row for row in envelopes if row["identity_scope"] == "known"]
        collision_games = [row for row in known if row["resolution_status"] == "collision-set-at-assignment-frontier"]
        collision_labels = Counter(str(_normalize_label(row.get("disclosed_label")) or "missing") for row in collision_games)
        no_match_labels = Counter(str(_normalize_label(row.get("disclosed_label")) or "missing") for row in known if row["resolution_status"] == "no-observed-public-label-match")
        collision_joint_sizes = Counter(len(row["joint_activity_label_ids"]) for row in collision_games)
        current_collision_labels = sorted({(row["family"], row["normalized_label"]) for row in collision_rows if row["end_sequence_exclusive"] is None})
        reserve = [row for row in envelopes if _normalize_label(row.get("disclosed_label")) == "reserve"]
        summary = {
            "contract": IDENTITY_REGISTRY_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only",
            "frontier_sequence": frontier,
            "source": {
                "reporter_database": str(self.reporter_database),
                "activity_summary": str(self.activity_summary),
                "activity_summary_sha256": _file_digest(self.activity_summary),
                "attribution_path": str(self.attribution_path),
                "attribution_sha256": _file_digest(self.attribution_path),
                "source_rows_sha256": effective_summary["source_rows_sha256"],
                "source_rows": registry.source_rows,
                "identity_source_rows_sha256": registry.source_digest.hexdigest(),
                "profile_sources": profile_sources,
            },
            "registry": {
                "player_family_records": len(registry.states),
                "label_intervals": sum(len(rows) for rows in registry.label_index.values()),
                "presence_intervals": sum(len(state.presence_intervals) for state in registry.states.values()),
                "renamed_player_family_records": sum(len(state.normalized_aliases) > 1 for state in registry.states.values()),
                "collision_intervals": len(collision_rows),
                "current_collision_labels": [{"family": family, "normalized_label": label} for family, label in current_collision_labels],
            },
            "game_envelopes": {
                "games": len(envelopes),
                "known_games": len(known),
                "hidden_games": len(envelopes) - len(known),
                "resolution_statuses": dict(sorted(envelope_statuses.items())),
                "known_unique_at_assignment": envelope_statuses["unique-at-assignment-frontier"],
                "known_collision_at_assignment": envelope_statuses["collision-set-at-assignment-frontier"],
                "known_without_assignment_match": len(known) - envelope_statuses["unique-at-assignment-frontier"] - envelope_statuses["collision-set-at-assignment-frontier"],
                "collision_games_by_normalized_label": dict(sorted(collision_labels.items())),
                "collision_joint_candidate_counts": {str(size): count for size, count in sorted(collision_joint_sizes.items())},
                "no_match_games_by_normalized_label": dict(sorted(no_match_labels.items())),
                "unknown_mass_is_calibrated": False,
            },
            "collision_audit": {
                "profile_records": len(profile_rows),
                "ambiguous_profile_records": len(collision_profiles),
                "reserve_game_envelopes": len(reserve),
                "reserve_resolution_statuses": dict(sorted(Counter(str(row["resolution_status"]) for row in reserve).items())),
            },
            "effective_event_receipt": {
                "effective_events_sha256": effective_summary["effective_events_sha256"],
                "effective_additions": effective_summary["totals"]["effective_additions"],
                "invariant_violations": effective_summary["invariants"]["violation_count"],
            },
        }
        _write_jsonl(self.output_dir / "identity-registry.jsonl", registry.registry_rows())
        _write_jsonl(self.output_dir / "label-intervals.jsonl", registry.label_rows())
        _write_jsonl(self.output_dir / "collision-intervals.jsonl", collision_rows)
        _write_jsonl(self.output_dir / "activity-frontiers.jsonl", activity_rows)
        _write_jsonl(self.output_dir / "game-identity-envelopes.jsonl", envelopes)
        _write_jsonl(self.output_dir / "profile-inventory.jsonl", profile_rows)
        _write_json(self.output_dir / "collision-audit.json", {"contract": IDENTITY_REGISTRY_CONTRACT, "schema_version": 1, "profiles": collision_profiles})
        _write_json(self.output_dir / "summary.json", summary)
        readme = "\n".join(
            [
                "# GLEE temporal identity registry v1",
                "",
                f"**Status:** Offline shadow receipt through reporter frontier `{frontier}`; no identity result is connected to matchmaking, prompts, actions, or dossier publication.",
                "",
                "## Result",
                "",
                f"The registry contains {summary['registry']['player_family_records']:,} player-family records, {summary['registry']['label_intervals']:,} half-open label intervals, and {summary['registry']['collision_intervals']:,} compressed collision intervals. At the pinned frontier, {len(current_collision_labels):,} family-label pairs are colliding.",
                "",
                f"The causal envelope pass covers {len(envelopes):,} local games: {len(known):,} disclose a label and {len(envelopes) - len(known):,} are hidden. Exactly {envelope_statuses['unique-at-assignment-frontier']:,} known games have one observed public ID carrying that label at the last public frontier completed before assignment; {envelope_statuses['collision-set-at-assignment-frontier']:,} retain a set-valued collision at assignment.",
                "",
                "`unknown_mass` remains null because Stage 1 establishes candidate support and exclusions, not calibrated posterior mass. Same-frontier effective activity narrows a candidate set but cannot turn a duplicated label into authenticated exact identity.",
                "",
                "## Artifacts",
                "",
                "`identity-registry.jsonl` stores one compact record per stable public player-family identity, `label-intervals.jsonl` stores temporal label ownership, `collision-intervals.jsonl` stores compressed concurrent collisions, `activity-frontiers.jsonl` stores each aligned frontier candidate set once, `game-identity-envelopes.jsonl` references those sets and stores only label-specific intersections and exclusions, `profile-inventory.jsonl` freezes the compact name-keyed profile inputs, and `collision-audit.json` lists profiles whose labels have mapped to several public IDs.",
                "",
            ]
        )
        _atomic_text(self.output_dir / "README.md", readme)
        artifacts = ("README.md", "summary.json", "identity-registry.jsonl", "label-intervals.jsonl", "collision-intervals.jsonl", "activity-frontiers.jsonl", "game-identity-envelopes.jsonl", "profile-inventory.jsonl", "collision-audit.json")
        manifest = {
            "contract": IDENTITY_REGISTRY_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "source_rows_sha256": effective_summary["source_rows_sha256"],
            "effective_events_sha256": effective_summary["effective_events_sha256"],
            "implementation_sha256": {
                "glee_identity_registry.py": _file_digest(Path(__file__)),
                "glee_effective_events.py": _file_digest(Path(__file__).with_name("glee_effective_events.py")),
            },
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {**summary, "output_dir": str(self.output_dir), "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
