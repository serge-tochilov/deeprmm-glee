"""Offline capacity-aware reconstruction of local GLEE games and public completion events."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from .glee_activity_eda import GLEE_FAMILIES, GleeActivityEDA, LocalGame, RatingEvidenceModel, _atomic_text, _file_digest, _read_only_database, _write_json, _write_jsonl
from .glee_effective_events import EffectiveEvent, derive_effective_events
from .glee_identity_registry import _assignment_frontier, _frontiers, _normalize_label
from .glee_joint_assignment import AssignmentDemand, AssignmentMarginals, AssignmentOption, CapacityAssignment, sample_capacity_marginals, solve_capacity_assignment, solve_capacity_assignment_auction


JOINT_ASSIGNMENT_ANALYSIS_CONTRACT = "glee-joint-game-assignment-v1"
ASSIGNMENT_MODELS = ("activity-only", "rating-only", "joint", "evidence-conditioned")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _iso(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="microseconds") if value is not None else None


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _logsumexp(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    maximum = max(materialized)
    return maximum + math.log(sum(math.exp(value - maximum) for value in materialized))


def _interval_distance(timestamp: float, start: float | None, end: float) -> float:
    start = end if start is None else start
    if timestamp < start:
        return start - timestamp
    if timestamp > end:
        return timestamp - end
    return 0.0


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """One corrected high-water public event retained in the bounded candidate store."""

    event_id: str
    source_change_sequence: int
    frontier_sequence: int
    family: str
    player_id: str
    observed_after: float | None
    observed_by: float
    games_delta: int
    clean_rating_delta: float | None
    rating_status: str
    prior_event_gap_s: float | None


@dataclass(frozen=True, slots=True)
class SelfAlignment:
    """One local game and its optional unique self-capacity slot."""

    game: LocalGame
    self_event: PublicEvent | None
    self_slot_index: int | None
    method: str
    distance_s: float | None


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    """One causally compatible opponent event with masked activity and rating features."""

    event: PublicEvent
    interval_distance_s: float
    frontier_distance: int
    publication_lag_s: float
    activity_utility: float
    rating_utility: float


@dataclass(frozen=True, slots=True)
class GameCandidates:
    """One self-aligned game, its public candidates, and causal evaluation metadata."""

    alignment: SelfAlignment
    assignment_frontier_sequence: int | None
    true_public_player_ids: tuple[str, ...]
    candidates: tuple[CandidateEdge, ...]
    unknown_probability_prior: float
    causal_rating_training_games: int


class IdentityLookup(Protocol):
    def ids_for_label(self, family: str, label: object, sequence: int) -> list[str]: ...

    def competitive_at(self, family: str, player_id: str, sequence: int) -> bool: ...


class FrozenIdentityRegistry:
    """Read the Stage 1 temporal registry directly from its hash-verified compact artifacts."""

    def __init__(self, *, summary_path: Path) -> None:
        root = summary_path.parent
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != "glee-public-identity-registry-v1":
            raise ValueError("identity artifact manifest has an incompatible contract")
        if _file_digest(summary_path) != manifest["artifacts"]["summary.json"]["sha256"]:
            raise RuntimeError("identity summary hash does not match its frozen manifest")
        self.frontier_sequence = int(manifest["frontier_sequence"])
        labels_path = root / "label-intervals.jsonl"
        registry_path = root / "identity-registry.jsonl"
        for name, path in (("label-intervals.jsonl", labels_path), ("identity-registry.jsonl", registry_path)):
            if _file_digest(path) != manifest["artifacts"][name]["sha256"]:
                raise RuntimeError(f"identity artifact hash mismatch: {name}")
        self.labels: dict[tuple[str, str], list[tuple[int, int | None, str]]] = defaultdict(list)
        with labels_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                self.labels[(str(row["family"]), str(row["normalized_label"]))].append((int(row["start_sequence"]), int(row["end_sequence_exclusive"]) if row["end_sequence_exclusive"] is not None else None, str(row["player_id"])))
        self.classifications: dict[tuple[str, str], list[tuple[int, int | None, bool]]] = defaultdict(list)
        with registry_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (str(row["family"]), str(row["player_id"]))
                for interval in row["classification_intervals"]:
                    competitive = not bool(interval["is_baseline"]) and not bool(interval["is_benchmark"])
                    self.classifications[key].append((int(interval["start_sequence"]), int(interval["end_sequence_exclusive"]) if interval["end_sequence_exclusive"] is not None else None, competitive))

    def ids_for_label(self, family: str, label: object, sequence: int) -> list[str]:
        normalized = _normalize_label(label)
        if normalized is None:
            return []
        return sorted({player_id for start, end, player_id in self.labels.get((family, normalized), ()) if start <= sequence and (end is None or sequence < end)})

    def competitive_at(self, family: str, player_id: str, sequence: int) -> bool:
        return any(start <= sequence and (end is None or sequence < end) and competitive for start, end, competitive in self.classifications.get((family, player_id), ()))


class _EventStore:
    """Bounded temporary SQLite store that avoids a permanent copy of the corrected event corpus."""

    def __init__(self, path: Path, *, create: bool = True) -> None:
        self.path = path
        self.create = create
        self.connection = sqlite3.connect(path) if create else sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        if create:
            self.connection.executescript(
                """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                source_change_sequence INTEGER NOT NULL,
                frontier_sequence INTEGER NOT NULL,
                family TEXT NOT NULL,
                player_id TEXT NOT NULL,
                observed_after REAL,
                observed_by REAL NOT NULL,
                games_delta INTEGER NOT NULL,
                clean_rating_delta REAL,
                rating_status TEXT NOT NULL,
                prior_event_gap_s REAL
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
            )
        else:
            self.connection.execute("PRAGMA query_only=ON")
        self.rows: list[tuple[object, ...]] = []
        self.previous_event_by_player: dict[tuple[str, str], float] = {}

    def add(self, event: EffectiveEvent) -> None:
        if not self.create:
            raise RuntimeError("cannot append to a read-only event cache")
        observed_by = _timestamp(event.observed_by)
        if observed_by is None:
            raise ValueError(f"effective event lacks an observation time: {event.source_change_sequence}")
        key = (event.family, event.player_id)
        previous = self.previous_event_by_player.get(key)
        self.previous_event_by_player[key] = observed_by
        self.rows.append(
            (
                str(event.source_change_sequence),
                event.source_change_sequence,
                event.frontier_sequence,
                event.family,
                event.player_id,
                _timestamp(event.observed_after),
                observed_by,
                event.games_delta,
                event.clean_rating_delta,
                event.rating_status,
                max(0.0, observed_by - previous) if previous is not None else None,
            )
        )
        if len(self.rows) >= 5_000:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", self.rows)
        self.rows.clear()

    def finish(self, metadata: Mapping[str, object]) -> None:
        self.flush()
        self.connection.executescript("CREATE INDEX events_family_frontier_idx ON events(family, frontier_sequence); CREATE INDEX events_player_frontier_idx ON events(family, player_id, frontier_sequence);")
        self.connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((str(key), _canonical(value)) for key, value in sorted(metadata.items())))
        self.connection.commit()

    def metadata(self) -> dict[str, object]:
        return {str(row["key"]): json.loads(str(row["value"])) for row in self.connection.execute("SELECT key, value FROM metadata ORDER BY key")}

    @staticmethod
    def _event(row: sqlite3.Row) -> PublicEvent:
        return PublicEvent(
            event_id=str(row["event_id"]),
            source_change_sequence=int(row["source_change_sequence"]),
            frontier_sequence=int(row["frontier_sequence"]),
            family=str(row["family"]),
            player_id=str(row["player_id"]),
            observed_after=float(row["observed_after"]) if row["observed_after"] is not None else None,
            observed_by=float(row["observed_by"]),
            games_delta=int(row["games_delta"]),
            clean_rating_delta=float(row["clean_rating_delta"]) if row["clean_rating_delta"] is not None else None,
            rating_status=str(row["rating_status"]),
            prior_event_gap_s=float(row["prior_event_gap_s"]) if row["prior_event_gap_s"] is not None else None,
        )

    def player_events(self, family: str, player_id: str) -> list[PublicEvent]:
        rows = self.connection.execute("SELECT * FROM events WHERE family = ? AND player_id = ? ORDER BY frontier_sequence, source_change_sequence", (family, player_id))
        return [self._event(row) for row in rows]

    def candidate_events(self, *, family: str, first_frontier: int, last_frontier: int) -> list[PublicEvent]:
        rows = self.connection.execute("SELECT * FROM events WHERE family = ? AND frontier_sequence BETWEEN ? AND ? ORDER BY frontier_sequence, source_change_sequence", (family, first_frontier, last_frontier))
        return [self._event(row) for row in rows]

    def close(self) -> None:
        self.connection.close()


def align_self_games(games: Iterable[LocalGame], events: Iterable[PublicEvent], *, alignment_slack_s: float, observation_quantum_s: float) -> tuple[list[SelfAlignment], dict[str, object]]:
    """Jointly align local games to interval-compatible self events with no capacity reuse."""

    ordered_games = sorted(games, key=lambda game: (game.completed_at, game.game_id))
    ordered_events = sorted(events, key=lambda event: (event.frontier_sequence, event.source_change_sequence))
    event_by_id = {event.event_id: event for event in ordered_events}
    capacities = {event.event_id: event.games_delta for event in ordered_events}
    demands: list[AssignmentDemand] = []
    for game in ordered_games:
        options: list[AssignmentOption] = []
        for event in ordered_events:
            distance = _interval_distance(game.completed_at, event.observed_after, event.observed_by)
            if distance > alignment_slack_s:
                continue
            start = event.observed_after if event.observed_after is not None else event.observed_by
            midpoint = (start + event.observed_by) / 2.0
            utility = -distance / max(1.0, observation_quantum_s) - abs(game.completed_at - midpoint) / max(100.0, 100.0 * observation_quantum_s)
            if game.rating_delta is not None and event.clean_rating_delta is not None:
                utility -= min(50.0, abs(game.rating_delta - event.clean_rating_delta) / 0.1)
            options.append(AssignmentOption(event.event_id, utility))
        demands.append(AssignmentDemand(game.game_id, tuple(options), -100.0))
    selected: dict[str, str | None] = {}
    used_events: Counter[str] = Counter()
    total_utility = 0.0
    components = _assignment_components(demands)
    for component in components:
        resource_ids = {option.resource_id for demand in component for option in demand.options}
        assignment = solve_capacity_assignment(component, {resource_id: capacities[resource_id] for resource_id in resource_ids})
        selected.update(assignment.selected)
        used_events.update(assignment.used_capacity)
        total_utility += assignment.total_utility
    alignments: list[SelfAlignment] = []
    method_counts: Counter[str] = Counter()
    slot_indexes: Counter[str] = Counter()
    for game in ordered_games:
        event_id = selected.get(game.game_id)
        if event_id is None:
            alignment = SelfAlignment(game, None, None, "unmatched", None)
        else:
            event = event_by_id[event_id]
            distance = _interval_distance(game.completed_at, event.observed_after, event.observed_by)
            method = "interval-contained" if distance == 0 else "slack-nearest"
            event_slot = slot_indexes[event_id]
            slot_indexes[event_id] += 1
            alignment = SelfAlignment(game, event, event_slot, method, distance)
        method_counts[alignment.method] += 1
        alignments.append(alignment)
    capacity_violations = [event_id for event_id, used in used_events.items() if used > capacities[event_id]]
    rating_mismatches = sorted(abs(alignment.game.rating_delta - alignment.self_event.clean_rating_delta) for alignment in alignments if alignment.game.rating_delta is not None and alignment.self_event is not None and alignment.self_event.clean_rating_delta is not None)
    summary = {
        "local_games": len(ordered_games),
        "self_public_capacity": sum(capacities.values()),
        "aligned_games": len(ordered_games) - method_counts["unmatched"],
        "unmatched_games": method_counts["unmatched"],
        "unused_public_capacity": sum(capacities.values()) - sum(used_events.values()),
        "capacity_violations": len(capacity_violations),
        "methods": dict(sorted(method_counts.items())),
        "assignment_components": len(components),
        "maximum_component_games": max((len(component) for component in components), default=0),
        "total_assignment_utility": _rounded(total_utility),
        "authenticated_public_rating_pairs": len(rating_mismatches),
        "rating_delta_mismatch_p50": _rounded(_quantile(rating_mismatches, 0.5)),
        "rating_delta_mismatch_p90": _rounded(_quantile(rating_mismatches, 0.9)),
        "rating_delta_mismatch_max": _rounded(max(rating_mismatches, default=0.0)),
        "rating_delta_mismatch_gt_0_11": sum(value > 0.11 for value in rating_mismatches),
        "rating_match_cost": "absolute authenticated-versus-clean-public displayed-delta mismatch divided by 0.1 and capped at 50",
    }
    return alignments, summary


def _activity_utility(*, event: PublicEvent, self_event: PublicEvent, game: LocalGame, observation_quantum_s: float) -> tuple[float, float, int, float]:
    interval_distance = _interval_distance(game.completed_at, event.observed_after, event.observed_by)
    frontier_distance = abs(event.frontier_sequence - self_event.frontier_sequence)
    publication_lag = abs(event.observed_by - self_event.observed_by)
    capacity_term = math.log(max(1, event.games_delta))
    session_term = 0.15 * math.exp(-event.prior_event_gap_s / 300.0) if event.prior_event_gap_s is not None else 0.0
    utility = capacity_term + session_term - 0.35 * interval_distance / max(1.0, observation_quantum_s) - 0.55 * frontier_distance - 0.10 * publication_lag / max(1.0, observation_quantum_s)
    return utility, interval_distance, frontier_distance, publication_lag


def _unknown_utility(candidate_utilities: Iterable[float], probability: float) -> float:
    values = list(candidate_utilities)
    if not values:
        return 0.0
    bounded = min(1.0 - 1e-9, max(1e-9, probability))
    return _logsumexp(values) + math.log(bounded / (1.0 - bounded))


def build_game_candidates(
    alignments: Iterable[SelfAlignment],
    *,
    event_store: _EventStore,
    registry: IdentityLookup,
    self_ids: Mapping[str, str],
    frontier_sequences: list[int],
    frontier_completed: list[float],
    frontier_radius: int,
    alignment_slack_s: float,
    observation_quantum_s: float,
    unknown_alpha: float,
    unknown_beta: float,
) -> tuple[list[GameCandidates], dict[str, PublicEvent], dict[str, object]]:
    """Build masked candidate edges and update rating evidence only after each known result has been scored."""

    rating_models = {family: RatingEvidenceModel() for family in GLEE_FAMILIES}
    prior_unique: Counter[str] = Counter()
    prior_misses: Counter[str] = Counter()
    candidate_events: dict[str, PublicEvent] = {}
    rows: list[GameCandidates] = []
    evaluation_inventory: dict[str, Counter[str]] = {family: Counter() for family in GLEE_FAMILIES}
    for alignment in sorted(alignments, key=lambda row: (row.game.completed_at, row.game.game_id)):
        game = alignment.game
        family = game.family
        self_event = alignment.self_event
        assignment_sequence = _assignment_frontier(frontier_sequences, frontier_completed, _iso(game.started_at)) if game.started_at is not None else None
        true_ids = tuple(registry.ids_for_label(family, game.opponent_name, assignment_sequence)) if game.identity_scope == "known" and assignment_sequence is not None else ()
        prior_total = prior_unique[family]
        prior_unknown = (prior_misses[family] + unknown_alpha) / (prior_total + unknown_alpha + unknown_beta)
        edges: list[CandidateEdge] = []
        if self_event is not None:
            events = event_store.candidate_events(family=family, first_frontier=max(1, self_event.frontier_sequence - frontier_radius), last_frontier=self_event.frontier_sequence + frontier_radius)
            for event in events:
                if event.player_id == self_ids.get(family) or not registry.competitive_at(family, event.player_id, event.frontier_sequence):
                    continue
                interval_distance = _interval_distance(game.completed_at, event.observed_after, event.observed_by)
                if interval_distance > alignment_slack_s:
                    continue
                activity_utility, interval_distance, frontier_distance, publication_lag = _activity_utility(event=event, self_event=self_event, game=game, observation_quantum_s=observation_quantum_s)
                rating_utility = 0.0
                model = rating_models[family]
                if self_event.clean_rating_delta is not None and event.clean_rating_delta is not None and model.training_games > 0:
                    rating_utility = model.score(self_delta=self_event.clean_rating_delta, opponent_delta=event.clean_rating_delta)
                edge = CandidateEdge(event, interval_distance, frontier_distance, publication_lag, activity_utility, rating_utility)
                edges.append(edge)
                candidate_events[event.event_id] = event
        edges.sort(key=lambda edge: (edge.event.frontier_sequence, edge.event.player_id, edge.event.source_change_sequence))
        row = GameCandidates(alignment, assignment_sequence, true_ids, tuple(edges), prior_unknown, rating_models[family].training_games)
        rows.append(row)
        inventory = evaluation_inventory[family]
        inventory["games"] += 1
        inventory["self_aligned"] += int(self_event is not None)
        inventory["candidate_edges"] += len(edges)
        inventory["candidate_public_ids"] += len({edge.event.player_id for edge in edges})
        if game.identity_scope == "known":
            inventory["known"] += 1
        if len(true_ids) == 1:
            inventory["unique_label"] += 1
            target_edges = [edge for edge in edges if edge.event.player_id == true_ids[0]]
            covered = bool(target_edges)
            inventory["unique_label_covered"] += int(covered)
            prior_unique[family] += 1
            prior_misses[family] += int(not covered)
            if self_event is not None and self_event.clean_rating_delta is not None:
                clean_targets = [edge for edge in target_edges if edge.event.clean_rating_delta is not None]
                if len(clean_targets) == 1:
                    negatives = [edge.event.clean_rating_delta for edge in edges if edge.event.player_id != true_ids[0] and edge.event.clean_rating_delta is not None]
                    rating_models[family].update(self_delta=self_event.clean_rating_delta, positive_delta=clean_targets[0].event.clean_rating_delta, negative_deltas=negatives)
                    inventory["causal_rating_updates"] += 1
        elif len(true_ids) > 1:
            inventory["collision_label"] += 1
        elif game.identity_scope == "known":
            inventory["known_without_public_label_match"] += 1
    summary = {
        "games": len(rows),
        "unique_candidate_events": len(candidate_events),
        "by_family": {family: dict(sorted(evaluation_inventory[family].items())) for family in GLEE_FAMILIES},
        "causal_rating_training_games": {family: rating_models[family].training_games for family in GLEE_FAMILIES},
        "unknown_prior": {"alpha": unknown_alpha, "beta": unknown_beta, "interpretation": "causal beta-binomial prior over unique-label candidate misses before the current game"},
    }
    return rows, candidate_events, summary


def _model_demands(rows: Iterable[GameCandidates], model: str) -> tuple[list[AssignmentDemand], dict[str, int]]:
    if model not in ASSIGNMENT_MODELS:
        raise ValueError(f"unknown assignment model: {model}")
    demands: list[AssignmentDemand] = []
    capacities: dict[str, int] = {}
    for row in rows:
        options: list[AssignmentOption] = []
        allowed_ids = set(row.true_public_player_ids) if model == "evidence-conditioned" and row.alignment.game.identity_scope == "known" else None
        for edge in row.candidates:
            if allowed_ids is not None and edge.event.player_id not in allowed_ids:
                continue
            if model == "activity-only":
                utility = edge.activity_utility
            elif model == "rating-only":
                utility = edge.rating_utility
            else:
                utility = edge.activity_utility + 0.5 * edge.rating_utility
            options.append(AssignmentOption(edge.event.event_id, utility))
            capacities[edge.event.event_id] = edge.event.games_delta
        unknown = _unknown_utility((option.utility for option in options), row.unknown_probability_prior)
        demands.append(AssignmentDemand(row.alignment.game.game_id, tuple(options), unknown))
    return demands, capacities


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            prior = self.parent[value]
            self.parent[value] = root
            value = prior
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def _assignment_components(demands: list[AssignmentDemand]) -> list[list[AssignmentDemand]]:
    disjoint = _DisjointSet(demand.demand_id for demand in demands)
    resource_owner: dict[str, str] = {}
    for demand in demands:
        for option in demand.options:
            owner = resource_owner.setdefault(option.resource_id, demand.demand_id)
            disjoint.union(owner, demand.demand_id)
    grouped: dict[str, list[AssignmentDemand]] = defaultdict(list)
    for demand in demands:
        grouped[disjoint.find(demand.demand_id)].append(demand)
    return [sorted(component, key=lambda demand: demand.demand_id) for _, component in sorted(grouped.items())]


def _softmax_probabilities(demand: AssignmentDemand, temperature: float) -> dict[str | None, float]:
    values = [(option.resource_id, option.utility / temperature) for option in demand.options]
    values.append((None, demand.unmatched_utility / temperature))
    maximum = max(value for _, value in values)
    denominator = sum(math.exp(value - maximum) for _, value in values)
    return {resource_id: math.exp(value - maximum) / denominator for resource_id, value in values}


def _independent_component(component: list[AssignmentDemand], capacities: Mapping[str, int]) -> bool:
    uses = Counter(option.resource_id for demand in component for option in demand.options)
    return all(count <= capacities[resource_id] for resource_id, count in uses.items())


def assign_with_marginals(rows: list[GameCandidates], *, model: str, samples: int, temperature: float, seed: str, auction_epsilon: float = 0.02) -> tuple[AssignmentMarginals, dict[str, object]]:
    """Solve independent conflict components and combine their capacity-feasible marginals."""

    demands, capacities = _model_demands(rows, model)
    components = _assignment_components(demands)
    selected: dict[str, str | None] = {}
    used: Counter[str] = Counter()
    total_utility = 0.0
    probabilities: dict[str, dict[str | None, float]] = {}
    sampled_components = 0
    independent_components = 0
    exact_conflict_components = 0
    auction_conflict_components = 0
    maximum_component = 0
    for component in components:
        maximum_component = max(maximum_component, len(component))
        resource_ids = {option.resource_id for demand in component for option in demand.options}
        component_capacities = {resource_id: capacities[resource_id] for resource_id in resource_ids}
        if _independent_component(component, component_capacities):
            independent_components += 1
            assignment = solve_capacity_assignment(component, component_capacities)
            component_probabilities = {demand.demand_id: _softmax_probabilities(demand, temperature) for demand in component}
        else:
            sampled_components += 1
            if len(component) <= 40:
                exact_conflict_components += 1
                solver = solve_capacity_assignment
            else:
                auction_conflict_components += 1
                solver = lambda values, limits: solve_capacity_assignment_auction(values, limits, epsilon=auction_epsilon)
            marginal = sample_capacity_marginals(component, component_capacities, samples=samples, temperature=temperature, seed=f"{seed}:{model}:{component[0].demand_id}", solver=solver)
            assignment = marginal.map_assignment
            component_probabilities = marginal.probabilities
        selected.update(assignment.selected)
        used.update(assignment.used_capacity)
        total_utility += assignment.total_utility
        probabilities.update(component_probabilities)
    violations = [resource_id for resource_id, count in used.items() if count > capacities[resource_id]]
    if violations:
        raise RuntimeError(f"joint assignment reused public capacity: {violations[:5]}")
    result = AssignmentMarginals(CapacityAssignment(dict(sorted(selected.items())), dict(sorted(used.items())), total_utility), dict(sorted(probabilities.items())), samples, temperature)
    summary = {
        "model": model,
        "demands": len(demands),
        "candidate_resources": len(capacities),
        "components": len(components),
        "independent_components": independent_components,
        "sampled_conflict_components": sampled_components,
        "exact_conflict_components": exact_conflict_components,
        "auction_conflict_components": auction_conflict_components,
        "maximum_component_games": maximum_component,
        "map_matched": len(demands) - len(result.map_assignment.unmatched),
        "map_unmatched": len(result.map_assignment.unmatched),
        "map_capacity_violations": len(violations),
        "perturbation_samples_per_conflict_component": samples,
        "temperature": temperature,
        "large_component_auction_epsilon": auction_epsilon,
        "marginal_status": "exact independent softmax, exact small-component edge-perturbed MAP frequency, or epsilon-auction large-component frequency; not calibrated posterior probability",
    }
    return result, summary


def _player_probabilities(probabilities: Mapping[str | None, float], event_by_id: Mapping[str, PublicEvent]) -> tuple[dict[str, float], float]:
    players: Counter[str] = Counter()
    unknown = float(probabilities.get(None, 0.0))
    for event_id, probability in probabilities.items():
        if event_id is None:
            continue
        event = event_by_id.get(event_id)
        if event is not None:
            players[event.player_id] += probability
    return dict(players), unknown


def evaluate_assignment(rows: list[GameCandidates], marginal: AssignmentMarginals, event_by_id: Mapping[str, PublicEvent]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate masked unique-label games and describe collision sets without converting them into exact-ID truth."""

    unique_total = 0
    candidate_covered = 0
    top1 = 0
    top5 = 0
    reciprocal_rank = 0.0
    candidate_counts: list[float] = []
    predictions: list[tuple[float, bool, bool]] = []
    collision_rows = 0
    collision_covered = 0
    collision_mass: list[float] = []
    output_rows: list[dict[str, object]] = []
    by_family: dict[str, Counter[str]] = {family: Counter() for family in GLEE_FAMILIES}
    for row in rows:
        game = row.alignment.game
        raw = marginal.probabilities.get(game.game_id, {None: 1.0})
        players, unknown = _player_probabilities(raw, event_by_id)
        ranked = sorted(players.items(), key=lambda item: (-item[1], item[0]))
        top = [{"public_player_id": player_id, "probability": _rounded(probability)} for player_id, probability in ranked[:10]]
        map_event = marginal.map_assignment.selected.get(game.game_id)
        output_rows.append(
            {
                "game_id": game.game_id,
                "family": game.family,
                "identity_scope": game.identity_scope,
                "disclosed_label": game.opponent_name,
                "assignment_frontier_sequence": row.assignment_frontier_sequence,
                "self_public_frontier_sequence": row.alignment.self_event.frontier_sequence if row.alignment.self_event is not None else None,
                "candidate_event_count": len(row.candidates),
                "candidate_public_id_count": len({edge.event.player_id for edge in row.candidates}),
                "unknown_probability_prior": _rounded(row.unknown_probability_prior),
                "unknown_assignment_probability": _rounded(unknown),
                "map_event_id": map_event,
                "map_public_player_id": event_by_id[map_event].player_id if map_event in event_by_id else None,
                "true_public_player_ids": list(row.true_public_player_ids),
                "top_public_players": top,
            }
        )
        if len(row.true_public_player_ids) == 1:
            unique_total += 1
            by_family[game.family]["unique_total"] += 1
            target = row.true_public_player_ids[0]
            candidate_ids = {edge.event.player_id for edge in row.candidates}
            covered = target in candidate_ids
            candidate_covered += int(covered)
            by_family[game.family]["candidate_covered"] += int(covered)
            candidate_counts.append(float(len(candidate_ids)))
            rank = next((index for index, (player_id, _probability) in enumerate(ranked, start=1) if player_id == target), None)
            top1 += int(rank == 1)
            top5 += int(rank is not None and rank <= 5)
            reciprocal_rank += 1.0 / rank if rank is not None else 0.0
            by_family[game.family]["top1"] += int(rank == 1)
            by_family[game.family]["top5"] += int(rank is not None and rank <= 5)
            confidence = ranked[0][1] if ranked else 0.0
            predicted = ranked[0][0] if ranked and confidence > unknown else None
            predictions.append((confidence, predicted == target, predicted is not None))
        elif len(row.true_public_player_ids) > 1:
            collision_rows += 1
            target_set = set(row.true_public_player_ids)
            candidate_ids = {edge.event.player_id for edge in row.candidates}
            collision_covered += int(bool(target_set.intersection(candidate_ids)))
            collision_mass.append(sum(players.get(player_id, 0.0) for player_id in target_set))
    curves: list[dict[str, object]] = []
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        accepted = [correct for confidence, correct, predicted in predictions if predicted and confidence >= threshold]
        curves.append({"minimum_top_probability": threshold, "accepted_games": len(accepted), "accepted_fraction": _rounded(len(accepted) / unique_total if unique_total else None), "accuracy": _rounded(sum(accepted) / len(accepted) if accepted else None)})
    summary = {
        "unique_label_games": unique_total,
        "candidate_coverage": _rounded(candidate_covered / unique_total if unique_total else None),
        "median_candidate_public_ids": _rounded(statistics.median(candidate_counts) if candidate_counts else None),
        "top1_accuracy": _rounded(top1 / unique_total if unique_total else None),
        "top5_accuracy": _rounded(top5 / unique_total if unique_total else None),
        "mean_reciprocal_rank": _rounded(reciprocal_rank / unique_total if unique_total else None),
        "accepted_coverage_curve": curves,
        "collision_set_games": collision_rows,
        "collision_candidate_coverage": _rounded(collision_covered / collision_rows if collision_rows else None),
        "collision_target_set_probability_mean": _rounded(statistics.mean(collision_mass) if collision_mass else None),
        "by_family": {
            family: {
                "unique_label_games": values["unique_total"],
                "candidate_coverage": _rounded(values["candidate_covered"] / values["unique_total"] if values["unique_total"] else None),
                "top1_accuracy": _rounded(values["top1"] / values["unique_total"] if values["unique_total"] else None),
                "top5_accuracy": _rounded(values["top5"] / values["unique_total"] if values["unique_total"] else None),
            }
            for family, values in by_family.items()
        },
    }
    return summary, output_rows


class GleeJointAssignmentAnalysis:
    """Run Stage 2 game assignment against frozen local and public evidence without touching live services."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        history_database: Path,
        game_archive_root: Path,
        activity_summary: Path,
        identity_summary: Path,
        output_dir: Path,
        cache_root: Path | None = None,
        reporter_frontier: int | None = None,
        self_name: str = "DeepRMM-01",
        alignment_slack_s: float = 30.0,
        frontier_radius: int = 2,
        marginal_samples: int = 24,
        marginal_temperature: float = 0.35,
        unknown_alpha: float = 1.0,
        unknown_beta: float = 19.0,
        auction_epsilon: float = 0.02,
    ) -> None:
        if alignment_slack_s < 0 or frontier_radius < 0 or marginal_samples < 1 or marginal_temperature <= 0 or unknown_alpha <= 0 or unknown_beta <= 0 or auction_epsilon <= 0:
            raise ValueError("joint-assignment parameters are invalid")
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.activity_summary_path = activity_summary.resolve()
        self.identity_summary_path = identity_summary.resolve()
        self.output_dir = output_dir.resolve()
        self.cache_root = cache_root.resolve() if cache_root is not None else (self.reporter_database.parent.parent / "glee-joint-assignment-cache-v1").resolve()
        self.reporter_frontier = reporter_frontier
        self.self_name = self_name
        self.alignment_slack_s = alignment_slack_s
        self.frontier_radius = frontier_radius
        self.marginal_samples = marginal_samples
        self.marginal_temperature = marginal_temperature
        self.unknown_alpha = unknown_alpha
        self.unknown_beta = unknown_beta
        self.auction_epsilon = auction_epsilon

    @staticmethod
    def _event_row(event: PublicEvent) -> dict[str, object]:
        return {
            "contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "event_id": event.event_id,
            "source_change_sequence": event.source_change_sequence,
            "frontier_sequence": event.frontier_sequence,
            "family": event.family,
            "public_player_id": event.player_id,
            "observed_after": _iso(event.observed_after),
            "observed_by": _iso(event.observed_by),
            "games_delta": event.games_delta,
            "clean_rating_delta": event.clean_rating_delta,
            "rating_status": event.rating_status,
            "prior_event_gap_s": _rounded(event.prior_event_gap_s),
        }

    @staticmethod
    def _alignment_row(alignment: SelfAlignment) -> dict[str, object]:
        event = alignment.self_event
        return {
            "contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "game_id": alignment.game.game_id,
            "family": alignment.game.family,
            "started_at": _iso(alignment.game.started_at),
            "completed_at": _iso(alignment.game.completed_at),
            "history_rating_delta": alignment.game.rating_delta,
            "history_revision": None,
            "archive_path": alignment.game.archive_path,
            "archive_sha256": alignment.game.archive_sha256,
            "self_event_id": event.event_id if event is not None else None,
            "self_event_slot_index": alignment.self_slot_index,
            "self_public_frontier_sequence": event.frontier_sequence if event is not None else None,
            "self_public_clean_rating_delta": event.clean_rating_delta if event is not None else None,
            "method": alignment.method,
            "interval_distance_s": _rounded(alignment.distance_s),
        }

    @staticmethod
    def _candidate_row(row: GameCandidates) -> dict[str, object]:
        return {
            "contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "game_id": row.alignment.game.game_id,
            "family": row.alignment.game.family,
            "assignment_frontier_sequence": row.assignment_frontier_sequence,
            "true_public_player_ids": list(row.true_public_player_ids),
            "unknown_probability_prior": _rounded(row.unknown_probability_prior),
            "causal_rating_training_games": row.causal_rating_training_games,
            "candidates": [
                {
                    "event_id": edge.event.event_id,
                    "activity_utility": _rounded(edge.activity_utility),
                    "rating_utility": _rounded(edge.rating_utility),
                    "interval_distance_s": _rounded(edge.interval_distance_s),
                    "frontier_distance": edge.frontier_distance,
                    "publication_lag_s": _rounded(edge.publication_lag_s),
                }
                for edge in row.candidates
            ],
        }

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        alignment = summary["self_alignment"]
        candidates = summary["candidate_construction"]
        evaluations = summary["evaluation"]
        activity = evaluations["activity-only"]
        rating = evaluations["rating-only"]
        joint = evaluations["joint"]
        return "\n".join(
            [
                "# GLEE capacity-aware joint assignment v1",
                "",
                f"**Status:** Offline shadow analysis through reporter frontier `{summary['frontier_sequence']}`. It starts no matchmaking, sends no model calls, changes no live prompt, and publishes no identity or dossier update.",
                "",
                "## Self-capacity reconstruction",
                "",
                f"The joint interval alignment maps {alignment['aligned_games']:,} of {alignment['local_games']:,} authenticated local games to {alignment['self_public_capacity']:,} corrected DeepRMM-01 public slots, leaves {alignment['unmatched_games']:,} local games unmatched, leaves {alignment['unused_public_capacity']:,} public slots unused, and has {alignment['capacity_violations']:,} capacity violations. Unlike the Stage 0 greedy baseline, it cannot reuse self capacity; concurrently completed games may map across publication order when their causal intervals overlap.",
                "",
                "## Masked identity evaluation",
                "",
                f"The candidate builder retains {candidates['unique_candidate_events']:,} unique public events and scores every game before exposing its disclosed identity to the causal rating learner. On unique-label KI games, activity-only reaches candidate coverage {activity['candidate_coverage']}, top-one {activity['top1_accuracy']}, top-5 {activity['top5_accuracy']}, and MRR {activity['mean_reciprocal_rank']}; rating-only reaches top-one {rating['top1_accuracy']}; the joint model reaches top-one {joint['top1_accuracy']}, top-5 {joint['top5_accuracy']}, and MRR {joint['mean_reciprocal_rank']}.",
                "",
                "Each MAP assignment obeys event capacity globally within every connected rolling-window conflict component. Independent components use exact softmax marginals, while constrained components use deterministic edge-perturbed MAP frequencies. Those frequencies are structured uncertainty estimates, not calibrated posterior probabilities; `unknown` begins with a declared beta prior and is updated only from causally earlier unique-label coverage outcomes.",
                "",
                "## Evidence-conditioned reconstruction",
                "",
                "The evidence-conditioned model applies the label visible in KI games while retaining collision sets and `unknown`; hidden games remain label-blind. It is the retrospective corpus reconstruction, while activity-only, rating-only, and joint evaluations hide the current game's label and measure actual identification power.",
                "",
                "## Artifacts and limits",
                "",
                "`self-alignments.jsonl` references authenticated games and corrected self slots, `candidate-events.jsonl` stores each referenced public event once, `game-candidate-refs.jsonl` stores only event references and compact edge features, and one assignment file per model stores public-ID marginals. The public counter remains interval-censored, polling can miss or aggregate events, the rating likelihood is deliberately coarse, and collision labels are evaluated only as sets.",
                "",
            ]
        )

    def run(self) -> dict[str, object]:
        started = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(not path.name.startswith(".") for path in self.output_dir.iterdir()):
            raise FileExistsError(f"joint-assignment output directory is not empty: {self.output_dir}")
        activity_summary = json.loads(self.activity_summary_path.read_text(encoding="utf-8"))
        identity_summary = json.loads(self.identity_summary_path.read_text(encoding="utf-8"))
        if activity_summary.get("contract") != "glee-activity-eda-v2":
            raise ValueError("joint assignment requires a glee-activity-eda-v2 source")
        if identity_summary.get("contract") != "glee-public-identity-registry-v1":
            raise ValueError("joint assignment requires a glee-public-identity-registry-v1 source")
        frontier = int(self.reporter_frontier or activity_summary["source_frontier"]["frontier_sequence"])
        if frontier != int(activity_summary["source_frontier"]["frontier_sequence"]) or frontier != int(identity_summary["frontier_sequence"]):
            raise ValueError("joint-assignment source frontiers differ")
        self_ids = {family: str(player_id) for family, player_id in activity_summary["source_frontier"]["self_player_ids"].items()}
        observation_quantum = float(activity_summary["source_frontier"]["poll_interval_s"])
        helper = GleeActivityEDA(reporter_database=self.reporter_database, history_database=self.history_database, game_archive_root=self.game_archive_root, output_dir=self.output_dir, self_name=self.self_name, reporter_frontier=frontier, alignment_slack_s=self.alignment_slack_s)
        registry = FrozenIdentityRegistry(summary_path=self.identity_summary_path)
        if registry.frontier_sequence != frontier:
            raise ValueError("frozen identity artifacts and requested frontier differ")
        progress: dict[str, object] = {"contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT, "frontier_sequence": frontier, "completed_stages": []}

        def checkpoint(stage: str, detail: Mapping[str, object] | None = None) -> None:
            progress["completed_stages"].append({"stage": stage, "elapsed_s": round(time.monotonic() - started, 3), "detail": dict(detail or {})})
            _write_json(self.output_dir / "progress.json", progress)

        reporter = _read_only_database(self.reporter_database)
        try:
            reporter.execute("BEGIN")
            snapshot = helper._snapshot(reporter)
            frontier_sequences, frontier_completed = _frontiers(reporter, frontier)
            reporter.execute("ROLLBACK")
        finally:
            reporter.close()
        expected_source_hash = str(activity_summary["effective_event_reconstruction"]["source_rows_sha256"])
        expected_event_hash = str(activity_summary["effective_event_reconstruction"]["effective_events_sha256"])
        if identity_summary["source"]["source_rows_sha256"] != expected_source_hash:
            raise RuntimeError("Stage 0 and Stage 1 source receipts differ")
        if identity_summary["effective_event_receipt"]["effective_events_sha256"] != expected_event_hash:
            raise RuntimeError("Stage 0 and Stage 1 effective-event receipts differ")
        cache_path = self.cache_root / f"frontier-{frontier}" / "effective-events.sqlite3"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_built = False
        if cache_path.is_file():
            store = _EventStore(cache_path, create=False)
            cache_metadata = store.metadata()
            if cache_metadata.get("contract") != "glee-joint-effective-event-cache-v1" or int(cache_metadata.get("frontier_sequence") or 0) != frontier or cache_metadata.get("source_rows_sha256") != expected_source_hash or cache_metadata.get("effective_events_sha256") != expected_event_hash:
                store.close()
                raise RuntimeError(f"effective-event cache does not match frozen sources: {cache_path}")
            effective_summary = cache_metadata["effective_summary"]
        else:
            cache_built = True
            descriptor, temporary_name = tempfile.mkstemp(prefix=".effective-events-", suffix=".sqlite3", dir=cache_path.parent)
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            try:
                store = _EventStore(temporary_path)
                reporter = _read_only_database(self.reporter_database)
                try:
                    reporter.execute("BEGIN")
                    effective_summary = derive_effective_events(reporter, frontier_sequence=frontier, event_sink=store.add)
                    reporter.execute("ROLLBACK")
                finally:
                    reporter.close()
                if effective_summary["source_rows_sha256"] != expected_source_hash or effective_summary["effective_events_sha256"] != expected_event_hash:
                    store.close()
                    raise RuntimeError("effective-event cache build does not reproduce the Stage 0 source receipt")
                store.finish({"contract": "glee-joint-effective-event-cache-v1", "frontier_sequence": frontier, "source_rows_sha256": effective_summary["source_rows_sha256"], "effective_events_sha256": effective_summary["effective_events_sha256"], "effective_summary": effective_summary})
                store.close()
                temporary_path.replace(cache_path)
                store = _EventStore(cache_path, create=False)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        checkpoint("effective-event-cache", {"cache_built": cache_built, "cache_path": str(cache_path), "effective_events": effective_summary["totals"]["effective_event_rows"]})
        try:
            history_rows, history_inventory = helper._history_games(first_at=str(snapshot["first_started_at"]), completed_by=str(snapshot["frontier_completed_at"]))
            local_games, archive_inventory = helper._load_local_games(history_rows)
            checkpoint("local-game-load", {"local_games": len(local_games)})
            alignments: list[SelfAlignment] = []
            family_alignment: dict[str, dict[str, object]] = {}
            for family in GLEE_FAMILIES:
                family_rows, family_summary = align_self_games((game for game in local_games if game.family == family), store.player_events(family, self_ids[family]), alignment_slack_s=self.alignment_slack_s, observation_quantum_s=observation_quantum)
                alignments.extend(family_rows)
                family_alignment[family] = family_summary
            alignment_summary = {
                "local_games": sum(int(row["local_games"]) for row in family_alignment.values()),
                "self_public_capacity": sum(int(row["self_public_capacity"]) for row in family_alignment.values()),
                "aligned_games": sum(int(row["aligned_games"]) for row in family_alignment.values()),
                "unmatched_games": sum(int(row["unmatched_games"]) for row in family_alignment.values()),
                "unused_public_capacity": sum(int(row["unused_public_capacity"]) for row in family_alignment.values()),
                "capacity_violations": sum(int(row["capacity_violations"]) for row in family_alignment.values()),
                "by_family": family_alignment,
                "stage0_greedy_baseline": activity_summary["alignment"],
            }
            checkpoint("self-capacity-alignment", alignment_summary)
            candidate_rows, candidate_events, candidate_summary = build_game_candidates(alignments, event_store=store, registry=registry, self_ids=self_ids, frontier_sequences=frontier_sequences, frontier_completed=frontier_completed, frontier_radius=self.frontier_radius, alignment_slack_s=self.alignment_slack_s, observation_quantum_s=observation_quantum, unknown_alpha=self.unknown_alpha, unknown_beta=self.unknown_beta)
            graph_inventory: dict[str, object] = {}
            for model in ASSIGNMENT_MODELS:
                demands, capacities = _model_demands(candidate_rows, model)
                components = _assignment_components(demands)
                sizes = [len(component) for component in components]
                graph_inventory[model] = {"components": len(components), "maximum_component_games": max(sizes, default=0), "component_games_p50": _rounded(statistics.median(sizes) if sizes else None), "component_games_p90": _rounded(_quantile([float(size) for size in sizes], 0.9)), "candidate_resources": len(capacities)}
            candidate_summary["conflict_graphs"] = graph_inventory
            checkpoint("candidate-construction", {"candidate_events": len(candidate_events), "conflict_graphs": graph_inventory})
            model_summaries: dict[str, object] = {}
            evaluations: dict[str, object] = {}
            assignment_outputs: dict[str, list[dict[str, object]]] = {}
            for model in ASSIGNMENT_MODELS:
                marginal, model_summary = assign_with_marginals(candidate_rows, model=model, samples=self.marginal_samples, temperature=self.marginal_temperature, seed=f"{JOINT_ASSIGNMENT_ANALYSIS_CONTRACT}:{frontier}", auction_epsilon=self.auction_epsilon)
                evaluation, output_rows = evaluate_assignment(candidate_rows, marginal, candidate_events)
                model_summaries[model] = model_summary
                evaluations[model] = evaluation
                assignment_outputs[model] = output_rows
                checkpoint(f"assignment-{model}", model_summary)
        finally:
            store.close()
        history_digest = hashlib.sha256()
        for row in history_rows:
            history_digest.update(_canonical({"game_id": row["game_id"], "family": row["family"], "completed_at": _iso(row["completed_at"]), "record_sha256": row["history_record_sha256"]}).encode("utf-8") + b"\n")
        summary: dict[str, object] = {
            "contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only",
            "frontier_sequence": frontier,
            "source_frontier": snapshot,
            "parameters": {
                "alignment_slack_s": self.alignment_slack_s,
                "frontier_radius": self.frontier_radius,
                "marginal_samples": self.marginal_samples,
                "marginal_temperature": self.marginal_temperature,
                "unknown_alpha": self.unknown_alpha,
                "unknown_beta": self.unknown_beta,
                "auction_epsilon": self.auction_epsilon,
            },
            "sources": {
                "reporter_database": str(self.reporter_database),
                "history_database": str(self.history_database),
                "game_archive_root": str(self.game_archive_root),
                "activity_summary": {"path": str(self.activity_summary_path), "sha256": _file_digest(self.activity_summary_path)},
                "identity_summary": {"path": str(self.identity_summary_path), "sha256": _file_digest(self.identity_summary_path)},
                "effective_event_cache": {"path": str(cache_path), "contract": "glee-joint-effective-event-cache-v1", "source_rows_sha256": expected_source_hash, "effective_events_sha256": expected_event_hash},
                "source_rows_sha256": effective_summary["source_rows_sha256"],
                "effective_events_sha256": effective_summary["effective_events_sha256"],
                "selected_history_rows_sha256": history_digest.hexdigest(),
            },
            "history_inventory": history_inventory,
            "archive_inventory": archive_inventory,
            "self_alignment": alignment_summary,
            "candidate_construction": candidate_summary,
            "assignment": model_summaries,
            "evaluation": evaluations,
            "interpretation": {
                "masked_models": ["activity-only", "rating-only", "joint"],
                "retrospective_model": "evidence-conditioned",
                "current_game_label_used_in_masked_models": False,
                "capacity_reuse_allowed": False,
                "probabilities_calibrated": False,
                "unknown_mass_status": "causal beta-prior plus structured assignment frequency; calibration remains a promotion blocker",
            },
        }
        _write_jsonl(self.output_dir / "self-alignments.jsonl", (self._alignment_row(row) for row in sorted(alignments, key=lambda value: (value.game.completed_at, value.game.game_id))))
        _write_jsonl(self.output_dir / "candidate-events.jsonl", (self._event_row(event) for _, event in sorted(candidate_events.items(), key=lambda item: int(item[0]))))
        _write_jsonl(self.output_dir / "game-candidate-refs.jsonl", (self._candidate_row(row) for row in sorted(candidate_rows, key=lambda value: (value.alignment.game.completed_at, value.alignment.game.game_id))))
        for model, rows in assignment_outputs.items():
            _write_jsonl(self.output_dir / f"assignments-{model}.jsonl", ({"contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT, "schema_version": 1, "model": model, **row} for row in rows))
        _write_json(self.output_dir / "effective-events-summary.json", effective_summary)
        _write_json(self.output_dir / "summary.json", summary)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        progress = {"contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT, "frontier_sequence": frontier, "status": "completed", "completed_stages": ["effective-event-cache", "local-game-load", "self-capacity-alignment", "candidate-construction", *(f"assignment-{model}" for model in ASSIGNMENT_MODELS), "artifact-write"]}
        _write_json(self.output_dir / "progress.json", progress)
        artifact_names = ["README.md", "summary.json", "effective-events-summary.json", "progress.json", "self-alignments.jsonl", "candidate-events.jsonl", "game-candidate-refs.jsonl", *(f"assignments-{model}.jsonl" for model in ASSIGNMENT_MODELS)]
        manifest = {
            "contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "source_rows_sha256": effective_summary["source_rows_sha256"],
            "effective_events_sha256": effective_summary["effective_events_sha256"],
            "selected_history_rows_sha256": history_digest.hexdigest(),
            "parameters": summary["parameters"],
            "implementation_sha256": {
                "glee_joint_assignment_analysis.py": _file_digest(Path(__file__)),
                "glee_joint_assignment.py": _file_digest(Path(__file__).with_name("glee_joint_assignment.py")),
                "glee_effective_events.py": _file_digest(Path(__file__).with_name("glee_effective_events.py")),
                "glee_identity_registry.py": _file_digest(Path(__file__).with_name("glee_identity_registry.py")),
            },
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifact_names},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": JOINT_ASSIGNMENT_ANALYSIS_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": frontier, "self_alignment": alignment_summary, "candidate_construction": candidate_summary, "assignment": model_summaries, "evaluation": evaluations, "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
