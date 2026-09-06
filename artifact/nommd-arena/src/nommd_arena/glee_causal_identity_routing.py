"""Evaluate causal open-set identity fusion and predictive identity routing offline."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_behavior_channel_analysis import CHANNELS, UNKNOWN_ID, action_features, channel_features, chronological_game_splits, fit_fingerprint_profile
from .glee_behavior_fusion_analysis import FusionExample, _group_metrics, _prediction_record, fit_conditional_stacker
from .glee_joint_assignment_analysis import _timestamp
from .glee_presence_forecast import EMPTY_SERIES, EventSeries, ForecastBundle, FrozenPresenceRegistry, PlattCalibrator, AdditiveHazardModel, _normalized_identity, causal_state


CAUSAL_ROUTING_CONTRACT = "glee-causal-identity-routing-v1"
IDENTITY_ARMS = ("uniform-open-set", "static-presence", "full-presence", "uniform-behavior", "static-behavior", "full-behavior")
BEHAVIOR_FEATURES = ("presence", "timing", "action", "lexical", "discourse")
MINIMUM_GALLERY_GAMES = 3
FUSION_RIDGE = 0.1
FUSION_ITERATIONS = 300
FUSION_LEARNING_RATE = 0.03
ROUTING_BOOTSTRAP_REPLICATES = 5000
ROUTING_BOOTSTRAP_SEED = 20260812
OOV_ACTION_TOKEN = "__oov_action__"
OOV_ACTION_CONTEXT = "__oov_action_context__"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(path, "".join(_canonical(value) + "\n" for value in values))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _first_move_record(record: Mapping[str, object]) -> dict[str, object]:
    moves = [dict(move) for move in record.get("moves", []) if isinstance(move, Mapping)]
    return {**dict(record), "moves": moves[:1], "channel_counts": {"moves": min(1, len(moves))}}


def _move_action_tokens(move: Mapping[str, object]) -> Counter[str]:
    record = {"moves": [dict(move)]}
    tokens = action_features(record)
    tokens.pop(next((token for token in tokens if token.startswith("terminal-style|")), ""), None)
    return tokens


def _move_action_observations(move: Mapping[str, object]) -> list[tuple[str, str]]:
    """Turn frozen action tokens into proper next-choice outcomes conditional on visible context."""
    tokens = _move_action_tokens(move)
    kind = str(move.get("kind") or "")
    if kind == "proposal":
        selected = [token for token in tokens if token.startswith("action|")]
        marker = "|value="
    else:
        selected = [token for token in tokens if token.startswith("decision|")]
        marker = "|decision="
        if not selected:
            selected = [token for token in tokens if token.startswith("action|")]
            marker = "|value="
    observations = []
    for token in sorted(selected):
        condition, separator, outcome = token.rpartition(marker)
        if separator:
            observations.append((condition, f"{marker[1:]}{outcome}"))
    return observations


@dataclass(frozen=True)
class FrozenPresenceContext:
    """Hash-verified Stage 3 models and causal public state needed at game start."""

    registry: FrozenPresenceRegistry
    self_ids: Mapping[str, str]
    series: Mapping[tuple[str, str], EventSeries]
    family_event_times: Mapping[str, Sequence[float]]
    global_event_times: Sequence[float]
    first_seen_at: Mapping[tuple[str, str], float]
    frontier_sequences: Sequence[int]
    frontier_times: Sequence[float]
    models: Mapping[tuple[str, int], ForecastBundle]

    def sequence_at(self, stamp: float) -> int | None:
        index = bisect.bisect_right(self.frontier_times, stamp) - 1
        return int(self.frontier_sequences[index]) if index >= 0 else None

    def traffic(self, stamp: float, family: str) -> tuple[int, int]:
        family_values = self.family_event_times[family]
        family_count = bisect.bisect_right(family_values, stamp) - bisect.bisect_right(family_values, stamp - 60.0)
        global_count = bisect.bisect_right(self.global_event_times, stamp) - bisect.bisect_right(self.global_event_times, stamp - 60.0)
        return family_count, global_count

    def raw_prior(self, family: str, stamp: float, arm: str) -> tuple[int, dict[str, float]]:
        sequence = self.sequence_at(stamp)
        if sequence is None:
            raise RuntimeError("game starts before the public reporter frontier")
        candidates = self.registry.candidates(family, sequence, self_id=self.self_ids.get(family))
        family_traffic, global_traffic = self.traffic(stamp, family)
        states = {}
        for player_id in candidates:
            others = [self.series.get((other, player_id), EMPTY_SERIES) for other in GLEE_FAMILIES if other != family]
            states[player_id] = causal_state(player_id=player_id, stamp=stamp, first_seen_at=self.first_seen_at.get((family, player_id), self.frontier_times[0]), series=self.series.get((family, player_id), EMPTY_SERIES), other_series=others, family_traffic_60=family_traffic, global_traffic_60=global_traffic)
        if arm == "uniform":
            scores = {player_id: 1.0 for player_id in candidates}
        else:
            bundle = self.models[(family, 60)]
            scores = {player_id: bundle.probability(arm, state, horizon=60) for player_id, state in states.items()}
        return sequence, _normalized_identity(scores)


FROZEN_STAGE_3_MANIFEST_SHA256 = "283efba4956e5af5c86cfa58963ecef3c78c504260ebbb92dcb761aa7decb7f7"


def _load_presence_context(*, presence_dir: Path, event_cache: Path, identity_dir: Path, reporter_database: Path, activity_summary: Path, expected_manifest_sha256: str | None = FROZEN_STAGE_3_MANIFEST_SHA256) -> FrozenPresenceContext:
    manifest = json.loads((presence_dir / "manifest.json").read_text(encoding="utf-8"))
    if expected_manifest_sha256 is not None and _file_digest(presence_dir / "manifest.json") != expected_manifest_sha256:
        raise RuntimeError("presence receipt is not the frozen Stage 3 manifest")
    for name, receipt in manifest["artifacts"].items():
        if _file_digest(presence_dir / name) != receipt["sha256"]:
            raise RuntimeError(f"presence artifact hash mismatch: {name}")
    summary = json.loads((presence_dir / "summary.json").read_text(encoding="utf-8"))
    activity = json.loads(activity_summary.read_text(encoding="utf-8"))
    registry = FrozenPresenceRegistry(identity_dir)
    connection = sqlite3.connect(f"file:{event_cache.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = {str(row["key"]): json.loads(str(row["value"])) for row in connection.execute("SELECT key, value FROM metadata")}
        if metadata.get("effective_events_sha256") != summary["sources"]["effective_events_sha256"]:
            raise RuntimeError("effective-event cache differs from Stage 3")
        event_rows = [dict(row) for row in connection.execute("SELECT family, player_id, observed_by, games_delta FROM events ORDER BY frontier_sequence, family, player_id, source_change_sequence")]
    finally:
        connection.close()
    connection = sqlite3.connect(f"file:{reporter_database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        frontiers = connection.execute("SELECT sequence, completed_at FROM frontiers WHERE sequence <= ? ORDER BY sequence", (int(summary["sources"]["frontier_sequence"]),)).fetchall()
    finally:
        connection.close()
    frontier_sequences = [int(row["sequence"]) for row in frontiers]
    frontier_times = [float(_timestamp(row["completed_at"])) for row in frontiers]
    digest = hashlib.sha256()
    for sequence, stamp in zip(frontier_sequences, frontier_times, strict=True):
        digest.update(f"{sequence}\0{stamp:.6f}\n".encode("ascii"))
    if digest.hexdigest() != summary["sources"]["frontier_times_sha256"]:
        raise RuntimeError("reporter frontier timestamps differ from Stage 3")
    rows_by_series: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    family_event_times: dict[str, list[float]] = defaultdict(list)
    global_event_times: list[float] = []
    for row in event_rows:
        stamp = float(row["observed_by"])
        rows_by_series[(str(row["family"]), str(row["player_id"]))].append((stamp, int(row["games_delta"])))
        family_event_times[str(row["family"])].append(stamp)
        global_event_times.append(stamp)
    series = {key: EventSeries.build(values) for key, values in rows_by_series.items()}
    for values in family_event_times.values():
        values.sort()
    global_event_times.sort()
    first_seen_at = {(family, record.player_id): frontier_times[max(0, bisect.bisect_left(frontier_sequences, record.first_seen_sequence))] for family, records in registry.by_family.items() for record in records}
    model_rows = json.loads((presence_dir / "models.json").read_text(encoding="utf-8"))
    models: dict[tuple[str, int], ForecastBundle] = {}
    for family, horizons in model_rows.items():
        for horizon_text, values in horizons.items():
            def hazard(raw: Mapping[str, object]) -> AdditiveHazardModel:
                return AdditiveHazardModel(tuple(raw["feature_names"]), float(raw["prevalence"]), raw["effects"], int(raw["training_rows"]), int(raw["training_positives"]))
            calibrators = {arm: PlattCalibrator(float(raw["slope"]), float(raw["intercept"]), int(raw["rows"])) for arm, raw in values["calibrators"].items()}
            models[(family, int(horizon_text))] = ForecastBundle(hazard(values["renewal"]), hazard(values["full"]), calibrators)
    return FrozenPresenceContext(registry, {str(family): str(player_id) for family, player_id in activity["source_frontier"]["self_player_ids"].items()}, series, family_event_times, global_event_times, first_seen_at, frontier_sequences, frontier_times, models)


def _collapse_prior(raw: Mapping[str, float], gallery: Sequence[str]) -> dict[str, float]:
    gallery_set = set(gallery)
    values = {player_id: float(raw.get(player_id, 0.0)) for player_id in gallery}
    values[UNKNOWN_ID] = sum(float(value) for player_id, value in raw.items() if player_id not in gallery_set)
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


@dataclass(frozen=True)
class CausalFusionExample:
    """One first-move identity query with 3 frozen pregame priors."""

    game_id: str
    family: str
    role: str
    split: str
    started_at: str
    true_public_player_id: str
    target: str
    gallery: tuple[str, ...]
    priors: Mapping[str, Mapping[str, float]]
    channel_logits: Mapping[str, Mapping[str, float]]
    evidence: Mapping[str, int]

    def fusion(self, prior_arm: str) -> FusionExample:
        candidates = (*self.gallery, UNKNOWN_ID)
        features = {
            label: {
                "presence": math.log(max(1e-12, float(self.priors[prior_arm][label]))),
                **{channel: float(self.channel_logits[channel].get(label, 0.0)) for channel in CHANNELS},
            }
            for label in candidates
        }
        return FusionExample(self.game_id, self.family, self.role, self.split, self.started_at, candidates, self.target, self.true_public_player_id, features, {}, self.evidence)


def _opponent_role(record: Mapping[str, object]) -> str:
    for move in record.get("moves", []):
        context = move.get("context") if isinstance(move, Mapping) and isinstance(move.get("context"), Mapping) else {}
        if context.get("opponent_role"):
            return str(context["opponent_role"]).casefold()
    return "none"


def _gallery(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    output = {}
    for family in GLEE_FAMILIES:
        support = Counter(str(row["public_player_id"]) for row in records if row["family"] == family and assignments.get(str(row["game_id"])) == "train")
        output[family] = tuple(sorted(identity for identity, count in support.items() if count >= MINIMUM_GALLERY_GAMES))
    return output


def _build_channel_profiles(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], galleries: Mapping[str, Sequence[str]], channel_summary: Mapping[str, object]) -> tuple[dict[tuple[str, str, str], object], dict[tuple[str, str], Counter[str]], dict[tuple[str, str], float]]:
    snapshots = {str(record["game_id"]): _first_move_record(record) for record in records}
    vectors: dict[tuple[str, str], Counter[str]] = {}
    profiles: dict[tuple[str, str, str], object] = {}
    temperatures: dict[tuple[str, str], float] = {}
    for channel in CHANNELS:
        for game_id, snapshot in snapshots.items():
            vector = channel_features(snapshot, channel)
            if channel == "action":
                vector = Counter({token: count for token, count in vector.items() if not token.startswith("terminal-style|")})
            vectors[(game_id, channel)] = vector
        for family in GLEE_FAMILIES:
            selected = channel_summary["families"][family][channel].get("selected_hyperparameters")
            if not selected:
                continue
            temperatures[(family, channel)] = float(selected["temperature"])
            family_records = [record for record in records if record["family"] == family]
            train = [record for record in family_records if assignments.get(str(record["game_id"])) == "train"]
            calibration = [record for record in family_records if assignments.get(str(record["game_id"])) == "calibration"]
            local_vectors = {str(record["game_id"]): vectors[(str(record["game_id"]), channel)] for record in family_records}
            profiles[(family, channel, "calibration")] = fit_fingerprint_profile(train, local_vectors, candidates=galleries[family], alpha=float(selected["alpha"]), channel=channel)
            profiles[(family, channel, "test")] = fit_fingerprint_profile(train + calibration, local_vectors, candidates=galleries[family], alpha=float(selected["alpha"]), channel=channel)
    return profiles, vectors, temperatures


def _build_examples(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], galleries: Mapping[str, tuple[str, ...]], profiles: Mapping[tuple[str, str, str], object], vectors: Mapping[tuple[str, str], Counter[str]], temperatures: Mapping[tuple[str, str], float], presence: FrozenPresenceContext, expected_test_posteriors: Mapping[str, Mapping[str, object]]) -> tuple[list[CausalFusionExample], dict[str, object]]:
    examples = []
    parity_rows = 0
    parity_mismatches = 0
    maximum_parity_error = 0.0
    for record in records:
        split = assignments.get(str(record["game_id"]))
        if split not in {"calibration", "test"}:
            continue
        family = str(record["family"])
        stamp = float(_timestamp(record["started_at"]))
        priors = {}
        sequence = None
        causal_candidates: set[str] = set()
        for arm in ("uniform", "static-rate", "full"):
            arm_sequence, raw = presence.raw_prior(family, stamp, arm)
            sequence = arm_sequence if sequence is None else sequence
            if sequence != arm_sequence:
                raise RuntimeError("presence arms disagree on causal frontier")
            if split == "test" and arm in {"static-rate", "full"}:
                expected_row = expected_test_posteriors[str(record["game_id"])]
                expected = {str(candidate["public_player_id"]): float(candidate["identity_probability"][arm]["60"]) for candidate in expected_row["candidates"]} | {UNKNOWN_ID: float(expected_row["unknown_identity_mass"])}
                labels = set(expected) | set(raw)
                error = max((abs(float(expected.get(label, 0.0)) - float(raw.get(label, 0.0))) for label in labels), default=0.0)
                maximum_parity_error = max(maximum_parity_error, error)
                parity_mismatches += int(error > 1e-15 or int(expected_row["assignment_frontier_sequence"]) != sequence)
                parity_rows += 1
            causal_candidates.update(label for label in raw if label != UNKNOWN_ID)
            priors[arm] = _collapse_prior(raw, galleries[family])
        channel_logits = {}
        evidence = {}
        for channel in CHANNELS:
            profile = profiles.get((family, channel, split))
            if profile is None:
                channel_logits[channel] = {candidate: 0.0 for candidate in galleries[family]}
                evidence[channel] = 0
                continue
            scores, count = profile.scores(vectors[(str(record["game_id"]), channel)])
            temperature = temperatures[(family, channel)]
            channel_logits[channel] = {candidate: score / temperature for candidate, score in scores.items()} | {UNKNOWN_ID: 0.0}
            evidence[channel] = count
        true_id = str(record["public_player_id"])
        target = true_id if true_id in galleries[family] and true_id in causal_candidates else UNKNOWN_ID
        examples.append(CausalFusionExample(str(record["game_id"]), family, _opponent_role(record), split, str(record["started_at"]), true_id, target, galleries[family], priors, channel_logits, evidence))
    return sorted(examples, key=lambda value: (value.started_at, value.game_id)), {"checked_arm_game_rows": parity_rows, "mismatches": parity_mismatches, "maximum_probability_error": maximum_parity_error, "tolerance": 1e-15}


def _raw_prediction(example: CausalFusionExample, arm: str, prior: str) -> dict[str, object]:
    fusion = example.fusion(prior)
    row = _prediction_record(fusion, arm, example.priors[prior])
    row["contract"] = CAUSAL_ROUTING_CONTRACT
    return row


def _fitted_prediction(example: CausalFusionExample, arm: str, prior: str, stacker: object) -> dict[str, object]:
    fusion = example.fusion(prior)
    row = _prediction_record(fusion, arm, stacker.probabilities(fusion))
    row["contract"] = CAUSAL_ROUTING_CONTRACT
    return row


@dataclass(frozen=True)
class ActionPackage:
    """Population-smoothed next-action distributions conditional on visible move context."""

    probabilities: Mapping[str, Mapping[str, float]]

    def probability(self, condition: str, outcome: str) -> float:
        distribution = self.probabilities.get(condition, self.probabilities[OOV_ACTION_CONTEXT])
        return float(distribution.get(outcome, distribution[OOV_ACTION_TOKEN]))

    def nll(self, observations: Sequence[tuple[str, str]]) -> float:
        if not observations:
            return 0.0
        return -sum(math.log(max(1e-12, self.probability(condition, outcome))) for condition, outcome in observations) / len(observations)


def _action_packages(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], galleries: Mapping[str, Sequence[str]], split: str, action_alphas: Mapping[str, float]) -> dict[tuple[str, str], ActionPackage]:
    allowed = {"train"} if split == "calibration" else {"train", "calibration"}
    output: dict[tuple[str, str], ActionPackage] = {}
    for family in GLEE_FAMILIES:
        selected = [record for record in records if record["family"] == family and assignments.get(str(record["game_id"])) in allowed]
        population: dict[str, Counter[str]] = defaultdict(Counter)
        identities: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for record in selected:
            for move in record.get("moves", []):
                if not isinstance(move, Mapping):
                    continue
                for condition, outcome in _move_action_observations(move):
                    population[condition][outcome] += 1
                    identities[str(record["public_player_id"])][condition][outcome] += 1
        global_counts: Counter[str] = Counter()
        for counts in population.values():
            global_counts.update(counts)
        global_counts[OOV_ACTION_TOKEN] += 1
        global_total = sum(global_counts.values())
        background: dict[str, dict[str, float]] = {OOV_ACTION_CONTEXT: {outcome: count / global_total for outcome, count in sorted(global_counts.items())}}
        for condition, counts in population.items():
            counts = counts.copy()
            counts[OOV_ACTION_TOKEN] += 1
            total = sum(counts.values())
            background[condition] = {outcome: count / total for outcome, count in sorted(counts.items())}
        output[(family, UNKNOWN_ID)] = ActionPackage(background)
        alpha = float(action_alphas[family])
        for identity in galleries[family]:
            probabilities: dict[str, dict[str, float]] = {}
            for condition, population_distribution in background.items():
                if condition == OOV_ACTION_CONTEXT:
                    probabilities[condition] = dict(population_distribution)
                    continue
                counts = identities[identity][condition]
                denominator = sum(counts.values()) + alpha
                probabilities[condition] = {outcome: (counts[outcome] + alpha * population_probability) / denominator for outcome, population_probability in population_distribution.items()}
            output[(family, identity)] = ActionPackage(probabilities)
    return output


def _mixture_nll(observations: Sequence[tuple[str, str]], posterior: Mapping[str, float], packages: Mapping[str, ActionPackage]) -> float:
    if not observations:
        return 0.0
    total = 0.0
    for condition, outcome in observations:
        probability = sum(float(weight) * packages[label].probability(condition, outcome) for label, weight in posterior.items())
        total -= math.log(max(1e-12, probability))
    return total / len(observations)


def _bootstrap_interval(rows: Sequence[Mapping[str, object]], left: str, right: str) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["true_public_player_id"])].append(float(row[left]) - float(row[right]))
    identities = sorted(grouped)
    if not identities:
        return {"left": left, "right": right, "identity_clusters": 0, "replicates": ROUTING_BOOTSTRAP_REPLICATES, "identity_macro_difference": None, "lower_95": None, "upper_95": None}
    identity_means = {identity: sum(values) / len(values) for identity, values in grouped.items()}
    generator = random.Random(ROUTING_BOOTSTRAP_SEED)
    samples = []
    for _ in range(ROUTING_BOOTSTRAP_REPLICATES):
        selected = [identities[generator.randrange(len(identities))] for _ in identities]
        samples.append(sum(identity_means[identity] for identity in selected) / len(selected))
    samples.sort()
    return {"left": left, "right": right, "identity_clusters": len(identities), "replicates": ROUTING_BOOTSTRAP_REPLICATES, "identity_macro_difference": sum(identity_means.values()) / len(identity_means), "lower_95": samples[math.floor(0.025 * len(samples))], "upper_95": samples[min(len(samples) - 1, math.floor(0.975 * len(samples)))]}


def _routing_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def mean(key: str, selected: Sequence[Mapping[str, object]] = rows) -> float | None:
        return sum(float(row[key]) for row in selected) / len(selected) if selected else None
    known = [row for row in rows if row["evaluation_target"] != UNKNOWN_ID]
    unknown = [row for row in rows if row["evaluation_target"] == UNKNOWN_ID]
    by_family = {family: {key: mean(key, [row for row in rows if row["family"] == family]) for key in ("generic_nll", "oracle_nll", "soft_nll", "hard_nll")} for family in GLEE_FAMILIES}
    generic = mean("generic_nll")
    oracle = mean("oracle_nll")
    soft = mean("soft_nll")
    denominator = float(generic) - float(oracle) if generic is not None and oracle is not None else 0.0
    return {"games": len(rows), "known_games": len(known), "unknown_games": len(unknown), "generic_nll": generic, "oracle_nll": oracle, "soft_nll": soft, "hard_nll": mean("hard_nll"), "soft_known_nll": mean("soft_nll", known), "soft_unknown_nll": mean("soft_nll", unknown), "generic_unknown_nll": mean("generic_nll", unknown), "hard_named_route_coverage": sum(bool(row["hard_named_route"]) for row in rows) / len(rows) if rows else None, "hard_wrong_route_excess_nll": mean("hard_minus_oracle_nll", [row for row in known if row["hard_route_target"] != row["evaluation_target"]]), "oracle_gap_recovered": (float(generic) - float(soft)) / denominator if denominator > 0.0 and soft is not None else None, "by_family": by_family, "bootstrap": {"oracle_minus_generic": _bootstrap_interval(rows, "oracle_nll", "generic_nll"), "soft_minus_generic": _bootstrap_interval(rows, "soft_nll", "generic_nll"), "hard_minus_generic": _bootstrap_interval(rows, "hard_nll", "generic_nll")}}


class GleeCausalIdentityRoutingAnalysis:
    """Run frozen causal identity fusion and predictive routing without live authority."""

    def __init__(self, *, presence_dir: Path, behavior_dir: Path, channel_dir: Path, event_cache: Path, identity_dir: Path, reporter_database: Path, activity_summary: Path, output_dir: Path) -> None:
        self.presence_dir = presence_dir.resolve()
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.event_cache = event_cache.resolve()
        self.identity_dir = identity_dir.resolve()
        self.reporter_database = reporter_database.resolve()
        self.activity_summary = activity_summary.resolve()
        self.output_dir = output_dir.resolve()

    def _verify(self, directory: Path, artifact: str, expected_manifest: str) -> dict[str, object]:
        if _file_digest(directory / "manifest.json") != expected_manifest:
            raise RuntimeError(f"source manifest differs from frozen protocol: {directory}")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if _file_digest(directory / artifact) != manifest["artifacts"][artifact]["sha256"]:
            raise RuntimeError(f"source artifact hash mismatch: {artifact}")
        return manifest

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"causal-routing output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        behavior_manifest = self._verify(self.behavior_dir, "behavior-games.jsonl", "206e0e0f8ea2b2a2898e7775f2af36c9ebcf5a9697bc63e0e435ecc7eceba1b3")
        self._verify(self.channel_dir, "summary.json", "aebf601db25e0383781293e7a50a72ee2320d2d4b65b5fe2f58d822c4043cbaf")
        presence = _load_presence_context(presence_dir=self.presence_dir, event_cache=self.event_cache, identity_dir=self.identity_dir, reporter_database=self.reporter_database, activity_summary=self.activity_summary)
        records = _load_jsonl(self.behavior_dir / "behavior-games.jsonl")
        records_by_game = {str(record["game_id"]): record for record in records}
        channel_summary = json.loads((self.channel_dir / "summary.json").read_text(encoding="utf-8"))
        expected_test_posteriors = {str(row["game_id"]): row for row in _load_jsonl(self.presence_dir / "game-presence-posteriors.jsonl")}
        assignments, boundaries = chronological_game_splits(records, train_fraction=float(channel_summary["design"]["train_fraction"]), calibration_fraction=float(channel_summary["design"]["calibration_fraction"]))
        galleries = _gallery(records, assignments)
        profiles, vectors, temperatures = _build_channel_profiles(records, assignments, galleries, channel_summary)
        examples, presence_parity = _build_examples(records, assignments, galleries, profiles, vectors, temperatures, presence, expected_test_posteriors)
        if presence_parity["mismatches"]:
            raise RuntimeError(f"Stage 3 causal-prior reconstruction failed parity: {presence_parity}")
        calibration = [example for example in examples if example.split == "calibration"]
        test = [example for example in examples if example.split == "test"]
        if len(calibration) != 376 or len(test) != 379:
            raise RuntimeError(f"unexpected causal-fusion split sizes: calibration={len(calibration)} test={len(test)}")
        models: dict[str, object] = {}
        calibration_predictions: list[dict[str, object]] = []
        test_predictions: list[dict[str, object]] = []
        for arm, prior in (("uniform-open-set", "uniform"), ("static-presence", "static-rate"), ("full-presence", "full")):
            calibration_rows = [_raw_prediction(example, arm, prior) for example in calibration]
            test_rows = [_raw_prediction(example, arm, prior) for example in test]
            models[arm] = {"kind": "frozen-causal-presence", "metrics": _group_metrics(test_rows)}
            calibration_predictions.extend(calibration_rows)
            test_predictions.extend(test_rows)
        stackers = {}
        for arm, prior in (("uniform-behavior", "uniform"), ("static-behavior", "static-rate"), ("full-behavior", "full")):
            stacker = fit_conditional_stacker([example.fusion(prior) for example in calibration], feature_names=BEHAVIOR_FEATURES, ridge_lambda=FUSION_RIDGE, iterations=FUSION_ITERATIONS, learning_rate=FUSION_LEARNING_RATE)
            stackers[arm] = stacker
            calibration_rows = [_fitted_prediction(example, arm, prior, stacker) for example in calibration]
            test_rows = [_fitted_prediction(example, arm, prior, stacker) for example in test]
            models[arm] = {"kind": "causal-first-move-conditional-fusion", "features": list(BEHAVIOR_FEATURES), "weights": dict(stacker.weights), "scales": dict(stacker.scales), "unknown_bias": stacker.unknown_bias, "metrics": _group_metrics(test_rows)}
            calibration_predictions.extend(calibration_rows)
            test_predictions.extend(test_rows)
        action_alphas = {family: float(channel_summary["families"][family]["action"]["selected_hyperparameters"]["alpha"]) for family in GLEE_FAMILIES}
        packages_by_split = {split: _action_packages(records, assignments, galleries, split, action_alphas) for split in ("calibration", "test")}
        routing_rows: list[dict[str, object]] = []
        routing_eligible: Counter[tuple[str, str]] = Counter()
        routing_excluded: Counter[tuple[str, str]] = Counter()
        for example in examples:
            record = records_by_game[example.game_id]
            if len(record.get("moves", [])) < 2:
                continue
            routing_eligible[(example.split, example.family)] += 1
            observations = _move_action_observations(record["moves"][1])
            if not observations:
                routing_excluded[(example.split, example.family)] += 1
                continue
            packages = {label: packages_by_split[example.split][(example.family, label)] for label in (*example.gallery, UNKNOWN_ID)}
            posterior = stackers["full-behavior"].probabilities(example.fusion("full"))
            generic = packages[UNKNOWN_ID]
            oracle = packages[example.target] if example.target != UNKNOWN_ID else generic
            hard_target = max(posterior, key=lambda label: (posterior[label], label))
            hard = packages[hard_target] if hard_target != UNKNOWN_ID else generic
            generic_nll = generic.nll(observations)
            oracle_nll = oracle.nll(observations)
            soft_nll = _mixture_nll(observations, posterior, packages)
            hard_nll = hard.nll(observations)
            routing_rows.append({"contract": CAUSAL_ROUTING_CONTRACT, "game_id": example.game_id, "family": example.family, "split": example.split, "started_at": example.started_at, "true_public_player_id": example.true_public_player_id, "evaluation_target": example.target, "posterior_true_probability": posterior[example.target], "posterior_unknown_probability": posterior[UNKNOWN_ID], "hard_route_target": hard_target, "hard_named_route": hard_target != UNKNOWN_ID, "target_observation_count": len(observations), "generic_nll": generic_nll, "oracle_nll": oracle_nll, "soft_nll": soft_nll, "hard_nll": hard_nll, "hard_minus_oracle_nll": hard_nll - oracle_nll})
        routing_test = [row for row in routing_rows if row["split"] == "test"]
        routing_calibration = [row for row in routing_rows if row["split"] == "calibration"]
        expected_test_eligible = {"bargaining": 59, "negotiation": 81, "persuasion": 40}
        observed_test_eligible = {family: routing_eligible[("test", family)] for family in GLEE_FAMILIES}
        if observed_test_eligible != expected_test_eligible:
            raise RuntimeError(f"unexpected pre-exclusion predictive-routing test counts: {observed_test_eligible}")
        routing_metrics = _routing_metrics(routing_test)
        full_metrics = models["full-behavior"]["metrics"]["pooled"]
        full_presence = models["full-presence"]["metrics"]["pooled"]
        uniform_behavior = models["uniform-behavior"]["metrics"]["pooled"]
        fusion_pass = bool(full_metrics["negative_log_likelihood"] < full_presence["negative_log_likelihood"] and full_metrics["negative_log_likelihood"] < uniform_behavior["negative_log_likelihood"] and full_metrics["multiclass_brier"] < full_presence["multiclass_brier"] and full_metrics["multiclass_brier"] < uniform_behavior["multiclass_brier"] and (full_metrics["known_conditional_top_5_accuracy"] > full_presence["known_conditional_top_5_accuracy"] or full_metrics["known_conditional_mean_reciprocal_rank"] > full_presence["known_conditional_mean_reciprocal_rank"]) and full_metrics["unknown_recall"] >= full_presence["unknown_recall"] - 0.05)
        routing_pass = bool(routing_metrics["oracle_nll"] is not None and routing_metrics["generic_nll"] is not None and routing_metrics["oracle_nll"] < routing_metrics["generic_nll"] and routing_metrics["bootstrap"]["soft_minus_generic"]["upper_95"] is not None and routing_metrics["bootstrap"]["soft_minus_generic"]["upper_95"] < 0.0 and routing_metrics["soft_unknown_nll"] is not None and routing_metrics["generic_unknown_nll"] is not None and float(routing_metrics["soft_unknown_nll"]) <= float(routing_metrics["generic_unknown_nll"]) + 0.02)
        unknown_by_family = {family: sum(example.target == UNKNOWN_ID for example in test if example.family == family) for family in GLEE_FAMILIES}
        gallery_inventory = {family: {"public_player_ids": list(galleries[family]), "gallery_size": len(galleries[family]), "test_known_games": sum(example.target != UNKNOWN_ID for example in test if example.family == family), "test_unknown_games": unknown_by_family[family]} for family in GLEE_FAMILIES}
        summary = {"contract": CAUSAL_ROUTING_CONTRACT, "schema_version": 1, "status": "offline-shadow-only", "sources": {"presence_manifest_sha256": _file_digest(self.presence_dir / "manifest.json"), "behavior_manifest_sha256": _file_digest(self.behavior_dir / "manifest.json"), "channel_manifest_sha256": _file_digest(self.channel_dir / "manifest.json"), "frontier_sequence": behavior_manifest.get("frontier_sequence")}, "design": {"chronological_boundaries": boundaries, "presence_reconstruction_parity": presence_parity, "gallery_minimum_train_games": MINIMUM_GALLERY_GAMES, "galleries": {family: list(values) for family, values in galleries.items()}, "calibration_games": len(calibration), "test_games": len(test), "test_unknown_games": sum(unknown_by_family.values()), "test_unknown_games_by_family": unknown_by_family, "identity_observation": "pregame 60-second public prior plus first opponent move", "routing_target": "second opponent move conditional action choice", "routing_pre_exclusion_test_games_by_family": observed_test_eligible, "routing_target_exclusions_by_split_family": {f"{split}:{family}": routing_excluded[(split, family)] for split in ("calibration", "test") for family in GLEE_FAMILIES}, "routing_test_games": len(routing_test), "routing_calibration_games": len(routing_calibration), "action_alphas": action_alphas, "bootstrap_replicates": ROUTING_BOOTSTRAP_REPLICATES, "test_selected_tuning": False}, "identity_models": models, "routing": routing_metrics, "gates": {"causal_fusion_pass": fusion_pass, "predictive_routing_pass": routing_pass, "live_authority": False, "economic_decision_value_evaluated": False}}
        calibration_predictions.sort(key=lambda row: (str(row["started_at"]), str(row["arm"]), str(row["game_id"])))
        test_predictions.sort(key=lambda row: (str(row["started_at"]), str(row["arm"]), str(row["game_id"])))
        routing_rows.sort(key=lambda row: (str(row["started_at"]), str(row["game_id"])))
        model_artifact = {"fusion": {arm: {"features": list(model.feature_names), "scales": dict(model.scales), "weights": dict(model.weights), "unknown_bias": model.unknown_bias} for arm, model in stackers.items()}, "routing": {"kind": "family-conditional-hierarchical-action-package", "family_action_alphas": action_alphas, "unknown_route": "family population", "target": "second opponent move"}}
        _write_json(self.output_dir / "summary.json", summary)
        _write_json(self.output_dir / "models.json", model_artifact)
        _write_json(self.output_dir / "gallery-inventory.json", gallery_inventory)
        _write_jsonl(self.output_dir / "identity-calibration-predictions.jsonl", calibration_predictions)
        _write_jsonl(self.output_dir / "identity-test-predictions.jsonl", test_predictions)
        _write_jsonl(self.output_dir / "routing-predictions.jsonl", routing_rows)
        artifacts = ("summary.json", "models.json", "gallery-inventory.json", "identity-calibration-predictions.jsonl", "identity-test-predictions.jsonl", "routing-predictions.jsonl")
        protocol_path = Path(__file__).resolve().parents[2] / "protocols" / "glee-causal-identity-routing-v1.md"
        manifest = {"contract": CAUSAL_ROUTING_CONTRACT, "schema_version": 1, "sources": summary["sources"], "implementation_sha256": {"glee-causal-identity-routing-v1.md": _file_digest(protocol_path), "glee_causal_identity_routing.py": _file_digest(Path(__file__)), "glee_presence_forecast.py": _file_digest(Path(__file__).with_name("glee_presence_forecast.py")), "glee_behavior_channel_analysis.py": _file_digest(Path(__file__).with_name("glee_behavior_channel_analysis.py")), "glee_behavior_fusion_analysis.py": _file_digest(Path(__file__).with_name("glee_behavior_fusion_analysis.py"))}, "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts}}
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": CAUSAL_ROUTING_CONTRACT, "output_dir": str(self.output_dir), "identity_test_games": len(test), "representative_unknown_games": sum(example.target == UNKNOWN_ID for example in test), "routing_test_games": len(routing_test), "identity": {arm: value["metrics"]["pooled"] for arm, value in models.items()}, "routing": routing_metrics, "gates": summary["gates"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
