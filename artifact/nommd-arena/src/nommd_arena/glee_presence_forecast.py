"""Evaluate causal GLEE activity-presence forecasts and linked-policy features offline."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import sqlite3
import statistics
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_activity_eda import GLEE_FAMILIES, _file_digest, _read_only_database
from .glee_behavior_channel_analysis import UNKNOWN_ID, chronological_game_splits
from .glee_joint_assignment_analysis import _timestamp


PRESENCE_FORECAST_CONTRACT = "glee-causal-presence-forecast-v1"
HORIZONS = (60, 300, 900)
LANDMARK_STEP_S = 60
WARMUP_S = 900
UNKNOWN_IDENTITY_MASS = 0.05
CATEGORY_SMOOTHING = 200.0
CATEGORY_EFFECT_CLIP = 3.0
PLATT_RIDGE = 0.01
PLATT_ITERATIONS = 40
RENEWAL_FEATURES = ("player", "idle", "last_gap", "lifetime_rate", "recent_60", "recent_300", "recent_900", "session_age", "regime")
FULL_FEATURES = (*RENEWAL_FEATURES, "utc_block", "family_traffic", "global_traffic", "other_family_recent")
ARMS = ("population", "static-rate", "renewal", "full")


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


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(40.0, value))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(-40.0, value))
    return exponent / (1.0 + exponent)


def _logit(probability: float) -> float:
    bounded = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(bounded / (1.0 - bounded))


def _bin(value: float | None, boundaries: Sequence[float], *, missing: str = "missing") -> str:
    if value is None or not math.isfinite(value):
        return missing
    for boundary in boundaries:
        if value <= boundary:
            return f"le-{boundary:g}"
    return f"gt-{boundaries[-1]:g}"


@dataclass(frozen=True)
class IdentityRecord:
    """Temporal public presence and competitive-status intervals for one family identity."""

    family: str
    player_id: str
    label: str | None
    first_seen_sequence: int
    presence: tuple[tuple[int, int | None], ...]
    competitive: tuple[tuple[int, int | None, bool], ...]

    def present_at(self, sequence: int) -> bool:
        return any(start <= sequence and (end is None or sequence < end) for start, end in self.presence)

    def competitive_at(self, sequence: int) -> bool:
        return any(start <= sequence and (end is None or sequence < end) and competitive for start, end, competitive in self.competitive)


class FrozenPresenceRegistry:
    """Load hash-verified temporal identities and expose causal competitive candidate sets."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("contract") != "glee-public-identity-registry-v1":
            raise ValueError("presence forecasting requires the Stage 1 identity registry")
        path = self.root / "identity-registry.jsonl"
        if _file_digest(path) != manifest["artifacts"]["identity-registry.jsonl"]["sha256"]:
            raise RuntimeError("identity-registry artifact hash mismatch")
        self.manifest_sha256 = _file_digest(self.root / "manifest.json")
        self.frontier_sequence = int(manifest["frontier_sequence"])
        self.records: dict[tuple[str, str], IdentityRecord] = {}
        self.by_family: dict[str, list[IdentityRecord]] = defaultdict(list)
        for row in _load_jsonl(path):
            record = IdentityRecord(
                family=str(row["family"]),
                player_id=str(row["player_id"]),
                label=str(row["current_label"]) if row.get("current_label") is not None else None,
                first_seen_sequence=int(row["first_seen_sequence"]),
                presence=tuple((int(value["start_sequence"]), int(value["end_sequence_exclusive"]) if value["end_sequence_exclusive"] is not None else None) for value in row["presence_intervals"]),
                competitive=tuple((int(value["start_sequence"]), int(value["end_sequence_exclusive"]) if value["end_sequence_exclusive"] is not None else None, not bool(value["is_baseline"]) and not bool(value["is_benchmark"])) for value in row["classification_intervals"]),
            )
            self.records[(record.family, record.player_id)] = record
            self.by_family[record.family].append(record)
        for family in self.by_family:
            self.by_family[family].sort(key=lambda value: value.player_id)

    def candidates(self, family: str, sequence: int, *, self_id: str | None = None) -> tuple[str, ...]:
        return tuple(record.player_id for record in self.by_family.get(family, ()) if record.player_id != self_id and record.present_at(sequence) and record.competitive_at(sequence))

    def competitive_at(self, family: str, player_id: str, sequence: int) -> bool:
        record = self.records.get((family, player_id))
        return bool(record and record.present_at(sequence) and record.competitive_at(sequence))

    def label(self, family: str, player_id: str) -> str | None:
        record = self.records.get((family, player_id))
        return record.label if record else None


@dataclass(frozen=True)
class EventSeries:
    """One corrected player-family pulse history with precomputed cumulative state."""

    times: tuple[float, ...]
    additions: tuple[int, ...]
    cumulative: tuple[int, ...]
    session_start_index: tuple[int, ...]
    session_number: tuple[int, ...]

    @classmethod
    def build(cls, rows: Sequence[tuple[float, int]]) -> EventSeries:
        ordered = sorted(rows)
        times = tuple(value[0] for value in ordered)
        additions = tuple(value[1] for value in ordered)
        cumulative_values = [0]
        starts: list[int] = []
        sessions: list[int] = []
        session_start = 0
        session_number = 0
        for index, (stamp, addition) in enumerate(ordered):
            if index and stamp - ordered[index - 1][0] > 300.0:
                session_start = index
                session_number += 1
            starts.append(session_start)
            sessions.append(session_number)
            cumulative_values.append(cumulative_values[-1] + addition)
        return cls(times, additions, tuple(cumulative_values), tuple(starts), tuple(sessions))

    def index_at(self, stamp: float) -> int:
        return bisect.bisect_right(self.times, stamp)

    def additions_between(self, start: float, end: float) -> int:
        left = bisect.bisect_right(self.times, start)
        right = bisect.bisect_right(self.times, end)
        return self.cumulative[right] - self.cumulative[left]

    def pulses_between(self, start: float, end: float) -> int:
        return bisect.bisect_right(self.times, end) - bisect.bisect_right(self.times, start)

    def future_event(self, stamp: float, horizon: int) -> bool:
        return bisect.bisect_right(self.times, stamp + horizon) > bisect.bisect_right(self.times, stamp)


EMPTY_SERIES = EventSeries((), (), (0,), (), ())


@dataclass(frozen=True)
class CausalState:
    """Features computable from public events observed by one query time."""

    features: Mapping[str, str]
    prior_additions: int
    exposure_s: float
    regime: str


def causal_state(*, player_id: str, stamp: float, first_seen_at: float, series: EventSeries, other_series: Sequence[EventSeries], family_traffic_60: int, global_traffic_60: int) -> CausalState:
    index = series.index_at(stamp)
    prior_additions = series.cumulative[index]
    exposure = max(0.0, stamp - first_seen_at)
    if index:
        last = series.times[index - 1]
        idle = max(0.0, stamp - last)
        last_gap = series.times[index - 1] - series.times[index - 2] if index >= 2 else None
        recent_gaps = [series.times[value] - series.times[value - 1] for value in range(max(1, index - 5), index)]
        median_gap = statistics.median(recent_gaps) if recent_gaps else None
        session_start = series.session_start_index[index - 1]
        session_age = max(0.0, stamp - series.times[session_start])
        session_additions = series.cumulative[index] - series.cumulative[session_start]
        prior_sessions = series.session_number[index - 1] + 1
        if index >= 12 and idle >= 300.0:
            latest = series.times[max(0, index - 20) : index]
            phase_counts = Counter(int(datetime.fromtimestamp(value, tz=timezone.utc).minute // 10) for value in latest)
            phase_concentration = max(phase_counts.values(), default=0) / len(latest)
            distinct_hours = len({int(value // 3600) for value in latest})
        else:
            phase_concentration = 0.0
            distinct_hours = 0
    else:
        idle = None
        last_gap = None
        median_gap = None
        session_age = None
        session_additions = 0
        prior_sessions = 0
        phase_concentration = 0.0
        distinct_hours = 0
    recent_60 = series.additions_between(stamp - 60.0, stamp)
    recent_300 = series.additions_between(stamp - 300.0, stamp)
    recent_900 = series.additions_between(stamp - 900.0, stamp)
    other_recent = sum(value.additions_between(stamp - 300.0, stamp) for value in other_series)
    if index == 0:
        regime = "unseen"
    elif prior_additions <= 5:
        regime = "canary-like"
    elif idle is not None and idle > 900.0:
        regime = "returning-after-idle"
    elif index >= 12 and distinct_hours >= 2 and phase_concentration >= 0.6 and idle is not None and idle >= 300.0:
        regime = "threshold-defense-like"
    elif session_additions >= 5 and median_gap is not None and median_gap <= 120.0:
        regime = "continuous"
    elif prior_sessions >= 2 and idle is not None and idle <= 300.0:
        regime = "periodic-burst"
    else:
        regime = "intermittent"
    lifetime_rate = prior_additions * 3600.0 / max(1.0, exposure)
    features = {
        "player": player_id,
        "idle": _bin(idle, (20, 40, 60, 120, 300, 900), missing="unseen"),
        "last_gap": _bin(last_gap, (20, 40, 60, 120, 300, 900), missing="unseen"),
        "lifetime_rate": _bin(lifetime_rate, (0.25, 1, 3, 10, 30, 90)),
        "recent_60": _bin(float(recent_60), (0, 1, 2, 4, 8, 16)),
        "recent_300": _bin(float(recent_300), (0, 1, 2, 4, 8, 16)),
        "recent_900": _bin(float(recent_900), (0, 1, 2, 4, 8, 16)),
        "session_age": _bin(session_age, (20, 40, 60, 120, 300, 900), missing="unseen"),
        "regime": regime,
        "utc_block": str(datetime.fromtimestamp(stamp, tz=timezone.utc).hour // 4),
        "family_traffic": _bin(float(family_traffic_60), (0, 5, 10, 20, 40, 80)),
        "global_traffic": _bin(float(global_traffic_60), (0, 5, 10, 20, 40, 80)),
        "other_family_recent": _bin(float(other_recent), (0, 1, 2, 4, 8, 16)),
    }
    return CausalState(features, prior_additions, exposure, regime)


class AdditiveHazardTrainer:
    """Accumulate categorical future-pulse counts without retaining training landmarks."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.total = 0
        self.positive = 0
        self.counts: dict[str, dict[str, list[int]]] = {name: defaultdict(lambda: [0, 0]) for name in self.feature_names}

    def update(self, features: Mapping[str, str], target: bool) -> None:
        self.total += 1
        self.positive += int(target)
        for name in self.feature_names:
            values = self.counts[name][features[name]]
            values[0] += 1
            values[1] += int(target)

    def build(self) -> AdditiveHazardModel:
        if not self.total:
            raise ValueError("hazard model has no training landmarks")
        prevalence = (self.positive + 1.0) / (self.total + 2.0)
        base = _logit(prevalence)
        effects: dict[str, dict[str, float]] = {}
        for name in self.feature_names:
            effects[name] = {}
            for category, (count, positive) in sorted(self.counts[name].items()):
                probability = (positive + CATEGORY_SMOOTHING * prevalence) / (count + CATEGORY_SMOOTHING)
                effects[name][category] = max(-CATEGORY_EFFECT_CLIP, min(CATEGORY_EFFECT_CLIP, _logit(probability) - base))
        return AdditiveHazardModel(self.feature_names, prevalence, effects, self.total, self.positive)


@dataclass(frozen=True)
class AdditiveHazardModel:
    """One additive empirical-Bayes categorical hazard score."""

    feature_names: tuple[str, ...]
    prevalence: float
    effects: Mapping[str, Mapping[str, float]]
    training_rows: int
    training_positives: int

    def raw_logit(self, features: Mapping[str, str]) -> float:
        score = _logit(self.prevalence)
        for name in self.feature_names:
            score += float(self.effects.get(name, {}).get(features[name], 0.0))
        return max(-20.0, min(20.0, score))

    def as_dict(self) -> dict[str, object]:
        return {"feature_names": list(self.feature_names), "prevalence": self.prevalence, "training_rows": self.training_rows, "training_positives": self.training_positives, "effects": {name: dict(sorted(values.items())) for name, values in self.effects.items()}}


@dataclass(frozen=True)
class PlattCalibrator:
    """A monotone scalar probability calibration fitted only on the calibration interval."""

    slope: float
    intercept: float
    rows: int

    def probability(self, raw_logit: float) -> float:
        return min(1.0 - 1e-9, max(1e-9, _sigmoid(self.intercept + self.slope * raw_logit)))


def fit_platt(scores: Sequence[float], targets: Sequence[int]) -> PlattCalibrator:
    if not scores or len(scores) != len(targets):
        raise ValueError("Platt calibration requires paired scores and targets")

    def objective(candidate_slope: float, candidate_intercept: float) -> float:
        total = 0.5 * PLATT_RIDGE * (candidate_slope * candidate_slope + candidate_intercept * candidate_intercept)
        for score, target in zip(scores, targets, strict=True):
            value = candidate_intercept + candidate_slope * score
            total += max(value, 0.0) + math.log1p(math.exp(-abs(value))) - target * value
        return total

    slope = 1.0
    intercept = 0.0
    for _ in range(PLATT_ITERATIONS):
        gradient_slope = PLATT_RIDGE * slope
        gradient_intercept = PLATT_RIDGE * intercept
        hessian_slope = PLATT_RIDGE
        hessian_intercept = PLATT_RIDGE
        hessian_cross = 0.0
        for score, target in zip(scores, targets, strict=True):
            probability = _sigmoid(intercept + slope * score)
            residual = probability - target
            weight = probability * (1.0 - probability)
            gradient_slope += residual * score
            gradient_intercept += residual
            hessian_slope += weight * score * score
            hessian_intercept += weight
            hessian_cross += weight * score
        determinant = hessian_slope * hessian_intercept - hessian_cross * hessian_cross
        if determinant <= 1e-12:
            break
        delta_slope = (gradient_slope * hessian_intercept - gradient_intercept * hessian_cross) / determinant
        delta_intercept = (gradient_intercept * hessian_slope - gradient_slope * hessian_cross) / determinant
        current_objective = objective(slope, intercept)
        accepted: tuple[float, float] | None = None
        scale = 1.0
        for _ in range(60):
            candidate_slope = max(0.0, slope - scale * delta_slope)
            candidate_intercept = intercept - scale * delta_intercept
            change_slope = candidate_slope - slope
            change_intercept = candidate_intercept - intercept
            directional_derivative = gradient_slope * change_slope + gradient_intercept * change_intercept
            if directional_derivative < 0.0 and objective(candidate_slope, candidate_intercept) <= current_objective + 1e-4 * directional_derivative:
                accepted = candidate_slope, candidate_intercept
                break
            scale *= 0.5
        if accepted is None:
            break
        new_slope, new_intercept = accepted
        if abs(new_slope - slope) + abs(new_intercept - intercept) < 1e-9:
            slope, intercept = accepted
            break
        slope, intercept = new_slope, new_intercept
    return PlattCalibrator(slope, intercept, len(scores))


class BinaryAccumulator:
    """Compact exact binary-probability accumulator for proper scores and calibration."""

    def __init__(self) -> None:
        self.probabilities = array("d")
        self.targets = bytearray()

    def add(self, probability: float, target: bool) -> None:
        self.probabilities.append(min(1.0 - 1e-15, max(1e-15, probability)))
        self.targets.append(int(target))

    def metrics(self) -> dict[str, object]:
        count = len(self.targets)
        positives = sum(self.targets)
        if not count:
            return {"rows": 0}
        nll = 0.0
        brier = 0.0
        bins = [[0, 0, 0.0] for _ in range(10)]
        for probability, target in zip(self.probabilities, self.targets, strict=True):
            nll -= target * math.log(probability) + (1 - target) * math.log1p(-probability)
            brier += (probability - target) ** 2
            index = min(9, int(probability * 10))
            bins[index][0] += 1
            bins[index][1] += target
            bins[index][2] += probability
        ece = sum(values[0] / count * abs(values[1] / values[0] - values[2] / values[0]) for values in bins if values[0])
        return {"rows": count, "positives": positives, "prevalence": positives / count, "negative_log_likelihood": nll / count, "brier": brier / count, "expected_calibration_error": ece, "roc_auc": _auc(self.probabilities, self.targets), "calibration_bins": [{"lower": index / 10, "upper": (index + 1) / 10, "rows": values[0], "observed": values[1] / values[0] if values[0] else None, "predicted": values[2] / values[0] if values[0] else None} for index, values in enumerate(bins)]}


def _auc(probabilities: Sequence[float], targets: Sequence[int]) -> float | None:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(probabilities, targets, strict=True), key=lambda value: value[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(target for _, target in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def static_rate_logit(state: CausalState, *, horizon: int, population_probability: float) -> float:
    prior_exposure = 3600.0
    population_rate = -math.log1p(-min(1.0 - 1e-9, max(1e-9, population_probability))) / horizon
    rate = (state.prior_additions + population_rate * prior_exposure) / (state.exposure_s + prior_exposure)
    return _logit(1.0 - math.exp(-max(0.0, rate) * horizon))


@dataclass(frozen=True)
class ForecastBundle:
    """Frozen family-and-horizon models and calibration maps."""

    renewal: AdditiveHazardModel
    full: AdditiveHazardModel
    calibrators: Mapping[str, PlattCalibrator]

    def probability(self, arm: str, state: CausalState, *, horizon: int) -> float:
        if arm == "population":
            return self.renewal.prevalence
        if arm == "static-rate":
            score = static_rate_logit(state, horizon=horizon, population_probability=self.renewal.prevalence)
        elif arm == "renewal":
            score = self.renewal.raw_logit(state.features)
        elif arm == "full":
            score = self.full.raw_logit(state.features)
        else:
            raise ValueError(f"unknown presence arm: {arm}")
        return self.calibrators[arm].probability(score)


def _identity_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"games": 0}
    return {
        "games": len(rows),
        "candidate_coverage": sum(bool(row["covered"]) for row in rows) / len(rows),
        "top_one_accuracy": sum(int(row["true_rank"]) == 1 for row in rows) / len(rows),
        "top_5_accuracy": sum(int(row["true_rank"]) <= 5 for row in rows) / len(rows),
        "mean_reciprocal_rank": sum(1.0 / int(row["true_rank"]) for row in rows) / len(rows),
        "negative_log_likelihood": sum(float(row["nll"]) for row in rows) / len(rows),
        "multiclass_brier": sum(float(row["brier"]) for row in rows) / len(rows),
        "candidate_count_median": statistics.median(int(row["candidate_count"]) for row in rows),
    }


def _rank_record(*, game_id: str, family: str, arm: str, horizon: int, true_id: str, probabilities: Mapping[str, float], candidate_count: int) -> dict[str, object]:
    target = true_id if true_id in probabilities else UNKNOWN_ID
    ranking = sorted(probabilities, key=lambda label: (-probabilities[label], label))
    probability = max(1e-15, float(probabilities[target]))
    return {"game_id": game_id, "family": family, "arm": arm, "horizon_s": horizon, "covered": target != UNKNOWN_ID, "true_rank": ranking.index(target) + 1, "true_probability": probability, "nll": -math.log(probability), "brier": sum((value - float(label == target)) ** 2 for label, value in probabilities.items()), "candidate_count": candidate_count}


def _normalized_identity(scores: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(1e-12, value) for value in scores.values())
    if not scores:
        return {UNKNOWN_ID: 1.0}
    probabilities = {player_id: (1.0 - UNKNOWN_IDENTITY_MASS) * max(1e-12, value) / total for player_id, value in scores.items()}
    probabilities[UNKNOWN_ID] = UNKNOWN_IDENTITY_MASS
    return probabilities


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum((value - left_mean) ** 2 for value in left) * sum((value - right_mean) ** 2 for value in right))
    return numerator / denominator if denominator > 0.0 else None


def _best_lag(left: Sequence[float], right: Sequence[float], maximum_lag: int = 5) -> tuple[int, float | None]:
    candidates: list[tuple[float, int, float]] = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            correlation = _pearson(left[-lag:], right[:lag])
        elif lag > 0:
            correlation = _pearson(left[:-lag], right[lag:])
        else:
            correlation = _pearson(left, right)
        if correlation is not None:
            candidates.append((abs(correlation), -abs(lag), correlation if lag >= 0 else -correlation))
    if not candidates:
        return 0, None
    selected = max(candidates)
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            correlation = _pearson(left[-lag:], right[:lag])
        elif lag > 0:
            correlation = _pearson(left[:-lag], right[lag:])
        else:
            correlation = _pearson(left, right)
        if correlation is not None and abs(correlation) == selected[0] and -abs(lag) == selected[1]:
            return lag, correlation
    return 0, None


def _dilated(values: Sequence[int], radius: int = 5) -> set[int]:
    result: set[int] = set()
    for index, value in enumerate(values):
        if value:
            result.update(range(max(0, index - radius), min(len(values), index + radius + 1)))
    return result


def _session_starts(values: Sequence[int], quiet_bins: int = 5) -> set[int]:
    return {index for index, value in enumerate(values) if value and not any(values[max(0, index - quiet_bins) : index])}


def _linked_policy_features(*, event_rows: Sequence[Mapping[str, object]], registry: FrozenPresenceRegistry, self_ids: Mapping[str, str], first_time: float, last_time: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    bin_count = int(math.ceil((last_time - first_time) / LANDMARK_STEP_S))
    vectors: dict[tuple[str, str], bytearray] = defaultdict(lambda: bytearray(bin_count))
    additions: Counter[tuple[str, str]] = Counter()
    platform = [0] * bin_count
    for row in event_rows:
        family = str(row["family"])
        player_id = str(row["player_id"])
        if player_id == self_ids.get(family) or not registry.competitive_at(family, player_id, int(row["frontier_sequence"])):
            continue
        index = min(bin_count - 1, max(0, int((float(row["observed_by"]) - first_time) // LANDMARK_STEP_S)))
        vectors[(family, player_id)][index] = 1
        additions[(family, player_id)] += int(row["games_delta"])
        platform[index] += 1
    ordered_traffic = sorted(platform)
    thresholds = [ordered_traffic[int(fraction * (len(ordered_traffic) - 1))] for fraction in (0.25, 0.5, 0.75)]
    strata = [(datetime.fromtimestamp(first_time + index * LANDMARK_STEP_S, tz=timezone.utc).hour // 4, bisect.bisect_right(thresholds, traffic)) for index, traffic in enumerate(platform)]
    stratum_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        stratum_indices[stratum].append(index)

    def residuals(values: Sequence[int]) -> list[float]:
        probabilities: dict[tuple[int, int], float] = {}
        for stratum, indices in stratum_indices.items():
            probabilities[stratum] = (sum(values[index] for index in indices) + 1.0) / (len(indices) + 2.0)
        return [value - probabilities[strata[index]] for index, value in enumerate(values)]

    def overlap_core(left: Sequence[int], right: Sequence[int]) -> dict[str, object]:
        observed = sum(bool(a and b) for a, b in zip(left, right, strict=True))
        expected = 0.0
        for indices in stratum_indices.values():
            left_probability = (sum(left[index] for index in indices) + 1.0) / (len(indices) + 2.0)
            right_probability = (sum(right[index] for index in indices) + 1.0) / (len(indices) + 2.0)
            expected += len(indices) * left_probability * right_probability
        union = sum(bool(a or b) for a, b in zip(left, right, strict=True))
        return {"left_active_bins": sum(left), "right_active_bins": sum(right), "observed_overlap_bins": observed, "expected_overlap_bins": expected, "standardized_excess_overlap": (observed - expected) / math.sqrt(max(1.0, expected)), "active_bin_jaccard": observed / union if union else None}

    def overlap_record(left: Sequence[int], right: Sequence[int], *, left_residual: Sequence[float] | None = None, right_residual: Sequence[float] | None = None) -> dict[str, object]:
        values = overlap_core(left, right)
        lag, correlation = _best_lag(left_residual if left_residual is not None else residuals(left), right_residual if right_residual is not None else residuals(right))
        left_dilated = _dilated(left)
        right_dilated = _dilated(right)
        session_union = len(left_dilated | right_dilated)
        left_starts = _session_starts(left)
        right_starts = _session_starts(right)
        shared_starts = sum(any(abs(value - other) <= 2 for other in right_starts) for value in left_starts)
        left_to_right = sum(any(right[min(len(right), value + 1) : min(len(right), value + 6)]) for value in left_starts)
        right_to_left = sum(any(left[min(len(left), value + 1) : min(len(left), value + 6)]) for value in right_starts)
        return {**values, "dilated_session_jaccard": len(left_dilated & right_dilated) / session_union if session_union else None, "left_session_starts": len(left_starts), "right_session_starts": len(right_starts), "shared_session_starts_120s": shared_starts, "left_to_right_starts_300s": left_to_right, "right_to_left_starts_300s": right_to_left, "best_residual_lag_s": lag * LANDMARK_STEP_S, "best_residual_lag_correlation": correlation}

    output: list[dict[str, object]] = []
    player_ids = sorted({player_id for _, player_id in vectors})
    for player_id in player_ids:
        available = [family for family in GLEE_FAMILIES if (family, player_id) in vectors]
        for left_family, right_family in combinations(available, 2):
            values = overlap_record(vectors[(left_family, player_id)], vectors[(right_family, player_id)])
            output.append({"contract": PRESENCE_FORECAST_CONTRACT, "kind": "same-public-id-cross-family", "left_public_player_id": player_id, "right_public_player_id": player_id, "left_family": left_family, "right_family": right_family, "labels": {left_family: registry.label(left_family, player_id), right_family: registry.label(right_family, player_id)}, "family_additions": {left_family: additions[(left_family, player_id)], right_family: additions[(right_family, player_id)]}, **values})
    global_vectors: dict[str, bytearray] = {}
    for player_id in player_ids:
        combined = bytearray(bin_count)
        for family in GLEE_FAMILIES:
            values = vectors.get((family, player_id))
            if values is not None:
                for index, value in enumerate(values):
                    combined[index] = int(bool(combined[index] or value))
        global_vectors[player_id] = combined
    candidates: list[tuple[float, str, str, dict[str, object]]] = []
    for left_id, right_id in combinations(player_ids, 2):
        left = global_vectors[left_id]
        right = global_vectors[right_id]
        if sum(left) < 10 or sum(right) < 10:
            continue
        values = overlap_core(left, right)
        candidates.append((float(values["standardized_excess_overlap"]), left_id, right_id, values))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    global_residuals = {player_id: residuals(values) for player_id, values in global_vectors.items()}
    for _, left_id, right_id, _values in candidates[:200]:
        values = overlap_record(global_vectors[left_id], global_vectors[right_id], left_residual=global_residuals[left_id], right_residual=global_residuals[right_id])
        output.append({"contract": PRESENCE_FORECAST_CONTRACT, "kind": "different-public-id-linkage-hypothesis", "left_public_player_id": left_id, "right_public_player_id": right_id, "labels": {family: [registry.label(family, left_id), registry.label(family, right_id)] for family in GLEE_FAMILIES}, **values})
    return output, {"bin_seconds": LANDMARK_STEP_S, "bins": bin_count, "competitive_player_ids": len(player_ids), "same_id_cross_family_records": sum(row["kind"] == "same-public-id-cross-family" for row in output), "different_id_records_retained": sum(row["kind"] == "different-public-id-linkage-hypothesis" for row in output), "different_id_minimum_active_bins": 10, "traffic_quartile_boundaries": thresholds, "interpretation": "retrospective scheduler or policy-similarity hypotheses only; no ownership, coordination, collusion, or identity claim"}


class GleePresenceForecastAnalysis:
    """Run frozen Stage 3 causal forecasting and descriptive linked-policy analysis."""

    def __init__(self, *, event_cache: Path, identity_dir: Path, behavior_dir: Path, channel_dir: Path, reporter_database: Path, activity_summary: Path, output_dir: Path) -> None:
        self.event_cache = event_cache.resolve()
        self.identity_dir = identity_dir.resolve()
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.reporter_database = reporter_database.resolve()
        self.activity_summary = activity_summary.resolve()
        self.output_dir = output_dir.resolve()

    def _event_rows(self, expected_sha256: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        connection = sqlite3.connect(f"file:{self.event_cache}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            metadata = {str(row["key"]): json.loads(str(row["value"])) for row in connection.execute("SELECT key, value FROM metadata")}
            if metadata.get("contract") != "glee-joint-effective-event-cache-v1" or metadata.get("effective_events_sha256") != expected_sha256:
                raise RuntimeError("effective-event cache does not match the frozen Stage 0 receipt")
            rows = [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY frontier_sequence, family, player_id, source_change_sequence")]
        finally:
            connection.close()
        return rows, metadata

    def _frontiers(self, frontier_sequence: int, expected_completed_at: str) -> tuple[list[int], list[float], str]:
        connection = _read_only_database(self.reporter_database)
        try:
            rows = connection.execute("SELECT sequence, completed_at FROM frontiers WHERE sequence <= ? ORDER BY sequence", (frontier_sequence,)).fetchall()
        finally:
            connection.close()
        sequences = [int(row["sequence"]) for row in rows]
        completed = [_timestamp(row["completed_at"]) for row in rows]
        if not sequences or sequences[-1] != frontier_sequence or completed[-1] is None or abs(completed[-1] - float(_timestamp(expected_completed_at))) > 1e-6:
            raise RuntimeError("reporter frontier timestamps do not reproduce the frozen frontier")
        digest = hashlib.sha256()
        for sequence, stamp in zip(sequences, completed, strict=True):
            digest.update(f"{sequence}\0{stamp:.6f}\n".encode("ascii"))
        return sequences, [float(value) for value in completed], digest.hexdigest()

    @staticmethod
    def _landmarks(start: float, end: float) -> list[float]:
        first = math.ceil(start / LANDMARK_STEP_S) * LANDMARK_STEP_S
        return [first + index * LANDMARK_STEP_S for index in range(max(0, int(math.floor((end - first) / LANDMARK_STEP_S)) + 1))]

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        lines = ["# GLEE causal presence forecasting v1", "", "**Status:** Completed offline Stage 3 baseline; no result is connected to matchmaking, prompts, actions, identity routing, SIC, or dossiers.", "", "## Untouched pulse-forecast test", "", "| Family | Horizon | Arm | NLL | Brier | ECE | AUC |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
        for family in GLEE_FAMILIES:
            for horizon in HORIZONS:
                for arm in ARMS:
                    metrics = summary["presence_test"][family][str(horizon)][arm]
                    auc = metrics.get("roc_auc")
                    lines.append(f"| {family.title()} | {horizon}s | {arm} | {metrics['negative_log_likelihood']:.4f} | {metrics['brier']:.4f} | {metrics['expected_calibration_error']:.4f} | {auc:.4f} |" if auc is not None else f"| {family.title()} | {horizon}s | {arm} | {metrics['negative_log_likelihood']:.4f} | {metrics['brier']:.4f} | {metrics['expected_calibration_error']:.4f} | — |")
        lines.extend(["", "## Masked identity ranking", "", "| Horizon | Arm | Top-one | Top-5 | MRR | NLL |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
        for horizon in HORIZONS:
            for arm in ("uniform", "static-rate", "renewal", "full"):
                metrics = summary["identity_test"]["pooled"][str(horizon)][arm]
                lines.append(f"| {horizon}s | {arm} | {metrics['top_one_accuracy']:.1%} | {metrics['top_5_accuracy']:.1%} | {metrics['mean_reciprocal_rank']:.3f} | {metrics['negative_log_likelihood']:.3f} |")
        lines.extend(["", "## Boundary", "", "The activity model sees only public evidence available by each query time. The exact-ID suffix contains no representative absent-from-board opponents, so the fixed 5% `unknown` reserve is not calibrated. Linked-policy outputs are descriptive schedule-similarity hypotheses and do not establish common control. A later Stage 6 rerun must test causal fusion and decision value before any live use.", ""])
        return "\n".join(lines)

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"presence-forecast output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        activity = json.loads(self.activity_summary.read_text(encoding="utf-8"))
        expected_events_sha256 = str(activity["effective_event_reconstruction"]["effective_events_sha256"])
        frontier_sequence = int(activity["source_frontier"]["frontier_sequence"])
        self_ids = {str(family): str(player_id) for family, player_id in activity["source_frontier"]["self_player_ids"].items()}
        registry = FrozenPresenceRegistry(self.identity_dir)
        if registry.frontier_sequence != frontier_sequence:
            raise RuntimeError("identity registry and activity frontier differ")
        event_rows, event_metadata = self._event_rows(expected_events_sha256)
        frontier_sequences, frontier_times, frontier_time_sha256 = self._frontiers(frontier_sequence, str(activity["source_frontier"]["frontier_completed_at"]))
        first_time = frontier_times[0]
        last_time = frontier_times[-1]
        event_series_rows: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        family_event_times: dict[str, list[float]] = defaultdict(list)
        global_event_times: list[float] = []
        for row in event_rows:
            stamp = float(row["observed_by"])
            event_series_rows[(str(row["family"]), str(row["player_id"]))].append((stamp, int(row["games_delta"])))
            family_event_times[str(row["family"])].append(stamp)
            global_event_times.append(stamp)
        series = {key: EventSeries.build(values) for key, values in event_series_rows.items()}
        for values in family_event_times.values():
            values.sort()
        global_event_times.sort()
        first_seen_at = {(family, record.player_id): frontier_times[max(0, bisect.bisect_left(frontier_sequences, record.first_seen_sequence))] for family, records in registry.by_family.items() for record in records}
        channel_manifest = json.loads((self.channel_dir / "manifest.json").read_text(encoding="utf-8"))
        if _file_digest(self.channel_dir / "summary.json") != channel_manifest["artifacts"]["summary.json"]["sha256"]:
            raise RuntimeError("behavior-channel summary hash mismatch")
        channel_summary = json.loads((self.channel_dir / "summary.json").read_text(encoding="utf-8"))
        boundaries = channel_summary["split_boundaries"]

        def sequence_at(stamp: float) -> int | None:
            index = bisect.bisect_right(frontier_times, stamp) - 1
            return frontier_sequences[index] if index >= 0 else None

        def traffic(stamp: float, family: str) -> tuple[int, int]:
            family_values = family_event_times[family]
            family_count = bisect.bisect_right(family_values, stamp) - bisect.bisect_right(family_values, stamp - 60.0)
            global_count = bisect.bisect_right(global_event_times, stamp) - bisect.bisect_right(global_event_times, stamp - 60.0)
            return family_count, global_count

        def state_for(family: str, player_id: str, stamp: float, family_traffic: int, global_traffic: int) -> CausalState:
            others = [series.get((other, player_id), EMPTY_SERIES) for other in GLEE_FAMILIES if other != family]
            return causal_state(player_id=player_id, stamp=stamp, first_seen_at=first_seen_at.get((family, player_id), first_time), series=series.get((family, player_id), EMPTY_SERIES), other_series=others, family_traffic_60=family_traffic, global_traffic_60=global_traffic)

        models: dict[tuple[str, int], ForecastBundle] = {}
        split_inventory: dict[str, dict[str, int]] = {}
        regime_inventory: dict[str, Counter[str]] = {family: Counter() for family in GLEE_FAMILIES}
        test_accumulators: dict[tuple[str, int, str], BinaryAccumulator] = {(family, horizon, arm): BinaryAccumulator() for family in GLEE_FAMILIES for horizon in HORIZONS for arm in ARMS}
        model_artifact: dict[str, object] = {}
        for family in GLEE_FAMILIES:
            calibration_start = float(_timestamp(boundaries[family]["calibration_start"]))
            test_start = float(_timestamp(boundaries[family]["test_start"]))
            train_landmarks = self._landmarks(first_time + WARMUP_S, calibration_start - WARMUP_S)
            calibration_landmarks = self._landmarks(calibration_start, test_start - WARMUP_S)
            test_landmarks = self._landmarks(test_start, last_time - WARMUP_S)
            split_inventory[family] = {"train_landmarks": len(train_landmarks), "calibration_landmarks": len(calibration_landmarks), "test_landmarks": len(test_landmarks)}
            renewal_trainers = {horizon: AdditiveHazardTrainer(RENEWAL_FEATURES) for horizon in HORIZONS}
            full_trainers = {horizon: AdditiveHazardTrainer(FULL_FEATURES) for horizon in HORIZONS}
            for stamp in train_landmarks:
                sequence = sequence_at(stamp)
                if sequence is None:
                    continue
                family_traffic, global_traffic = traffic(stamp, family)
                for player_id in registry.candidates(family, sequence, self_id=self_ids.get(family)):
                    state = state_for(family, player_id, stamp, family_traffic, global_traffic)
                    player_series = series.get((family, player_id), EMPTY_SERIES)
                    for horizon in HORIZONS:
                        target = player_series.future_event(stamp, horizon)
                        renewal_trainers[horizon].update(state.features, target)
                        full_trainers[horizon].update(state.features, target)
            family_models: dict[int, tuple[AdditiveHazardModel, AdditiveHazardModel]] = {horizon: (renewal_trainers[horizon].build(), full_trainers[horizon].build()) for horizon in HORIZONS}
            calibration_scores: dict[tuple[int, str], list[float]] = {(horizon, arm): [] for horizon in HORIZONS for arm in ("static-rate", "renewal", "full")}
            calibration_targets: dict[int, list[int]] = {horizon: [] for horizon in HORIZONS}
            for stamp in calibration_landmarks:
                sequence = sequence_at(stamp)
                if sequence is None:
                    continue
                family_traffic, global_traffic = traffic(stamp, family)
                for player_id in registry.candidates(family, sequence, self_id=self_ids.get(family)):
                    state = state_for(family, player_id, stamp, family_traffic, global_traffic)
                    player_series = series.get((family, player_id), EMPTY_SERIES)
                    for horizon in HORIZONS:
                        renewal_model, full_model = family_models[horizon]
                        calibration_targets[horizon].append(int(player_series.future_event(stamp, horizon)))
                        calibration_scores[(horizon, "static-rate")].append(static_rate_logit(state, horizon=horizon, population_probability=renewal_model.prevalence))
                        calibration_scores[(horizon, "renewal")].append(renewal_model.raw_logit(state.features))
                        calibration_scores[(horizon, "full")].append(full_model.raw_logit(state.features))
            model_artifact[family] = {}
            for horizon in HORIZONS:
                renewal_model, full_model = family_models[horizon]
                calibrators = {arm: fit_platt(calibration_scores[(horizon, arm)], calibration_targets[horizon]) for arm in ("static-rate", "renewal", "full")}
                models[(family, horizon)] = ForecastBundle(renewal_model, full_model, calibrators)
                model_artifact[family][str(horizon)] = {"renewal": renewal_model.as_dict(), "full": full_model.as_dict(), "calibrators": {arm: {"slope": value.slope, "intercept": value.intercept, "rows": value.rows} for arm, value in calibrators.items()}}
            for stamp in test_landmarks:
                sequence = sequence_at(stamp)
                if sequence is None:
                    continue
                family_traffic, global_traffic = traffic(stamp, family)
                for player_id in registry.candidates(family, sequence, self_id=self_ids.get(family)):
                    state = state_for(family, player_id, stamp, family_traffic, global_traffic)
                    regime_inventory[family][state.regime] += 1
                    player_series = series.get((family, player_id), EMPTY_SERIES)
                    for horizon in HORIZONS:
                        target = player_series.future_event(stamp, horizon)
                        bundle = models[(family, horizon)]
                        for arm in ARMS:
                            test_accumulators[(family, horizon, arm)].add(bundle.probability(arm, state, horizon=horizon), target)
        presence_test = {family: {str(horizon): {arm: test_accumulators[(family, horizon, arm)].metrics() for arm in ARMS} for horizon in HORIZONS} for family in GLEE_FAMILIES}

        behavior_manifest = json.loads((self.behavior_dir / "manifest.json").read_text(encoding="utf-8"))
        behavior_path = self.behavior_dir / "behavior-games.jsonl"
        if _file_digest(behavior_path) != behavior_manifest["artifacts"]["behavior-games.jsonl"]["sha256"]:
            raise RuntimeError("behavior-game corpus hash mismatch")
        behavior_records = _load_jsonl(behavior_path)
        assignments, behavior_boundaries = chronological_game_splits(behavior_records, train_fraction=float(channel_summary["design"]["train_fraction"]), calibration_fraction=float(channel_summary["design"]["calibration_fraction"]))
        envelopes = {str(row["game_id"]): row for row in _load_jsonl(self.identity_dir / "game-identity-envelopes.jsonl")}
        identity_rows: list[dict[str, object]] = []
        posterior_rows: list[dict[str, object]] = []
        test_games = [row for row in behavior_records if assignments.get(str(row["game_id"])) == "test"]
        expected_test_games = int(channel_summary["pooled"]["action"]["test"]["games"])
        if len(test_games) != expected_test_games:
            raise RuntimeError(f"Stage 3 identity-test size differs from the behavior-channel split: {len(test_games)} != {expected_test_games}")
        for game in sorted(test_games, key=lambda row: (str(row["started_at"]), str(row["game_id"]))):
            game_id = str(game["game_id"])
            family = str(game["family"])
            stamp = float(_timestamp(game["started_at"]))
            envelope = envelopes[game_id]
            sequence = int(envelope["assignment_frontier_sequence"])
            true_id = str(game["public_player_id"])
            candidates = registry.candidates(family, sequence, self_id=self_ids.get(family))
            family_traffic, global_traffic = traffic(stamp, family)
            states = {player_id: state_for(family, player_id, stamp, family_traffic, global_traffic) for player_id in candidates}
            candidate_output: dict[str, object] = {player_id: {"event_probability": {}, "identity_probability": {}} for player_id in candidates}
            for horizon in HORIZONS:
                bundle = models[(family, horizon)]
                uniform = _normalized_identity({player_id: 1.0 for player_id in candidates})
                identity_rows.append(_rank_record(game_id=game_id, family=family, arm="uniform", horizon=horizon, true_id=true_id, probabilities=uniform, candidate_count=len(candidates)))
                for arm in ("static-rate", "renewal", "full"):
                    event_probabilities = {player_id: bundle.probability(arm, state, horizon=horizon) for player_id, state in states.items()}
                    identity_probabilities = _normalized_identity(event_probabilities)
                    identity_rows.append(_rank_record(game_id=game_id, family=family, arm=arm, horizon=horizon, true_id=true_id, probabilities=identity_probabilities, candidate_count=len(candidates)))
                    for player_id in candidates:
                        candidate_output[player_id]["event_probability"].setdefault(arm, {})[str(horizon)] = event_probabilities[player_id]
                        candidate_output[player_id]["identity_probability"].setdefault(arm, {})[str(horizon)] = identity_probabilities[player_id]
            posterior_rows.append({"contract": PRESENCE_FORECAST_CONTRACT, "game_id": game_id, "family": family, "started_at": game["started_at"], "assignment_frontier_sequence": sequence, "true_public_player_id": true_id, "unknown_identity_mass": UNKNOWN_IDENTITY_MASS, "candidates": [{"public_player_id": player_id, **candidate_output[player_id]} for player_id in candidates]})
        identity_test = {
            "pooled": {str(horizon): {arm: _identity_metrics([row for row in identity_rows if row["horizon_s"] == horizon and row["arm"] == arm]) for arm in ("uniform", "static-rate", "renewal", "full")} for horizon in HORIZONS},
            "by_family": {family: {str(horizon): {arm: _identity_metrics([row for row in identity_rows if row["family"] == family and row["horizon_s"] == horizon and row["arm"] == arm]) for arm in ("uniform", "static-rate", "renewal", "full")} for horizon in HORIZONS} for family in GLEE_FAMILIES},
        }
        linked_rows, linked_summary = _linked_policy_features(event_rows=event_rows, registry=registry, self_ids=self_ids, first_time=first_time, last_time=last_time)
        summary = {
            "contract": PRESENCE_FORECAST_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only-stage-3-baseline",
            "sources": {"frontier_sequence": frontier_sequence, "effective_events_sha256": expected_events_sha256, "event_cache_metadata": event_metadata, "identity_manifest_sha256": registry.manifest_sha256, "behavior_manifest_sha256": _file_digest(self.behavior_dir / "manifest.json"), "channel_manifest_sha256": _file_digest(self.channel_dir / "manifest.json"), "activity_summary_sha256": _file_digest(self.activity_summary), "frontier_times_sha256": frontier_time_sha256},
            "design": {"horizons_s": list(HORIZONS), "landmark_step_s": LANDMARK_STEP_S, "warmup_and_boundary_purge_s": WARMUP_S, "category_smoothing": CATEGORY_SMOOTHING, "category_effect_clip": CATEGORY_EFFECT_CLIP, "platt_ridge": PLATT_RIDGE, "platt_iterations": PLATT_ITERATIONS, "unknown_identity_mass": UNKNOWN_IDENTITY_MASS, "test_selected_tuning": False, "behavior_split_boundaries": behavior_boundaries, "identity_test_games_expected_from_channel_split": expected_test_games, "landmark_inventory": split_inventory, "causal_feature_cutoff": "observed_by <= query time"},
            "presence_test": presence_test,
            "test_regimes": {family: dict(sorted(values.items())) for family, values in regime_inventory.items()},
            "identity_test": identity_test,
            "linked_policy": linked_summary,
            "promotion": {"online_shadow_authority": False, "identity_routing_authority": False, "sic_authority": False, "policy_clusters_implemented": False, "decision_value_evaluated": False, "blockers": ["unknown identity reserve lacks representative absent-from-board test support", "linked-policy records are retrospective similarity hypotheses", "causal presence prior has not been fused with behavior channels", "opponent-specific strategy decision value is not evaluated"]},
        }
        _write_json(self.output_dir / "summary.json", summary)
        _write_json(self.output_dir / "models.json", model_artifact)
        _write_jsonl(self.output_dir / "identity-test-predictions.jsonl", identity_rows)
        _write_jsonl(self.output_dir / "game-presence-posteriors.jsonl", posterior_rows)
        _write_jsonl(self.output_dir / "linked-policy-features.jsonl", linked_rows)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifacts = ("README.md", "summary.json", "models.json", "identity-test-predictions.jsonl", "game-presence-posteriors.jsonl", "linked-policy-features.jsonl")
        manifest = {"contract": PRESENCE_FORECAST_CONTRACT, "schema_version": 1, "sources": summary["sources"], "implementation_sha256": {"glee_presence_forecast.py": _file_digest(Path(__file__))}, "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts}}
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": PRESENCE_FORECAST_CONTRACT, "output_dir": str(self.output_dir), "identity_test_games": len(test_games), "presence_test": presence_test, "identity_test": identity_test["pooled"], "linked_policy": linked_summary, "promotion": summary["promotion"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
