"""Read-only exploratory analysis of GLEE public activity and local game receipts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator


ACTIVITY_EDA_CONTRACT = "glee-activity-eda-v1"
GLEE_FAMILIES = ("bargaining", "negotiation", "persuasion")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(value: str | None) -> float | None:
    parsed = _parse_time(value)
    return parsed.timestamp() if parsed is not None else None


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="microseconds")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _normalize_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized.casefold() if normalized else None


def _sign(value: float | None, tolerance: float = 1e-9) -> str:
    if value is None:
        return "missing"
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _magnitude_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    magnitude = abs(value)
    if magnitude < 0.5:
        return "lt-0.5"
    if magnitude < 2.0:
        return "0.5-to-2"
    if magnitude < 5.0:
        return "2-to-5"
    if magnitude < 10.0:
        return "5-to-10"
    return "ge-10"


def _pearson(count: int, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float | None:
    if count < 2:
        return None
    numerator = count * sum_xy - sum_x * sum_y
    denominator_x = count * sum_x2 - sum_x * sum_x
    denominator_y = count * sum_y2 - sum_y * sum_y
    denominator = math.sqrt(max(0.0, denominator_x) * max(0.0, denominator_y))
    return numerator / denominator if denominator > 0 else None


def _read_only_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


@dataclass(frozen=True)
class PublicPlayer:
    family: str
    player_id: str
    name: str
    normalized_name: str | None
    is_baseline: bool
    is_benchmark: bool
    is_owner_best: bool | None
    current_games: int | None
    current_rating: float | None


@dataclass(frozen=True)
class Pulse:
    change_sequence: int
    frontier_sequence: int
    family: str
    player_id: str
    observed_after: float | None
    observed_by: float
    games_delta: int
    rating_delta: float | None

    @property
    def width_s(self) -> float | None:
        if self.observed_after is None:
            return None
        return max(0.0, self.observed_by - self.observed_after)


@dataclass(frozen=True)
class LocalGame:
    game_id: str
    family: str
    started_at: float | None
    completed_at: float
    rating_delta: float | None
    identity_scope: str
    opponent_name: str | None
    archive_path: str | None
    archive_sha256: str | None
    archive_conflict: bool


@dataclass
class ActivityAccumulator:
    session_gap_s: float
    observation_quantum_s: float
    pulse_count: int = 0
    observed_games: int = 0
    multi_game_pulses: int = 0
    first_at: float | None = None
    last_at: float | None = None
    last_pulse_at: float | None = None
    gaps: list[float] = field(default_factory=list)
    hour_games: list[int] = field(default_factory=lambda: [0] * 24)
    rating_changes: int = 0
    rating_abs_sum: float = 0.0
    session_count: int = 0
    current_session_start: float | None = None
    current_session_last: float | None = None
    current_session_games: int = 0
    current_session_pulses: int = 0
    session_durations: list[float] = field(default_factory=list)
    session_games: list[int] = field(default_factory=list)

    def add(self, pulse: Pulse) -> None:
        observed_at = pulse.observed_by
        if self.last_pulse_at is not None:
            gap = max(0.0, observed_at - self.last_pulse_at)
            self.gaps.append(gap)
            if gap > self.session_gap_s:
                self._close_session()
        if self.current_session_start is None:
            self.current_session_start = observed_at
            self.current_session_last = observed_at
            self.current_session_games = 0
            self.current_session_pulses = 0
        self.current_session_last = observed_at
        self.current_session_games += pulse.games_delta
        self.current_session_pulses += 1
        self.pulse_count += 1
        self.observed_games += pulse.games_delta
        self.multi_game_pulses += int(pulse.games_delta > 1)
        self.first_at = observed_at if self.first_at is None else min(self.first_at, observed_at)
        self.last_at = observed_at if self.last_at is None else max(self.last_at, observed_at)
        self.last_pulse_at = observed_at
        hour = datetime.fromtimestamp(observed_at, tz=timezone.utc).hour
        self.hour_games[hour] += pulse.games_delta
        if pulse.rating_delta is not None:
            self.rating_changes += 1
            self.rating_abs_sum += abs(pulse.rating_delta)

    def _close_session(self) -> None:
        if self.current_session_start is None or self.current_session_last is None:
            return
        duration = max(self.observation_quantum_s, self.current_session_last - self.current_session_start + self.observation_quantum_s)
        self.session_count += 1
        self.session_durations.append(duration)
        self.session_games.append(self.current_session_games)
        self.current_session_start = None
        self.current_session_last = None
        self.current_session_games = 0
        self.current_session_pulses = 0

    def finish(self) -> None:
        self._close_session()

    def as_dict(self, *, reference_at: float) -> dict[str, object]:
        mean_gap = statistics.fmean(self.gaps) if self.gaps else None
        std_gap = statistics.pstdev(self.gaps) if len(self.gaps) >= 2 else None
        burstiness = None
        if mean_gap is not None and std_gap is not None and mean_gap + std_gap > 0:
            burstiness = (std_gap - mean_gap) / (std_gap + mean_gap)
        total_hour_games = sum(self.hour_games)
        probabilities = [count / total_hour_games for count in self.hour_games if count > 0] if total_hour_games else []
        entropy = -sum(probability * math.log2(probability) for probability in probabilities)
        normalized_entropy = entropy / math.log2(24) if probabilities else 0.0
        active_s = sum(self.session_durations)
        dominant_hour = max(range(24), key=self.hour_games.__getitem__) if total_hour_games else None
        seconds_since_last = max(0.0, reference_at - self.last_at) if self.last_at is not None else None

        def forecast(horizon_s: float) -> float | None:
            if seconds_since_last is None:
                return None
            eligible = [gap for gap in self.gaps if gap > seconds_since_last]
            if not eligible:
                return None
            return sum(gap <= seconds_since_last + horizon_s for gap in eligible) / len(eligible)

        return {
            "pulse_count": self.pulse_count,
            "observed_games": self.observed_games,
            "multi_game_pulse_fraction": _rounded(self.multi_game_pulses / self.pulse_count if self.pulse_count else None),
            "first_observed_at": _iso(self.first_at),
            "last_observed_at": _iso(self.last_at),
            "observed_span_hours": _rounded((self.last_at - self.first_at) / 3600.0 if self.first_at is not None and self.last_at is not None else None),
            "inter_pulse_mean_s": _rounded(mean_gap),
            "inter_pulse_std_s": _rounded(std_gap),
            "inter_pulse_p10_s": _rounded(_quantile(self.gaps, 0.1)),
            "inter_pulse_p50_s": _rounded(_quantile(self.gaps, 0.5)),
            "inter_pulse_p90_s": _rounded(_quantile(self.gaps, 0.9)),
            "inter_pulse_max_s": _rounded(max(self.gaps) if self.gaps else None),
            "burstiness": _rounded(burstiness),
            "seconds_since_last_pulse": _rounded(seconds_since_last),
            "currently_inside_session_gap": bool(seconds_since_last is not None and seconds_since_last <= self.session_gap_s),
            "empirical_next_pulse_probability_60s": _rounded(forecast(60.0)),
            "empirical_next_pulse_probability_300s": _rounded(forecast(300.0)),
            "empirical_next_pulse_probability_900s": _rounded(forecast(900.0)),
            "session_gap_s": self.session_gap_s,
            "session_count": self.session_count,
            "session_duration_p50_s": _rounded(_quantile(self.session_durations, 0.5)),
            "session_duration_p90_s": _rounded(_quantile(self.session_durations, 0.9)),
            "session_games_p50": _rounded(_quantile([float(value) for value in self.session_games], 0.5)),
            "session_games_p90": _rounded(_quantile([float(value) for value in self.session_games], 0.9)),
            "games_per_active_hour": _rounded(self.observed_games / (active_s / 3600.0) if active_s > 0 else None),
            "utc_hour_entropy_normalized": _rounded(normalized_entropy),
            "dominant_utc_hour": dominant_hour,
            "active_utc_hour_count": sum(count > 0 for count in self.hour_games),
            "utc_hour_games": self.hour_games,
            "rating_change_count": self.rating_changes,
            "mean_abs_public_rating_delta": _rounded(self.rating_abs_sum / self.rating_changes if self.rating_changes else None),
        }


@dataclass
class PairAccumulator:
    co_frontiers: int = 0
    co_game_capacity: int = 0
    single_single_frontiers: int = 0
    rating_pairs: int = 0
    opposite_sign: int = 0
    same_sign: int = 0
    zero_involved: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, first: Pulse, second: Pulse) -> None:
        self.co_frontiers += 1
        self.co_game_capacity += min(first.games_delta, second.games_delta)
        self.single_single_frontiers += int(first.games_delta == 1 and second.games_delta == 1)
        if first.rating_delta is None or second.rating_delta is None:
            return
        self.rating_pairs += 1
        first_sign = _sign(first.rating_delta)
        second_sign = _sign(second.rating_delta)
        if "zero" in {first_sign, second_sign}:
            self.zero_involved += 1
        elif first_sign == second_sign:
            self.same_sign += 1
        else:
            self.opposite_sign += 1
        self.sum_x += first.rating_delta
        self.sum_y += second.rating_delta
        self.sum_x2 += first.rating_delta * first.rating_delta
        self.sum_y2 += second.rating_delta * second.rating_delta
        self.sum_xy += first.rating_delta * second.rating_delta


class RatingEvidenceModel:
    """Causal exploratory likelihood ratios for public rating-pair features."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.positive: dict[str, Counter[str]] = defaultdict(Counter)
        self.negative: dict[str, Counter[str]] = defaultdict(Counter)
        self.positive_totals: Counter[str] = Counter()
        self.negative_totals: Counter[str] = Counter()
        self.vocabularies: dict[str, set[str]] = defaultdict(set)
        self.training_games = 0

    @staticmethod
    def features(self_delta: float | None, opponent_delta: float | None) -> dict[str, str]:
        self_sign = _sign(self_delta)
        opponent_sign = _sign(opponent_delta)
        return {
            "sign_pair": f"{self_sign}:{opponent_sign}",
            "magnitude_pair": f"{_magnitude_bin(self_delta)}:{_magnitude_bin(opponent_delta)}",
            "sum_magnitude": _magnitude_bin((self_delta or 0.0) + (opponent_delta or 0.0)) if self_delta is not None and opponent_delta is not None else "missing",
        }

    def _add(self, target: dict[str, Counter[str]], totals: Counter[str], features: dict[str, str], weight: float) -> None:
        for category, value in features.items():
            target[category][value] += weight
            totals[category] += weight
            self.vocabularies[category].add(value)

    def update(self, *, self_delta: float | None, positive_delta: float | None, negative_deltas: list[float | None]) -> None:
        self._add(self.positive, self.positive_totals, self.features(self_delta, positive_delta), 1.0)
        if negative_deltas:
            weight = 1.0 / len(negative_deltas)
            for delta in negative_deltas:
                self._add(self.negative, self.negative_totals, self.features(self_delta, delta), weight)
        self.training_games += 1

    def score(self, *, self_delta: float | None, opponent_delta: float | None) -> float:
        score = 0.0
        for category, value in self.features(self_delta, opponent_delta).items():
            vocabulary_size = max(2, len(self.vocabularies[category]) + int(value not in self.vocabularies[category]))
            positive_probability = (self.positive[category][value] + self.alpha) / (self.positive_totals[category] + self.alpha * vocabulary_size)
            negative_probability = (self.negative[category][value] + self.alpha) / (self.negative_totals[category] + self.alpha * vocabulary_size)
            score += max(-2.0, min(2.0, math.log(positive_probability / negative_probability)))
        return score


def _game_id_from_archive(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem
    for family in GLEE_FAMILIES:
        prefix = f"{family}-"
        if stem.startswith(prefix):
            return family, stem[len(prefix) :]
    return None, None


def _archive_identity(game: dict[str, Any]) -> tuple[str, str | None]:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    opponent_type = str(opponent.get("type") or "").casefold()
    name = opponent.get("name") if isinstance(opponent.get("name"), str) else None
    name = " ".join(name.split()).strip() if name else None
    if opponent_type == "hidden" or not name:
        return "hidden", None
    return "known", name


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _atomic_text(path, "".join(_canonical(row) + "\n" for row in rows))


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    temporary.replace(path)


class GleeActivityEDA:
    """Derive bounded activity traits and exploratory identity candidates from immutable frontiers."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        history_database: Path,
        game_archive_root: Path,
        output_dir: Path,
        self_name: str = "DeepRMM-01",
        reporter_frontier: int | None = None,
        session_gap_s: float = 300.0,
        alignment_slack_s: float = 30.0,
        top_candidates: int = 5,
    ) -> None:
        if session_gap_s <= 0 or alignment_slack_s < 0 or top_candidates < 1:
            raise ValueError("activity EDA thresholds are invalid")
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.output_dir = output_dir.resolve()
        self.self_name = self_name
        self.self_normalized_name = _normalize_name(self_name)
        self.requested_frontier = reporter_frontier
        self.session_gap_s = session_gap_s
        self.alignment_slack_s = alignment_slack_s
        self.top_candidates = top_candidates

    def _snapshot(self, connection: sqlite3.Connection) -> dict[str, object]:
        maximum = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM frontiers").fetchone()[0])
        frontier = maximum if self.requested_frontier is None else self.requested_frontier
        if frontier < 1 or frontier > maximum:
            raise ValueError(f"reporter frontier must be between 1 and {maximum}: {frontier}")
        row = connection.execute("SELECT sequence, frontier_id, started_at, completed_at FROM frontiers WHERE sequence = ?", (frontier,)).fetchone()
        if row is None:
            raise RuntimeError(f"missing reporter frontier {frontier}")
        first = connection.execute("SELECT MIN(started_at) FROM frontiers WHERE sequence <= ?", (frontier,)).fetchone()[0]
        change_sequence = int(connection.execute("SELECT COALESCE(MAX(change_sequence), 0) FROM changes WHERE frontier_sequence <= ?", (frontier,)).fetchone()[0])
        manifest_path = self.reporter_database.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        poll_interval = _finite(manifest.get("poll_interval_s")) or 10.0
        return {
            "frontier_sequence": frontier,
            "frontier_id": str(row["frontier_id"]),
            "first_started_at": str(first),
            "frontier_started_at": str(row["started_at"]),
            "frontier_completed_at": str(row["completed_at"]),
            "change_sequence": change_sequence,
            "poll_interval_s": poll_interval,
            "reporter_contract": manifest.get("contract"),
            "reporter_manifest_sha256": _file_digest(manifest_path) if manifest_path.is_file() else None,
        }

    def _players(self, connection: sqlite3.Connection, frontier: int) -> tuple[dict[tuple[str, str], PublicPlayer], dict[tuple[str, str], set[str]]]:
        current_rows = connection.execute(
            """
            SELECT rv.family, rv.player_id, rv.row_json
            FROM row_versions AS rv
            JOIN (
                SELECT family, player_id, MAX(sequence) AS sequence
                FROM row_versions
                WHERE sequence <= ? AND row_json IS NOT NULL
                GROUP BY family, player_id
            ) AS latest
            ON latest.family = rv.family AND latest.player_id = rv.player_id AND latest.sequence = rv.sequence
            """,
            (frontier,),
        )
        players: dict[tuple[str, str], PublicPlayer] = {}
        for row in current_rows:
            payload = json.loads(str(row["row_json"]))
            family = str(row["family"])
            player_id = str(row["player_id"])
            name = str(payload.get("player_name") or player_id)
            players[(family, player_id)] = PublicPlayer(
                family=family,
                player_id=player_id,
                name=name,
                normalized_name=_normalize_name(name),
                is_baseline=bool(payload.get("is_baseline")),
                is_benchmark=bool(payload.get("is_benchmark")),
                is_owner_best=bool(payload["is_owner_best"]) if payload.get("is_owner_best") is not None else None,
                current_games=int(payload["games_played"]) if isinstance(payload.get("games_played"), int) else None,
                current_rating=_finite(payload.get("rating")),
            )
        aliases: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in connection.execute(
            "SELECT DISTINCT family, player_id, json_extract(row_json, '$.player_name') AS player_name FROM row_versions WHERE sequence <= ? AND row_json IS NOT NULL",
            (frontier,),
        ):
            normalized = _normalize_name(row["player_name"])
            if normalized:
                aliases[(str(row["family"]), normalized)].add(str(row["player_id"]))
        return players, aliases

    def _self_ids(self, players: dict[tuple[str, str], PublicPlayer], aliases: dict[tuple[str, str], set[str]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for family in GLEE_FAMILIES:
            candidates = aliases.get((family, self.self_normalized_name or ""), set())
            if len(candidates) != 1:
                current = {player.player_id for (candidate_family, _), player in players.items() if candidate_family == family and player.normalized_name == self.self_normalized_name}
                candidates = current
            if len(candidates) != 1:
                raise RuntimeError(f"cannot uniquely resolve {self.self_name!r} on {family}: {sorted(candidates)}")
            result[family] = next(iter(candidates))
        return result

    def _self_pulses(self, connection: sqlite3.Connection, frontier: int, self_ids: dict[str, str]) -> dict[str, list[Pulse]]:
        pulses: dict[str, list[Pulse]] = {family: [] for family in GLEE_FAMILIES}
        for family, player_id in self_ids.items():
            rows = connection.execute(
                "SELECT change_sequence, frontier_sequence, family, player_id, observed_after, observed_by, games_delta, rating_delta FROM changes WHERE frontier_sequence <= ? AND family = ? AND player_id = ? AND games_delta > 0 ORDER BY frontier_sequence, change_sequence",
                (frontier, family, player_id),
            )
            for row in rows:
                observed_by = _timestamp(str(row["observed_by"]))
                if observed_by is None:
                    continue
                pulses[family].append(
                    Pulse(
                        change_sequence=int(row["change_sequence"]),
                        frontier_sequence=int(row["frontier_sequence"]),
                        family=family,
                        player_id=player_id,
                        observed_after=_timestamp(row["observed_after"]),
                        observed_by=observed_by,
                        games_delta=int(row["games_delta"]),
                        rating_delta=_finite(row["rating_delta"]),
                    )
                )
        return pulses

    def _history_games(self, *, first_at: str, completed_by: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        connection = _read_only_database(self.history_database)
        try:
            rows = connection.execute(
                "SELECT game_id, game_family, started_at, completed_at, rating_delta, revision, record_sha256 FROM games WHERE completed_at IS NOT NULL AND completed_at >= ? AND completed_at <= ? ORDER BY completed_at, game_id",
                (first_at, completed_by),
            ).fetchall()
            revision = connection.execute("SELECT COALESCE(MAX(revision), 0), COUNT(*) FROM games WHERE completed_at IS NOT NULL AND completed_at <= ?", (completed_by,)).fetchone()
        finally:
            connection.close()
        games = [
            {
                "game_id": str(row["game_id"]),
                "family": str(row["game_family"]),
                "started_at": _timestamp(row["started_at"]),
                "completed_at": _timestamp(row["completed_at"]),
                "rating_delta": _finite(row["rating_delta"]),
                "history_revision": int(row["revision"]),
                "history_record_sha256": str(row["record_sha256"]),
            }
            for row in rows
            if str(row["game_family"]) in GLEE_FAMILIES and _timestamp(row["completed_at"]) is not None
        ]
        return games, {"games_through_frontier": int(revision[1]), "maximum_game_revision": int(revision[0]), "selected_games": len(games)}

    def _load_local_games(self, history_rows: list[dict[str, object]]) -> tuple[list[LocalGame], dict[str, object]]:
        wanted = {str(row["game_id"]): row for row in history_rows}
        selected: dict[str, tuple[str, Path, dict[str, Any]]] = {}
        duplicate_files = 0
        conflicting_games: set[str] = set()
        malformed_files = 0
        scanned_files = 0
        for path in sorted(self.game_archive_root.glob("*/games/*.json")):
            family, game_id = _game_id_from_archive(path)
            if family is None or game_id not in wanted:
                continue
            scanned_files += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or str(payload.get("game_id") or "") != game_id or str(payload.get("game_family") or "") != family:
                    malformed_files += 1
                    continue
            except (OSError, json.JSONDecodeError):
                malformed_files += 1
                continue
            digest = _digest(payload)
            previous = selected.get(game_id)
            if previous is None:
                selected[game_id] = (digest, path, payload)
            elif previous[0] == digest:
                duplicate_files += 1
            else:
                conflicting_games.add(game_id)
        local_games: list[LocalGame] = []
        scope_counts: Counter[str] = Counter()
        for row in history_rows:
            game_id = str(row["game_id"])
            archive = selected.get(game_id)
            if archive is None:
                scope = "archive-missing"
                opponent_name = None
                archive_path = None
                archive_sha = None
            elif game_id in conflicting_games:
                scope = "archive-conflict"
                opponent_name = None
                archive_path = str(archive[1].relative_to(self.game_archive_root))
                archive_sha = archive[0]
            else:
                scope, opponent_name = _archive_identity(archive[2])
                archive_path = str(archive[1].relative_to(self.game_archive_root))
                archive_sha = archive[0]
            scope_counts[scope] += 1
            local_games.append(
                LocalGame(
                    game_id=game_id,
                    family=str(row["family"]),
                    started_at=_finite(row["started_at"]),
                    completed_at=float(row["completed_at"]),
                    rating_delta=_finite(row["rating_delta"]),
                    identity_scope=scope,
                    opponent_name=opponent_name,
                    archive_path=archive_path,
                    archive_sha256=archive_sha,
                    archive_conflict=game_id in conflicting_games,
                )
            )
        return local_games, {
            "selected_history_games": len(history_rows),
            "archive_files_scanned": scanned_files,
            "unique_archived_games": len(selected),
            "duplicate_archive_files": duplicate_files,
            "conflicting_archive_games": len(conflicting_games),
            "malformed_archive_files": malformed_files,
            "identity_scopes": dict(sorted(scope_counts.items())),
        }

    def _align_games(self, games: list[LocalGame], self_pulses: dict[str, list[Pulse]]) -> tuple[dict[tuple[int, str], list[LocalGame]], dict[str, object]]:
        aligned: dict[tuple[int, str], list[LocalGame]] = defaultdict(list)
        assigned: Counter[tuple[str, int]] = Counter()
        methods: Counter[str] = Counter()
        family_methods: dict[str, Counter[str]] = defaultdict(Counter)
        unmatched: list[str] = []
        for game in sorted(games, key=lambda value: (value.completed_at, value.game_id)):
            candidates: list[tuple[float, float, Pulse, str]] = []
            for pulse in self_pulses.get(game.family, []):
                start = pulse.observed_after if pulse.observed_after is not None else pulse.observed_by
                if game.completed_at < start - self.alignment_slack_s:
                    if start - game.completed_at > self.alignment_slack_s:
                        break
                    continue
                if game.completed_at > pulse.observed_by + self.alignment_slack_s:
                    continue
                distance = 0.0 if start <= game.completed_at <= pulse.observed_by else min(abs(game.completed_at - start), abs(game.completed_at - pulse.observed_by))
                method = "interval-contained" if distance == 0 else "slack-nearest"
                capacity_used = assigned[(game.family, pulse.frontier_sequence)]
                capacity_penalty = 0.0 if capacity_used < pulse.games_delta else 1_000_000.0
                midpoint = (start + pulse.observed_by) / 2.0
                candidates.append((capacity_penalty + distance, abs(game.completed_at - midpoint), pulse, method))
            if not candidates:
                methods["unmatched"] += 1
                family_methods[game.family]["unmatched"] += 1
                unmatched.append(game.game_id)
                continue
            _, _, selected, method = min(candidates, key=lambda value: (value[0], value[1], value[2].frontier_sequence))
            if assigned[(game.family, selected.frontier_sequence)] >= selected.games_delta:
                method = "capacity-overflow"
            assigned[(game.family, selected.frontier_sequence)] += 1
            aligned[(selected.frontier_sequence, game.family)].append(game)
            methods[method] += 1
            family_methods[game.family][method] += 1
        pulse_capacity = sum(pulse.games_delta for pulses in self_pulses.values() for pulse in pulses)
        return aligned, {
            "local_games": len(games),
            "aligned_games": sum(len(value) for value in aligned.values()),
            "unmatched_games": len(unmatched),
            "self_public_pulse_capacity": pulse_capacity,
            "methods": dict(sorted(methods.items())),
            "by_family": {family: dict(sorted(family_methods[family].items())) for family in GLEE_FAMILIES},
            "unmatched_game_ids_sample": unmatched[:20],
        }

    @staticmethod
    def _pulse(row: sqlite3.Row) -> Pulse | None:
        observed_by = _timestamp(str(row["observed_by"]))
        if observed_by is None:
            return None
        return Pulse(
            change_sequence=int(row["change_sequence"]),
            frontier_sequence=int(row["frontier_sequence"]),
            family=str(row["family"]),
            player_id=str(row["player_id"]),
            observed_after=_timestamp(row["observed_after"]),
            observed_by=observed_by,
            games_delta=int(row["games_delta"]),
            rating_delta=_finite(row["rating_delta"]),
        )

    @staticmethod
    def _groups(rows: Iterable[sqlite3.Row]) -> Iterator[tuple[int, list[Pulse]]]:
        current_frontier: int | None = None
        group: list[Pulse] = []
        for row in rows:
            pulse = GleeActivityEDA._pulse(row)
            if pulse is None:
                continue
            if current_frontier is not None and pulse.frontier_sequence != current_frontier:
                yield current_frontier, group
                group = []
            current_frontier = pulse.frontier_sequence
            group.append(pulse)
        if current_frontier is not None:
            yield current_frontier, group

    @staticmethod
    def _candidate_record(pulse: Pulse, player: PublicPlayer | None, score: float, exploratory_weight: float, rating_evidence_used: bool) -> dict[str, object]:
        return {
            "player_id": pulse.player_id,
            "player_name": player.name if player is not None else pulse.player_id,
            "games_delta": pulse.games_delta,
            "rating_delta": pulse.rating_delta,
            "score": _rounded(score),
            "exploratory_weight": _rounded(exploratory_weight),
            "rating_evidence_used": rating_evidence_used,
        }

    def _attribution_record(
        self,
        *,
        game: LocalGame,
        self_pulse: Pulse,
        candidates: list[Pulse],
        players: dict[tuple[str, str], PublicPlayer],
        aliases: dict[tuple[str, str], set[str]],
        rating_model: RatingEvidenceModel,
        prior_ki_total: int,
        prior_ki_misses: int,
    ) -> tuple[dict[str, object], list[str], list[tuple[Pulse, float]]]:
        clean_self_rating = self_pulse.rating_delta if self_pulse.games_delta == 1 else None
        scored: list[tuple[Pulse, float, bool]] = []
        for pulse in candidates:
            rating_used = clean_self_rating is not None and pulse.games_delta == 1 and pulse.rating_delta is not None and rating_model.training_games > 0
            score = math.log(max(1, pulse.games_delta))
            if rating_used:
                score += rating_model.score(self_delta=clean_self_rating, opponent_delta=pulse.rating_delta)
            scored.append((pulse, score, rating_used))
        maximum = max((score for _, score, _ in scored), default=0.0)
        exponentials = [math.exp(score - maximum) for _, score, _ in scored]
        unknown_weight = (prior_ki_misses + 1.0) / (prior_ki_total + 2.0)
        candidate_mass = 1.0 - unknown_weight if scored else 0.0
        denominator = sum(exponentials)
        ranked_records: list[dict[str, object]] = []
        ranked_pairs: list[tuple[Pulse, float]] = []
        for (pulse, score, rating_used), exponential in sorted(zip(scored, exponentials), key=lambda value: (-value[0][1], value[0][0].player_id)):
            weight = candidate_mass * exponential / denominator if denominator > 0 else 0.0
            ranked_records.append(self._candidate_record(pulse, players.get((game.family, pulse.player_id)), score, weight, rating_used))
            ranked_pairs.append((pulse, score))
        normalized_name = _normalize_name(game.opponent_name)
        true_ids = sorted(aliases.get((game.family, normalized_name or ""), set())) if normalized_name else []
        rank = None
        if true_ids:
            for index, record in enumerate(ranked_records, start=1):
                if record["player_id"] in true_ids:
                    rank = index
                    break
        return (
            {
                "schema_version": 1,
                "contract": ACTIVITY_EDA_CONTRACT,
                "game_id": game.game_id,
                "family": game.family,
                "started_at": _iso(game.started_at),
                "completed_at": _iso(game.completed_at),
                "identity_scope": game.identity_scope,
                "opponent_name": game.opponent_name,
                "true_public_player_ids": true_ids,
                "true_candidate_rank": rank,
                "self_rating_delta": game.rating_delta,
                "self_public_games_delta": self_pulse.games_delta,
                "self_public_rating_delta": self_pulse.rating_delta,
                "public_frontier_sequence": self_pulse.frontier_sequence,
                "candidate_count": len(ranked_records),
                "unknown_weight": _rounded(unknown_weight if scored else 1.0),
                "candidate_model": "causal-rating-feature-likelihood-plus-public-completion-multiplicity; exploratory and uncalibrated",
                "top_candidates": ranked_records[: self.top_candidates],
                "archive_path": game.archive_path,
                "archive_sha256": game.archive_sha256,
            },
            true_ids,
            ranked_pairs,
        )

    def _analyze(
        self,
        *,
        connection: sqlite3.Connection,
        snapshot: dict[str, object],
        players: dict[tuple[str, str], PublicPlayer],
        aliases: dict[tuple[str, str], set[str]],
        self_ids: dict[str, str],
        aligned_games: dict[tuple[int, str], list[LocalGame]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        frontier_limit = int(snapshot["frontier_sequence"])
        observation_quantum = float(snapshot["poll_interval_s"])
        activity: dict[tuple[str, str], ActivityAccumulator] = {}
        pair_edges: dict[tuple[str, str, str], PairAccumulator] = defaultdict(PairAccumulator)
        cross_family: Counter[tuple[str, str, str]] = Counter()
        rating_models = {family: RatingEvidenceModel() for family in GLEE_FAMILIES}
        attribution_rows: list[dict[str, object]] = []
        ki_total = 0
        ki_misses = 0
        ki_evaluable = 0
        ki_top1 = 0
        ki_top5 = 0
        causal_ki_total: Counter[str] = Counter()
        causal_ki_misses: Counter[str] = Counter()
        clean_ki_rating_pairs: list[tuple[float, float, float | None]] = []
        clean_ki_rating_pairs_by_family: dict[str, list[tuple[float, float, float | None]]] = {family: [] for family in GLEE_FAMILIES}
        family_attribution: dict[str, Counter[str]] = {family: Counter() for family in GLEE_FAMILIES}
        family_candidate_counts: dict[str, list[float]] = {family: [] for family in GLEE_FAMILIES}
        family_frontiers: Counter[str] = Counter()
        frontier_count = 0
        rows = connection.execute(
            "SELECT change_sequence, frontier_sequence, family, player_id, observed_after, observed_by, games_delta, rating_delta FROM changes WHERE frontier_sequence <= ? AND games_delta > 0 ORDER BY frontier_sequence, family, change_sequence",
            (frontier_limit,),
        )
        for frontier_sequence, pulses in self._groups(rows):
            frontier_count += 1
            by_family: dict[str, list[Pulse]] = defaultdict(list)
            by_player_families: dict[str, set[str]] = defaultdict(set)
            for pulse in pulses:
                by_family[pulse.family].append(pulse)
                by_player_families[pulse.player_id].add(pulse.family)
                key = (pulse.family, pulse.player_id)
                accumulator = activity.get(key)
                if accumulator is None:
                    accumulator = ActivityAccumulator(session_gap_s=self.session_gap_s, observation_quantum_s=observation_quantum)
                    activity[key] = accumulator
                accumulator.add(pulse)
            for player_id, families in by_player_families.items():
                for first, second in combinations(sorted(families), 2):
                    cross_family[(player_id, first, second)] += 1
            pending_training: list[tuple[str, float | None, float | None, list[float | None]]] = []
            pending_ki_coverage: list[tuple[str, bool]] = []
            for family, family_pulses in by_family.items():
                family_frontiers[family] += 1
                eligible = [pulse for pulse in family_pulses if not (players.get((family, pulse.player_id)) and (players[(family, pulse.player_id)].is_baseline or players[(family, pulse.player_id)].is_benchmark))]
                for first, second in combinations(sorted(eligible, key=lambda value: value.player_id), 2):
                    key = (family, first.player_id, second.player_id)
                    pair_edges[key].add(first, second)
                games = aligned_games.get((frontier_sequence, family), [])
                if not games:
                    continue
                self_pulse = next((pulse for pulse in family_pulses if pulse.player_id == self_ids[family]), None)
                if self_pulse is None:
                    continue
                candidates = [pulse for pulse in eligible if pulse.player_id != self_ids[family]]
                for game in sorted(games, key=lambda value: (value.completed_at, value.game_id)):
                    record, true_ids, ranked = self._attribution_record(
                        game=game,
                        self_pulse=self_pulse,
                        candidates=candidates,
                        players=players,
                        aliases=aliases,
                        rating_model=rating_models[family],
                        prior_ki_total=causal_ki_total[family],
                        prior_ki_misses=causal_ki_misses[family],
                    )
                    attribution_rows.append(record)
                    family_candidate_counts[family].append(float(record["candidate_count"]))
                    if game.identity_scope != "known":
                        family_attribution[family][game.identity_scope] += 1
                        continue
                    family_attribution[family]["ki_total"] += 1
                    candidate_ids = {pulse.player_id for pulse in candidates}
                    target_ids = candidate_ids.intersection(true_ids)
                    pending_ki_coverage.append((family, bool(target_ids)))
                    if not target_ids:
                        family_attribution[family]["ki_miss"] += 1
                        continue
                    ki_evaluable += 1
                    family_attribution[family]["ki_evaluable"] += 1
                    ranked_ids = [pulse.player_id for pulse, _ in ranked]
                    rank = min((ranked_ids.index(player_id) + 1 for player_id in target_ids), default=None)
                    ki_top1 += int(rank == 1)
                    ki_top5 += int(rank is not None and rank <= 5)
                    family_attribution[family]["ki_top1"] += int(rank == 1)
                    family_attribution[family]["ki_top5"] += int(rank is not None and rank <= 5)
                    if len(target_ids) == 1:
                        target_id = next(iter(target_ids))
                        positive = next(pulse for pulse in candidates if pulse.player_id == target_id)
                        if self_pulse.games_delta == 1 and positive.games_delta == 1 and self_pulse.rating_delta is not None and positive.rating_delta is not None:
                            clean_ki_rating_pairs.append((self_pulse.rating_delta, positive.rating_delta, game.rating_delta))
                            clean_ki_rating_pairs_by_family[family].append((self_pulse.rating_delta, positive.rating_delta, game.rating_delta))
                            negatives = [pulse.rating_delta for pulse in candidates if pulse.player_id != target_id and pulse.games_delta == 1 and pulse.rating_delta is not None]
                            pending_training.append((family, self_pulse.rating_delta, positive.rating_delta, negatives))
            for family, self_delta, positive_delta, negative_deltas in pending_training:
                rating_models[family].update(self_delta=self_delta, positive_delta=positive_delta, negative_deltas=negative_deltas)
            ki_total += len(pending_ki_coverage)
            ki_misses += sum(not covered for _, covered in pending_ki_coverage)
            for family, covered in pending_ki_coverage:
                causal_ki_total[family] += 1
                causal_ki_misses[family] += int(not covered)
        for accumulator in activity.values():
            accumulator.finish()
        traits: list[dict[str, object]] = []
        pulse_counts: dict[tuple[str, str], int] = {}
        for (family, player_id), accumulator in sorted(activity.items()):
            player = players.get((family, player_id))
            trait = accumulator.as_dict(reference_at=_timestamp(str(snapshot["frontier_completed_at"])) or accumulator.last_at or 0.0)
            pulse_counts[(family, player_id)] = int(trait["pulse_count"])
            traits.append(
                {
                    "family": family,
                    "player_id": player_id,
                    "player_name": player.name if player is not None else player_id,
                    "is_baseline": player.is_baseline if player is not None else None,
                    "is_benchmark": player.is_benchmark if player is not None else None,
                    "is_owner_best": player.is_owner_best if player is not None else None,
                    "current_games": player.current_games if player is not None else None,
                    "current_rating": player.current_rating if player is not None else None,
                    **trait,
                }
            )
        games_by_player: Counter[str] = Counter()
        for row in traits:
            games_by_player[str(row["player_id"])] += int(row["observed_games"])
        shares_by_player: dict[str, list[float]] = defaultdict(list)
        for row in traits:
            total_games = games_by_player[str(row["player_id"])]
            share = int(row["observed_games"]) / total_games if total_games else 0.0
            row["family_observed_game_share"] = _rounded(share)
            shares_by_player[str(row["player_id"])].append(share)
        for row in traits:
            shares = [share for share in shares_by_player[str(row["player_id"])] if share > 0]
            entropy = -sum(share * math.log2(share) for share in shares)
            row["cross_family_game_entropy_normalized"] = _rounded(entropy / math.log2(3) if len(shares) > 1 else 0.0)
        edge_rows: list[dict[str, object]] = []
        for (family, first_id, second_id), edge in sorted(pair_edges.items()):
            first_player = players.get((family, first_id))
            second_player = players.get((family, second_id))
            first_pulses = pulse_counts.get((family, first_id), 0)
            second_pulses = pulse_counts.get((family, second_id), 0)
            available_frontiers = family_frontiers[family]
            expected = first_pulses * second_pulses / available_frontiers if available_frontiers else None
            union = first_pulses + second_pulses - edge.co_frontiers
            edge_rows.append(
                {
                    "family": family,
                    "player_1_id": first_id,
                    "player_1_name": first_player.name if first_player is not None else first_id,
                    "player_2_id": second_id,
                    "player_2_name": second_player.name if second_player is not None else second_id,
                    "co_frontiers": edge.co_frontiers,
                    "co_game_capacity": edge.co_game_capacity,
                    "single_single_frontiers": edge.single_single_frontiers,
                    "coactivity_jaccard": _rounded(edge.co_frontiers / union if union > 0 else None),
                    "independence_expected_co_frontiers": _rounded(expected),
                    "coactivity_lift": _rounded(edge.co_frontiers / expected if expected and expected > 0 else None),
                    "rating_pairs": edge.rating_pairs,
                    "opposite_sign_fraction": _rounded(edge.opposite_sign / edge.rating_pairs if edge.rating_pairs else None),
                    "same_sign_fraction": _rounded(edge.same_sign / edge.rating_pairs if edge.rating_pairs else None),
                    "zero_involved_fraction": _rounded(edge.zero_involved / edge.rating_pairs if edge.rating_pairs else None),
                    "rating_delta_correlation": _rounded(_pearson(edge.rating_pairs, edge.sum_x, edge.sum_y, edge.sum_x2, edge.sum_y2, edge.sum_xy)),
                }
            )
        cross_rows: list[dict[str, object]] = []
        for (player_id, first_family, second_family), co_frontiers in sorted(cross_family.items()):
            first_count = pulse_counts.get((first_family, player_id), 0)
            second_count = pulse_counts.get((second_family, player_id), 0)
            player = players.get((first_family, player_id)) or players.get((second_family, player_id))
            expected = first_count * second_count / max(1, frontier_count)
            cross_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player.name if player is not None else player_id,
                    "family_1": first_family,
                    "family_2": second_family,
                    "co_frontiers": co_frontiers,
                    "family_1_pulse_frontiers": first_count,
                    "family_2_pulse_frontiers": second_count,
                    "conditional_1_given_2": _rounded(co_frontiers / second_count if second_count else None),
                    "conditional_2_given_1": _rounded(co_frontiers / first_count if first_count else None),
                    "independence_expected_co_frontiers": _rounded(expected),
                    "cross_family_lift": _rounded(co_frontiers / expected if expected > 0 else None),
                }
            )
        rating_summary = self._rating_pair_summary(clean_ki_rating_pairs)
        family_attribution_summary = {}
        for family in GLEE_FAMILIES:
            counters = family_attribution[family]
            total = counters["ki_total"]
            evaluable = counters["ki_evaluable"]
            family_attribution_summary[family] = {
                "ki_total": total,
                "ki_candidate_coverage": _rounded((total - counters["ki_miss"]) / total if total else None),
                "ki_evaluable": evaluable,
                "ki_top1_exploratory_accuracy": _rounded(counters["ki_top1"] / evaluable if evaluable else None),
                "ki_top5_exploratory_accuracy": _rounded(counters["ki_top5"] / evaluable if evaluable else None),
                "hidden_games": counters["hidden"],
                "candidate_count_p50": _rounded(_quantile(family_candidate_counts[family], 0.5)),
                "candidate_count_p90": _rounded(_quantile(family_candidate_counts[family], 0.9)),
                "causal_rating_training_games": rating_models[family].training_games,
                "clean_ki_rating_pairs": self._rating_pair_summary(clean_ki_rating_pairs_by_family[family]),
            }
        identity_counts = Counter(str(row["identity_scope"]) for row in attribution_rows)
        candidate_counts = [int(row["candidate_count"]) for row in attribution_rows]
        hidden_rows = [row for row in attribution_rows if row["identity_scope"] == "hidden"]
        summary = {
            "public_activity": {
                "positive_change_rows": sum(accumulator.pulse_count for accumulator in activity.values()),
                "observed_game_increments": sum(accumulator.observed_games for accumulator in activity.values()),
                "active_player_family_profiles": len(traits),
                "family_pulse_frontiers": dict(sorted(family_frontiers.items())),
                "coactivity_edges": len(edge_rows),
                "cross_family_profiles": len(cross_rows),
            },
            "attribution": {
                "aligned_identity_scopes": dict(sorted(identity_counts.items())),
                "ki_total": ki_total,
                "ki_candidate_coverage": _rounded((ki_total - ki_misses) / ki_total if ki_total else None),
                "ki_evaluable": ki_evaluable,
                "ki_top1_exploratory_accuracy": _rounded(ki_top1 / ki_evaluable if ki_evaluable else None),
                "ki_top5_exploratory_accuracy": _rounded(ki_top5 / ki_evaluable if ki_evaluable else None),
                "causal_rating_training_games": sum(model.training_games for model in rating_models.values()),
                "causal_rating_training_games_by_family": {family: rating_models[family].training_games for family in GLEE_FAMILIES},
                "hidden_games_with_candidates": sum(int(row["candidate_count"]) > 0 for row in hidden_rows),
                "hidden_games": len(hidden_rows),
                "candidate_count_p50": _rounded(_quantile([float(value) for value in candidate_counts], 0.5)),
                "candidate_count_p90": _rounded(_quantile([float(value) for value in candidate_counts], 0.9)),
                "candidate_weights_are_calibrated": False,
                "by_family": family_attribution_summary,
            },
            "clean_ki_rating_pairs": rating_summary,
            "top_activity": {
                family: sorted((row for row in traits if row["family"] == family and not row["is_baseline"] and not row["is_benchmark"]), key=lambda row: (-int(row["observed_games"]), str(row["player_name"]).casefold()))[:10]
                for family in GLEE_FAMILIES
            },
            "self_activity": [row for row in traits if row["player_id"] in set(self_ids.values())],
            "top_hidden_examples": sorted((row for row in hidden_rows if row["top_candidates"]), key=lambda row: -float(row["top_candidates"][0]["exploratory_weight"] or 0.0))[:20],
        }
        return traits, edge_rows, cross_rows, attribution_rows, summary

    @staticmethod
    def _rating_pair_summary(rows: list[tuple[float, float, float | None]]) -> dict[str, object]:
        opposite = 0
        same = 0
        zero = 0
        sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
        public_self_errors: list[float] = []
        for self_public, opponent_public, self_exact in rows:
            signs = {_sign(self_public), _sign(opponent_public)}
            if "zero" in signs:
                zero += 1
            elif _sign(self_public) == _sign(opponent_public):
                same += 1
            else:
                opposite += 1
            sum_x += self_public
            sum_y += opponent_public
            sum_x2 += self_public * self_public
            sum_y2 += opponent_public * opponent_public
            sum_xy += self_public * opponent_public
            if self_exact is not None:
                public_self_errors.append(self_public - self_exact)
        count = len(rows)
        return {
            "count": count,
            "opposite_sign_fraction": _rounded(opposite / count if count else None),
            "same_sign_fraction": _rounded(same / count if count else None),
            "zero_involved_fraction": _rounded(zero / count if count else None),
            "rating_delta_correlation": _rounded(_pearson(count, sum_x, sum_y, sum_x2, sum_y2, sum_xy)),
            "public_minus_authenticated_self_delta_p50": _rounded(_quantile(public_self_errors, 0.5)),
            "public_minus_authenticated_self_delta_p90_abs": _rounded(_quantile([abs(value) for value in public_self_errors], 0.9)),
            "interpretation": "Clean KI pairs require one DeepRMM-01 public game and one true-opponent public game in the aligned frontier. Opposite-sign movement is measured evidence, not a pairing rule.",
        }

    def _readme(self, summary: dict[str, object]) -> str:
        snapshot = summary["source_frontier"]
        public = summary["public_activity"]
        attribution = summary["attribution"]
        rating = summary["clean_ki_rating_pairs"]
        lines = [
            "# GLEE activity corpus EDA v1",
            "",
            f"**Status:** Read-only exploratory snapshot through reporter frontier `{snapshot['frontier_sequence']}` at {snapshot['frontier_completed_at']}; no result is connected to live routing or a named dossier.",
            "",
            "## Corpus",
            "",
            f"The public corpus spans {snapshot['first_started_at']} through {snapshot['frontier_completed_at']} and contains {public['positive_change_rows']:,} positive player-family change rows representing {public['observed_game_increments']:,} public game-count increments. The EDA derives {public['active_player_family_profiles']:,} player-family activity profiles, {public['coactivity_edges']:,} same-family coactivity edges, and {public['cross_family_profiles']:,} cross-family coupling profiles without copying source rows.",
            "",
            "## Local-game alignment",
            "",
            f"The authenticated local-history join aligned {summary['alignment']['aligned_games']:,} of {summary['alignment']['local_games']:,} games to DeepRMM-01 public count pulses. Identity-bearing archives classify the aligned games as `{attribution['aligned_identity_scopes']}`. KI candidate coverage is {attribution['ki_candidate_coverage']}; among {attribution['ki_evaluable']:,} evaluable KI games, the causal exploratory ranking reaches top-one accuracy {attribution['ki_top1_exploratory_accuracy']} and top-5 accuracy {attribution['ki_top5_exploratory_accuracy']}.",
            "",
            "## Rating-pair finding",
            "",
            f"Among {rating['count']:,} clean KI single-game pulse pairs, opposite-sign public rating movement occurs in {rating['opposite_sign_fraction']} of cases, same-sign movement in {rating['same_sign_fraction']}, and the Pearson correlation is {rating['rating_delta_correlation']}. This directly rejects use of anti-alignment as a hard same-game rule: it is one likelihood feature whose value must be learned conditionally.",
            "",
            "## Hidden-game candidates",
            "",
            f"The snapshot contains {attribution['hidden_games']:,} aligned HI games, of which {attribution['hidden_games_with_candidates']:,} have at least one same-frontier public candidate. The median candidate-set size is {attribution['candidate_count_p50']} and p90 is {attribution['candidate_count_p90']}. Candidate weights use only causally prior KI rating-feature counts plus public completion multiplicity; they are exploratory and explicitly uncalibrated, preserve unknown mass, and cannot update named evidence.",
            "",
            "## Artifacts",
            "",
            "- `summary.json` contains the bounded overview, top activity profiles, DeepRMM-01's own activity profile, and selected HI examples.",
            "- `activity-traits.csv` contains one derived activity profile per public player and family.",
            "- `coactivity-edges.csv` contains same-family public co-pulse, lift, and rating-sign features.",
            "- `cross-family-coupling.csv` contains within-player cross-family synchrony features.",
            "- `game-attribution.jsonl` contains KI validation and HI candidate sets keyed by local `game_id` and immutable archive reference.",
            "- `manifest.json` pins the logical source frontier and artifact hashes.",
            "",
            "## Limits",
            "",
            "The public window is short, completion timestamps are interval-censored, one polling frontier can aggregate several games, public ratings are asymmetric and can be revised by platform mechanics, names can change or collide, and the current candidate scorer does not yet use move timing or game style. Same-frontier coactivity is not proof that 2 players faced each other, empirical activity forecasts mix behavioral regimes and are not calibrated probabilities, and high-confidence-looking HI weights are not authenticated identity.",
            "",
        ]
        return "\n".join(lines)

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.output_dir.iterdir() if not path.name.startswith(".")]
        if existing:
            raise FileExistsError(f"activity EDA output directory is not empty: {self.output_dir}")
        connection = _read_only_database(self.reporter_database)
        try:
            connection.execute("BEGIN")
            snapshot = self._snapshot(connection)
            players, aliases = self._players(connection, int(snapshot["frontier_sequence"]))
            self_ids = self._self_ids(players, aliases)
            self_pulses = self._self_pulses(connection, int(snapshot["frontier_sequence"]), self_ids)
            history_rows, history_inventory = self._history_games(first_at=str(snapshot["first_started_at"]), completed_by=str(snapshot["frontier_completed_at"]))
            local_games, archive_inventory = self._load_local_games(history_rows)
            aligned_games, alignment = self._align_games(local_games, self_pulses)
            traits, edges, cross_rows, attribution_rows, analysis_summary = self._analyze(
                connection=connection,
                snapshot=snapshot,
                players=players,
                aliases=aliases,
                self_ids=self_ids,
                aligned_games=aligned_games,
            )
            connection.execute("ROLLBACK")
        finally:
            connection.close()
        summary: dict[str, object] = {
            "schema_version": 1,
            "contract": ACTIVITY_EDA_CONTRACT,
            "status": "offline-exploratory",
            "source_frontier": {
                **snapshot,
                "reporter_database": str(self.reporter_database),
                "history_database": str(self.history_database),
                "game_archive_root": str(self.game_archive_root),
                "self_name": self.self_name,
                "self_player_ids": self_ids,
            },
            "parameters": {
                "session_gap_s": self.session_gap_s,
                "alignment_slack_s": self.alignment_slack_s,
                "top_candidates": self.top_candidates,
            },
            "history_inventory": history_inventory,
            "archive_inventory": archive_inventory,
            "alignment": alignment,
            **analysis_summary,
        }
        trait_fields = (
            "family", "player_id", "player_name", "is_baseline", "is_benchmark", "is_owner_best", "current_games", "current_rating", "pulse_count", "observed_games", "family_observed_game_share", "cross_family_game_entropy_normalized", "multi_game_pulse_fraction", "first_observed_at", "last_observed_at", "observed_span_hours", "inter_pulse_mean_s", "inter_pulse_std_s", "inter_pulse_p10_s", "inter_pulse_p50_s", "inter_pulse_p90_s", "inter_pulse_max_s", "burstiness", "seconds_since_last_pulse", "currently_inside_session_gap", "empirical_next_pulse_probability_60s", "empirical_next_pulse_probability_300s", "empirical_next_pulse_probability_900s", "session_gap_s", "session_count", "session_duration_p50_s", "session_duration_p90_s", "session_games_p50", "session_games_p90", "games_per_active_hour", "utc_hour_entropy_normalized", "dominant_utc_hour", "active_utc_hour_count", "utc_hour_games", "rating_change_count", "mean_abs_public_rating_delta",
        )
        edge_fields = (
            "family", "player_1_id", "player_1_name", "player_2_id", "player_2_name", "co_frontiers", "co_game_capacity", "single_single_frontiers", "coactivity_jaccard", "independence_expected_co_frontiers", "coactivity_lift", "rating_pairs", "opposite_sign_fraction", "same_sign_fraction", "zero_involved_fraction", "rating_delta_correlation",
        )
        cross_fields = (
            "player_id", "player_name", "family_1", "family_2", "co_frontiers", "family_1_pulse_frontiers", "family_2_pulse_frontiers", "conditional_1_given_2", "conditional_2_given_1", "independence_expected_co_frontiers", "cross_family_lift",
        )
        _write_csv(self.output_dir / "activity-traits.csv", trait_fields, traits)
        _write_csv(self.output_dir / "coactivity-edges.csv", edge_fields, edges)
        _write_csv(self.output_dir / "cross-family-coupling.csv", cross_fields, cross_rows)
        _write_jsonl(self.output_dir / "game-attribution.jsonl", attribution_rows)
        _write_json(self.output_dir / "summary.json", summary)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifact_names = ("README.md", "summary.json", "activity-traits.csv", "coactivity-edges.csv", "cross-family-coupling.csv", "game-attribution.jsonl")
        manifest = {
            "schema_version": 1,
            "contract": ACTIVITY_EDA_CONTRACT,
            "source_frontier": summary["source_frontier"],
            "parameters": summary["parameters"],
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifact_names},
            "implementation_sha256": _file_digest(Path(__file__)),
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {
            "contract": ACTIVITY_EDA_CONTRACT,
            "output_dir": str(self.output_dir),
            "frontier_sequence": snapshot["frontier_sequence"],
            "public_activity": summary["public_activity"],
            "alignment": summary["alignment"],
            "attribution": summary["attribution"],
            "clean_ki_rating_pairs": summary["clean_ki_rating_pairs"],
            "manifest_sha256": _file_digest(self.output_dir / "manifest.json"),
        }
