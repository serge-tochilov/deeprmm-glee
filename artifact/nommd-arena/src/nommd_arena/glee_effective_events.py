"""Monotonic high-water reconstruction of public GLEE game-count events."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


EFFECTIVE_EVENTS_CONTRACT = "glee-public-effective-events-v2"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class EffectiveEvent:
    """One public count increase beyond every count previously observed for the player-family."""

    source_change_sequence: int
    frontier_sequence: int
    family: str
    player_id: str
    observed_after: str | None
    observed_by: str
    previous_high_water: int
    new_high_water: int
    games_delta: int
    previous_rating_anchor: float | None
    current_rating: float | None
    rating_delta_from_anchor: float | None
    clean_rating_delta: float | None
    rating_status: str
    raw_previous_count: int | None
    raw_games_delta: int | None
    crossed_regression: bool
    crossed_absence: bool
    source_change_kind: str
    source_row_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract": EFFECTIVE_EVENTS_CONTRACT,
            "source_change_sequence": self.source_change_sequence,
            "frontier_sequence": self.frontier_sequence,
            "family": self.family,
            "player_id": self.player_id,
            "observed_after": self.observed_after,
            "observed_by": self.observed_by,
            "previous_high_water": self.previous_high_water,
            "new_high_water": self.new_high_water,
            "games_delta": self.games_delta,
            "previous_rating_anchor": self.previous_rating_anchor,
            "current_rating": self.current_rating,
            "rating_delta_from_anchor": self.rating_delta_from_anchor,
            "clean_rating_delta": self.clean_rating_delta,
            "rating_status": self.rating_status,
            "raw_previous_count": self.raw_previous_count,
            "raw_games_delta": self.raw_games_delta,
            "crossed_regression": self.crossed_regression,
            "crossed_absence": self.crossed_absence,
            "source_change_kind": self.source_change_kind,
            "source_row_sha256": self.source_row_sha256,
        }


@dataclass(frozen=True)
class CounterObservation:
    """One non-baseline public observation classified relative to the running high water."""

    source_change_sequence: int
    frontier_sequence: int
    family: str
    player_id: str
    kind: str
    observed_after: str | None
    observed_by: str
    previous_observed_count: int | None
    observed_count: int | None
    high_water_before: int | None
    high_water_after: int | None
    raw_games_delta: int | None
    raw_rating_delta: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract": EFFECTIVE_EVENTS_CONTRACT,
            "source_change_sequence": self.source_change_sequence,
            "frontier_sequence": self.frontier_sequence,
            "family": self.family,
            "player_id": self.player_id,
            "kind": self.kind,
            "observed_after": self.observed_after,
            "observed_by": self.observed_by,
            "previous_observed_count": self.previous_observed_count,
            "observed_count": self.observed_count,
            "high_water_before": self.high_water_before,
            "high_water_after": self.high_water_after,
            "raw_games_delta": self.raw_games_delta,
            "raw_rating_delta": self.raw_rating_delta,
        }


@dataclass
class _HighWaterState:
    baseline_count: int
    high_water: int
    rating_anchor: float | None
    rating_anchor_sequence: int
    high_water_observed_by: str
    current_count: int | None
    current_rating: float | None
    present: bool
    effective_additions: int = 0


class HighWaterDeriver:
    """Consume reporter row versions in frontier order and emit effective events without reusing recovered capacity."""

    def __init__(self) -> None:
        self.states: dict[tuple[str, str], _HighWaterState] = {}
        self.totals: Counter[str] = Counter()
        self.by_family: dict[str, Counter[str]] = defaultdict(Counter)
        self.source_digest = hashlib.sha256()
        self.event_digest = hashlib.sha256()
        self.observation_digest = hashlib.sha256()
        self.invariant_violations: list[dict[str, object]] = []

    def _increment(self, family: str, key: str, amount: int = 1) -> None:
        self.totals[key] += amount
        self.by_family[family][key] += amount

    def _digest_source(self, row: Mapping[str, object]) -> None:
        payload = {
            "change_sequence": row["change_sequence"],
            "frontier_sequence": row["frontier_sequence"],
            "family": row["family"],
            "player_id": row["player_id"],
            "change_kind": row["change_kind"],
            "observed_after": row["observed_after"],
            "observed_by": row["observed_by"],
            "games_delta": row["games_delta"],
            "rating_delta": row["rating_delta"],
            "row_sha256": row["row_sha256"],
            "observed_count": row["observed_count"],
            "observed_rating": row["observed_rating"],
        }
        self.source_digest.update(_canonical(payload).encode("utf-8") + b"\n")

    @staticmethod
    def _payload(row: Mapping[str, object]) -> tuple[int | None, float | None]:
        return _integer(row["observed_count"]), _finite(row["observed_rating"])

    def consume(self, row: Mapping[str, object]) -> tuple[EffectiveEvent | None, CounterObservation | None]:
        self._digest_source(row)
        family = str(row["family"])
        player_id = str(row["player_id"])
        key = (family, player_id)
        sequence = int(row["frontier_sequence"])
        change_sequence = int(row["change_sequence"])
        change_kind = str(row["change_kind"])
        observed_after = str(row["observed_after"]) if row["observed_after"] is not None else None
        observed_by = str(row["observed_by"])
        raw_games_delta = _integer(row["games_delta"])
        raw_rating_delta = _finite(row["rating_delta"])
        observed_count, observed_rating = self._payload(row)
        self._increment(family, "source_rows")
        if raw_games_delta is None:
            self._increment(family, "raw_null_rows")
        elif raw_games_delta > 0:
            self._increment(family, "raw_positive_rows")
            self._increment(family, "raw_positive_additions", raw_games_delta)
        elif raw_games_delta < 0:
            self._increment(family, "raw_negative_rows")
            self._increment(family, "raw_negative_removals", -raw_games_delta)
        else:
            self._increment(family, "raw_zero_rows")
        state = self.states.get(key)
        if state is None:
            if observed_count is None:
                self._increment(family, "unusable_initial_rows")
                return None, None
            self.states[key] = _HighWaterState(
                baseline_count=observed_count,
                high_water=observed_count,
                rating_anchor=observed_rating,
                rating_anchor_sequence=sequence,
                high_water_observed_by=observed_by,
                current_count=observed_count,
                current_rating=observed_rating,
                present=True,
            )
            self._increment(family, "baseline_rows")
            return None, None
        previous_count = state.current_count
        high_water_before = state.high_water
        if observed_count is None:
            state.current_count = None
            state.current_rating = None
            state.present = False
            self._increment(family, "disappearance_rows")
            observation = CounterObservation(change_sequence, sequence, family, player_id, "disappearance", observed_after, observed_by, previous_count, None, high_water_before, state.high_water, raw_games_delta, raw_rating_delta)
            self.observation_digest.update(_canonical(observation.as_dict()).encode("utf-8") + b"\n")
            return None, observation
        crossed_absence = not state.present or previous_count is None
        crossed_regression = previous_count is not None and previous_count < state.high_water
        if observed_count > state.high_water:
            effective_delta = observed_count - state.high_water
            interval_start = observed_after if previous_count == state.high_water and not crossed_absence else state.high_water_observed_by
            rating_delta = round(observed_rating - state.rating_anchor, 3) if observed_rating is not None and state.rating_anchor is not None else None
            clean = effective_delta == 1 and raw_games_delta == 1 and previous_count == state.high_water and not crossed_absence and rating_delta is not None
            if clean:
                rating_status = "clean-single"
                clean_rating_delta = rating_delta
                self._increment(family, "clean_rating_events")
            elif effective_delta > 1:
                rating_status = "multi-game"
                clean_rating_delta = None
            elif crossed_absence:
                rating_status = "absence-crossed"
                clean_rating_delta = None
            elif crossed_regression:
                rating_status = "regression-crossed"
                clean_rating_delta = None
            else:
                rating_status = "unclean-single"
                clean_rating_delta = None
            event = EffectiveEvent(
                source_change_sequence=change_sequence,
                frontier_sequence=sequence,
                family=family,
                player_id=player_id,
                observed_after=interval_start,
                observed_by=observed_by,
                previous_high_water=state.high_water,
                new_high_water=observed_count,
                games_delta=effective_delta,
                previous_rating_anchor=state.rating_anchor,
                current_rating=observed_rating,
                rating_delta_from_anchor=rating_delta,
                clean_rating_delta=clean_rating_delta,
                rating_status=rating_status,
                raw_previous_count=previous_count,
                raw_games_delta=raw_games_delta,
                crossed_regression=crossed_regression,
                crossed_absence=crossed_absence,
                source_change_kind=change_kind,
                source_row_sha256=str(row["row_sha256"]) if row["row_sha256"] is not None else None,
            )
            state.effective_additions += effective_delta
            state.high_water = observed_count
            state.rating_anchor = observed_rating
            state.rating_anchor_sequence = sequence
            state.high_water_observed_by = observed_by
            state.current_count = observed_count
            state.current_rating = observed_rating
            state.present = True
            self._increment(family, "effective_event_rows")
            self._increment(family, "effective_additions", effective_delta)
            self._increment(family, "multi_game_event_rows", int(effective_delta > 1))
            self._increment(family, "events_crossing_regression", int(crossed_regression))
            self._increment(family, "events_crossing_absence", int(crossed_absence))
            if raw_games_delta is not None and raw_games_delta > effective_delta:
                self._increment(family, "recovered_raw_additions", raw_games_delta - effective_delta)
            self.event_digest.update(_canonical(event.as_dict()).encode("utf-8") + b"\n")
            observation = CounterObservation(change_sequence, sequence, family, player_id, "effective-increment", observed_after, observed_by, previous_count, observed_count, high_water_before, state.high_water, raw_games_delta, raw_rating_delta)
            self.observation_digest.update(_canonical(observation.as_dict()).encode("utf-8") + b"\n")
            return event, observation
        if observed_count == state.high_water:
            recovered = previous_count is None or previous_count < state.high_water
            rating_changed = observed_rating != state.rating_anchor
            if recovered:
                kind = "recovery"
                self._increment(family, "recovery_rows")
                if raw_games_delta is not None and raw_games_delta > 0:
                    self._increment(family, "recovered_raw_additions", raw_games_delta)
            elif rating_changed:
                kind = "rating-only-at-high-water"
                self._increment(family, "rating_only_rows")
            else:
                kind = "metadata-only-at-high-water"
                self._increment(family, "metadata_only_rows")
            state.rating_anchor = observed_rating
            state.rating_anchor_sequence = sequence
            state.high_water_observed_by = observed_by
            state.current_count = observed_count
            state.current_rating = observed_rating
            state.present = True
            observation = CounterObservation(change_sequence, sequence, family, player_id, kind, observed_after, observed_by, previous_count, observed_count, high_water_before, state.high_water, raw_games_delta, raw_rating_delta)
            self.observation_digest.update(_canonical(observation.as_dict()).encode("utf-8") + b"\n")
            return None, observation
        state.current_count = observed_count
        state.current_rating = observed_rating
        state.present = True
        self._increment(family, "regression_rows")
        if raw_games_delta is not None and raw_games_delta > 0:
            self._increment(family, "recovered_raw_additions", raw_games_delta)
        observation = CounterObservation(change_sequence, sequence, family, player_id, "regression", observed_after, observed_by, previous_count, observed_count, high_water_before, state.high_water, raw_games_delta, raw_rating_delta)
        self.observation_digest.update(_canonical(observation.as_dict()).encode("utf-8") + b"\n")
        return None, observation

    def finish(self, *, frontier_sequence: int) -> dict[str, object]:
        expected_additions = 0
        for (family, player_id), state in sorted(self.states.items()):
            expected = state.high_water - state.baseline_count
            expected_additions += expected
            if expected != state.effective_additions:
                self.invariant_violations.append({"family": family, "player_id": player_id, "baseline_count": state.baseline_count, "high_water": state.high_water, "expected_additions": expected, "observed_effective_additions": state.effective_additions})
        effective = self.totals["effective_additions"]
        raw_positive = self.totals["raw_positive_additions"]
        summary = {
            "schema_version": 1,
            "contract": EFFECTIVE_EVENTS_CONTRACT,
            "frontier_sequence": frontier_sequence,
            "player_family_states": len(self.states),
            "source_rows_sha256": self.source_digest.hexdigest(),
            "effective_events_sha256": self.event_digest.hexdigest(),
            "counter_observations_sha256": self.observation_digest.hexdigest(),
            "totals": dict(sorted(self.totals.items())),
            "by_family": {family: dict(sorted(values.items())) for family, values in sorted(self.by_family.items())},
            "invariants": {
                "expected_additions_from_high_water_minus_baseline": expected_additions,
                "effective_additions": effective,
                "violation_count": len(self.invariant_violations),
                "violations_sample": self.invariant_violations[:20],
            },
            "raw_positive_inflation_fraction": round((raw_positive - effective) / effective, 9) if effective else None,
            "raw_positive_recovery_fraction": round((raw_positive - effective) / raw_positive, 9) if raw_positive else None,
        }
        return summary


def reporter_observation_rows(connection: sqlite3.Connection, *, frontier_sequence: int) -> Iterable[sqlite3.Row]:
    """Stream every stored row version through one pinned frontier with its causal observation interval."""
    return connection.execute(
        """
        SELECT ch.change_sequence, ch.frontier_sequence, ch.family, ch.player_id, ch.change_kind,
               ch.observed_after, ch.observed_by, ch.games_delta, ch.rating_delta,
               rv.row_sha256,
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


def derive_effective_events(
    connection: sqlite3.Connection,
    *,
    frontier_sequence: int,
    event_sink: Callable[[EffectiveEvent], None] | None = None,
    observation_sink: Callable[[CounterObservation], None] | None = None,
) -> dict[str, object]:
    """Derive one deterministic high-water corpus from an open read-only reporter transaction."""
    deriver = HighWaterDeriver()
    for row in reporter_observation_rows(connection, frontier_sequence=frontier_sequence):
        event, observation = deriver.consume(row)
        if event is not None and event_sink is not None:
            event_sink(event)
        if observation is not None and observation_sink is not None:
            observation_sink(observation)
    return deriver.finish(frontier_sequence=frontier_sequence)
