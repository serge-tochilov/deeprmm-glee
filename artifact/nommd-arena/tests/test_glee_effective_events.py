import json
import sqlite3
from datetime import datetime, timedelta, timezone

from nommd_arena.glee_effective_events import CounterObservation, EffectiveEvent, derive_effective_events


def _stamp(second: int) -> str:
    return (datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat(timespec="microseconds")


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE row_versions (family TEXT NOT NULL, player_id TEXT NOT NULL, sequence INTEGER NOT NULL, change_kind TEXT NOT NULL, row_sha256 TEXT, row_json TEXT, PRIMARY KEY (family, player_id, sequence));
        CREATE TABLE changes (change_sequence INTEGER PRIMARY KEY, frontier_sequence INTEGER NOT NULL, family TEXT NOT NULL, player_id TEXT NOT NULL, change_kind TEXT NOT NULL, observed_after TEXT, observed_by TEXT NOT NULL, games_delta INTEGER, rating_delta REAL);
        """
    )
    return connection


def _insert(connection: sqlite3.Connection, sequence: int, *, count: int | None, rating: float | None, games_delta: int | None, rating_delta: float | None, kind: str = "changed", family: str = "bargaining", player_id: str = "agent-a") -> None:
    payload = None if count is None else json.dumps({"games_played": count, "rating": rating, "player_name": "Agent A"}, sort_keys=True)
    connection.execute("INSERT INTO row_versions VALUES (?, ?, ?, ?, ?, ?)", (family, player_id, sequence, kind, f"row-{sequence}" if payload is not None else None, payload))
    connection.execute("INSERT INTO changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (sequence, sequence, family, player_id, kind, _stamp((sequence - 1) * 10), _stamp(sequence * 10), games_delta, rating_delta))


def _derive(connection: sqlite3.Connection, frontier: int) -> tuple[list[EffectiveEvent], list[CounterObservation], dict[str, object]]:
    events: list[EffectiveEvent] = []
    observations: list[CounterObservation] = []
    summary = derive_effective_events(connection, frontier_sequence=frontier, event_sink=events.append, observation_sink=observations.append)
    return events, observations, summary


def test_regression_recovery_does_not_reuse_game_capacity() -> None:
    connection = _connection()
    _insert(connection, 1, count=100, rating=1500.0, games_delta=None, rating_delta=None, kind="appeared")
    _insert(connection, 2, count=99, rating=1499.0, games_delta=-1, rating_delta=-1.0)
    _insert(connection, 3, count=100, rating=1500.5, games_delta=1, rating_delta=1.5)
    _insert(connection, 4, count=101, rating=1502.0, games_delta=1, rating_delta=1.5)

    events, observations, summary = _derive(connection, 4)

    assert len(events) == 1
    assert events[0].previous_high_water == 100
    assert events[0].new_high_water == 101
    assert events[0].games_delta == 1
    assert events[0].clean_rating_delta == 1.5
    assert [row.kind for row in observations] == ["regression", "recovery", "effective-increment"]
    assert summary["totals"]["raw_positive_additions"] == 2
    assert summary["totals"]["effective_additions"] == 1
    assert summary["totals"]["recovered_raw_additions"] == 1
    assert summary["invariants"]["violation_count"] == 0


def test_absence_and_multi_game_jump_remain_aggregate_and_unclean() -> None:
    connection = _connection()
    _insert(connection, 1, count=10, rating=1400.0, games_delta=None, rating_delta=None, kind="appeared")
    _insert(connection, 2, count=None, rating=None, games_delta=None, rating_delta=None, kind="disappeared")
    _insert(connection, 3, count=12, rating=1404.0, games_delta=None, rating_delta=None, kind="appeared")
    _insert(connection, 4, count=15, rating=1410.0, games_delta=3, rating_delta=6.0)

    events, observations, summary = _derive(connection, 4)

    assert [event.games_delta for event in events] == [2, 3]
    assert events[0].crossed_absence is True
    assert events[0].rating_status == "multi-game"
    assert events[0].clean_rating_delta is None
    assert events[1].rating_status == "multi-game"
    assert events[1].observed_after == _stamp(30)
    assert [row.kind for row in observations] == ["disappearance", "effective-increment", "effective-increment"]
    assert summary["invariants"]["expected_additions_from_high_water_minus_baseline"] == 5
    assert summary["invariants"]["effective_additions"] == 5


def test_rating_only_high_water_observation_refreshes_clean_anchor() -> None:
    connection = _connection()
    _insert(connection, 1, count=100, rating=1500.0, games_delta=None, rating_delta=None, kind="appeared")
    _insert(connection, 2, count=100, rating=1502.0, games_delta=0, rating_delta=2.0)
    _insert(connection, 3, count=101, rating=1503.25, games_delta=1, rating_delta=1.25)

    events, observations, summary = _derive(connection, 3)

    assert [row.kind for row in observations] == ["rating-only-at-high-water", "effective-increment"]
    assert events[0].previous_rating_anchor == 1502.0
    assert events[0].rating_delta_from_anchor == 1.25
    assert events[0].clean_rating_delta == 1.25
    assert summary["totals"]["rating_only_rows"] == 1


def test_reconstruction_is_deterministic() -> None:
    connection = _connection()
    _insert(connection, 1, count=7, rating=1300.0, games_delta=None, rating_delta=None, kind="appeared")
    _insert(connection, 2, count=8, rating=1302.0, games_delta=1, rating_delta=2.0)

    first_events, first_observations, first_summary = _derive(connection, 2)
    second_events, second_observations, second_summary = _derive(connection, 2)

    assert first_events == second_events
    assert first_observations == second_observations
    assert first_summary == second_summary
