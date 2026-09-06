import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nommd_arena.glee_activity_eda import GleeActivityEDA, RatingEvidenceModel


def _stamp(origin: datetime, seconds: int) -> str:
    return (origin + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _player(name: str, games: int, rating: float, *, owner_best: bool = True) -> str:
    return json.dumps({"player_name": name, "games_played": games, "rating": rating, "is_baseline": False, "is_benchmark": False, "is_owner_best": owner_best}, sort_keys=True)


def _reporter_database(path: Path, origin: datetime) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE frontiers (sequence INTEGER PRIMARY KEY, frontier_id TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL);
        CREATE TABLE row_versions (family TEXT NOT NULL, player_id TEXT NOT NULL, sequence INTEGER NOT NULL, row_json TEXT);
        CREATE TABLE changes (change_sequence INTEGER PRIMARY KEY, frontier_sequence INTEGER NOT NULL, family TEXT NOT NULL, player_id TEXT NOT NULL, observed_after TEXT, observed_by TEXT NOT NULL, games_delta INTEGER, rating_delta REAL);
        """
    )
    for sequence, second in enumerate((0, 10, 20, 30, 700), start=1):
        connection.execute("INSERT INTO frontiers VALUES (?, ?, ?, ?)", (sequence, f"f-{sequence}", _stamp(origin, second), _stamp(origin, second + 1)))
    rows = []
    for family in ("bargaining", "negotiation", "persuasion"):
        rows.append((family, "self-id", 1, _player("DeepRMM-01", 100, 1800.0)))
    rows.extend(
        [
            ("bargaining", "alice-id", 1, _player("Alice", 80, 1900.0)),
            ("bargaining", "bob-id", 1, _player("Bob", 60, 1700.0)),
            ("negotiation", "alice-id", 1, _player("Alice", 40, 1800.0)),
        ]
    )
    connection.executemany("INSERT INTO row_versions VALUES (?, ?, ?, ?)", rows)
    changes = [
        (1, 2, "bargaining", "self-id", _stamp(origin, 1), _stamp(origin, 11), 1, 2.0),
        (2, 2, "bargaining", "alice-id", _stamp(origin, 1), _stamp(origin, 11), 1, -3.0),
        (3, 2, "bargaining", "bob-id", _stamp(origin, 1), _stamp(origin, 11), 1, 1.0),
        (4, 3, "bargaining", "self-id", _stamp(origin, 11), _stamp(origin, 21), 1, -1.0),
        (5, 3, "bargaining", "alice-id", _stamp(origin, 11), _stamp(origin, 21), 1, 2.0),
        (6, 3, "bargaining", "bob-id", _stamp(origin, 11), _stamp(origin, 21), 1, -2.0),
        (7, 4, "negotiation", "self-id", _stamp(origin, 21), _stamp(origin, 31), 1, 1.0),
        (8, 4, "negotiation", "alice-id", _stamp(origin, 21), _stamp(origin, 31), 1, -1.0),
        (9, 5, "bargaining", "self-id", _stamp(origin, 31), _stamp(origin, 701), 1, 3.0),
        (10, 5, "bargaining", "alice-id", _stamp(origin, 31), _stamp(origin, 701), 1, -2.0),
    ]
    connection.executemany("INSERT INTO changes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", changes)
    connection.commit()
    connection.close()
    (path.parent / "manifest.json").write_text(json.dumps({"contract": "glee-arena-reporter-v1", "poll_interval_s": 10.0}), encoding="utf-8")


def _history_database(path: Path, origin: datetime) -> list[dict[str, object]]:
    games = [
        {"game_id": "g-known-1", "family": "bargaining", "started": 2, "completed": 5, "rating_delta": 2.0, "opponent": {"type": "agent", "name": "Alice"}},
        {"game_id": "g-hidden", "family": "bargaining", "started": 12, "completed": 15, "rating_delta": -1.0, "opponent": {"type": "hidden", "name": None}},
        {"game_id": "g-known-2", "family": "bargaining", "started": 690, "completed": 695, "rating_delta": 3.0, "opponent": {"type": "agent", "name": "Alice"}},
    ]
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY, game_family TEXT, started_at TEXT, completed_at TEXT, rating_delta REAL, revision INTEGER, record_sha256 TEXT)")
    for game in games:
        connection.execute("INSERT INTO games VALUES (?, ?, ?, ?, ?, 1, ?)", (game["game_id"], game["family"], _stamp(origin, int(game["started"])), _stamp(origin, int(game["completed"])), game["rating_delta"], f"history-{game['game_id']}"))
    connection.commit()
    connection.close()
    return games


def _archives(root: Path, games: list[dict[str, object]]) -> None:
    game_root = root / "run-a" / "games"
    game_root.mkdir(parents=True)
    for game in games:
        payload = {"game_id": game["game_id"], "game_family": game["family"], "status": "completed", "your_player": "player_1", "opponent": game["opponent"], "game_state": {"phase": "completed", "history": []}}
        (game_root / f"{game['family']}-{game['game_id']}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_activity_eda_derives_sessions_coactivity_and_causal_candidates(tmp_path: Path) -> None:
    origin = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    reporter_root = tmp_path / "reporter"
    reporter_root.mkdir()
    reporter_database = reporter_root / "reporter.sqlite3"
    history_database = tmp_path / "history.sqlite3"
    archive_root = tmp_path / "runs"
    output = tmp_path / "output"
    _reporter_database(reporter_database, origin)
    games = _history_database(history_database, origin)
    _archives(archive_root, games)

    result = GleeActivityEDA(reporter_database=reporter_database, history_database=history_database, game_archive_root=archive_root, output_dir=output, session_gap_s=300.0).run()

    assert result["frontier_sequence"] == 5
    assert result["alignment"]["aligned_games"] == 3
    assert result["attribution"]["ki_total"] == 2
    assert result["attribution"]["hidden_games"] == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    self_bargaining = next(row for row in summary["self_activity"] if row["family"] == "bargaining")
    assert self_bargaining["session_count"] == 2
    assert summary["clean_ki_rating_pairs"]["opposite_sign_fraction"] == 1.0
    attribution = [json.loads(line) for line in (output / "game-attribution.jsonl").read_text(encoding="utf-8").splitlines()]
    hidden = next(row for row in attribution if row["game_id"] == "g-hidden")
    assert hidden["top_candidates"][0]["player_name"] == "Alice"
    with (output / "coactivity-edges.csv").open(encoding="utf-8", newline="") as stream:
        edges = list(csv.DictReader(stream))
    self_alice = next(row for row in edges if {row["player_1_id"], row["player_2_id"]} == {"self-id", "alice-id"} and row["family"] == "bargaining")
    assert int(self_alice["co_frontiers"]) == 3
    assert float(self_alice["opposite_sign_fraction"]) == 1.0
    assert (output / "manifest.json").is_file()


def test_activity_eda_refuses_to_overwrite_results(tmp_path: Path) -> None:
    origin = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    reporter_root = tmp_path / "reporter"
    reporter_root.mkdir()
    reporter_database = reporter_root / "reporter.sqlite3"
    history_database = tmp_path / "history.sqlite3"
    archive_root = tmp_path / "runs"
    output = tmp_path / "output"
    _reporter_database(reporter_database, origin)
    games = _history_database(history_database, origin)
    _archives(archive_root, games)
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        GleeActivityEDA(reporter_database=reporter_database, history_database=history_database, game_archive_root=archive_root, output_dir=output).run()

    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_rating_evidence_model_prefers_learned_sign_relation() -> None:
    model = RatingEvidenceModel()
    model.update(self_delta=2.0, positive_delta=-3.0, negative_deltas=[1.0, 2.0])
    model.update(self_delta=-2.0, positive_delta=3.0, negative_deltas=[-1.0, -4.0])

    assert model.score(self_delta=-1.0, opponent_delta=2.0) > model.score(self_delta=-1.0, opponent_delta=-2.0)
