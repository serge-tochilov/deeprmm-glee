import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nommd_arena.glee_effective_events import derive_effective_events
from nommd_arena.glee_identity_registry import GleeIdentityRegistryAnalysis, TemporalIdentityRegistry


def _stamp(origin: datetime, seconds: int) -> str:
    return (origin + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _row(name: str, games: int, rating: float, *, baseline: bool = False, benchmark: bool = False) -> str:
    return json.dumps({"player_name": name, "games_played": games, "rating": rating, "is_baseline": baseline, "is_benchmark": benchmark, "is_owner_best": True}, sort_keys=True)


def _mapping(sequence: int, family: str, player_id: str, row_json: str | None, *, kind: str = "changed") -> dict[str, object]:
    return {"frontier_sequence": sequence, "family": family, "player_id": player_id, "row_json": row_json, "change_kind": kind, "row_sha256": f"row-{sequence}-{family}-{player_id}" if row_json is not None else None}


def test_temporal_registry_preserves_rename_disappearance_and_collision_sets() -> None:
    registry = TemporalIdentityRegistry(frontier_sequence=4)
    registry.consume(_mapping(1, "bargaining", "id-a", _row("RESERVE", 1, 1500.0), kind="appeared"))
    registry.consume(_mapping(1, "bargaining", "id-b", _row("Other", 1, 1500.0), kind="appeared"))
    registry.consume(_mapping(2, "bargaining", "id-b", _row("reserve", 2, 1501.0)))
    registry.consume(_mapping(3, "bargaining", "id-a", None, kind="disappeared"))
    registry.consume(_mapping(4, "bargaining", "id-a", _row("New Reserve", 2, 1502.0), kind="appeared"))
    registry.finish()

    assert registry.ids_for_label("bargaining", " reserve ", 1) == ["id-a"]
    assert registry.ids_for_label("bargaining", "RESERVE", 2) == ["id-a", "id-b"]
    assert registry.ids_for_label("bargaining", "reserve", 3) == ["id-b"]
    assert registry.ids_for_label("bargaining", "new reserve", 4) == ["id-a"]
    assert registry.ever_ids_for_label("bargaining", "reserve") == ["id-a", "id-b"]
    collision = registry.collision_rows()
    assert collision == [
        {
            "contract": "glee-public-identity-registry-v1",
            "schema_version": 1,
            "family": "bargaining",
            "normalized_label": "reserve",
            "display_labels": ["RESERVE", "reserve"],
            "player_ids": ["id-a", "id-b"],
            "start_sequence": 2,
            "end_sequence_exclusive": 3,
        }
    ]


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    origin = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    reporter = root / "reporter.sqlite3"
    connection = sqlite3.connect(reporter)
    connection.executescript(
        """
        CREATE TABLE frontiers (sequence INTEGER PRIMARY KEY, frontier_id TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL);
        CREATE TABLE family_polls (sequence INTEGER NOT NULL, family TEXT NOT NULL, status TEXT NOT NULL, truncated INTEGER NOT NULL);
        CREATE TABLE row_versions (family TEXT NOT NULL, player_id TEXT NOT NULL, sequence INTEGER NOT NULL, change_kind TEXT NOT NULL, row_sha256 TEXT, row_json TEXT, PRIMARY KEY (family, player_id, sequence));
        CREATE TABLE changes (change_sequence INTEGER PRIMARY KEY, frontier_sequence INTEGER NOT NULL, family TEXT NOT NULL, player_id TEXT NOT NULL, change_kind TEXT NOT NULL, observed_after TEXT, observed_by TEXT NOT NULL, games_delta INTEGER, rating_delta REAL);
        """
    )
    for sequence, seconds in enumerate((0, 10, 20, 30), start=1):
        connection.execute("INSERT INTO frontiers VALUES (?, ?, ?, ?)", (sequence, f"f-{sequence}", _stamp(origin, seconds), _stamp(origin, seconds + 1)))
        connection.execute("INSERT INTO family_polls VALUES (?, 'bargaining', 'ok', 0)", (sequence,))
    rows = [
        (1, "self", _row("DeepRMM-01", 10, 1800.0), "appeared", None, None),
        (1, "id-a", _row("RESERVE", 5, 1700.0), "appeared", None, None),
        (1, "id-b", _row("Other", 5, 1700.0), "appeared", None, None),
        (2, "self", _row("DeepRMM-01", 11, 1801.0), "changed", 1, 1.0),
        (2, "id-a", _row("RESERVE", 6, 1699.0), "changed", 1, -1.0),
        (3, "id-b", _row("RESERVE", 5, 1700.0), "changed", 0, 0.0),
        (4, "self", _row("DeepRMM-01", 12, 1802.0), "changed", 1, 1.0),
        (4, "id-b", _row("RESERVE", 6, 1702.0), "changed", 1, 2.0),
    ]
    for change_sequence, (sequence, player_id, payload, kind, games_delta, rating_delta) in enumerate(rows, start=1):
        connection.execute("INSERT INTO row_versions VALUES ('bargaining', ?, ?, ?, ?, ?)", (player_id, sequence, kind, f"row-{change_sequence}", payload))
        connection.execute("INSERT INTO changes VALUES (?, ?, 'bargaining', ?, ?, ?, ?, ?, ?)", (change_sequence, sequence, player_id, kind, _stamp(origin, (sequence - 1) * 10), _stamp(origin, (sequence - 1) * 10 + 1), games_delta, rating_delta))
    connection.commit()
    connection.row_factory = sqlite3.Row
    effective = derive_effective_events(connection, frontier_sequence=4)
    connection.close()
    activity_summary = root / "activity-summary.json"
    activity_summary.write_text(
        json.dumps(
            {
                "contract": "glee-activity-eda-v2",
                "source_frontier": {"frontier_sequence": 4, "self_player_ids": {"bargaining": "self"}},
                "effective_event_reconstruction": {"source_rows_sha256": effective["source_rows_sha256"]},
            }
        ),
        encoding="utf-8",
    )
    attribution = root / "attribution.jsonl"
    games = [
        {"contract": "glee-activity-eda-v2", "game_id": "g-unique", "family": "bargaining", "started_at": _stamp(origin, 5), "completed_at": _stamp(origin, 15), "identity_scope": "known", "opponent_name": "RESERVE", "public_frontier_sequence": 2, "archive_path": "a.json", "archive_sha256": "a"},
        {"contract": "glee-activity-eda-v2", "game_id": "g-collision", "family": "bargaining", "started_at": _stamp(origin, 25), "completed_at": _stamp(origin, 35), "identity_scope": "known", "opponent_name": "reserve", "public_frontier_sequence": 4, "archive_path": "b.json", "archive_sha256": "b"},
    ]
    attribution.write_text("".join(json.dumps(game, sort_keys=True) + "\n" for game in games), encoding="utf-8")
    dossier_index = root / "dossier-index.json"
    dossier_index.write_text(json.dumps({"opponents": {"name-hash": {"name": "RESERVE", "aliases": ["reserve"]}}}), encoding="utf-8")
    return reporter, activity_summary, attribution, dossier_index


def test_analysis_emits_causal_set_valued_envelopes_and_is_deterministic(tmp_path: Path) -> None:
    reporter, activity_summary, attribution, dossier_index = _fixture(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = GleeIdentityRegistryAnalysis(reporter_database=reporter, activity_summary=activity_summary, attribution_path=attribution, output_dir=first_output, dossier_index=dossier_index).run()
    second = GleeIdentityRegistryAnalysis(reporter_database=reporter, activity_summary=activity_summary, attribution_path=attribution, output_dir=second_output, dossier_index=dossier_index).run()

    assert first["manifest_sha256"] == second["manifest_sha256"]
    envelopes = [json.loads(line) for line in (first_output / "game-identity-envelopes.jsonl").read_text(encoding="utf-8").splitlines()]
    unique, collision = envelopes
    assert unique["resolution_status"] == "unique-at-assignment-frontier"
    assert unique["exact_public_player_id"] == "id-a"
    assert unique["joint_activity_label_ids"] == ["id-a"]
    assert unique["activity_frontier_ref"] == {"family": "bargaining", "frontier_sequence": 2}
    assert collision["resolution_status"] == "collision-set-at-assignment-frontier"
    assert collision["exact_public_player_id"] is None
    assert collision["assignment_label_ids"] == ["id-a", "id-b"]
    assert collision["joint_activity_label_ids"] == ["id-b"]
    assert collision["unknown_mass"] is None
    frontiers = [json.loads(line) for line in (first_output / "activity-frontiers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert frontiers[1]["candidate_capacity_by_id"] == {"id-b": 1}
    audit = json.loads((first_output / "collision-audit.json").read_text(encoding="utf-8"))
    assert audit["profiles"][0]["status"] == "current-collision"
    profiles = [json.loads(line) for line in (first_output / "profile-inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {profile["name"] for profile in profiles} == {"RESERVE", "reserve"}
    assert {profile["source_kind"] for profile in profiles} == {"dossier-index"}
    assert first["effective_event_receipt"]["invariant_violations"] == 0
    registry_rows = [json.loads(line) for line in (first_output / "identity-registry.jsonl").read_text(encoding="utf-8").splitlines()]
    self_row = next(row for row in registry_rows if row["player_id"] == "self")
    assert self_row["last_observed_sequence"] == 4
