"""Build one frozen Polars/Parquet analytics lake from GLEE operational receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .glee_activity_eda import GLEE_FAMILIES, _file_digest, _game_id_from_archive, _read_only_database


GLEE_ANALYTICS_LAKE_CONTRACT = "glee-polars-analytics-lake-v1"
DEFAULT_WINDOWS_S = (30, 60, 120, 300, 600, 1200, 1800)
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 7
PARQUET_ROW_GROUP_SIZE = 131072


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_parquet(frame: pl.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL, statistics=True, row_group_size=PARQUET_ROW_GROUP_SIZE)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    reopened_rows = int(pl.scan_parquet(path).select(pl.len()).collect().item(0, 0))
    reopened_schema = pl.read_parquet_schema(path)
    if reopened_rows != frame.height or list(reopened_schema) != frame.columns:
        raise RuntimeError(f"Parquet reopen validation failed for {path}: rows={reopened_rows}/{frame.height}, columns={list(reopened_schema)}/{frame.columns}")
    return {"rows": frame.height, "columns": frame.width, "bytes": path.stat().st_size, "sha256": _file_digest(path), "reopen_validated": True}


def _normalize_name_expr(column: str) -> pl.Expr:
    return pl.col(column).str.replace_all(r"\s+", " ").str.strip_chars().str.to_lowercase()


def _parse_utc_expr(column: str) -> pl.Expr:
    return pl.col(column).str.to_datetime(time_zone="UTC", strict=False)


def _fleet_frames(path: Path) -> tuple[pl.DataFrame, dict[str, str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        raise ValueError(f"fleet group file has no groups list: {path}")
    rows: list[dict[str, object]] = []
    fleet_for_id: dict[str, str] = {}
    confidence_for_key: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("key"), str) or not isinstance(group.get("members"), dict):
            raise ValueError("malformed fleet group")
        key = str(group["key"])
        confidence = str(group.get("confidence") or "unknown")
        confidence_for_key[key] = confidence
        for member_name, player_id in group["members"].items():
            member_id = str(player_id)
            if member_id in fleet_for_id and fleet_for_id[member_id] != key:
                raise ValueError(f"player {member_id} appears in multiple fleets")
            fleet_for_id[member_id] = key
            rows.append({"fleet_key": key, "confidence": confidence, "member_name": str(member_name), "member_name_normalized": " ".join(str(member_name).split()).strip().casefold(), "player_id": member_id})
    fleet_schema = {"fleet_key": pl.String, "confidence": pl.String, "member_name": pl.String, "member_name_normalized": pl.String, "player_id": pl.String}
    frame = pl.DataFrame(rows, schema=fleet_schema).sort(["fleet_key", "member_name_normalized"])
    return frame, fleet_for_id, confidence_for_key


def _manifest_engine(run_dir: Path, family: str, game_id: str) -> tuple[str, str | None, str | None, int]:
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            manifest = {}
    assignment_revision: str | None = None
    assignment_path = run_dir / f"{family}-live-policy-assignments.jsonl"
    if assignment_path.is_file():
        try:
            for line in assignment_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                assignment = json.loads(line)
                if isinstance(assignment, dict) and assignment.get("game_id") == game_id and isinstance(assignment.get("revision"), str):
                    assignment_revision = str(assignment["revision"])
        except (OSError, json.JSONDecodeError):
            assignment_revision = None
    advisor = manifest.get(f"{family}_advisor")
    advisor = advisor if isinstance(advisor, dict) else {}
    advisor_version = advisor.get("model_version") if isinstance(advisor.get("model_version"), str) else None
    base_version = advisor.get("base_engine_version") if isinstance(advisor.get("base_engine_version"), str) else None
    live_policy = manifest.get(f"{family}_live_policy")
    live_policy = live_policy if isinstance(live_policy, dict) else {}
    policy_revision = assignment_revision or (str(live_policy["current_revision_at_startup"]) if isinstance(live_policy.get("current_revision_at_startup"), str) else None)
    if advisor_version or policy_revision:
        engine = "|".join([advisor_version or base_version or "advisor-unknown", policy_revision or "policy-none"])
        return engine, advisor_version or base_version, policy_revision, 3 + int(assignment_revision is not None)
    run_label = re.sub(r"-\d{8}T\d{6}Z$", "", run_dir.name)
    mode = str(manifest.get("mode") or "engine-unknown")
    model = str(manifest.get("model") or "model-unknown")
    effort = str(manifest.get("effort") or "effort-unknown")
    return f"{run_label}|{mode}|{model}|{effort}", None, None, int(bool(manifest))


def _game_role(payload: Mapping[str, object], family: str) -> str:
    player = str(payload.get("your_player") or "unknown")
    state = payload.get("game_state") if isinstance(payload.get("game_state"), Mapping) else {}
    if family == "bargaining":
        return player
    role = state.get(f"{player}_role")
    return str(role) if isinstance(role, str) and role else player


def _archive_identity(payload: Mapping[str, object]) -> tuple[str, str | None]:
    opponent = payload.get("opponent") if isinstance(payload.get("opponent"), Mapping) else {}
    opponent_type = str(opponent.get("type") or "").casefold()
    raw_name = opponent.get("name")
    name = " ".join(raw_name.split()).strip() if isinstance(raw_name, str) and raw_name.strip() else None
    return ("hidden", None) if opponent_type == "hidden" or name is None else ("known", name)


def _archive_metadata(game_archive_root: Path, wanted: set[str]) -> tuple[pl.DataFrame, dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    malformed = 0
    scanned = 0
    for path in sorted(game_archive_root.glob("*/games/*.json")):
        family, game_id = _game_id_from_archive(path)
        if family is None or game_id not in wanted:
            continue
        scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        if not isinstance(payload, dict) or payload.get("game_id") != game_id or payload.get("game_family") != family:
            malformed += 1
            continue
        engine, advisor, policy, specificity = _manifest_engine(path.parent.parent, family, game_id)
        identity_scope, opponent_name = _archive_identity(payload)
        candidates[game_id].append(
            {
                "game_id": game_id,
                "family_archive": family,
                "identity_scope": identity_scope,
                "opponent_name": opponent_name,
                "role": _game_role(payload, family),
                "engine_version": engine,
                "advisor_version": advisor,
                "policy_revision": policy,
                "archive_path": str(path.relative_to(game_archive_root)),
                "archive_sha256": _digest(payload),
                "archive_specificity": specificity,
            }
        )
    selected: list[dict[str, object]] = []
    conflicts = 0
    duplicates = 0
    for game_id, options in sorted(candidates.items()):
        if len({str(option["archive_sha256"]) for option in options}) != 1:
            conflicts += 1
            continue
        duplicates += max(0, len(options) - 1)
        selected.append(max(options, key=lambda option: (int(option["archive_specificity"]), str(option["engine_version"]), str(option["archive_path"]))))
    schema = {
        "game_id": pl.String,
        "family_archive": pl.String,
        "identity_scope": pl.String,
        "opponent_name": pl.String,
        "role": pl.String,
        "engine_version": pl.String,
        "advisor_version": pl.String,
        "policy_revision": pl.String,
        "archive_path": pl.String,
        "archive_sha256": pl.String,
        "archive_specificity": pl.Int64,
    }
    frame = pl.DataFrame(selected, schema=schema) if selected else pl.DataFrame(schema=schema)
    return frame, {"archive_files_scanned": scanned, "selected_archives": frame.height, "missing_archives": len(wanted) - frame.height - conflicts, "conflicting_archives": conflicts, "duplicate_archive_files": duplicates, "malformed_archive_files": malformed}


def _history_frame(connection: Any, *, first_at: str, completed_by: str) -> pl.DataFrame:
    query = """
        SELECT game_id, game_family AS family, started_at, completed_at, rating_delta, revision AS history_revision, record_sha256 AS history_record_sha256
        FROM games
        WHERE started_at IS NOT NULL
          AND completed_at IS NOT NULL
          AND rating_delta IS NOT NULL
          AND started_at >= ?
          AND completed_at <= ?
        ORDER BY completed_at, game_id
    """
    frame = pl.read_database(query, connection, execute_options={"parameters": (first_at, completed_by)}, infer_schema_length=None)
    return frame.with_columns(_parse_utc_expr("started_at"), _parse_utc_expr("completed_at"), pl.col("rating_delta").cast(pl.Float64), pl.col("history_revision").cast(pl.Int64)).filter(pl.col("family").is_in(list(GLEE_FAMILIES)))


def _public_updates_frame(connection: Any, *, frontier: int) -> pl.DataFrame:
    query = """
        SELECT ch.frontier_sequence, ch.change_sequence, ch.family, ch.player_id, ch.change_kind,
               ch.observed_by AS observed_at,
               CAST(json_extract(rv.row_json, '$.player_name') AS TEXT) AS player_name_raw,
               CAST(json_extract(rv.row_json, '$.rating') AS REAL) AS rating_raw,
               CAST(json_extract(rv.row_json, '$.games_played') AS INTEGER) AS games_raw,
               CAST(json_extract(rv.row_json, '$.is_baseline') AS INTEGER) AS baseline_raw,
               CAST(json_extract(rv.row_json, '$.is_benchmark') AS INTEGER) AS benchmark_raw
        FROM changes AS ch
        JOIN row_versions AS rv
          ON rv.sequence = ch.frontier_sequence
         AND rv.family = ch.family
         AND rv.player_id = ch.player_id
        WHERE ch.frontier_sequence <= ?
          AND (
              ch.change_kind != 'changed'
              OR instr(ch.changed_fields_json, 'games_played') > 0
              OR instr(ch.changed_fields_json, 'rating') > 0
              OR instr(ch.changed_fields_json, 'player_name') > 0
          )
        ORDER BY ch.frontier_sequence, ch.family, ch.player_id, ch.change_sequence
    """
    frame = pl.read_database(query, connection, execute_options={"parameters": (frontier,)}, infer_schema_length=None)
    group = ["family", "player_id"]
    return (
        frame.with_columns(
            _parse_utc_expr("observed_at"),
            pl.col("player_name_raw").forward_fill().over(group).alias("player_name"),
            pl.col("rating_raw").forward_fill().over(group).cast(pl.Float64).alias("rating"),
            pl.col("games_raw").forward_fill().over(group).cast(pl.Int64).alias("games"),
            pl.col("baseline_raw").forward_fill().over(group).fill_null(0).cast(pl.Boolean).alias("is_baseline"),
            pl.col("benchmark_raw").forward_fill().over(group).fill_null(0).cast(pl.Boolean).alias("is_benchmark"),
            (pl.col("change_kind") != "disappeared").alias("present"),
        )
        .with_columns(_normalize_name_expr("player_name").alias("player_name_normalized"))
        .drop(["player_name_raw", "rating_raw", "baseline_raw", "benchmark_raw"])
        .sort(["observed_at", "frontier_sequence", "family", "player_id", "change_sequence"])
    )


def _effective_activity(updates: pl.DataFrame, fleets: pl.DataFrame, self_name: str) -> pl.DataFrame:
    group = ["family", "player_id"]
    normalized_self = " ".join(self_name.split()).strip().casefold()
    return (
        updates.sort(["family", "player_id", "observed_at", "change_sequence"])
        .with_columns(pl.col("games_raw").cum_max().over(group).alias("raw_running_high_water"))
        .with_columns(pl.col("raw_running_high_water").forward_fill().over(group).alias("running_high_water"))
        .with_columns(pl.col("running_high_water").shift(1).over(group).alias("previous_high_water"))
        .filter(pl.col("games_raw").is_not_null() & pl.col("previous_high_water").is_not_null() & (pl.col("games_raw") > pl.col("previous_high_water")))
        .with_columns((pl.col("games_raw") - pl.col("previous_high_water")).cast(pl.Int64).alias("games_delta"))
        .filter(~pl.col("is_baseline") & ~pl.col("is_benchmark") & (pl.col("player_name_normalized").fill_null("") != normalized_self))
        .join(fleets.select(["player_id", "fleet_key", "confidence"]), on="player_id", how="left")
        .select(pl.col("observed_at").alias("event_at"), "frontier_sequence", "change_sequence", "family", "player_id", "player_name", "games_delta", "fleet_key", pl.col("confidence").alias("fleet_confidence"))
        .sort(["event_at", "family", "player_id"])
    )


def _activity_invariant(updates: pl.DataFrame, activity: pl.DataFrame, self_name: str) -> dict[str, object]:
    normalized_self = " ".join(self_name.split()).strip().casefold()
    states = (
        updates.filter(pl.col("games_raw").is_not_null())
        .sort(["family", "player_id", "observed_at", "change_sequence"])
        .group_by(["family", "player_id"])
        .agg(
            pl.col("games_raw").first().alias("baseline_games"),
            pl.col("games_raw").max().alias("high_water_games"),
            pl.col("player_name_normalized").drop_nulls().first().alias("player_name_normalized"),
            pl.col("is_baseline").first().alias("is_baseline"),
            pl.col("is_benchmark").first().alias("is_benchmark"),
        )
        .filter(~pl.col("is_baseline") & ~pl.col("is_benchmark") & (pl.col("player_name_normalized").fill_null("") != normalized_self))
        .with_columns((pl.col("high_water_games") - pl.col("baseline_games")).alias("expected_additions"))
    )
    expected = int(states["expected_additions"].sum() or 0)
    actual = int(activity["games_delta"].sum() or 0)
    receipt = {"eligible_player_family_states": states.height, "expected_additions_from_high_water_minus_baseline": expected, "materialized_effective_additions": actual, "difference": actual - expected, "valid": actual == expected}
    if not receipt["valid"]:
        raise RuntimeError(f"effective public activity violates high-water conservation: {receipt}")
    return receipt


def _public_context(games: pl.DataFrame, updates: pl.DataFrame, fleets: pl.DataFrame, self_name: str) -> pl.DataFrame:
    aliases = updates.filter(pl.col("player_name_normalized").is_not_null()).group_by(["family", "player_name_normalized"]).agg(pl.col("player_id").n_unique().alias("alias_id_count"))
    unique_updates = (
        updates.join(aliases.filter(pl.col("alias_id_count") == 1), on=["family", "player_name_normalized"], how="inner")
        .join(fleets.select(["player_id", "fleet_key"]), on="player_id", how="left")
        .select("family", pl.col("player_name_normalized").alias("opponent_name_normalized"), "observed_at", pl.col("player_id").alias("resolved_opponent_id"), pl.col("rating").alias("resolved_opponent_rating"), pl.col("present").alias("opponent_publicly_present"), pl.col("fleet_key").alias("direct_fleet_key"))
        .sort(["observed_at"])
    )
    with_names = games.with_columns(_normalize_name_expr("opponent_name").alias("opponent_name_normalized")).sort("started_at")
    resolved = with_names.join_asof(unique_updates, left_on="started_at", right_on="observed_at", by=["family", "opponent_name_normalized"], strategy="backward", check_sortedness=False)
    player_dimension = updates.select(["family", "player_id"]).unique()
    game_player_grid = games.select(["game_id", "family", "started_at"]).join(player_dimension, on="family", how="inner").sort("started_at")
    state_timeline = updates.select(["family", "player_id", "observed_at", "rating", "present", "is_baseline", "is_benchmark", "player_name_normalized"]).sort("observed_at")
    state_at_game = game_player_grid.join_asof(state_timeline, left_on="started_at", right_on="observed_at", by=["family", "player_id"], strategy="backward", check_sortedness=False)
    family_medians = (
        state_at_game.filter(pl.col("present").fill_null(False) & ~pl.col("is_baseline").fill_null(True) & ~pl.col("is_benchmark").fill_null(True) & pl.col("rating").is_not_null())
        .group_by("game_id")
        .agg(pl.col("rating").median().alias("family_rating_median"))
    )
    normalized_self = " ".join(self_name.split()).strip().casefold()
    self_ratings = (
        state_at_game.filter((pl.col("player_name_normalized") == normalized_self) & pl.col("rating").is_not_null())
        .group_by("game_id")
        .agg(pl.col("rating").last().alias("self_rating"))
    )
    return (
        resolved.join(family_medians, on="game_id", how="left")
        .join(self_ratings, on="game_id", how="left")
        .with_columns(
            pl.when((pl.col("identity_scope") == "known") & pl.col("resolved_opponent_rating").is_not_null()).then(pl.col("resolved_opponent_rating")).otherwise(pl.col("family_rating_median")).alias("opponent_rating"),
            ((pl.col("identity_scope") == "known") & pl.col("resolved_opponent_rating").is_not_null()).cast(pl.Int8).alias("opponent_rating_observed"),
            pl.col("self_rating").fill_null(pl.col("family_rating_median")),
            ((pl.col("identity_scope") == "known") & pl.col("direct_fleet_key").is_not_null()).cast(pl.Int8).alias("direct_named_fleet_opponent"),
        )
        .with_columns((1 - pl.col("opponent_rating_observed")).cast(pl.Int8).alias("opponent_rating_imputed"))
        .sort("started_at")
    )


def _window_features(games: pl.DataFrame, activity: pl.DataFrame, windows_s: Sequence[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    game_columns = games.select(["game_id", "family", "started_at"])
    aggregates: list[pl.DataFrame] = []
    exposures: list[pl.DataFrame] = []
    for window_s in windows_s:
        window_games = game_columns.with_columns((pl.col("started_at") - pl.duration(seconds=int(window_s))).alias("window_start"))
        joined = window_games.lazy().join_where(
            activity.lazy(),
            pl.col("event_at") <= pl.col("started_at"),
            pl.col("event_at") > pl.col("window_start"),
            suffix="_activity",
        )
        aggregate = (
            joined.group_by("game_id")
            .agg(
                pl.col("fleet_key").drop_nulls().n_unique().alias("fleet_count"),
                pl.when(pl.col("fleet_confidence").is_in(["high", "very-high"])).then(pl.col("fleet_key")).otherwise(None).drop_nulls().n_unique().alias("confident_fleet_count"),
                pl.when(pl.col("fleet_key").is_not_null()).then(pl.struct(["fleet_key", "player_id"])).otherwise(None).drop_nulls().n_unique().alias("fleet_member_count"),
                pl.when(pl.col("fleet_key").is_not_null()).then(pl.col("games_delta")).otherwise(0).sum().alias("fleet_additions"),
                pl.when(pl.col("fleet_key").is_null()).then(pl.col("games_delta")).otherwise(0).sum().alias("nonfleet_additions"),
                pl.col("games_delta").sum().alias("all_additions"),
                pl.col("fleet_key").drop_nulls().unique().sort().alias("fleet_keys"),
            )
            .with_columns(pl.lit(int(window_s)).alias("window_s"), (pl.col("fleet_count") > 0).cast(pl.Int8).alias("fleet_online_any"))
            .collect()
        )
        full = (
            game_columns.select(["game_id", "family"]).join(aggregate, on="game_id", how="left")
            .with_columns(
                pl.col("window_s").fill_null(int(window_s)),
                pl.col("fleet_count").fill_null(0),
                pl.col("confident_fleet_count").fill_null(0),
                pl.col("fleet_member_count").fill_null(0),
                pl.col("fleet_additions").fill_null(0),
                pl.col("nonfleet_additions").fill_null(0),
                pl.col("all_additions").fill_null(0),
                pl.col("fleet_online_any").fill_null(0),
                pl.col("fleet_keys").fill_null(pl.lit([]).cast(pl.List(pl.String))),
            )
        )
        aggregates.append(full)
        exposure = (
            joined.filter(pl.col("fleet_key").is_not_null())
            .group_by(["game_id", "fleet_key"])
            .agg(pl.col("player_id").n_unique().alias("active_member_count"), pl.col("games_delta").sum().alias("fleet_game_additions"), pl.col("event_at").max().alias("last_event_at"))
            .with_columns(pl.lit(int(window_s)).alias("window_s"))
            .collect()
        )
        exposures.append(exposure)
    return pl.concat(aggregates, how="diagonal_relaxed").sort(["game_id", "window_s"]), pl.concat(exposures, how="diagonal_relaxed").sort(["game_id", "window_s", "fleet_key"])


class GleeAnalyticsLakeBuilder:
    """Freeze operational sources into compact typed Parquet facts and reusable Polars joins."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        history_database: Path,
        game_archive_root: Path,
        fleet_groups: Path,
        output_dir: Path,
        reporter_frontier: int | None = None,
        self_name: str = "DeepRMM-01",
        windows_s: Sequence[int] = DEFAULT_WINDOWS_S,
    ) -> None:
        windows = tuple(sorted(set(int(value) for value in windows_s)))
        if not windows or min(windows) <= 0:
            raise ValueError("analytics windows must be positive")
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.fleet_groups = fleet_groups.resolve()
        self.output_dir = output_dir.resolve()
        self.requested_frontier = reporter_frontier
        self.self_name = self_name
        self.windows_s = windows

    def _snapshot(self, connection: Any) -> dict[str, object]:
        maximum = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM frontiers").fetchone()[0])
        frontier = maximum if self.requested_frontier is None else int(self.requested_frontier)
        if frontier < 1 or frontier > maximum:
            raise ValueError(f"reporter frontier must be between 1 and {maximum}: {frontier}")
        row = connection.execute("SELECT sequence, frontier_id, started_at, completed_at FROM frontiers WHERE sequence = ?", (frontier,)).fetchone()
        first = connection.execute("SELECT MIN(started_at) FROM frontiers WHERE sequence <= ?", (frontier,)).fetchone()[0]
        if row is None or first is None:
            raise RuntimeError(f"incomplete reporter frontier: {frontier}")
        return {"frontier_sequence": frontier, "frontier_id": str(row["frontier_id"]), "first_started_at": str(first), "frontier_started_at": str(row["started_at"]), "frontier_completed_at": str(row["completed_at"])}

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"analytics lake output directory already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging_dir.mkdir(mode=0o700)
        _fsync_directory(staging_dir.parent)
        fleet_frame, _fleet_for_id, _confidence = _fleet_frames(self.fleet_groups)
        reporter = _read_only_database(self.reporter_database)
        reporter.execute("BEGIN")
        history = _read_only_database(self.history_database)
        history.execute("BEGIN")
        try:
            reporter_journal_mode = str(reporter.execute("PRAGMA journal_mode").fetchone()[0])
            history_journal_mode = str(history.execute("PRAGMA journal_mode").fetchone()[0])
            snapshot = self._snapshot(reporter)
            games = _history_frame(history, first_at=str(snapshot["first_started_at"]), completed_by=str(snapshot["frontier_completed_at"]))
            archives, archive_inventory = _archive_metadata(self.game_archive_root, set(games["game_id"].to_list()))
            updates = _public_updates_frame(reporter, frontier=int(snapshot["frontier_sequence"]))
            reporter.execute("ROLLBACK")
            history.execute("ROLLBACK")
        finally:
            reporter.close()
            history.close()
        games = games.join(archives, on="game_id", how="left").with_columns(pl.col("identity_scope").fill_null("archive-missing"), pl.col("role").fill_null("unknown"), pl.col("engine_version").fill_null("engine-unknown"))
        activity = _effective_activity(updates, fleet_frame, self.self_name)
        activity_invariant = _activity_invariant(updates, activity, self.self_name)
        games = _public_context(games, updates, fleet_frame, self.self_name)
        windows, exposures = _window_features(games, activity, self.windows_s)
        artifact_frames = {
            "games.parquet": games,
            "public-updates.parquet": updates,
            "public-activity.parquet": activity,
            "fleet-members.parquet": fleet_frame,
            "game-windows.parquet": windows,
            "game-fleet-exposures.parquet": exposures,
        }
        artifacts = {name: _write_parquet(frame, staging_dir / name) for name, frame in artifact_frames.items()}
        manifest = {
            "schema_version": 1,
            "contract": GLEE_ANALYTICS_LAKE_CONTRACT,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "source_frontier": {
                **snapshot,
                "reporter_database": str(self.reporter_database),
                "history_database": str(self.history_database),
                "game_archive_root": str(self.game_archive_root),
                "fleet_groups": str(self.fleet_groups),
                "fleet_groups_sha256": _file_digest(self.fleet_groups),
            },
            "source_consistency": {"reporter_sqlite_journal_mode": reporter_journal_mode, "history_sqlite_journal_mode": history_journal_mode, "read_transactions": "explicit read-only snapshots held through extraction"},
            "storage": {"format": "Apache Parquet", "compression": f"Zstandard level {PARQUET_COMPRESSION_LEVEL}", "statistics": True, "row_group_size": PARQUET_ROW_GROUP_SIZE, "sort_keys": {"games.parquet": ["started_at"], "public-updates.parquet": ["observed_at", "frontier_sequence"], "public-activity.parquet": ["event_at"], "game-windows.parquet": ["game_id", "window_s"], "game-fleet-exposures.parquet": ["game_id", "window_s", "fleet_key"]}, "commit_protocol": "write hidden sibling staging directory; fsync and reopen-validate every Parquet artifact; hash all artifacts; write manifest last; fsync staging directory; atomically rename directory; fsync parent directory"},
            "parameters": {"self_name": self.self_name, "windows_s": list(self.windows_s), "strictly_prior_activity": True},
            "inventory": {
                **archive_inventory,
                "history_games": games.height,
                "public_updates": updates.height,
                "effective_public_activity_events": activity.height,
                "effective_public_game_additions": int(activity["games_delta"].sum() or 0),
                "fleet_members": fleet_frame.height,
                "candidate_fleets": fleet_frame["fleet_key"].n_unique(),
                "game_window_rows": windows.height,
                "game_fleet_exposure_rows": exposures.height,
                "games_by_family": dict(sorted(Counter(games["family"].to_list()).items())),
                "games_by_identity": dict(sorted(Counter(games["identity_scope"].to_list()).items())),
            },
            "invariants": {"effective_activity_high_water_conservation": activity_invariant},
            "artifacts": artifacts,
            "implementation_sha256": _file_digest(Path(__file__)),
        }
        _atomic_json(staging_dir / "manifest.json", manifest)
        committed_manifest_sha256 = _file_digest(staging_dir / "manifest.json")
        for name, receipt in artifacts.items():
            path = staging_dir / name
            if path.stat().st_size != receipt["bytes"] or _file_digest(path) != receipt["sha256"]:
                raise RuntimeError(f"staged artifact changed before commit: {path}")
        _fsync_directory(staging_dir)
        os.replace(staging_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        if _file_digest(self.output_dir / "manifest.json") != committed_manifest_sha256:
            raise RuntimeError(f"committed analytics-lake manifest changed during publication: {self.output_dir}")
        return {"contract": GLEE_ANALYTICS_LAKE_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": snapshot["frontier_sequence"], "games": games.height, "activity_events": activity.height, "exposure_rows": exposures.height, "manifest_sha256": committed_manifest_sha256, "commit_protocol": "fsync-validated-staging-plus-atomic-directory-rename-v1"}
