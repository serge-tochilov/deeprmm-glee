"""Continuously publish crash-safe Parquet views of authoritative live GLEE data."""

from __future__ import annotations

import fcntl
import glob
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import polars as pl

from .glee_activity_eda import GLEE_FAMILIES, _file_digest, _read_only_database
from .glee_analytics_incremental import PUBLIC_ACTIVITY_ARTIFACT, PUBLIC_STATE_ARTIFACT, PUBLIC_UPDATES_ARTIFACT, ONLINE_PUBLIC_ACTIVITY_SCHEMA, ONLINE_PUBLIC_STATE_SCHEMA, ONLINE_PUBLIC_UPDATE_SCHEMA, derive_public_increment, empty_frame, normalize_frame, reporter_change_rows
from .glee_analytics_lake import DEFAULT_WINDOWS_S, GLEE_ANALYTICS_LAKE_CONTRACT, _archive_identity, _archive_metadata, _atomic_json, _digest, _fleet_frames, _fsync_directory, _game_role, _history_frame, _manifest_engine, _parse_utc_expr, _write_parquet


ONLINE_ANALYTICS_LAKE_CONTRACT = "glee-online-polars-analytics-lake-v1"
ONLINE_ANALYTICS_RELEASE_CONTRACT = "glee-online-terminal-game-lake-v1"
ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT = "glee-online-incremental-game-lake-v2"
ONLINE_INCREMENTAL_RELEASE_CONTRACT = "glee-online-incremental-analytics-lake-v3"
ONLINE_ANALYTICS_POINTER_CONTRACT = "glee-online-polars-analytics-lake-pointer-v1"
ONLINE_ANALYTICS_STATUS_CONTRACT = "glee-online-polars-analytics-lake-status-v1"
GAMES_ARTIFACT = "games.parquet"
ONLINE_REQUIRED_ARTIFACTS = (GAMES_ARTIFACT, PUBLIC_UPDATES_ARTIFACT, PUBLIC_ACTIVITY_ARTIFACT)
ONLINE_INTERNAL_ARTIFACTS = (PUBLIC_STATE_ARTIFACT,)
ONLINE_INCREMENTAL_PROFILE = "all-games-and-public-activity-incremental"
ONLINE_GAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String,
    "family": pl.String,
    "started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "completed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "rating_delta": pl.Float64,
    "history_revision": pl.Int64,
    "history_record_sha256": pl.String,
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
ONLINE_GAME_COLUMNS = tuple(ONLINE_GAME_SCHEMA)
ONLINE_HISTORY_COLUMNS = ("game_id", "family", "started_at", "completed_at", "rating_delta", "history_revision", "history_record_sha256")
ONLINE_ARCHIVE_COLUMNS = tuple(column for column in ONLINE_GAME_COLUMNS if column not in ONLINE_HISTORY_COLUMNS)
ONLINE_DATASET_SCHEMAS: dict[str, Mapping[str, pl.DataType]] = {
    GAMES_ARTIFACT: ONLINE_GAME_SCHEMA,
    PUBLIC_UPDATES_ARTIFACT: ONLINE_PUBLIC_UPDATE_SCHEMA,
    PUBLIC_ACTIVITY_ARTIFACT: ONLINE_PUBLIC_ACTIVITY_SCHEMA,
    PUBLIC_STATE_ARTIFACT: ONLINE_PUBLIC_STATE_SCHEMA,
}
SEGMENT_PATH_PATTERN = re.compile(r"segments/[0-9a-f]{64}\.parquet")
LEGACY_FULL_ARTIFACTS = (
    "games.parquet",
    "public-updates.parquet",
    "public-activity.parquet",
    "fleet-members.parquet",
    "game-windows.parquet",
    "game-fleet-exposures.parquet",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_time(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _schema_sha256(schema: Mapping[str, object]) -> str:
    return _digest({str(name): str(dtype) for name, dtype in schema.items()})


def _empty_games() -> pl.DataFrame:
    return pl.DataFrame(schema=ONLINE_GAME_SCHEMA)


def _normalize_games(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in ONLINE_GAME_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in ONLINE_GAME_COLUMNS]
    if missing or extra:
        raise RuntimeError(f"online game segment schema differs: missing={missing}, extra={extra}")
    return frame.select(pl.col(column).cast(dtype) for column, dtype in ONLINE_GAME_SCHEMA.items())


def _logical_games(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [_normalize_games(frame) for frame in frames if frame.height]
    if not nonempty:
        return _empty_games()
    stacked = pl.concat([frame.with_columns(pl.lit(index).alias("_segment_ordinal")) for index, frame in enumerate(nonempty)], how="vertical")
    conflicts = (
        stacked.group_by(["game_id", "history_revision"])
        .agg(pl.col("history_record_sha256").n_unique().alias("hashes"))
        .filter(pl.col("hashes") > 1)
    )
    if conflicts.height:
        raise RuntimeError(f"incremental game lake contains conflicting history hashes for {conflicts.height} game revisions")
    return (
        stacked.sort(["game_id", "history_revision", "_segment_ordinal"])
        .unique(subset=["game_id"], keep="last", maintain_order=True)
        .drop("_segment_ordinal")
        .sort(["completed_at", "game_id"])
    )


def _history_rows(connection: Any, *, first_at: str | None = None, game_ids: Sequence[str] | None = None) -> pl.DataFrame:
    parameters: list[object] = []
    lower_bound = ""
    if first_at is not None:
        lower_bound = " AND started_at >= ?"
        parameters.append(first_at)
    game_filter = ""
    if game_ids is not None:
        if not game_ids:
            return pl.DataFrame(schema={column: ONLINE_GAME_SCHEMA[column] for column in ONLINE_HISTORY_COLUMNS})
        placeholders = ",".join("?" for _game_id in game_ids)
        game_filter = f" AND game_id IN ({placeholders})"
        parameters.extend(game_ids)
    query = f"""
        SELECT game_id, game_family AS family, started_at, completed_at, rating_delta, revision AS history_revision, record_sha256 AS history_record_sha256
        FROM games
        WHERE started_at IS NOT NULL
          AND completed_at IS NOT NULL
          AND rating_delta IS NOT NULL
          {lower_bound}
          {game_filter}
        ORDER BY completed_at, game_id
    """
    history_schema = {"game_id": pl.String, "family": pl.String, "started_at": pl.String, "completed_at": pl.String, "rating_delta": pl.Float64, "history_revision": pl.Int64, "history_record_sha256": pl.String}
    frame = pl.read_database(query, connection, execute_options={"parameters": tuple(parameters)}, schema_overrides=history_schema, infer_schema_length=None)
    return frame.with_columns(_parse_utc_expr("started_at"), _parse_utc_expr("completed_at"), pl.col("rating_delta").cast(pl.Float64), pl.col("history_revision").cast(pl.Int64)).filter(pl.col("family").is_in(list(GLEE_FAMILIES)))


def _targeted_archive_metadata(game_archive_root: Path, family_by_game: Mapping[str, str]) -> tuple[pl.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    scanned = 0
    malformed = 0
    conflicts = 0
    duplicates = 0
    for game_id, family in sorted(family_by_game.items()):
        if family not in GLEE_FAMILIES:
            continue
        filename = glob.escape(f"{family}-{game_id}.json")
        options: list[dict[str, object]] = []
        for path in sorted(game_archive_root.glob(f"*/games/{filename}")):
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
            options.append(
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
        if len({str(option["archive_sha256"]) for option in options}) > 1:
            conflicts += 1
            continue
        duplicates += max(0, len(options) - 1)
        if options:
            rows.append(max(options, key=lambda option: (int(option["archive_specificity"]), str(option["engine_version"]), str(option["archive_path"]))))
    schema = {"game_id": pl.String, **{column: ONLINE_GAME_SCHEMA[column] for column in ONLINE_ARCHIVE_COLUMNS}}
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return frame, {"archive_files_scanned": scanned, "selected_archives": frame.height, "missing_archives": len(family_by_game) - frame.height - conflicts, "conflicting_archives": conflicts, "duplicate_archive_files": duplicates, "malformed_archive_files": malformed}


def _join_history_archives(history: pl.DataFrame, archives: pl.DataFrame) -> pl.DataFrame:
    return _normalize_games(
        history.join(archives, on="game_id", how="left").with_columns(
            pl.col("identity_scope").fill_null("archive-missing"),
            pl.col("role").fill_null("unknown"),
            pl.col("engine_version").fill_null("engine-unknown"),
        )
    )


class GleeOnlineAnalyticsLakeReader:
    """Select a verified current Parquet frontier while exposing SQLite/raw fallback explicitly."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.releases_root = self.root / "releases"
        self.segments_root = self.root / "segments"
        self.pointer_path = self.root / "current.json"

    def _release_path(self, release: str) -> Path:
        if Path(release).name != release or release in {"", ".", ".."}:
            raise RuntimeError("online analytics pointer contains an invalid release")
        path = (self.releases_root / release).resolve()
        if path.parent != self.releases_root.resolve():
            raise RuntimeError("online analytics release escapes its root")
        return path

    def _segment_path(self, receipt: Mapping[str, object]) -> Path:
        relative = receipt.get("path")
        if not isinstance(relative, str) or not SEGMENT_PATH_PATTERN.fullmatch(relative):
            raise RuntimeError("incremental release contains an invalid segment path")
        path = (self.root / relative).resolve()
        if path.parent != self.segments_root.resolve():
            raise RuntimeError("incremental segment escapes its root")
        return path

    def _validate_segment(self, receipt: Mapping[str, object], *, artifact: str, reopen: bool) -> Path:
        path = self._segment_path(receipt)
        sha256 = receipt.get("sha256")
        rows = receipt.get("rows")
        size = receipt.get("bytes")
        schema_sha256 = receipt.get("schema_sha256")
        if not isinstance(sha256, str) or path.name != f"{sha256}.parquet" or not isinstance(rows, int) or rows < 0 or not isinstance(size, int) or size < 1 or not isinstance(schema_sha256, str):
            raise RuntimeError("incremental release contains an incomplete segment receipt")
        if not path.is_file() or path.stat().st_size != size or _file_digest(path) != sha256:
            raise RuntimeError(f"incremental segment verification failed: {path.name}")
        if reopen:
            schema = pl.read_parquet_schema(path)
            reopened_rows = int(pl.scan_parquet(path).select(pl.len()).collect().item(0, 0))
            expected = ONLINE_DATASET_SCHEMAS.get(artifact)
            if expected is None:
                raise RuntimeError(f"incremental release names an unsupported dataset: {artifact}")
            if reopened_rows != rows or list(schema) != list(expected) or _schema_sha256(schema) != schema_sha256:
                raise RuntimeError(f"incremental segment reopen validation failed: {path.name}")
            if artifact in {PUBLIC_UPDATES_ARTIFACT, PUBLIC_ACTIVITY_ARTIFACT}:
                metrics = pl.scan_parquet(path).select(pl.col("change_sequence").min().alias("key_min"), pl.col("change_sequence").max().alias("key_max"), pl.col("change_sequence").n_unique().alias("key_unique")).collect().row(0, named=True)
                if metrics["key_min"] != receipt.get("key_min") or metrics["key_max"] != receipt.get("key_max") or int(metrics["key_unique"] or 0) != rows:
                    raise RuntimeError(f"incremental append segment key receipt failed: {path.name}")
        return path

    @staticmethod
    def _dataset(manifest: Mapping[str, object], artifact: str) -> Mapping[str, object]:
        if manifest.get("contract") == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT:
            if artifact != GAMES_ARTIFACT:
                raise RuntimeError(f"legacy incremental release does not contain {artifact}")
            dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), Mapping) else None
        else:
            datasets = manifest.get("datasets") if isinstance(manifest.get("datasets"), Mapping) else None
            dataset = datasets.get(artifact) if isinstance(datasets, Mapping) and isinstance(datasets.get(artifact), Mapping) else None
        if not isinstance(dataset, Mapping):
            raise RuntimeError(f"incremental release omits dataset {artifact}")
        return dataset

    @classmethod
    def _dataset_receipts(cls, manifest: Mapping[str, object], artifact: str = GAMES_ARTIFACT) -> list[Mapping[str, object]]:
        dataset = cls._dataset(manifest, artifact)
        base = dataset.get("base")
        deltas = dataset.get("deltas")
        if not isinstance(base, Mapping) or not isinstance(deltas, list) or any(not isinstance(value, Mapping) for value in deltas):
            raise RuntimeError("incremental release has malformed base or delta receipts")
        receipts = [base, *deltas]
        paths = [receipt.get("path") for receipt in receipts]
        if len(paths) != len(set(paths)):
            raise RuntimeError("incremental release repeats a segment")
        return receipts

    def _scan_incremental(self, manifest: Mapping[str, object], artifact: str = GAMES_ARTIFACT) -> pl.LazyFrame:
        dataset = self._dataset(manifest, artifact)
        scans = []
        for ordinal, receipt in enumerate(self._dataset_receipts(manifest, artifact)):
            scans.append(pl.scan_parquet(self._segment_path(receipt)).with_columns(pl.lit(ordinal).alias("_segment_ordinal")))
        stacked = pl.concat(scans, how="vertical")
        mode = str(dataset.get("mode") or "upsert")
        if mode == "upsert":
            return stacked.sort(["game_id", "history_revision", "_segment_ordinal"]).unique(subset=["game_id"], keep="last", maintain_order=True).drop("_segment_ordinal").sort(["completed_at", "game_id"])
        if mode == "append":
            sort_by = dataset.get("sort_by")
            if not isinstance(sort_by, list) or not sort_by or any(not isinstance(value, str) for value in sort_by):
                raise RuntimeError(f"append dataset {artifact} omits its sort order")
            return stacked.drop("_segment_ordinal").sort(sort_by)
        if mode == "snapshot":
            if len(scans) != 1:
                raise RuntimeError(f"snapshot dataset {artifact} contains deltas")
            return stacked.drop("_segment_ordinal")
        raise RuntimeError(f"incremental dataset {artifact} has unsupported mode {mode!r}")

    def _validate_incremental_dataset(self, manifest: Mapping[str, object], artifact: str, *, reopen: bool) -> int:
        dataset = self._dataset(manifest, artifact)
        receipts = self._dataset_receipts(manifest, artifact)
        for receipt in receipts:
            self._validate_segment(receipt, artifact=artifact, reopen=reopen)
        logical_rows = dataset.get("logical_rows")
        physical_rows = dataset.get("physical_rows")
        if not isinstance(logical_rows, int) or logical_rows < 0 or not isinstance(physical_rows, int) or physical_rows != sum(int(receipt["rows"]) for receipt in receipts):
            raise RuntimeError(f"incremental dataset {artifact} contains invalid logical or physical row counts")
        mode = dataset.get("mode") or ("upsert" if manifest.get("contract") == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT else None)
        if mode == "upsert":
            if dataset.get("primary_key") != ["game_id"] or dataset.get("resolution_order") != ["history_revision", "segment_ordinal"]:
                raise RuntimeError(f"incremental dataset {artifact} has an unsupported upsert contract")
            if reopen:
                physical = pl.concat([pl.read_parquet(self._segment_path(receipt)).with_columns(pl.lit(index).alias("_segment_ordinal")) for index, receipt in enumerate(receipts)], how="vertical")
                conflicts = physical.group_by(["game_id", "history_revision"]).agg(pl.col("history_record_sha256").n_unique().alias("hashes")).filter(pl.col("hashes") > 1)
                if conflicts.height:
                    raise RuntimeError(f"incremental release contains {conflicts.height} conflicting history revisions")
                rows = int(self._scan_incremental(manifest, artifact).select(pl.len()).collect().item(0, 0))
                if rows != logical_rows:
                    raise RuntimeError(f"incremental logical row count differs for {artifact}: {rows} != {logical_rows}")
        elif mode == "append":
            if logical_rows != physical_rows:
                raise RuntimeError(f"append dataset {artifact} has unequal logical and physical row counts")
            previous_max: int | None = None
            for receipt in receipts:
                rows = int(receipt["rows"])
                key_min = receipt.get("key_min")
                key_max = receipt.get("key_max")
                if rows == 0:
                    if key_min is not None or key_max is not None:
                        raise RuntimeError(f"empty append segment for {artifact} has a key range")
                    continue
                if not isinstance(key_min, int) or not isinstance(key_max, int) or key_min > key_max or (previous_max is not None and key_min <= previous_max):
                    raise RuntimeError(f"append dataset {artifact} has overlapping or invalid key ranges")
                previous_max = key_max
        elif mode == "snapshot":
            if len(receipts) != 1 or logical_rows != physical_rows:
                raise RuntimeError(f"snapshot dataset {artifact} has deltas or inconsistent row counts")
        else:
            raise RuntimeError(f"incremental dataset {artifact} has unsupported mode {mode!r}")
        return logical_rows

    def _validate_release(self, release: str, *, expected_manifest_sha256: str | None = None, reopen: bool = True) -> dict[str, object]:
        release_path = self._release_path(release)
        manifest_path = release_path / "manifest.json"
        manifest_sha256 = _file_digest(manifest_path)
        if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError("online analytics manifest differs from its pointer")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("online analytics manifest is not an object")
        release_contract = manifest.get("contract")
        incremental_logical_rows: int | None = None
        if release_contract not in {ONLINE_INCREMENTAL_RELEASE_CONTRACT, ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT, ONLINE_ANALYTICS_RELEASE_CONTRACT, GLEE_ANALYTICS_LAKE_CONTRACT}:
            raise RuntimeError("online analytics release has an unsupported lake contract")
        if release_contract in {ONLINE_INCREMENTAL_RELEASE_CONTRACT, ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT}:
            artifacts = (GAMES_ARTIFACT,) if release_contract == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT else (*ONLINE_REQUIRED_ARTIFACTS, *ONLINE_INTERNAL_ARTIFACTS)
            logical_by_artifact = {artifact: self._validate_incremental_dataset(manifest, artifact, reopen=reopen) for artifact in artifacts}
            incremental_logical_rows = logical_by_artifact[GAMES_ARTIFACT]
            if release_contract == ONLINE_INCREMENTAL_RELEASE_CONTRACT:
                reporter_cursor = manifest.get("reporter_cursor") if isinstance(manifest.get("reporter_cursor"), Mapping) else {}
                change_sequence = reporter_cursor.get("change_sequence")
                change_events = reporter_cursor.get("change_events")
                if not isinstance(change_sequence, int) or change_sequence < 0 or not isinstance(change_events, int) or change_events < 0:
                    raise RuntimeError("incremental analytics release omits its reporter change cursor")
                if logical_by_artifact[PUBLIC_UPDATES_ARTIFACT] != change_events:
                    raise RuntimeError("public update rows differ from the reporter change cursor")
                materialized = int((manifest.get("activity_invariant") or {}).get("materialized_game_additions") or 0) if isinstance(manifest.get("activity_invariant"), Mapping) else -1
                if materialized < 0:
                    raise RuntimeError("incremental analytics release omits its public-activity invariant")
                if reopen:
                    activity_sum = int(self._scan_incremental(manifest, PUBLIC_ACTIVITY_ARTIFACT).select(pl.col("games_delta").sum()).collect().item(0, 0) or 0)
                    state_metrics = self._scan_incremental(manifest, PUBLIC_STATE_ARTIFACT).select(pl.len().alias("rows"), pl.col("materialized_additions").sum().alias("materialized"), pl.col("effective_additions").sum().alias("effective")).collect().row(0, named=True)
                    invariant = manifest.get("activity_invariant") if isinstance(manifest.get("activity_invariant"), Mapping) else {}
                    expected = (int(invariant.get("player_family_states") or 0), int(invariant.get("materialized_game_additions") or 0), int(invariant.get("effective_game_additions") or 0))
                    actual = (int(state_metrics["rows"]), int(state_metrics["materialized"] or 0), int(state_metrics["effective"] or 0))
                    if actual != expected or activity_sum != expected[1]:
                        raise RuntimeError(f"incremental public-activity invariant differs: state={actual}, activity={activity_sum}, manifest={expected}")
        else:
            required_artifacts = (GAMES_ARTIFACT,) if release_contract == ONLINE_ANALYTICS_RELEASE_CONTRACT else LEGACY_FULL_ARTIFACTS
            artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}
            for name in required_artifacts:
                receipt = artifacts.get(name) if isinstance(artifacts, Mapping) else None
                if not isinstance(receipt, Mapping) or not isinstance(receipt.get("sha256"), str) or not isinstance(receipt.get("rows"), int):
                    raise RuntimeError(f"online analytics manifest omits a complete receipt for {name}")
                path = release_path / name
                if not path.is_file() or path.stat().st_size != int(receipt.get("bytes") or -1) or _file_digest(path) != receipt["sha256"]:
                    raise RuntimeError(f"online analytics artifact verification failed: {name}")
                if reopen:
                    rows = int(pl.scan_parquet(path).select(pl.len()).collect().item(0, 0))
                    if rows != int(receipt["rows"]):
                        raise RuntimeError(f"online analytics Parquet row count differs for {name}: {rows} != {receipt['rows']}")
        source = manifest.get("source_frontier") if isinstance(manifest.get("source_frontier"), Mapping) else {}
        inventory = manifest.get("inventory") if isinstance(manifest.get("inventory"), Mapping) else {}
        frontier = source.get("frontier_sequence")
        history_games = inventory.get("history_games")
        if not isinstance(frontier, int) or not isinstance(history_games, int):
            raise RuntimeError("online analytics manifest omits its source frontier or game count")
        if incremental_logical_rows is not None and history_games != incremental_logical_rows:
            raise RuntimeError("incremental release inventory differs from its logical row count")
        history_cursor = manifest.get("history_cursor") if isinstance(manifest.get("history_cursor"), Mapping) else {}
        history_revision_rowid = history_cursor.get("revision_rowid", 0)
        if release_contract in {ONLINE_INCREMENTAL_RELEASE_CONTRACT, ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT} and (not isinstance(history_revision_rowid, int) or history_revision_rowid < 0):
            raise RuntimeError("incremental release omits its history revision cursor")
        reporter_cursor = manifest.get("reporter_cursor") if isinstance(manifest.get("reporter_cursor"), Mapping) else {}
        reporter_change_sequence = reporter_cursor.get("change_sequence", 0)
        return {
            "contract": ONLINE_ANALYTICS_LAKE_CONTRACT,
            "status": "verified",
            "release": release,
            "path": str(release_path),
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "frontier_sequence": frontier,
            "history_games": history_games,
            "history_revision_rowid": history_revision_rowid,
            "reporter_change_sequence": reporter_change_sequence,
            "release_contract": release_contract,
            "profile": ONLINE_INCREMENTAL_PROFILE if release_contract == ONLINE_INCREMENTAL_RELEASE_CONTRACT else ("terminal-games-incremental" if release_contract == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT else ("terminal-games-online" if release_contract == ONLINE_ANALYTICS_RELEASE_CONTRACT else "legacy-full-analytics")),
        }

    def try_resolve(self, *, reopen: bool = True) -> dict[str, object]:
        errors: list[str] = []
        if self.pointer_path.is_file():
            try:
                pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
                if pointer.get("contract") != ONLINE_ANALYTICS_POINTER_CONTRACT:
                    raise RuntimeError("online analytics pointer has an unsupported contract")
                selected = self._validate_release(str(pointer.get("release") or ""), expected_manifest_sha256=str(pointer.get("manifest_sha256") or ""), reopen=reopen)
                if selected["frontier_sequence"] != pointer.get("frontier_sequence") or selected["history_games"] != pointer.get("history_games"):
                    raise RuntimeError("online analytics pointer metadata differs from its release")
                if pointer.get("history_revision_rowid") is not None and selected["history_revision_rowid"] != pointer.get("history_revision_rowid"):
                    raise RuntimeError("online analytics pointer history cursor differs from its release")
                if pointer.get("reporter_change_sequence") is not None and selected["reporter_change_sequence"] != pointer.get("reporter_change_sequence"):
                    raise RuntimeError("online analytics pointer reporter cursor differs from its release")
                if pointer.get("release_contract") is not None and selected["release_contract"] != pointer.get("release_contract"):
                    raise RuntimeError("online analytics pointer release contract differs from its release")
                if pointer.get("profile") is not None and selected["profile"] != pointer.get("profile"):
                    raise RuntimeError("online analytics pointer profile differs from its release")
                return {**selected, "selection": "current-pointer", "pointer": pointer, "errors": []}
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
                errors.append(f"current pointer rejected: {type(error).__name__}: {error}")
        candidates: list[dict[str, object]] = []
        if self.releases_root.is_dir():
            for manifest_path in self.releases_root.glob("*/manifest.json"):
                try:
                    candidates.append(self._validate_release(manifest_path.parent.name, reopen=reopen))
                except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
                    errors.append(f"release {manifest_path.parent.name!r} rejected: {type(error).__name__}: {error}")
        if candidates:
            selected = max(candidates, key=lambda item: (int(item.get("reporter_change_sequence") or 0), int(item.get("history_revision_rowid") or 0), int((item.get("manifest") or {}).get("generation") or 0), int(item["frontier_sequence"]), int(item["history_games"]), str(item["release"])))
            return {**selected, "selection": "last-verified-parquet-fallback", "pointer": None, "errors": errors}
        return {
            "contract": ONLINE_ANALYTICS_LAKE_CONTRACT,
            "status": "unavailable",
            "selection": "sqlite-wal-and-archive-fallback-required",
            "path": None,
            "pointer": None,
            "errors": errors or ["no published Parquet release exists"],
        }

    def resolve(self, *, reopen: bool = True) -> dict[str, object]:
        selection = self.try_resolve(reopen=reopen)
        if selection["status"] != "verified":
            raise RuntimeError("no verified online Parquet frontier is available; use the SQLite WAL and immutable game archives")
        return selection

    def scan(self, artifact: str) -> tuple[pl.LazyFrame, dict[str, object]]:
        if artifact not in {*ONLINE_REQUIRED_ARTIFACTS, *ONLINE_INTERNAL_ARTIFACTS}:
            raise ValueError(f"unsupported online analytics artifact: {artifact}")
        selection = self.resolve()
        if selection["release_contract"] in {ONLINE_INCREMENTAL_RELEASE_CONTRACT, ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT}:
            return self._scan_incremental(selection["manifest"], artifact), selection
        if artifact not in ((GAMES_ARTIFACT,) if selection["release_contract"] == ONLINE_ANALYTICS_RELEASE_CONTRACT else LEGACY_FULL_ARTIFACTS):
            raise RuntimeError(f"selected legacy release does not contain {artifact}")
        return pl.scan_parquet(Path(str(selection["path"])) / artifact), selection


class GleeOnlineGameLakeBuilder:
    """Materialize only terminal game facts for the low-cost online profile."""

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
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.fleet_groups = fleet_groups.resolve()
        self.output_dir = output_dir.resolve()
        self.reporter_frontier = reporter_frontier
        self.self_name = self_name
        self.windows_s = tuple(windows_s)

    def _snapshot(self, connection: Any) -> dict[str, object]:
        maximum = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM frontiers").fetchone()[0])
        frontier = maximum if self.reporter_frontier is None else int(self.reporter_frontier)
        if frontier < 1 or frontier > maximum:
            raise ValueError(f"reporter frontier must be between 1 and {maximum}: {frontier}")
        row = connection.execute("SELECT sequence, frontier_id, started_at, completed_at FROM frontiers WHERE sequence = ?", (frontier,)).fetchone()
        first = connection.execute("SELECT MIN(started_at) FROM frontiers WHERE sequence <= ?", (frontier,)).fetchone()[0]
        if row is None or first is None:
            raise RuntimeError(f"incomplete reporter frontier: {frontier}")
        return {"frontier_sequence": frontier, "frontier_id": str(row["frontier_id"]), "first_started_at": str(first), "frontier_started_at": str(row["started_at"]), "frontier_completed_at": str(row["completed_at"])}

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"online game-lake output directory already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging_dir.mkdir(mode=0o700)
        _fsync_directory(staging_dir.parent)
        reporter = _read_only_database(self.reporter_database)
        history = _read_only_database(self.history_database)
        reporter.execute("BEGIN")
        history.execute("BEGIN")
        try:
            reporter_mode = str(reporter.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            history_mode = str(history.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            if reporter_mode != "wal" or history_mode != "wal":
                raise RuntimeError(f"online game-lake authorities must use SQLite WAL: reporter={reporter_mode}, history={history_mode}")
            snapshot = self._snapshot(reporter)
            games = _history_frame(history, first_at=str(snapshot["first_started_at"]), completed_by=str(snapshot["frontier_completed_at"]))
            archives, archive_inventory = _archive_metadata(self.game_archive_root, set(games["game_id"].to_list()))
            reporter.execute("ROLLBACK")
            history.execute("ROLLBACK")
        finally:
            if reporter.in_transaction:
                reporter.execute("ROLLBACK")
            if history.in_transaction:
                history.execute("ROLLBACK")
            reporter.close()
            history.close()
        games = (
            games.join(archives, on="game_id", how="left")
            .with_columns(pl.col("identity_scope").fill_null("archive-missing"), pl.col("role").fill_null("unknown"), pl.col("engine_version").fill_null("engine-unknown"))
            .sort(["completed_at", "game_id"])
        )
        artifacts = {"games.parquet": _write_parquet(games, staging_dir / "games.parquet")}
        manifest = {
            "schema_version": 1,
            "contract": ONLINE_ANALYTICS_RELEASE_CONTRACT,
            "profile": "terminal-games-online",
            "created_at": _now(),
            "source_frontier": {**snapshot, "reporter_database": str(self.reporter_database), "history_database": str(self.history_database), "game_archive_root": str(self.game_archive_root)},
            "source_consistency": {"reporter_sqlite_journal_mode": reporter_mode, "history_sqlite_journal_mode": history_mode, "read_transactions": "explicit read-only snapshots held through extraction"},
            "storage": {"format": "Apache Parquet", "compression": "Zstandard level 7", "sort_keys": {"games.parquet": ["completed_at", "game_id"]}, "commit_protocol": "fsync-reopen-validate-hash-stage-then-atomic-directory-rename"},
            "parameters": {"self_name": self.self_name, "heavy_public_activity_windows": "offline-only"},
            "inventory": {**archive_inventory, "history_games": games.height, "games_by_family": dict(sorted(Counter(games["family"].to_list()).items())), "games_by_identity": dict(sorted(Counter(games["identity_scope"].to_list()).items()))},
            "artifacts": artifacts,
            "implementation_sha256": _file_digest(Path(__file__)),
        }
        _atomic_json(staging_dir / "manifest.json", manifest)
        manifest_sha256 = _file_digest(staging_dir / "manifest.json")
        for name, receipt in artifacts.items():
            path = staging_dir / name
            if path.stat().st_size != receipt["bytes"] or _file_digest(path) != receipt["sha256"]:
                raise RuntimeError(f"staged online game-lake artifact changed before commit: {path}")
        _fsync_directory(staging_dir)
        os.replace(staging_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        if _file_digest(self.output_dir / "manifest.json") != manifest_sha256:
            raise RuntimeError(f"committed online game-lake manifest changed during publication: {self.output_dir}")
        return {"contract": ONLINE_ANALYTICS_RELEASE_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": snapshot["frontier_sequence"], "games": games.height, "manifest_sha256": manifest_sha256}


class GleeIncrementalGameLakeBuilder(GleeOnlineGameLakeBuilder):
    """Publish immutable revision upserts and compact them into a shared content-addressed base."""

    def __init__(
        self,
        *,
        output_root: Path,
        current: Mapping[str, object],
        max_delta_segments: int = 32,
        compaction_growth_ratio: float = 0.10,
        **values: object,
    ) -> None:
        super().__init__(**values)
        if max_delta_segments < 2:
            raise ValueError("incremental game lake must allow at least 2 delta segments")
        if not 0 < compaction_growth_ratio <= 1:
            raise ValueError("incremental compaction growth ratio must be greater than zero and at most one")
        self.output_root = output_root.resolve()
        self.segments_root = self.output_root / "segments"
        self.reader = GleeOnlineAnalyticsLakeReader(self.output_root)
        self.current = current
        self.max_delta_segments = int(max_delta_segments)
        self.compaction_growth_ratio = float(compaction_growth_ratio)

    @staticmethod
    def _history_cursor(connection: Any) -> dict[str, object]:
        events, rowid = connection.execute("SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM revisions").fetchone()
        cursor: dict[str, object] = {"revision_events": int(events), "revision_rowid": int(rowid)}
        if rowid:
            head = connection.execute("SELECT game_id, revision, record_sha256 FROM revisions WHERE rowid = ?", (int(rowid),)).fetchone()
            if head is None:
                raise RuntimeError("history revision cursor head disappeared")
            cursor.update({"head_game_id": str(head[0]), "head_game_revision": int(head[1]), "head_record_sha256": str(head[2])})
        return cursor

    @staticmethod
    def _validate_previous_cursor(connection: Any, previous: Mapping[str, object], current: Mapping[str, object]) -> None:
        previous_rowid = previous.get("revision_rowid")
        previous_events = previous.get("revision_events")
        if not isinstance(previous_rowid, int) or previous_rowid < 0 or not isinstance(previous_events, int) or previous_events < 0:
            raise RuntimeError("previous incremental release contains an invalid history cursor")
        if int(current["revision_rowid"]) < previous_rowid or int(current["revision_events"]) < previous_events:
            raise RuntimeError("authoritative history revision stream moved backwards")
        if previous_rowid:
            head = connection.execute("SELECT game_id, revision, record_sha256 FROM revisions WHERE rowid = ?", (previous_rowid,)).fetchone()
            expected = (previous.get("head_game_id"), previous.get("head_game_revision"), previous.get("head_record_sha256"))
            actual = (str(head[0]), int(head[1]), str(head[2])) if head is not None else None
            if actual != expected:
                raise RuntimeError("authoritative history changed at the published revision cursor")

    def _write_segment(self, frame: pl.DataFrame) -> dict[str, object]:
        normalized = _normalize_games(frame)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        temporary = self.segments_root / f".segment-{os.getpid()}-{uuid.uuid4().hex}.parquet"
        try:
            receipt = _write_parquet(normalized, temporary)
            sha256 = str(receipt["sha256"])
            final = self.segments_root / f"{sha256}.parquet"
            segment = {**receipt, "path": str(final.relative_to(self.output_root)), "schema_sha256": _schema_sha256(normalized.schema)}
            if final.exists():
                self.reader._validate_segment(segment, artifact=GAMES_ARTIFACT, reopen=True)
                temporary.unlink()
            else:
                os.replace(temporary, final)
                _fsync_directory(self.segments_root)
                self.reader._validate_segment(segment, artifact=GAMES_ARTIFACT, reopen=True)
            return segment
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _manifest_dataset(current: Mapping[str, object]) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
        manifest = current.get("manifest") if isinstance(current.get("manifest"), Mapping) else None
        dataset = manifest.get("dataset") if isinstance(manifest, Mapping) and isinstance(manifest.get("dataset"), Mapping) else None
        if not isinstance(dataset, Mapping) or not isinstance(dataset.get("base"), Mapping) or not isinstance(dataset.get("deltas"), list):
            raise RuntimeError("current incremental release has no reusable dataset")
        deltas = dataset["deltas"]
        if any(not isinstance(value, Mapping) for value in deltas):
            raise RuntimeError("current incremental release has malformed delta receipts")
        return dataset["base"], list(deltas)

    def _current_frame(self) -> pl.DataFrame:
        manifest = self.current.get("manifest")
        if not isinstance(manifest, Mapping):
            raise RuntimeError("current incremental release has no manifest")
        return _normalize_games(self.reader._scan_incremental(manifest).collect())

    @staticmethod
    def _rows_by_id(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
        return {str(row["game_id"]): row for row in frame.to_dicts()}

    def _incremental_rows(self, history: Any, *, first_at: str, cursor: Mapping[str, object], current_frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, object]]:
        previous_manifest = self.current.get("manifest") if isinstance(self.current.get("manifest"), Mapping) else {}
        previous_cursor = previous_manifest.get("history_cursor") if isinstance(previous_manifest, Mapping) and isinstance(previous_manifest.get("history_cursor"), Mapping) else {}
        previous_rowid = int(previous_cursor.get("revision_rowid") or 0)
        current_rowid = int(cursor["revision_rowid"])
        changed_ids = [str(row[0]) for row in history.execute("SELECT DISTINCT game_id FROM revisions WHERE rowid > ? AND rowid <= ? ORDER BY game_id", (previous_rowid, current_rowid))]
        history_rows = _history_rows(history, first_at=first_at, game_ids=changed_ids)
        history_by_id = self._rows_by_id(history_rows)
        current_by_id = self._rows_by_id(current_frame)
        changed_existing = set(changed_ids).intersection(current_by_id)
        missing_changed = changed_existing.difference(history_by_id)
        if missing_changed:
            raise RuntimeError(f"authoritative history removed {len(missing_changed)} published games")
        pending_ids = {game_id for game_id, row in current_by_id.items() if row.get("identity_scope") == "archive-missing"}
        archive_targets = {
            game_id: str(history_by_id.get(game_id, current_by_id.get(game_id, {})).get("family") or "")
            for game_id in set(history_by_id).union(pending_ids)
            if game_id not in current_by_id or current_by_id[game_id].get("identity_scope") == "archive-missing"
        }
        archives, archive_inventory = _targeted_archive_metadata(self.game_archive_root, archive_targets)
        archive_by_id = self._rows_by_id(archives)
        upsert_ids = set(history_by_id).union(pending_ids.intersection(archive_by_id))
        upserts: list[dict[str, object]] = []
        for game_id in sorted(upsert_ids):
            previous = current_by_id.get(game_id)
            candidate = dict(previous or {})
            history_row = history_by_id.get(game_id)
            if history_row is not None:
                candidate.update(history_row)
            archive_row = archive_by_id.get(game_id)
            if archive_row is not None:
                candidate.update(archive_row)
            elif previous is None:
                candidate.update({"family_archive": None, "identity_scope": "archive-missing", "opponent_name": None, "role": "unknown", "engine_version": "engine-unknown", "advisor_version": None, "policy_revision": None, "archive_path": None, "archive_sha256": None, "archive_specificity": None})
            candidate = {column: candidate.get(column) for column in ONLINE_GAME_COLUMNS}
            if previous != candidate:
                upserts.append(candidate)
        frame = pl.DataFrame(upserts, schema=ONLINE_GAME_SCHEMA) if upserts else _empty_games()
        return _normalize_games(frame), {**archive_inventory, "history_revision_ids": len(changed_ids), "history_upserts": history_rows.height, "archive_enrichments": len(pending_ids.intersection(archive_by_id)), "published_upserts": frame.height}

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"incremental game-lake output directory already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        staging_dir = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging_dir.mkdir(mode=0o700)
        _fsync_directory(staging_dir.parent)
        reporter = _read_only_database(self.reporter_database)
        history = _read_only_database(self.history_database)
        reporter.execute("BEGIN")
        history.execute("BEGIN")
        try:
            reporter_mode = str(reporter.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            history_mode = str(history.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            if reporter_mode != "wal" or history_mode != "wal":
                raise RuntimeError(f"incremental game-lake authorities must use SQLite WAL: reporter={reporter_mode}, history={history_mode}")
            snapshot = self._snapshot(reporter)
            cursor = self._history_cursor(history)
            current_incremental = self.current.get("status") == "verified" and self.current.get("release_contract") == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT
            previous_manifest = self.current.get("manifest") if current_incremental and isinstance(self.current.get("manifest"), Mapping) else {}
            previous_source = previous_manifest.get("source_frontier") if isinstance(previous_manifest, Mapping) and isinstance(previous_manifest.get("source_frontier"), Mapping) else {}
            if current_incremental and previous_source.get("first_started_at") != snapshot["first_started_at"]:
                current_incremental = False
            if current_incremental:
                previous_cursor = previous_manifest.get("history_cursor") if isinstance(previous_manifest.get("history_cursor"), Mapping) else {}
                self._validate_previous_cursor(history, previous_cursor, cursor)
                current_frame = self._current_frame()
                upserts, build_inventory = self._incremental_rows(history, first_at=str(snapshot["first_started_at"]), cursor=cursor, current_frame=current_frame)
            else:
                history_rows = _history_rows(history, first_at=str(snapshot["first_started_at"]))
                archives, build_inventory = _archive_metadata(self.game_archive_root, set(history_rows["game_id"].to_list()))
                current_frame = _empty_games()
                upserts = _join_history_archives(history_rows, archives)
            expected_games = int(history.execute("SELECT COUNT(*) FROM games WHERE started_at IS NOT NULL AND completed_at IS NOT NULL AND rating_delta IS NOT NULL AND started_at >= ? AND game_family IN ('bargaining','negotiation','persuasion')", (str(snapshot["first_started_at"]),)).fetchone()[0])
            reporter.execute("ROLLBACK")
            history.execute("ROLLBACK")
        finally:
            if reporter.in_transaction:
                reporter.execute("ROLLBACK")
            if history.in_transaction:
                history.execute("ROLLBACK")
            reporter.close()
            history.close()
        if current_incremental:
            base, deltas = self._manifest_dataset(self.current)
            prospective_deltas = list(deltas)
            delta_receipt = self._write_segment(upserts) if upserts.height else None
            if delta_receipt is not None:
                prospective_deltas.append(delta_receipt)
            base_rows = int(base["rows"])
            delta_rows = sum(int(receipt["rows"]) for receipt in prospective_deltas)
            growth_limit = max(1, math.ceil(base_rows * self.compaction_growth_ratio))
            compact = len(prospective_deltas) >= self.max_delta_segments or delta_rows > growth_limit
            logical = _logical_games([current_frame, upserts])
            if compact:
                base = self._write_segment(logical)
                prospective_deltas = []
                publication_mode = "compacted"
            else:
                publication_mode = "delta" if delta_receipt is not None else "metadata-only"
            generation = int(previous_manifest.get("generation") or 0) + 1
            parent_manifest_sha256 = self.current.get("manifest_sha256")
        else:
            logical = _normalize_games(upserts).sort(["completed_at", "game_id"])
            base = self._write_segment(logical)
            prospective_deltas = []
            publication_mode = "bootstrap" if self.current.get("status") != "verified" else "migrated-from-legacy"
            generation = 1
            parent_manifest_sha256 = self.current.get("manifest_sha256") if self.current.get("status") == "verified" else None
        if logical.height != expected_games:
            raise RuntimeError(f"incremental logical game count differs from authoritative history: {logical.height} != {expected_games}")
        physical_rows = int(base["rows"]) + sum(int(receipt["rows"]) for receipt in prospective_deltas)
        manifest = {
            "schema_version": 2,
            "contract": ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT,
            "profile": ONLINE_INCREMENTAL_PROFILE,
            "generation": generation,
            "created_at": _now(),
            "parent_manifest_sha256": parent_manifest_sha256,
            "source_frontier": {**snapshot, "reporter_database": str(self.reporter_database), "history_database": str(self.history_database), "game_archive_root": str(self.game_archive_root)},
            "history_cursor": cursor,
            "source_consistency": {"reporter_sqlite_journal_mode": reporter_mode, "history_sqlite_journal_mode": history_mode, "read_transactions": "explicit read-only snapshots held through revision extraction", "revision_stream": "append-only rowid cursor with head receipt and rewind detection"},
            "storage": {"format": "Apache Parquet base plus immutable upsert deltas", "compression": "Zstandard level 7", "segment_addressing": "SHA-256 content-addressed", "commit_protocol": "fsync segment, fsync manifest stage, atomic release rename, atomic pointer replace", "compaction": {"maximum_delta_segments": self.max_delta_segments, "maximum_delta_rows_over_base": self.compaction_growth_ratio}},
            "dataset": {"primary_key": ["game_id"], "resolution_order": ["history_revision", "segment_ordinal"], "base": base, "deltas": prospective_deltas, "logical_rows": logical.height, "physical_rows": physical_rows, "publication_mode": publication_mode},
            "parameters": {"self_name": self.self_name, "heavy_public_activity_windows": "offline-only"},
            "inventory": {**build_inventory, "history_games": logical.height, "games_by_family": dict(sorted(Counter(logical["family"].to_list()).items())), "games_by_identity": dict(sorted(Counter(logical["identity_scope"].to_list()).items())), "missing_archives": int(logical.filter(pl.col("identity_scope") == "archive-missing").height), "delta_segments": len(prospective_deltas), "delta_rows": sum(int(receipt["rows"]) for receipt in prospective_deltas)},
            "implementation_sha256": _file_digest(Path(__file__)),
        }
        _atomic_json(staging_dir / "manifest.json", manifest)
        manifest_sha256 = _file_digest(staging_dir / "manifest.json")
        _fsync_directory(staging_dir)
        os.replace(staging_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        if _file_digest(self.output_dir / "manifest.json") != manifest_sha256:
            raise RuntimeError(f"committed incremental game-lake manifest changed during publication: {self.output_dir}")
        return {"contract": ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": snapshot["frontier_sequence"], "history_revision_rowid": cursor["revision_rowid"], "games": logical.height, "publication_mode": publication_mode, "manifest_sha256": manifest_sha256}


class GleeIncrementalAnalyticsLakeBuilder(GleeIncrementalGameLakeBuilder):
    """Publish all rated games and every public-board change through 2 independent append cursors."""

    @staticmethod
    def _reporter_cursor(connection: Any, *, frontier_sequence: int) -> dict[str, object]:
        events, sequence = connection.execute("SELECT COUNT(*), COALESCE(MAX(change_sequence), 0) FROM changes WHERE frontier_sequence <= ?", (frontier_sequence,)).fetchone()
        cursor: dict[str, object] = {"change_events": int(events), "change_sequence": int(sequence)}
        if sequence:
            row = connection.execute(
                """
                SELECT ch.change_sequence, ch.frontier_sequence, ch.family, ch.player_id, ch.change_kind,
                       ch.observed_after, ch.observed_by, ch.games_delta, ch.rating_delta, ch.changed_fields_json,
                       rv.row_sha256, rv.previous_row_sha256
                FROM changes AS ch
                JOIN row_versions AS rv
                  ON rv.sequence = ch.frontier_sequence
                 AND rv.family = ch.family
                 AND rv.player_id = ch.player_id
                WHERE ch.change_sequence = ?
                """,
                (int(sequence),),
            ).fetchone()
            if row is None:
                raise RuntimeError("reporter change cursor head disappeared")
            payload = {key: row[key] for key in row.keys()}
            cursor.update({"head_frontier_sequence": int(row["frontier_sequence"]), "head_family": str(row["family"]), "head_player_id": str(row["player_id"]), "head_sha256": _digest(payload)})
        return cursor

    @classmethod
    def _validate_previous_reporter_cursor(cls, connection: Any, previous: Mapping[str, object], current: Mapping[str, object]) -> None:
        previous_sequence = previous.get("change_sequence")
        previous_events = previous.get("change_events")
        if not isinstance(previous_sequence, int) or previous_sequence < 0 or not isinstance(previous_events, int) or previous_events < 0:
            raise RuntimeError("previous incremental release contains an invalid reporter cursor")
        if int(current["change_sequence"]) < previous_sequence or int(current["change_events"]) < previous_events:
            raise RuntimeError("authoritative reporter change stream moved backwards")
        if previous_sequence:
            observed = cls._reporter_cursor(connection, frontier_sequence=int(previous.get("head_frontier_sequence") or 0))
            if int(observed.get("change_sequence") or 0) != previous_sequence or observed.get("head_sha256") != previous.get("head_sha256"):
                raise RuntimeError("authoritative reporter changed at the published change cursor")

    def _write_dataset_segment(self, frame: pl.DataFrame, artifact: str, *, append_key: str | None = None) -> dict[str, object]:
        schema = ONLINE_DATASET_SCHEMAS[artifact]
        normalized = normalize_frame(frame, schema, label=artifact)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        temporary = self.segments_root / f".segment-{os.getpid()}-{uuid.uuid4().hex}.parquet"
        try:
            receipt = _write_parquet(normalized, temporary)
            sha256 = str(receipt["sha256"])
            final = self.segments_root / f"{sha256}.parquet"
            segment: dict[str, object] = {**receipt, "path": str(final.relative_to(self.output_root)), "schema_sha256": _schema_sha256(normalized.schema), "artifact": artifact}
            if append_key is not None:
                segment["key_min"] = int(normalized[append_key].min()) if normalized.height else None
                segment["key_max"] = int(normalized[append_key].max()) if normalized.height else None
            if final.exists():
                self.reader._validate_segment(segment, artifact=artifact, reopen=True)
                temporary.unlink()
            else:
                os.replace(temporary, final)
                _fsync_directory(self.segments_root)
                self.reader._validate_segment(segment, artifact=artifact, reopen=True)
            return segment
        finally:
            temporary.unlink(missing_ok=True)

    def _current_dataset(self, artifact: str) -> Mapping[str, object]:
        manifest = self.current.get("manifest") if isinstance(self.current.get("manifest"), Mapping) else {}
        return self.reader._dataset(manifest, artifact)

    def _compact_due(self, base: Mapping[str, object], deltas: Sequence[Mapping[str, object]]) -> bool:
        base_rows = int(base["rows"])
        delta_rows = sum(int(receipt["rows"]) for receipt in deltas)
        return len(deltas) >= self.max_delta_segments or delta_rows > max(1, math.ceil(base_rows * self.compaction_growth_ratio))

    def _publish_append_dataset(self, artifact: str, additions: pl.DataFrame, *, current_incremental: bool, force_rewrite: bool = False, fleets: pl.DataFrame | None = None) -> dict[str, object]:
        append_key = "change_sequence"
        if not current_incremental:
            base = self._write_dataset_segment(additions, artifact, append_key=append_key)
            return {"mode": "append", "primary_key": [append_key], "sort_by": [append_key], "base": base, "deltas": [], "logical_rows": additions.height, "physical_rows": additions.height, "publication_mode": "bootstrap"}
        previous = self._current_dataset(artifact)
        base = previous["base"]
        deltas = list(previous["deltas"])
        delta = self._write_dataset_segment(additions, artifact, append_key=append_key) if additions.height else None
        if delta is not None:
            previous_max = next((receipt.get("key_max") for receipt in reversed([base, *deltas]) if receipt.get("key_max") is not None), None)
            if previous_max is not None and int(delta["key_min"]) <= int(previous_max):
                raise RuntimeError(f"new {artifact} segment does not advance its append key")
            deltas.append(delta)
        compact = force_rewrite or self._compact_due(base, deltas)
        logical_rows = int(previous["logical_rows"]) + additions.height
        if compact:
            existing = self.reader._scan_incremental(self.current["manifest"], artifact).collect()
            logical = normalize_frame(pl.concat([existing, additions], how="vertical"), ONLINE_DATASET_SCHEMAS[artifact], label=artifact).sort(append_key)
            if artifact == PUBLIC_ACTIVITY_ARTIFACT and force_rewrite:
                if fleets is None:
                    raise RuntimeError("public activity fleet rewrite has no fleet dimension")
                logical = normalize_frame(
                    logical.drop("fleet_key", "fleet_confidence").join(fleets.select("player_id", "fleet_key", pl.col("confidence").alias("fleet_confidence")), on="player_id", how="left"),
                    ONLINE_PUBLIC_ACTIVITY_SCHEMA,
                    label=artifact,
                ).sort(append_key)
            if logical.height != logical_rows:
                raise RuntimeError(f"compacted append dataset {artifact} changed its row count")
            base = self._write_dataset_segment(logical, artifact, append_key=append_key)
            deltas = []
            mode = "dimension-rewrite" if force_rewrite else "compacted"
        else:
            mode = "delta" if delta is not None else "metadata-only"
        return {"mode": "append", "primary_key": [append_key], "sort_by": [append_key], "base": base, "deltas": deltas, "logical_rows": logical_rows, "physical_rows": int(base["rows"]) + sum(int(receipt["rows"]) for receipt in deltas), "publication_mode": mode}

    def _publish_game_dataset(self, upserts: pl.DataFrame, current_frame: pl.DataFrame, *, current_incremental: bool) -> tuple[dict[str, object], pl.DataFrame]:
        if not current_incremental:
            logical = _normalize_games(upserts).sort(["completed_at", "game_id"])
            base = self._write_dataset_segment(logical, GAMES_ARTIFACT)
            dataset = {"mode": "upsert", "primary_key": ["game_id"], "resolution_order": ["history_revision", "segment_ordinal"], "base": base, "deltas": [], "logical_rows": logical.height, "physical_rows": logical.height, "publication_mode": "bootstrap"}
            return dataset, logical
        previous = self._current_dataset(GAMES_ARTIFACT)
        base = previous["base"]
        deltas = list(previous["deltas"])
        delta = self._write_dataset_segment(upserts, GAMES_ARTIFACT) if upserts.height else None
        if delta is not None:
            deltas.append(delta)
        logical = _logical_games([current_frame, upserts])
        if self._compact_due(base, deltas):
            base = self._write_dataset_segment(logical, GAMES_ARTIFACT)
            deltas = []
            mode = "compacted"
        else:
            mode = "delta" if delta is not None else "metadata-only"
        dataset = {"mode": "upsert", "primary_key": ["game_id"], "resolution_order": ["history_revision", "segment_ordinal"], "base": base, "deltas": deltas, "logical_rows": logical.height, "physical_rows": int(base["rows"]) + sum(int(receipt["rows"]) for receipt in deltas), "publication_mode": mode}
        return dataset, logical

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"incremental analytics-lake output directory already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        staging_dir = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging_dir.mkdir(mode=0o700)
        _fsync_directory(staging_dir.parent)
        fleets, _fleet_for_id, _confidence = _fleet_frames(self.fleet_groups)
        fleet_sha256 = _file_digest(self.fleet_groups)
        reporter = _read_only_database(self.reporter_database)
        history = _read_only_database(self.history_database)
        reporter.execute("BEGIN")
        history.execute("BEGIN")
        try:
            reporter_mode = str(reporter.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            history_mode = str(history.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            if reporter_mode != "wal" or history_mode != "wal":
                raise RuntimeError(f"incremental analytics-lake authorities must use SQLite WAL: reporter={reporter_mode}, history={history_mode}")
            snapshot = self._snapshot(reporter)
            history_cursor = self._history_cursor(history)
            reporter_cursor = self._reporter_cursor(reporter, frontier_sequence=int(snapshot["frontier_sequence"]))
            current_incremental = self.current.get("status") == "verified" and self.current.get("release_contract") == ONLINE_INCREMENTAL_RELEASE_CONTRACT
            previous_manifest = self.current.get("manifest") if current_incremental and isinstance(self.current.get("manifest"), Mapping) else {}
            if current_incremental:
                previous_history_cursor = previous_manifest.get("history_cursor") if isinstance(previous_manifest.get("history_cursor"), Mapping) else {}
                previous_reporter_cursor = previous_manifest.get("reporter_cursor") if isinstance(previous_manifest.get("reporter_cursor"), Mapping) else {}
                self._validate_previous_cursor(history, previous_history_cursor, history_cursor)
                self._validate_previous_reporter_cursor(reporter, previous_reporter_cursor, reporter_cursor)
                current_games = _normalize_games(self.reader._scan_incremental(previous_manifest, GAMES_ARTIFACT).collect())
                game_upserts, game_inventory = self._incremental_rows(history, first_at=None, cursor=history_cursor, current_frame=current_games)
                prior_state = normalize_frame(self.reader._scan_incremental(previous_manifest, PUBLIC_STATE_ARTIFACT).collect(), ONLINE_PUBLIC_STATE_SCHEMA, label=PUBLIC_STATE_ARTIFACT)
                previous_change_sequence = int(previous_reporter_cursor.get("change_sequence") or 0)
            else:
                history_rows = _history_rows(history)
                archives, game_inventory = _archive_metadata(self.game_archive_root, set(history_rows["game_id"].to_list()))
                current_games = _empty_games()
                game_upserts = _join_history_archives(history_rows, archives)
                prior_state = empty_frame(ONLINE_PUBLIC_STATE_SCHEMA)
                previous_change_sequence = 0
            raw_updates = reporter_change_rows(reporter, after_change_sequence=previous_change_sequence, through_change_sequence=int(reporter_cursor["change_sequence"]))
            updates, activity, state, activity_receipt = derive_public_increment(raw_updates, prior_state, fleets, self_name=self.self_name)
            expected_games = int(history.execute("SELECT COUNT(*) FROM games WHERE started_at IS NOT NULL AND completed_at IS NOT NULL AND rating_delta IS NOT NULL AND game_family IN ('bargaining','negotiation','persuasion')").fetchone()[0])
            reporter.execute("ROLLBACK")
            history.execute("ROLLBACK")
        finally:
            if reporter.in_transaction:
                reporter.execute("ROLLBACK")
            if history.in_transaction:
                history.execute("ROLLBACK")
            reporter.close()
            history.close()
        games_dataset, logical_games = self._publish_game_dataset(game_upserts, current_games, current_incremental=current_incremental)
        previous_fleet_sha256 = None
        if current_incremental:
            previous_source = previous_manifest.get("source_frontier") if isinstance(previous_manifest.get("source_frontier"), Mapping) else {}
            previous_fleet_sha256 = previous_source.get("fleet_groups_sha256")
        fleet_changed = current_incremental and previous_fleet_sha256 != fleet_sha256
        updates_dataset = self._publish_append_dataset(PUBLIC_UPDATES_ARTIFACT, updates, current_incremental=current_incremental)
        activity_dataset = self._publish_append_dataset(PUBLIC_ACTIVITY_ARTIFACT, activity, current_incremental=current_incremental, force_rewrite=fleet_changed, fleets=fleets)
        state_receipt = self._write_dataset_segment(state, PUBLIC_STATE_ARTIFACT)
        state_dataset = {"mode": "snapshot", "primary_key": ["family", "player_id"], "base": state_receipt, "deltas": [], "logical_rows": state.height, "physical_rows": state.height, "publication_mode": "updated" if updates.height else "reused-content"}
        datasets = {GAMES_ARTIFACT: games_dataset, PUBLIC_UPDATES_ARTIFACT: updates_dataset, PUBLIC_ACTIVITY_ARTIFACT: activity_dataset, PUBLIC_STATE_ARTIFACT: state_dataset}
        if logical_games.height != expected_games:
            raise RuntimeError(f"incremental logical game count differs from authoritative history: {logical_games.height} != {expected_games}")
        if int(updates_dataset["logical_rows"]) != int(reporter_cursor["change_events"]):
            raise RuntimeError(f"incremental public update count differs from authoritative reporter: {updates_dataset['logical_rows']} != {reporter_cursor['change_events']}")
        cumulative_materialized = int(state["materialized_additions"].sum() or 0)
        previous_materialized = int((previous_manifest.get("activity_invariant") or {}).get("materialized_game_additions") or 0) if current_incremental and isinstance(previous_manifest.get("activity_invariant"), Mapping) else 0
        if cumulative_materialized != previous_materialized + int(activity_receipt["materialized_game_additions"]):
            raise RuntimeError("incremental public activity additions differ from the continuation state")
        generation = int(previous_manifest.get("generation") or 0) + 1 if current_incremental else 1
        parent_manifest_sha256 = self.current.get("manifest_sha256") if self.current.get("status") == "verified" else None
        publication_modes = {artifact: str(dataset["publication_mode"]) for artifact, dataset in datasets.items()}
        manifest = {
            "schema_version": 3,
            "contract": ONLINE_INCREMENTAL_RELEASE_CONTRACT,
            "profile": ONLINE_INCREMENTAL_PROFILE,
            "generation": generation,
            "created_at": _now(),
            "parent_manifest_sha256": parent_manifest_sha256,
            "source_frontier": {**snapshot, "reporter_database": str(self.reporter_database), "history_database": str(self.history_database), "game_archive_root": str(self.game_archive_root), "fleet_groups": str(self.fleet_groups), "fleet_groups_sha256": fleet_sha256},
            "history_cursor": history_cursor,
            "reporter_cursor": reporter_cursor,
            "source_consistency": {"reporter_sqlite_journal_mode": reporter_mode, "history_sqlite_journal_mode": history_mode, "read_transactions": "explicit read-only snapshots held through both cursor extractions", "history_revision_stream": "append-only rowid cursor with head receipt and rewind detection", "reporter_change_stream": "append-only change_sequence cursor with head receipt and rewind detection"},
            "storage": {"format": "Apache Parquet bases plus immutable deltas", "compression": "Zstandard level 7", "segment_addressing": "SHA-256 content-addressed", "commit_protocol": "fsync segment, fsync manifest stage, atomic release rename, atomic pointer replace", "compaction": {"maximum_delta_segments": self.max_delta_segments, "maximum_delta_rows_over_base": self.compaction_growth_ratio}},
            "datasets": datasets,
            "parameters": {"self_name": self.self_name, "public_updates": "all reporter change rows", "public_activity": "monotonic game-count high-water events excluding self and benchmark rows", "heavy_asof_windows": "offline-only"},
            "activity_invariant": {"effective_game_additions": int(state["effective_additions"].sum() or 0), "materialized_game_additions": cumulative_materialized, "materialized_event_rows": int(activity_dataset["logical_rows"]), "player_family_states": state.height, "current_batch": activity_receipt},
            "inventory": {**game_inventory, "history_games": logical_games.height, "games_by_family": dict(sorted(Counter(logical_games["family"].to_list()).items())), "games_by_identity": dict(sorted(Counter(logical_games["identity_scope"].to_list()).items())), "missing_archives": int(logical_games.filter(pl.col("identity_scope") == "archive-missing").height), "public_update_rows": int(updates_dataset["logical_rows"]), "public_activity_rows": int(activity_dataset["logical_rows"]), "public_activity_states": state.height, "publication_modes": publication_modes},
            "implementation_sha256": _file_digest(Path(__file__)),
        }
        _atomic_json(staging_dir / "manifest.json", manifest)
        manifest_sha256 = _file_digest(staging_dir / "manifest.json")
        _fsync_directory(staging_dir)
        os.replace(staging_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        if _file_digest(self.output_dir / "manifest.json") != manifest_sha256:
            raise RuntimeError(f"committed incremental analytics-lake manifest changed during publication: {self.output_dir}")
        return {"contract": ONLINE_INCREMENTAL_RELEASE_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": snapshot["frontier_sequence"], "history_revision_rowid": history_cursor["revision_rowid"], "reporter_change_sequence": reporter_cursor["change_sequence"], "games": logical_games.height, "public_updates": int(updates_dataset["logical_rows"]), "public_activity": int(activity_dataset["logical_rows"]), "publication_modes": publication_modes, "manifest_sha256": manifest_sha256}


class GleeOnlineAnalyticsLake:
    """Refresh an immutable Parquet frontier without placing Parquet on the operational critical path."""

    def __init__(
        self,
        *,
        reporter_database: Path,
        history_database: Path,
        game_archive_root: Path,
        fleet_groups: Path,
        output_root: Path,
        self_name: str = "DeepRMM-01",
        windows_s: Sequence[int] = DEFAULT_WINDOWS_S,
        poll_interval_s: float = 30.0,
        minimum_history_game_advance: int = 10,
        maximum_frontier_advance: int = 60,
        maximum_age_s: float = 900.0,
        retained_releases: int = 3,
        max_delta_segments: int = 32,
        compaction_growth_ratio: float = 0.10,
        builder_factory: Callable[..., GleeOnlineGameLakeBuilder] = GleeIncrementalAnalyticsLakeBuilder,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_s <= 0 or maximum_age_s <= 0:
            raise ValueError("online analytics polling and maximum age must be positive")
        if minimum_history_game_advance < 1 or maximum_frontier_advance < 1:
            raise ValueError("online analytics advance thresholds must be positive")
        if retained_releases < 2:
            raise ValueError("online analytics must retain at least 2 verified releases")
        if max_delta_segments < 2:
            raise ValueError("online analytics must allow at least 2 delta segments")
        if not 0 < compaction_growth_ratio <= 1:
            raise ValueError("online analytics compaction growth ratio must be greater than zero and at most one")
        self.reporter_database = reporter_database.resolve()
        self.history_database = history_database.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.fleet_groups = fleet_groups.resolve()
        self.output_root = output_root.resolve()
        self.releases_root = self.output_root / "releases"
        self.reader = GleeOnlineAnalyticsLakeReader(self.output_root)
        self.self_name = self_name
        self.windows_s = tuple(sorted(set(int(value) for value in windows_s)))
        self.poll_interval_s = float(poll_interval_s)
        self.minimum_history_game_advance = int(minimum_history_game_advance)
        self.maximum_frontier_advance = int(maximum_frontier_advance)
        self.maximum_age_s = float(maximum_age_s)
        self.retained_releases = int(retained_releases)
        self.max_delta_segments = int(max_delta_segments)
        self.compaction_growth_ratio = float(compaction_growth_ratio)
        self.builder_factory = builder_factory
        self.clock = clock
        self.sleeper = sleeper
        self.status_path = self.output_root / "status.json"
        self.events_path = self.output_root / "events.jsonl"
        self.stop_path = self.output_root / "stop.requested"
        self.stopped_path = self.output_root / "stopped.json"
        self.lock_path = self.output_root / "publisher.lock"

    def _source_state(self) -> dict[str, object]:
        reporter = None
        history = None
        try:
            reporter = _read_only_database(self.reporter_database)
            history = _read_only_database(self.history_database)
            reporter.execute("BEGIN")
            history.execute("BEGIN")
            reporter_mode = str(reporter.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            history_mode = str(history.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
            if reporter_mode != "wal" or history_mode != "wal":
                raise RuntimeError(f"online analytics authorities must use SQLite WAL: reporter={reporter_mode}, history={history_mode}")
            maximum = int(reporter.execute("SELECT COALESCE(MAX(sequence), 0) FROM frontiers").fetchone()[0])
            if maximum < 1:
                raise RuntimeError("reporter has not published a frontier")
            row = reporter.execute("SELECT completed_at FROM frontiers WHERE sequence = ?", (maximum,)).fetchone()
            first = reporter.execute("SELECT MIN(started_at) FROM frontiers WHERE sequence <= ?", (maximum,)).fetchone()[0]
            if row is None or first is None:
                raise RuntimeError("reporter frontier is incomplete")
            completed_at = str(row["completed_at"])
            placeholders = ",".join("?" for _family in GLEE_FAMILIES)
            query = f"SELECT COUNT(*) FROM games WHERE started_at IS NOT NULL AND completed_at IS NOT NULL AND rating_delta IS NOT NULL AND game_family IN ({placeholders})"
            history_games = int(history.execute(query, tuple(GLEE_FAMILIES)).fetchone()[0])
            cursor = GleeIncrementalGameLakeBuilder._history_cursor(history)
            reporter_cursor = GleeIncrementalAnalyticsLakeBuilder._reporter_cursor(reporter, frontier_sequence=maximum)
            reporter.execute("ROLLBACK")
            history.execute("ROLLBACK")
            return {"frontier_sequence": maximum, "frontier_completed_at": completed_at, "first_started_at": str(first), "history_games": history_games, "history_cursor": cursor, "history_revision_rowid": cursor["revision_rowid"], "history_revision_events": cursor["revision_events"], "reporter_cursor": reporter_cursor, "reporter_change_sequence": reporter_cursor["change_sequence"], "reporter_change_events": reporter_cursor["change_events"], "fleet_groups_sha256": _file_digest(self.fleet_groups), "reporter_journal_mode": reporter_mode, "history_journal_mode": history_mode}
        finally:
            if reporter is not None and reporter.in_transaction:
                reporter.execute("ROLLBACK")
            if history is not None and history.in_transaction:
                history.execute("ROLLBACK")
            if reporter is not None:
                reporter.close()
            if history is not None:
                history.close()

    def _due(self, source: Mapping[str, object], current: Mapping[str, object]) -> tuple[bool, str]:
        if current.get("status") != "verified":
            return True, "no-verified-parquet"
        if current.get("selection") != "current-pointer":
            return True, "repair-current-pointer"
        if current.get("release_contract") != ONLINE_INCREMENTAL_RELEASE_CONTRACT:
            return True, "upgrade-to-incremental-online-profile"
        frontier_advance = int(source["frontier_sequence"]) - int(current["frontier_sequence"])
        history_advance = int(source["history_games"]) - int(current["history_games"])
        revision_advance = int(source["history_revision_rowid"]) - int(current.get("history_revision_rowid") or 0)
        reporter_change_advance = int(source.get("reporter_change_sequence") or 0) - int(current.get("reporter_change_sequence") or 0)
        current_manifest = current.get("manifest") if isinstance(current.get("manifest"), Mapping) else {}
        current_cursor = current_manifest.get("history_cursor") if isinstance(current_manifest.get("history_cursor"), Mapping) else {}
        source_cursor = source.get("history_cursor") if isinstance(source.get("history_cursor"), Mapping) else {}
        current_reporter_cursor = current_manifest.get("reporter_cursor") if isinstance(current_manifest.get("reporter_cursor"), Mapping) else {}
        source_reporter_cursor = source.get("reporter_cursor") if isinstance(source.get("reporter_cursor"), Mapping) else {}
        if revision_advance < 0 or int(source.get("history_revision_events") or 0) < int(current_cursor.get("revision_events") or 0):
            return True, "history-revision-stream-regressed"
        if revision_advance == 0 and current_cursor.get("head_record_sha256") != source_cursor.get("head_record_sha256"):
            return True, "history-revision-head-changed"
        if reporter_change_advance < 0 or int(source.get("reporter_change_events") or 0) < int(current_reporter_cursor.get("change_events") or 0):
            return True, "reporter-change-stream-regressed"
        if reporter_change_advance == 0 and current_reporter_cursor.get("head_sha256") != source_reporter_cursor.get("head_sha256"):
            return True, "reporter-change-head-changed"
        current_source = current_manifest.get("source_frontier") if isinstance(current_manifest.get("source_frontier"), Mapping) else {}
        if source.get("fleet_groups_sha256") != current_source.get("fleet_groups_sha256"):
            return True, "fleet-dimension-changed"
        if revision_advance >= self.minimum_history_game_advance:
            return True, "history-revision-threshold"
        if frontier_advance >= self.maximum_frontier_advance:
            return True, "reporter-frontier-threshold"
        published_at = _parse_time((current.get("pointer") or {}).get("published_at") if isinstance(current.get("pointer"), Mapping) else None)
        if published_at is None:
            published_at = _parse_time((current.get("manifest") or {}).get("created_at") if isinstance(current.get("manifest"), Mapping) else None)
        pending_archives = int((current_manifest.get("inventory") or {}).get("missing_archives") or 0) if isinstance(current_manifest.get("inventory"), Mapping) else 0
        source_ahead = frontier_advance > 0 or history_advance > 0 or revision_advance > 0 or reporter_change_advance > 0
        if (source_ahead or pending_archives > 0) and (published_at is None or self.clock() - published_at >= self.maximum_age_s):
            return True, "maximum-age"
        if not source_ahead:
            return False, "source-not-ahead"
        return False, "below-refresh-thresholds"

    def _release_name(self, source: Mapping[str, object]) -> str:
        return f"frontier-{int(source['frontier_sequence'])}-games-{int(source['history_games'])}-rev-{int(source.get('history_revision_rowid') or 0)}-chg-{int(source.get('reporter_change_sequence') or 0)}"

    def _build(self, source: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
        release = self._release_name(source)
        output_dir = self.releases_root / release
        if output_dir.exists():
            try:
                existing = self.reader._validate_release(release)
                if current.get("release") != release:
                    return existing
                generation = int(((current.get("manifest") or {}).get("generation") or 0) if isinstance(current.get("manifest"), Mapping) else 0) + 1
                release = f"{release}-gen-{generation}"
                output_dir = self.releases_root / release
                if output_dir.exists():
                    return self.reader._validate_release(release)
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
                release = f"{release}-retry-{uuid.uuid4().hex[:8]}"
                output_dir = self.releases_root / release
        before = set(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
        try:
            builder = self.builder_factory(
                reporter_database=self.reporter_database,
                history_database=self.history_database,
                game_archive_root=self.game_archive_root,
                fleet_groups=self.fleet_groups,
                output_dir=output_dir,
                output_root=self.output_root,
                current=current,
                reporter_frontier=int(source["frontier_sequence"]),
                self_name=self.self_name,
                windows_s=self.windows_s,
                max_delta_segments=self.max_delta_segments,
                compaction_growth_ratio=self.compaction_growth_ratio,
            )
            builder.run()
        except Exception:
            for staging in set(output_dir.parent.glob(f".{output_dir.name}.staging-*")) - before:
                if staging.parent == output_dir.parent and staging.name.startswith(f".{output_dir.name}.staging-"):
                    shutil.rmtree(staging, ignore_errors=True)
            raise
        return self.reader._validate_release(release)

    def _publish_pointer(self, release: Mapping[str, object]) -> dict[str, object]:
        pointer = {
            "contract": ONLINE_ANALYTICS_POINTER_CONTRACT,
            "schema_version": 1,
            "release": release["release"],
            "frontier_sequence": release["frontier_sequence"],
            "history_games": release["history_games"],
            "history_revision_rowid": release.get("history_revision_rowid", 0),
            "reporter_change_sequence": release.get("reporter_change_sequence", 0),
            "manifest_sha256": release["manifest_sha256"],
            "release_contract": release["release_contract"],
            "profile": release["profile"],
            "published_at": _now(),
            "authoritative_fallback": "reporter/history SQLite WAL plus immutable terminal game archives",
        }
        previous_pointer = None
        if self.reader.pointer_path.is_file():
            try:
                loaded = json.loads(self.reader.pointer_path.read_text(encoding="utf-8"))
                previous_pointer = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                previous_pointer = None
        _atomic_json(self.reader.pointer_path, pointer)
        try:
            verified = self.reader.resolve()
            if verified["release"] != release["release"] or verified["manifest_sha256"] != release["manifest_sha256"]:
                raise RuntimeError("online analytics pointer did not reopen to the published release")
        except Exception:
            if previous_pointer is not None:
                _atomic_json(self.reader.pointer_path, previous_pointer)
            else:
                rejected = self.reader.pointer_path.with_name(f".rejected-current-{os.getpid()}-{uuid.uuid4().hex}.json")
                os.replace(self.reader.pointer_path, rejected)
                _fsync_directory(self.output_root)
            raise
        return pointer

    def _prune(self, current_release: str) -> tuple[list[str], list[str]]:
        valid: list[dict[str, object]] = []
        warnings: list[str] = []
        for manifest_path in self.releases_root.glob("*/manifest.json"):
            try:
                valid.append(self.reader._validate_release(manifest_path.parent.name, reopen=False))
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
                warnings.append(f"retention skipped invalid release {manifest_path.parent.name!r}: {type(error).__name__}: {error}")
        ordered = sorted(valid, key=lambda item: (int(item.get("reporter_change_sequence") or 0), int(item.get("history_revision_rowid") or 0), int((item.get("manifest") or {}).get("generation") or 0), int(item["frontier_sequence"]), int(item["history_games"]), str(item["release"])), reverse=True)
        keep = {str(item["release"]) for item in ordered[: self.retained_releases]} | {current_release}
        pruned: list[str] = []
        for item in ordered:
            release = str(item["release"])
            if release in keep:
                continue
            path = self.reader._release_path(release)
            if path.parent != self.releases_root.resolve() or not (path / "manifest.json").is_file():
                warnings.append(f"retention refused unsafe release path: {path}")
                continue
            try:
                shutil.rmtree(path)
                _fsync_directory(self.releases_root)
                pruned.append(release)
            except OSError as error:
                warnings.append(f"retention could not prune {release!r}: {type(error).__name__}: {error}")
        reachable_segments: set[str] = set()
        for item in ordered:
            if str(item["release"]) not in keep or item.get("release_contract") not in {ONLINE_INCREMENTAL_RELEASE_CONTRACT, ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT}:
                continue
            try:
                artifacts = (GAMES_ARTIFACT,) if item.get("release_contract") == ONLINE_INCREMENTAL_GAME_RELEASE_CONTRACT else (*ONLINE_REQUIRED_ARTIFACTS, *ONLINE_INTERNAL_ARTIFACTS)
                for artifact in artifacts:
                    for receipt in self.reader._dataset_receipts(item["manifest"], artifact):
                        reachable_segments.add(str(receipt["path"]))
            except (KeyError, RuntimeError, TypeError) as error:
                warnings.append(f"retention could not enumerate segments for {item['release']!r}: {type(error).__name__}: {error}")
        if self.reader.segments_root.is_dir():
            for path in self.reader.segments_root.glob("*.parquet"):
                relative = str(path.relative_to(self.output_root))
                if not SEGMENT_PATH_PATTERN.fullmatch(relative) or relative in reachable_segments:
                    continue
                try:
                    path.unlink()
                    pruned.append(relative)
                except OSError as error:
                    warnings.append(f"retention could not prune segment {path.name!r}: {type(error).__name__}: {error}")
            _fsync_directory(self.reader.segments_root)
        return pruned, warnings

    def cycle(self) -> dict[str, object]:
        attempted_at = _now()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.releases_root.mkdir(parents=True, exist_ok=True)
        self.reader.segments_root.mkdir(parents=True, exist_ok=True)
        current = self.reader.try_resolve(reopen=False)
        try:
            source = self._source_state()
            due, reason = self._due(source, current)
            if not due:
                status = {"contract": ONLINE_ANALYTICS_STATUS_CONTRACT, "schema_version": 1, "status": "healthy", "cycle": "skipped", "reason": reason, "attempted_at": attempted_at, "source": source, "current": {key: current.get(key) for key in ("status", "selection", "release", "release_contract", "profile", "frontier_sequence", "history_games", "history_revision_rowid", "reporter_change_sequence", "manifest_sha256")}, "operational_authority": "sqlite-wal-and-immutable-game-archives"}
                _atomic_json(self.status_path, status)
                return status
            built = self._build(source, current)
            pointer = self._publish_pointer(built)
            pruned, warnings = self._prune(str(built["release"]))
            status = {"contract": ONLINE_ANALYTICS_STATUS_CONTRACT, "schema_version": 1, "status": "healthy", "cycle": "published", "reason": reason, "attempted_at": attempted_at, "completed_at": _now(), "source": source, "current": pointer, "pruned_releases": pruned, "warnings": warnings, "operational_authority": "sqlite-wal-and-immutable-game-archives"}
            _atomic_json(self.status_path, status)
            _append_jsonl(self.events_path, {**status, "contract": ONLINE_ANALYTICS_LAKE_CONTRACT, "kind": "parquet_frontier_published"})
            return status
        except Exception as error:
            fallback = self.reader.try_resolve(reopen=False)
            status = {"contract": ONLINE_ANALYTICS_STATUS_CONTRACT, "schema_version": 1, "status": "degraded", "cycle": "failed-open-to-operational-authority", "attempted_at": attempted_at, "error": f"{type(error).__name__}: {error}", "parquet_fallback": {key: fallback.get(key) for key in ("status", "selection", "release", "release_contract", "profile", "frontier_sequence", "history_games", "history_revision_rowid", "reporter_change_sequence", "manifest_sha256", "errors")}, "operational_authority": "sqlite-wal-and-immutable-game-archives", "matchmaking_effect": "none"}
            try:
                loaded = json.loads(self.status_path.read_text(encoding="utf-8")) if self.status_path.is_file() else None
                previous = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                previous = None
            _atomic_json(self.status_path, status)
            if not isinstance(previous, Mapping) or previous.get("error") != status["error"]:
                _append_jsonl(self.events_path, {**status, "contract": ONLINE_ANALYTICS_LAKE_CONTRACT, "kind": "parquet_refresh_failed"})
            return status

    def run(self, *, max_cycles: int | None = None) -> dict[str, object]:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.releases_root.mkdir(parents=True, exist_ok=True)
        self.reader.segments_root.mkdir(parents=True, exist_ok=True)
        self.stopped_path.unlink(missing_ok=True)
        cycles = 0
        with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another online analytics publisher owns the output root") from error
            _append_jsonl(self.events_path, {"contract": ONLINE_ANALYTICS_LAKE_CONTRACT, "kind": "publisher_started", "at": _now(), "pid": os.getpid()})
            last = self.cycle()
            cycles += 1
            while (max_cycles is None or cycles < max_cycles) and not self.stop_path.is_file():
                deadline = self.clock() + self.poll_interval_s
                while self.clock() < deadline and not self.stop_path.is_file():
                    self.sleeper(min(1.0, max(0.0, deadline - self.clock())))
                if self.stop_path.is_file():
                    break
                last = self.cycle()
                cycles += 1
            stopped = {"contract": ONLINE_ANALYTICS_STATUS_CONTRACT, "schema_version": 1, "status": "stopped" if self.stop_path.is_file() else "complete", "stopped_at": _now(), "cycles": cycles, "last_cycle": last}
            _atomic_json(self.stopped_path, stopped)
            _append_jsonl(self.events_path, {"contract": ONLINE_ANALYTICS_LAKE_CONTRACT, "kind": "publisher_stopped", "at": stopped["stopped_at"], "cycles": cycles})
            return stopped
