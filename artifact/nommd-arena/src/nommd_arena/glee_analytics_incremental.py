"""Pure schemas and transformations for the incremental GLEE analytics lake."""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from .glee_analytics_lake import _normalize_name_expr, _parse_utc_expr


PUBLIC_UPDATES_ARTIFACT = "public-updates.parquet"
PUBLIC_ACTIVITY_ARTIFACT = "public-activity.parquet"
PUBLIC_STATE_ARTIFACT = "public-activity-state.parquet"

ONLINE_PUBLIC_UPDATE_SCHEMA: dict[str, pl.DataType] = {
    "frontier_sequence": pl.Int64,
    "change_sequence": pl.Int64,
    "family": pl.String,
    "player_id": pl.String,
    "change_kind": pl.String,
    "observed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "games_raw": pl.Int64,
    "player_name": pl.String,
    "rating": pl.Float64,
    "games": pl.Int64,
    "is_baseline": pl.Boolean,
    "is_benchmark": pl.Boolean,
    "present": pl.Boolean,
    "player_name_normalized": pl.String,
}

ONLINE_PUBLIC_ACTIVITY_SCHEMA: dict[str, pl.DataType] = {
    "event_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "frontier_sequence": pl.Int64,
    "change_sequence": pl.Int64,
    "family": pl.String,
    "player_id": pl.String,
    "player_name": pl.String,
    "games_delta": pl.Int64,
    "fleet_key": pl.String,
    "fleet_confidence": pl.String,
}

ONLINE_PUBLIC_STATE_SCHEMA: dict[str, pl.DataType] = {
    "family": pl.String,
    "player_id": pl.String,
    "player_name": pl.String,
    "rating": pl.Float64,
    "games": pl.Int64,
    "is_baseline": pl.Boolean,
    "is_benchmark": pl.Boolean,
    "present": pl.Boolean,
    "player_name_normalized": pl.String,
    "baseline_games": pl.Int64,
    "high_water_games": pl.Int64,
    "effective_additions": pl.Int64,
    "materialized_additions": pl.Int64,
    "last_change_sequence": pl.Int64,
}

_RAW_UPDATE_SCHEMA: dict[str, pl.DataType] = {
    "frontier_sequence": pl.Int64,
    "change_sequence": pl.Int64,
    "family": pl.String,
    "player_id": pl.String,
    "change_kind": pl.String,
    "observed_at": pl.String,
    "games_raw": pl.Int64,
    "player_name_raw": pl.String,
    "rating_raw": pl.Float64,
    "baseline_raw": pl.Int64,
    "benchmark_raw": pl.Int64,
}


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def normalize_frame(frame: pl.DataFrame, schema: Mapping[str, pl.DataType], *, label: str) -> pl.DataFrame:
    columns = tuple(schema)
    missing = [column for column in columns if column not in frame.columns]
    extra = [column for column in frame.columns if column not in columns]
    if missing or extra:
        raise RuntimeError(f"{label} schema differs: missing={missing}, extra={extra}")
    return frame.select(pl.col(column).cast(dtype) for column, dtype in schema.items())


def reporter_change_rows(connection: Any, *, after_change_sequence: int, through_change_sequence: int) -> pl.DataFrame:
    """Read one contiguous append-only reporter interval with every public row change preserved."""
    if through_change_sequence <= after_change_sequence:
        return empty_frame(_RAW_UPDATE_SCHEMA)
    query = """
        SELECT ch.frontier_sequence, ch.change_sequence, ch.family, ch.player_id, ch.change_kind,
               ch.observed_by AS observed_at,
               CAST(json_extract(rv.row_json, '$.games_played') AS INTEGER) AS games_raw,
               CAST(json_extract(rv.row_json, '$.player_name') AS TEXT) AS player_name_raw,
               CAST(json_extract(rv.row_json, '$.rating') AS REAL) AS rating_raw,
               CAST(json_extract(rv.row_json, '$.is_baseline') AS INTEGER) AS baseline_raw,
               CAST(json_extract(rv.row_json, '$.is_benchmark') AS INTEGER) AS benchmark_raw
        FROM changes AS ch
        JOIN row_versions AS rv
          ON rv.sequence = ch.frontier_sequence
         AND rv.family = ch.family
         AND rv.player_id = ch.player_id
        WHERE ch.change_sequence > ?
          AND ch.change_sequence <= ?
        ORDER BY ch.change_sequence
    """
    frame = pl.read_database(
        query,
        connection,
        execute_options={"parameters": (after_change_sequence, through_change_sequence)},
        schema_overrides=_RAW_UPDATE_SCHEMA,
        infer_schema_length=None,
    )
    return normalize_frame(frame, _RAW_UPDATE_SCHEMA, label="raw public update")


def derive_public_increment(
    raw: pl.DataFrame,
    prior_state: pl.DataFrame,
    fleets: pl.DataFrame,
    *,
    self_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Normalize one reporter interval, derive new high-water activity, and return a complete continuation state."""
    prior = normalize_frame(prior_state, ONLINE_PUBLIC_STATE_SCHEMA, label="public activity state")
    fleet_dimension = fleets.select("player_id", "fleet_key", pl.col("confidence").alias("fleet_confidence")) if {"player_id", "fleet_key", "confidence"}.issubset(fleets.columns) else pl.DataFrame(schema={"player_id": pl.String, "fleet_key": pl.String, "fleet_confidence": pl.String})
    if raw.is_empty():
        return empty_frame(ONLINE_PUBLIC_UPDATE_SCHEMA), empty_frame(ONLINE_PUBLIC_ACTIVITY_SCHEMA), prior, {
            "source_rows": 0,
            "effective_event_rows": 0,
            "effective_game_additions": 0,
            "materialized_event_rows": 0,
            "materialized_game_additions": 0,
            "player_family_states": prior.height,
        }
    keys = ["family", "player_id"]
    prior_columns = {
        "player_name": "_prior_player_name",
        "rating": "_prior_rating",
        "games": "_prior_games",
        "is_baseline": "_prior_is_baseline",
        "is_benchmark": "_prior_is_benchmark",
        "player_name_normalized": "_prior_player_name_normalized",
        "baseline_games": "_prior_baseline_games",
        "high_water_games": "_prior_high_water_games",
        "effective_additions": "_prior_effective_additions",
        "materialized_additions": "_prior_materialized_additions",
    }
    prior_join = prior.select(keys + [pl.col(source).alias(target) for source, target in prior_columns.items()])
    work = (
        raw.sort([*keys, "change_sequence"])
        .join(prior_join, on=keys, how="left")
        .with_columns(
            _parse_utc_expr("observed_at"),
            pl.col("player_name_raw").forward_fill().over(keys).fill_null(pl.col("_prior_player_name")).alias("player_name"),
            pl.col("rating_raw").forward_fill().over(keys).fill_null(pl.col("_prior_rating")).cast(pl.Float64).alias("rating"),
            pl.col("games_raw").forward_fill().over(keys).fill_null(pl.col("_prior_games")).cast(pl.Int64).alias("games"),
            pl.col("baseline_raw").forward_fill().over(keys).cast(pl.Boolean).fill_null(pl.col("_prior_is_baseline")).fill_null(False).alias("is_baseline"),
            pl.col("benchmark_raw").forward_fill().over(keys).cast(pl.Boolean).fill_null(pl.col("_prior_is_benchmark")).fill_null(False).alias("is_benchmark"),
            (pl.col("change_kind") != "disappeared").alias("present"),
            pl.col("games_raw").cum_max().over(keys).alias("_batch_running_high_water"),
        )
        .with_columns(
            pl.col("_batch_running_high_water").forward_fill().over(keys),
            _normalize_name_expr("player_name").fill_null(pl.col("_prior_player_name_normalized")).alias("player_name_normalized"),
        )
        .with_columns(pl.max_horizontal("_batch_running_high_water", "_prior_high_water_games").alias("_running_high_water"))
        .with_columns(pl.col("_running_high_water").shift(1).over(keys).fill_null(pl.col("_prior_high_water_games")).alias("_previous_high_water"))
        .with_columns(
            pl.when(pl.col("games_raw").is_not_null() & pl.col("_previous_high_water").is_not_null() & (pl.col("games_raw") > pl.col("_previous_high_water")))
            .then(pl.col("games_raw") - pl.col("_previous_high_water"))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("_effective_delta")
        )
    )
    normalized_self = " ".join(self_name.split()).strip().casefold()
    work = work.with_columns(
        pl.when(
            (pl.col("_effective_delta") > 0)
            & ~pl.col("is_baseline")
            & ~pl.col("is_benchmark")
            & (pl.col("player_name_normalized").fill_null("") != normalized_self)
        )
        .then(pl.col("_effective_delta"))
        .otherwise(0)
        .cast(pl.Int64)
        .alias("_materialized_delta")
    )
    updates = normalize_frame(work.select(*ONLINE_PUBLIC_UPDATE_SCHEMA), ONLINE_PUBLIC_UPDATE_SCHEMA, label="public update").sort("change_sequence")
    activity = (
        work.filter(pl.col("_materialized_delta") > 0)
        .select(
            pl.col("observed_at").alias("event_at"),
            "frontier_sequence",
            "change_sequence",
            "family",
            "player_id",
            "player_name",
            pl.col("_materialized_delta").alias("games_delta"),
        )
        .join(fleet_dimension, on="player_id", how="left")
    )
    activity = normalize_frame(activity, ONLINE_PUBLIC_ACTIVITY_SCHEMA, label="public activity").sort("change_sequence")
    aggregated = (
        work.group_by(keys)
        .agg(
            pl.col("player_name").last(),
            pl.col("rating").last(),
            pl.col("games").last(),
            pl.col("is_baseline").last(),
            pl.col("is_benchmark").last(),
            pl.col("present").last(),
            pl.col("player_name_normalized").last(),
            pl.col("games_raw").drop_nulls().first().alias("_batch_first_games"),
            pl.col("games_raw").max().alias("_batch_max_games"),
            pl.col("_effective_delta").sum().alias("_batch_effective_additions"),
            pl.col("_materialized_delta").sum().alias("_batch_materialized_additions"),
            pl.col("change_sequence").max().alias("last_change_sequence"),
        )
        .join(prior, on=keys, how="left", suffix="_prior")
    )
    state_rows: list[dict[str, object]] = []
    for row in aggregated.iter_rows(named=True):
        baseline = row.get("baseline_games")
        if baseline is None:
            baseline = row.get("_batch_first_games")
        prior_high_water = row.get("high_water_games")
        batch_high_water = row.get("_batch_max_games")
        high_water_values = [int(value) for value in (prior_high_water, batch_high_water) if value is not None]
        high_water = max(high_water_values) if high_water_values else None
        effective_additions = int(high_water - baseline) if high_water is not None and baseline is not None else 0
        expected_effective = int(row.get("effective_additions") or 0) + int(row.get("_batch_effective_additions") or 0)
        if effective_additions != expected_effective:
            raise RuntimeError(f"public activity high-water continuation differs for {row['family']}/{row['player_id']}: {effective_additions} != {expected_effective}")
        state_rows.append(
            {
                "family": row["family"],
                "player_id": row["player_id"],
                "player_name": row.get("player_name"),
                "rating": row.get("rating"),
                "games": row.get("games"),
                "is_baseline": bool(row.get("is_baseline")),
                "is_benchmark": bool(row.get("is_benchmark")),
                "present": bool(row.get("present")),
                "player_name_normalized": row.get("player_name_normalized"),
                "baseline_games": baseline,
                "high_water_games": high_water,
                "effective_additions": effective_additions,
                "materialized_additions": int(row.get("materialized_additions") or 0) + int(row.get("_batch_materialized_additions") or 0),
                "last_change_sequence": int(row["last_change_sequence"]),
            }
        )
    touched = pl.DataFrame(state_rows, schema=ONLINE_PUBLIC_STATE_SCHEMA)
    untouched = prior.join(touched.select(keys), on=keys, how="anti") if prior.height else prior
    state = normalize_frame(pl.concat([untouched, touched], how="vertical"), ONLINE_PUBLIC_STATE_SCHEMA, label="public activity state").sort(keys)
    if state.filter(pl.col("materialized_additions") > pl.col("effective_additions")).height:
        raise RuntimeError("materialized public activity exceeds monotonic high-water additions")
    receipt = {
        "source_rows": updates.height,
        "effective_event_rows": int(work.filter(pl.col("_effective_delta") > 0).height),
        "effective_game_additions": int(work["_effective_delta"].sum() or 0),
        "materialized_event_rows": activity.height,
        "materialized_game_additions": int(activity["games_delta"].sum() or 0),
        "player_family_states": state.height,
        "cumulative_effective_game_additions": int(state["effective_additions"].sum() or 0),
        "cumulative_materialized_game_additions": int(state["materialized_additions"].sum() or 0),
    }
    return updates, activity, state, receipt
