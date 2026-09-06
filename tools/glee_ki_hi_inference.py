#!/usr/bin/env python3
"""Reproduce dependence-aware KI-versus-HI rating-delta inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


FAMILIES = ("bargaining", "negotiation", "persuasion")
TERMINAL_EVENTS = {"game_completed", "game_completed_during_opponent_turn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eligible-input", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--history-db", type=Path)
    parser.add_argument("--eligible-output", type=Path)
    parser.add_argument("--frontier", type=int, default=797)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--sensitivity-block-lengths", type=int, nargs="*", default=(10, 40))
    parser.add_argument("--repetitions", type=int, default=50_000)
    parser.add_argument("--sensitivity-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1_729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_rows(run_dir: Path, history_db: Path, frontier: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal: list[dict[str, Any]] = []
    seen: set[str] = set()
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            game_id = event.get("game_id")
            if event.get("kind") not in TERMINAL_EVENTS or not game_id or game_id in seen:
                continue
            terminal.append(event)
            seen.add(game_id)
            if len(terminal) == frontier:
                break
    if len(terminal) != frontier:
        raise ValueError(f"expected {frontier} terminal games, found {len(terminal)}")

    game_paths = {path.name.split("-", 1)[1][:-5]: path for path in (run_dir / "games").glob("*.json")}
    connection = sqlite3.connect(f"file:{history_db}?mode=ro", uri=True)
    try:
        ratings = dict(connection.execute("SELECT game_id, rating_delta FROM games WHERE rating_delta IS NOT NULL"))
    finally:
        connection.close()

    rows: list[dict[str, Any]] = []
    family_totals: Counter[str] = Counter()
    family_rated: Counter[str] = Counter()
    family_timeouts: Counter[str] = Counter()
    for index, event in enumerate(terminal, start=1):
        game_id = event["game_id"]
        game = json.loads(game_paths[game_id].read_text(encoding="utf-8"))
        family = event["family"]
        outcome = game.get("result", {}).get("outcome")
        rating_delta = ratings.get(game_id)
        family_totals[family] += 1
        if rating_delta is not None:
            family_rated[family] += 1
        if outcome == "timeout":
            family_timeouts[family] += 1
        if rating_delta is None or outcome == "timeout":
            continue
        rows.append(
            {
                "family": family,
                "frontier_index": index,
                "mode": "HI" if game["opponent"]["type"] == "hidden" else "KI",
                "rating_delta": float(rating_delta),
                "terminal_at": event["ts"],
            }
        )

    canonical_rows = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    source = {
        "events_path": str((run_dir / "events.jsonl").resolve()),
        "events_sha256": sha256(run_dir / "events.jsonl"),
        "first_terminal_at": terminal[0]["ts"],
        "last_terminal_at": terminal[-1]["ts"],
        "frontier_terminal_games": len(terminal),
        "family_terminal_games": dict(sorted(family_totals.items())),
        "family_authenticated_ratings": dict(sorted(family_rated.items())),
        "family_timeout_outcomes": dict(sorted(family_timeouts.items())),
        "eligible_record_count": len(rows),
        "eligible_records_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "history_db_path": str(history_db.resolve()),
    }
    return rows, source


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "family": row["family"],
            "frontier_index": row["frontier_index"],
            "mode": row["mode"],
            "rating_delta": row["rating_delta"],
        }
        for row in rows
    ]


def write_eligible_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in public_rows(rows))
    path.write_text(payload, encoding="utf-8")


def load_eligible_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_fields = {"family", "frontier_index", "mode", "rating_delta"}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if set(row) != expected_fields:
                raise ValueError(f"eligible row {line_number} has fields {sorted(row)}")
            if row["family"] not in FAMILIES or row["mode"] not in {"KI", "HI"}:
                raise ValueError(f"eligible row {line_number} has invalid family or mode")
            rows.append(row)
    if not rows or len({row["frontier_index"] for row in rows}) != len(rows):
        raise ValueError("eligible input must contain uniquely indexed rows")
    return rows, {
        "eligible_input_path": path.as_posix(),
        "eligible_input_sha256": sha256(path),
        "eligible_record_count": len(rows),
    }


def contrast(rows: list[dict[str, Any]], value_key: str = "rating_delta") -> float:
    hi = [row[value_key] for row in rows if row["mode"] == "HI"]
    ki = [row[value_key] for row in rows if row["mode"] == "KI"]
    if not hi or not ki:
        raise ValueError("bootstrap sample lacks one identity mode")
    return fmean(hi) - fmean(ki)


def bootstrap_family(rows: list[dict[str, Any]], block_length: int, repetitions: int, seed: int) -> dict[str, Any]:
    observed = contrast(rows)
    mode_means = {mode: fmean(row["rating_delta"] for row in rows if row["mode"] == mode) for mode in ("KI", "HI")}
    overall = fmean(row["rating_delta"] for row in rows)
    centered = [{**row, "null_delta": row["rating_delta"] - mode_means[row["mode"]] + overall} for row in rows]
    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    null_differences: list[float] = []
    for _ in range(repetitions):
        starts: list[int] = []
        remaining = len(rows)
        while remaining > 0:
            starts.append(rng.randrange(len(rows)))
            remaining -= min(block_length, remaining)
        sampled: list[dict[str, Any]] = []
        sampled_null: list[dict[str, Any]] = []
        remaining = len(rows)
        for start in starts:
            take = min(block_length, remaining)
            sampled.extend(rows[(start + offset) % len(rows)] for offset in range(take))
            sampled_null.extend(centered[(start + offset) % len(rows)] for offset in range(take))
            remaining -= take
        bootstrap_differences.append(contrast(sampled))
        null_differences.append(contrast(sampled_null, "null_delta"))
    unadjusted = [quantile(bootstrap_differences, 0.025), quantile(bootstrap_differences, 0.975)]
    familywise_alpha = 0.05 / len(FAMILIES)
    simultaneous = [quantile(bootstrap_differences, familywise_alpha / 2.0), quantile(bootstrap_differences, 1.0 - familywise_alpha / 2.0)]
    exceedances = sum(abs(value) >= abs(observed) for value in null_differences)
    p_value = (exceedances + 1.0) / (repetitions + 1.0)
    return {
        "block_length_eligible_games": block_length,
        "bootstrap_repetitions": repetitions,
        "difference_hi_minus_ki": observed,
        "familywise_95_percent_bonferroni_interval": simultaneous,
        "ki_count": sum(row["mode"] == "KI" for row in rows),
        "ki_mean": mode_means["KI"],
        "hi_count": sum(row["mode"] == "HI" for row in rows),
        "hi_mean": mode_means["HI"],
        "null_centered_two_sided_p": p_value,
        "unadjusted_95_percent_interval": unadjusted,
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (family, value) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        adjusted[family] = min(1.0, running)
    return adjusted


def main() -> None:
    args = parse_args()
    if args.eligible_input is not None:
        if args.history_db is not None or args.eligible_output is not None:
            raise ValueError("--eligible-input cannot be combined with --history-db or --eligible-output")
        rows, source = load_eligible_rows(args.eligible_input)
    else:
        if args.history_db is None:
            raise ValueError("--history-db is required with --run-dir")
        rows, source = load_rows(args.run_dir, args.history_db, args.frontier)
        if args.eligible_output is not None:
            write_eligible_rows(args.eligible_output, rows)
    grouped = {family: [row for row in rows if row["family"] == family] for family in FAMILIES}
    primary = {family: bootstrap_family(grouped[family], args.block_length, args.repetitions, args.seed + index * 10_000) for index, family in enumerate(FAMILIES)}
    adjusted = holm_adjust({family: result["null_centered_two_sided_p"] for family, result in primary.items()})
    for family in FAMILIES:
        primary[family]["holm_adjusted_p"] = adjusted[family]
        interval = primary[family]["familywise_95_percent_bonferroni_interval"]
        primary[family]["familywise_nonzero"] = interval[0] > 0.0 or interval[1] < 0.0
    sensitivity: dict[str, Any] = {}
    for block_length in args.sensitivity_block_lengths:
        sensitivity[str(block_length)] = {
            family: bootstrap_family(grouped[family], block_length, args.sensitivity_repetitions, args.seed + block_length * 1_000 + index * 10_000)
            for index, family in enumerate(FAMILIES)
        }
    result = {
        "contract": "glee-ki-hi-moving-block-bootstrap-v1",
        "estimand": "mean authenticated rating delta in HI games minus mean authenticated rating delta in KI games, by family, excluding timeout outcomes",
        "inference": {
            "bootstrap": "circular moving-block bootstrap over each family chronological eligible-game stream",
            "multiplicity": "Bonferroni 98.333% per-family intervals and Holm-adjusted null-centered bootstrap p-values control the 3-family family-wise error rate at 0.05",
            "primary_block_length_eligible_games": args.block_length,
            "primary_repetitions": args.repetitions,
            "seed": args.seed,
            "sensitivity_block_lengths_eligible_games": list(args.sensitivity_block_lengths),
            "sensitivity_repetitions": args.sensitivity_repetitions,
        },
        "limitations": "The intervals address sampling uncertainty under local temporal dependence but do not identify a causal SIC effect because identity mode was not randomized and opponent, role, configuration, and policy composition varied.",
        "primary": primary,
        "sensitivity": sensitivity,
        "source": source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
