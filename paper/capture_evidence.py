"""Capture a hash-bound, read-only paper snapshot from append-only GLEE journals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_complete_prefix(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_size = path.stat().st_size
    with path.open("rb") as stream:
        payload = stream.read(initial_size)
    final_newline = payload.rfind(b"\n")
    complete = payload[: final_newline + 1] if final_newline >= 0 else b""
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    sequences = [int(row["event_sequence"]) for row in rows if isinstance(row.get("event_sequence"), int)]
    timestamps = [str(row["ts"]) for row in rows if row.get("ts")]
    receipt = {
        "path": str(path),
        "initial_file_size_bytes": initial_size,
        "complete_prefix_size_bytes": len(complete),
        "complete_prefix_sha256": hashlib.sha256(complete).hexdigest(),
        "line_count": len(rows),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "last_event_sequence": max(sequences) if sequences else None,
    }
    return rows, receipt


def contains_model_call(value: Any) -> bool:
    if isinstance(value, dict):
        metadata = value.get("call_metadata")
        if isinstance(metadata, dict) and metadata:
            return True
        return any(contains_model_call(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_model_call(child) for child in value)
    return False


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts = Counter(str(row.get("kind")) for row in rows)
    turn_families: dict[str, str] = {}
    for row in rows:
        turn_id = row.get("turn_id")
        family = row.get("family")
        if isinstance(turn_id, str) and isinstance(family, str):
            turn_families[turn_id] = family
    decisions = [row for row in rows if row.get("kind") == "worker_finished" and isinstance(row.get("decision"), dict)]
    family_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    latencies: list[float] = []
    fallback_count = 0
    nonnull_tetrad_updates = 0
    for row in decisions:
        decision = row["decision"]
        family_counts[turn_families.get(str(row.get("turn_id")), "unknown")] += 1
        path_counts[str(decision.get("selection_branch") or decision.get("role") or "unspecified")] += 1
        route_counts["cloud-assisted" if contains_model_call(decision) else "local"] += 1
        if decision.get("fallback") is True:
            fallback_count += 1
        if decision.get("tetrad_update") is not None:
            nonnull_tetrad_updates += 1
        elapsed = decision.get("elapsed_s")
        if isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)):
            latencies.append(float(elapsed))
    submissions = [row for row in rows if row.get("kind") == "move_submitted"]
    invalid_submissions = sum(1 for row in submissions if row.get("result", {}).get("valid") is not True)
    terminal_rows = [row for row in rows if row.get("kind") in {"game_completed", "game_completed_during_opponent_turn"}]
    terminal_games = {str(row.get("game_id")) for row in terminal_rows if row.get("game_id")}
    return {
        "events_by_kind_selected": {kind: kind_counts.get(kind, 0) for kind in ("worker_finished", "move_submitted", "game_completed", "game_completed_during_opponent_turn", "move_submission_terminal_race", "move_submission_transport_ambiguous")},
        "worker_decisions": len(decisions),
        "worker_decisions_by_family": dict(sorted(family_counts.items())),
        "decision_routes": dict(sorted(route_counts.items())),
        "decision_route_fractions": {key: round(value / len(decisions), 6) for key, value in sorted(route_counts.items())} if decisions else {},
        "top_selection_branches": dict(path_counts.most_common(12)),
        "fallback_count": fallback_count,
        "fallback_fraction": round(fallback_count / len(decisions), 6) if decisions else None,
        "nonnull_tetrad_updates": nonnull_tetrad_updates,
        "decision_latency_seconds": {
            "mean": round(statistics.fmean(latencies), 6) if latencies else None,
            "median": round(statistics.median(latencies), 6) if latencies else None,
            "p95": round(quantile(latencies, 0.95), 6) if latencies else None,
            "maximum": round(max(latencies), 6) if latencies else None,
        },
        "move_submissions": len(submissions),
        "invalid_move_submissions": invalid_submissions,
        "distinct_terminal_games": len(terminal_games),
    }


def latest_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [row for row in rows if row.get("kind") == "agent_stats_observed" and isinstance(row.get("stats"), dict)]
    if not observations:
        return {"status": "unavailable"}
    latest = observations[-1]
    scores = latest["stats"].get("scores", {})
    sanitized = {
        family: {
            "games_played": int(value["games_played"]),
            "rating": float(value["rating"]),
        }
        for family, value in scores.items()
        if isinstance(value, dict) and "games_played" in value and "rating" in value
    }
    ratings = [value["rating"] for value in sanitized.values()]
    return {
        "timestamp": latest.get("ts"),
        "active_games": latest["stats"].get("active_games"),
        "families": dict(sorted(sanitized.items())),
        "arithmetic_mean_family_rating": round(statistics.fmean(ratings), 6) if ratings else None,
        "total_games_played": sum(value["games_played"] for value in sanitized.values()),
    }


def verify_evidence(path: Path) -> None:
    evidence_bytes = path.read_bytes()
    evidence = json.loads(evidence_bytes)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for source_name in ("run_source", "sensor_source"):
        source = evidence[source_name]
        source_path = Path(source["path"])
        byte_limit = int(source["complete_prefix_size_bytes"])
        with source_path.open("rb") as stream:
            payload = stream.read(byte_limit)
        digest = hashlib.sha256(payload).hexdigest()
        complete = len(payload) == byte_limit and (not payload or payload.endswith(b"\n"))
        matched = complete and digest == source["complete_prefix_sha256"]
        checks[source_name] = {
            "path": str(source_path),
            "bytes_read": len(payload),
            "expected_bytes": byte_limit,
            "actual_sha256": digest,
            "expected_sha256": source["complete_prefix_sha256"],
            "complete_newline_terminated_prefix": complete,
            "matched": matched,
        }
        if not matched:
            failures.append(source_name)
    result = {
        "contract": "glee-competition-paper-evidence-verification-v1",
        "evidence_path": str(path.resolve()),
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "status": "verified" if not failures else "failed",
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"evidence verification failed: {', '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path)
    parser.add_argument("--sensor-events", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        if args.events is not None or args.sensor_events is not None:
            parser.error("--verify cannot be combined with capture inputs")
        verify_evidence(args.verify.resolve())
        return
    if args.events is None or args.sensor_events is None:
        parser.error("capture requires both --events and --sensor-events")
    event_rows, event_receipt = read_complete_prefix(args.events.resolve())
    sensor_rows, sensor_receipt = read_complete_prefix(args.sensor_events.resolve())
    result = {
        "schema_version": 1,
        "contract": "glee-competition-paper-evidence-prefix-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "run_source": event_receipt,
        "sensor_source": sensor_receipt,
        "run_summary": summarize_run(event_rows),
        "live_endpoint": latest_scores(sensor_rows),
        "interpretation": "The sources were live append-only journals. Every statistic is bound to the complete newline-terminated prefix and SHA-256 recorded above; later appends are outside this snapshot.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
