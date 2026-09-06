#!/usr/bin/env python3
"""Recompute the paper's frozen metrics from deidentified sufficient data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("bargaining", "negotiation", "persuasion")
CONDITIONAL_ARMS = ("convex_stack", "sequence_ensemble", "engineered_ensemble")
SELF_MIRROR_SEEDS = (1729, 2718)
DOSSIER_BOOTSTRAP_REPLICATES = 5_000
DOSSIER_BOOTSTRAP_SEED = 20_260_810
PERSUASION_BOOTSTRAP_REPLICATES = 2_000
PERSUASION_BOOTSTRAP_SEED = 72_014
GROUP_PATTERNS = {
    "conditional_twin": {"game_group": re.compile(r"ctg\d{6}")},
    "persuasion_continuation": {"game_group": re.compile(r"pcg\d{6}")},
    "self_mirror": {"game_group": re.compile(r"smg\d{6}")},
    "dossier_comparison": {
        "opponent_group": re.compile(r"dc-o\d{6}"),
        "game_group": re.compile(r"dc-g\d{6}"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "reproduction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(value: float | None) -> float | None:
    return round(value, 12) if value is not None else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def probabilities(row: Mapping[str, Any], prefix: str) -> list[float]:
    vector = row.get(f"{prefix}_probabilities")
    if vector is not None:
        result = [float(value) for value in vector]
    else:
        values = [row.get(f"{prefix}_p{index}") for index in range(3)]
        result = [float(value) for value in values if value is not None]
    if len(result) < 2 or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"invalid probability vector for {prefix}")
    if not math.isclose(sum(result), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"probability vector for {prefix} does not sum to one")
    return result


def classification_summary(rows: Sequence[Mapping[str, Any]], prefix: str, *, family_balanced: bool = False) -> dict[str, Any]:
    losses: list[float] = []
    correct = 0
    family_losses: defaultdict[str, list[float]] = defaultdict(list)
    family_correct: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        values = probabilities(row, prefix)
        actual = int(row["actual_action"])
        if actual < 0 or actual >= len(values):
            raise ValueError(f"actual action is outside the {prefix} probability vector")
        loss = -math.log(max(values[actual], 1e-12))
        selected = max(range(len(values)), key=values.__getitem__)
        losses.append(loss)
        correct += int(selected == actual)
        if family_balanced:
            family = str(row["family"])
            family_losses[family].append(loss)
            family_correct[family].append(int(selected == actual))
    result: dict[str, Any] = {
        "rows": len(rows),
        "accuracy": rounded(correct / len(rows)),
        "negative_log_likelihood": rounded(mean(losses)),
    }
    if family_balanced:
        if set(family_losses) != set(FAMILIES):
            raise ValueError("conditional-twin rows do not cover all families")
        by_family = {
            family: {
                "rows": len(family_losses[family]),
                "negative_log_likelihood": rounded(mean(family_losses[family])),
                "accuracy": rounded(mean(family_correct[family])),
            }
            for family in FAMILIES
        }
        result["by_family"] = by_family
        result["equal_family_negative_log_likelihood"] = rounded(mean([float(by_family[family]["negative_log_likelihood"]) for family in FAMILIES]))
    return result


def conditional_twin_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts()
    return {
        "rows": frame.height,
        "games": frame["game_group"].n_unique(),
        "arms": {arm: classification_summary(rows, arm, family_balanced=True) for arm in CONDITIONAL_ARMS},
        "selection_objective": "validation equal-family negative log likelihood",
    }


def continuation_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    losses: defaultdict[str, list[float]] = defaultdict(list)
    recalls: defaultdict[int, list[int]] = defaultdict(list)
    correct = 0
    total_loss = 0.0
    for row in rows:
        values = probabilities(row, prefix)
        actual = int(row["actual_action"])
        selected = max(range(len(values)), key=values.__getitem__)
        loss = -math.log(max(values[actual], 1e-12))
        total_loss += loss
        correct += int(selected == actual)
        recalls[actual].append(int(selected == actual))
        losses[str(row["game_group"])].append(loss)
    game_means = [sum(values) / len(values) for values in losses.values()]
    return {
        "rows": len(rows),
        "games": len(game_means),
        "negative_log_likelihood": rounded(total_loss / len(rows)),
        "game_macro_negative_log_likelihood": rounded(mean(game_means)),
        "accuracy": rounded(correct / len(rows)),
        "balanced_accuracy": rounded(mean([sum(values) / len(values) for values in recalls.values()])),
    }


def persuasion_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_game: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        actual = int(row["actual_action"])
        selected = probabilities(row, "selected")
        baseline = probabilities(row, "markov")
        delta = -math.log(max(selected[actual], 1e-12)) + math.log(max(baseline[actual], 1e-12))
        by_game[str(row["game_group"])].append(delta)
    game_deltas = [sum(values) / len(values) for values in by_game.values()]
    observed = mean(game_deltas)
    generator = random.Random(PERSUASION_BOOTSTRAP_SEED)
    draws = sorted(mean([generator.choice(game_deltas) for _index in game_deltas]) for _replicate in range(PERSUASION_BOOTSTRAP_REPLICATES))
    return {
        "games": len(game_deltas),
        "selected_minus_markov_game_macro_negative_log_likelihood": rounded(observed),
        "bootstrap_replicates": PERSUASION_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": PERSUASION_BOOTSTRAP_SEED,
        "bootstrap_95_percent_interval": [
            rounded(draws[int(0.025 * PERSUASION_BOOTSTRAP_REPLICATES)]),
            rounded(draws[min(PERSUASION_BOOTSTRAP_REPLICATES - 1, int(0.975 * PERSUASION_BOOTSTRAP_REPLICATES))]),
        ],
    }


def persuasion_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts()
    return {
        "selected": continuation_summary(rows, "selected"),
        "preceding_signal_and_action_markov_baseline": continuation_summary(rows, "markov"),
        "paired_game_bootstrap": persuasion_bootstrap(rows),
    }


def self_mirror_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts()
    action_rows = [row for row in rows if bool(row["action_scored"])]
    value_rows = [row for row in rows if row["actual_value"] is not None]
    seeds: dict[str, Any] = {}
    for seed in SELF_MIRROR_SEEDS:
        prefix = f"seed{seed}"
        summary = classification_summary(action_rows, prefix)
        errors = [abs(float(row[f"{prefix}_predicted_value"]) - float(row["actual_value"])) for row in value_rows]
        seeds[str(seed)] = {
            "action_rows": len(action_rows),
            "action_accuracy": summary["accuracy"],
            "proposal_rows": len(value_rows),
            "proposal_coordinate_mean_absolute_error": rounded(mean(errors)),
        }
    return {
        "targets": frame.height,
        "action_scored_games": pl.DataFrame(action_rows)["game_group"].n_unique(),
        "baseline_status": "no simple baseline was frozen; metrics are descriptive",
        "seeds": seeds,
    }


def rating_model_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILIES:
        rows = frame.filter(pl.col("family") == family).to_dicts()
        result[family] = {
            "held_out_rows": len(rows),
            "held_out_mean_absolute_error": rounded(mean([abs(float(row["actual_rating_delta"]) - float(row["v3_prediction"])) for row in rows])),
            "zero_baseline_held_out_mean_absolute_error": rounded(mean([abs(float(row["actual_rating_delta"])) for row in rows])),
        }
    return {"families": result}


def safe_probability(value: float) -> float:
    return min(1 - 1e-9, max(1e-9, value))


def normal_nll(observed: float, center: float, sigma: float) -> float:
    residual = (observed - center) / sigma
    return 0.5 * residual * residual + math.log(sigma * math.sqrt(2 * math.pi))


def dossier_scored_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        action_type = str(row["action_type"])
        scored: dict[str, Any] = {
            "opponent_group": str(row["opponent_group"]),
            "game_group": str(row["game_group"]),
            "action_type": action_type,
            "predictions": {},
        }
        for model in ("hierarchical-program", "direct-context", "prose-dossier"):
            if not bool(row[f"{model}-available"]):
                continue
            if action_type == "response":
                probability = safe_probability(float(row[f"{model}-response-probability"]))
                actual = bool(row["actual_response"])
                scored["predictions"][model] = {
                    "nll": -math.log(probability if actual else 1 - probability),
                    "brier": (probability - float(actual)) ** 2,
                }
            else:
                actual = float(row["actual_proposal"])
                center = float(row[f"{model}-proposal-mean"])
                sigma = float(row[f"{model}-proposal-sigma"])
                scored["predictions"][model] = {
                    "nll": normal_nll(actual, center, sigma),
                    "absolute_error": abs(actual - center),
                    "squared_error": (actual - center) ** 2,
                }
        output.append(scored)
    return output


def dossier_task_metrics(records: Sequence[Mapping[str, Any]], model: str) -> dict[str, dict[str, float | int | None]]:
    response = [record["predictions"][model] for record in records if record["action_type"] == "response" and model in record["predictions"]]
    proposal = [record["predictions"][model] for record in records if record["action_type"] == "proposal" and model in record["predictions"]]
    return {
        "response": {
            "count": len(response),
            "nll": mean([float(value["nll"]) for value in response]),
            "brier": mean([float(value["brier"]) for value in response]),
        },
        "proposal": {
            "count": len(proposal),
            "nll": mean([float(value["nll"]) for value in proposal]),
            "mae": mean([float(value["absolute_error"]) for value in proposal]),
            "rmse": math.sqrt(sum(float(value["squared_error"]) for value in proposal) / len(proposal)) if proposal else None,
        },
    }


def dossier_macro(records: Sequence[Mapping[str, Any]], model: str) -> dict[str, dict[str, float | int | None]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["opponent_group"])].append(record)
    blocks = [dossier_task_metrics(values, model) for values in grouped.values()]
    result: dict[str, dict[str, float | int | None]] = {}
    for task, names in (("response", ("nll", "brier")), ("proposal", ("nll", "mae", "rmse"))):
        result[task] = {"opponent_count": sum(int(block[task]["count"]) > 0 for block in blocks)}
        for name in names:
            values = [float(block[task][name]) for block in blocks if block[task][name] is not None]
            result[task][name] = mean(values)
    return result


def dossier_comparison(records: Sequence[Mapping[str, Any]], left: str, right: str, *, macro: bool) -> dict[str, float | None]:
    paired = [record for record in records if left in record["predictions"] and right in record["predictions"]]
    left_metrics = dossier_macro(paired, left) if macro else dossier_task_metrics(paired, left)
    right_metrics = dossier_macro(paired, right) if macro else dossier_task_metrics(paired, right)
    output: dict[str, float | None] = {}
    for task, names in (("response", ("nll", "brier")), ("proposal", ("nll", "mae", "rmse"))):
        for name in names:
            left_value = left_metrics[task][name]
            right_value = right_metrics[task][name]
            output[f"{task}_{name}_delta"] = float(left_value) - float(right_value) if left_value is not None and right_value is not None else None
    return output


def dossier_bootstrap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generator = random.Random(DOSSIER_BOOTSTRAP_SEED)
    grouped: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["opponent_group"])][str(record["game_group"])].append(record)
    opponent_ids = sorted(grouped)
    metric_names = ("response_nll_delta", "response_brier_delta", "proposal_nll_delta", "proposal_mae_delta", "proposal_rmse_delta")
    distributions: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(DOSSIER_BOOTSTRAP_REPLICATES):
        sampled_by_slot: list[list[Mapping[str, Any]]] = []
        for _slot in opponent_ids:
            opponent_id = generator.choice(opponent_ids)
            games = list(grouped[opponent_id].values())
            slot_records: list[Mapping[str, Any]] = []
            for _game in games:
                slot_records.extend(generator.choice(games))
            sampled_by_slot.append(slot_records)
        slot_values: defaultdict[str, list[float]] = defaultdict(list)
        for slot_records in sampled_by_slot:
            values = dossier_comparison(slot_records, "direct-context", "prose-dossier", macro=False)
            for metric, value in values.items():
                if value is not None:
                    slot_values[metric].append(float(value))
        for metric in metric_names:
            value = mean(slot_values[metric])
            if value is not None:
                distributions[metric].append(value)
    point = dossier_comparison(records, "direct-context", "prose-dossier", macro=True)
    output: dict[str, Any] = {}
    for metric in metric_names:
        values = distributions[metric]
        output[metric] = {
            "point": rounded(point[metric]),
            "lower_95": rounded(percentile(values, 0.025)),
            "upper_95": rounded(percentile(values, 0.975)),
            "interval_crosses_zero": bool(percentile(values, 0.025) <= 0.0 <= percentile(values, 0.975)),
            "finite_replicates": len(values),
        }
    return {
        "method": "paired 2-stage opponent-and-complete-game cluster bootstrap",
        "replicates": DOSSIER_BOOTSTRAP_REPLICATES,
        "seed": DOSSIER_BOOTSTRAP_SEED,
        "opponent_clusters": len(opponent_ids),
        "direct_context_minus_prose_dossier": output,
    }


def successful_call_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    prefix = arm.replace("_", "-")
    selected = [row for row in rows if bool(row[f"{prefix}-call-ok"])]
    return {
        "successful_calls": len(selected),
        "mean_elapsed_seconds": rounded(mean([float(row[f"{prefix}-elapsed-s"]) for row in selected])),
        "mean_reasoning_tokens": rounded(mean([float(row[f"{prefix}-reasoning-tokens"]) for row in selected])),
        "mean_input_tokens": rounded(mean([float(row[f"{prefix}-input-tokens"]) for row in selected])),
    }


def dossier_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts()
    direct = successful_call_summary(rows, "direct_context")
    dossier = successful_call_summary(rows, "prose_dossier")
    return {
        "decisions": frame.height,
        "requested_calls": frame.height * 2,
        "successful_calls": int(direct["successful_calls"]) + int(dossier["successful_calls"]),
        "failed_calls": frame.height * 2 - int(direct["successful_calls"]) - int(dossier["successful_calls"]),
        "opponent_clusters": frame["opponent_group"].n_unique(),
        "direct_context": direct,
        "prose_dossier": dossier,
        "dossier_minus_direct": {
            "mean_elapsed_seconds": rounded(float(dossier["mean_elapsed_seconds"]) - float(direct["mean_elapsed_seconds"])),
            "mean_reasoning_tokens": rounded(float(dossier["mean_reasoning_tokens"]) - float(direct["mean_reasoning_tokens"])),
            "mean_input_tokens": rounded(float(dossier["mean_input_tokens"]) - float(direct["mean_input_tokens"])),
        },
        "uncertainty": dossier_bootstrap(dossier_scored_rows(frame)),
    }


def nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def execution_decision_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["elapsed_s"]) for row in rows]
    route_counts = {route: sum(str(row["route"]) == route for row in rows) for route in ("local", "cloud-assisted")}
    return {
        "worker_decisions": len(rows),
        "decision_routes": route_counts,
        "fallback_count": sum(bool(row["fallback"]) for row in rows),
        "fallback_fraction": rounded(sum(bool(row["fallback"]) for row in rows) / len(rows)),
        "decision_latency_seconds": {
            "mean": rounded(statistics.fmean(latencies)),
            "median": rounded(statistics.median(latencies)),
            "p95": rounded(nearest_rank(latencies, 0.95)),
            "maximum": rounded(max(latencies)),
        },
    }


def execution_metrics(decisions: pl.DataFrame, submissions: pl.DataFrame, terminals: pl.DataFrame) -> dict[str, Any]:
    all_decisions = decisions.to_dicts()
    final_decisions = decisions.filter(pl.col("in_final_v108_prefix")).to_dicts()
    all_submissions = submissions.to_dicts()
    final_submissions = submissions.filter(pl.col("in_final_v108_prefix")).to_dicts()
    return {
        "all_history": {
            **execution_decision_summary(all_decisions),
            "move_submissions": len(all_submissions),
            "invalid_move_submissions": sum(row["valid"] is not True for row in all_submissions),
            "distinct_terminal_games": terminals.height,
        },
        "final_v108_prefix": {
            **execution_decision_summary(final_decisions),
            "move_submissions": len(final_submissions),
            "invalid_move_submissions": sum(row["valid"] is not True for row in final_submissions),
            "distinct_terminal_games": terminals.filter(pl.col("in_final_v108_prefix")).height,
        },
    }


def load_frames(data_dir: Path) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    filenames = {
        "conditional_twin": "conditional-twin-test.parquet",
        "persuasion_continuation": "persuasion-continuation-test.parquet",
        "self_mirror": "self-mirror-test.parquet",
        "rating_model": "rating-model-heldout.parquet",
        "dossier_comparison": "dossier-comparison.parquet",
        "execution_decisions": "execution-decisions.parquet",
        "execution_submissions": "execution-submissions.parquet",
        "execution_terminal_games": "execution-terminal-games.parquet",
    }
    frames: dict[str, pl.DataFrame] = {}
    hashes: dict[str, str] = {}
    for name, filename in filenames.items():
        path = data_dir / filename
        frames[name] = pl.read_parquet(path)
        hashes[filename] = sha256(path)
    validate_public_schema(frames)
    return frames, hashes


def validate_public_schema(frames: Mapping[str, pl.DataFrame]) -> None:
    expected_columns = {
        "conditional_twin": {"game_group", "family", "actual_action", *(f"{arm}_p{index}" for arm in CONDITIONAL_ARMS for index in range(3))},
        "persuasion_continuation": {"game_group", "actual_action", *(f"{arm}_p{index}" for arm in ("selected", "markov") for index in range(3))},
        "self_mirror": {"game_group", "family", "target_kind", "action_scored", "actual_action", "actual_value", *(field for seed in SELF_MIRROR_SEEDS for field in (f"seed{seed}_probabilities", f"seed{seed}_predicted_value"))},
        "rating_model": {"family", "actual_rating_delta", "v3_prediction"},
        "dossier_comparison": {
            "opponent_group", "game_group", "action_type", "actual_response", "actual_proposal",
            *(f"{model}-{field}" for model in ("hierarchical-program", "direct-context", "prose-dossier") for field in ("available", "response-probability", "proposal-mean", "proposal-sigma")),
            *(f"{model}-{field}" for model in ("direct-context", "prose-dossier") for field in ("call-ok", "elapsed-s", "reasoning-tokens", "input-tokens")),
        },
        "execution_decisions": {"in_final_v108_prefix", "route", "fallback", "elapsed_s"},
        "execution_submissions": {"in_final_v108_prefix", "valid"},
        "execution_terminal_games": {"in_final_v108_prefix"},
    }
    for name, expected in expected_columns.items():
        actual = set(frames[name].columns)
        if actual != expected:
            raise ValueError(f"unexpected columns in {name}: {sorted(actual ^ expected)}")
    for name, columns in GROUP_PATTERNS.items():
        for column, pattern in columns.items():
            if any(pattern.fullmatch(str(value)) is None for value in frames[name][column].unique().to_list()):
                raise ValueError(f"non-anonymous group label in {name}.{column}")
    allowed_categories = {
        ("conditional_twin", "family"): set(FAMILIES),
        ("self_mirror", "family"): set(FAMILIES),
        ("self_mirror", "target_kind"): {"proposal", "response", "signal"},
        ("rating_model", "family"): set(FAMILIES),
        ("dossier_comparison", "action_type"): {"proposal", "response"},
        ("execution_decisions", "route"): {"local", "cloud-assisted"},
    }
    for (name, column), allowed in allowed_categories.items():
        observed = set(frames[name][column].unique().to_list())
        if not observed <= allowed:
            raise ValueError(f"unexpected categorical value in {name}.{column}: {sorted(observed - allowed)}")


def build_receipt(data_dir: Path) -> dict[str, Any]:
    frames, hashes = load_frames(data_dir)
    return {
        "schema_version": 1,
        "contract": "deeprmm-glee-deidentified-paper-metric-reproduction-v1",
        "source_files": hashes,
        "metrics": {
            "conditional_twin": conditional_twin_metrics(frames["conditional_twin"]),
            "persuasion_buyer_continuation": persuasion_metrics(frames["persuasion_continuation"]),
            "public_self_mirror": self_mirror_metrics(frames["self_mirror"]),
            "rating_model": rating_model_metrics(frames["rating_model"]),
            "dossier_comparison": dossier_metrics(frames["dossier_comparison"]),
            "execution": execution_metrics(frames["execution_decisions"], frames["execution_submissions"], frames["execution_terminal_games"]),
        },
        "boundary": "All inputs are deidentified metric-level sufficient data. The reproduction neither retrains models nor replays the nonstationary live competition.",
    }


def main() -> None:
    args = parse_args()
    receipt = build_receipt(args.data_dir.resolve())
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.expected is not None:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        if receipt != expected:
            raise SystemExit("reproduced metrics differ from the frozen expected receipt")
        print(f"reproduced metrics match {args.expected}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
