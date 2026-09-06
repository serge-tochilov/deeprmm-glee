"""Evaluate causal account-level identity inference and KI/HI performance offline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_analytics_lake import _digest, _fsync_directory, _write_parquet
from .glee_behavior_channel_analysis import CHANNELS, UNKNOWN_ID, channel_features, chronological_game_splits
from .glee_behavior_corpus import extract_behavior_moves
from .glee_behavior_fusion_analysis import ConditionalStacker, fit_conditional_stacker
from .glee_causal_identity_routing import BEHAVIOR_FEATURES, FUSION_ITERATIONS, FUSION_LEARNING_RATE, FUSION_RIDGE, CausalFusionExample, FrozenPresenceContext, _build_channel_profiles, _build_examples, _collapse_prior, _first_move_record, _gallery, _load_jsonl, _load_presence_context


ACCOUNT_ANALYSIS_CONTRACT = "glee-account-identity-performance-v1"
UNKNOWN_ACCOUNT = "__unlinked_or_unknown__"
CONFIDENCE_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
BOOTSTRAP_BLOCK_SECONDS = 3600
DEFAULT_BOOTSTRAP_REPLICATES = 2000
MINIMUM_KI_SUPPORT = 10
MINIMUM_HI_MASS = 5.0
MINIMUM_HI_EFFECTIVE_SUPPORT = 10.0


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def load_account_map(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups") if isinstance(payload, Mapping) else None
    if not isinstance(groups, list):
        raise ValueError("account-linkage file has no groups list")
    account_for_id: dict[str, str] = {}
    confidence_for_account: dict[str, str] = {}
    members_by_account: dict[str, tuple[str, ...]] = {}
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("key"), str) or not isinstance(group.get("members"), Mapping):
            raise ValueError("malformed account-linkage group")
        account = str(group["key"])
        confidence_for_account[account] = str(group.get("confidence") or "unknown")
        members = tuple(sorted(str(value) for value in group["members"].values()))
        members_by_account[account] = members
        for public_id in members:
            if public_id in account_for_id:
                raise ValueError(f"public ID belongs to several candidate accounts: {public_id}")
            account_for_id[public_id] = account
    return account_for_id, confidence_for_account, members_by_account


def collapse_account_posterior(probabilities: Mapping[str, float], account_for_id: Mapping[str, str]) -> dict[str, float]:
    collapsed: defaultdict[str, float] = defaultdict(float)
    for public_id, probability in probabilities.items():
        account = account_for_id.get(public_id, UNKNOWN_ACCOUNT) if public_id != UNKNOWN_ID else UNKNOWN_ACCOUNT
        collapsed[account] += float(probability)
    total = sum(collapsed.values())
    if total <= 0.0:
        raise ValueError("identity posterior has no probability mass")
    return {account: value / total for account, value in sorted(collapsed.items()) if value > 0.0}


def _ranking(probabilities: Mapping[str, float]) -> list[str]:
    return sorted(probabilities, key=lambda label: (-float(probabilities[label]), label))


def classification_metrics(rows: Sequence[Mapping[str, object]], *, probability_key: str, target_key: str) -> dict[str, object]:
    if not rows:
        return {"games": 0, "top_one_accuracy": None, "top_3_accuracy": None, "mean_reciprocal_rank": None, "negative_log_likelihood": None, "multiclass_brier": None, "named_prediction_coverage": None, "named_prediction_precision": None}
    top_one = 0
    top_3 = 0
    reciprocal = 0.0
    nll = 0.0
    brier = 0.0
    named = 0
    named_correct = 0
    for row in rows:
        probabilities = {str(label): float(value) for label, value in dict(row[probability_key]).items()}
        target = str(row[target_key])
        ranking = _ranking(probabilities)
        rank = ranking.index(target) + 1 if target in probabilities else len(ranking) + 1
        predicted = ranking[0]
        top_one += int(predicted == target)
        top_3 += int(target in ranking[:3])
        reciprocal += 1.0 / rank
        nll -= math.log(max(1e-15, probabilities.get(target, 0.0)))
        labels = set(probabilities) | {target}
        brier += sum((probabilities.get(label, 0.0) - float(label == target)) ** 2 for label in labels)
        if predicted != UNKNOWN_ACCOUNT:
            named += 1
            named_correct += int(predicted == target)
    return {
        "games": len(rows),
        "top_one_accuracy": top_one / len(rows),
        "top_3_accuracy": top_3 / len(rows),
        "mean_reciprocal_rank": reciprocal / len(rows),
        "negative_log_likelihood": nll / len(rows),
        "multiclass_brier": brier / len(rows),
        "named_prediction_coverage": named / len(rows),
        "named_prediction_precision": named_correct / named if named else None,
    }


def closed_set_metrics(rows: Sequence[Mapping[str, object]], *, probability_key: str, target_key: str, unknown_label: str) -> dict[str, object]:
    transformed = []
    for row in rows:
        probabilities = {str(label): float(value) for label, value in dict(row[probability_key]).items() if str(label) != unknown_label}
        total = sum(probabilities.values())
        if total <= 0.0:
            continue
        transformed.append(dict(row) | {"closed_probabilities": {label: value / total for label, value in probabilities.items()}})
    return classification_metrics(transformed, probability_key="closed_probabilities", target_key=target_key)


def _named_prediction_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        probabilities = dict(row["account_probabilities"])
        predicted = _ranking(probabilities)[0]
        if predicted == UNKNOWN_ACCOUNT:
            continue
        counts[predicted][0] += 1
        counts[predicted][1] += int(predicted == str(row["true_account"]))
    return {account: {"named_predictions": values[0], "correct": values[1], "precision": values[1] / values[0]} for account, values in sorted(counts.items())}


def _confidence_curve(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = []
        for row in rows:
            probabilities = dict(row["account_probabilities"])
            predicted = _ranking(probabilities)[0]
            if predicted != UNKNOWN_ACCOUNT and float(probabilities[predicted]) >= threshold:
                accepted.append((predicted, str(row["true_account"])))
        output.append({"minimum_confidence": threshold, "accepted_games": len(accepted), "coverage": len(accepted) / len(rows) if rows else None, "precision": sum(left == right for left, right in accepted) / len(accepted) if accepted else None})
    return output


@dataclass(frozen=True)
class FrozenIdentityComponents:
    presence: FrozenPresenceContext
    records: tuple[Mapping[str, object], ...]
    assignments: Mapping[str, str]
    galleries: Mapping[str, tuple[str, ...]]
    profiles: Mapping[tuple[str, str, str], object]
    temperatures: Mapping[tuple[str, str], float]
    stacker: ConditionalStacker
    calibration_examples: tuple[CausalFusionExample, ...]
    test_examples: tuple[CausalFusionExample, ...]
    parity: Mapping[str, object]


def _rebuild_frozen_identity_model(*, routing_dir: Path, presence_dir: Path, behavior_dir: Path, channel_dir: Path, event_cache: Path, identity_dir: Path, reporter_database: Path, activity_summary: Path) -> FrozenIdentityComponents:
    routing_manifest = json.loads((routing_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact, receipt in routing_manifest["artifacts"].items():
        if _file_digest(routing_dir / artifact) != receipt["sha256"]:
            raise RuntimeError(f"frozen routing artifact hash mismatch: {artifact}")
    implementation = routing_manifest["implementation_sha256"]
    implementation_paths = {
        "glee_causal_identity_routing.py": Path(__file__).with_name("glee_causal_identity_routing.py"),
        "glee_behavior_channel_analysis.py": Path(__file__).with_name("glee_behavior_channel_analysis.py"),
        "glee_behavior_fusion_analysis.py": Path(__file__).with_name("glee_behavior_fusion_analysis.py"),
        "glee_presence_forecast.py": Path(__file__).with_name("glee_presence_forecast.py"),
    }
    for name, path in implementation_paths.items():
        if _file_digest(path) != implementation[name]:
            raise RuntimeError(f"frozen identity implementation changed: {name}")
    presence = _load_presence_context(presence_dir=presence_dir, event_cache=event_cache, identity_dir=identity_dir, reporter_database=reporter_database, activity_summary=activity_summary)
    records = _load_jsonl(behavior_dir / "behavior-games.jsonl")
    channel_summary = json.loads((channel_dir / "summary.json").read_text(encoding="utf-8"))
    expected_test_posteriors = {str(row["game_id"]): row for row in _load_jsonl(presence_dir / "game-presence-posteriors.jsonl")}
    assignments, _boundaries = chronological_game_splits(records, train_fraction=float(channel_summary["design"]["train_fraction"]), calibration_fraction=float(channel_summary["design"]["calibration_fraction"]))
    galleries = _gallery(records, assignments)
    profiles, vectors, temperatures = _build_channel_profiles(records, assignments, galleries, channel_summary)
    examples, presence_parity = _build_examples(records, assignments, galleries, profiles, vectors, temperatures, presence, expected_test_posteriors)
    if presence_parity["mismatches"]:
        raise RuntimeError(f"causal-presence parity failed: {presence_parity}")
    calibration = tuple(example for example in examples if example.split == "calibration")
    test = tuple(example for example in examples if example.split == "test")
    stacker = fit_conditional_stacker([example.fusion("full") for example in calibration], feature_names=BEHAVIOR_FEATURES, ridge_lambda=FUSION_RIDGE, iterations=FUSION_ITERATIONS, learning_rate=FUSION_LEARNING_RATE)
    frozen_model = json.loads((routing_dir / "models.json").read_text(encoding="utf-8"))["fusion"]["full-behavior"]
    model_error = max([abs(float(stacker.unknown_bias) - float(frozen_model["unknown_bias"]))] + [abs(float(stacker.weights[name]) - float(frozen_model["weights"][name])) for name in BEHAVIOR_FEATURES] + [abs(float(stacker.scales[name]) - float(frozen_model["scales"][name])) for name in BEHAVIOR_FEATURES])
    stored_rows = {str(row["game_id"]): row for row in _load_jsonl(routing_dir / "identity-test-predictions.jsonl") if row["arm"] == "full-behavior"}
    maximum_probability_error = 0.0
    ranking_mismatches = 0
    for example in test:
        probabilities = stacker.probabilities(example.fusion("full"))
        stored = stored_rows[example.game_id]
        maximum_probability_error = max(maximum_probability_error, abs(float(stored["true_probability"]) - float(probabilities[example.target])), abs(float(stored["confidence"]) - max(probabilities.values())))
        ranking = _ranking(probabilities)
        expected_top = [(label, float(probabilities[label])) for label in ranking[:5]]
        actual_top = [(str(value["target"]), float(value["probability"])) for value in stored["top_candidates"]]
        ranking_mismatches += int([label for label, _value in expected_top] != [label for label, _value in actual_top])
        maximum_probability_error = max(maximum_probability_error, max((abs(left[1] - right[1]) for left, right in zip(expected_top, actual_top, strict=True)), default=0.0))
    if model_error > 1e-15 or maximum_probability_error > 1e-15 or ranking_mismatches or len(stored_rows) != len(test):
        raise RuntimeError(f"frozen identity reconstruction failed: model={model_error} probability={maximum_probability_error} rankings={ranking_mismatches} rows={len(stored_rows)}/{len(test)}")
    parity = {"test_games": len(test), "model_maximum_error": model_error, "prediction_maximum_error": maximum_probability_error, "top_5_ranking_mismatches": ranking_mismatches, "presence": presence_parity}
    return FrozenIdentityComponents(presence, tuple(records), assignments, galleries, profiles, temperatures, stacker, calibration, test, parity)


def _account_prediction(example: CausalFusionExample, stacker: ConditionalStacker, account_for_id: Mapping[str, str]) -> dict[str, object]:
    agent_probabilities = stacker.probabilities(example.fusion("full"))
    account_probabilities = collapse_account_posterior(agent_probabilities, account_for_id)
    account_ranking = _ranking(account_probabilities)
    true_account = account_for_id.get(example.true_public_player_id, UNKNOWN_ACCOUNT)
    return {
        "game_id": example.game_id,
        "family": example.family,
        "role": example.role,
        "started_at": example.started_at,
        "true_public_player_id": example.true_public_player_id,
        "agent_evaluation_target": example.target,
        "true_account": true_account,
        "top_account": account_ranking[0],
        "top_account_probability": float(account_probabilities[account_ranking[0]]),
        "account_probabilities": account_probabilities,
        "agent_probabilities": agent_probabilities,
    }


def _hidden_prediction(*, game_row: Mapping[str, object], game_archive_root: Path, components: FrozenIdentityComponents, account_for_id: Mapping[str, str]) -> tuple[dict[str, object] | None, str | None]:
    archive_path = game_archive_root / str(game_row["archive_path"])
    try:
        game = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable-archive"
    if _digest(game) != str(game_row["archive_sha256"]):
        return None, "archive-hash-mismatch"
    moves = extract_behavior_moves(game)
    if not moves:
        return None, "no-opponent-move"
    family = str(game_row["family"])
    started = game_row["started_at"]
    if not isinstance(started, datetime):
        return None, "invalid-started-at"
    record = {"game_id": str(game_row["game_id"]), "family": family, "started_at": started.isoformat(), "moves": moves, "channel_counts": {"moves": len(moves)}}
    snapshot = _first_move_record(record)
    channel_logits: dict[str, dict[str, float]] = {}
    evidence: dict[str, int] = {}
    for channel in CHANNELS:
        profile = components.profiles.get((family, channel, "test"))
        if profile is None:
            channel_logits[channel] = {candidate: 0.0 for candidate in components.galleries[family]} | {UNKNOWN_ID: 0.0}
            evidence[channel] = 0
            continue
        vector = channel_features(snapshot, channel)
        if channel == "action":
            vector = Counter({token: count for token, count in vector.items() if not token.startswith("terminal-style|")})
        scores, count = profile.scores(vector)
        temperature = components.temperatures[(family, channel)]
        channel_logits[channel] = {candidate: score / temperature for candidate, score in scores.items()} | {UNKNOWN_ID: 0.0}
        evidence[channel] = count
    _sequence, raw_prior = components.presence.raw_prior(family, started.timestamp(), "full")
    prior = _collapse_prior(raw_prior, components.galleries[family])
    example = CausalFusionExample(str(game_row["game_id"]), family, str(game_row["role"]), "test", started.isoformat(), UNKNOWN_ID, UNKNOWN_ID, components.galleries[family], {"full": prior}, channel_logits, evidence)
    agent_probabilities = components.stacker.probabilities(example.fusion("full"))
    account_probabilities = collapse_account_posterior(agent_probabilities, account_for_id)
    ranking = _ranking(account_probabilities)
    return {
        "game_id": str(game_row["game_id"]),
        "family": family,
        "role": str(game_row["role"]),
        "started_at": started,
        "top_account": ranking[0],
        "top_account_probability": float(account_probabilities[ranking[0]]),
        "unknown_account_probability": float(account_probabilities.get(UNKNOWN_ACCOUNT, 0.0)),
        "account_probabilities": account_probabilities,
        "evidence_tokens": sum(evidence.values()),
        "archive_path": str(game_row["archive_path"]),
        "archive_sha256": str(game_row["archive_sha256"]),
    }, None


def _standardized(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = max(math.sqrt(variance), 1e-9)
    return [(value - mean) / scale for value in values]


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    augmented = [list(row) + [float(vector[index])] for index, row in enumerate(matrix)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-14:
            divisor = 1e-14
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column], strict=True)]
    return [augmented[index][-1] for index in range(size)]


def _outcome_residuals(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, float], list[str]]:
    started = [float(row["started_at"].timestamp()) for row in rows]
    time_values = _standardized(started)
    self_values = _standardized([float(row["self_rating"]) for row in rows])
    opponent_values = _standardized([float(row["opponent_rating"]) for row in rows])
    engine_counts = Counter(str(row["engine_version"]) for row in rows)
    normalized_engines = [str(row["engine_version"]) if engine_counts[str(row["engine_version"])] >= 20 else "__sparse_engine__" for row in rows]
    categorical_values = {
        "family": [str(row["family"]) for row in rows],
        "family_role": [f"{row['family']}:{row['role']}" for row in rows],
        "engine": normalized_engines,
    }
    category_columns: list[tuple[str, str]] = []
    for name, values in categorical_values.items():
        categories = sorted(set(values))
        category_columns.extend((name, category) for category in categories[1:])
    feature_names = ["intercept", "time", "time_squared", "self_rating", "self_rating_squared", "opponent_rating", "opponent_rating_squared", "opponent_rating_imputed", *[f"{name}={category}" for name, category in category_columns]]
    design = []
    for index, row in enumerate(rows):
        categories = {"family": str(row["family"]), "family_role": f"{row['family']}:{row['role']}", "engine": normalized_engines[index]}
        design.append([1.0, time_values[index], time_values[index] ** 2, self_values[index], self_values[index] ** 2, opponent_values[index], opponent_values[index] ** 2, float(row["opponent_rating_imputed"]), *[float(categories[name] == category) for name, category in category_columns]])
    outcomes = [float(row["rating_delta"]) for row in rows]
    width = len(feature_names)
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    vector = [0.0 for _ in range(width)]
    for features, outcome in zip(design, outcomes, strict=True):
        for left in range(width):
            vector[left] += features[left] * outcome
            for right in range(left, width):
                gram[left][right] += features[left] * features[right]
    for left in range(width):
        for right in range(left):
            gram[left][right] = gram[right][left]
        if left:
            gram[left][left] += 1e-4
    coefficients = _solve(gram, vector)
    residuals = {str(row["game_id"]): outcome - sum(coefficient * feature for coefficient, feature in zip(coefficients, features, strict=True)) for row, outcome, features in zip(rows, outcomes, design, strict=True)}
    return residuals, feature_names


def _effective_support(weights: Sequence[float]) -> float:
    total = sum(weights)
    squares = sum(value * value for value in weights)
    return total * total / squares if squares > 0.0 else 0.0


def _block_bootstrap_difference(*, ki_rows: Sequence[Mapping[str, object]], hi_rows: Sequence[Mapping[str, object]], account: str, value_key: str, replicates: int, seed: int) -> dict[str, object]:
    contributions: defaultdict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for row in ki_rows:
        block = int(row["started_at"].timestamp()) // BOOTSTRAP_BLOCK_SECONDS
        contributions[block][0] += float(row[value_key])
        contributions[block][1] += 1.0
    for row in hi_rows:
        weight = float(dict(row["account_probabilities"]).get(account, 0.0))
        block = int(row["started_at"].timestamp()) // BOOTSTRAP_BLOCK_SECONDS
        contributions[block][2] += weight * float(row[value_key])
        contributions[block][3] += weight
    blocks = sorted(contributions)
    if not blocks:
        return {"replicates": replicates, "lower_95": None, "upper_95": None, "valid_replicates": 0}
    generator = random.Random(seed)
    values = []
    for _ in range(replicates):
        selected = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        ki_sum = sum(contributions[block][0] for block in selected)
        ki_count = sum(contributions[block][1] for block in selected)
        hi_sum = sum(contributions[block][2] for block in selected)
        hi_weight = sum(contributions[block][3] for block in selected)
        if ki_count > 0.0 and hi_weight > 0.0:
            values.append(hi_sum / hi_weight - ki_sum / ki_count)
    return {"replicates": replicates, "valid_replicates": len(values), "lower_95": _quantile(values, 0.025), "upper_95": _quantile(values, 0.975)}


def _block_bootstrap_hard_difference(*, ki_rows: Sequence[Mapping[str, object]], hi_rows: Sequence[Mapping[str, object]], value_key: str, replicates: int, seed: int) -> dict[str, object]:
    contributions: defaultdict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for row in ki_rows:
        block = int(row["started_at"].timestamp()) // BOOTSTRAP_BLOCK_SECONDS
        contributions[block][0] += float(row[value_key])
        contributions[block][1] += 1.0
    for row in hi_rows:
        block = int(row["started_at"].timestamp()) // BOOTSTRAP_BLOCK_SECONDS
        contributions[block][2] += float(row[value_key])
        contributions[block][3] += 1.0
    blocks = sorted(contributions)
    if not blocks:
        return {"replicates": replicates, "lower_95": None, "upper_95": None, "valid_replicates": 0}
    generator = random.Random(seed)
    values = []
    for _ in range(replicates):
        selected = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        ki_sum = sum(contributions[block][0] for block in selected)
        ki_count = sum(contributions[block][1] for block in selected)
        hi_sum = sum(contributions[block][2] for block in selected)
        hi_count = sum(contributions[block][3] for block in selected)
        if ki_count > 0.0 and hi_count > 0.0:
            values.append(hi_sum / hi_count - ki_sum / ki_count)
    return {"replicates": replicates, "valid_replicates": len(values), "lower_95": _quantile(values, 0.025), "upper_95": _quantile(values, 0.975)}


def _performance_rows(*, games: Sequence[Mapping[str, object]], hidden_predictions: Mapping[str, Mapping[str, object]], accounts: Sequence[str], confidence_for_account: Mapping[str, str], validation_by_account: Mapping[str, Mapping[str, object]], residuals: Mapping[str, float], replicates: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    known = [dict(row) | {"adjusted_rating_delta": residuals[str(row["game_id"])]} for row in games if row["identity_scope"] == "known" and row.get("direct_fleet_key") in accounts]
    hidden = [dict(row) | {"adjusted_rating_delta": residuals[str(row["game_id"])], "account_probabilities": hidden_predictions[str(row["game_id"])]["account_probabilities"], "top_account": hidden_predictions[str(row["game_id"])]["top_account"], "top_account_probability": hidden_predictions[str(row["game_id"])]["top_account_probability"]} for row in games if row["identity_scope"] == "hidden" and str(row["game_id"]) in hidden_predictions]
    soft_rows: list[dict[str, object]] = []
    hard_rows: list[dict[str, object]] = []
    for account in accounts:
        for family in (*GLEE_FAMILIES, "all"):
            ki = [row for row in known if row["direct_fleet_key"] == account and (family == "all" or row["family"] == family)]
            hi = [row for row in hidden if family == "all" or row["family"] == family]
            weights = [float(dict(row["account_probabilities"]).get(account, 0.0)) for row in hi]
            mass = sum(weights)
            ess = _effective_support(weights)
            concentration = len(hi) / ess if ess > 0.0 else None
            ki_mean = sum(float(row["rating_delta"]) for row in ki) / len(ki) if ki else None
            hi_mean = sum(weight * float(row["rating_delta"]) for weight, row in zip(weights, hi, strict=True)) / mass if mass > 0.0 else None
            ki_adjusted = sum(float(row["adjusted_rating_delta"]) for row in ki) / len(ki) if ki else None
            hi_adjusted = sum(weight * float(row["adjusted_rating_delta"]) for weight, row in zip(weights, hi, strict=True)) / mass if mass > 0.0 else None
            numerical_support = len(ki) >= MINIMUM_KI_SUPPORT and mass >= MINIMUM_HI_MASS and ess >= MINIMUM_HI_EFFECTIVE_SUPPORT
            validation = validation_by_account.get(account, {})
            validation_predictions = int(validation.get("named_predictions") or 0)
            validation_precision = float(validation["precision"]) if validation.get("precision") is not None else None
            signal_grade = "validated" if validation_predictions >= 3 and validation_precision is not None and validation_precision >= 0.8 else "contradicted" if validation_predictions and validation_precision is not None and validation_precision < 0.5 else "sparse-or-unvalidated"
            seed_text = f"{account}\0{family}\0{replicates}"
            seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
            raw_interval = _block_bootstrap_difference(ki_rows=ki, hi_rows=hi, account=account, value_key="rating_delta", replicates=replicates, seed=seed) if numerical_support else {"replicates": replicates, "valid_replicates": 0, "lower_95": None, "upper_95": None}
            adjusted_interval = _block_bootstrap_difference(ki_rows=ki, hi_rows=hi, account=account, value_key="adjusted_rating_delta", replicates=replicates, seed=seed + 1) if numerical_support else {"replicates": replicates, "valid_replicates": 0, "lower_95": None, "upper_95": None}
            soft_rows.append({"account": account, "linkage_confidence": confidence_for_account[account], "family": family, "masked_named_predictions": validation_predictions, "masked_named_precision": _rounded(validation_precision), "account_signal_grade": signal_grade, "ki_games": len(ki), "ki_mean_rating_delta": _rounded(ki_mean), "hi_games_in_pool": len(hi), "hi_posterior_mass": _rounded(mass), "hi_effective_support": _rounded(ess), "hi_posterior_concentration_ratio": _rounded(concentration), "hi_posterior_mean_rating_delta": _rounded(hi_mean), "hi_minus_ki": _rounded(hi_mean - ki_mean) if hi_mean is not None and ki_mean is not None else None, "hi_minus_ki_lower_95": _rounded(raw_interval["lower_95"]), "hi_minus_ki_upper_95": _rounded(raw_interval["upper_95"]), "ki_adjusted_mean": _rounded(ki_adjusted), "hi_adjusted_posterior_mean": _rounded(hi_adjusted), "adjusted_hi_minus_ki": _rounded(hi_adjusted - ki_adjusted) if hi_adjusted is not None and ki_adjusted is not None else None, "adjusted_lower_95": _rounded(adjusted_interval["lower_95"]), "adjusted_upper_95": _rounded(adjusted_interval["upper_95"]), "numerical_support": numerical_support, "interpretable_support": bool(numerical_support and signal_grade == "validated" and concentration is not None and concentration >= 1.25)})
            for threshold in CONFIDENCE_THRESHOLDS:
                selected = [row for row in hi if row["top_account"] == account and float(row["top_account_probability"]) >= threshold]
                hard_supported = len(ki) >= MINIMUM_KI_SUPPORT and len(selected) >= 5
                hard_raw_interval = _block_bootstrap_hard_difference(ki_rows=ki, hi_rows=selected, value_key="rating_delta", replicates=replicates, seed=seed + int(threshold * 1000) + 2) if hard_supported else {"lower_95": None, "upper_95": None}
                hard_adjusted_interval = _block_bootstrap_hard_difference(ki_rows=ki, hi_rows=selected, value_key="adjusted_rating_delta", replicates=replicates, seed=seed + int(threshold * 1000) + 3) if hard_supported else {"lower_95": None, "upper_95": None}
                hard_adjusted_mean = sum(float(row["adjusted_rating_delta"]) for row in selected) / len(selected) if selected else None
                hard_rows.append({"account": account, "linkage_confidence": confidence_for_account[account], "masked_named_predictions": validation_predictions, "masked_named_precision": _rounded(validation_precision), "account_signal_grade": signal_grade, "family": family, "minimum_confidence": threshold, "ki_games": len(ki), "hi_assigned_games": len(selected), "ki_mean_rating_delta": _rounded(ki_mean), "hi_hard_mean_rating_delta": _rounded(sum(float(row["rating_delta"]) for row in selected) / len(selected)) if selected else None, "hi_hard_minus_ki": _rounded(sum(float(row["rating_delta"]) for row in selected) / len(selected) - ki_mean) if selected and ki_mean is not None else None, "hi_hard_minus_ki_lower_95": _rounded(hard_raw_interval["lower_95"]), "hi_hard_minus_ki_upper_95": _rounded(hard_raw_interval["upper_95"]), "ki_adjusted_mean": _rounded(ki_adjusted), "hi_hard_adjusted_mean": _rounded(hard_adjusted_mean), "adjusted_hi_hard_minus_ki": _rounded(hard_adjusted_mean - ki_adjusted) if hard_adjusted_mean is not None and ki_adjusted is not None else None, "adjusted_lower_95": _rounded(hard_adjusted_interval["lower_95"]), "adjusted_upper_95": _rounded(hard_adjusted_interval["upper_95"]), "interpretable_support": bool(hard_supported and signal_grade == "validated")})
    return soft_rows, hard_rows


class GleeAccountIdentityPerformanceAnalysis:
    """Run retrospective account pooling and probabilistic KI/HI outcome analysis."""

    def __init__(self, *, routing_dir: Path, presence_dir: Path, behavior_dir: Path, channel_dir: Path, event_cache: Path, identity_dir: Path, reporter_database: Path, activity_summary: Path, fleet_groups: Path, analytics_dir: Path, game_archive_root: Path, output_dir: Path, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES) -> None:
        self.routing_dir = routing_dir.resolve()
        self.presence_dir = presence_dir.resolve()
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.event_cache = event_cache.resolve()
        self.identity_dir = identity_dir.resolve()
        self.reporter_database = reporter_database.resolve()
        self.activity_summary = activity_summary.resolve()
        self.fleet_groups = fleet_groups.resolve()
        self.analytics_dir = analytics_dir.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.output_dir = output_dir.resolve()
        self.bootstrap_replicates = bootstrap_replicates

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"account-analysis output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        account_for_id, confidence_for_account, members_by_account = load_account_map(self.fleet_groups)
        components = _rebuild_frozen_identity_model(routing_dir=self.routing_dir, presence_dir=self.presence_dir, behavior_dir=self.behavior_dir, channel_dir=self.channel_dir, event_cache=self.event_cache, identity_dir=self.identity_dir, reporter_database=self.reporter_database, activity_summary=self.activity_summary)
        calibration_rows = [_account_prediction(example, components.stacker, account_for_id) for example in components.calibration_examples]
        test_rows = [_account_prediction(example, components.stacker, account_for_id) for example in components.test_examples]
        linked_test = [row for row in test_rows if row["true_account"] != UNKNOWN_ACCOUNT]
        paired_test = [row for row in linked_test if row["agent_evaluation_target"] == row["true_public_player_id"]]
        calibration_named_diagnostics = _named_prediction_diagnostics(calibration_rows)
        test_named_diagnostics = _named_prediction_diagnostics(test_rows)
        account_metrics = {
            "all_open_set": classification_metrics(test_rows, probability_key="account_probabilities", target_key="true_account"),
            "linked_accounts": classification_metrics(linked_test, probability_key="account_probabilities", target_key="true_account"),
            "linked_accounts_closed_set": closed_set_metrics(linked_test, probability_key="account_probabilities", target_key="true_account", unknown_label=UNKNOWN_ACCOUNT),
            "paired_exact_agent_account": classification_metrics(paired_test, probability_key="account_probabilities", target_key="true_account"),
            "paired_exact_agent_account_closed_set": closed_set_metrics(paired_test, probability_key="account_probabilities", target_key="true_account", unknown_label=UNKNOWN_ACCOUNT),
            "by_family": {family: classification_metrics([row for row in linked_test if row["family"] == family], probability_key="account_probabilities", target_key="true_account") for family in GLEE_FAMILIES},
            "by_linkage_confidence": {confidence: classification_metrics([row for row in linked_test if confidence_for_account.get(str(row["true_account"])) == confidence], probability_key="account_probabilities", target_key="true_account") for confidence in ("very-high", "high", "medium")},
        }
        agent_paired_metrics = classification_metrics(paired_test, probability_key="agent_probabilities", target_key="true_public_player_id")
        agent_paired_closed_metrics = closed_set_metrics(paired_test, probability_key="agent_probabilities", target_key="true_public_player_id", unknown_label=UNKNOWN_ID)
        account_by_true = {account: classification_metrics([row for row in linked_test if row["true_account"] == account], probability_key="account_probabilities", target_key="true_account") for account in sorted(members_by_account)}
        analytics_manifest = json.loads((self.analytics_dir / "manifest.json").read_text(encoding="utf-8"))
        for artifact, receipt in analytics_manifest["artifacts"].items():
            if _file_digest(self.analytics_dir / artifact) != receipt["sha256"]:
                raise RuntimeError(f"analytics lake artifact hash mismatch: {artifact}")
        games_frame = pl.read_parquet(self.analytics_dir / "games.parquet")
        cutoff_epoch = components.presence.frontier_times[-1]
        eligible_frame = games_frame.filter(pl.col("completed_at").dt.epoch("us") <= int(cutoff_epoch * 1_000_000)).sort("started_at")
        game_rows = eligible_frame.to_dicts()
        hidden_rows = [row for row in game_rows if row["identity_scope"] == "hidden"]
        hidden_predictions: list[dict[str, object]] = []
        hidden_exclusions: Counter[str] = Counter()
        for row in hidden_rows:
            prediction, exclusion = _hidden_prediction(game_row=row, game_archive_root=self.game_archive_root, components=components, account_for_id=account_for_id)
            if prediction is None:
                hidden_exclusions[str(exclusion)] += 1
            else:
                hidden_predictions.append(prediction)
        hidden_by_game = {str(row["game_id"]): row for row in hidden_predictions}
        analysis_game_rows = [row for row in game_rows if row["identity_scope"] == "known" or str(row["game_id"]) in hidden_by_game]
        residuals, control_features = _outcome_residuals(analysis_game_rows)
        soft_performance, hard_performance = _performance_rows(games=analysis_game_rows, hidden_predictions=hidden_by_game, accounts=sorted(members_by_account), confidence_for_account=confidence_for_account, validation_by_account=test_named_diagnostics, residuals=residuals, replicates=self.bootstrap_replicates)
        hidden_top_counts = Counter(str(row["top_account"]) for row in hidden_predictions)
        account_prediction_rows = []
        for row in test_rows:
            account_prediction_rows.append({key: value for key, value in row.items() if key not in {"agent_probabilities", "account_probabilities"}} | {"account_probabilities_json": _canonical(row["account_probabilities"]), "agent_probabilities_json": _canonical(row["agent_probabilities"])})
        hidden_prediction_rows = [{key: value for key, value in row.items() if key != "account_probabilities"} | {"account_probabilities_json": _canonical(row["account_probabilities"])} for row in hidden_predictions]
        artifacts: dict[str, Mapping[str, object]] = {}
        artifacts["masked-ki-account-predictions.parquet"] = _write_parquet(pl.DataFrame(account_prediction_rows, infer_schema_length=None), self.output_dir / "masked-ki-account-predictions.parquet")
        artifacts["hidden-account-posteriors.parquet"] = _write_parquet(pl.DataFrame(hidden_prediction_rows, infer_schema_length=None), self.output_dir / "hidden-account-posteriors.parquet")
        artifacts["account-performance.parquet"] = _write_parquet(pl.DataFrame(soft_performance, infer_schema_length=None), self.output_dir / "account-performance.parquet")
        artifacts["hard-assignment-sensitivity.parquet"] = _write_parquet(pl.DataFrame(hard_performance, infer_schema_length=None), self.output_dir / "hard-assignment-sensitivity.parquet")
        summary = {
            "contract": ACCOUNT_ANALYSIS_CONTRACT,
            "schema_version": 1,
            "status": "offline-retrospective-only",
            "sources": {"routing_manifest_sha256": _file_digest(self.routing_dir / "manifest.json"), "account_groups_sha256": _file_digest(self.fleet_groups), "analytics_manifest_sha256": _file_digest(self.analytics_dir / "manifest.json"), "identity_frontier_sequence": 18108, "identity_frontier_completed_at": datetime.fromtimestamp(cutoff_epoch, tz=timezone.utc).isoformat()},
            "design": {"account_count": len(members_by_account), "mapped_public_ids": len(account_for_id), "unknown_account_class": UNKNOWN_ACCOUNT, "identity_observation": "strictly pregame public prior plus exactly one opponent move", "masked_ki_test_games": len(test_rows), "linked_account_test_games": len(linked_test), "paired_exact_agent_account_games": len(paired_test), "eligible_outcome_games": len(game_rows), "eligible_hidden_games": len(hidden_rows), "inferred_hidden_games": len(hidden_predictions), "hidden_exclusions": dict(sorted(hidden_exclusions.items())), "bootstrap_replicates": self.bootstrap_replicates, "bootstrap_block_seconds": BOOTSTRAP_BLOCK_SECONDS, "outcome_control_features": control_features},
            "reconstruction_parity": components.parity,
            "masked_ki": {"account_metrics": account_metrics, "paired_agent_metrics": agent_paired_metrics, "paired_agent_closed_set_metrics": agent_paired_closed_metrics, "confidence_curve_calibration": _confidence_curve(calibration_rows), "confidence_curve_test": _confidence_curve(test_rows), "named_prediction_by_account_calibration": calibration_named_diagnostics, "named_prediction_by_account_test": test_named_diagnostics, "by_true_account": account_by_true},
            "hidden_inference": {"top_account_counts": dict(sorted(hidden_top_counts.items())), "named_top_games": sum(account != UNKNOWN_ACCOUNT for account in hidden_top_counts.elements()), "unknown_top_games": hidden_top_counts[UNKNOWN_ACCOUNT]},
            "performance": {"soft_cells": len(soft_performance), "numerically_supported_soft_cells": sum(bool(row["numerical_support"]) for row in soft_performance), "interpretable_soft_cells": sum(bool(row["interpretable_support"]) for row in soft_performance), "interpretable_hard_cells": sum(bool(row["interpretable_support"]) for row in hard_performance), "interpretation": "posterior-weighted HI estimates are descriptive shrinkage; high effective support can reflect diffuse weights, and hard-assignment estimates require masked-KI account validation"},
            "authority": {"live_identity_routing": False, "account_assignment": False, "causal_performance_claim": False},
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        artifacts["summary.json"] = {"rows": 1, "bytes": (self.output_dir / "summary.json").stat().st_size, "sha256": _file_digest(self.output_dir / "summary.json")}
        protocol_path = Path(__file__).resolve().parents[2] / "protocols" / "glee-account-identity-performance-v1.md"
        manifest = {"contract": ACCOUNT_ANALYSIS_CONTRACT, "schema_version": 1, "sources": summary["sources"], "implementation_sha256": {"glee_account_identity_analysis.py": _file_digest(Path(__file__)), "glee-account-identity-performance-v1.md": _file_digest(protocol_path)}, "artifacts": artifacts}
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return {"contract": ACCOUNT_ANALYSIS_CONTRACT, "output_dir": str(self.output_dir), "masked_ki": summary["masked_ki"], "hidden_inference": summary["hidden_inference"], "interpretable_soft_cells": summary["performance"]["interpretable_soft_cells"], "interpretable_hard_cells": summary["performance"]["interpretable_hard_cells"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
