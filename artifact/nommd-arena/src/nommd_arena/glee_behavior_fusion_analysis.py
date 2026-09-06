"""Run an offline, activity-conditioned fusion of collision-safe GLEE behavior channels."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_activity_eda import _file_digest
from .glee_behavior_channel_analysis import CHANNELS, FAMILIES, UNKNOWN_ID, channel_features, chronological_game_splits, classification_metrics, fit_fingerprint_profile
from .glee_joint_assignment import AssignmentDemand, AssignmentOption, sample_capacity_marginals, solve_capacity_assignment, solve_capacity_assignment_auction
from .glee_joint_assignment_analysis import JOINT_ASSIGNMENT_ANALYSIS_CONTRACT, _assignment_components, _independent_component, _softmax_probabilities, _unknown_utility


BEHAVIOR_FUSION_CONTRACT = "glee-behavior-activity-fusion-v1"
ASSIGNMENT_MODELS = ("activity-only", "rating-only", "joint")
FEATURE_NAMES = ("activity", "rating", "timing", "action", "lexical", "discourse")
FITTED_ARMS = {
    "activity-calibrated": ("activity",),
    "timing-only": ("timing",),
    "action-only": ("action",),
    "language-only": ("lexical", "discourse"),
    "activity-language": ("activity", "lexical", "discourse"),
    "fused": FEATURE_NAMES,
}
RIDGE_LAMBDA = 0.1
OPTIMIZER_ITERATIONS = 300
OPTIMIZER_LEARNING_RATE = 0.03
PROBABILITY_FLOOR = 1e-8
SERIALIZATION_PARITY_TOLERANCE = 1e-5


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(path, "".join(_canonical(value) + "\n" for value in values))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    maximum = max(logits.values())
    weights = {label: math.exp(value - maximum) for label, value in logits.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _assignment_utility(edge: Mapping[str, object], model: str) -> float:
    activity = float(edge.get("activity_utility") or 0.0)
    rating = float(edge.get("rating_utility") or 0.0)
    if model == "activity-only":
        return activity
    if model == "rating-only":
        return rating
    if model == "joint":
        return activity + 0.5 * rating
    raise ValueError(f"unsupported assignment model: {model}")


def reconstruct_assignment_probabilities(candidate_rows: Sequence[Mapping[str, object]], event_rows: Sequence[Mapping[str, object]], *, model: str, samples: int, temperature: float, frontier_sequence: int, auction_epsilon: float) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    """Reconstruct full player marginals from the compact Stage 2 event-reference substrate."""
    events = {str(row["event_id"]): row for row in event_rows}
    capacities = {event_id: int(row["games_delta"]) for event_id, row in events.items()}
    demands: list[AssignmentDemand] = []
    for row in candidate_rows:
        options = tuple(AssignmentOption(str(edge["event_id"]), _assignment_utility(edge, model)) for edge in row.get("candidates", []) if str(edge["event_id"]) in events)
        unknown = _unknown_utility((option.utility for option in options), float(row.get("unknown_probability_prior") or 0.0))
        demands.append(AssignmentDemand(str(row["game_id"]), options, unknown))
    components = _assignment_components(demands)
    event_probabilities: dict[str, dict[str | None, float]] = {}
    sampled_components = 0
    for component in components:
        resource_ids = {option.resource_id for demand in component for option in demand.options}
        component_capacities = {resource_id: capacities[resource_id] for resource_id in resource_ids}
        if _independent_component(component, component_capacities):
            component_probabilities = {demand.demand_id: _softmax_probabilities(demand, temperature) for demand in component}
        else:
            sampled_components += 1
            solver = solve_capacity_assignment if len(component) <= 40 else lambda values, limits: solve_capacity_assignment_auction(values, limits, epsilon=auction_epsilon)
            marginal = sample_capacity_marginals(component, component_capacities, samples=samples, temperature=temperature, seed=f"{JOINT_ASSIGNMENT_ANALYSIS_CONTRACT}:{frontier_sequence}:{model}:{component[0].demand_id}", solver=solver)
            component_probabilities = marginal.probabilities
        event_probabilities.update(component_probabilities)
    player_probabilities: dict[str, dict[str, float]] = {}
    for game_id, probabilities in event_probabilities.items():
        players: Counter[str] = Counter()
        players[UNKNOWN_ID] = float(probabilities.get(None, 0.0))
        for event_id, probability in probabilities.items():
            if event_id is None:
                continue
            event = events.get(str(event_id))
            if event is not None:
                players[str(event["public_player_id"])] += float(probability)
        player_probabilities[game_id] = dict(players)
    return player_probabilities, {"model": model, "demands": len(demands), "components": len(components), "sampled_conflict_components": sampled_components, "samples": samples, "temperature": temperature}


def assignment_parity(reconstructed: Mapping[str, Mapping[str, float]], expected_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Verify reconstructed marginals against Stage 2 labels and six-decimal probabilities."""
    mismatches = 0
    label_mismatches = 0
    probability_mismatches = 0
    strict_probability_differences = 0
    strict_unknown_probability_differences = 0
    maximum_probability_error = 0.0
    for row in expected_rows:
        game_id = str(row["game_id"])
        probabilities = reconstructed[game_id]
        unknown_error = abs(float(row["unknown_assignment_probability"]) - round(float(probabilities.get(UNKNOWN_ID, 0.0)), 6))
        maximum_probability_error = max(maximum_probability_error, unknown_error)
        actual_top = sorted(((label, probability) for label, probability in probabilities.items() if label != UNKNOWN_ID), key=lambda item: (-item[1], item[0]))[:10]
        expected_top = [(str(value["public_player_id"]), float(value["probability"])) for value in row.get("top_public_players", [])]
        actual_rounded = [(label, round(probability, 6)) for label, probability in actual_top]
        if len(actual_rounded) != len(expected_top):
            mismatches += 1
            label_mismatches += 1
            continue
        row_label_mismatch = False
        row_probability_mismatch = unknown_error > SERIALIZATION_PARITY_TOLERANCE
        row_strict_probability_difference = unknown_error > 1e-9
        strict_unknown_probability_differences += int(unknown_error > 1e-9)
        for actual, expected in zip(actual_rounded, expected_top, strict=True):
            maximum_probability_error = max(maximum_probability_error, abs(actual[1] - expected[1]))
            row_label_mismatch = row_label_mismatch or actual[0] != expected[0]
            row_probability_mismatch = row_probability_mismatch or abs(actual[1] - expected[1]) > SERIALIZATION_PARITY_TOLERANCE
            row_strict_probability_difference = row_strict_probability_difference or abs(actual[1] - expected[1]) > 1e-9
        label_mismatches += int(row_label_mismatch)
        probability_mismatches += int(row_probability_mismatch)
        strict_probability_differences += int(row_strict_probability_difference)
        mismatches += int(row_label_mismatch or row_probability_mismatch)
    return {
        "rows": len(expected_rows),
        "mismatches": mismatches,
        "label_mismatches": label_mismatches,
        "probability_mismatches": probability_mismatches,
        "strict_probability_differences": strict_probability_differences,
        "strict_unknown_probability_differences": strict_unknown_probability_differences,
        "serialization_tolerance": SERIALIZATION_PARITY_TOLERANCE,
        "maximum_rounded_probability_error": maximum_probability_error,
    }


def _opponent_role(record: Mapping[str, object]) -> str:
    for move in record.get("moves", []):
        if not isinstance(move, Mapping):
            continue
        context = move.get("context") if isinstance(move.get("context"), Mapping) else {}
        role = context.get("opponent_role")
        if role:
            return str(role).casefold()
    return "none"


@dataclass(frozen=True)
class ChannelProfilePair:
    """Earlier-only and final channel profiles under one frozen Stage 5 calibration."""

    candidates: tuple[str, ...]
    calibration_profile: object
    test_profile: object
    temperature: float
    unknown_bias: float

    def logits(self, features: Mapping[str, int], *, split: str) -> tuple[dict[str, float], int]:
        profile = self.calibration_profile if split == "calibration" else self.test_profile
        scores, evidence = profile.scores(features)
        if evidence == 0:
            return {candidate: 0.0 for candidate in self.candidates} | {UNKNOWN_ID: 0.0}, 0
        logits = {candidate: score / self.temperature for candidate, score in scores.items()}
        logits[UNKNOWN_ID] = math.log(max(1, len(self.candidates))) + self.unknown_bias
        return logits, evidence


def build_channel_profiles(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], channel_summary: Mapping[str, object]) -> tuple[dict[tuple[str, str], ChannelProfilePair], dict[tuple[str, str], Counter[str]]]:
    """Rebuild Stage 5 profiles for calibration and untouched-test feature emission."""
    profiles: dict[tuple[str, str], ChannelProfilePair] = {}
    vectors: dict[tuple[str, str], Counter[str]] = {}
    for channel in CHANNELS:
        for record in records:
            vectors[(str(record["game_id"]), channel)] = channel_features(record, channel)
        for family in FAMILIES:
            result = channel_summary["families"][family][channel]
            if result.get("status") == "insufficient-support":
                continue
            candidates = tuple(str(value) for value in result["final_profile"]["candidate_ids"])
            selected = result["selected_hyperparameters"]
            family_records = [record for record in records if record.get("family") == family]
            train = [record for record in family_records if assignments.get(str(record["game_id"])) == "train"]
            calibration = [record for record in family_records if assignments.get(str(record["game_id"])) == "calibration"]
            channel_vectors = {str(record["game_id"]): vectors[(str(record["game_id"]), channel)] for record in family_records}
            calibration_profile = fit_fingerprint_profile(train, channel_vectors, candidates=candidates, alpha=float(selected["alpha"]), channel=channel)
            test_profile = fit_fingerprint_profile(train + calibration, channel_vectors, candidates=candidates, alpha=float(selected["alpha"]), channel=channel)
            profiles[(family, channel)] = ChannelProfilePair(candidates, calibration_profile, test_profile, float(selected["temperature"]), float(selected["unknown_bias"]))
    return profiles, vectors


@dataclass(frozen=True)
class FusionExample:
    """One activity-compatible identity choice with candidate-specific channel features."""

    game_id: str
    family: str
    role: str
    split: str
    started_at: str
    candidates: tuple[str, ...]
    target: str
    true_public_player_id: str
    features: Mapping[str, Mapping[str, float]]
    raw_probabilities: Mapping[str, Mapping[str, float]]
    evidence: Mapping[str, int]


def build_fusion_examples(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], candidate_rows: Sequence[Mapping[str, object]], event_rows: Sequence[Mapping[str, object]], assignment_probabilities: Mapping[str, Mapping[str, Mapping[str, float]]], profiles: Mapping[tuple[str, str], ChannelProfilePair], vectors: Mapping[tuple[str, str], Counter[str]]) -> list[FusionExample]:
    """Join causal behavior profiles to the full retrospective Stage 2 candidate universe."""
    candidate_by_game = {str(row["game_id"]): row for row in candidate_rows}
    event_identity = {str(row["event_id"]): str(row["public_player_id"]) for row in event_rows}
    examples: list[FusionExample] = []
    for record in records:
        game_id = str(record["game_id"])
        split = assignments.get(game_id)
        if split not in {"calibration", "test"} or game_id not in candidate_by_game:
            continue
        candidate_row = candidate_by_game[game_id]
        public_ids = sorted({event_identity[str(edge["event_id"])] for edge in candidate_row.get("candidates", []) if str(edge["event_id"]) in event_identity})
        labels = tuple([*public_ids, UNKNOWN_ID])
        family = str(record["family"])
        channel_logits: dict[str, dict[str, float]] = {}
        evidence: dict[str, int] = {}
        for channel in CHANNELS:
            profile = profiles.get((family, channel))
            if profile is None:
                logits, count = ({UNKNOWN_ID: 0.0}, 0)
            else:
                logits, count = profile.logits(vectors[(game_id, channel)], split=split)
            channel_logits[channel] = logits
            evidence[channel] = count
        raw = {model: dict(assignment_probabilities[model][game_id]) for model in ASSIGNMENT_MODELS}
        features: dict[str, dict[str, float]] = {}
        for label in labels:
            features[label] = {
                "activity": math.log(max(PROBABILITY_FLOOR, raw["activity-only"].get(label, 0.0))),
                "rating": math.log(max(PROBABILITY_FLOOR, raw["rating-only"].get(label, 0.0))),
                "timing": channel_logits["timing"].get(label, 0.0),
                "action": channel_logits["action"].get(label, 0.0),
                "lexical": channel_logits["lexical"].get(label, 0.0),
                "discourse": channel_logits["discourse"].get(label, 0.0),
            }
        truth = str(record["public_player_id"])
        target = truth if truth in public_ids else UNKNOWN_ID
        examples.append(FusionExample(game_id, family, _opponent_role(record), split, str(record["started_at"]), labels, target, truth, features, raw, evidence))
    examples.sort(key=lambda example: (example.started_at, example.game_id))
    return examples


@dataclass(frozen=True)
class ConditionalStacker:
    """One nonnegative conditional log-linear stacker with a fitted unknown intercept."""

    feature_names: tuple[str, ...]
    scales: Mapping[str, float]
    weights: Mapping[str, float]
    unknown_bias: float

    def probabilities(self, example: FusionExample) -> dict[str, float]:
        logits: dict[str, float] = {}
        means = {feature: sum(float(example.features[label][feature]) for label in example.candidates) / len(example.candidates) for feature in self.feature_names}
        for label in example.candidates:
            score = sum(self.weights[feature] * (float(example.features[label][feature]) - means[feature]) / self.scales[feature] for feature in self.feature_names)
            logits[label] = score + (self.unknown_bias if label == UNKNOWN_ID else 0.0)
        return _softmax(logits)


def _feature_scales(examples: Sequence[FusionExample], feature_names: Sequence[str]) -> dict[str, float]:
    sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for example in examples:
        for feature in feature_names:
            values = [float(example.features[label][feature]) for label in example.candidates]
            mean = sum(values) / len(values)
            sums[feature] += sum((value - mean) ** 2 for value in values)
            counts[feature] += len(values)
    return {feature: max(0.1, math.sqrt(sums[feature] / max(1, counts[feature]))) for feature in feature_names}


def fit_conditional_stacker(examples: Sequence[FusionExample], *, feature_names: Sequence[str], ridge_lambda: float = RIDGE_LAMBDA, iterations: int = OPTIMIZER_ITERATIONS, learning_rate: float = OPTIMIZER_LEARNING_RATE) -> ConditionalStacker:
    """Fit a deterministic nonnegative conditional model by full-batch Adam updates."""
    if not examples or not feature_names:
        raise ValueError("conditional stacker requires examples and features")
    names = tuple(feature_names)
    scales = _feature_scales(examples, names)
    weights = {feature: 0.1 for feature in names}
    unknown_bias = 0.0
    first_moment = {feature: 0.0 for feature in names} | {"__unknown_bias__": 0.0}
    second_moment = dict(first_moment)
    for iteration in range(1, iterations + 1):
        gradient = {feature: 0.0 for feature in names}
        bias_gradient = 0.0
        model = ConditionalStacker(names, scales, weights, unknown_bias)
        for example in examples:
            probabilities = model.probabilities(example)
            means = {feature: sum(float(example.features[label][feature]) for label in example.candidates) / len(example.candidates) for feature in names}
            for feature in names:
                expected = sum(probabilities[label] * (float(example.features[label][feature]) - means[feature]) / scales[feature] for label in example.candidates)
                observed = (float(example.features[example.target][feature]) - means[feature]) / scales[feature]
                gradient[feature] += expected - observed
            bias_gradient += probabilities[UNKNOWN_ID] - float(example.target == UNKNOWN_ID)
        for feature in names:
            gradient[feature] = gradient[feature] / len(examples) + ridge_lambda * weights[feature]
        bias_gradient = bias_gradient / len(examples) + 0.01 * ridge_lambda * unknown_bias
        gradients = gradient | {"__unknown_bias__": bias_gradient}
        for parameter, value in gradients.items():
            first_moment[parameter] = 0.9 * first_moment[parameter] + 0.1 * value
            second_moment[parameter] = 0.999 * second_moment[parameter] + 0.001 * value * value
            corrected_first = first_moment[parameter] / (1.0 - 0.9**iteration)
            corrected_second = second_moment[parameter] / (1.0 - 0.999**iteration)
            update = learning_rate * corrected_first / (math.sqrt(corrected_second) + 1e-8)
            if parameter == "__unknown_bias__":
                unknown_bias -= update
            else:
                weights[parameter] = max(0.0, weights[parameter] - update)
    return ConditionalStacker(names, scales, weights, unknown_bias)


def _prediction_record(example: FusionExample, arm: str, probabilities: Mapping[str, float]) -> dict[str, object]:
    ranking = sorted(probabilities, key=lambda label: (-probabilities[label], label))
    candidate_ranking = [label for label in ranking if label != UNKNOWN_ID]
    predicted = ranking[0]
    true_probability = max(1e-15, float(probabilities.get(example.target, 0.0)))
    return {
        "contract": BEHAVIOR_FUSION_CONTRACT,
        "game_id": example.game_id,
        "family": example.family,
        "role": example.role,
        "arm": arm,
        "started_at": example.started_at,
        "true_public_player_id": example.true_public_player_id,
        "evaluation_target": example.target,
        "predicted_target": predicted,
        "true_rank": ranking.index(example.target) + 1,
        "known_candidate_rank": candidate_ranking.index(example.target) + 1 if example.target != UNKNOWN_ID else None,
        "true_probability": true_probability,
        "confidence": probabilities[predicted],
        "correct": predicted == example.target,
        "evidence_tokens": sum(example.evidence.values()),
        "evidence_by_channel": dict(example.evidence),
        "candidate_count": len(example.candidates) - 1,
        "top_candidates": [{"target": label, "probability": probabilities[label]} for label in ranking[:5]],
        "nll": -math.log(true_probability),
        "brier": sum((probability - float(label == example.target)) ** 2 for label, probability in probabilities.items()),
    }


def _normalized_raw(example: FusionExample, model: str) -> dict[str, float]:
    values = {label: max(0.0, float(example.raw_probabilities[model].get(label, 0.0))) for label in example.candidates}
    total = sum(values.values())
    return {label: value / total for label, value in values.items()} if total > 0.0 else {label: 1.0 / len(values) for label in values}


def _abstention(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for threshold in (0.1, 0.2, 0.3, 0.5, 0.7):
        accepted = [row for row in rows if row["predicted_target"] != UNKNOWN_ID and float(row["confidence"]) >= threshold]
        values.append({"minimum_confidence": threshold, "accepted_games": len(accepted), "coverage": len(accepted) / len(rows) if rows else None, "accuracy": sum(bool(row["correct"]) for row in accepted) / len(accepted) if accepted else None})
    return values


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum((value - left_mean) ** 2 for value in left) * sum((value - right_mean) ** 2 for value in right))
    return numerator / denominator if denominator > 0.0 else None


def _phi(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    both = sum(a and b for a, b in zip(left, right, strict=True))
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(not a and b for a, b in zip(left, right, strict=True))
    neither = len(left) - both - left_only - right_only
    denominator = math.sqrt((both + left_only) * (right_only + neither) * (both + right_only) * (left_only + neither))
    return (both * neither - left_only * right_only) / denominator if denominator > 0.0 else None


def channel_error_correlations(examples: Sequence[FusionExample]) -> dict[str, object]:
    """Measure held-out dependence between channel margins and top-5 successes."""
    output: dict[str, object] = {}
    for left_index, left_channel in enumerate(CHANNELS):
        for right_channel in CHANNELS[left_index + 1 :]:
            left_margins: list[float] = []
            right_margins: list[float] = []
            left_top_5: list[bool] = []
            right_top_5: list[bool] = []
            for example in examples:
                if example.target == UNKNOWN_ID or example.evidence[left_channel] == 0 or example.evidence[right_channel] == 0:
                    continue
                known = [label for label in example.candidates if label != UNKNOWN_ID]
                if example.target not in known or len(known) < 2:
                    continue
                left_ranking = sorted(known, key=lambda label: (-float(example.features[label][left_channel]), label))
                right_ranking = sorted(known, key=lambda label: (-float(example.features[label][right_channel]), label))
                left_other = max(float(example.features[label][left_channel]) for label in known if label != example.target)
                right_other = max(float(example.features[label][right_channel]) for label in known if label != example.target)
                left_margins.append(float(example.features[example.target][left_channel]) - left_other)
                right_margins.append(float(example.features[example.target][right_channel]) - right_other)
                left_top_5.append(left_ranking.index(example.target) < 5)
                right_top_5.append(right_ranking.index(example.target) < 5)
            output[f"{left_channel}+{right_channel}"] = {"games": len(left_margins), "true_margin_correlation": _pearson(left_margins, right_margins), "top_5_success_phi": _phi(left_top_5, right_top_5), "both_top_5": sum(a and b for a, b in zip(left_top_5, right_top_5, strict=True))}
    return output


def _group_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family = {family: classification_metrics([row for row in rows if row["family"] == family]) for family in FAMILIES}
    roles = sorted({str(row["role"]) for row in rows})
    by_role = {role: classification_metrics([row for row in rows if row["role"] == role]) for role in roles}
    return {"pooled": classification_metrics(rows), "by_family": by_family, "by_role": by_role, "abstention": _abstention(rows)}


class GleeBehaviorFusionAnalysis:
    """Run the first partial Stage 6 identity-fusion baseline without live authority."""

    def __init__(self, *, behavior_dir: Path, channel_dir: Path, assignment_dir: Path, output_dir: Path) -> None:
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.assignment_dir = assignment_dir.resolve()
        self.output_dir = output_dir.resolve()

    def _verify_source(self, directory: Path, artifact: str) -> Mapping[str, object]:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {}).get(artifact, {}).get("sha256")
        if expected and _file_digest(directory / artifact) != expected:
            raise ValueError(f"source artifact hash mismatch: {directory / artifact}")
        return manifest

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        lines = [
            "# GLEE activity-conditioned behavior fusion v1",
            "",
            "**Status:** Completed partial Stage 6 retrospective identity-fusion baseline; no posterior, identity route, policy cluster, prompt, dossier, or action is connected to live matchmaking.",
            "",
            "## Design boundary",
            "",
            "The experiment reconstructs the full Stage 2 capacity-aware activity and rating marginals, joins the Stage 5 timing, action, lexical, and discourse scores, and fits one pooled nonnegative conditional stacker on the chronological calibration block. The 379-game test suffix remains untouched by fitting. Stage 2 candidates rely on public completion pulses observed after each game, so the result evaluates retrospective deanonymization and cannot support live decision routing.",
            "",
            "## Untouched test result",
            "",
            "| Arm | Top-one | Top-5 | MRR | NLL | Brier | Unknown recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, result in summary["models"].items():
            metrics = result["metrics"]["pooled"]
            unknown = metrics.get("unknown_recall")
            lines.append(f"| {arm} | {metrics['top_one_accuracy']:.1%} | {metrics['top_5_accuracy']:.1%} | {metrics['mean_reciprocal_rank']:.3f} | {metrics['negative_log_likelihood']:.3f} | {metrics['multiclass_brier']:.3f} | {unknown:.1%} |" if unknown is not None else f"| {arm} | {metrics['top_one_accuracy']:.1%} | {metrics['top_5_accuracy']:.1%} | {metrics['mean_reciprocal_rank']:.3f} | {metrics['negative_log_likelihood']:.3f} | {metrics['multiclass_brier']:.3f} | — |")
        lines.extend(
            [
                "",
                "## Promotion boundary",
                "",
                "This v1 output supplies an identity posterior only. Policy-cluster inference is not implemented, `unknown` support is sparse, the activity evidence is postgame, and no observed-trajectory replay can establish the causal value or safety of choosing an opponent-specific strategy package. Live identity routing and SIC therefore remain prohibited.",
                "",
            ]
        )
        return "\n".join(lines)

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"behavior-fusion output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        behavior_manifest = self._verify_source(self.behavior_dir, "behavior-games.jsonl")
        channel_manifest = self._verify_source(self.channel_dir, "test-predictions.jsonl")
        assignment_manifest = self._verify_source(self.assignment_dir, "game-candidate-refs.jsonl")
        records = _load_jsonl(self.behavior_dir / "behavior-games.jsonl")
        channel_summary = json.loads((self.channel_dir / "summary.json").read_text(encoding="utf-8"))
        assignment_summary = json.loads((self.assignment_dir / "summary.json").read_text(encoding="utf-8"))
        candidate_rows = _load_jsonl(self.assignment_dir / "game-candidate-refs.jsonl")
        event_rows = _load_jsonl(self.assignment_dir / "candidate-events.jsonl")
        assignments, boundaries = chronological_game_splits(records, train_fraction=float(channel_summary["design"]["train_fraction"]), calibration_fraction=float(channel_summary["design"]["calibration_fraction"]))
        parameters = assignment_summary["parameters"]
        assignment_probabilities: dict[str, dict[str, dict[str, float]]] = {}
        assignment_reconstruction: dict[str, object] = {}
        for model in ASSIGNMENT_MODELS:
            probabilities, solver_summary = reconstruct_assignment_probabilities(candidate_rows, event_rows, model=model, samples=int(parameters["marginal_samples"]), temperature=float(parameters["marginal_temperature"]), frontier_sequence=int(assignment_summary["frontier_sequence"]), auction_epsilon=float(parameters["auction_epsilon"]))
            expected = _load_jsonl(self.assignment_dir / f"assignments-{model}.jsonl")
            parity = assignment_parity(probabilities, expected)
            if parity["mismatches"]:
                raise RuntimeError(f"Stage 2 {model} marginal reconstruction failed parity: {parity}")
            assignment_probabilities[model] = probabilities
            assignment_reconstruction[model] = {"solver": solver_summary, "parity": parity}
        profiles, vectors = build_channel_profiles(records, assignments, channel_summary)
        examples = build_fusion_examples(records, assignments, candidate_rows, event_rows, assignment_probabilities, profiles, vectors)
        calibration = [example for example in examples if example.split == "calibration"]
        test = [example for example in examples if example.split == "test"]
        if len(test) != 379:
            raise RuntimeError(f"unexpected Stage 6 test size: {len(test)}")
        models: dict[str, object] = {}
        prediction_rows: list[dict[str, object]] = []
        for arm, assignment_model in (("activity-raw", "activity-only"), ("stage2-joint-raw", "joint")):
            rows = [_prediction_record(example, arm, _normalized_raw(example, assignment_model)) for example in test]
            models[arm] = {"kind": "frozen-stage2-baseline", "metrics": _group_metrics(rows)}
            prediction_rows.extend(rows)
        for arm, features in FITTED_ARMS.items():
            stacker = fit_conditional_stacker(calibration, feature_names=features)
            rows = [_prediction_record(example, arm, stacker.probabilities(example)) for example in test]
            models[arm] = {"kind": "nonnegative-conditional-log-linear", "features": list(features), "ridge_lambda": RIDGE_LAMBDA, "iterations": OPTIMIZER_ITERATIONS, "learning_rate": OPTIMIZER_LEARNING_RATE, "weights": dict(stacker.weights), "scales": dict(stacker.scales), "unknown_bias": stacker.unknown_bias, "metrics": _group_metrics(rows)}
            prediction_rows.extend(rows)
        correlations = channel_error_correlations(test)
        summary = {
            "contract": BEHAVIOR_FUSION_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only-partial-stage-6",
            "sources": {"behavior_manifest_sha256": _file_digest(self.behavior_dir / "manifest.json"), "channel_manifest_sha256": _file_digest(self.channel_dir / "manifest.json"), "assignment_manifest_sha256": _file_digest(self.assignment_dir / "manifest.json"), "frontier_sequence": behavior_manifest.get("frontier_sequence")},
            "design": {"chronological_boundaries": boundaries, "calibration_games": len(calibration), "test_games": len(test), "fixed_ridge_lambda": RIDGE_LAMBDA, "optimizer_iterations": OPTIMIZER_ITERATIONS, "test_selected_tuning": False, "activity_evidence_timing": "postgame completion pulse; retrospective only", "identity_prior": "activity-conditioned", "policy_clusters_implemented": False},
            "assignment_reconstruction": assignment_reconstruction,
            "correlations": correlations,
            "models": models,
            "promotion": {"identity_routing_authority": False, "sic_authority": False, "decision_value_evaluated": False, "blockers": ["Stage 2 completion-pulse candidates are postgame", "policy clusters are not implemented", "unknown test support is sparse", "observed-trajectory replay is not a causal action-value estimate"], "next_gate": "complete Stage 3 pre-decision presence forecasting, then rerun fusion and strategy-routing replay under causal candidate evidence"},
        }
        prediction_rows.sort(key=lambda row: (str(row["started_at"]), str(row["arm"]), str(row["game_id"])))
        _write_json(self.output_dir / "summary.json", summary)
        _write_json(self.output_dir / "channel-error-correlations.json", correlations)
        _write_jsonl(self.output_dir / "test-predictions.jsonl", prediction_rows)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifacts = ("README.md", "summary.json", "channel-error-correlations.json", "test-predictions.jsonl")
        manifest = {
            "contract": BEHAVIOR_FUSION_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": summary["sources"],
            "implementation_sha256": {"glee_behavior_fusion_analysis.py": _file_digest(Path(__file__)), "glee_behavior_channel_analysis.py": _file_digest(Path(__file__).with_name("glee_behavior_channel_analysis.py")), "glee_joint_assignment_analysis.py": _file_digest(Path(__file__).with_name("glee_joint_assignment_analysis.py")), "glee_joint_assignment.py": _file_digest(Path(__file__).with_name("glee_joint_assignment.py"))},
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": BEHAVIOR_FUSION_CONTRACT, "output_dir": str(self.output_dir), "test_games": len(test), "models": {arm: value["metrics"]["pooled"] for arm, value in models.items()}, "promotion": summary["promotion"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
