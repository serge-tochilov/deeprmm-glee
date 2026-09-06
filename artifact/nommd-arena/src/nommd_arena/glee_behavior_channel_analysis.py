"""Evaluate collision-safe GLEE behavior channels under chronological open-set splits."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_activity_eda import _file_digest


BEHAVIOR_CHANNEL_CONTRACT = "glee-behavior-channel-evaluation-v1"
UNKNOWN_ID = "__unknown__"
CHANNELS = ("timing", "action", "lexical", "discourse")
FAMILIES = ("bargaining", "negotiation", "persuasion")
_ALPHAS = (5.0, 20.0, 100.0)
_TEMPERATURES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_UNKNOWN_BIASES = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


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


def _timestamp(value: object) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _phase(value: object) -> str:
    number = min(1.0, max(0.0, _number(value)))
    return "early" if number < 1.0 / 3.0 else "middle" if number < 2.0 / 3.0 else "late"


def _context(move: Mapping[str, object]) -> str:
    context = move.get("context") if isinstance(move.get("context"), Mapping) else {}
    role = str(context.get("opponent_role") or "none").casefold()
    information = "complete" if context.get("complete_information") is True else "incomplete"
    horizon = "known" if context.get("horizon_known") is True else "unknown"
    messages = "enabled" if context.get("messages_allowed") is True else "disabled"
    return f"kind={move.get('kind')}|role={role}|information={information}|horizon={horizon}|phase={_phase(context.get('round_phase'))}|messages={messages}"


def _bounded_bin(value: float, *, width: float, lower: float, upper: float) -> str:
    bounded = min(upper, max(lower, value))
    index = math.floor((bounded - lower) / width + 1e-12)
    return str(index)


def _delay_bin(delay_ms: float) -> str:
    seconds = max(0.0, delay_ms / 1000.0)
    boundaries = (1.0, 3.0, 8.0, 15.0, 30.0, 60.0, 90.0, 105.0, 115.0)
    for index, boundary in enumerate(boundaries):
        if seconds < boundary:
            return str(index)
    return str(len(boundaries))


def _delay_change_bin(current_ms: float, previous_ms: float) -> str:
    log_ratio = math.log2((max(0.0, current_ms) + 1000.0) / (max(0.0, previous_ms) + 1000.0))
    return _bounded_bin(log_ratio, width=0.75, lower=-4.5, upper=4.5)


def _count_bin(value: object, boundaries: Sequence[float]) -> str:
    number = max(0.0, _number(value))
    for index, boundary in enumerate(boundaries):
        if number < boundary:
            return str(index)
    return str(len(boundaries))


def _ratio_bin(value: object) -> str:
    return _bounded_bin(_number(value), width=0.1, lower=0.0, upper=1.0)


def _decision(value: object) -> str:
    normalized = str(value or "none").strip().casefold().replace(" ", "-").replace("_", "-")
    return normalized[:48] or "none"


def timing_features(game: Mapping[str, object]) -> Counter[str]:
    """Build state-conditioned exact-server-delay tokens for one whole game."""
    features: Counter[str] = Counter()
    previous_delay: float | None = None
    for move in game.get("moves", []):
        if not isinstance(move, Mapping) or move.get("response_time_ms") is None:
            continue
        delay = max(0.0, _number(move.get("response_time_ms")))
        context = _context(move)
        features[f"timing|{context}|delay={_delay_bin(delay)}"] += 1
        features[f"timing|kind={move.get('kind')}|deadline={int(delay >= 100_000.0)}"] += 1
        if previous_delay is not None:
            features[f"timing-sequence|{context}|change={_delay_change_bin(delay, previous_delay)}"] += 1
        previous_delay = delay
    return features


def action_features(game: Mapping[str, object]) -> Counter[str]:
    """Build scale-normalized, state-conditioned opponent-choice tokens for one whole game."""
    features: Counter[str] = Counter()
    previous_by_kind: dict[str, float] = {}
    last_move: Mapping[str, object] | None = None
    for move in game.get("moves", []):
        if not isinstance(move, Mapping):
            continue
        kind = str(move.get("kind") or "unknown")
        context = _context(move)
        value = _number(move.get("action_value"))
        features[f"action|{context}|value={_bounded_bin(value, width=0.1, lower=-2.0, upper=2.0)}"] += 1
        decision = _decision(move.get("decision"))
        if decision != "none":
            features[f"decision|{context}|value={_bounded_bin(value, width=0.1, lower=-2.0, upper=2.0)}|decision={decision}"] += 1
        if kind in previous_by_kind:
            change = value - previous_by_kind[kind]
            features[f"action-sequence|kind={kind}|change={_bounded_bin(change, width=0.05, lower=-1.0, upper=1.0)}"] += 1
            features[f"action-sequence|kind={kind}|repeat={int(abs(change) <= 0.005)}"] += 1
        previous_by_kind[kind] = value
        last_move = move
    if last_move is not None:
        features[f"terminal-style|kind={last_move.get('kind')}|decision={_decision(last_move.get('decision'))}"] += 1
    return features


def lexical_features(game: Mapping[str, object]) -> Counter[str]:
    """Build hashed lexical and visible style tokens while excluding exact-message hashes and prose."""
    features: Counter[str] = Counter()
    for move in game.get("moves", []):
        if not isinstance(move, Mapping) or not isinstance(move.get("language"), Mapping):
            continue
        language = move["language"]
        context = _context(move)
        present = language.get("present") is True
        features[f"lexical-presence|{context}|present={int(present)}"] += 1
        if not present:
            continue
        style = language.get("style") if isinstance(language.get("style"), Mapping) else {}
        features[f"style|words={_count_bin(style.get('words'), (4, 8, 16, 32, 64))}"] += 1
        features[f"style|chars={_count_bin(style.get('chars'), (16, 32, 64, 128, 256))}"] += 1
        features[f"style|sentences={_count_bin(style.get('sentences'), (2, 3, 5))}"] += 1
        features[f"style|uppercase={_ratio_bin(style.get('uppercase_ratio'))}"] += 1
        features[f"style|digits={_ratio_bin(style.get('digit_ratio'))}"] += 1
        for key in ("question_marks", "exclamation_marks", "commas", "semicolons", "currency_marks", "percent_marks", "decimal_numbers", "contractions"):
            features[f"style|{key}={_count_bin(style.get(key), (1, 2, 4))}"] += 1
        opening = style.get("opening_sha256")
        ending = style.get("ending_sha256")
        if opening:
            features[f"opening|{opening}"] += 1
        if ending:
            features[f"ending|{ending}"] += 1
        lexemes = language.get("hashed_lexemes") if isinstance(language.get("hashed_lexemes"), Mapping) else {}
        for feature_hash, raw_count in lexemes.items():
            count = max(0, min(3, int(_number(raw_count))))
            if count:
                features[f"lexeme|{feature_hash}"] += count
    return features


def discourse_features(game: Mapping[str, object]) -> Counter[str]:
    """Build context-conditioned communicative-act tokens independently of lexical hashes."""
    features: Counter[str] = Counter()
    previous_acts: tuple[str, ...] | None = None
    for move in game.get("moves", []):
        if not isinstance(move, Mapping) or not isinstance(move.get("language"), Mapping):
            continue
        language = move["language"]
        context = _context(move)
        acts = tuple(sorted(str(act) for act in language.get("discourse_acts", []) if act)) or ("other",)
        family_act = _decision(language.get("family_act"))
        features[f"discourse-presence|{context}|present={int(language.get('present') is True)}"] += 1
        features[f"family-act|{context}|act={family_act}"] += 1
        for act in acts:
            features[f"discourse-act|{context}|act={act}"] += 1
        features[f"discourse-combination|{context}|acts={'+'.join(acts)}"] += 1
        if previous_acts is not None:
            features[f"discourse-sequence|from={'+'.join(previous_acts)}|to={'+'.join(acts)}"] += 1
        previous_acts = acts
    return features


def channel_features(game: Mapping[str, object], channel: str) -> Counter[str]:
    """Dispatch one game to one behavior channel without crossing channel boundaries."""
    if channel == "timing":
        return timing_features(game)
    if channel == "action":
        return action_features(game)
    if channel == "lexical":
        return lexical_features(game)
    if channel == "discourse":
        return discourse_features(game)
    raise ValueError(f"unsupported behavior channel: {channel}")


def chronological_game_splits(records: Sequence[Mapping[str, object]], *, train_fraction: float = 0.6, calibration_fraction: float = 0.2) -> tuple[dict[str, str], dict[str, object]]:
    """Split whole games by family and purge games crossing either chronological boundary."""
    if train_fraction <= 0.0 or calibration_fraction <= 0.0 or train_fraction + calibration_fraction >= 1.0:
        raise ValueError("chronological fractions must leave positive train, calibration, and test blocks")
    assignments: dict[str, str] = {}
    boundaries: dict[str, object] = {}
    for family in FAMILIES:
        selected = sorted((record for record in records if record.get("family") == family), key=lambda record: (_timestamp(record["started_at"]), str(record["game_id"])))
        if len(selected) < 5:
            continue
        train_end = max(1, min(len(selected) - 2, math.floor(len(selected) * train_fraction)))
        calibration_end = max(train_end + 1, min(len(selected) - 1, math.floor(len(selected) * (train_fraction + calibration_fraction))))
        calibration_start = _timestamp(selected[train_end]["started_at"])
        test_start = _timestamp(selected[calibration_end]["started_at"])
        counts: Counter[str] = Counter()
        for record in selected:
            started = _timestamp(record["started_at"])
            completed = _timestamp(record["completed_at"])
            if started < calibration_start:
                split = "train" if completed < calibration_start else "purged-train-calibration-overlap"
            elif started < test_start:
                split = "calibration" if completed < test_start else "purged-calibration-test-overlap"
            else:
                split = "test"
            assignments[str(record["game_id"])] = split
            counts[split] += 1
        boundaries[family] = {
            "calibration_start": datetime.fromtimestamp(calibration_start, tz=timezone.utc).isoformat(),
            "test_start": datetime.fromtimestamp(test_start, tz=timezone.utc).isoformat(),
            "counts": dict(sorted(counts.items())),
        }
    return assignments, boundaries


@dataclass(frozen=True)
class FingerprintProfile:
    """One hierarchical token profile with a population background and uniform identity prior."""

    candidates: tuple[str, ...]
    vocabulary: frozenset[str]
    global_probability: Mapping[str, float]
    identity_counts: Mapping[str, Mapping[str, float]]
    identity_totals: Mapping[str, float]
    alpha: float

    def scores(self, features: Mapping[str, int]) -> tuple[dict[str, float], int]:
        filtered = {token: float(count) for token, count in features.items() if token in self.vocabulary and count > 0}
        evidence = int(sum(filtered.values()))
        if not filtered:
            return {candidate: 0.0 for candidate in self.candidates}, 0
        scale = math.sqrt(evidence)
        scores: dict[str, float] = {}
        for candidate in self.candidates:
            counts = self.identity_counts[candidate]
            total = self.identity_totals[candidate]
            score = evidence * math.log(self.alpha / (total + self.alpha))
            for token, observed_count in filtered.items():
                identity_count = float(counts.get(token, 0.0))
                if identity_count > 0.0:
                    score += observed_count * math.log1p(identity_count / (self.alpha * self.global_probability[token]))
            scores[candidate] = score / scale
        return scores, evidence


def fit_fingerprint_profile(records: Sequence[Mapping[str, object]], vectors: Mapping[str, Counter[str]], *, candidates: Sequence[str], alpha: float, channel: str) -> FingerprintProfile:
    """Fit one dependency-free hierarchical multinomial profile from whole-game vectors."""
    minimum_document_frequency = 2 if channel == "lexical" else 1
    document_frequency: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    identity_document_frequency: dict[str, Counter[str]] = defaultdict(Counter)
    raw_identity_counts: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_set = set(candidates)
    for record in records:
        vector = vectors[str(record["game_id"])]
        document_frequency.update(vector.keys())
        global_counts.update(vector)
        identity = str(record["public_player_id"])
        if identity in candidate_set:
            identity_document_frequency[identity].update(vector.keys())
            raw_identity_counts[identity].update(vector)
    vocabulary = frozenset(token for token, count in document_frequency.items() if count >= minimum_document_frequency)
    global_total = sum(global_counts[token] for token in vocabulary)
    vocabulary_size = max(1, len(vocabulary))
    background_smoothing = 0.1
    denominator = global_total + background_smoothing * vocabulary_size
    global_probability = {token: (global_counts[token] + background_smoothing) / denominator for token in vocabulary}
    identity_counts: dict[str, dict[str, float]] = {}
    identity_totals: dict[str, float] = {}
    for candidate in sorted(candidates):
        filtered: dict[str, float] = {}
        for token, count in raw_identity_counts[candidate].items():
            if token not in vocabulary:
                continue
            if channel == "lexical" and token.startswith(("lexeme|", "opening|", "ending|")) and identity_document_frequency[candidate][token] < 2:
                continue
            filtered[token] = float(count)
        identity_counts[candidate] = filtered
        identity_totals[candidate] = sum(filtered.values())
    return FingerprintProfile(candidates=tuple(sorted(candidates)), vocabulary=vocabulary, global_probability=global_probability, identity_counts=identity_counts, identity_totals=identity_totals, alpha=alpha)


def _probabilities(scores: Mapping[str, float], *, temperature: float, unknown_bias: float) -> dict[str, float]:
    candidate_count = max(1, len(scores))
    logits = {candidate: score / temperature for candidate, score in scores.items()}
    logits[UNKNOWN_ID] = math.log(candidate_count) + unknown_bias
    maximum = max(logits.values())
    weights = {label: math.exp(logit - maximum) for label, logit in logits.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _prediction_record(record: Mapping[str, object], channel: str, candidates: Sequence[str], probabilities: Mapping[str, float], evidence_tokens: int) -> dict[str, object]:
    truth_id = str(record["public_player_id"])
    truth = truth_id if truth_id in candidates else UNKNOWN_ID
    ranking = sorted(probabilities, key=lambda label: (-probabilities[label], label))
    candidate_ranking = [label for label in ranking if label != UNKNOWN_ID]
    predicted = ranking[0]
    true_probability = max(1e-15, probabilities.get(truth, 0.0))
    return {
        "contract": BEHAVIOR_CHANNEL_CONTRACT,
        "game_id": record["game_id"],
        "family": record["family"],
        "channel": channel,
        "started_at": record["started_at"],
        "true_public_player_id": truth_id,
        "evaluation_target": truth,
        "predicted_target": predicted,
        "true_rank": ranking.index(truth) + 1,
        "known_candidate_rank": candidate_ranking.index(truth) + 1 if truth != UNKNOWN_ID else None,
        "true_probability": true_probability,
        "confidence": probabilities[predicted],
        "correct": predicted == truth,
        "evidence_tokens": evidence_tokens,
        "top_candidates": [{"target": label, "probability": probabilities[label]} for label in ranking[:5]],
        "nll": -math.log(true_probability),
        "brier": sum((probability - float(label == truth)) ** 2 for label, probability in probabilities.items()),
    }


def _expected_calibration_error(rows: Sequence[Mapping[str, object]], bins: int = 10) -> float | None:
    if not rows:
        return None
    buckets: list[list[Mapping[str, object]]] = [[] for _ in range(bins)]
    for row in rows:
        confidence = min(1.0, max(0.0, float(row["confidence"])))
        buckets[min(bins - 1, int(confidence * bins))].append(row)
    total = len(rows)
    return sum(len(bucket) / total * abs(sum(float(row["confidence"]) for row in bucket) / len(bucket) - sum(float(bool(row["correct"])) for row in bucket) / len(bucket)) for bucket in buckets if bucket)


def classification_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute game-level open-set ranking and calibration metrics."""
    if not rows:
        return {"games": 0}
    known = [row for row in rows if row["evaluation_target"] != UNKNOWN_ID]
    unknown = [row for row in rows if row["evaluation_target"] == UNKNOWN_ID]
    predicted_unknown = [row for row in rows if row["predicted_target"] == UNKNOWN_ID]
    identity_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in known:
        identity_groups[str(row["evaluation_target"])].append(row)
    always_unknown_accuracy = len(unknown) / len(rows)
    return {
        "games": len(rows),
        "evidence_games": sum(int(row["evidence_tokens"]) > 0 for row in rows),
        "evidence_coverage": sum(int(row["evidence_tokens"]) > 0 for row in rows) / len(rows),
        "top_one_accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "always_unknown_top_one_accuracy": always_unknown_accuracy,
        "top_one_lift_over_always_unknown": sum(bool(row["correct"]) for row in rows) / len(rows) - always_unknown_accuracy,
        "top_5_accuracy": sum(int(row["true_rank"]) <= 5 for row in rows) / len(rows),
        "mean_reciprocal_rank": sum(1.0 / int(row["true_rank"]) for row in rows) / len(rows),
        "negative_log_likelihood": sum(float(row["nll"]) for row in rows) / len(rows),
        "multiclass_brier": sum(float(row["brier"]) for row in rows) / len(rows),
        "expected_calibration_error": _expected_calibration_error(rows),
        "known_games": len(known),
        "known_top_one_accuracy": sum(bool(row["correct"]) for row in known) / len(known) if known else None,
        "known_top_5_accuracy": sum(int(row["true_rank"]) <= 5 for row in known) / len(known) if known else None,
        "known_mean_reciprocal_rank": sum(1.0 / int(row["true_rank"]) for row in known) / len(known) if known else None,
        "known_conditional_top_one_accuracy": sum(int(row["known_candidate_rank"]) == 1 for row in known) / len(known) if known else None,
        "known_conditional_top_5_accuracy": sum(int(row["known_candidate_rank"]) <= 5 for row in known) / len(known) if known else None,
        "known_conditional_mean_reciprocal_rank": sum(1.0 / int(row["known_candidate_rank"]) for row in known) / len(known) if known else None,
        "known_identity_macro_top_one": sum(sum(bool(row["correct"]) for row in group) / len(group) for group in identity_groups.values()) / len(identity_groups) if identity_groups else None,
        "known_identity_macro_conditional_top_one": sum(sum(int(row["known_candidate_rank"]) == 1 for row in group) / len(group) for group in identity_groups.values()) / len(identity_groups) if identity_groups else None,
        "unknown_games": len(unknown),
        "unknown_recall": sum(row["predicted_target"] == UNKNOWN_ID for row in unknown) / len(unknown) if unknown else None,
        "unknown_precision": sum(row["evaluation_target"] == UNKNOWN_ID for row in predicted_unknown) / len(predicted_unknown) if predicted_unknown else None,
        "predicted_unknown_games": len(predicted_unknown),
    }


def _select_calibration(train: Sequence[Mapping[str, object]], calibration: Sequence[Mapping[str, object]], vectors: Mapping[str, Counter[str]], *, candidates: Sequence[str], channel: str) -> tuple[dict[str, float], dict[str, object]]:
    trials: list[tuple[float, float, float, float, float, dict[str, object]]] = []
    for alpha in _ALPHAS:
        profile = fit_fingerprint_profile(train, vectors, candidates=candidates, alpha=alpha, channel=channel)
        score_rows = []
        for record in calibration:
            scores, evidence = profile.scores(vectors[str(record["game_id"])])
            score_rows.append((record, scores, evidence))
        for temperature in _TEMPERATURES:
            for unknown_bias in _UNKNOWN_BIASES:
                predictions = [_prediction_record(record, channel, candidates, _probabilities(scores, temperature=temperature, unknown_bias=unknown_bias), evidence) for record, scores, evidence in score_rows]
                metrics = classification_metrics(predictions)
                trials.append((float(metrics["negative_log_likelihood"]), float(metrics["multiclass_brier"]), alpha, temperature, unknown_bias, metrics))
    if not trials:
        raise ValueError("channel calibration requires at least one calibration game")
    trials.sort(key=lambda trial: trial[:5])
    nll, brier, alpha, temperature, unknown_bias, metrics = trials[0]
    return {"alpha": alpha, "temperature": temperature, "unknown_bias": unknown_bias}, {"selected": metrics, "candidate_trials": len(trials), "nll_gap_to_second": trials[1][0] - nll if len(trials) > 1 else None, "brier": brier}


def _channel_evaluation(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str], *, family: str, channel: str, minimum_profile_games: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    family_records = [record for record in records if record.get("family") == family and assignments.get(str(record["game_id"])) in {"train", "calibration", "test"}]
    vectors = {str(record["game_id"]): channel_features(record, channel) for record in family_records}
    train = [record for record in family_records if assignments[str(record["game_id"])] == "train"]
    calibration = [record for record in family_records if assignments[str(record["game_id"])] == "calibration"]
    test = [record for record in family_records if assignments[str(record["game_id"])] == "test"]
    support: Counter[str] = Counter(str(record["public_player_id"]) for record in train if vectors[str(record["game_id"])])
    candidates = tuple(sorted(identity for identity, count in support.items() if count >= minimum_profile_games))
    if not candidates or not calibration or not test:
        return {"status": "insufficient-support", "candidate_public_player_ids": len(candidates), "counts": {"train": len(train), "calibration": len(calibration), "test": len(test)}}, []
    selected, calibration_report = _select_calibration(train, calibration, vectors, candidates=candidates, channel=channel)
    final_fit = train + calibration
    profile = fit_fingerprint_profile(final_fit, vectors, candidates=candidates, alpha=selected["alpha"], channel=channel)
    predictions: list[dict[str, object]] = []
    for record in test:
        scores, evidence = profile.scores(vectors[str(record["game_id"])])
        probabilities = _probabilities(scores, temperature=selected["temperature"], unknown_bias=selected["unknown_bias"])
        predictions.append(_prediction_record(record, channel, candidates, probabilities, evidence))
    metrics = classification_metrics(predictions)
    evidence_metrics = classification_metrics([row for row in predictions if int(row["evidence_tokens"]) > 0])
    uniform_known_mrr = sum(1.0 / rank for rank in range(1, len(candidates) + 1)) / len(candidates)
    return {
        "status": "offline-shadow-only",
        "counts": {"train": len(train), "calibration": len(calibration), "test": len(test), "final_fit": len(final_fit)},
        "candidate_public_player_ids": len(candidates),
        "candidate_support": {"minimum_profile_games": minimum_profile_games, "minimum": min((support[candidate] for candidate in candidates), default=0), "median": sorted(support[candidate] for candidate in candidates)[len(candidates) // 2] if candidates else 0, "maximum": max((support[candidate] for candidate in candidates), default=0)},
        "selected_hyperparameters": selected,
        "calibration": calibration_report,
        "final_profile": {"vocabulary_size": len(profile.vocabulary), "population_games": len(final_fit), "candidate_ids": list(candidates), "uniform_open_set_class_chance": 1.0 / (len(candidates) + 1), "uniform_known_ranking": {"top_one_accuracy": 1.0 / len(candidates), "top_5_accuracy": min(5, len(candidates)) / len(candidates), "mean_reciprocal_rank": uniform_known_mrr}},
        "test": metrics,
        "test_with_evidence": evidence_metrics,
    }, predictions


class GleeBehaviorChannelAnalysis:
    """Run separate, non-operational fingerprint-channel evaluations from one frozen corpus."""

    def __init__(self, *, behavior_dir: Path, output_dir: Path, train_fraction: float = 0.6, calibration_fraction: float = 0.2, minimum_profile_games: int = 3) -> None:
        self.behavior_dir = behavior_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.train_fraction = train_fraction
        self.calibration_fraction = calibration_fraction
        self.minimum_profile_games = minimum_profile_games

    def _load(self) -> tuple[list[dict[str, object]], Mapping[str, object]]:
        manifest_path = self.behavior_dir / "manifest.json"
        corpus_path = self.behavior_dir / "behavior-games.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {}).get("behavior-games.jsonl", {}).get("sha256")
        if expected and _file_digest(corpus_path) != expected:
            raise ValueError("behavior-corpus artifact hash mismatch")
        records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
        return records, manifest

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        lines = [
            "# GLEE separate behavior-channel evaluation v1",
            "",
            "**Status:** Completed offline Stage 5 channel evaluation; no model, probability, identity route, prompt, dossier, or action is connected to live matchmaking.",
            "",
            "## Design",
            "",
            "The evaluation fits timing, action, lexical, and discourse fingerprints separately within each game family. Every split is chronological and game-disjoint, games crossing a boundary are purged, identity priors are uniform, and only exact temporal public IDs enter. An identity needs repeated training games with channel evidence to become a candidate; every other identity maps to an explicit `unknown` target.",
            "",
            "Profiles use population-smoothed token likelihood ratios. Calibration selects profile shrinkage, score temperature, and unknown bias without reading the test suffix. Lexical profiles exclude exact-message hashes, require hashed lexical, opening, and ending features to recur across at least 2 training games for the same candidate, and retain no raw prose.",
            "",
            "## Untouched chronological suffix",
            "",
            "| Family | Channel | Candidates | Test games | Evidence | Known rank top-one | Known rank top-5 | Open-set top-one | Lift over always-unknown | Unknown recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for family in FAMILIES:
            for channel in CHANNELS:
                result = summary["families"][family][channel]
                if result.get("status") == "insufficient-support":
                    lines.append(f"| {family.title()} | {channel.title()} | {result['candidate_public_player_ids']} | {result['counts']['test']} | — | — | — | — | — | — |")
                    continue
                metrics = result["test"]
                unknown_recall = metrics["unknown_recall"]
                lines.append(f"| {family.title()} | {channel.title()} | {result['candidate_public_player_ids']} | {metrics['games']} | {metrics['evidence_coverage']:.1%} | {metrics['known_conditional_top_one_accuracy']:.1%} | {metrics['known_conditional_top_5_accuracy']:.1%} | {metrics['top_one_accuracy']:.1%} | {metrics['top_one_lift_over_always_unknown']:+.1%} | {unknown_recall:.1%} |" if unknown_recall is not None else f"| {family.title()} | {channel.title()} | {result['candidate_public_player_ids']} | {metrics['games']} | {metrics['evidence_coverage']:.1%} | {metrics['known_conditional_top_one_accuracy']:.1%} | {metrics['known_conditional_top_5_accuracy']:.1%} | {metrics['top_one_accuracy']:.1%} | {metrics['top_one_lift_over_always_unknown']:+.1%} | — |")
        lines.extend(
            [
                "",
                "## Boundary",
                "",
                "The `unknown` target means absent or too sparse at the training cutoff for this channel; it is a defensible open-set proxy, not ground truth for an anonymous live opponent. Separate-channel accuracy measures retrospective identifiability under this frozen sample and does not authorize identity fusion, online routing, or SIC. Stage 6 may consume these channels only after inspecting support, calibration, correlated errors, and incremental value over corrected activity evidence.",
                "",
            ]
        )
        return "\n".join(lines)

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"behavior-channel output directory is not empty: {self.output_dir}")
        if self.minimum_profile_games < 2:
            raise ValueError("minimum profile games must be at least 2")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records, source_manifest = self._load()
        assignments, boundaries = chronological_game_splits(records, train_fraction=self.train_fraction, calibration_fraction=self.calibration_fraction)
        family_results: dict[str, object] = {}
        all_predictions: list[dict[str, object]] = []
        for family in FAMILIES:
            family_results[family] = {}
            for channel in CHANNELS:
                result, predictions = _channel_evaluation(records, assignments, family=family, channel=channel, minimum_profile_games=self.minimum_profile_games)
                family_results[family][channel] = result
                all_predictions.extend(predictions)
        pooled: dict[str, object] = {}
        for channel in CHANNELS:
            selected = [row for row in all_predictions if row["channel"] == channel]
            pooled[channel] = {"test": classification_metrics(selected), "test_with_evidence": classification_metrics([row for row in selected if int(row["evidence_tokens"]) > 0])}
        summary = {
            "contract": BEHAVIOR_CHANNEL_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only",
            "source": {"behavior_dir": str(self.behavior_dir), "behavior_manifest_sha256": _file_digest(self.behavior_dir / "manifest.json"), "frontier_sequence": source_manifest.get("frontier_sequence")},
            "design": {"train_fraction": self.train_fraction, "calibration_fraction": self.calibration_fraction, "test_fraction": 1.0 - self.train_fraction - self.calibration_fraction, "minimum_profile_games": self.minimum_profile_games, "identity_prior": "uniform", "unknown_semantics": "not enrolled for this family-channel at the training cutoff", "lexical_single-game_authentication": False, "live_authority": False},
            "split_boundaries": boundaries,
            "families": family_results,
            "pooled": pooled,
            "promotion": {"identity_routing_authority": False, "sic_authority": False, "next_gate": "inspect channel support, calibration, correlated errors, and incremental value before frozen Stage 6 fusion"},
        }
        all_predictions.sort(key=lambda row: (str(row["started_at"]), str(row["family"]), str(row["channel"]), str(row["game_id"])))
        _write_json(self.output_dir / "summary.json", summary)
        _write_jsonl(self.output_dir / "test-predictions.jsonl", all_predictions)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifacts = ("README.md", "summary.json", "test-predictions.jsonl")
        manifest = {
            "contract": BEHAVIOR_CHANNEL_CONTRACT,
            "schema_version": 1,
            "source_behavior_manifest_sha256": summary["source"]["behavior_manifest_sha256"],
            "implementation_sha256": _file_digest(Path(__file__)),
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": BEHAVIOR_CHANNEL_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": summary["source"]["frontier_sequence"], "pooled": pooled, "promotion": summary["promotion"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
