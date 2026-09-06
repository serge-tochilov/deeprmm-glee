"""Whole-game rolling-origin validation for GLEE Negotiation v2.0."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .glee_negotiation_twin_v2 import ENGINE_VERSION, MODEL_VERSION, NegotiationGameEvidence, NegotiationModelConfig, NegotiationOpponentModelV2, PriorObservation, _sha, _sha_file, load_negotiation_corpus, model_receipt, prior_observations


VALIDATION_SCHEMA_VERSION = 1
VALIDATION_VERSION = f"negotiation-validation-{ENGINE_VERSION}-development"
MODEL_NAMES = ("adaptive_v2", "target_kernel", "population_kernel")
PRIMARY_MODEL = "adaptive_v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    _atomic_text(path, "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for value in values))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class NegotiationValidationConfig:
    """Frozen corpus eligibility and causal-origin settings."""

    min_games: int = 10
    warmup_games: int = 4

    def validate(self) -> None:
        if self.warmup_games < 1:
            raise ValueError("warmup_games must be positive")
        if self.min_games <= self.warmup_games:
            raise ValueError("min_games must exceed warmup_games")


def _response_prediction(probability: float, accepted: bool) -> dict[str, float]:
    bounded = min(1 - 1e-6, max(1e-6, probability))
    label = 1.0 if accepted else 0.0
    return {"probability": bounded, "nll": -math.log(bounded if accepted else 1 - bounded), "brier": (bounded - label) ** 2}


def _proposal_prediction(distribution: Any, observed: float) -> dict[str, float | int]:
    summary = distribution.as_dict()
    return {**summary, "nll": distribution.nll(observed), "absolute_error": abs(float(summary["mean"]) - observed), "squared_error": (float(summary["mean"]) - observed) ** 2, "covered_80": float(summary["q10"]) <= observed <= float(summary["q90"])}


def _metric_block(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    response_records = [record for record in records if record["action_type"] == "response"]
    proposal_records = [record for record in records if record["action_type"] == "proposal"]
    response: dict[str, dict[str, float | int | None]] = {}
    proposal: dict[str, dict[str, float | int | None]] = {}
    for model in MODEL_NAMES:
        if response_records:
            response[model] = {
                "count": len(response_records),
                "nll": _mean([float(record["predictions"][model]["nll"]) for record in response_records]),
                "brier": _mean([float(record["predictions"][model]["brier"]) for record in response_records]),
                "accuracy": _mean([float((float(record["predictions"][model]["probability"]) >= 0.5) == bool(record["actual"]["accepted"])) for record in response_records]),
            }
        else:
            response[model] = {"count": 0, "nll": None, "brier": None, "accuracy": None}
        if proposal_records:
            proposal[model] = {
                "count": len(proposal_records),
                "nll": _mean([float(record["predictions"][model]["nll"]) for record in proposal_records]),
                "mae": _mean([float(record["predictions"][model]["absolute_error"]) for record in proposal_records]),
                "rmse": math.sqrt(float(_mean([float(record["predictions"][model]["squared_error"]) for record in proposal_records]) or 0.0)),
                "interval_80_coverage": _mean([float(bool(record["predictions"][model]["covered_80"])) for record in proposal_records]),
            }
        else:
            proposal[model] = {"count": 0, "nll": None, "mae": None, "rmse": None, "interval_80_coverage": None}
    return {"response": response, "proposal": proposal}


def _macro_metric(records: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, dict[str, float | int | None]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    blocks = [_metric_block(group) for group in groups.values()]
    result: dict[str, dict[str, dict[str, float | int | None]]] = {"response": {}, "proposal": {}}
    metric_names = {"response": ("nll", "brier", "accuracy"), "proposal": ("nll", "mae", "rmse", "interval_80_coverage")}
    for action_type, names in metric_names.items():
        for model in MODEL_NAMES:
            model_values: dict[str, float | int | None] = {"group_count": 0}
            contributing = [block[action_type][model] for block in blocks if int(block[action_type][model]["count"] or 0) > 0]
            model_values["group_count"] = len(contributing)
            for name in names:
                values = [float(block[name]) for block in contributing if block[name] is not None]
                model_values[name] = _mean(values)
            result[action_type][model] = model_values
    return result


def _comparison(metric: dict[str, dict[str, dict[str, float | int | None]]], comparator: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for action_type, names in (("response", ("nll", "brier")), ("proposal", ("nll", "mae", "rmse"))):
        for name in names:
            primary = metric[action_type][PRIMARY_MODEL].get(name)
            other = metric[action_type][comparator].get(name)
            result[f"{action_type}_{name}"] = float(primary) - float(other) if primary is not None and other is not None else None
    return result


def _calibration(records: Sequence[dict[str, Any]], bins: int = 10) -> tuple[list[dict[str, float | int]], float | None]:
    selected = [record for record in records if record["action_type"] == "response"]
    result: list[dict[str, float | int]] = []
    weighted_error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [record for record in selected if low <= float(record["predictions"][PRIMARY_MODEL]["probability"]) < high or (index == bins - 1 and float(record["predictions"][PRIMARY_MODEL]["probability"]) == 1.0)]
        if not bucket:
            continue
        predicted = float(_mean([float(record["predictions"][PRIMARY_MODEL]["probability"]) for record in bucket]) or 0.0)
        observed = float(_mean([float(bool(record["actual"]["accepted"])) for record in bucket]) or 0.0)
        weighted_error += len(bucket) * abs(predicted - observed)
        result.append({"low": low, "high": high, "count": len(bucket), "mean_prediction": predicted, "observed_acceptance": observed})
    return result, weighted_error / len(selected) if selected else None


class NegotiationRollingValidationV2:
    """Evaluate one fixed Negotiation v2.0 model under causal whole-game origins."""

    def __init__(
        self,
        *,
        dossier_root: Path,
        output_dir: Path,
        project_root: Path,
        validation_config: NegotiationValidationConfig | None = None,
        model_config: NegotiationModelConfig | None = None,
        opponents: set[str] | None = None,
        bootstrap_replicates: int = 2000,
        bootstrap_seed: int = 20260810,
    ) -> None:
        self.dossier_root = dossier_root
        self.output_dir = output_dir
        self.project_root = project_root
        self.validation_config = validation_config or NegotiationValidationConfig()
        self.model_config = model_config or NegotiationModelConfig()
        self.validation_config.validate()
        self.model_config.validate()
        if bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        self.opponents = opponents
        self.bootstrap_replicates = bootstrap_replicates
        self.bootstrap_seed = bootstrap_seed
        self.model = NegotiationOpponentModelV2(self.model_config)

    def _eligible(self, games: Sequence[NegotiationGameEvidence]) -> dict[str, dict[str, str]]:
        grouped: dict[str, list[NegotiationGameEvidence]] = defaultdict(list)
        for game in games:
            grouped[game.opponent_id].append(game)
        return {
            opponent_id: {"id": opponent_id, "name": target_games[0].opponent_name}
            for opponent_id, target_games in grouped.items()
            if len(target_games) >= self.validation_config.min_games and (self.opponents is None or opponent_id in self.opponents or target_games[0].opponent_name in self.opponents)
        }

    def _records(self, games: Sequence[NegotiationGameEvidence], eligible: Mapping[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        origins: list[dict[str, Any]] = []
        for global_index, game in enumerate(games):
            if game.opponent_id not in eligible:
                continue
            prior_games = list(games[:global_index])
            prior_target_games = [prior for prior in prior_games if prior.opponent_id == game.opponent_id]
            if len(prior_target_games) < self.validation_config.warmup_games:
                continue
            observations, target_counts = prior_observations(prior_games)
            population_prior = [observation for observation in observations if observation.row.context.opponent_id != game.opponent_id]
            target_prior = [observation for observation in observations if observation.row.context.opponent_id == game.opponent_id]
            target_index = target_counts.get(game.opponent_id, 0)
            origin_index = len(origins)
            prefix: list[Any] = []
            game_records: list[dict[str, Any]] = []
            for decision_index, row in enumerate(game.rows):
                if row.action_type == "response":
                    forecasts = self.model.response_forecast(
                        row,
                        population_prior=population_prior,
                        target_prior=target_prior,
                        prefix=prefix,
                        current_global_game_index=global_index,
                        current_target_game_index=target_index,
                    )
                    actual = {"accepted": bool(row.accepted), "decision": row.decision, "offered_demand": row.offered_demand, "offered_opponent_surplus_share": row.offered_opponent_surplus_share}
                    predictions = {name: {**_response_prediction(float(forecast["probability"]), bool(row.accepted)), "effective_mass": float(forecast["effective_mass"])} for name, forecast in forecasts.items()}
                else:
                    distributions = self.model.proposal_forecast(
                        row,
                        population_prior=population_prior,
                        target_prior=target_prior,
                        prefix=prefix,
                        current_global_game_index=global_index,
                        current_target_game_index=target_index,
                    )
                    actual = {"proposal_demand": row.proposal_demand, "proposal_price_ratio": row.proposal_price_ratio, "proposal_opponent_surplus_share": row.proposal_opponent_surplus_share, "message_act": row.message_act}
                    predictions = {name: _proposal_prediction(distribution, float(row.proposal_demand)) for name, distribution in distributions.items()}
                record = {
                    "schema_version": VALIDATION_SCHEMA_VERSION,
                    "validation_version": VALIDATION_VERSION,
                    "model_version": MODEL_VERSION,
                    "origin_index": origin_index,
                    "decision_index": decision_index,
                    "game_id": game.game_id,
                    "game_outcome": game.outcome,
                    "opponent": eligible[game.opponent_id],
                    "opponent_id": game.opponent_id,
                    "opponent_name": game.opponent_name,
                    "action_type": row.action_type,
                    "prefix_action_count": len(prefix),
                    "prior_target_game_count": len(prior_target_games),
                    "prior_population_game_count": len(prior_games) - len(prior_target_games),
                    "context": row.as_dict()["context"],
                    "actual": actual,
                    "predictions": predictions,
                    "source": {"job_id": row.job_id, "job_path": row.job_path, "job_sha256": row.job_sha256},
                }
                game_records.append(record)
                prefix.append(row)
            records.extend(game_records)
            origins.append(
                {
                    "schema_version": VALIDATION_SCHEMA_VERSION,
                    "validation_version": VALIDATION_VERSION,
                    "model_version": MODEL_VERSION,
                    "origin_index": origin_index,
                    "game_id": game.game_id,
                    "opponent": eligible[game.opponent_id],
                    "completed_at": game.completed_at,
                    "completion_order": game.completion_order,
                    "prior_target_game_count": len(prior_target_games),
                    "prior_population_game_count": len(prior_games) - len(prior_target_games),
                    "decision_count": len(game_records),
                    "frontier": "All intergame evidence predates the target game; within-game updates use only opponent actions already visible in the target transcript.",
                }
            )
        return records, origins

    def _bootstrap(self, records: Sequence[dict[str, Any]]) -> dict[str, object]:
        by_opponent: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            by_opponent[str(record["opponent_id"])][str(record["game_id"])].append(record)
        opponent_ids = sorted(by_opponent)
        comparisons = ("target_kernel", "population_kernel")
        metric_paths = (("response", "nll"), ("response", "brier"), ("proposal", "nll"), ("proposal", "mae"), ("proposal", "rmse"))
        samples: dict[tuple[str, str, str], list[float]] = {(comparator, action, metric): [] for comparator in comparisons for action, metric in metric_paths}
        generator = random.Random(self.bootstrap_seed)
        if opponent_ids and self.bootstrap_replicates:
            for _replicate in range(self.bootstrap_replicates):
                selected: list[dict[str, Any]] = []
                for opponent_draw, opponent_id in enumerate(generator.choices(opponent_ids, k=len(opponent_ids))):
                    games = list(by_opponent[opponent_id].values())
                    for game_draw, game in enumerate(generator.choices(games, k=len(games))):
                        selected.extend({**record, "opponent_id": f"{opponent_draw}:{opponent_id}", "game_id": f"{opponent_draw}:{game_draw}:{record['game_id']}"} for record in game)
                macro = _macro_metric(selected, "opponent_id")
                for comparator in comparisons:
                    difference = _comparison(macro, comparator)
                    for action, metric in metric_paths:
                        value = difference[f"{action}_{metric}"]
                        if value is not None:
                            samples[(comparator, action, metric)].append(value)
        result: dict[str, object] = {}
        for (comparator, action, metric), values in samples.items():
            ordered = sorted(values)
            if ordered:
                low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
                high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
                favorable = sum(value < 0 for value in ordered) / len(ordered)
            else:
                low = high = favorable = None
            result[f"{comparator}:{action}_{metric}"] = {"low_95": low, "high_95": high, "primary_favorable_fraction": favorable}
        return {"method": "paired-2-stage-opponent-and-complete-game-cluster-bootstrap", "replicates": self.bootstrap_replicates, "random_seed": self.bootstrap_seed, "opponent_cluster_count": len(opponent_ids), "comparisons": result}

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite validation output: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        games, rejected = load_negotiation_corpus(self.dossier_root)
        eligible = self._eligible(games)
        records, origins = self._records(games, eligible)
        micro = _metric_block(records)
        game_macro = _macro_metric(records, "game_id")
        opponent_macro = _macro_metric(records, "opponent_id")
        calibration, calibration_error = _calibration(records)
        slices: dict[str, object] = {}
        slice_keys = {
            "opponent_buyer": lambda record: record["context"]["opponent_role"] == "buyer",
            "opponent_seller": lambda record: record["context"]["opponent_role"] == "seller",
            "complete_information": lambda record: bool(record["context"]["complete_information"]),
            "incomplete_information": lambda record: not bool(record["context"]["complete_information"]),
            "known_horizon": lambda record: bool(record["context"]["horizon_known"]),
            "unknown_horizon": lambda record: not bool(record["context"]["horizon_known"]),
            "one_round": lambda record: record["context"]["max_rounds"] == 1,
            "long_game": lambda record: int(record["context"]["round_number"]) > 10,
        }
        for name, predicate in slice_keys.items():
            selected = [record for record in records if predicate(record)]
            slices[name] = {"decision_count": len(selected), "metrics": _metric_block(selected)}
        per_opponent = []
        for opponent_id, opponent in sorted(eligible.items(), key=lambda item: item[1]["name"].casefold()):
            selected = [record for record in records if record["opponent_id"] == opponent_id]
            per_opponent.append({"opponent": opponent, "game_count": len({record["game_id"] for record in selected}), "decision_count": len(selected), "metrics": _metric_block(selected)})
        evaluation = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-negotiation-rolling-origin-evaluation-v2",
            "validation_version": VALIDATION_VERSION,
            "model_version": MODEL_VERSION,
            "status": "retrospective-development-shadow-only",
            "origin_count": len(origins),
            "decision_count": len(records),
            "response_count": sum(record["action_type"] == "response" for record in records),
            "proposal_count": sum(record["action_type"] == "proposal" for record in records),
            "opponent_count": len(eligible),
            "micro": micro,
            "complete_game_macro": game_macro,
            "opponent_macro": opponent_macro,
            "comparisons": {"micro": {comparator: _comparison(micro, comparator) for comparator in MODEL_NAMES if comparator != PRIMARY_MODEL}, "complete_game_macro": {comparator: _comparison(game_macro, comparator) for comparator in MODEL_NAMES if comparator != PRIMARY_MODEL}, "opponent_macro": {comparator: _comparison(opponent_macro, comparator) for comparator in MODEL_NAMES if comparator != PRIMARY_MODEL}},
            "response_calibration": calibration,
            "response_expected_calibration_error": calibration_error,
            "slices": slices,
            "per_opponent": per_opponent,
            "bootstrap": self._bootstrap(records),
            "interpretation_boundary": "Prediction is not a decision rule; retrospective fit cannot authorize live offer, response, message, or matchmaking changes.",
        }
        source_receipt = [
            {"game_id": game.game_id, "opponent_id": game.opponent_id, "completed_at": game.completed_at, "completion_order": game.completion_order, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256, "decision_count": len(game.rows)}
            for game in games
        ]
        corpus = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-negotiation-validation-corpus-v2",
            "validation_version": VALIDATION_VERSION,
            "game_count": len(games),
            "opponent_count": len({game.opponent_id for game in games}),
            "decision_count": sum(len(game.rows) for game in games),
            "eligible_opponents": list(eligible.values()),
            "rejected": list(rejected),
            "source_receipt": source_receipt,
            "source_receipt_sha256": _sha(source_receipt),
        }
        _atomic_json(self.output_dir / "corpus.json", corpus)
        _jsonl(self.output_dir / "origins.jsonl", origins)
        _jsonl(self.output_dir / "predictions.jsonl", records)
        _atomic_json(self.output_dir / "evaluation.json", evaluation)
        protocol_path = self.project_root / "protocols" / "glee-negotiation-opponent-model-v2.md"
        module_paths = {
            "model": self.project_root / "src" / "nommd_arena" / "glee_negotiation_twin_v2.py",
            "validation": self.project_root / "src" / "nommd_arena" / "glee_negotiation_validation_v2.py",
            "protocol": protocol_path,
        }
        manifest = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "kind": "glee-negotiation-validation-run-v2",
            "created_at": _now(),
            "validation_version": VALIDATION_VERSION,
            "model": model_receipt(self.model_config),
            "validation_config": asdict(self.validation_config),
            "dossier_root": str(self.dossier_root.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "artifacts": {name: {"path": path.name, "sha256": _sha_file(path)} for name, path in (("corpus", self.output_dir / "corpus.json"), ("origins", self.output_dir / "origins.jsonl"), ("predictions", self.output_dir / "predictions.jsonl"), ("evaluation", self.output_dir / "evaluation.json"))},
            "implementation": {name: {"path": str(path.relative_to(self.project_root)), "sha256": _sha_file(path)} for name, path in module_paths.items()},
        }
        manifest["manifest_sha256"] = _sha(manifest)
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return {"manifest": manifest, "corpus": {key: corpus[key] for key in ("game_count", "opponent_count", "decision_count", "source_receipt_sha256")}, "evaluation": {key: evaluation[key] for key in ("origin_count", "decision_count", "response_count", "proposal_count", "opponent_count", "micro", "complete_game_macro", "opponent_macro", "comparisons", "response_expected_calibration_error")}}
