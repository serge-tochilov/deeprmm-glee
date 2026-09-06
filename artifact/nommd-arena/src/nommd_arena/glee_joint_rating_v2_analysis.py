"""Retrospective exact-configuration hybrid for GLEE displayed-rating deltas."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_joint_rating_analysis import FAMILY_BASE_FEATURES, RIDGE_GRID, _quantile, _structural_target, eta_for_game_count, predict_display_delta, prediction_metrics
from .glee_negotiation_rating_v2_4 import RidgeRatingModel, fit_ridge


JOINT_RATING_V2_CONTRACT = "glee-joint-rating-reconstruction-v2"
EXPECTED_V1_MANIFEST_SHA256 = "3bcb817869ad85a7869010691bc394fa5f76e0ab6381363f3f5ceabf209ee8a8"
ALPHA_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
CROSS_FIT_FOLDS = 5
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260812
PAIR_FEATURES = ("bias", "own_structural", "other_structural", "own_direct", "other_direct", "target_self", "own_pregame_scaled", "other_pregame_scaled", "own_log_count", "other_log_count")


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


def cross_fit_fold(game_id: str, folds: int = CROSS_FIT_FOLDS) -> int:
    """Assign a whole game deterministically to one cross-fit fold."""
    if folds < 2:
        raise ValueError("cross-fitting requires at least 2 folds")
    return int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def empirical_midrank(payoff: float, sorted_payoffs: Sequence[float], *, remove_one: bool = False) -> tuple[float | None, int]:
    """Return a smoothed empirical percentile and effective support, optionally leaving one tied observation out."""
    left = bisect.bisect_left(sorted_payoffs, payoff)
    right = bisect.bisect_right(sorted_payoffs, payoff)
    count = len(sorted_payoffs)
    equal = right - left
    if remove_one:
        if equal < 1:
            raise ValueError("leave-one-out payoff is absent from its configuration reference")
        count -= 1
        equal -= 1
    if count == 0:
        return None, 0
    return (left + 0.5 * equal + 0.5) / (count + 1.0), count


def shrinkage_weight(support: int, alpha: float) -> float:
    """Return the bounded empirical or residual weight for one support count."""
    if support < 0 or alpha <= 0:
        raise ValueError("shrinkage requires nonnegative support and positive alpha")
    return support / (support + alpha)


def asymmetric_interval(residuals: Sequence[float]) -> dict[str, float]:
    """Fit family-level asymmetric 80% and 95% residual offsets."""
    if not residuals:
        raise ValueError("interval calibration requires residuals")
    return {
        "lower_80": float(_quantile(residuals, 0.10)),
        "upper_80": float(_quantile(residuals, 0.90)),
        "lower_95": float(_quantile(residuals, 0.025)),
        "upper_95": float(_quantile(residuals, 0.975)),
    }


def _interval_metrics(rows: Sequence[Mapping[str, object]], interval: Mapping[str, float]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0, "coverage_80": 0.0, "coverage_95": 0.0, "below_80": 0.0, "above_80": 0.0, "below_95": 0.0, "above_95": 0.0, "width_80": 0.0, "width_95": 0.0}
    residuals = [float(row["actual_rating_delta"]) - float(row["hybrid_prediction"]) for row in rows]
    lower_80 = float(interval["lower_80"])
    upper_80 = float(interval["upper_80"])
    lower_95 = float(interval["lower_95"])
    upper_95 = float(interval["upper_95"])
    return {
        "count": len(rows),
        "coverage_80": statistics.fmean(float(lower_80 <= value <= upper_80) for value in residuals),
        "coverage_95": statistics.fmean(float(lower_95 <= value <= upper_95) for value in residuals),
        "below_80": statistics.fmean(float(value < lower_80) for value in residuals),
        "above_80": statistics.fmean(float(value > upper_80) for value in residuals),
        "below_95": statistics.fmean(float(value < lower_95) for value in residuals),
        "above_95": statistics.fmean(float(value > upper_95) for value in residuals),
        "width_80": upper_80 - lower_80,
        "width_95": upper_95 - lower_95,
    }


def _prediction_metrics(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    return prediction_metrics([float(row["actual_rating_delta"]) for row in rows], [float(row[key]) for row in rows])


def _support_band(support: int) -> str:
    return "unseen" if support == 0 else "one" if support == 1 else "2-to-4" if support <= 4 else "5-or-more"


def _game_count_band(count: int) -> str:
    return "lt-120" if count < 120 else "120-to-499" if count < 500 else "500-to-1999" if count < 2000 else "ge-2000"


def _slice_metrics(rows: Sequence[Mapping[str, object]], key) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: {"structural": _prediction_metrics(group, "structural_prediction"), "hybrid": _prediction_metrics(group, "hybrid_prediction")} for name, group in sorted(groups.items())}


def _references(rows: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["configuration_sha256"])].append(float(row["own_payoff"]))
    return {key: sorted(payoffs) for key, payoffs in sorted(values.items())}


def _rank_prediction(row: Mapping[str, object], references: Mapping[str, Sequence[float]], alpha: float, schedule: Mapping[str, object], *, base_percentile: float | None = None, remove_one: bool = False) -> tuple[float, float, float | None, int]:
    structural_percentile = float(row["structural_percentile"] if base_percentile is None else base_percentile)
    empirical, support = empirical_midrank(float(row["own_payoff"]), references.get(str(row["configuration_sha256"]), ()), remove_one=remove_one) if str(row["configuration_sha256"]) in references else (None, 0)
    weight = shrinkage_weight(support, alpha)
    percentile = structural_percentile if empirical is None else (1.0 - weight) * structural_percentile + weight * empirical
    games_before = int(row["pregame_game_count"])
    delta = predict_display_delta(float(row["pregame_display_rating"]), games_before, percentile, eta_for_game_count(schedule, games_before))
    return delta, percentile, empirical, support


def _residual_table(rows: Sequence[Mapping[str, object]], predictions: Mapping[tuple[str, str], float]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["game_id"]), str(row["target_scope"]))
        values[str(row["configuration_sha256"])].append(float(row["actual_rating_delta"]) - float(predictions[key]))
    return {configuration: {"mean": statistics.fmean(residuals), "count": len(residuals)} for configuration, residuals in sorted(values.items())}


def _apply_residual(delta: float, configuration: str, table: Mapping[str, Mapping[str, float | int]], alpha: float) -> tuple[float, int, float]:
    record = table.get(configuration)
    if record is None:
        return delta, 0, 0.0
    support = int(record["count"])
    correction = shrinkage_weight(support, alpha) * float(record["mean"])
    return delta + correction, support, correction


def _fit_structural_model(rows: Sequence[Mapping[str, object]], family: str, ridge_lambda: float, schedule: Mapping[str, object]) -> RidgeRatingModel:
    fit_rows = [{"features": row["structural_features"], "rating_delta": _structural_target(row["sample"], schedule)} for row in rows]
    return fit_ridge(fit_rows, feature_key="features", feature_names=FAMILY_BASE_FEATURES[family], ridge_lambda=ridge_lambda)


def _cross_fitted_structural(rows: Sequence[Mapping[str, object]], family: str, ridge_lambda: float, schedule: Mapping[str, object]) -> dict[tuple[str, str], tuple[float, float]]:
    output: dict[tuple[str, str], tuple[float, float]] = {}
    for fold in range(CROSS_FIT_FOLDS):
        fit_rows = [row for row in rows if cross_fit_fold(str(row["game_id"])) != fold]
        held_rows = [row for row in rows if cross_fit_fold(str(row["game_id"])) == fold]
        if not fit_rows or not held_rows:
            raise ValueError(f"{family} cross-fit fold {fold} lacks fit or held rows")
        model = _fit_structural_model(fit_rows, family, ridge_lambda, schedule)
        for row in held_rows:
            percentile = model.predict(row["structural_features"])
            games_before = int(row["pregame_game_count"])
            delta = predict_display_delta(float(row["pregame_display_rating"]), games_before, percentile, eta_for_game_count(schedule, games_before))
            output[(str(row["game_id"]), str(row["target_scope"]))] = (delta, percentile)
    if len(output) != len(rows):
        raise RuntimeError(f"{family} cross-fitting did not predict every row")
    return output


def _bootstrap_mae_difference(rows: Sequence[Mapping[str, object]], *, replicates: int, seed: int) -> dict[str, float | int]:
    by_game: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game_id"])].append(row)
    groups = [by_game[game_id] for game_id in sorted(by_game)]
    if not groups:
        return {"games": 0, "samples": 0, "point": 0.0, "lower_95": 0.0, "upper_95": 0.0, "replicates": replicates, "seed": seed}
    point = statistics.fmean(abs(float(row["hybrid_prediction"]) - float(row["actual_rating_delta"])) - abs(float(row["structural_prediction"]) - float(row["actual_rating_delta"])) for row in rows)
    generator = random.Random(seed)
    values = []
    for _replicate in range(replicates):
        selected = [groups[generator.randrange(len(groups))] for _index in groups]
        sampled = [row for group in selected for row in group]
        values.append(statistics.fmean(abs(float(row["hybrid_prediction"]) - float(row["actual_rating_delta"])) - abs(float(row["structural_prediction"]) - float(row["actual_rating_delta"])) for row in sampled))
    return {"games": len(groups), "samples": len(rows), "point": point, "lower_95": float(_quantile(values, 0.025)), "upper_95": float(_quantile(values, 0.975)), "replicates": replicates, "seed": seed}


def _pair_features(row: Mapping[str, object], other: Mapping[str, object]) -> dict[str, float]:
    return {
        "bias": 1.0,
        "own_structural": float(row["structural_prediction"]),
        "other_structural": float(other["structural_prediction"]),
        "own_direct": float(row["direct_prediction"]),
        "other_direct": float(other["direct_prediction"]),
        "target_self": float(row["target_scope"] == "self"),
        "own_pregame_scaled": (float(row["pregame_display_rating"]) - 2000.0) / 1000.0,
        "other_pregame_scaled": (float(other["pregame_display_rating"]) - 2000.0) / 1000.0,
        "own_log_count": math.log1p(int(row["pregame_game_count"])) / 10.0,
        "other_log_count": math.log1p(int(other["pregame_game_count"])) / 10.0,
    }


def _joint_pair_diagnostic(rows: Sequence[Mapping[str, object]], selected_predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_game: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        by_game[str(row["game_id"])][str(row["target_scope"])] = row
    selected_by_key = {(str(row["game_id"]), str(row["target_scope"])): row for row in selected_predictions}
    output: dict[str, object] = {"feature_names": list(PAIR_FEATURES), "families": {}}
    for family in GLEE_FAMILIES:
        model_rows = []
        for pair in by_game.values():
            if set(pair) != {"self", "opponent"} or pair["self"]["family"] != family:
                continue
            for scope, other_scope in (("self", "opponent"), ("opponent", "self")):
                row = pair[scope]
                other = pair[other_scope]
                selected = selected_by_key.get((str(row["game_id"]), str(row["target_scope"])))
                model_rows.append({"split": row["split"], "features": _pair_features(row, other), "rating_delta": float(row["actual_rating_delta"]) - float(row["structural_prediction"]), "actual": float(row["actual_rating_delta"]), "base": float(row["structural_prediction"]), "selected": float(selected["hybrid_prediction"]) if selected is not None else None})
        train = [row for row in model_rows if row["split"] == "train"]
        calibration = [row for row in model_rows if row["split"] == "calibration"]
        test = [row for row in model_rows if row["split"] == "test"]
        candidates = []
        for ridge_lambda in RIDGE_GRID:
            model = fit_ridge(train, feature_key="features", feature_names=PAIR_FEATURES, ridge_lambda=ridge_lambda)
            estimates = [float(row["base"]) + model.predict(row["features"]) for row in calibration]
            metrics = prediction_metrics([float(row["actual"]) for row in calibration], estimates)
            candidates.append((float(metrics["mae"]), float(metrics["rmse"]), ridge_lambda, model, metrics))
        candidates.sort(key=lambda value: value[:3])
        _mae, _rmse, ridge_lambda, model, calibration_metrics = candidates[0]
        test_estimates = [float(row["base"]) + model.predict(row["features"]) for row in test]
        base_calibration = prediction_metrics([float(row["actual"]) for row in calibration], [float(row["base"]) for row in calibration])
        base_test = prediction_metrics([float(row["actual"]) for row in test], [float(row["base"]) for row in test])
        selected_calibration = prediction_metrics([float(row["actual"]) for row in calibration], [float(row["selected"]) for row in calibration])
        selected_test = prediction_metrics([float(row["actual"]) for row in test], [float(row["selected"]) for row in test])
        test_metrics = prediction_metrics([float(row["actual"]) for row in test], test_estimates)
        promoted = float(calibration_metrics["mae"]) < float(selected_calibration["mae"]) and float(calibration_metrics["rmse"]) < float(selected_calibration["rmse"]) and float(test_metrics["mae"]) <= float(selected_test["mae"]) and float(test_metrics["rmse"]) <= float(selected_test["rmse"])
        output["families"][family] = {"counts": {"train": len(train), "calibration": len(calibration), "test": len(test)}, "ridge_lambda": ridge_lambda, "structural_calibration": base_calibration, "selected_nonjoint_calibration": selected_calibration, "pair_calibration": calibration_metrics, "structural_test": base_test, "selected_nonjoint_test": selected_test, "pair_test": test_metrics, "promoted": promoted, "model": model.as_dict()}
    output["any_promoted"] = any(bool(result["promoted"]) for result in output["families"].values())
    return output


class GleeJointRatingV2Analysis:
    """Fit and report the frozen retrospective exact-configuration rating hybrid."""

    def __init__(self, *, rating_dir: Path, output_dir: Path, bootstrap_replicates: int = BOOTSTRAP_REPLICATES, bootstrap_seed: int = BOOTSTRAP_SEED) -> None:
        if bootstrap_replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        self.rating_dir = rating_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.bootstrap_replicates = bootstrap_replicates
        self.bootstrap_seed = bootstrap_seed

    def _load_source(self) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        manifest_path = self.rating_dir / "manifest.json"
        if _file_digest(manifest_path) != EXPECTED_V1_MANIFEST_SHA256:
            raise RuntimeError("rating v2 requires the frozen frontier-18108 v1 manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != "glee-joint-rating-reconstruction-v1":
            raise ValueError("rating v2 source contract is invalid")
        for name, receipt in manifest["artifacts"].items():
            if _file_digest(self.rating_dir / name) != receipt["sha256"]:
                raise RuntimeError(f"rating v1 artifact hash mismatch: {name}")
        structural_artifact = json.loads((self.rating_dir / "structural-model.json").read_text(encoding="utf-8"))
        samples = _load_jsonl(self.rating_dir / "joint-rating-samples.jsonl")
        predictions = _load_jsonl(self.rating_dir / "predictions.jsonl")
        sample_by_key = {(str(sample["game_id"]), str(sample["target_scope"])): sample for sample in samples}
        rows = []
        for prediction in predictions:
            key = (str(prediction["game_id"]), str(prediction["target_scope"]))
            sample = sample_by_key.get(key)
            if sample is None:
                raise RuntimeError(f"v1 prediction lacks sample: {key}")
            rows.append(
                {
                    "game_id": key[0],
                    "family": str(sample["family"]),
                    "split": str(sample["split"]),
                    "completed_at": str(sample["completed_at"]),
                    "target_scope": key[1],
                    "target_player": str(sample["target_player"]),
                    "target_public_player_id": str(sample["target_public_player_id"]),
                    "role": str(sample["terminal"]["role"]),
                    "outcome": str(sample["terminal"]["outcome"]),
                    "configuration_sha256": str(sample["terminal"]["observed_configuration_sha256"]),
                    "own_payoff": float(sample["terminal"]["own_payoff"]),
                    "actual_rating_delta": float(sample["rating_delta"]),
                    "pregame_game_count": int(sample["pregame_game_count"]),
                    "pregame_display_rating": float(sample["pregame_display_rating"]),
                    "structural_prediction": float(prediction["structural_prediction"]),
                    "structural_percentile": float(prediction["predicted_adjusted_percentile"]),
                    "direct_prediction": float(prediction["direct_prediction"]),
                    "structural_features": sample["structural_features"],
                    "sample": sample,
                }
            )
        if len(rows) != len(samples):
            raise RuntimeError("rating v1 sample and prediction counts differ")
        rows.sort(key=lambda row: (float(row["sample"]["completed_at_timestamp"]), str(row["game_id"]), str(row["target_scope"])))
        return manifest, structural_artifact, rows

    def _fit_family(self, family: str, rows: Sequence[Mapping[str, object]], structural_artifact: Mapping[str, object]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        family_rows = [row for row in rows if row["family"] == family]
        train = [row for row in family_rows if row["split"] == "train"]
        calibration = [row for row in family_rows if row["split"] == "calibration"]
        test = [row for row in family_rows if row["split"] == "test"]
        schedule = structural_artifact["eta_schedule"]
        ridge_lambda = float(structural_artifact["models"][family]["ridge_lambda"])
        cross_fitted = _cross_fitted_structural(train, family, ridge_lambda, schedule)
        references = _references(train)

        config_candidates = []
        oof_base = {key: prediction[0] for key, prediction in cross_fitted.items()}
        config_table = _residual_table(train, oof_base)
        for alpha in ALPHA_GRID:
            estimates = [_apply_residual(float(row["structural_prediction"]), str(row["configuration_sha256"]), config_table, alpha)[0] for row in calibration]
            metrics = prediction_metrics([float(row["actual_rating_delta"]) for row in calibration], estimates)
            config_candidates.append((float(metrics["mae"]), float(metrics["rmse"]), alpha, metrics))
        config_candidates.sort(key=lambda value: value[:3])
        _mae, _rmse, config_alpha, config_calibration = config_candidates[0]

        rank_candidates = []
        for rank_alpha in ALPHA_GRID:
            estimates = [_rank_prediction(row, references, rank_alpha, schedule)[0] for row in calibration]
            metrics = prediction_metrics([float(row["actual_rating_delta"]) for row in calibration], estimates)
            rank_candidates.append((float(metrics["mae"]), float(metrics["rmse"]), rank_alpha, metrics))
        rank_candidates.sort(key=lambda value: value[:3])
        _mae, _rmse, rank_alpha, rank_calibration = rank_candidates[0]

        hybrid_candidates = []
        for candidate_rank_alpha in ALPHA_GRID:
            training_predictions = {}
            for row in train:
                key = (str(row["game_id"]), str(row["target_scope"]))
                training_predictions[key] = _rank_prediction(row, references, candidate_rank_alpha, schedule, base_percentile=cross_fitted[key][1], remove_one=True)[0]
            table = _residual_table(train, training_predictions)
            for residual_alpha in ALPHA_GRID:
                estimates = []
                for row in calibration:
                    rank_delta = _rank_prediction(row, references, candidate_rank_alpha, schedule)[0]
                    estimates.append(_apply_residual(rank_delta, str(row["configuration_sha256"]), table, residual_alpha)[0])
                metrics = prediction_metrics([float(row["actual_rating_delta"]) for row in calibration], estimates)
                hybrid_candidates.append((float(metrics["mae"]), float(metrics["rmse"]), candidate_rank_alpha, residual_alpha, metrics, table))
        hybrid_candidates.sort(key=lambda value: value[:4])
        _mae, _rmse, selected_rank_alpha, residual_alpha, hybrid_calibration, hybrid_table = hybrid_candidates[0]

        output_rows = []
        for row in (*calibration, *test):
            rank_delta, rank_percentile, empirical, rank_support = _rank_prediction(row, references, rank_alpha, schedule)
            hybrid_rank_delta, hybrid_percentile, hybrid_empirical, hybrid_rank_support = _rank_prediction(row, references, selected_rank_alpha, schedule)
            configuration_delta, configuration_support, configuration_correction = _apply_residual(float(row["structural_prediction"]), str(row["configuration_sha256"]), config_table, config_alpha)
            hybrid_delta, residual_support, residual_correction = _apply_residual(hybrid_rank_delta, str(row["configuration_sha256"]), hybrid_table, residual_alpha)
            if rank_support != hybrid_rank_support or empirical != hybrid_empirical:
                raise RuntimeError("rank support differs across alpha candidates")
            output_rows.append(
                {
                    "contract": JOINT_RATING_V2_CONTRACT,
                    "schema_version": 1,
                    "game_id": row["game_id"],
                    "family": family,
                    "completed_at": row["completed_at"],
                    "split": row["split"],
                    "target_scope": row["target_scope"],
                    "target_player": row["target_player"],
                    "target_public_player_id": row["target_public_player_id"],
                    "role": row["role"],
                    "outcome": row["outcome"],
                    "configuration_sha256": row["configuration_sha256"],
                    "own_payoff": row["own_payoff"],
                    "pregame_game_count": row["pregame_game_count"],
                    "actual_rating_delta": row["actual_rating_delta"],
                    "direct_prediction": row["direct_prediction"],
                    "structural_prediction": row["structural_prediction"],
                    "structural_percentile": row["structural_percentile"],
                    "configuration_prediction": configuration_delta,
                    "configuration_support": configuration_support,
                    "configuration_correction": configuration_correction,
                    "rank_prediction": rank_delta,
                    "rank_percentile": rank_percentile,
                    "empirical_percentile": empirical,
                    "rank_support": rank_support,
                    "hybrid_prediction": hybrid_delta,
                    "hybrid_percentile": hybrid_percentile,
                    "hybrid_residual_support": residual_support,
                    "hybrid_residual_correction": residual_correction,
                }
            )
        calibration_rows = [row for row in output_rows if row["split"] == "calibration"]
        test_rows = [row for row in output_rows if row["split"] == "test"]
        intervals = asymmetric_interval([float(row["actual_rating_delta"]) - float(row["hybrid_prediction"]) for row in calibration_rows])
        for row in output_rows:
            prediction = float(row["hybrid_prediction"])
            row["interval_80"] = [prediction + intervals["lower_80"], prediction + intervals["upper_80"]]
            row["interval_95"] = [prediction + intervals["lower_95"], prediction + intervals["upper_95"]]
        result = {
            "counts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
            "selected": {"configuration_alpha": config_alpha, "rank_alpha": rank_alpha, "hybrid_rank_alpha": selected_rank_alpha, "hybrid_residual_alpha": residual_alpha, "structural_ridge_lambda": ridge_lambda},
            "calibration": {"structural": _prediction_metrics(calibration_rows, "structural_prediction"), "configuration": config_calibration, "rank": rank_calibration, "hybrid": hybrid_calibration},
            "test": {"structural": _prediction_metrics(test_rows, "structural_prediction"), "configuration": _prediction_metrics(test_rows, "configuration_prediction"), "rank": _prediction_metrics(test_rows, "rank_prediction"), "hybrid": _prediction_metrics(test_rows, "hybrid_prediction")},
            "support": {"training_configurations": len(references), "test_seen": sum(int(row["rank_support"]) > 0 for row in test_rows), "test_unseen": sum(int(row["rank_support"]) == 0 for row in test_rows), "test_median_seen": statistics.median(int(row["rank_support"]) for row in test_rows if int(row["rank_support"]) > 0) if any(int(row["rank_support"]) > 0 for row in test_rows) else None},
            "intervals": {"residual_offsets": intervals, "test": _interval_metrics(test_rows, intervals)},
        }
        evaluation_model = {"references": references, "configuration_residuals": hybrid_table, "rank_alpha": selected_rank_alpha, "residual_alpha": residual_alpha, "intervals": intervals}
        return result, output_rows, evaluation_model

    def _prospective_family_model(self, family: str, rows: Sequence[Mapping[str, object]], structural_artifact: Mapping[str, object], selected: Mapping[str, object]) -> dict[str, object]:
        family_rows = [row for row in rows if row["family"] == family]
        schedule = structural_artifact["eta_schedule"]
        ridge_lambda = float(structural_artifact["models"][family]["ridge_lambda"])
        cross_fitted = _cross_fitted_structural(family_rows, family, ridge_lambda, schedule)
        references = _references(family_rows)
        rank_alpha = float(selected["hybrid_rank_alpha"])
        training_predictions = {}
        for row in family_rows:
            key = (str(row["game_id"]), str(row["target_scope"]))
            training_predictions[key] = _rank_prediction(row, references, rank_alpha, schedule, base_percentile=cross_fitted[key][1], remove_one=True)[0]
        table = _residual_table(family_rows, training_predictions)
        model = _fit_structural_model(family_rows, family, ridge_lambda, schedule)
        return {"structural_model": model.as_dict(), "references": references, "configuration_residuals": table, "rank_alpha": rank_alpha, "residual_alpha": float(selected["hybrid_residual_alpha"]), "training_samples": len(family_rows), "latest_completed_at": max(str(row["completed_at"]) for row in family_rows)}

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        families = summary["families"]
        pooled = summary["pooled_test"]
        pair = summary["joint_pair_diagnostic"]
        return "\n".join(
            [
                "# GLEE joint rating-delta reconstruction v2",
                "",
                "**Status:** Completed retrospective architecture-development analysis through reporter frontier `18108`; the source suffix was inspected while designing v2, so these metrics are not confirmatory and no model has live authority.",
                "",
                "## Result",
                "",
                f"The exact-configuration hybrid lowers pooled displayed-delta MAE from `{pooled['structural']['mae']:.4f}` to `{pooled['hybrid']['mae']:.4f}` and RMSE from `{pooled['structural']['rmse']:.4f}` to `{pooled['hybrid']['rmse']:.4f}`. Sign accuracy changes from `{pooled['structural']['sign_accuracy']:.2%}` to `{pooled['hybrid']['sign_accuracy']:.2%}`. The whole-game bootstrap interval for hybrid-minus-structural MAE is `[{summary['bootstrap']['pooled']['lower_95']:+.4f}, {summary['bootstrap']['pooled']['upper_95']:+.4f}]`; it is descriptive because the architecture was selected retrospectively.",
                "",
                "| Family | Structural MAE | Hybrid MAE | Structural sign | Hybrid sign | Seen test samples | Selected rank/residual alpha |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[f"| {family.title()} | {families[family]['test']['structural']['mae']:.4f} | {families[family]['test']['hybrid']['mae']:.4f} | {families[family]['test']['structural']['sign_accuracy']:.2%} | {families[family]['test']['hybrid']['sign_accuracy']:.2%} | {families[family]['support']['test_seen']}/{families[family]['counts']['test']} | {families[family]['selected']['hybrid_rank_alpha']:g}/{families[family]['selected']['hybrid_residual_alpha']:g} |" for family in GLEE_FAMILIES],
                "",
                "## Interpretation",
                "",
                "The main gain comes from approximating the published same-exact-configuration and same-role payoff percentile directly. A smoothed empirical payoff rank is blended with the structural percentile, then a cross-fitted, shrunken configuration residual captures persistent cell error. Unseen configurations fall back to the v1 structural model.",
                "",
                f"The generic paired residual model beats the selected non-joint hybrid under the frozen calibration-and-test gate for {sum(bool(value['promoted']) for value in pair['families'].values())} of 3 families. The 2 players' anti-correlated residuals therefore do not by themselves justify imposing a zero-sum or generic affine pair constraint; exact scoring-cell structure explains more of the recoverable error.",
                "",
                "## Boundary",
                "",
                "The separately serialized prospective-shadow model is fitted on all frozen samples using the already selected hyperparameters. It may register forecasts for later games, but it cannot advise actions until a fresh chronological suffix confirms point accuracy, uncertainty, and decision value. The model does not recover GLEE's private hourly opponent-strength adjustment or weakly identified early learning-rate schedule.",
                "",
            ]
        )

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"joint-rating v2 output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        v1_manifest, structural_artifact, rows = self._load_source()
        family_reports = {}
        prediction_rows = []
        evaluation_models = {}
        for family in GLEE_FAMILIES:
            report, outputs, model = self._fit_family(family, rows, structural_artifact)
            family_reports[family] = report
            prediction_rows.extend(outputs)
            evaluation_models[family] = model
        prediction_rows.sort(key=lambda row: (str(row["completed_at"]), str(row["game_id"]), str(row["target_scope"])))
        test_rows = [row for row in prediction_rows if row["split"] == "test"]
        pooled = {arm: _prediction_metrics(test_rows, key) for arm, key in (("structural", "structural_prediction"), ("configuration", "configuration_prediction"), ("rank", "rank_prediction"), ("hybrid", "hybrid_prediction"))}
        pair_diagnostic = _joint_pair_diagnostic(rows, prediction_rows)
        bootstrap = {family: _bootstrap_mae_difference([row for row in test_rows if row["family"] == family], replicates=self.bootstrap_replicates, seed=self.bootstrap_seed) for family in GLEE_FAMILIES}
        bootstrap["pooled"] = _bootstrap_mae_difference(test_rows, replicates=self.bootstrap_replicates, seed=self.bootstrap_seed)
        slices = {
            "target_scope": _slice_metrics(test_rows, lambda row: row["target_scope"]),
            "role": _slice_metrics(test_rows, lambda row: f"{row['family']}:{row['role']}"),
            "outcome": _slice_metrics(test_rows, lambda row: f"{row['family']}:{row['outcome']}"),
            "configuration_support": _slice_metrics(test_rows, lambda row: _support_band(int(row["rank_support"]))),
            "game_count_band": _slice_metrics(test_rows, lambda row: _game_count_band(int(row["pregame_game_count"]))),
        }
        prospective_models = {family: self._prospective_family_model(family, rows, structural_artifact, family_reports[family]["selected"]) for family in GLEE_FAMILIES}
        family_regressions = {family: float(family_reports[family]["test"]["hybrid"]["mae"]) / float(family_reports[family]["test"]["structural"]["mae"]) - 1.0 for family in GLEE_FAMILIES}
        retrospective_gate = float(pooled["hybrid"]["mae"]) < float(pooled["structural"]["mae"]) and float(pooled["hybrid"]["rmse"]) < float(pooled["structural"]["rmse"]) and float(pooled["hybrid"]["sign_accuracy"]) >= float(pooled["structural"]["sign_accuracy"]) and max(family_regressions.values()) <= 0.02
        summary: dict[str, object] = {
            "contract": JOINT_RATING_V2_CONTRACT,
            "schema_version": 1,
            "status": "retrospective-development-only",
            "frontier_sequence": int(v1_manifest["frontier_sequence"]),
            "parameters": {"alpha_grid": list(ALPHA_GRID), "cross_fit_folds": CROSS_FIT_FOLDS, "bootstrap_replicates": self.bootstrap_replicates, "bootstrap_seed": self.bootstrap_seed, "pair_ridge_grid": list(RIDGE_GRID)},
            "sources": {"rating_dir": str(self.rating_dir), "v1_manifest_sha256": EXPECTED_V1_MANIFEST_SHA256, "latest_admitted_completion": max(str(row["completed_at"]) for row in rows)},
            "inventory": {"samples": len(rows), "logical_games": len({str(row["game_id"]) for row in rows}), "by_split": {split: sum(row["split"] == split for row in rows) for split in ("train", "calibration", "test")}, "by_family": {family: sum(row["family"] == family for row in rows) for family in GLEE_FAMILIES}},
            "families": family_reports,
            "pooled_test": pooled,
            "bootstrap": bootstrap,
            "slices": slices,
            "joint_pair_diagnostic": pair_diagnostic,
            "promotion": {"retrospective_gate_passed": retrospective_gate, "prospective_shadow_model_written": True, "live_authority": False, "confirmatory_evidence": False, "next_gate": "register predictions on a fresh post-frontier suffix before reading rating deltas"},
        }
        model_artifact = {"contract": JOINT_RATING_V2_CONTRACT, "schema_version": 1, "eta_schedule": structural_artifact["eta_schedule"], "evaluation_models": evaluation_models, "prospective_shadow_models": prospective_models, "boundary": "Offline prospective-shadow artifact only; it supplies no live action, prompt, identity, dossier, or matchmaking authority."}
        _write_jsonl(self.output_dir / "hybrid-predictions.jsonl", prediction_rows)
        _write_json(self.output_dir / "hybrid-model.json", model_artifact)
        _write_json(self.output_dir / "joint-pair-diagnostic.json", pair_diagnostic)
        _write_json(self.output_dir / "summary.json", summary)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifact_names = ("README.md", "summary.json", "hybrid-predictions.jsonl", "hybrid-model.json", "joint-pair-diagnostic.json")
        manifest = {
            "contract": JOINT_RATING_V2_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": summary["frontier_sequence"],
            "latest_admitted_completion": summary["sources"]["latest_admitted_completion"],
            "parameters": summary["parameters"],
            "v1_manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
            "implementation_sha256": {"glee_joint_rating_v2_analysis.py": _file_digest(Path(__file__)), "glee_joint_rating_analysis.py": _file_digest(Path(__file__).with_name("glee_joint_rating_analysis.py")), "glee_negotiation_rating_v2_4.py": _file_digest(Path(__file__).with_name("glee_negotiation_rating_v2_4.py")), "protocol": _file_digest(Path(__file__).resolve().parents[2] / "protocols/glee-joint-rating-v2.md")},
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifact_names},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": JOINT_RATING_V2_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": summary["frontier_sequence"], "pooled_test": pooled, "promotion": summary["promotion"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
