"""Paired, game-clustered analysis for sequence-twin experiment arms."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from .corpus import file_sha256


ANALYSIS_CONTRACT = "glee-hierarchical-sequence-twin-analysis-v3"
EXPECTED_EXPERIMENT_CONTRACTS = {"glee-hierarchical-sequence-twin-experiment-v2", "glee-hierarchical-sequence-twin-experiment-v3"}


def _single_key(value: object) -> str:
    return str(value[0] if isinstance(value, tuple) and len(value) == 1 else value)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _metrics(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty():
        return {"targets": 0, "games": 0, "action": {"rows": 0}, "value": {"rows": 0}, "delay": {"rows": 0}}
    action = frame.filter(pl.col("action_scored"))
    values = frame.filter(pl.col("actual_value").is_not_null())
    delays = frame.filter(pl.col("actual_delay").is_not_null())
    games = action.group_by("game_id").agg(pl.col("action_negative_log_likelihood").mean().alias("nll"))
    accounts = action.filter(pl.col("account_key").is_not_null()).group_by("account_key").agg(pl.col("action_negative_log_likelihood").mean().alias("nll"))
    fusion = {}
    for target in ("action", "value", "delay"):
        column = f"{target}_message_gate"
        gate_values = frame[column].drop_nulls() if column in frame.columns else None
        fusion[target] = {"rows": len(gate_values) if gate_values is not None else 0, "mean_gate": gate_values.mean() if gate_values is not None and len(gate_values) else None}
    return {
        "targets": frame.height,
        "games": frame["game_id"].n_unique(),
        "action": {"rows": action.height, "negative_log_likelihood": action["action_negative_log_likelihood"].mean() if not action.is_empty() else None, "accuracy": (action["actual_action"] == action["predicted_action"]).mean() if not action.is_empty() else None, "game_macro_negative_log_likelihood": games["nll"].mean() if not games.is_empty() else None, "known_account_macro_negative_log_likelihood": accounts["nll"].mean() if not accounts.is_empty() else None},
        "value": {"rows": values.height, "negative_log_likelihood": values["value_negative_log_likelihood"].mean() if not values.is_empty() else None, "mean_absolute_error": (values["actual_value"] - values["predicted_value"]).abs().mean() if not values.is_empty() else None},
        "delay": {"rows": delays.height, "negative_log_likelihood": delays["delay_negative_log_likelihood"].mean() if not delays.is_empty() else None, "log_mean_absolute_error": (delays["actual_delay"] - delays["predicted_delay"]).abs().mean() if not delays.is_empty() else None},
        "message_fusion": fusion,
    }


def _stratified(frame: pl.DataFrame) -> dict[str, object]:
    strata: dict[str, object] = {"all": _metrics(frame)}
    for column in ("family", "identity_scope", "target_kind"):
        strata[f"by_{column}"] = {_single_key(key): _metrics(group) for key, group in frame.partition_by(column, as_dict=True).items()}
    strata["by_family_and_target_kind"] = {f"{family}/{kind}": _metrics(group) for (family, kind), group in frame.partition_by(["family", "target_kind"], as_dict=True).items()}
    return strata


def _bootstrap_delta(frame: pl.DataFrame, *, replicates: int, seed: int) -> dict[str, object]:
    grouped = frame.group_by("game_id").agg(pl.col("delta").sum().alias("sum"), pl.len().alias("count"), pl.col("delta").mean().alias("mean"))
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    means = grouped["mean"].to_numpy()
    if len(sums) == 0:
        return {"replicates": 0, "micro_mean": None, "micro_interval_95": None, "micro_favorable_fraction": None, "game_macro_mean": None, "game_macro_interval_95": None, "game_macro_favorable_fraction": None}
    rng = np.random.default_rng(seed)
    micro = np.empty(replicates, dtype=np.float64)
    macro = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(sums), size=len(sums))
        micro[index] = sums[selected].sum() / counts[selected].sum()
        macro[index] = means[selected].mean()
    observed_micro = float(sums.sum() / counts.sum())
    observed_macro = float(means.mean())
    return {
        "replicates": replicates,
        "cluster": "complete game",
        "direction": "arm minus baseline; negative favors arm",
        "micro_mean": observed_micro,
        "micro_interval_95": [float(value) for value in np.quantile(micro, [0.025, 0.975])],
        "game_macro_mean": observed_macro,
        "game_macro_interval_95": [float(value) for value in np.quantile(macro, [0.025, 0.975])],
        "micro_favorable_fraction": float((micro < 0.0).mean()),
        "game_macro_favorable_fraction": float((macro < 0.0).mean()),
    }


def _comparison(baseline: pl.DataFrame, arm: pl.DataFrame, *, replicates: int, seed: int) -> dict[str, object]:
    columns = ["sample_id", "game_id", "family", "identity_scope", "target_kind", "account_key", "action_scored", "actual_action", "action_negative_log_likelihood", "actual_value", "predicted_value", "value_negative_log_likelihood", "actual_delay", "delay_negative_log_likelihood"]
    merged = baseline.select(columns).join(arm.select(columns), on="sample_id", how="inner", suffix="_arm", validate="1:1")
    if merged.height != baseline.height or merged.height != arm.height:
        raise ValueError("experiment arms do not cover the same test samples")
    for column in ("game_id", "family", "identity_scope", "target_kind", "action_scored", "actual_action"):
        if merged.filter(pl.col(column).ne_missing(pl.col(f"{column}_arm"))).height:
            raise ValueError(f"paired experiment coordinate mismatch: {column}")
    paired = merged.filter(pl.col("action_scored")).with_columns((pl.col("action_negative_log_likelihood_arm") - pl.col("action_negative_log_likelihood")).alias("delta"))
    value_nll = merged.filter(pl.col("actual_value").is_not_null()).with_columns((pl.col("value_negative_log_likelihood_arm") - pl.col("value_negative_log_likelihood")).alias("delta"))
    value_mae = merged.filter(pl.col("actual_value").is_not_null()).with_columns(((pl.col("actual_value_arm") - pl.col("predicted_value_arm")).abs() - (pl.col("actual_value") - pl.col("predicted_value")).abs()).alias("delta"))
    delay = merged.filter(pl.col("actual_delay").is_not_null()).with_columns((pl.col("delay_negative_log_likelihood_arm") - pl.col("delay_negative_log_likelihood")).alias("delta"))
    receipt: dict[str, object] = {"tasks": {"action_negative_log_likelihood": _bootstrap_delta(paired, replicates=replicates, seed=seed), "value_negative_log_likelihood": _bootstrap_delta(value_nll, replicates=replicates, seed=seed + 1), "value_mean_absolute_error": _bootstrap_delta(value_mae, replicates=replicates, seed=seed + 2), "delay_negative_log_likelihood": _bootstrap_delta(delay, replicates=replicates, seed=seed + 3)}}
    for column in ("family", "identity_scope", "target_kind"):
        receipt[f"by_{column}"] = {_single_key(key): _bootstrap_delta(group, replicates=replicates, seed=seed + index + 1) for index, (key, group) in enumerate(paired.partition_by(column, as_dict=True).items())}
    receipt["by_family_and_target_kind"] = {f"{family}/{kind}": _bootstrap_delta(group, replicates=replicates, seed=seed + index + 101) for index, ((family, kind), group) in enumerate(paired.partition_by(["family", "target_kind"], as_dict=True).items())}
    return receipt


def _experiment(directory: Path) -> tuple[dict[str, object], pl.DataFrame]:
    resolved = directory.resolve()
    result = json.loads((resolved / "result.json").read_text(encoding="utf-8"))
    if result.get("contract") not in EXPECTED_EXPERIMENT_CONTRACTS:
        raise ValueError(f"incompatible experiment contract: {result.get('contract')!r}")
    prediction = resolved / str(result["test_predictions"]["path"])
    if file_sha256(prediction) != result["test_predictions"]["sha256"]:
        raise ValueError(f"prediction hash mismatch: {prediction}")
    return result, pl.read_parquet(prediction)


def compare_account_ablation(*, experiment_dir: Path, output_dir: Path, bootstrap_replicates: int = 2_000, seed: int = 81_077) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    result, full = _experiment(experiment_dir)
    reference = result.get("account_ablation_test_predictions")
    if not isinstance(reference, dict):
        raise ValueError("experiment has no account-ablation predictions")
    ablation_path = experiment_dir.resolve() / str(reference["path"])
    if file_sha256(ablation_path) != reference["sha256"]:
        raise ValueError(f"account-ablation prediction hash mismatch: {ablation_path}")
    population = pl.read_parquet(ablation_path)
    receipt = {
        "contract": ANALYSIS_CONTRACT,
        "comparison": "trained hierarchical model with account conditioning minus the same model forced to its population path",
        "direction": "full hierarchical minus forced population; negative loss delta favors account conditioning",
        "experiment": {"arm": result["arm"], "directory": str(experiment_dir.resolve()), "result_sha256": file_sha256(experiment_dir.resolve() / "result.json")},
        "forced_population": {"metrics": _stratified(population), "predictions_sha256": reference["sha256"]},
        "full_hierarchical": {"metrics": _stratified(full), "predictions_sha256": result["test_predictions"]["sha256"]},
        "paired": _comparison(population, full, replicates=bootstrap_replicates, seed=seed),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }
    _atomic_json(output_dir / "analysis.json", receipt)
    return receipt


def compare_experiments(*, baseline_dir: Path, arm_dirs: Sequence[Path], output_dir: Path, bootstrap_replicates: int = 2_000, seed: int = 81_077) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    baseline_result, baseline = _experiment(baseline_dir)
    arms: dict[str, object] = {}
    for arm_index, directory in enumerate(arm_dirs):
        result, predictions = _experiment(directory)
        if result["vocabulary_sha256"] != baseline_result["vocabulary_sha256"]:
            raise ValueError("experiment vocabulary mismatch")
        label = str(result.get("label") or result["arm"])
        if label in arms:
            label = f"{label}-seed{result['training']['seed']}"
        arms[label] = {
            "directory": str(directory.resolve()),
            "result_sha256": file_sha256(directory.resolve() / "result.json"),
            "metrics": _stratified(predictions),
            "paired_against_baseline": _comparison(baseline, predictions, replicates=bootstrap_replicates, seed=seed + arm_index * 10_000),
        }
    receipt = {
        "contract": ANALYSIS_CONTRACT,
        "baseline": {"arm": baseline_result["arm"], "label": baseline_result.get("label") or baseline_result["arm"], "directory": str(baseline_dir.resolve()), "result_sha256": file_sha256(baseline_dir.resolve() / "result.json"), "metrics": _stratified(baseline)},
        "arms": arms,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "interpretation": "Paired deltas use complete-game cluster resampling; negative arm-minus-baseline values favor the arm, and micro and game-macro favorable fractions are reported separately.",
    }
    _atomic_json(output_dir / "analysis.json", receipt)
    return receipt
