"""Retrospective sensitivity diagnostics for the post-planner conditional twin."""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from nommd_arena.glee_persuasion_twin_v2 import classify_persuasion_signal

from .conditional_release import ConditionalTwinRelease
from .corpus import _message_features, file_sha256
from .data import CorpusIndex
from .pre_terra_conditional_v2 import CandidateAction


WORDING_SENSITIVITY_CONTRACT = "glee-post-planner-wording-sensitivity-audit-v1"
_TEMPLATES = {
    "signal_positive": "I recommend buying this product.",
    "signal_negative": "I recommend passing on this product.",
}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"rows": 0}
    deltas = [float(row["absolute_buy_probability_delta"]) for row in rows]
    ordered = sorted(deltas)
    return {
        "rows": len(rows),
        "games": len({str(row["game_id"]) for row in rows}),
        "mean_absolute_buy_probability_delta": statistics.fmean(deltas),
        "median_absolute_buy_probability_delta": statistics.median(deltas),
        "p95_absolute_buy_probability_delta": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum_absolute_buy_probability_delta": max(deltas),
        "nonzero_delta_fraction": sum(value > 1e-8 for value in deltas) / len(deltas),
        "argmax_changed_fraction": sum(bool(row["argmax_changed"]) for row in rows) / len(rows),
    }


def run_wording_sensitivity_audit(*, release_dir: Path, corpus_dir: Path, output_dir: Path) -> dict[str, object]:
    release_dir = release_dir.resolve()
    corpus_dir = corpus_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"wording-sensitivity output already exists: {output_dir}")
    release = ConditionalTwinRelease(release_dir)
    index = CorpusIndex([corpus_dir])
    features = {str(row["sample_id"]): row for row in pl.read_parquet(corpus_dir / "features.parquet").to_dicts()}
    rows: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    for target in index.targets:
        if target.get("chronological_split") != "test":
            continue
        game_id = str(target["game_id"])
        game = index.games[game_id]
        if game.get("family") != "persuasion":
            continue
        feature = features[str(target["sample_id"])]
        if feature.get("phase") != "seller_message":
            exclusions["not-text-seller-message"] += 1
            continue
        actual = CandidateAction.from_feature_row(feature, family="persuasion", phase="seller_message")
        template = _TEMPLATES.get(actual.action_label)
        if template is None:
            exclusions["unsupported-polarity"] += 1
            continue
        polarity, family_act, _fingerprint = classify_persuasion_signal(template, channel="text")
        if f"signal_{polarity}" != actual.action_label:
            raise RuntimeError("wording-control template changed candidate polarity")
        alternative_payload = actual.receipt()
        alternative_payload.update(_message_features(template, family_act=family_act))
        alternative = CandidateAction.from_mapping(alternative_payload, family="persuasion", phase="seller_message")
        if alternative.message_sha256 == actual.message_sha256:
            exclusions["template-identical"] += 1
            continue
        prefix_length = int(target["pre_candidate_prefix_length"])
        sample = {"game": game, "events": index.events[game_id][:prefix_length], "target": target}
        predictions = release.predict_candidates(
            sample=sample,
            family="persuasion",
            phase="seller_message",
            base_feature_indices=feature["feature_indices"],
            base_feature_values=feature["feature_values"],
            base_feature_vector_sha256=str(feature["feature_vector_sha256"]),
            candidates=[actual.receipt(), alternative.receipt()],
        )
        if len(predictions) != 2 or predictions[0]["labels"] != ["buy", "pass"] or predictions[1]["labels"] != ["buy", "pass"]:
            raise RuntimeError("wording-sensitivity prediction contract changed")
        actual_buy = float(predictions[0]["response_probabilities"][0])
        alternative_buy = float(predictions[1]["response_probabilities"][0])
        rows.append(
            {
                "sample_id": str(target["sample_id"]),
                "game_id": game_id,
                "identity_scope": str(target["identity_scope"]),
                "polarity": actual.action_label.removeprefix("signal_"),
                "actual_message_sha256": actual.message_sha256,
                "control_message_sha256": alternative.message_sha256,
                "actual_buy_probability": actual_buy,
                "control_buy_probability": alternative_buy,
                "absolute_buy_probability_delta": abs(actual_buy - alternative_buy),
                "argmax_changed": (actual_buy >= 0.5) != (alternative_buy >= 0.5),
            }
        )
    if not rows:
        raise RuntimeError("wording-sensitivity audit found no eligible test rows")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[f"polarity:{row['polarity']}"].append(row)
        grouped[f"identity:{row['identity_scope']}"].append(row)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        rows_path = staging / "rows.parquet"
        pl.DataFrame(rows, infer_schema_length=None).sort(["game_id", "sample_id"]).write_parquet(rows_path, compression="zstd", compression_level=7, statistics=True)
        result = {
            "schema_version": 1,
            "contract": WORDING_SENSITIVITY_CONTRACT,
            "status": "complete",
            "release": {"path": str(release_dir), "manifest_sha256": file_sha256(release_dir / "manifest.json")},
            "corpus": {"path": str(corpus_dir), "manifest_sha256": file_sha256(corpus_dir / "manifest.json")},
            "control": "Replace the historical seller message with one fixed same-polarity template while preserving the authenticated prefix and all non-message candidate coordinates.",
            "overall": _summary(rows),
            "strata": {name: _summary(values) for name, values in sorted(grouped.items())},
            "exclusions": dict(sorted(exclusions.items())),
            "rows": {"path": rows_path.name, "sha256": file_sha256(rows_path), "count": len(rows)},
            "interpretation": "Sensitivity is necessary for candidate comparison but is not counterfactual accuracy evidence because the control wording was not submitted historically.",
        }
        _atomic_json(staging / "result.json", result)
        os.replace(staging, output_dir)
        return {**result, "output_dir": str(output_dir)}
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
