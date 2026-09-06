"""Build public metric-level reproduction tables from the private evidence archive.

The output retains labels, frozen predictions, cluster structure, and execution
telemetry needed by the paper's estimators. It removes names, stable identifiers,
messages, prompts, exact timestamps, source paths, and account linkages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl
import torch

from glee_sequence_lab.data import CorpusVocabs
from glee_sequence_lab.model import HierarchicalSequenceTwin, ModelConfig
from glee_sequence_lab.persuasion_buyer_continuation import (
    BUYER_CONTINUATION_LABELS,
    _ContinuationIndex,
    _ReversedPersuasionHead,
    _encode_partition,
    _markov_baseline,
    _predict_head,
)


CONDITIONAL_ARMS = ("convex-stack", "sequence-ensemble", "engineered-ensemble")
FAMILIES = ("bargaining", "negotiation", "persuasion")
TARGET_EVENT_KINDS = {
    "game_completed",
    "game_completed_during_opponent_turn",
    "move_submitted",
    "worker_finished",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.write_parquet(temporary, compression="zstd", compression_level=7, statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _probability_columns(prefix: str, values: Iterable[float]) -> dict[str, float | None]:
    probabilities = [float(value) for value in values]
    return {f"{prefix}_p{index}": probabilities[index] if index < len(probabilities) else None for index in range(3)}


def _anonymous_groups(values: Iterable[str], prefix: str) -> dict[str, str]:
    return {value: f"{prefix}{index:06d}" for index, value in enumerate(sorted(set(values)), 1)}


def export_conditional_twin(source: Path, output: Path) -> dict[str, Any]:
    path = source / "runs/glee-post-planner-conditional-v3-experiments/deeprmm-final-9587-20260820/predictions.parquet"
    frame = pl.read_parquet(path).filter((pl.col("chronological_split") == "test") & pl.col("arm").is_in(CONDITIONAL_ARMS))
    records = frame.select("sample_id", "game_id", "family", "identity_scope", "arm", "actual_action", "probabilities").to_dicts()
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(str(record["sample_id"]), {})[str(record["arm"])] = record
    if any(set(arms) != set(CONDITIONAL_ARMS) for arms in by_sample.values()):
        raise RuntimeError("conditional-twin test arms are incomplete")
    games = _anonymous_groups((str(record["game_id"]) for record in records), "ctg")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        arms = by_sample[sample_id]
        reference = arms[CONDITIONAL_ARMS[0]]
        row: dict[str, Any] = {
            "game_group": games[str(reference["game_id"])],
            "family": str(reference["family"]),
            "actual_action": int(reference["actual_action"]),
        }
        for arm in CONDITIONAL_ARMS:
            if int(arms[arm]["actual_action"]) != row["actual_action"]:
                raise RuntimeError("conditional-twin arm targets disagree")
            row.update(_probability_columns(arm.replace("-", "_"), arms[arm]["probabilities"]))
        rows.append(row)
    destination = output / "conditional-twin-test.parquet"
    _write_parquet(pl.DataFrame(rows, infer_schema_length=None), destination)
    return {"file": destination.name, "rows": len(rows), "games": len(games), "sha256": _sha256(destination)}


def _ensemble_rows(components: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    aligned = [sorted(rows, key=lambda row: str(row["sample_id"])) for rows in components]
    sample_ids = [str(row["sample_id"]) for row in aligned[0]]
    if any([str(row["sample_id"]) for row in rows] != sample_ids for rows in aligned[1:]):
        raise RuntimeError("Persuasion component predictions are not aligned")
    result: list[dict[str, Any]] = []
    for rows in zip(*aligned, strict=True):
        reference = rows[0]
        probabilities = [sum(float(row["probabilities"][index]) for row in rows) / len(rows) for index in range(len(BUYER_CONTINUATION_LABELS))]
        result.append({**reference, "probabilities": probabilities})
    return result


def export_persuasion_continuation(source: Path, output: Path) -> dict[str, Any]:
    corpus_dir = source / "runs/glee-persuasion-buyer-continuation-corpus-v1/deeprmm-final-9587-20260820"
    sequence_release = source / "runs/glee-post-planner-conditional-v3-sequence/deeprmm-final-9587-20260820/release"
    continuation_release = source / "opponent-sequence-models/releases/persuasion-buyer-continuation-v1-deeprmm-final-9587-20260820"
    sequence_manifest = json.loads((sequence_release / "manifest.json").read_text(encoding="utf-8"))
    continuation_manifest = json.loads((continuation_release / "manifest.json").read_text(encoding="utf-8"))
    source_vocabs = CorpusVocabs.from_receipt(json.loads((sequence_release / str(sequence_manifest["vocabulary"]["path"])).read_text(encoding="utf-8")))
    continuation_vocabs = replace(source_vocabs, target_labels={**source_vocabs.target_labels, "persuasion": type(source_vocabs.target_labels["persuasion"])(BUYER_CONTINUATION_LABELS)})
    model_config = ModelConfig(**sequence_manifest["model"])
    index = _ContinuationIndex(corpus_dir, continuation_vocabs)
    dataset = index.dataset(split="test", source_types={"deeprmm"})
    target_lookup = {str(row["sample_id"]): row for row in index.targets}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_rows: list[list[dict[str, Any]]] = []
    heads = {int(component["seed"]): continuation_release / str(component["path"]) for component in continuation_manifest["components"]}
    for component in sequence_manifest["components"]:
        seed = int(component["seed"])
        sequence_payload = torch.load(sequence_release / str(component["path"]), map_location="cpu", weights_only=False)
        model = HierarchicalSequenceTwin(source_vocabs, model_config)
        model.load_state_dict(sequence_payload["model"])
        model.to(device)
        encoded = _encode_partition(
            model,
            dataset,
            vocabs=continuation_vocabs,
            batch_size=256,
            seed=seed,
            device=device,
            target_lookup=target_lookup,
            mixed_precision=device.type == "cuda",
        )
        head_payload = torch.load(heads[seed], map_location="cpu", weights_only=False)
        head = _ReversedPersuasionHead(model_config)
        head.load_state_dict(head_payload["head"])
        head.to(device)
        component_rows.append(_predict_head(head, encoded, batch_size=512, device=device))
        del model, head, encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = _ensemble_rows(component_rows)
    _baseline_receipt, baseline_by_split = _markov_baseline(index)
    baseline_rows = baseline_by_split["test"]
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    games = _anonymous_groups((str(row["game_id"]) for row in selected), "pcg")
    rows: list[dict[str, Any]] = []
    for other in baseline_rows:
        sample_id = str(other["sample_id"])
        row = selected_by_id[sample_id]
        if int(row["actual"]) != int(other["actual"]):
            raise RuntimeError("Persuasion continuation targets disagree")
        rows.append(
            {
                "game_group": games[str(row["game_id"])],
                "actual_action": int(row["actual"]),
                **_probability_columns("selected", row["probabilities"]),
                **_probability_columns("markov", other["probabilities"]),
            }
        )
    destination = output / "persuasion-continuation-test.parquet"
    _write_parquet(pl.DataFrame(rows, infer_schema_length=None), destination)
    return {"file": destination.name, "rows": len(rows), "games": len(games), "sha256": _sha256(destination), "device": device.type}


def export_self_mirror(source: Path, output: Path) -> dict[str, Any]:
    base = source / "runs/glee-public-self-mirror-v1-experiments/public-self-mirror-v1-deeprmm-final-9587-20260820"
    seeds = (1729, 2718)
    components: dict[int, dict[str, Mapping[str, Any]]] = {}
    all_games: list[str] = []
    for seed in seeds:
        records = pl.read_parquet(base / f"seed{seed}/test-predictions.parquet").to_dicts()
        components[seed] = {str(row["sample_id"]): row for row in records}
        all_games.extend(str(row["game_id"]) for row in records)
    sample_ids = set(components[seeds[0]])
    if any(set(components[seed]) != sample_ids for seed in seeds[1:]):
        raise RuntimeError("self-mirror component predictions are not aligned")
    games = _anonymous_groups(all_games, "smg")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        reference = components[seeds[0]][sample_id]
        row: dict[str, Any] = {
            "game_group": games[str(reference["game_id"])],
            "family": str(reference["family"]),
            "target_kind": str(reference["target_kind"]),
            "action_scored": bool(reference["action_scored"]),
            "actual_action": int(reference["actual_action"]) if reference["actual_action"] is not None else None,
            "actual_value": float(reference["actual_value"]) if reference["actual_value"] is not None else None,
        }
        for seed in seeds:
            current = components[seed][sample_id]
            if current["actual_action"] != reference["actual_action"] or current["actual_value"] != reference["actual_value"]:
                raise RuntimeError("self-mirror component targets disagree")
            row[f"seed{seed}_probabilities"] = [float(value) for value in current["action_probabilities"]] if current["action_probabilities"] is not None else None
            row[f"seed{seed}_predicted_value"] = float(current["predicted_value"]) if current["predicted_value"] is not None else None
        rows.append(row)
    destination = output / "self-mirror-test.parquet"
    _write_parquet(pl.DataFrame(rows, infer_schema_length=None), destination)
    action_games = {row["game_group"] for row in rows if row["action_scored"]}
    return {"file": destination.name, "rows": len(rows), "action_scored_games": len(action_games), "sha256": _sha256(destination)}


def export_rating_model(source: Path, output: Path) -> dict[str, Any]:
    path = source / "rating-models/releases/v3.0-deeprmm-final-9587-20260820/chronological-predictions.jsonl"
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["block"]) != 4:
            continue
        rows.append(
            {
                "family": str(record["family"]),
                "actual_rating_delta": float(record["actual_rating_delta"]),
                "v3_prediction": float(record["v3_prediction"]),
            }
        )
    destination = output / "rating-model-heldout.parquet"
    _write_parquet(pl.DataFrame(rows, infer_schema_length=None).sort(["family", "actual_rating_delta", "v3_prediction"]), destination)
    return {"file": destination.name, "rows": len(rows), "sha256": _sha256(destination)}


def _cloud_fields(record: Mapping[str, Any], arm: str, calls: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    receipt = record.get("cloud_receipts", {}).get(arm, {})
    call = calls.get(str(receipt.get("call_id") or ""))
    prefix = arm.replace("_", "-")
    if call is None or not call.get("ok"):
        return {f"{prefix}-call-ok": False, f"{prefix}-elapsed-s": None, f"{prefix}-reasoning-tokens": None, f"{prefix}-input-tokens": None}
    provider = call["provider"]
    return {
        f"{prefix}-call-ok": True,
        f"{prefix}-elapsed-s": float(call["elapsed_s"]),
        f"{prefix}-reasoning-tokens": int(provider["reasoning_tokens"]),
        f"{prefix}-input-tokens": int(provider["tokens_in"]),
    }


def export_dossier_comparison(source: Path, output: Path) -> dict[str, Any]:
    base = source / "reports/glee-bargaining-cloud-baselines-v1"
    scored = [json.loads(line) for line in (base / "scored-predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = {str(row["call_id"]): row for row in (json.loads(line) for line in (base / "calls.jsonl").read_text(encoding="utf-8").splitlines())}
    opponents = _anonymous_groups((str(row["opponent"]["id"]) for row in scored), "dc-o")
    games = _anonymous_groups((str(row["game_id"]) for row in scored), "dc-g")
    rows: list[dict[str, Any]] = []
    for record in scored:
        action_type = str(record["action_type"])
        row: dict[str, Any] = {
            "opponent_group": opponents[str(record["opponent"]["id"])],
            "game_group": games[str(record["game_id"])],
            "action_type": action_type,
            "actual_response": int(bool(record["actual"]["accepted"])) if action_type == "response" else None,
            "actual_proposal": float(record["actual"]["proposal_share"]) if action_type == "proposal" else None,
        }
        for arm in ("hierarchical_program", "direct_context", "prose_dossier"):
            prediction = record["predictions"].get(arm)
            prefix = arm.replace("_", "-")
            row[f"{prefix}-available"] = prediction is not None
            row[f"{prefix}-response-probability"] = float(prediction["probability"]) if prediction is not None and action_type == "response" else None
            row[f"{prefix}-proposal-mean"] = float(prediction["mean"]) if prediction is not None and action_type == "proposal" else None
            row[f"{prefix}-proposal-sigma"] = float(prediction["sigma"]) if prediction is not None and action_type == "proposal" else None
        row.update(_cloud_fields(record, "direct_context", calls))
        row.update(_cloud_fields(record, "prose_dossier", calls))
        rows.append(row)
    destination = output / "dossier-comparison.parquet"
    _write_parquet(pl.DataFrame(rows, infer_schema_length=None), destination)
    return {"file": destination.name, "rows": len(rows), "opponent_groups": len(opponents), "games": len(games), "sha256": _sha256(destination)}


def _contains_model_call(value: Any) -> bool:
    if isinstance(value, dict):
        metadata = value.get("call_metadata")
        return bool(isinstance(metadata, dict) and metadata) or any(_contains_model_call(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_model_call(child) for child in value)
    return False


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def export_execution_telemetry(source: Path, output: Path) -> dict[str, Any]:
    paper = source / "papers/glee-competition-2026"
    receipt = json.loads((paper / "all-history-evidence.json").read_text(encoding="utf-8"))
    prefix = json.loads((paper / "evidence.json").read_text(encoding="utf-8"))["run_source"]
    cutoff = _parse_timestamp(str(receipt["retirement_frontier"]["cutoff"]))
    agent_id = str(receipt["agent"]["agent_id"])
    runs_root = source / "runs"
    final_path = Path(str(prefix["path"])).resolve()
    final_limit = int(prefix["complete_prefix_size_bytes"])
    sources: list[Path] = []
    for manifest_path in sorted(runs_root.glob("glee-*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest.get("agent"), dict) or manifest["agent"].get("agent_id") != agent_id:
            continue
        events_path = manifest_path.parent / "events.jsonl"
        if events_path.is_file():
            sources.append(events_path)
    if len(sources) != int(receipt["source_selection"]["selected_journals"]):
        raise RuntimeError("all-history source selection changed")
    decisions: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    terminals: dict[str, bool] = {}
    for events_path in sources:
        is_final_source = events_path.resolve() == final_path
        with events_path.open("rb") as stream:
            while raw := stream.readline():
                if not raw.endswith(b"\n") or not any(f'"{kind}"'.encode() in raw for kind in TARGET_EVENT_KINDS):
                    continue
                row = json.loads(raw)
                raw_timestamp = row.get("ts")
                if isinstance(raw_timestamp, str) and _parse_timestamp(raw_timestamp) > cutoff:
                    continue
                in_final_prefix = is_final_source and stream.tell() <= final_limit
                kind = row.get("kind")
                if kind == "worker_finished":
                    decision = row.get("decision")
                    if isinstance(decision, dict):
                        decisions.append(
                            {
                                "in_final_v108_prefix": in_final_prefix,
                                "route": "cloud-assisted" if _contains_model_call(decision) else "local",
                                "fallback": decision.get("fallback") is True,
                                "elapsed_s": float(decision["elapsed_s"]),
                            }
                        )
                elif kind == "move_submitted":
                    result = row.get("result")
                    valid = result.get("valid") if isinstance(result, dict) else None
                    submissions.append({"in_final_v108_prefix": in_final_prefix, "valid": valid})
                    decision = row.get("decision")
                    if isinstance(decision, dict):
                        decisions.append(
                            {
                                "in_final_v108_prefix": in_final_prefix,
                                "route": "cloud-assisted" if _contains_model_call(decision) else "local",
                                "fallback": decision.get("fallback") is True,
                                "elapsed_s": float(decision["elapsed_s"]),
                            }
                        )
                elif kind in {"game_completed", "game_completed_during_opponent_turn"}:
                    game_id = row.get("game_id")
                    if isinstance(game_id, str):
                        terminals[game_id] = terminals.get(game_id, False) or in_final_prefix
    decision_path = output / "execution-decisions.parquet"
    submission_path = output / "execution-submissions.parquet"
    terminal_path = output / "execution-terminal-games.parquet"
    _write_parquet(pl.DataFrame(decisions, infer_schema_length=None).sort(["in_final_v108_prefix", "route", "fallback", "elapsed_s"]), decision_path)
    _write_parquet(pl.DataFrame(submissions, infer_schema_length=None).sort(["in_final_v108_prefix", "valid"]), submission_path)
    terminal_rows = [{"in_final_v108_prefix": in_final_prefix} for in_final_prefix in sorted(terminals.values())]
    _write_parquet(pl.DataFrame(terminal_rows, infer_schema_length=None), terminal_path)
    return {
        "decision_file": decision_path.name,
        "decision_rows": len(decisions),
        "decision_sha256": _sha256(decision_path),
        "submission_file": submission_path.name,
        "submission_rows": len(submissions),
        "submission_sha256": _sha256(submission_path),
        "terminal_file": terminal_path.name,
        "terminal_rows": len(terminals),
        "terminal_sha256": _sha256(terminal_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "conditional_twin": export_conditional_twin(source, output),
        "persuasion_continuation": export_persuasion_continuation(source, output),
        "self_mirror": export_self_mirror(source, output),
        "rating_model": export_rating_model(source, output),
        "dossier_comparison": export_dossier_comparison(source, output),
        "execution_telemetry": export_execution_telemetry(source, output),
    }
    print(json.dumps({"contract": "deeprmm-glee-deidentified-reproduction-export-v1", "artifacts": artifacts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
