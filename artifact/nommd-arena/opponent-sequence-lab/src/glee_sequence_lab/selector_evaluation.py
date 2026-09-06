"""Evaluate one frozen local selector on a post-frontier chronological suffix exactly once."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from nommd_arena.glee_selector_backend_replay import read_event_prefix, selector_examples
from nommd_arena.glee_selector_model import LOCAL_SELECTOR_MODEL_CONTRACT, FrozenLinearSelector, admissible_candidate_ids

from .selector_training import FAMILIES, LOCAL_SELECTOR_FRONTIER_CONTRACT, load_frontier


LOCAL_SELECTOR_PROSPECTIVE_EVALUATION_CONTRACT = "glee-local-selector-prospective-evaluation-v1"
MINIMUM_TURNS_PER_FAMILY = 30
MAXIMUM_INFERENCE_MILLISECONDS = 100.0
PROXY_TOLERANCE = 1e-9


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)], 9)


def _release(release_dir: Path) -> tuple[FrozenLinearSelector, dict[str, object]]:
    release_dir = release_dir.resolve()
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != LOCAL_SELECTOR_MODEL_CONTRACT or manifest.get("status") != "frozen-development-candidate-awaiting-prospective-suffix" or manifest.get("live_authority") is not False:
        raise ValueError("prospective evaluation requires one inert frozen selector candidate")
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("selector release has no model receipt")
    model_path = release_dir / str(model.get("path"))
    if _file_sha256(model_path) != model.get("sha256"):
        raise ValueError("selector release model digest mismatch")
    policy = FrozenLinearSelector(model_path)
    if policy.backend_id != manifest.get("release_id"):
        raise ValueError("selector model and manifest release identifiers differ")
    return policy, {"path": str(release_dir), "manifest_sha256": _file_sha256(manifest_path), "model_sha256": policy.model_sha256, "release_id": policy.backend_id}


def _new_examples(frontier_path: Path) -> tuple[dict[str, list[object]], dict[str, object]]:
    development, verified_frontier = load_frontier(frontier_path)
    development_ids = {(group.family, group.turn_id) for group in development}
    frontier_path = frontier_path.resolve()
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    if frontier.get("kind") != LOCAL_SELECTOR_FRONTIER_CONTRACT:
        raise ValueError("unsupported selector frontier")
    arena_root = frontier_path.parents[2]
    output: dict[str, list[object]] = {}
    snapshots: dict[str, object] = {}
    for family in FAMILIES:
        source_receipt = frontier["families"][family]
        source = arena_root / str(source_receipt["path"])
        events, snapshot = read_event_prefix(source)
        examples = [example for example in selector_examples(events) if (family, example.request.turn_id) not in development_ids]
        if any(example.request.family != family for example in examples):
            raise ValueError(f"prospective selector examples crossed the {family} source boundary")
        output[family] = examples
        snapshots[family] = {**snapshot, "frontier_byte_limit": int(source_receipt["byte_limit"]), "frontier_sha256": str(source_receipt["sha256"]), "new_selector_examples": len(examples)}
    return output, {"verified_frontier": verified_frontier, "snapshots": snapshots}


def _evaluate_family(family: str, examples: Sequence[object], policy: FrozenLinearSelector) -> tuple[dict[str, object], list[dict[str, object]]]:
    runtimes = []
    proxy_fallback_deltas = []
    proxy_cloud_deltas = []
    cloud_agreements = 0
    illegal = 0
    nondeterministic = 0
    disagreements = []
    for example in examples:
        wire = example.request.wire_payload()
        candidate_ids = [str(value) for value in wire["candidate_ids"]]
        started = time.perf_counter()
        first = policy(wire)
        runtimes.append(1_000 * (time.perf_counter() - started))
        second = policy(wire)
        nondeterministic += first != second
        selected_id = str(first.get("candidate_id") or "")
        if selected_id not in candidate_ids:
            illegal += 1
            continue
        selected_presentation_index = candidate_ids.index(selected_id)
        selected_canonical_index = next(candidate.index for candidate in example.request.candidate_set.candidates if candidate.action_sha256 == selected_id)
        cloud_canonical_index = int(example.cloud_candidate_index)
        cloud_id = example.request.candidate_set.candidates[cloud_canonical_index].action_sha256
        cloud_agreements += selected_id == cloud_id
        _eligible, scores = admissible_candidate_ids(wire)
        fallback_id = str(wire["fallback_candidate_id"])
        selected_score = scores[selected_id]
        fallback_score = scores[fallback_id]
        cloud_score = scores[cloud_id]
        fallback_delta = selected_score - fallback_score if selected_score is not None and fallback_score is not None else None
        cloud_delta = selected_score - cloud_score if selected_score is not None and cloud_score is not None else None
        if fallback_delta is not None:
            proxy_fallback_deltas.append(fallback_delta)
        if cloud_delta is not None:
            proxy_cloud_deltas.append(cloud_delta)
        if selected_id != cloud_id:
            disagreements.append({"ts": example.ts, "turn_id": example.request.turn_id, "local_candidate_index": selected_canonical_index, "cloud_candidate_index": cloud_canonical_index, "local_presentation_index": selected_presentation_index, "fallback_candidate_index": next(candidate.index for candidate in example.request.candidate_set.candidates if candidate.action_sha256 == fallback_id), "local_minus_fallback_proxy": round(fallback_delta, 9) if fallback_delta is not None else None, "local_minus_cloud_proxy": round(cloud_delta, 9) if cloud_delta is not None else None})
    count = len(examples)
    summary = {
        "family": family,
        "turns": count,
        "support_gate": count >= MINIMUM_TURNS_PER_FAMILY,
        "cloud_agreement": round(cloud_agreements / count, 9) if count else None,
        "illegal_or_non_candidate_actions": illegal,
        "nondeterministic_replays": nondeterministic,
        "runtime_ms_mean": round(statistics.fmean(runtimes), 9) if runtimes else None,
        "runtime_ms_p95": _percentile(runtimes, 0.95),
        "runtime_ms_max": round(max(runtimes), 9) if runtimes else None,
        "deadline_gate": bool(runtimes) and max(runtimes) <= MAXIMUM_INFERENCE_MILLISECONDS,
        "comparable_fallback_proxy_turns": len(proxy_fallback_deltas),
        "local_minus_fallback_proxy_mean": round(statistics.fmean(proxy_fallback_deltas), 9) if proxy_fallback_deltas else None,
        "local_minus_fallback_proxy_minimum": round(min(proxy_fallback_deltas), 9) if proxy_fallback_deltas else None,
        "fallback_proxy_gate": bool(proxy_fallback_deltas) and statistics.fmean(proxy_fallback_deltas) >= -PROXY_TOLERANCE and min(proxy_fallback_deltas) >= -PROXY_TOLERANCE,
        "comparable_cloud_proxy_turns": len(proxy_cloud_deltas),
        "local_minus_cloud_proxy_mean": round(statistics.fmean(proxy_cloud_deltas), 9) if proxy_cloud_deltas else None,
        "cloud_disagreements": len(disagreements),
    }
    summary["safety_gate"] = summary["support_gate"] and summary["deadline_gate"] and summary["fallback_proxy_gate"] and illegal == 0 and nondeterministic == 0
    return summary, disagreements


def evaluate_local_selector(*, frontier_path: Path, release_dir: Path, output_dir: Path) -> dict[str, object]:
    """Open one post-seal suffix and issue a non-activating promotion receipt."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"prospective selector evaluation output already exists: {output_dir}")
    policy, release = _release(release_dir)
    examples, source = _new_examples(frontier_path)
    families = {}
    disagreements = {}
    for family in FAMILIES:
        families[family], disagreements[family] = _evaluate_family(family, examples[family], policy)
    all_gates_pass = all(bool(families[family]["safety_gate"]) for family in FAMILIES)
    result = {
        "schema_version": 1,
        "contract": LOCAL_SELECTOR_PROSPECTIVE_EVALUATION_CONTRACT,
        "status": "proxy-gates-passed-controlled-collector-pilot-eligible" if all_gates_pass else "prospective-block-insufficient-or-gate-failed",
        "release": release,
        "source": source,
        "predeclared_gates": {"minimum_successful_selector_turns_per_family": MINIMUM_TURNS_PER_FAMILY, "maximum_single_inference_milliseconds": MAXIMUM_INFERENCE_MILLISECONDS, "proxy_tolerance": PROXY_TOLERANCE, "zero_illegal_actions": True, "zero_nondeterministic_replays": True, "family_mean_and_minimum_local_minus_fallback_proxy_nonnegative": True},
        "families": families,
        "all_proxy_safety_gates_pass": all_gates_pass,
        "disagreements": disagreements,
        "authority_boundary": "Passing licenses only a separately controlled collector pilot. It does not activate this selector, alter DeepRMM-01, establish observed counterfactual payoff, or authorize broad production authority.",
        "implementation": {"selector_evaluation_sha256": _file_sha256(Path(__file__))},
    }
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, mode=0o700)
    try:
        _write_json(staging / "evaluation.json", result)
        readme = f"""# Local selector prospective evaluation\n\nThe evaluator opened the append-only suffix after the predeclared development frontier and found {families['bargaining']['turns']} Bargaining, {families['negotiation']['turns']} Negotiation, and {families['persuasion']['turns']} Persuasion selector turns. The minimum support requirement is {MINIMUM_TURNS_PER_FAMILY} turns in every family.\n\nThe result status is `{result['status']}`. Passing this proxy gate licenses only a separately controlled collector pilot; it does not activate the release, alter DeepRMM-01, or establish the unobserved payoff of replayed alternatives.\n"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"contract": result["contract"], "status": result["status"], "output_dir": str(output_dir), "all_proxy_safety_gates_pass": all_gates_pass, "turns": {family: families[family]["turns"] for family in FAMILIES}, "evaluation_sha256": _file_sha256(output_dir / "evaluation.json")}
