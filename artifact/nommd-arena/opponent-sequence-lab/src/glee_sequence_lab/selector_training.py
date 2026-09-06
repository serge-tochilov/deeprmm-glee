"""Fit and freeze a compact family-balanced local selector over sealed GLEE prefixes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from nommd_arena.glee_selector_backend_replay import SelectorReplayExample, read_event_prefix, selector_examples
from nommd_arena.glee_selector_model import LOCAL_SELECTOR_ADMISSIBILITY_CONTRACT, LOCAL_SELECTOR_FEATURE_CONTRACT, LOCAL_SELECTOR_MODEL_CONTRACT, FrozenLinearSelector, admissible_candidate_ids, candidate_feature_maps


LOCAL_SELECTOR_TRAINING_CONTRACT = "glee-local-selector-training-v1"
LOCAL_SELECTOR_FRONTIER_CONTRACT = "glee-local-selector-evidence-frontier-v1"
FAMILIES = ("bargaining", "negotiation", "persuasion")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ChoiceGroup:
    family: str
    ts: str
    turn_id: str
    wire: Mapping[str, object]
    candidate_ids: tuple[str, ...]
    features: tuple[Mapping[str, float], ...]
    target_index: int
    cloud_index: int
    fallback_index: int
    reference_scores: tuple[float | None, ...]
    cloud_was_admissible: bool


def _choice_group(example: SelectorReplayExample) -> ChoiceGroup:
    wire = example.request.wire_payload()
    feature_rows = candidate_feature_maps(wire)
    candidate_ids = tuple(str(value) for value in wire["candidate_ids"])
    fallback_id = str(wire["fallback_candidate_id"])
    eligible, scores = admissible_candidate_ids(wire)
    cloud_id = candidate_ids[example.cloud_candidate_index]
    if cloud_id in eligible:
        target_id = cloud_id
    else:
        ranked = sorted(eligible, key=lambda candidate_id: (-(float(scores[candidate_id]) if scores[candidate_id] is not None else -math.inf), candidate_id != fallback_id, candidate_id))
        target_id = ranked[0]
    return ChoiceGroup(
        family=example.request.family,
        ts=example.ts,
        turn_id=example.request.turn_id,
        wire=wire,
        candidate_ids=candidate_ids,
        features=tuple(feature_rows),
        target_index=candidate_ids.index(target_id),
        cloud_index=example.cloud_candidate_index,
        fallback_index=candidate_ids.index(fallback_id),
        reference_scores=tuple(scores[candidate_id] for candidate_id in candidate_ids),
        cloud_was_admissible=cloud_id in eligible,
    )


def load_frontier(frontier_path: Path) -> tuple[list[ChoiceGroup], dict[str, object]]:
    """Verify and parse only the development side of one predeclared evidence frontier."""
    frontier_path = frontier_path.resolve()
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    if frontier.get("kind") != LOCAL_SELECTOR_FRONTIER_CONTRACT:
        raise ValueError("unsupported local selector frontier")
    families = frontier.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("local selector frontier must pin all 3 families")
    arena_root = frontier_path.parents[2]
    groups: list[ChoiceGroup] = []
    inputs: dict[str, object] = {}
    for family in FAMILIES:
        receipt = families[family]
        if not isinstance(receipt, Mapping):
            raise ValueError(f"invalid frontier receipt for {family}")
        source = arena_root / str(receipt["path"])
        events, parsed = read_event_prefix(source, byte_limit=int(receipt["byte_limit"]), expected_sha256=str(receipt["sha256"]))
        examples = selector_examples(events)
        if any(example.request.family != family for example in examples):
            raise ValueError(f"selector examples crossed the {family} source boundary")
        groups.extend(_choice_group(example) for example in examples)
        inputs[family] = {**parsed, "selector_examples": len(examples)}
    groups.sort(key=lambda group: (group.family, group.ts, group.turn_id))
    return groups, {"path": str(frontier_path.relative_to(arena_root)), "sha256": _file_sha256(frontier_path), "sealed_at": frontier.get("sealed_at"), "source_commit": frontier.get("source_commit"), "inputs": inputs}


def _split(groups: Sequence[ChoiceGroup]) -> dict[str, list[ChoiceGroup]]:
    output: dict[str, list[ChoiceGroup]] = {"train": [], "validation": [], "test": []}
    for family in FAMILIES:
        rows = sorted((group for group in groups if group.family == family), key=lambda group: (group.ts, group.turn_id))
        if len(rows) < 6:
            raise ValueError(f"local selector needs at least 6 development groups for {family}")
        train_end = max(1, math.floor(0.70 * len(rows)))
        validation_end = max(train_end + 1, math.floor(0.85 * len(rows)))
        validation_end = min(validation_end, len(rows) - 1)
        output["train"].extend(rows[:train_end])
        output["validation"].extend(rows[train_end:validation_end])
        output["test"].extend(rows[validation_end:])
    return output


def _feature_names(groups: Sequence[ChoiceGroup], *, minimum_varying_groups: int = 2) -> list[str]:
    counts: Counter[str] = Counter()
    for group in groups:
        names = {name for row in group.features for name in row}
        for name in names:
            if len({row.get(name, 0.0) for row in group.features}) > 1:
                counts[name] += 1
    selected = sorted(name for name, count in counts.items() if count >= minimum_varying_groups)
    if not selected:
        raise ValueError("local selector feature projection has no recurring within-choice variation")
    return selected


def _scaler(groups: Sequence[ChoiceGroup], feature_names: Sequence[str]) -> tuple[list[float], list[float]]:
    rows = [features for group in groups for features in group.features]
    means = [sum(row.get(name, 0.0) for row in rows) / len(rows) for name in feature_names]
    scales = []
    for name, mean in zip(feature_names, means, strict=True):
        variance = sum((row.get(name, 0.0) - mean) ** 2 for row in rows) / len(rows)
        scales.append(max(math.sqrt(variance), 1e-6))
    return means, scales


def _tensors(groups: Sequence[ChoiceGroup], feature_names: Sequence[str], means: Sequence[float], scales: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_candidates = max(len(group.features) for group in groups)
    features = torch.zeros((len(groups), max_candidates, len(feature_names)), dtype=torch.float64)
    mask = torch.zeros((len(groups), max_candidates), dtype=torch.bool)
    targets = torch.tensor([group.target_index for group in groups], dtype=torch.long)
    families = torch.tensor([FAMILIES.index(group.family) for group in groups], dtype=torch.long)
    family_counts = Counter(group.family for group in groups)
    weights = torch.tensor([1.0 / family_counts[group.family] for group in groups], dtype=torch.float64)
    weights /= weights.sum()
    for group_index, group in enumerate(groups):
        for candidate_index, row in enumerate(group.features):
            mask[group_index, candidate_index] = True
            for feature_index, name in enumerate(feature_names):
                features[group_index, candidate_index, feature_index] = (row.get(name, 0.0) - means[feature_index]) / scales[feature_index]
    return features, mask, targets, families, weights


def _fit(groups: Sequence[ChoiceGroup], feature_names: Sequence[str], means: Sequence[float], scales: Sequence[float], *, global_l2: float, family_l2: float) -> tuple[list[float], dict[str, list[float]], dict[str, object]]:
    torch.set_num_threads(1)
    torch.manual_seed(17_291)
    features, mask, targets, families, group_weights = _tensors(groups, feature_names, means, scales)
    global_weights = torch.zeros(len(feature_names), dtype=torch.float64, requires_grad=True)
    family_residuals = torch.zeros((len(FAMILIES), len(feature_names)), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([global_weights, family_residuals], lr=1.0, max_iter=350, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad()
        effective = global_weights.unsqueeze(0) + family_residuals[families]
        logits = (features * effective.unsqueeze(1)).sum(dim=-1).masked_fill(~mask, -1e30)
        losses = F.cross_entropy(logits, targets, reduction="none")
        objective = (losses * group_weights).sum() + 0.5 * global_l2 * global_weights.square().sum() + 0.5 * family_l2 * family_residuals.square().sum()
        objective.backward()
        evaluations += 1
        return objective

    final_loss = float(optimizer.step(closure).detach())
    return (
        [float(value) for value in global_weights.detach()],
        {family: [float(value) for value in family_residuals.detach()[index]] for index, family in enumerate(FAMILIES)},
        {"optimizer": "torch-lbfgs-strong-wolfe", "closure_evaluations": evaluations, "objective": final_loss, "global_l2": global_l2, "family_l2": family_l2},
    )


def _probabilities(group: ChoiceGroup, feature_names: Sequence[str], means: Sequence[float], scales: Sequence[float], global_weights: Sequence[float], family_residuals: Mapping[str, Sequence[float]]) -> list[float]:
    effective = [global_weights[index] + family_residuals[group.family][index] for index in range(len(feature_names))]
    scores = []
    for row in group.features:
        scores.append(sum(((row.get(name, 0.0) - means[index]) / scales[index]) * effective[index] for index, name in enumerate(feature_names)))
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _metrics(groups: Sequence[ChoiceGroup], feature_names: Sequence[str], means: Sequence[float], scales: Sequence[float], global_weights: Sequence[float], family_residuals: Mapping[str, Sequence[float]]) -> dict[str, object]:
    by_family: dict[str, list[ChoiceGroup]] = defaultdict(list)
    for group in groups:
        by_family[group.family].append(group)

    def summarize(rows: Sequence[ChoiceGroup]) -> dict[str, object]:
        losses = []
        correct = 0
        cloud_agreement = 0
        fallback_correct = 0
        reference_correct = 0
        uniform_losses = []
        proxy_deltas = []
        for group in rows:
            probabilities = _probabilities(group, feature_names, means, scales, global_weights, family_residuals)
            prediction = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index != group.fallback_index, group.candidate_ids[index]))[0]
            comparable = [index for index, score in enumerate(group.reference_scores) if score is not None]
            reference_prediction = sorted(comparable, key=lambda index: (-float(group.reference_scores[index]), index != group.fallback_index, group.candidate_ids[index]))[0] if comparable else group.fallback_index
            losses.append(-math.log(max(probabilities[group.target_index], 1e-12)))
            uniform_losses.append(math.log(len(probabilities)))
            correct += prediction == group.target_index
            cloud_agreement += prediction == group.cloud_index
            fallback_correct += group.fallback_index == group.target_index
            reference_correct += reference_prediction == group.target_index
            selected_score = group.reference_scores[prediction]
            fallback_score = group.reference_scores[group.fallback_index]
            if selected_score is not None and fallback_score is not None:
                proxy_deltas.append(selected_score - fallback_score)
        return {
            "groups": len(rows),
            "nll": round(sum(losses) / len(losses), 9) if losses else None,
            "uniform_nll": round(sum(uniform_losses) / len(uniform_losses), 9) if uniform_losses else None,
            "target_accuracy": round(correct / len(rows), 9) if rows else None,
            "cloud_agreement": round(cloud_agreement / len(rows), 9) if rows else None,
            "fallback_target_accuracy": round(fallback_correct / len(rows), 9) if rows else None,
            "reference_max_target_accuracy": round(reference_correct / len(rows), 9) if rows else None,
            "mean_reference_value_minus_fallback": round(sum(proxy_deltas) / len(proxy_deltas), 9) if proxy_deltas else None,
            "comparable_proxy_groups": len(proxy_deltas),
        }

    families = {family: summarize(by_family[family]) for family in FAMILIES}
    equal_family_nll = sum(float(families[family]["nll"]) for family in FAMILIES) / len(FAMILIES)
    equal_family_accuracy = sum(float(families[family]["target_accuracy"]) for family in FAMILIES) / len(FAMILIES)
    return {"all": summarize(groups), "families": families, "equal_family_nll": round(equal_family_nll, 9), "equal_family_target_accuracy": round(equal_family_accuracy, 9)}


def _model_payload(*, release_id: str, feature_names: Sequence[str], means: Sequence[float], scales: Sequence[float], global_weights: Sequence[float], family_residuals: Mapping[str, Sequence[float]], hyperparameters: Mapping[str, object], source: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": LOCAL_SELECTOR_MODEL_CONTRACT,
        "release_id": release_id,
        "feature_contract": LOCAL_SELECTOR_FEATURE_CONTRACT,
        "admissibility_contract": LOCAL_SELECTOR_ADMISSIBILITY_CONTRACT,
        "families": list(FAMILIES),
        "feature_names": list(feature_names),
        "scaler": {"means": [round(value, 12) for value in means], "scales": [round(value, 12) for value in scales]},
        "weights": {"global": [round(value, 12) for value in global_weights], "family_residuals": {family: [round(value, 12) for value in family_residuals[family]] for family in FAMILIES}},
        "hyperparameters": dict(hyperparameters),
        "source": dict(source),
        "selection": "maximum pooled-plus-family conditional-logit score within the nonnegative-versus-fallback bounded-proxy admissible set",
        "tie_break": "fallback-then-candidate-id",
    }


def train_local_selector(*, frontier_path: Path, output_dir: Path, release_id: str) -> dict[str, object]:
    """Select regularization chronologically, refit all sealed development groups, and freeze one inert release."""
    started = time.perf_counter()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"local selector output already exists: {output_dir}")
    if not release_id.strip():
        raise ValueError("local selector release ID cannot be empty")
    groups, frontier = load_frontier(frontier_path)
    splits = _split(groups)
    development_names = _feature_names(splits["train"])
    development_means, development_scales = _scaler(splits["train"], development_names)
    trials = []
    fitted = {}
    for global_l2 in (0.003, 0.03, 0.3):
        for family_l2 in (0.03, 0.3, 3.0):
            global_weights, family_residuals, fit = _fit(splits["train"], development_names, development_means, development_scales, global_l2=global_l2, family_l2=family_l2)
            validation = _metrics(splits["validation"], development_names, development_means, development_scales, global_weights, family_residuals)
            key = (global_l2, family_l2)
            fitted[key] = (global_weights, family_residuals, fit)
            trials.append({"global_l2": global_l2, "family_l2": family_l2, "fit": fit, "validation": validation})
    selected = min(trials, key=lambda trial: (float(trial["validation"]["equal_family_nll"]), -float(trial["family_l2"]), -float(trial["global_l2"])))
    key = (float(selected["global_l2"]), float(selected["family_l2"]))
    selected_global, selected_residuals, selected_fit = fitted[key]
    internal_test = _metrics(splits["test"], development_names, development_means, development_scales, selected_global, selected_residuals)
    final_names = _feature_names(groups)
    final_means, final_scales = _scaler(groups, final_names)
    final_global, final_residuals, final_fit = _fit(groups, final_names, final_means, final_scales, global_l2=key[0], family_l2=key[1])
    counts = Counter(group.family for group in groups)
    source = {"frontier_path": frontier["path"], "frontier_sha256": frontier["sha256"], "development_groups": len(groups), "family_groups": dict(sorted(counts.items()))}
    model = _model_payload(release_id=release_id, feature_names=final_names, means=final_means, scales=final_scales, global_weights=final_global, family_residuals=final_residuals, hyperparameters={"global_l2": key[0], "family_l2": key[1], "optimizer": final_fit["optimizer"], "family_objective_weighting": "equal-total-weight-per-family"}, source=source)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, mode=0o700)
    try:
        _write_json(staging / "model.json", model)
        policy = FrozenLinearSelector(staging / "model.json")
        deterministic = True
        release_choices = []
        runtimes = []
        for group in groups:
            before = time.perf_counter()
            first = policy(group.wire)
            runtimes.append(1_000 * (time.perf_counter() - before))
            second = policy(group.wire)
            deterministic = deterministic and first == second
            release_choices.append(group.candidate_ids.index(str(first["candidate_id"])))
        if not deterministic:
            raise RuntimeError("frozen local selector inference is nondeterministic")
        release_proxy = {family: [] for family in FAMILIES}
        for group, prediction in zip(groups, release_choices, strict=True):
            selected_score = group.reference_scores[prediction]
            fallback_score = group.reference_scores[group.fallback_index]
            if selected_score is not None and fallback_score is not None:
                release_proxy[group.family].append(selected_score - fallback_score)
        if any(values and min(values) < -1e-10 for values in release_proxy.values()):
            raise RuntimeError("frozen selector violated its nonnegative fallback-proxy gate")
        evaluation = {
            "contract": LOCAL_SELECTOR_TRAINING_CONTRACT,
            "status": "frozen-development-candidate-awaiting-prospective-suffix",
            "release_id": release_id,
            "frontier": frontier,
            "split_counts": {name: dict(sorted(Counter(group.family for group in rows).items())) for name, rows in splits.items()},
            "feature_selection": {"development_features": len(development_names), "release_features": len(final_names), "minimum_varying_groups": 2},
            "hyperparameter_trials": trials,
            "selected_hyperparameters": {"global_l2": key[0], "family_l2": key[1], "fit": selected_fit},
            "internal_development_test": internal_test,
            "release_refit": {"fit": final_fit, "deterministic_replay": deterministic, "all_choices_admissible": True, "runtime_ms_mean": round(sum(runtimes) / len(runtimes), 9), "runtime_ms_max": round(max(runtimes), 9), "minimum_reference_value_minus_fallback": {family: round(min(values), 9) if values else None for family, values in release_proxy.items()}},
            "label_semantics": {"cloud_choice_when_admissible": sum(group.cloud_was_admissible for group in groups), "reference-max_substitution_when_cloud_choice_violated_fallback_proxy": sum(not group.cloud_was_admissible for group in groups)},
            "boundary": "This is an inert development candidate. It has no live authority and cannot be promoted until the post-frontier suffix passes the predeclared legality, determinism, deadline, and family-wise nonnegative fallback-proxy gates.",
        }
        _write_json(staging / "evaluation.json", evaluation)
        implementation = {"selector_training_sha256": _file_sha256(Path(__file__))}
        implementation["selector_model_sha256"] = _file_sha256(Path(__file__).resolve().parents[3] / "src" / "nommd_arena" / "glee_selector_model.py")
        implementation["selector_replay_sha256"] = _file_sha256(Path(__file__).resolve().parents[3] / "src" / "nommd_arena" / "glee_selector_backend_replay.py")
        manifest = {
            "schema_version": 1,
            "contract": LOCAL_SELECTOR_MODEL_CONTRACT,
            "release_id": release_id,
            "status": "frozen-development-candidate-awaiting-prospective-suffix",
            "model": {"path": "model.json", "sha256": _file_sha256(staging / "model.json"), "bytes": (staging / "model.json").stat().st_size},
            "evaluation": {"path": "evaluation.json", "sha256": _file_sha256(staging / "evaluation.json")},
            "source": source,
            "implementation": implementation,
            "live_authority": False,
        }
        _write_json(staging / "manifest.json", manifest)
        readme = f"""# Local selector {release_id}\n\nThis release distills the successful Terra selector choices that satisfy the existing bounded-payoff fallback gate, substitutes the maximum reference-value candidate when a historical cloud choice violates that gate, and fits one pooled conditional-logit policy with strongly regularized family residuals. Each family has equal aggregate objective weight despite unequal turn counts.\n\nThe sealed development corpus contains {len(groups)} staged turns: {counts['bargaining']} Bargaining, {counts['negotiation']} Negotiation, and {counts['persuasion']} Persuasion. Regularization was selected on chronological family-specific validation blocks, inspected once on family-specific development-test suffixes, and then refit over the full sealed development prefix. This internal suffix is development evidence, not the prospective promotion block.\n\nThe model can select only a frozen candidate whose established response-weighted bounded value is no lower than the deterministic fallback candidate. Missing comparable evidence collapses authority to fallback. Inference is standard-library-only, deterministic, and leaves legality, mechanics, final safeguards, timing, scheduling, and submission outside the model.\n\nThe release is behaviorally inert. Bytes written after the source frontier remain untouched, and live collector use is forbidden until the frozen evaluator opens a sufficiently supported prospective suffix and all predeclared family gates pass.\n"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"contract": LOCAL_SELECTOR_TRAINING_CONTRACT, "release_id": release_id, "output_dir": str(output_dir), "manifest_sha256": _file_sha256(output_dir / "manifest.json"), "development_groups": len(groups), "family_groups": dict(sorted(counts.items())), "internal_development_test": internal_test, "elapsed_seconds": round(time.perf_counter() - started, 3)}
