"""Compile and read collision-safe opponent statistical packages for GLEE."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_behavior_channel_analysis import action_features


STATISTICAL_PACKAGE_CONTRACT = "glee-opponent-statistical-package-v1"
STATISTICAL_PACKAGE_POINTER_CONTRACT = "glee-opponent-statistical-package-pointer-v1"
STATISTICAL_DECISION_FORECAST_CONTRACT = "glee-opponent-statistical-decision-forecast-v2"
OOV_ACTION_TOKEN = "__oov_action__"
OOV_ACTION_CONTEXT = "__oov_action_context__"
VALUE_BIN_LOWER = -2.0
VALUE_BIN_UPPER = 2.0
VALUE_BIN_WIDTH = 0.1
DEFAULT_PROJECTION_CONTEXTS = 8
DEFAULT_PROJECTION_OUTCOMES = 5
BARGAINING_FORECAST_SHARES = (0.05, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.45, 0.5, 0.55, 0.6, 2 / 3, 0.75, 0.8, 0.9, 0.95)


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


def _normalize_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized.casefold() if normalized else None


def _action_observations(move: Mapping[str, object]) -> list[tuple[str, str]]:
    tokens = action_features({"moves": [dict(move)]})
    tokens.pop(next((token for token in tokens if token.startswith("terminal-style|")), ""), None)
    kind = str(move.get("kind") or "")
    if kind == "proposal":
        selected = [token for token in tokens if token.startswith("action|")]
        marker = "|value="
    else:
        selected = [token for token in tokens if token.startswith("decision|")]
        marker = "|decision="
        if not selected:
            selected = [token for token in tokens if token.startswith("action|")]
            marker = "|value="
    observations = []
    for token in sorted(selected):
        condition, separator, outcome = token.rpartition(marker)
        if separator:
            observations.append((condition, f"{marker[1:]}{outcome}"))
    return observations


def _verified_manifest(directory: Path, required_artifacts: tuple[str, ...]) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}
    for name in required_artifacts:
        receipt = artifacts.get(name) if isinstance(artifacts, Mapping) else None
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("sha256"), str):
            raise RuntimeError(f"source manifest omits a hash for {name}: {manifest_path}")
        if _file_digest(directory / name) != receipt["sha256"]:
            raise RuntimeError(f"source artifact hash mismatch: {directory / name}")
    return manifest


def _evidence_tier(observations: int) -> str:
    if observations >= 20:
        return "direct-high"
    if observations >= 8:
        return "direct-moderate"
    if observations >= 3:
        return "direct-sparse"
    if observations > 0:
        return "direct-minimal"
    return "population-only"


def _population_distribution(counts: Mapping[str, int]) -> dict[str, float]:
    augmented = Counter({str(outcome): int(count) for outcome, count in counts.items()})
    augmented[OOV_ACTION_TOKEN] += 1
    total = sum(augmented.values())
    if total <= 0:
        raise RuntimeError("statistical package population distribution is empty")
    return {outcome: count / total for outcome, count in sorted(augmented.items())}


def _smoothed_distribution(identity_counts: Mapping[str, int], population_counts: Mapping[str, int], alpha: float) -> dict[str, float]:
    population = _population_distribution(population_counts)
    direct = Counter({str(outcome): int(count) for outcome, count in identity_counts.items()})
    denominator = sum(direct.values()) + alpha
    return {outcome: (direct[outcome] + alpha * probability) / denominator for outcome, probability in population.items()}


def _context_fields(condition: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in condition.split("|"):
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value
    return fields


def _phase(round_number: int, state: Mapping[str, object]) -> str:
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True and isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 1:
        fraction = min(1.0, max(0.0, (round_number - 1) / (maximum - 1)))
    else:
        total = state.get("total_rounds")
        if isinstance(total, int) and not isinstance(total, bool) and total > 1:
            fraction = min(1.0, max(0.0, (round_number - 1) / (total - 1)))
        else:
            fraction = 1.0 - math.exp(-max(0, round_number - 1) / 12.0)
    return "early" if fraction < 1.0 / 3.0 else "middle" if fraction < 2.0 / 3.0 else "late"


def _other_player(our_player: str) -> str:
    return "player_2" if our_player == "player_1" else "player_1"


def _visible_context(game: Mapping[str, object]) -> dict[str, str]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    round_value = state.get("round")
    round_number = int(round_value) if isinstance(round_value, (int, float)) and not isinstance(round_value, bool) else 1
    family = str(game.get("game_family") or "")
    our_player = str(game.get("your_player") or "")
    opponent_role = str(state.get(f"{_other_player(our_player)}_role") or "none").casefold() if family in {"negotiation", "persuasion"} else "none"
    messages = state.get("messages_allowed") is True or state.get("seller_message_type") in {"binary", "text"}
    return {
        "role": opponent_role,
        "information": "complete" if state.get("complete_information") is True else "incomplete",
        "horizon": "known" if state.get("horizon_known") is True else "unknown",
        "phase": _phase(round_number, state),
        "messages": "enabled" if messages else "disabled",
    }


def _context_rank(condition: str, visible: Mapping[str, str], direct_support: int, population_support: int) -> tuple[int, int, int, str]:
    fields = _context_fields(condition)
    matches = sum(fields.get(key) == value for key, value in visible.items())
    return (matches, direct_support, population_support, condition)


def _project_distribution(distribution: Mapping[str, float], limit: int) -> dict[str, object]:
    ranked = sorted(((str(outcome), float(probability)) for outcome, probability in distribution.items()), key=lambda item: (-item[1], item[0]))
    selected = ranked[:limit]
    omitted = sum(probability for _outcome, probability in ranked[limit:])
    return {
        "outcomes": [{"outcome": outcome, "probability": round(probability, 6)} for outcome, probability in selected],
        "omitted_probability": round(omitted, 6),
    }


def _value_bin(value: float) -> int:
    bounded = min(VALUE_BIN_UPPER, max(VALUE_BIN_LOWER, float(value)))
    return math.floor((bounded - VALUE_BIN_LOWER) / VALUE_BIN_WIDTH + 1e-12)


def _target_condition(prefix: str, kind: str, visible: Mapping[str, str], *, value_bin: int | None = None) -> str:
    condition = f"{prefix}|kind={kind}|role={visible['role']}|information={visible['information']}|horizon={visible['horizon']}|phase={visible['phase']}|messages={visible['messages']}"
    return f"{condition}|value={value_bin}" if value_bin is not None else condition


def _context_weight(candidate: str, target: str) -> float:
    candidate_prefix = candidate.partition("|")[0]
    target_prefix = target.partition("|")[0]
    candidate_fields = _context_fields(candidate)
    target_fields = _context_fields(target)
    if candidate_prefix != target_prefix or candidate_fields.get("kind") != target_fields.get("kind") or candidate_fields.get("role") != target_fields.get("role"):
        return 0.0
    weight = 1.0
    if candidate_fields.get("information") != target_fields.get("information"):
        weight *= 0.35
    if candidate_fields.get("horizon") != target_fields.get("horizon"):
        weight *= 0.65
    phase_order = {"early": 0, "middle": 1, "late": 2}
    candidate_phase = phase_order.get(str(candidate_fields.get("phase")))
    target_phase = phase_order.get(str(target_fields.get("phase")))
    if candidate_phase is None or target_phase is None:
        return 0.0
    phase_distance = abs(candidate_phase - target_phase)
    weight *= (1.0, 0.6, 0.3)[phase_distance]
    if candidate_fields.get("messages") != target_fields.get("messages"):
        weight *= 0.75
    if "value" in target_fields:
        try:
            distance = abs(int(candidate_fields.get("value", "-1000")) - int(target_fields["value"]))
        except ValueError:
            return 0.0
        if distance > 3:
            return 0.0
        weight *= 0.5**distance
    return weight


def _weighted_counts(contexts: Mapping[str, Mapping[str, int]], target: str, *, include_exact: bool) -> tuple[dict[str, float], float, int]:
    result: Counter[str] = Counter()
    effective_support = 0.0
    source_contexts = 0
    for condition, counts in contexts.items():
        if condition == target and not include_exact:
            continue
        weight = _context_weight(str(condition), target)
        if weight <= 0:
            continue
        support = sum(int(value) for value in counts.values())
        if support <= 0:
            continue
        source_contexts += 1
        effective_support += weight * support
        for outcome, count in counts.items():
            result[str(outcome)] += weight * int(count)
    return dict(result), effective_support, source_contexts


def _float_population_distribution(counts: Mapping[str, float]) -> dict[str, float]:
    augmented = Counter({str(outcome): float(count) for outcome, count in counts.items() if float(count) > 0})
    augmented[OOV_ACTION_TOKEN] += 1.0
    total = sum(augmented.values())
    if total <= 0:
        raise RuntimeError("statistical decision forecast population distribution is empty")
    return {outcome: count / total for outcome, count in sorted(augmented.items())}


def _float_smoothed_distribution(identity_counts: Mapping[str, float], population_counts: Mapping[str, float], alpha: float) -> dict[str, float]:
    population = _float_population_distribution(population_counts)
    direct = Counter({str(outcome): float(count) for outcome, count in identity_counts.items() if float(count) > 0})
    outcomes = sorted(set(population) | set(direct))
    denominator = sum(direct.values()) + alpha
    return {outcome: (direct[outcome] + alpha * population.get(outcome, 0.0)) / denominator for outcome in outcomes}


def _accept_probability(distribution: Mapping[str, float]) -> float:
    return sum(float(probability) for outcome, probability in distribution.items() if outcome in {"decision=accept", "decision=acceptoffer"})


def _isotonic_nondecreasing(values: list[float], weights: list[float]) -> list[float]:
    """Return the weighted least-squares nondecreasing projection using pooled adjacent violators."""
    if len(values) != len(weights):
        raise ValueError("isotonic values and weights must have equal lengths")
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("isotonic values must be finite and weights must be positive")
        blocks.append({"start": index, "end": index, "weight": weight, "weighted_sum": weight * value})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["weighted_sum"]) / float(left["weight"])
            right_mean = float(right["weighted_sum"]) / float(right["weight"])
            if left_mean <= right_mean + 1e-15:
                break
            blocks[-2:] = [
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": float(left["weight"]) + float(right["weight"]),
                    "weighted_sum": float(left["weighted_sum"]) + float(right["weighted_sum"]),
                }
            ]
    projected = [0.0] * len(values)
    for block in blocks:
        mean = min(1.0, max(0.0, float(block["weighted_sum"]) / float(block["weight"])))
        for index in range(int(block["start"]), int(block["end"]) + 1):
            projected[index] = mean
    return projected


def _project_response_curves_monotone(rows: list[dict[str, object]], *, alpha: float) -> None:
    """Preserve raw response estimates and expose only monotone curves to counterfactual policy code."""
    if not rows:
        return
    rows.sort(key=lambda row: int(row["offered_opponent_share_bin"]))
    specifications = (
        ("population_accept_probability", lambda row: max(1.0, float(row.get("population_support") or 0) + float(row.get("nearby_population_effective_support") or 0.0))),
        ("exact_identity_accept_probability", lambda row: max(1.0, float(row.get("exact_direct_support") or 0) + alpha)),
        ("hierarchical_identity_accept_probability", lambda row: max(1.0, float(row.get("exact_direct_support") or 0) + float(row.get("nearby_direct_effective_support") or 0.0) + alpha)),
    )
    maximum_adjustment = 0.0
    adjusted_points = 0
    for field, weight_for in specifications:
        raw = [float(row[field]) for row in rows]
        projected = _isotonic_nondecreasing(raw, [weight_for(row) for row in rows])
        raw_field = f"raw_{field}"
        adjustment_field = f"{field}_monotone_adjustment"
        for row, before, after in zip(rows, raw, projected, strict=True):
            adjustment = after - before
            row[raw_field] = round(before, 6)
            row[field] = round(after, 6)
            row[adjustment_field] = round(adjustment, 6)
            maximum_adjustment = max(maximum_adjustment, abs(adjustment))
            adjusted_points += int(abs(adjustment) > 5e-7)
    for row in rows:
        row["monotone_projection"] = {
            "method": "weighted-pooled-adjacent-violators",
            "direction": "acceptance probability nondecreasing in offered opponent share",
            "raw_probabilities_retained": True,
        }
    rows[0]["curve_projection_summary"] = {"adjusted_field_points": adjusted_points, "maximum_absolute_adjustment": round(maximum_adjustment, 6)}


def _mean_value_bin(distribution: Mapping[str, float]) -> float | None:
    weighted = 0.0
    support = 0.0
    for outcome, probability in distribution.items():
        if not outcome.startswith("value="):
            continue
        try:
            value = int(outcome.partition("=")[2])
        except ValueError:
            continue
        weighted += value * float(probability)
        support += float(probability)
    return weighted / support if support > 0 else None


def _bargaining_decision_model(*, visible: Mapping[str, str], population_contexts: Mapping[str, Mapping[str, int]], direct_contexts: Mapping[str, Mapping[str, int]], alpha: float, identity_exact: bool) -> dict[str, object]:
    response_rows = []
    seen_bins: set[int] = set()
    for share in BARGAINING_FORECAST_SHARES:
        value_bin = _value_bin(share)
        if value_bin in seen_bins:
            continue
        seen_bins.add(value_bin)
        target = _target_condition("decision", "response", visible, value_bin=value_bin)
        exact_direct = {str(outcome): float(count) for outcome, count in direct_contexts.get(target, {}).items()} if identity_exact else {}
        nearby_direct, nearby_support, nearby_contexts = _weighted_counts(direct_contexts, target, include_exact=False) if identity_exact else ({}, 0.0, 0)
        exact_population = {str(outcome): float(count) for outcome, count in population_contexts.get(target, {}).items()}
        nearby_population, nearby_population_support, nearby_population_contexts = _weighted_counts(population_contexts, target, include_exact=False)
        population_basis = exact_population if exact_population else nearby_population
        if not population_basis:
            continue
        exact_support = sum(exact_direct.values())
        combined_direct = Counter(exact_direct)
        combined_direct.update(nearby_direct)
        population_distribution = _float_population_distribution(population_basis)
        exact_distribution = _float_smoothed_distribution(exact_direct, population_basis, alpha) if identity_exact else population_distribution
        backoff_distribution = _float_smoothed_distribution(combined_direct, population_basis, alpha) if identity_exact else population_distribution
        evidence_scope = "exact-current-context" if exact_support > 0 else "nearby-context" if nearby_support > 0 else "population-only"
        response_rows.append(
            {
                "offered_opponent_share_bin": value_bin,
                "representative_opponent_share": round(VALUE_BIN_LOWER + VALUE_BIN_WIDTH * value_bin + VALUE_BIN_WIDTH / 2, 6),
                "target_condition": target,
                "evidence_scope": evidence_scope,
                "exact_direct_support": int(exact_support),
                "nearby_direct_effective_support": round(nearby_support, 6),
                "nearby_direct_contexts": nearby_contexts,
                "population_support": int(sum(exact_population.values())),
                "nearby_population_effective_support": round(nearby_population_support, 6),
                "nearby_population_contexts": nearby_population_contexts,
                "population_accept_probability": round(_accept_probability(population_distribution), 6),
                "exact_identity_accept_probability": round(_accept_probability(exact_distribution), 6),
                "hierarchical_identity_accept_probability": round(_accept_probability(backoff_distribution), 6),
            }
        )
    _project_response_curves_monotone(response_rows, alpha=alpha)
    proposal_target = _target_condition("action", "proposal", visible)
    exact_direct = {str(outcome): float(count) for outcome, count in direct_contexts.get(proposal_target, {}).items()} if identity_exact else {}
    nearby_direct, nearby_support, nearby_contexts = _weighted_counts(direct_contexts, proposal_target, include_exact=False) if identity_exact else ({}, 0.0, 0)
    exact_population = {str(outcome): float(count) for outcome, count in population_contexts.get(proposal_target, {}).items()}
    nearby_population, nearby_population_support, nearby_population_contexts = _weighted_counts(population_contexts, proposal_target, include_exact=False)
    population_basis = exact_population if exact_population else nearby_population
    proposal: dict[str, object] = {"target_condition": proposal_target, "status": "unavailable"}
    if population_basis:
        combined_direct = Counter(exact_direct)
        combined_direct.update(nearby_direct)
        population_distribution = _float_population_distribution(population_basis)
        backoff_distribution = _float_smoothed_distribution(combined_direct, population_basis, alpha) if identity_exact else population_distribution
        exact_support = sum(exact_direct.values())
        proposal = {
            "target_condition": proposal_target,
            "status": "available",
            "evidence_scope": "exact-current-context" if exact_support > 0 else "nearby-context" if nearby_support > 0 else "population-only",
            "exact_direct_support": int(exact_support),
            "nearby_direct_effective_support": round(nearby_support, 6),
            "nearby_direct_contexts": nearby_contexts,
            "population_support": int(sum(exact_population.values())),
            "nearby_population_effective_support": round(nearby_population_support, 6),
            "nearby_population_contexts": nearby_population_contexts,
            "population_mean_opponent_share": round((float(_mean_value_bin(population_distribution)) - 20 + 0.5) / 10, 6) if _mean_value_bin(population_distribution) is not None else None,
            "hierarchical_identity_mean_opponent_share": round((float(_mean_value_bin(backoff_distribution)) - 20 + 0.5) / 10, 6) if _mean_value_bin(backoff_distribution) is not None else None,
            "hierarchical_distribution": _project_distribution(backoff_distribution, 5),
        }
    return {
        "contract": STATISTICAL_DECISION_FORECAST_CONTRACT,
        "authority": "bounded-offer-candidate-scoring-for-support-gated-exact-identities",
        "backoff_contract": "same move kind and role; information 0.35, horizon 0.65, adjacent/far phase 0.6/0.3, message mode 0.75, and adjacent response-value bins 0.5 per bin through distance 3",
        "response_to_our_offer": response_rows,
        "opponent_proposal": proposal,
        "policy_gate": {"status": "bounded-authoritative", "scope": "offer candidate scoring only", "identity_requirement": "exact-current-label", "minimum_effective_support": 2.0, "maximum_blend_weight": 0.35, "minimum_expected-value_improvement": 0.02},
    }


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _nearest_share_row(rows: Iterable[Mapping[str, object]], share: float) -> dict[str, object] | None:
    selected = [dict(row) for row in rows if isinstance(row.get("offered_opponent_share_bin"), int)]
    return min(selected, key=lambda row: abs(int(row["offered_opponent_share_bin"]) - _value_bin(share))) if selected else None


def compact_bargaining_decision_forecast(package_context: object, game: Mapping[str, object], advisor_context: object) -> dict[str, object] | None:
    """Project only current-decision package evidence while retaining the immutable raw snapshot upstream."""
    if not isinstance(package_context, Mapping) or package_context.get("family") != "bargaining":
        return None
    local = package_context.get("decision_local_model") if isinstance(package_context.get("decision_local_model"), Mapping) else None
    if local is None or local.get("contract") != STATISTICAL_DECISION_FORECAST_CONTRACT:
        return None
    identity = package_context.get("identity_resolution") if isinstance(package_context.get("identity_resolution"), Mapping) else {}
    evidence = package_context.get("evidence") if isinstance(package_context.get("evidence"), Mapping) else {}
    overlay = package_context.get("live_overlay") if isinstance(package_context.get("live_overlay"), Mapping) else None
    action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or game.get("phase") or "")
    result: dict[str, object] = {
        "contract": STATISTICAL_DECISION_FORECAST_CONTRACT,
        "family": "bargaining",
        "frontier": "current-turn-snapshot-before-model-inference",
        "identity_resolution": dict(identity),
        "identity_wide_evidence": {key: evidence.get(key) for key in ("games", "observations", "contexts", "tier")},
        "authority": "bounded-offer-candidate-scoring" if identity.get("status") == "exact-current-label" else "diagnostic-only",
        "policy_gate": dict(local.get("policy_gate") or {}),
        "raw_package_retained_upstream": True,
    }
    if overlay is not None:
        result["live_overlay"] = {key: overlay.get(key) for key in ("contract", "revision", "state_sha256")}
    if action_type == "offer":
        rows = local.get("response_to_our_offer") if isinstance(local.get("response_to_our_offer"), list) else []
        advisor = advisor_context if isinstance(advisor_context, Mapping) else {}
        response = advisor.get("response_to_our_numeric_offer") if isinstance(advisor.get("response_to_our_numeric_offer"), Mapping) else {}
        curve = response.get("curve_by_opponent_share") if isinstance(response.get("curve_by_opponent_share"), list) else []
        advisor_curve = [(float(row["opponent_share"]), float(row["accept_probability_v2"])) for row in curve if isinstance(row, Mapping) and _finite_number(row.get("opponent_share")) is not None and _finite_number(row.get("accept_probability_v2")) is not None]
        continuation = advisor.get("behavioral_continuation") if isinstance(advisor.get("behavioral_continuation"), Mapping) else {}
        modeled = continuation.get("modeled_offer_policy") if isinstance(continuation.get("modeled_offer_policy"), Mapping) else {}
        myopic = response.get("myopic_no_continuation_candidate") if isinstance(response.get("myopic_no_continuation_candidate"), Mapping) else {}
        requested: list[tuple[str, float]] = []
        for source, value in (("specialized-modeled-offer", modeled.get("opponent_share")), ("specialized-myopic-offer", myopic.get("opponent_share"))):
            numeric = _finite_number(value)
            if numeric is not None:
                requested.append((source, numeric))
        compared = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            share = float(raw.get("representative_opponent_share") or 0.5)
            advisor_probability = min(advisor_curve, key=lambda pair: abs(pair[0] - share))[1] if advisor_curve else None
            package_probability = _finite_number(raw.get("hierarchical_identity_accept_probability"))
            divergence = package_probability - advisor_probability if package_probability is not None and advisor_probability is not None else None
            compared.append((dict(raw), advisor_probability, divergence))
        if compared:
            largest = max((item for item in compared if item[2] is not None), key=lambda item: abs(float(item[2])), default=None)
            if largest is not None:
                requested.append(("largest-package-advisor-divergence", float(largest[0]["representative_opponent_share"])))
            strongest = max(compared, key=lambda item: (int(item[0].get("exact_direct_support") or 0), float(item[0].get("nearby_direct_effective_support") or 0.0)))
            requested.append(("strongest-current-decision-support", float(strongest[0]["representative_opponent_share"])))
        projected = []
        selected_bins: set[int] = set()
        for source, share in requested:
            row = _nearest_share_row(rows, share)
            if row is None or int(row["offered_opponent_share_bin"]) in selected_bins:
                continue
            selected_bins.add(int(row["offered_opponent_share_bin"]))
            advisor_probability = min(advisor_curve, key=lambda pair: abs(pair[0] - share))[1] if advisor_curve else None
            package_probability = _finite_number(row.get("hierarchical_identity_accept_probability"))
            projected.append(
                {
                    "source": source,
                    "requested_opponent_share": round(share, 6),
                    "offered_opponent_share_bin": row.get("offered_opponent_share_bin"),
                    "evidence_scope": row.get("evidence_scope"),
                    "exact_direct_support": row.get("exact_direct_support"),
                    "nearby_direct_effective_support": row.get("nearby_direct_effective_support"),
                    "population_accept_probability": row.get("population_accept_probability"),
                    "raw_population_accept_probability": row.get("raw_population_accept_probability"),
                    "hierarchical_identity_accept_probability": package_probability,
                    "raw_hierarchical_identity_accept_probability": row.get("raw_hierarchical_identity_accept_probability"),
                    "monotone_projection": row.get("monotone_projection"),
                    "specialized_advisor_accept_probability": round(advisor_probability, 6) if advisor_probability is not None else None,
                    "package_minus_specialized_probability": round(package_probability - advisor_probability, 6) if package_probability is not None and advisor_probability is not None else None,
                }
            )
            if len(projected) >= 4:
                break
        result.update(
            {
                "decision": "choose-our-offer",
                "response_forecasts": projected,
                "local_support_summary": {
                    "projected_candidates": len(projected),
                    "candidates_with_exact_direct_support": sum(int(row.get("exact_direct_support") or 0) > 0 for row in projected),
                    "candidates_with_nearby_direct_support": sum(float(row.get("nearby_direct_effective_support") or 0.0) > 0 for row in projected),
                },
                "interpretation": "Current-decision exact support and nearby-context support are separate. V2.17 may blend the monotone hierarchical curve into offer candidate scoring only for exact identities that clear the frozen support gate; all deterministic guards remain authoritative.",
            }
        )
    else:
        proposal = dict(local.get("opponent_proposal") or {})
        advisor = advisor_context if isinstance(advisor_context, Mapping) else {}
        specialized = advisor.get("opponent_proposal_share") if isinstance(advisor.get("opponent_proposal_share"), Mapping) else {}
        specialized_mean = _finite_number(specialized.get("mean"))
        package_mean = _finite_number(proposal.get("hierarchical_identity_mean_opponent_share"))
        result.update(
            {
                "decision": "respond-to-observed-offer",
                "proposal_forecast": proposal,
                "specialized_advisor_proposal_mean": specialized_mean,
                "package_minus_specialized_proposal_mean": round(package_mean - specialized_mean, 6) if package_mean is not None and specialized_mean is not None else None,
                "interpretation": "The current offer is observed and authoritative. This proposal prior is only a surprise and continuation diagnostic.",
            }
        )
    return result


def _decoded_projected_distribution(distribution: object) -> dict[str, object]:
    if not isinstance(distribution, Mapping):
        return {"outcomes": [], "omitted_probability": None}
    decoded = []
    for raw in distribution.get("outcomes", []):
        if not isinstance(raw, Mapping):
            continue
        outcome = str(raw.get("outcome") or "")
        item = {"outcome": outcome, "probability": raw.get("probability")}
        if outcome.startswith("value="):
            try:
                value_bin = int(outcome.partition("=")[2])
            except ValueError:
                value_bin = None
            if value_bin is not None:
                item["normalized_value_bin"] = value_bin
                item["normalized_value_midpoint"] = round(VALUE_BIN_LOWER + VALUE_BIN_WIDTH * value_bin + VALUE_BIN_WIDTH / 2, 6)
        decoded.append(item)
    return {"outcomes": decoded, "omitted_probability": distribution.get("omitted_probability")}


def _expected_next_opponent_prefix(family: str, action_type: str) -> tuple[str, str] | None:
    if family == "negotiation":
        return ("decision", "response") if action_type in {"offer", "proposal"} else ("action", "proposal") if action_type == "decision" else None
    if family == "persuasion":
        return ("decision", "response") if action_type in {"seller_recommendation", "seller_message"} else ("decision", "signal") if action_type == "buyer_decision" else None
    return None


def compact_opponent_decision_forecast(package_context: object, game: Mapping[str, object], advisor_context: object = None) -> dict[str, object] | None:
    """Project a bounded current-turn opponent prior for every family while retaining the raw snapshot upstream."""
    family = str(game.get("game_family") or "")
    if family == "bargaining":
        return compact_bargaining_decision_forecast(package_context, game, advisor_context)
    if family not in {"negotiation", "persuasion"} or not isinstance(package_context, Mapping) or package_context.get("family") != family:
        return None
    action_model = package_context.get("action_model") if isinstance(package_context.get("action_model"), Mapping) else {}
    contexts = action_model.get("contexts") if isinstance(action_model.get("contexts"), list) else []
    action_type = str((game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}).get("type") or game.get("phase") or "")
    expected = _expected_next_opponent_prefix(family, action_type)
    selected = []
    for raw in contexts:
        if not isinstance(raw, Mapping):
            continue
        condition = str(raw.get("condition") or "")
        prefix = condition.partition("|")[0]
        fields = _context_fields(condition)
        if expected is not None and (prefix, fields.get("kind")) != expected:
            continue
        selected.append(
            {
                "condition": condition,
                "direct_support": int(raw.get("direct_support") or 0),
                "population_support": int(raw.get("population_support") or 0),
                "distribution": _decoded_projected_distribution(raw.get("distribution")),
            }
        )
        if len(selected) >= 4:
            break
    if not selected:
        for raw in contexts[:3]:
            if isinstance(raw, Mapping):
                selected.append({"condition": raw.get("condition"), "direct_support": int(raw.get("direct_support") or 0), "population_support": int(raw.get("population_support") or 0), "distribution": _decoded_projected_distribution(raw.get("distribution"))})
    identity = package_context.get("identity_resolution") if isinstance(package_context.get("identity_resolution"), Mapping) else {}
    evidence = package_context.get("evidence") if isinstance(package_context.get("evidence"), Mapping) else {}
    overlay = package_context.get("live_overlay") if isinstance(package_context.get("live_overlay"), Mapping) else None
    result: dict[str, object] = {
        "contract": STATISTICAL_DECISION_FORECAST_CONTRACT,
        "family": family,
        "frontier": "current-turn-snapshot-before-model-inference",
        "decision": action_type,
        "expected_next_opponent_move": {"prefix": expected[0], "kind": expected[1]} if expected is not None else None,
        "identity_resolution": dict(identity),
        "identity_wide_evidence": {key: evidence.get(key) for key in ("games", "observations", "contexts", "tier")},
        "current_context_projections": selected,
        "authority": "advisory-only",
        "raw_package_retained_upstream": True,
        "interpretation": "A population-smoothed prior over the opponent's likely next move. Current authenticated state, the family-specific causal advisor, payoff arithmetic, and legal-action guards remain authoritative.",
    }
    if overlay is not None:
        result["live_overlay"] = {key: overlay.get(key) for key in ("contract", "revision", "state_sha256")}
    return result


def bargaining_submitted_offer_forecast(package_context: object, game: Mapping[str, object], action: Mapping[str, object], advisor_submission: object) -> dict[str, object] | None:
    """Freeze population, exact, and hierarchical response probabilities for one submitted offer."""
    if not isinstance(package_context, Mapping):
        return None
    local = package_context.get("decision_local_model") if isinstance(package_context.get("decision_local_model"), Mapping) else {}
    rows = local.get("response_to_our_offer") if isinstance(local.get("response_to_our_offer"), list) else []
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    money = _finite_number(state.get("money_to_divide"))
    your_player = str(game.get("your_player") or "")
    opponent_gain_keys = ("player_2_gain", "bob_gain") if your_player == "player_1" else ("player_1_gain", "alice_gain")
    opponent_gain = next((_finite_number(action.get(key)) for key in opponent_gain_keys if _finite_number(action.get(key)) is not None), None)
    if money is None or money <= 0 or opponent_gain is None:
        return None
    share = opponent_gain / money
    row = _nearest_share_row(rows, share)
    if row is None:
        return None
    advisor = advisor_submission if isinstance(advisor_submission, Mapping) else {}
    advisor_probability = _finite_number(advisor.get("opponent_acceptance_probability_v2"))
    package_probability = _finite_number(row.get("hierarchical_identity_accept_probability"))
    identity = package_context.get("identity_resolution") if isinstance(package_context.get("identity_resolution"), Mapping) else {}
    return {
        "contract": STATISTICAL_DECISION_FORECAST_CONTRACT,
        "frontier": "registered-before-network-submission",
        "game_id": game.get("game_id"),
        "round": int(state.get("round") or 1),
        "your_player": your_player,
        "offered_opponent_share": round(share, 8),
        "offered_opponent_share_bin": row.get("offered_opponent_share_bin"),
        "identity_resolution": dict(identity),
        "evidence_scope": row.get("evidence_scope"),
        "exact_direct_support": row.get("exact_direct_support"),
        "nearby_direct_effective_support": row.get("nearby_direct_effective_support"),
        "population_accept_probability": row.get("population_accept_probability"),
        "raw_population_accept_probability": row.get("raw_population_accept_probability"),
        "exact_identity_accept_probability": row.get("exact_identity_accept_probability"),
        "raw_exact_identity_accept_probability": row.get("raw_exact_identity_accept_probability"),
        "hierarchical_identity_accept_probability": package_probability,
        "raw_hierarchical_identity_accept_probability": row.get("raw_hierarchical_identity_accept_probability"),
        "monotone_projection": row.get("monotone_projection"),
        "specialized_advisor_accept_probability": advisor_probability,
        "package_minus_specialized_probability": round(package_probability - advisor_probability, 6) if package_probability is not None and advisor_probability is not None else None,
        "authority": "bounded-offer-candidate-scoring" if identity.get("status") == "exact-current-label" and row.get("evidence_scope") != "population-only" and float(row.get("exact_direct_support") or 0) + float(row.get("nearby_direct_effective_support") or 0.0) >= 2.0 else "diagnostic-only",
        "action_changed": False,
    }


class OpponentStatisticalPackageCompiler:
    """Compile one deterministic full public-ID by family package matrix."""

    def __init__(self, *, identity_dir: Path, behavior_dir: Path, channel_dir: Path, activity_summary: Path, output_root: Path, release: str | None = None) -> None:
        self.identity_dir = identity_dir.resolve()
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.activity_summary = activity_summary.resolve()
        self.output_root = output_root.resolve()
        self.release = release

    def run(self) -> dict[str, object]:
        identity_manifest = _verified_manifest(self.identity_dir, ("identity-registry.jsonl", "summary.json"))
        behavior_manifest = _verified_manifest(self.behavior_dir, ("behavior-games.jsonl", "summary.json"))
        channel_manifest = _verified_manifest(self.channel_dir, ("summary.json",))
        frontier = int(identity_manifest["frontier_sequence"])
        if int(behavior_manifest["frontier_sequence"]) != frontier:
            raise RuntimeError("identity and behavior source frontiers differ")
        channel_summary = json.loads((self.channel_dir / "summary.json").read_text(encoding="utf-8"))
        if int(channel_summary["source"]["frontier_sequence"]) != frontier:
            raise RuntimeError("behavior-channel and identity source frontiers differ")
        activity = json.loads(self.activity_summary.read_text(encoding="utf-8"))
        if int(activity["source_frontier"]["frontier_sequence"]) != frontier:
            raise RuntimeError("activity and identity source frontiers differ")
        self_ids = {str(value) for value in activity["source_frontier"]["self_player_ids"].values()}
        release = self.release or f"frontier-{frontier}-v1"
        if Path(release).name != release or release in {"", ".", ".."}:
            raise ValueError("release must be one path component")
        release_dir = self.output_root / release
        if release_dir.exists() and any(release_dir.iterdir()):
            raise FileExistsError(f"statistical-package release is not empty: {release_dir}")
        release_dir.mkdir(parents=True, exist_ok=True)

        registry_rows = [json.loads(line) for line in (self.identity_dir / "identity-registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
        registry_by_key = {(str(row["family"]), str(row["player_id"])): row for row in registry_rows}
        canonical_by_id: dict[str, dict[str, object]] = {}
        for row in sorted(registry_rows, key=lambda item: (str(item["player_id"]), str(item["family"]))):
            player_id = str(row["player_id"])
            canonical_by_id.setdefault(player_id, row)
        opponent_ids = sorted(player_id for player_id in canonical_by_id if player_id not in self_ids)

        population: dict[str, dict[str, Counter[str]]] = {family: defaultdict(Counter) for family in GLEE_FAMILIES}
        direct: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        game_counts: Counter[tuple[str, str]] = Counter()
        with (self.behavior_dir / "behavior-games.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                family = str(record["family"])
                player_id = str(record["public_player_id"])
                if player_id in self_ids:
                    raise RuntimeError("behavior corpus unexpectedly attributes an opponent move to Self")
                if player_id not in canonical_by_id:
                    raise RuntimeError(f"behavior corpus ID is absent from the identity registry: {player_id}")
                game_counts[(family, player_id)] += 1
                for move in record.get("moves", []):
                    if not isinstance(move, Mapping):
                        continue
                    for condition, outcome in _action_observations(move):
                        population[family][condition][outcome] += 1
                        direct[(family, player_id)][condition][outcome] += 1

        alphas = {family: float(channel_summary["families"][family]["action"]["selected_hyperparameters"]["alpha"]) for family in GLEE_FAMILIES}
        population_artifact = {
            "contract": STATISTICAL_PACKAGE_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "families": {
                family: {
                    "alpha": alphas[family],
                    "games": sum(game_counts[(family, player_id)] for player_id in opponent_ids),
                    "observations": sum(sum(counts.values()) for counts in population[family].values()),
                    "contexts": {condition: dict(sorted(counts.items())) for condition, counts in sorted(population[family].items())},
                }
                for family in GLEE_FAMILIES
            },
        }
        package_rows = []
        for player_id in opponent_ids:
            canonical = canonical_by_id[player_id]
            for family in GLEE_FAMILIES:
                registry = registry_by_key.get((family, player_id))
                counts = direct[(family, player_id)]
                observations = sum(sum(values.values()) for values in counts.values())
                label = str((registry or canonical).get("current_label") or "")
                aliases = sorted({str(value) for value in (registry or canonical).get("aliases", []) if value})
                package_rows.append(
                    {
                        "contract": STATISTICAL_PACKAGE_CONTRACT,
                        "schema_version": 1,
                        "family": family,
                        "public_player_id": player_id,
                        "current_label": label,
                        "current_normalized_label": _normalize_label(label),
                        "aliases": aliases,
                        "registry_status": "observed-in-family" if registry is not None else "not-observed-in-family",
                        "present_at_frontier": bool(registry and registry.get("present_at_frontier") is True),
                        "classification": {
                            "is_baseline": bool(registry and isinstance(registry.get("latest_present_metadata"), Mapping) and registry["latest_present_metadata"].get("is_baseline") is True),
                            "is_benchmark": bool(registry and isinstance(registry.get("latest_present_metadata"), Mapping) and registry["latest_present_metadata"].get("is_benchmark") is True),
                        },
                        "evidence": {
                            "games": int(game_counts[(family, player_id)]),
                            "observations": observations,
                            "contexts": len(counts),
                            "tier": _evidence_tier(observations),
                        },
                        "direct_counts": {condition: dict(sorted(values.items())) for condition, values in sorted(counts.items())},
                    }
                )

        _write_json(release_dir / "population.json", population_artifact)
        _write_jsonl(release_dir / "packages.jsonl", package_rows)
        family_summary = {}
        for family in GLEE_FAMILIES:
            rows = [row for row in package_rows if row["family"] == family]
            current_labels: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                if row["present_at_frontier"] and row["current_normalized_label"]:
                    current_labels[str(row["current_normalized_label"])].append(str(row["public_player_id"]))
            family_summary[family] = {
                "packages": len(rows),
                "registry_observed": sum(row["registry_status"] == "observed-in-family" for row in rows),
                "present_at_frontier": sum(bool(row["present_at_frontier"]) for row in rows),
                "with_direct_evidence": sum(int(row["evidence"]["observations"]) > 0 for row in rows),
                "population_only": sum(int(row["evidence"]["observations"]) == 0 for row in rows),
                "direct_games": sum(int(row["evidence"]["games"]) for row in rows),
                "direct_observations": sum(int(row["evidence"]["observations"]) for row in rows),
                "current_collision_labels": sorted(label for label, ids in current_labels.items() if len(ids) > 1),
            }
        summary = {
            "contract": STATISTICAL_PACKAGE_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "release": release,
            "status": "compiled-live-context-advisory",
            "matrix": {"public_opponents": len(opponent_ids), "families": len(GLEE_FAMILIES), "packages": len(package_rows), "complete_cross_product": len(package_rows) == len(opponent_ids) * len(GLEE_FAMILIES)},
            "families": family_summary,
            "design": {
                "representation": "one family population count model plus sparse per-public-ID count deltas",
                "smoothing": "Dirichlet-style interpolation toward the family population distribution",
                "identity_resolution": "current stable public ID only; collisions, hidden opponents, and absent labels use family population",
                "action_authority": "advisory",
                "value_binning": {"lower": VALUE_BIN_LOWER, "upper": VALUE_BIN_UPPER, "width": VALUE_BIN_WIDTH},
            },
        }
        _write_json(release_dir / "summary.json", summary)
        readme = f"# GLEE opponent statistical package {release}\n\nThis release compiles a complete {len(opponent_ids)}-opponent by 3-family matrix at public frontier {frontier}. It stores each family population once and stores only sparse direct count deltas per stable public ID, so population-only cells do not duplicate data.\n\nExact current labels may select an ID package only when the public registry resolves the label uniquely. Hidden identities, missing labels, and collision sets such as `RESERVE` use the family-population package unless a separately frozen identity-routing protocol authorizes a candidate route.\n\nThe package is advisory evidence, not an action command. Current authenticated state, legal-action constraints, deterministic guards, and family-specific executable advisors retain their existing authority.\n"
        _atomic_text(release_dir / "README.md", readme)
        artifact_names = ("README.md", "packages.jsonl", "population.json", "summary.json")
        manifest = {
            "contract": STATISTICAL_PACKAGE_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": frontier,
            "release": release,
            "sources": {
                "identity_manifest_sha256": _file_digest(self.identity_dir / "manifest.json"),
                "behavior_manifest_sha256": _file_digest(self.behavior_dir / "manifest.json"),
                "channel_manifest_sha256": _file_digest(self.channel_dir / "manifest.json"),
                "activity_summary_sha256": _file_digest(self.activity_summary),
                "identity_contract": identity_manifest.get("contract"),
                "behavior_contract": behavior_manifest.get("contract"),
                "channel_contract": channel_manifest.get("contract"),
            },
            "implementation_sha256": _file_digest(Path(__file__)),
            "artifacts": {name: {"bytes": (release_dir / name).stat().st_size, "sha256": _file_digest(release_dir / name)} for name in artifact_names},
        }
        _write_json(release_dir / "manifest.json", manifest)
        pointer = {
            "contract": STATISTICAL_PACKAGE_POINTER_CONTRACT,
            "schema_version": 1,
            "release": release,
            "frontier_sequence": frontier,
            "manifest_sha256": _file_digest(release_dir / "manifest.json"),
        }
        _write_json(self.output_root / "current.json", pointer)
        return summary


class OpponentStatisticalPackageReader:
    """Load one immutable package release and emit bounded live projections."""

    def __init__(self, root: Path, *, context_limit: int = DEFAULT_PROJECTION_CONTEXTS, outcome_limit: int = DEFAULT_PROJECTION_OUTCOMES) -> None:
        if context_limit < 1 or outcome_limit < 1:
            raise ValueError("statistical-package projection limits must be positive")
        self.root = root.resolve()
        pointer_path = self.root / "current.json"
        self.pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if self.pointer.get("contract") != STATISTICAL_PACKAGE_POINTER_CONTRACT:
            raise RuntimeError("opponent statistical-package pointer has an unsupported contract")
        release = str(self.pointer.get("release") or "")
        if Path(release).name != release or release in {"", ".", ".."}:
            raise RuntimeError("opponent statistical-package pointer contains an invalid release")
        self.release_dir = (self.root / release).resolve()
        if self.release_dir.parent != self.root:
            raise RuntimeError("opponent statistical-package release escapes its root")
        manifest_path = self.release_dir / "manifest.json"
        if _file_digest(manifest_path) != self.pointer.get("manifest_sha256"):
            raise RuntimeError("opponent statistical-package manifest differs from current pointer")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != STATISTICAL_PACKAGE_CONTRACT:
            raise RuntimeError("opponent statistical-package manifest has an unsupported contract")
        for name, receipt in self.manifest["artifacts"].items():
            if _file_digest(self.release_dir / name) != receipt["sha256"]:
                raise RuntimeError(f"opponent statistical-package artifact hash mismatch: {name}")
        self.population = json.loads((self.release_dir / "population.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (self.release_dir / "packages.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.rows = {(str(row["family"]), str(row["public_player_id"])): row for row in rows}
        self.current_labels: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in rows:
            label = row.get("current_normalized_label")
            if row.get("present_at_frontier") is True and isinstance(label, str) and label:
                self.current_labels[(str(row["family"]), label)].append(str(row["public_player_id"]))
        self.context_limit = context_limit
        self.outcome_limit = outcome_limit
        self.receipt = {
            "contract": STATISTICAL_PACKAGE_CONTRACT,
            "release": release,
            "frontier_sequence": int(self.manifest["frontier_sequence"]),
            "manifest_sha256": str(self.pointer["manifest_sha256"]),
            "projection_context_limit": context_limit,
            "projection_outcome_limit": outcome_limit,
        }

    def _resolve(self, game: Mapping[str, object]) -> tuple[str, str | None, list[str], str | None]:
        family = str(game.get("game_family") or "")
        opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
        name = " ".join(str(opponent.get("name") or "").split()).strip()
        if opponent.get("type") == "hidden" or not name:
            return "hidden-population", None, [], name or None
        normalized = _normalize_label(name)
        candidates = sorted(self.current_labels.get((family, str(normalized)), [])) if normalized else []
        if len(candidates) == 1:
            return "exact-current-label", candidates[0], candidates, name
        if len(candidates) > 1:
            return "current-label-collision", None, candidates, name
        return "no-current-label-match", None, [], name

    def view(self, game: Mapping[str, object]) -> dict[str, object]:
        family = str(game.get("game_family") or "")
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported statistical-package family: {family}")
        resolution, player_id, candidates, label = self._resolve(game)
        row = self.rows.get((family, player_id)) if player_id is not None else None
        family_population = self.population["families"][family]
        population_contexts = family_population["contexts"]
        direct_contexts = row["direct_counts"] if isinstance(row, Mapping) else {}
        visible = _visible_context(game)
        ranked = sorted(
            population_contexts,
            key=lambda condition: _context_rank(condition, visible, sum(direct_contexts.get(condition, {}).values()), sum(population_contexts[condition].values())),
            reverse=True,
        )
        selected = ranked[: self.context_limit if row is not None else min(4, self.context_limit)]
        alpha = float(family_population["alpha"])
        contexts = []
        for condition in selected:
            population_counts = population_contexts[condition]
            direct_counts = direct_contexts.get(condition, {})
            distribution = _smoothed_distribution(direct_counts, population_counts, alpha) if row is not None else _population_distribution(population_counts)
            contexts.append(
                {
                    "condition": condition,
                    "direct_support": sum(int(value) for value in direct_counts.values()),
                    "population_support": sum(int(value) for value in population_counts.values()),
                    "distribution": _project_distribution(distribution, self.outcome_limit),
                }
            )
        evidence = dict(row["evidence"]) if isinstance(row, Mapping) else {"games": 0, "observations": 0, "contexts": 0, "tier": "population-only"}
        result = {
            "contract": STATISTICAL_PACKAGE_CONTRACT,
            "release": self.receipt["release"],
            "frontier_sequence": self.receipt["frontier_sequence"],
            "family": family,
            "identity_resolution": {
                "status": resolution,
                "display_label": label,
                "public_player_id": player_id,
                "candidate_public_player_ids": candidates,
                "collision_safe": player_id is not None or not candidates,
            },
            "evidence": {**evidence, "alpha": alpha},
            "action_model": {
                "authority": "advisory",
                "semantics": "population-smoothed opponent next-action distribution conditional on visible context",
                "value_bin_contract": {"index_i_interval": "[-2.0 + 0.1*i, -2.0 + 0.1*(i+1)) with endpoint clipping", "normalized_scale": True},
                "visible_context": visible,
                "contexts": contexts,
                "projected_contexts": len(contexts),
                "stored_contexts": len(population_contexts),
            },
        }
        if family == "bargaining":
            result["decision_local_model"] = _bargaining_decision_model(visible=visible, population_contexts=population_contexts, direct_contexts=direct_contexts, alpha=alpha, identity_exact=row is not None)
        return result
