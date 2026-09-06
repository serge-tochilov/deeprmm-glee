"""Sealed, restart-safe live advisor for GLEE Negotiation v2.10."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glee_advisor_contracts import NEGOTIATION_ADVISOR_MODEL_VERSION as MODEL_VERSION, NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE, NEGOTIATION_BUYER_SCALE_AWARE_DECISION_VALUES, NEGOTIATION_LIVE_ENGINE_VERSION as LIVE_ENGINE_VERSION, NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE, NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_VALUES, NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE, NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_VALUES
from .glee_named_dossier import named_opponent_id, normalize_opponent_name
from .glee_negotiation_rating_v2_4 import NegotiationRatingSurrogate
from .glee_negotiation_twin_v2 import GaussianComponent, GaussianMixture, NegotiationContext, NegotiationDecisionRow, NegotiationGameEvidence, NegotiationOpponentModelV2, PriorObservation, _history_context, extract_negotiation_game, opponent_demand, opponent_surplus_share, price_from_opponent_demand, prior_observations
from .glee_negotiation_twin_v2_1 import ENGINE_VERSION as BASE_ENGINE_VERSION, NegotiationModelConfigV21, NegotiationOpponentModelV21
from .glee_negotiation_validation_v2 import NegotiationValidationConfig
from .immutable_pack import load_ordered_json_objects, seal_json_objects


SCHEMA_VERSION = 1
SEED_KIND = "glee-negotiation-live-v2-seed"
SEED_REFERENCE_KIND = "glee-negotiation-live-v2-seed-reference"
JOURNAL_KIND = "glee-negotiation-live-v2-completion"
FRONTIER_JOURNAL_KIND = "glee-negotiation-live-v2-game-frontier"
RESPONSE_EXPERTS = ("adaptive_v2_1", "adaptive_v2", "target_only")
PROPOSAL_EXPERTS = RESPONSE_EXPERTS
NEGOTIATION_LIVE_POLICY_CONTRACT = "glee-negotiation-live-policy-v1"
NEGOTIATION_HOT_PARAMETERS = (
    "positive_surplus_minimum_tick",
    "positive_surplus_relative_tick",
    "reciprocal_concession_match_ratio",
    "shadow_reject_max_current_share",
    "shadow_reject_min_expected_gain_share",
    "shadow_reject_min_candidate_share",
    "shadow_reject_min_population_games",
    "one_round_incomplete_seller_markup",
    "complete_information_min_own_surplus_share",
    "stalled_exit_min_opponent_offers",
    "stalled_exit_recent_offer_window",
    "stalled_exit_min_reservation_gap_ratio",
    "stalled_exit_min_projected_opponent_offers",
)


@dataclass(frozen=True)
class NegotiationLiveConfig:
    """Frozen v2.10 settings for calibrated forecasts and narrow policy authority."""

    complete_response_weights: tuple[float, float, float] = (0.9, 0.05, 0.05)
    incomplete_response_weights: tuple[float, float, float] = (0.2, 0.45, 0.35)
    proposal_weights: tuple[float, float, float] = (0.45, 0.35, 0.2)
    proposal_final_contraction: float = 0.85
    within_game_learning_rate: float = 0.75
    within_game_uniform_mix: float = 0.03
    expert_loss_floor: float = -4.0
    expert_loss_ceiling: float = 8.0
    adverse_counterproposal_quantile: float = 0.8
    candidate_demand_grid: tuple[float, ...] = (-1.0, -0.6, -0.35, -0.18, -0.08, 0.0)
    consistency_value_tolerance: float = 1e-6
    consistency_relative_tolerance: float = 0.001
    minimum_support_rows: int = 20
    unknown_horizon_round_quantile: float = 0.99
    continuation_max_opponent_offers: int = 4
    continuation_concession_window: int = 3
    continuation_concession_decay: float = 0.75
    continuation_plateau_decay: float = 0.5
    unknown_horizon_default_stop_hazard: float = 0.08
    unknown_horizon_hazard_prior_games: float = 4.0
    latent_threshold_complete_weight: float = 0.25
    latent_threshold_incomplete_weight: float = 0.25
    latent_threshold_min_scale: float = 0.04
    latent_threshold_slope_scale: float = 2.0
    latent_threshold_projected_concession: float = 0.5
    latent_threshold_rejection_margin_scale: float = 0.1
    positive_surplus_minimum_tick: float = 0.01
    positive_surplus_relative_tick: float = 1e-6
    reciprocal_concession_match_ratio: float = 1.0
    shadow_reject_max_current_share: float = 0.05
    shadow_reject_min_expected_gain_share: float = 0.1
    shadow_reject_min_candidate_share: float = 0.2
    shadow_reject_min_population_games: int = 100
    one_round_incomplete_seller_markup: float = 0.25
    complete_information_min_own_surplus_share: float = 0.5
    buyer_minimum_accept_surplus_share: float = 0.01
    stalled_exit_min_opponent_offers: int = 6
    stalled_exit_recent_offer_window: int = 4
    stalled_exit_min_reservation_gap_ratio: float = 0.1
    stalled_exit_min_projected_opponent_offers: float = 20.0
    hardening_probe_budget: int = 1
    hardening_response_minimum_relative_progress: float = 0.001

    def validate(self) -> None:
        for name in ("complete_response_weights", "incomplete_response_weights", "proposal_weights"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or any(value <= 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
                raise ValueError(f"{name} must contain 3 positive weights summing to one")
        if self.within_game_learning_rate < 0 or not 0 <= self.within_game_uniform_mix < 1:
            raise ValueError("within-game expert adaptation settings are invalid")
        if self.expert_loss_floor >= self.expert_loss_ceiling:
            raise ValueError("expert loss bounds are invalid")
        if not 0 < self.adverse_counterproposal_quantile < 1:
            raise ValueError("counterproposal quantile must lie in (0, 1)")
        if not 0 < self.proposal_final_contraction <= 1:
            raise ValueError("proposal_final_contraction must lie in (0, 1]")
        if not self.candidate_demand_grid or any(not math.isfinite(value) or value > 0 for value in self.candidate_demand_grid):
            raise ValueError("candidate demands must be finite and no greater than our reservation boundary")
        if self.consistency_value_tolerance < 0 or self.consistency_relative_tolerance < 0 or self.minimum_support_rows < 1:
            raise ValueError("consistency tolerance and support threshold are invalid")
        if not 0.5 < self.unknown_horizon_round_quantile < 1:
            raise ValueError("unknown-horizon round quantile must lie in (0.5, 1)")
        if self.continuation_max_opponent_offers < 1 or self.continuation_concession_window < 1:
            raise ValueError("continuation path limits must be positive")
        if not 0 < self.continuation_concession_decay <= 1 or not 0 < self.continuation_plateau_decay <= 1:
            raise ValueError("continuation concession decay settings must lie in (0, 1]")
        if not 0 < self.unknown_horizon_default_stop_hazard < 1 or self.unknown_horizon_hazard_prior_games <= 0:
            raise ValueError("unknown-horizon stopping-hazard settings are invalid")
        if not 0 <= self.latent_threshold_complete_weight <= 1 or not 0 <= self.latent_threshold_incomplete_weight <= 1:
            raise ValueError("latent-threshold blend weights must lie in [0, 1]")
        if self.latent_threshold_min_scale <= 0 or self.latent_threshold_slope_scale <= 0:
            raise ValueError("latent-threshold scale settings must be positive")
        if not 0 <= self.latent_threshold_projected_concession <= 1 or self.latent_threshold_rejection_margin_scale < 0:
            raise ValueError("latent-threshold location settings are invalid")
        if self.positive_surplus_minimum_tick <= 0 or self.positive_surplus_relative_tick <= 0:
            raise ValueError("positive-surplus tick settings must be positive")
        if not 0 < self.reciprocal_concession_match_ratio <= 1:
            raise ValueError("reciprocal concession match ratio must lie in (0, 1]")
        if not 0 <= self.shadow_reject_max_current_share < self.shadow_reject_min_candidate_share <= 1:
            raise ValueError("shadow-reject share thresholds are invalid")
        if self.shadow_reject_min_expected_gain_share <= 0 or self.shadow_reject_min_population_games < 1:
            raise ValueError("shadow-reject support thresholds are invalid")
        if self.one_round_incomplete_seller_markup <= 0:
            raise ValueError("one-round incomplete-information seller markup must be positive")
        if not 0 < self.complete_information_min_own_surplus_share < 1:
            raise ValueError("complete-information minimum own-surplus share must lie in (0, 1)")
        if not 0 < self.buyer_minimum_accept_surplus_share < self.complete_information_min_own_surplus_share:
            raise ValueError("buyer minimum acceptance share must be positive and below the proposal surplus boundary")
        if self.stalled_exit_min_opponent_offers < 2 or self.stalled_exit_recent_offer_window < 2:
            raise ValueError("stalled-exit offer counts must be at least 2")
        if self.stalled_exit_recent_offer_window > self.stalled_exit_min_opponent_offers:
            raise ValueError("stalled-exit recent window cannot exceed its minimum opponent-offer count")
        if self.stalled_exit_min_reservation_gap_ratio <= 0 or self.stalled_exit_min_projected_opponent_offers <= 0:
            raise ValueError("stalled-exit gap and projection thresholds must be positive")
        if self.hardening_probe_budget != 1 or self.hardening_response_minimum_relative_progress <= 0:
            raise ValueError("Negotiation uses exactly one hardening probe and requires positive relative response progress")


def negotiation_turn_config(base: NegotiationLiveConfig, live_policy: Mapping[str, object] | None) -> tuple[NegotiationLiveConfig, dict[str, object] | None]:
    """Overlay only the stable narrow-control contract for one externally pinned game."""
    if live_policy is None:
        return base, None
    if live_policy.get("schema_version") != 1 or live_policy.get("contract") != NEGOTIATION_LIVE_POLICY_CONTRACT or not isinstance(live_policy.get("revision"), str):
        raise ValueError("Negotiation live policy has an incompatible contract")
    parameters = live_policy.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(NEGOTIATION_HOT_PARAMETERS):
        raise ValueError("Negotiation live policy has an incomplete parameter set")
    raw_features = live_policy.get("features")
    allowed_features = {NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE, NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE, NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE}
    if raw_features is not None and (not isinstance(raw_features, Mapping) or set(raw_features) - allowed_features):
        raise ValueError("Negotiation live policy has invalid features")
    features = dict(raw_features) if isinstance(raw_features, Mapping) else {}
    if NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE in features and features[NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE] not in NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_VALUES:
        raise ValueError("Negotiation live policy has an invalid positive-surplus exit authority")
    if NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE in features and features[NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE] not in NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_VALUES:
        raise ValueError("Negotiation live policy has an invalid one-round incomplete-information seller authority")
    if NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE in features and features[NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE] not in NEGOTIATION_BUYER_SCALE_AWARE_DECISION_VALUES:
        raise ValueError("Negotiation live policy has an invalid buyer scale-aware decision mode")
    config = replace(base, **dict(parameters))
    config.validate()
    receipt = {
        "contract": NEGOTIATION_LIVE_POLICY_CONTRACT,
        "revision": live_policy["revision"],
        "release_sha256": live_policy.get("release_sha256"),
        "parameters": {name: getattr(config, name) for name in NEGOTIATION_HOT_PARAMETERS},
        "features": features,
        "scope": "externally pinned per game; narrow action controls only",
    }
    return config, receipt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _seed_source_reference(source: Path, project_root: Path) -> dict[str, str]:
    source = source.resolve()
    try:
        relative = source.relative_to(project_root.resolve())
    except ValueError:
        return {"kind": "absolute", "path": str(source)}
    return {"kind": "project-relative", "path": str(relative)}


def _resolve_seed_source(reference: Mapping[str, object], project_root: Path) -> Path:
    path = Path(str(reference.get("path") or ""))
    if reference.get("kind") == "project-relative":
        root = project_root.resolve()
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"negotiation seed reference escapes the project root: {path}") from error
        return resolved
    if reference.get("kind") == "absolute":
        return path.resolve()
    raise RuntimeError(f"unsupported negotiation seed source reference: {reference.get('kind')!r}")


def _load_seed_document(path: Path, project_root: Path) -> tuple[dict[str, Any], Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"negotiation advisor seed is not a JSON object: {path}")
    if value.get("kind") != SEED_REFERENCE_KIND:
        return value, path.resolve()
    source_reference = value.get("source")
    if not isinstance(source_reference, dict):
        raise RuntimeError(f"installed negotiation seed has no source reference: {path}")
    source = _resolve_seed_source(source_reference, project_root)
    if not source.is_file() or _sha_file(source) != value.get("source_file_sha256"):
        raise RuntimeError(f"installed negotiation seed source failed SHA-256 verification: {source}")
    seed = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(seed, dict) or seed.get("kind") != SEED_KIND:
        raise RuntimeError(f"installed negotiation seed source is unsupported: {source}")
    if seed.get("seed_sha256") != value.get("seed_sha256"):
        raise RuntimeError(f"installed negotiation seed logical identity changed: {source}")
    return seed, source


def install_seed(source: Path, destination: Path, *, project_root: Path) -> None:
    """Install a small immutable reference to one canonical Negotiation seed."""
    seed, seed_source = _load_seed_document(source.resolve(), project_root)
    if seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source):
        raise RuntimeError(f"invalid negotiation advisor seed: {seed_source}")
    reference = {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_REFERENCE_KIND,
        "source": _seed_source_reference(seed_source, project_root),
        "source_file_sha256": _sha_file(seed_source),
        "seed_sha256": seed["seed_sha256"],
        "model_version": seed.get("model_version"),
        "corpus_sha256": seed.get("corpus_sha256"),
    }
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != reference:
            raise RuntimeError(f"negotiation advisor seed reference differs on resume: {destination}")
        return
    _atomic_json(destination, reference)


def _implementation_paths(project_root: Path) -> dict[str, Path]:
    return {
        "live_advisor_module": project_root / "src" / "nommd_arena" / "glee_negotiation_live_v2.py",
        "twin_module": project_root / "src" / "nommd_arena" / "glee_negotiation_twin_v2.py",
        "exact_surplus_module": project_root / "src" / "nommd_arena" / "glee_negotiation_twin_v2_1.py",
        "worker_module": project_root / "src" / "nommd_arena" / "glee_worker.py",
        "supervisor_module": project_root / "src" / "nommd_arena" / "glee_parallel.py",
        "transport_client_module": project_root / "src" / "nommd_arena" / "glee_transport.py",
        "dossier_broker_module": project_root / "src" / "nommd_arena" / "glee_dossier.py",
        "activity_scheduler_module": project_root / "src" / "nommd_arena" / "glee_activity_scheduler.py",
        "negotiation_prompt": project_root / "prompts" / "glee_nommd_negotiation.md",
        "meta_controller_module": project_root / "src" / "nommd_arena" / "glee_meta_controller_v2.py",
        "meta_prompt_transport_module": project_root / "src" / "nommd_arena" / "model_runner.py",
        "meta_common_prompt": project_root / "prompts" / "glee_meta_controller_common.md",
        "meta_planner_stage_prompt": project_root / "prompts" / "glee_meta_controller_planner.md",
        "meta_selector_stage_prompt": project_root / "prompts" / "glee_meta_controller_selector.md",
        "meta_planner_prompt": project_root / "prompts" / "glee_meta_controller_planner_negotiation.md",
        "meta_selector_prompt": project_root / "prompts" / "glee_meta_controller_selector_negotiation.md",
        "meta_controller_protocol": project_root / "protocols" / "glee-terra-meta-controller-live-v1.md",
        "rating_surrogate_module": project_root / "src" / "nommd_arena" / "glee_negotiation_rating_v2_4.py",
        "live_protocol": project_root / "protocols" / "glee-negotiation-live-v2-14.md",
        "transport_protocol": project_root / "protocols" / "glee-transport-fault-containment-v2.md",
    }


def _seed_games(seed: Mapping[str, object], *, seed_path: Path) -> list[dict[str, Any]]:
    inline = seed.get("games")
    if isinstance(inline, list):
        if any(not isinstance(value, dict) for value in inline):
            raise RuntimeError(f"negotiation seed contains a non-object game: {seed_path}")
        return copy.deepcopy(inline)
    reference = seed.get("games_ref")
    if not isinstance(reference, dict):
        raise RuntimeError(f"negotiation seed has neither inline games nor a game-pack reference: {seed_path}")
    games = load_ordered_json_objects(root=seed_path.parent, reference=reference)
    if len(games) != int(seed.get("game_count") or -1):
        raise RuntimeError(f"negotiation seed game-pack count differs from its manifest: {seed_path}")
    return games


def _seed_digest(seed: Mapping[str, object], *, seed_path: Path) -> str:
    logical = {key: copy.deepcopy(value) for key, value in seed.items() if key not in {"seed_sha256", "games_ref"}}
    if "games_ref" in seed:
        logical["games"] = _seed_games(seed, seed_path=seed_path)
    return _sha(logical)


def reseal_negotiation_seed_implementation(*, source_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Preserve one Negotiation frontier exactly while refreshing implementation receipts."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Negotiation advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path.resolve(), project_root)
    parent_model_version = str(seed.get("model_version") or "")
    supported_parent_versions = {MODEL_VERSION, "negotiation-live-advisor-v2.9"}
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or parent_model_version not in supported_parent_versions:
        raise RuntimeError(f"unsupported parent Negotiation advisor seed: {source_path}")
    if seed.get("base_engine_version") != BASE_ENGINE_VERSION or seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source_path):
        raise RuntimeError(f"invalid parent Negotiation advisor seed: {source_path}")
    if "games_ref" in seed and output_path.parent.resolve() != seed_source_path.parent.resolve():
        raise ValueError("a referenced Negotiation corpus can be resealed only beside its source manifest")
    parent_seed_sha256 = str(seed["seed_sha256"])
    seed["generated_at"] = _now()
    seed["parent_seed_sha256"] = parent_seed_sha256
    seed["parent_model_version"] = parent_model_version
    seed["model_version"] = MODEL_VERSION
    live_config = seed.get("live_config") if isinstance(seed.get("live_config"), dict) else None
    if live_config is None:
        raise RuntimeError(f"Negotiation advisor seed has no live configuration: {source_path}")
    live_config.setdefault("buyer_minimum_accept_surplus_share", NegotiationLiveConfig().buyer_minimum_accept_surplus_share)
    seed["reseal_reason"] = "implementation-receipt-refresh-with-identical-embedded-historical-frontier"
    seed["implementation_receipts"] = {label: {"path": str(path.relative_to(project_root)), "sha256": _sha_file(path)} for label, path in _implementation_paths(project_root).items()}
    seed["seed_sha256"] = _seed_digest(seed, seed_path=output_path)
    _atomic_json(output_path, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "path": str(output_path),
        "model_version": MODEL_VERSION,
        "base_engine_version": BASE_ENGINE_VERSION,
        "parent_seed_sha256": parent_seed_sha256,
        "seed_sha256": seed["seed_sha256"],
        "corpus_sha256": seed["corpus_sha256"],
        "game_count": seed["game_count"],
        "row_count": seed["row_count"],
    }


def promote_negotiation_seed_journal(*, source_path: Path, journal_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Fold one verified Negotiation completion journal into its exact parent frontier."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Negotiation advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path.resolve(), project_root)
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or seed.get("model_version") != MODEL_VERSION:
        raise RuntimeError(f"unsupported parent Negotiation advisor seed: {source_path}")
    if seed.get("base_engine_version") != BASE_ENGINE_VERSION or seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source_path):
        raise RuntimeError(f"invalid parent Negotiation advisor seed: {source_path}")
    if "games_ref" in seed and output_path.parent.resolve() != seed_source_path.parent.resolve():
        raise ValueError("a referenced Negotiation corpus can be promoted only beside its source manifest")
    if not journal_path.is_file():
        raise FileNotFoundError(f"Negotiation completion journal does not exist: {journal_path}")

    parent_seed_sha256 = str(seed["seed_sha256"])
    embedded_games = _seed_games(seed, seed_path=seed_source_path)
    parent_game_count = len(embedded_games)
    promoted_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        expected_sequence = len(promoted_records) + 1
        if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != JOURNAL_KIND or record.get("model_version") != MODEL_VERSION or record.get("seed_sha256") != parent_seed_sha256:
            raise RuntimeError(f"invalid Negotiation completion journal record at line {line_number}")
        if record.get("journal_sequence") != expected_sequence:
            raise RuntimeError(f"non-contiguous Negotiation completion journal sequence at line {line_number}")
        embedded = record.get("embedded_game")
        if not isinstance(embedded, dict):
            raise RuntimeError(f"Negotiation completion journal record has no embedded game at line {line_number}")
        promoted_records.append(copy.deepcopy(embedded))
    if not promoted_records:
        raise RuntimeError(f"Negotiation completion journal has no records: {journal_path}")

    embedded_games.extend(promoted_records)
    games: list[NegotiationGameEvidence] = []
    completed_hashes: dict[str, str] = {}
    named_ids: set[str] = set()
    source_counts = {"named": 0, "hidden": 0}
    for index, embedded in enumerate(embedded_games, start=1):
        job = embedded.get("job")
        if not isinstance(job, dict) or embedded.get("job_object_sha256") != _sha(job) or embedded.get("job_sha256") != _sha(job):
            raise RuntimeError(f"invalid embedded Negotiation game at promoted position {index}")
        final_game = job.get("final_game")
        final_game_sha256 = str(job.get("final_game_sha256") or "")
        if not isinstance(final_game, dict) or final_game_sha256 != _sha(final_game):
            raise RuntimeError(f"invalid terminal Negotiation game at promoted position {index}")
        game = extract_negotiation_game(job, job_path=Path(f"promoted/{job.get('job_id', index)}.json"), job_sha256=str(embedded["job_sha256"]))
        previous = completed_hashes.get(game.game_id)
        if previous is not None:
            if previous != game.final_game_sha256:
                raise RuntimeError(f"conflicting completed Negotiation game in promoted frontier: {game.game_id}")
            raise RuntimeError(f"duplicate completed Negotiation game in promoted frontier: {game.game_id}")
        completed_hashes[game.game_id] = game.final_game_sha256
        games.append(game)
        opponent = job.get("opponent") if isinstance(job.get("opponent"), dict) else {}
        opponent_id = str(opponent.get("id") or game.opponent_id)
        opponent_name = normalize_opponent_name(opponent.get("name"))
        identity_scope = str(job.get("opponent_identity_scope") or "")
        named = identity_scope == "named" or (not identity_scope and bool(opponent_name) and opponent_name != "hidden opponent" and not opponent_id.startswith("hidden-"))
        if named:
            named_ids.add(game.opponent_id)
            source_counts["named"] += 1
        else:
            source_counts["hidden"] += 1

    validation_config = NegotiationValidationConfig(**dict(seed["validation_config"]))
    game_counts = Counter(game.opponent_id for game in games)
    paper_targets = [
        {"id": opponent_id, "name": next(game.opponent_name for game in games if game.opponent_id == opponent_id), "seed_game_count": count}
        for opponent_id, count in sorted(game_counts.items())
        if opponent_id in named_ids and count >= validation_config.min_games
    ]
    promoted_seed: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "model_version": MODEL_VERSION,
        "base_engine_version": BASE_ENGINE_VERSION,
        "generated_at": _now(),
        "parent_seed_sha256": parent_seed_sha256,
        "parent_model_version": MODEL_VERSION,
        "frontier": "Only games embedded in this seed initialize the live epoch; later authenticated terminal games enter through the append-only run journal.",
        "source": {
            "kind": "parent-seed-plus-completion-journal",
            "parent_seed_path": str(seed_source_path),
            "parent_seed_sha256": parent_seed_sha256,
            "parent_game_count": parent_game_count,
            "journal_path": str(journal_path.resolve()),
            "journal_sha256": _sha_file(journal_path),
            "journal_game_count": len(promoted_records),
            "counts": source_counts,
        },
        "promotion_reason": "completed-epoch-journal-folded-for-an-operational-body-cutover",
        "validation_config": copy.deepcopy(seed["validation_config"]),
        "model_config": copy.deepcopy(seed["model_config"]),
        "live_config": copy.deepcopy(seed["live_config"]),
        "calibration_receipt": copy.deepcopy(seed.get("calibration_receipt")),
        "rating_surrogate": copy.deepcopy(seed.get("rating_surrogate")),
        "paper_targets": paper_targets,
        "game_count": len(games),
        "row_count": sum(len(game.rows) for game in games),
        "rejected": copy.deepcopy(seed.get("rejected") or []),
        "corpus_sha256": _sha([{"game_id": game.game_id, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256} for game in games]),
        "implementation_receipts": {label: {"path": str(path.relative_to(project_root)), "sha256": _sha_file(path)} for label, path in _implementation_paths(project_root).items()},
    }
    games_reference, _unique_games = seal_json_objects(root=output_path.parent, directory=output_path.parent / "corpora", prefix="negotiation-games", objects=embedded_games)
    promoted_seed["games_ref"] = games_reference
    promoted_seed["seed_sha256"] = _seed_digest(promoted_seed, seed_path=output_path)
    _atomic_json(output_path, promoted_seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "path": str(output_path),
        "model_version": MODEL_VERSION,
        "base_engine_version": BASE_ENGINE_VERSION,
        "parent_seed_sha256": parent_seed_sha256,
        "seed_sha256": promoted_seed["seed_sha256"],
        "corpus_sha256": promoted_seed["corpus_sha256"],
        "parent_game_count": parent_game_count,
        "journal_game_count": len(promoted_records),
        "game_count": len(games),
        "row_count": promoted_seed["row_count"],
        "paper_target_count": len(paper_targets),
        "source_counts": source_counts,
    }


class NegotiationLiveSeed:
    """Seal every authenticated archived Negotiation game at one causal frontier."""

    def __init__(
        self,
        *,
        output_path: Path,
        project_root: Path,
        game_archive_root: Path,
        rating_history_path: Path,
        validation_config: NegotiationValidationConfig | None = None,
        model_config: NegotiationModelConfigV21 | None = None,
        live_config: NegotiationLiveConfig | None = None,
        calibration_path: Path | None = None,
    ) -> None:
        self.output_path = output_path
        self.project_root = project_root
        self.game_archive_root = game_archive_root
        self.rating_history_path = rating_history_path
        self.validation_config = validation_config or NegotiationValidationConfig()
        self.model_config = model_config or NegotiationModelConfigV21()
        self.calibration_path = calibration_path
        self.calibration: dict[str, object] | None = None
        if calibration_path is not None:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not isinstance(calibration, dict) or calibration.get("kind") != "glee-negotiation-v2.4-calibration":
                raise ValueError(f"unsupported Negotiation v2.4 calibration: {calibration_path}")
            calibrated_values = {key: tuple(value) if isinstance(value, list) else value for key, value in dict(calibration["selected_live_config"]).items()}
            calibrated_config = NegotiationLiveConfig(**calibrated_values)
            if live_config is not None and asdict(live_config) != asdict(calibrated_config):
                raise ValueError("explicit live_config differs from the supplied Negotiation calibration")
            self.live_config = calibrated_config
            self.calibration = calibration
        else:
            self.live_config = live_config or NegotiationLiveConfig()
        self.validation_config.validate()
        self.model_config.validate()
        self.live_config.validate()

    def archive_games(self) -> tuple[list[NegotiationGameEvidence], list[dict[str, object]], list[dict[str, str]], set[str], dict[str, int]]:
        rating_history = json.loads(self.rating_history_path.read_text(encoding="utf-8"))
        deltas = rating_history.get("game_deltas")
        if not isinstance(deltas, dict):
            raise ValueError(f"rating history has no game_deltas map: {self.rating_history_path}")
        accepted: dict[str, tuple[str, Path, dict[str, Any], bool]] = {}
        rejected: list[dict[str, str]] = []
        for path in sorted(self.game_archive_root.glob("*/games/negotiation-*.json")):
            try:
                final_game = json.loads(path.read_text(encoding="utf-8"))
                if final_game.get("game_family") != "negotiation":
                    continue
                game_id = str(final_game.get("game_id") or "")
                delta = deltas.get(game_id)
                if not game_id or not isinstance(delta, dict) or not str(delta.get("completed_at") or ""):
                    raise ValueError("game lacks an authenticated completion timestamp")
                result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
                if str(final_game.get("status") or "").casefold() in {"timeout", "cancelled", "abandoned"} or str(result.get("outcome") or "").casefold() in {"timeout", "cancelled", "abandoned"}:
                    rejected.append({"path": str(path), "reason": "censored_terminal_state"})
                    continue
                final_sha = _sha(final_game)
                previous = accepted.get(game_id)
                if previous is not None:
                    if _sha(previous[2]) != final_sha:
                        raise RuntimeError(f"conflicting archived negotiation games for {game_id}: {previous[1]} and {path}")
                    rejected.append({"path": str(path), "reason": f"duplicate_game:{previous[1]}"})
                    continue
                opponent = final_game.get("opponent") if isinstance(final_game.get("opponent"), dict) else {}
                name = normalize_opponent_name(opponent.get("name"))
                named = bool(name and str(opponent.get("type") or "agent") != "hidden")
                accepted[game_id] = (str(delta["completed_at"]), path, final_game, named)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
        games: list[NegotiationGameEvidence] = []
        embedded_games: list[dict[str, object]] = []
        named_ids: set[str] = set()
        counts = {"named": 0, "hidden": 0}
        ordered = sorted(accepted.items(), key=lambda item: (item[1][0], item[0]))
        for completion_order, (game_id, (completed_at, path, final_game, named)) in enumerate(ordered, start=1):
            opponent = final_game.get("opponent") if isinstance(final_game.get("opponent"), dict) else {}
            normalized_name = normalize_opponent_name(opponent.get("name"))
            opponent_name = normalized_name if named and normalized_name else "hidden opponent"
            opponent_id = named_opponent_id(opponent_name) if named else f"hidden-history-{hashlib.sha256(game_id.encode('utf-8')).hexdigest()[:20]}"
            if named:
                named_ids.add(opponent_id)
                counts["named"] += 1
            else:
                counts["hidden"] += 1
            final_sha = _sha(final_game)
            job = {
                "schema_version": SCHEMA_VERSION,
                "kind": "negotiation-archive-seed-job",
                "job_id": _sha({"game_id": game_id, "final_game_sha256": final_sha, "completed_at": completed_at}),
                "opponent": {"id": opponent_id, "name": opponent_name},
                "opponent_identity_scope": "named" if named else "hidden-game-only",
                "game_id": game_id,
                "game_family": "negotiation",
                "completed_at": completed_at,
                "completion_order": completion_order,
                "rating_delta": float(deltas[game_id]["rating_delta"]) if isinstance(deltas[game_id].get("rating_delta"), (int, float)) and not isinstance(deltas[game_id].get("rating_delta"), bool) else None,
                "rating_record_sha256": str(deltas[game_id].get("record_sha256") or ""),
                "final_game_sha256": final_sha,
                "final_game": final_game,
                "source_game_path": str(path.resolve()),
            }
            job_sha = _sha(job)
            games.append(extract_negotiation_game(job, job_path=path, job_sha256=job_sha))
            embedded_games.append({"job": job, "job_sha256": job_sha, "job_object_sha256": job_sha})
        return games, embedded_games, rejected, named_ids, counts

    def run(self) -> dict[str, object]:
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite negotiation advisor seed: {self.output_path}")
        games, embedded_games, rejected, named_ids, source_counts = self.archive_games()
        if not games:
            raise RuntimeError("negotiation advisor seed has no accepted games")
        game_counts = Counter(game.opponent_id for game in games)
        paper_targets = [
            {"id": opponent_id, "name": next(game.opponent_name for game in games if game.opponent_id == opponent_id), "seed_game_count": count}
            for opponent_id, count in sorted(game_counts.items())
            if opponent_id in named_ids and count >= self.validation_config.min_games
        ]
        implementation_receipts = {label: {"path": str(path.relative_to(self.project_root)), "sha256": _sha_file(path)} for label, path in _implementation_paths(self.project_root).items()}
        seed: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": SEED_KIND,
            "model_version": MODEL_VERSION,
            "base_engine_version": BASE_ENGINE_VERSION,
            "generated_at": _now(),
            "frontier": "Only games embedded in this seed initialize the live epoch; later authenticated terminal games enter through the append-only run journal.",
            "source": {"kind": "authenticated-run-game-archive", "game_archive_root": str(self.game_archive_root.resolve()), "rating_history_path": str(self.rating_history_path.resolve()), "counts": source_counts},
            "validation_config": asdict(self.validation_config),
            "model_config": asdict(self.model_config),
            "live_config": asdict(self.live_config),
            "calibration_receipt": {"calibration_version": self.calibration.get("calibration_version"), "calibration_sha256": _sha(self.calibration), "source_file_sha256": _sha_file(self.calibration_path)} if self.calibration is not None and self.calibration_path is not None else None,
            "rating_surrogate": copy.deepcopy(self.calibration.get("rating_surrogate")) if self.calibration is not None else None,
            "paper_targets": paper_targets,
            "game_count": len(games),
            "row_count": sum(len(game.rows) for game in games),
            "rejected": rejected,
            "corpus_sha256": _sha([{"game_id": game.game_id, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256} for game in games]),
            "implementation_receipts": implementation_receipts,
        }
        games_reference, _unique_games = seal_json_objects(root=self.output_path.parent, directory=self.output_path.parent / "corpora", prefix="negotiation-games", objects=embedded_games)
        seed["games_ref"] = games_reference
        seed["seed_sha256"] = _seed_digest(seed, seed_path=self.output_path)
        _atomic_json(self.output_path, seed)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": SEED_KIND,
            "path": str(self.output_path),
            "model_version": MODEL_VERSION,
            "base_engine_version": BASE_ENGINE_VERSION,
            "seed_sha256": seed["seed_sha256"],
            "game_count": len(games),
            "row_count": seed["row_count"],
            "paper_target_count": len(paper_targets),
            "paper_targets": paper_targets,
            "source_counts": source_counts,
        }


def _weights(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _support_adjusted_weights(names: Sequence[str], values: Sequence[float], *, target_rows: int, full_support_rows: int) -> tuple[dict[str, float], float]:
    weights = _weights(names, values)
    support_factor = min(1.0, max(0.0, target_rows / full_support_rows))
    original_target = weights["target_only"]
    adjusted_target = original_target * support_factor
    released = original_target - adjusted_target
    other_names = [name for name in names if name != "target_only"]
    other_total = sum(weights[name] for name in other_names)
    for name in other_names:
        weights[name] += released * weights[name] / other_total
    weights["target_only"] = adjusted_target
    return weights, support_factor


def _updated_weights(prior: Mapping[str, float], losses: Mapping[str, float], config: NegotiationLiveConfig) -> dict[str, float]:
    names = tuple(prior)
    if not names or any(name not in losses or not math.isfinite(float(losses[name])) for name in names):
        raise ValueError("expert update requires one finite loss per expert")
    clipped = {name: min(config.expert_loss_ceiling, max(config.expert_loss_floor, float(losses[name]))) for name in names}
    minimum = min(clipped.values())
    raw = {name: max(float(prior[name]), 1e-300) * math.exp(-config.within_game_learning_rate * (clipped[name] - minimum)) for name in names}
    total = sum(raw.values())
    uniform = 1.0 / len(names)
    return {name: (1 - config.within_game_uniform_mix) * raw[name] / total + config.within_game_uniform_mix * uniform for name in names}


def _blend_mixtures(distributions: Mapping[str, GaussianMixture], weights: Mapping[str, float]) -> GaussianMixture:
    components = [GaussianComponent(component.mean, component.sigma, float(weights[name]) * component.weight, name) for name, distribution in distributions.items() for component in distribution.components]
    return GaussianMixture(components)


def _fast_quantile(distribution: GaussianMixture, probability: float, *, iterations: int = 20) -> float:
    """Resolve a live-only mixture quantile to bounded numerical precision without changing its density."""
    if not 0 <= probability <= 1 or iterations < 1:
        raise ValueError("quantile request is invalid")
    low = min(component.mean - 8 * component.sigma for component in distribution.components)
    high = max(component.mean + 8 * component.sigma for component in distribution.components)
    for _iteration in range(iterations):
        middle = (low + high) / 2
        if distribution.cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _fast_contract_mixture(distribution: GaussianMixture, contraction: float) -> GaussianMixture:
    center = _fast_quantile(distribution, 0.5)
    return GaussianMixture(tuple(GaussianComponent(center + contraction * (component.mean - center), contraction * component.sigma, component.weight, "live-v2.1-affine-contraction") for component in distribution.components))


def _logistic(value: float) -> float:
    if value >= 0:
        decay = math.exp(-value)
        return 1 / (1 + decay)
    growth = math.exp(value)
    return growth / (1 + growth)


def _latent_threshold_response(row: NegotiationDecisionRow, prefix: Sequence[NegotiationDecisionRow], config: NegotiationLiveConfig) -> dict[str, object]:
    """Estimate a moving acceptance threshold from rejected bids and the opponent's closing asks."""
    if row.offered_demand is None:
        return {"status": "unavailable", "reason": "candidate-has-no-demand-coordinate", "blend_weight": 0.0}
    proposals = [float(previous.proposal_demand) for previous in prefix if previous.action_type == "proposal" and previous.proposal_demand is not None]
    if not proposals:
        return {"status": "unavailable", "reason": "no-visible-opponent-proposal", "blend_weight": 0.0}
    concessions = [previous - current for previous, current in zip(proposals, proposals[1:])]
    latest_concession = concessions[-1] if concessions else None
    if not row.context.complete_information and (latest_concession is None or latest_concession <= 0):
        return {
            "status": "unavailable",
            "reason": "incomplete-information-overlay-requires-positive-latest-opponent-concession",
            "blend_weight": 0.0,
            "latest_opponent_concession_demand": latest_concession,
            "opponent_proposal_count": len(proposals),
        }
    recent_positive = [value for value in concessions[-config.continuation_concession_window :] if value > 0]
    concession_slope = _median(recent_positive)
    scale = max(config.latent_threshold_min_scale, config.latent_threshold_slope_scale * concession_slope)
    rejected_demands = [float(previous.offered_demand) for previous in prefix if previous.action_type == "response" and previous.accepted is False and previous.offered_demand is not None]
    rejected_boundary = max(rejected_demands) if rejected_demands else None
    projected_ask = proposals[-1] - config.latent_threshold_projected_concession * concession_slope
    threshold = projected_ask
    if rejected_boundary is not None:
        threshold = max(threshold, rejected_boundary + config.latent_threshold_rejection_margin_scale * scale)
    candidate = float(row.offered_demand)
    probability = _logistic((candidate - threshold) / scale)
    configured_weight = config.latent_threshold_complete_weight if row.context.complete_information else config.latent_threshold_incomplete_weight
    support_factor = min(1.0, len(proposals) / 2)
    blend_weight = configured_weight * support_factor
    return {
        "status": "available",
        "method": "interval-censored-moving-threshold",
        "probability": min(1 - 1e-6, max(1e-6, probability)),
        "blend_weight": blend_weight,
        "candidate_demand": candidate,
        "latest_opponent_ask_demand": proposals[-1],
        "gap_to_latest_ask": max(0.0, proposals[-1] - candidate),
        "rejected_offer_lower_boundary": rejected_boundary,
        "projected_threshold_demand": threshold,
        "recent_positive_concession_count": len(recent_positive),
        "latest_opponent_concession_demand": latest_concession,
        "median_recent_concession_demand": concession_slope,
        "scale": scale,
        "opponent_proposal_count": len(proposals),
        "opponent_rejection_count": len(rejected_demands),
        "boundary": "Visible-prefix behavioral overlay; opponent asks are soft upper evidence, rejections are lower-threshold evidence, and no hidden value is inferred.",
    }


@dataclass(frozen=True)
class _ForecastBundle:
    """One historical frontier plus causally updated visible-game evidence."""

    opponent_id: str
    opponent_name: str
    global_game_index: int
    target_game_index: int
    population_prior: tuple[PriorObservation, ...]
    target_prior: tuple[PriorObservation, ...]
    prefix: tuple[NegotiationDecisionRow, ...]
    model: NegotiationOpponentModelV21
    complete_response_weights: dict[str, float]
    incomplete_response_weights: dict[str, float]
    proposal_weights: dict[str, float]
    target_support_factor: float
    config: NegotiationLiveConfig

    def response(self, row: NegotiationDecisionRow) -> dict[str, object]:
        forecasts = self.model.response_forecast(row, population_prior=self.population_prior, target_prior=self.target_prior, prefix=self.prefix, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index)
        target_forecast = NegotiationOpponentModelV2.response_forecast(self.model, row, population_prior=(), target_prior=self.target_prior, prefix=self.prefix, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index)["adaptive_v2"]
        experts = {
            "adaptive_v2_1": float(forecasts["adaptive_v2_1"]["probability"]),
            "adaptive_v2": float(forecasts["adaptive_v2"]["probability"]),
            "target_only": float(target_forecast["probability"]),
        }
        weights = self.complete_response_weights if row.context.complete_information else self.incomplete_response_weights
        base_probability = sum(weights[name] * experts[name] for name in RESPONSE_EXPERTS)
        threshold = _latent_threshold_response(row, self.prefix, self.config)
        blend_weight = float(threshold.get("blend_weight") or 0.0)
        threshold_probability = float(threshold["probability"]) if threshold.get("status") == "available" else base_probability
        probability = (1 - blend_weight) * base_probability + blend_weight * threshold_probability
        return {
            "probability": min(1 - 1e-6, max(1e-6, probability)),
            "base_probability": min(1 - 1e-6, max(1e-6, base_probability)),
            "experts": experts,
            "weights": dict(weights),
            "latent_threshold": copy.deepcopy(threshold),
            "v2_1": copy.deepcopy(forecasts["adaptive_v2_1"]),
        }

    def proposal(self, row: NegotiationDecisionRow) -> dict[str, object]:
        forecasts = NegotiationOpponentModelV2.proposal_forecast(self.model, row, population_prior=self.population_prior, target_prior=self.target_prior, prefix=self.prefix, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index)
        target_forecast = NegotiationOpponentModelV2.proposal_forecast(self.model, row, population_prior=(), target_prior=self.target_prior, prefix=self.prefix, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index)["adaptive_v2"]
        experts = {"adaptive_v2_1": _fast_contract_mixture(forecasts["adaptive_v2"], self.model.config.proposal_uncertainty_contraction), "adaptive_v2": forecasts["adaptive_v2"], "target_only": target_forecast}
        blended = _blend_mixtures(experts, self.proposal_weights)
        distribution = _fast_contract_mixture(blended, self.config.proposal_final_contraction) if self.config.proposal_final_contraction < 1 else blended
        return {"distribution": distribution, "pre_final_contraction_distribution": blended, "final_contraction": self.config.proposal_final_contraction, "experts": experts, "weights": dict(self.proposal_weights)}

    def incorporate(self, row: NegotiationDecisionRow, *, precomputed_forecast: Mapping[str, object] | None = None) -> _ForecastBundle:
        complete_weights = dict(self.complete_response_weights)
        incomplete_weights = dict(self.incomplete_response_weights)
        proposal_weights = dict(self.proposal_weights)
        if row.action_type == "response" and row.accepted is not None:
            forecast = dict(precomputed_forecast) if precomputed_forecast is not None else self.response(row)
            losses = {name: -math.log(max(1e-12, probability if row.accepted else 1 - probability)) for name, probability in dict(forecast["experts"]).items()}
            if row.context.complete_information:
                complete_weights = _updated_weights(complete_weights, losses, self.config)
            else:
                incomplete_weights = _updated_weights(incomplete_weights, losses, self.config)
        elif row.action_type == "proposal" and row.proposal_demand is not None:
            forecast = self.proposal(row)
            losses = {name: distribution.nll(float(row.proposal_demand)) for name, distribution in dict(forecast["experts"]).items()}
            proposal_weights = _updated_weights(proposal_weights, losses, self.config)
        return replace(self, prefix=(*self.prefix, row), complete_response_weights=complete_weights, incomplete_response_weights=incomplete_weights, proposal_weights=proposal_weights)


@dataclass(frozen=True)
class _WithinGameEvidence:
    """Opponent actions recoverable from the visible Negotiation transcript."""

    rows: tuple[NegotiationDecisionRow, ...]
    history: tuple[dict[str, Any], ...]
    transcript_sha256: str
    current_offer_included: bool

    def receipt(self) -> dict[str, object]:
        return {
            "method": "sequential-visible-prefix-expert-update",
            "source": "authenticated_history_plus_current_opponent_offer",
            "transcript_sha256": self.transcript_sha256,
            "turn_local_only": True,
            "restart_reconstruction": "deterministic-from-visible-transcript",
            "terminal_journal_mutated": False,
            "opponent_response_count": sum(row.action_type == "response" for row in self.rows),
            "opponent_proposal_count": sum(row.action_type == "proposal" for row in self.rows),
            "current_offer_included": self.current_offer_included,
        }


def _row_fingerprint(row: NegotiationDecisionRow) -> str:
    value = asdict(row)
    for key in ("job_id", "job_path", "job_sha256"):
        value.pop(key, None)
    return _sha(value)


@dataclass(frozen=True)
class _GamePrefixCache:
    """One causally reconstructed game prefix at its frozen historical revision."""

    state_revision: int
    opponent_id: str
    row_fingerprints: tuple[str, ...]
    bundle: _ForecastBundle


@dataclass(frozen=True)
class _GameHistoricalFrontier:
    """One persistent historical corpus boundary fixed before a game's first inference."""

    game_id: str
    opponent_id: str
    opponent_name: str
    state_revision: int
    history_prefix_sha256: str


def _identity(game: Mapping[str, object]) -> tuple[str, str, bool]:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    name = normalize_opponent_name(opponent.get("name"))
    if name and str(opponent.get("type") or "agent") != "hidden":
        return named_opponent_id(name), name, True
    game_id = str(game.get("game_id") or "unknown")
    return f"hidden-{hashlib.sha256(game_id.encode('utf-8')).hexdigest()[:20]}", "hidden opponent", False


def _other_player(player: str) -> str:
    try:
        return {"player_1": "player_2", "player_2": "player_1"}[player]
    except KeyError as error:
        raise ValueError(f"unsupported player identity: {player}") from error


def _live_evidence(game: dict[str, Any], *, opponent_id: str, opponent_name: str) -> _WithinGameEvidence:
    state = game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("live negotiation game has no state")
    raw_history = state.get("history") or []
    if not isinstance(raw_history, list):
        raise ValueError("live negotiation history is not a list")
    history = [copy.deepcopy(entry) for entry in raw_history]
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    opponent_player = _other_player(our_player)
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    our_value = float(state.get(f"{our_player}_value"))
    opponent_value = float(state[f"{opponent_player}_value"]) if state.get("complete_information") is True and state.get(f"{opponent_player}_value") is not None else None
    rows: list[NegotiationDecisionRow] = []
    prior: list[dict[str, Any]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or not isinstance(entry.get("offer"), dict):
            raise ValueError("live negotiation history contains a malformed entry")
        offer = entry["offer"]
        from_player = str(offer.get("from_player") or "")
        decided_by = str(entry.get("decided_by") or "")
        if from_player not in {our_player, opponent_player} or decided_by != _other_player(from_player):
            raise ValueError("live negotiation history has inconsistent actors")
        round_number = int(entry.get("round") or offer.get("round") or index + 1)
        context = _history_context(game_id=str(game.get("game_id") or "live"), opponent_id=opponent_id, opponent_name=opponent_name, completed_at="", completion_order=0, state=state, our_player=our_player, opponent_player=opponent_player, round_number=round_number, prior=prior, current_offer=offer if decided_by == opponent_player else None)
        price = float(offer["price"])
        demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
        share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
        decision = str(entry.get("decision") or "")
        if from_player == opponent_player:
            from .glee_negotiation_twin_v2 import classify_negotiation_message

            message = str(offer.get("message") or "")
            rows.append(NegotiationDecisionRow(context=context, action_type="proposal", proposal_demand=demand, proposal_price_ratio=price / our_value, proposal_opponent_surplus_share=share, message=message, message_act=classify_negotiation_message(message, messages_allowed=context.messages_allowed), job_id="within-game-history-proposal", job_path="live-transcript", job_sha256=""))
        elif decision in {"AcceptOffer", "RejectOffer", "WalkAway"}:
            response_time = entry.get("response_time_ms")
            rows.append(NegotiationDecisionRow(context=context, action_type="response", offered_demand=demand, offered_opponent_surplus_share=share, accepted=decision == "AcceptOffer", decision=decision, response_time_ms=float(response_time) if isinstance(response_time, (int, float)) and not isinstance(response_time, bool) else None, job_id="within-game-history-response", job_path="live-transcript", job_sha256=""))
        prior.append(entry)
    current_offer_included = False
    current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
    if str(game.get("phase") or state.get("phase") or "") == "decision" and current_offer is not None and str(current_offer.get("from_player") or "") == opponent_player:
        current_key = (str(current_offer.get("from_player") or ""), int(current_offer.get("round") or state.get("round") or 0), float(current_offer.get("price") or 0.0))
        represented = any(isinstance(entry, dict) and isinstance(entry.get("offer"), dict) and (str(entry["offer"].get("from_player") or ""), int(entry.get("round") or entry["offer"].get("round") or 0), float(entry["offer"].get("price") or 0.0)) == current_key for entry in history)
        if not represented:
            from .glee_negotiation_twin_v2 import classify_negotiation_message

            round_number = current_key[1]
            context = _history_context(game_id=str(game.get("game_id") or "live"), opponent_id=opponent_id, opponent_name=opponent_name, completed_at="", completion_order=0, state=state, our_player=our_player, opponent_player=opponent_player, round_number=round_number, prior=prior, current_offer=None)
            price = current_key[2]
            demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
            share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
            message = str(current_offer.get("message") or "")
            rows.append(NegotiationDecisionRow(context=context, action_type="proposal", proposal_demand=demand, proposal_price_ratio=price / our_value, proposal_opponent_surplus_share=share, message=message, message_act=classify_negotiation_message(message, messages_allowed=context.messages_allowed), job_id="within-game-current-proposal", job_path="live-transcript", job_sha256=""))
            current_offer_included = True
    transcript = {"history": history, "current_opponent_offer": current_offer if current_offer_included else None}
    return _WithinGameEvidence(rows=tuple(rows), history=tuple(history), transcript_sha256=_sha(transcript), current_offer_included=current_offer_included)


def _response_row(game: dict[str, Any], evidence: _WithinGameEvidence, *, opponent_id: str, opponent_name: str, price: float, message: str = "") -> NegotiationDecisionRow:
    state = game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("live negotiation game has no state")
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    opponent_player = _other_player(our_player)
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    our_role = str(state.get(f"{our_player}_role") or "")
    our_value = float(state.get(f"{our_player}_value"))
    opponent_value = float(state[f"{opponent_player}_value"]) if state.get("complete_information") is True and state.get(f"{opponent_player}_value") is not None else None
    phase = str(game.get("phase") or state.get("phase") or "")
    round_number = int(state.get("round") or 1)
    prior = [copy.deepcopy(entry) for entry in evidence.history]
    if phase == "decision":
        current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
        if current_offer is None:
            raise ValueError("decision turn has no opponent offer")
        prior.append({"round": int(current_offer.get("round") or round_number), "offer": copy.deepcopy(current_offer), "decided_by": our_player, "decision": "RejectOffer", "counteroffer": price})
        round_number += 1
    offer = {"round": round_number, "from_player": our_player, "price": price, "message": message}
    context = _history_context(game_id=str(game.get("game_id") or "live"), opponent_id=opponent_id, opponent_name=opponent_name, completed_at="", completion_order=0, state=state, our_player=our_player, opponent_player=opponent_player, round_number=round_number, prior=prior, current_offer=offer)
    demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
    share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
    return NegotiationDecisionRow(context=context, action_type="response", offered_demand=demand, offered_opponent_surplus_share=share, accepted=False, decision="RejectOffer", job_id="hypothetical-opponent-response", job_path="live-forecast", job_sha256="")


def _proposal_row_after_rejection(response: NegotiationDecisionRow) -> NegotiationDecisionRow:
    context = replace(response.context, round_number=response.context.round_number + 1, observed_opponent_actions=response.context.observed_opponent_actions + 1, previous_our_demand=response.offered_demand, previous_opponent_response="RejectOffer", current_offer_demand=None, current_offer_opponent_surplus_share=None, current_offer_message_act="none")
    return NegotiationDecisionRow(context=context, action_type="proposal", proposal_demand=0.0, proposal_price_ratio=1.0, job_id="conditional-opponent-counterproposal", job_path="live-forecast", job_sha256="")


def _own_surplus(price: float, context: NegotiationContext) -> float:
    return context.our_value - price if context.our_role == "buyer" else price - context.our_value


def _later_round_available(context: NegotiationContext) -> bool:
    return not (context.horizon_known and context.max_rounds is not None and context.round_number >= context.max_rounds)


def _counteroffer_available(game: dict[str, Any]) -> bool:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    phase = str(game.get("phase") or state.get("phase") or "")
    if phase != "decision":
        return False
    valid_actions = game.get("valid_actions") if isinstance(game.get("valid_actions"), dict) else state.get("valid_actions")
    if isinstance(valid_actions, dict):
        fields = valid_actions.get("fields")
        return isinstance(fields, dict) and "product_price" in fields
    round_number = int(state.get("round") or 1)
    max_rounds = int(state["max_rounds"]) if state.get("max_rounds") is not None else None
    horizon_known = bool(state.get("horizon_known"))
    return not (horizon_known and max_rounds is not None and round_number >= max_rounds)


def _message_available(game: Mapping[str, object]) -> bool:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    if state.get("messages_allowed") is False:
        return False
    valid_actions = game.get("valid_actions") if isinstance(game.get("valid_actions"), dict) else {}
    fields = valid_actions.get("fields") if isinstance(valid_actions, dict) else None
    return not isinstance(fields, dict) or "message" in fields


def _reference_context(game: dict[str, Any], evidence: _WithinGameEvidence, *, opponent_id: str, opponent_name: str) -> NegotiationContext:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    phase = str(game.get("phase") or state.get("phase") or "")
    if phase == "decision":
        for row in reversed(evidence.rows):
            if row.action_type == "proposal":
                return row.context
    current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
    reference_price = float(current_offer["price"]) if current_offer is not None and current_offer.get("price") is not None else float(state.get(f"{game.get('your_player')}_value"))
    return _response_row(game, evidence, opponent_id=opponent_id, opponent_name=opponent_name, price=reference_price).context


def _candidate_prices(context: NegotiationContext, evidence: _WithinGameEvidence, config: NegotiationLiveConfig) -> tuple[float, ...]:
    prices: set[float] = set()
    if context.complete_information and context.opponent_value is not None:
        buyer_value = context.opponent_value if context.opponent_role == "buyer" else context.our_value
        seller_value = context.opponent_value if context.opponent_role == "seller" else context.our_value
        surplus = buyer_value - seller_value
        if surplus > 0:
            one_round_complete_seller = context.our_role == "seller" and context.opponent_role == "buyer" and context.horizon_known and context.max_rounds == 1 and context.round_number == 1
            for opponent_share in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                if one_round_complete_seller and opponent_share == 0.0:
                    continue
                price = buyer_value - opponent_share * surplus if context.opponent_role == "buyer" else seller_value + opponent_share * surplus
                prices.add(price)
            if one_round_complete_seller:
                prices.add(buyer_value - _positive_surplus_tick(buyer_value, config))
    else:
        prices.update(price_from_opponent_demand(demand, opponent_role=context.opponent_role, our_value=context.our_value) for demand in config.candidate_demand_grid)
    return tuple(sorted({_rounded(price) for price in prices if math.isfinite(price) and price > 0 and _own_surplus(price, context) >= -config.consistency_value_tolerance}))


def _expert_probability_receipt(forecast: Mapping[str, object]) -> dict[str, object]:
    receipt: dict[str, object] = {
        "weighted": _rounded(float(forecast["probability"])),
        "pre_threshold_weighted": _rounded(float(forecast.get("base_probability") or forecast["probability"])),
        "experts": {name: _rounded(float(value)) for name, value in dict(forecast["experts"]).items()},
        "weights": {name: _rounded(float(value)) for name, value in dict(forecast["weights"]).items()},
    }
    threshold = forecast.get("latent_threshold")
    if isinstance(threshold, Mapping):
        receipt["latent_threshold"] = {key: _rounded(float(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else copy.deepcopy(value) for key, value in threshold.items()}
    return receipt


def _prompt_response_profile(candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Expose forecast shape without turning fixed numeric prices into a model-facing action menu."""
    if not candidates:
        return {"representation": "qualitative-range-summary-without-price-knots", "status": "no-numeric-offer-available", "exact_offer_evaluations": "shadow-receipt-only"}
    ordered = sorted(candidates, key=lambda candidate: float(candidate["opponent_demand"]))
    probabilities = [float(dict(candidate["opponent_response"])["weighted"]) for candidate in ordered]
    differences = [current - previous for previous, current in zip(probabilities, probabilities[1:])]
    net_change = probabilities[-1] - probabilities[0]
    if net_change > 0.02 and sum(value < -0.01 for value in differences) <= 1:
        trend = "acceptance-rises-as-the-offer-favors-the-opponent"
    elif net_change < -0.02 and sum(value > 0.01 for value in differences) <= 1:
        trend = "acceptance-falls-as-the-offer-favors-the-opponent"
    elif max(probabilities) - min(probabilities) <= 0.03:
        trend = "acceptance-nearly-flat-across-the-feasible-range"
    else:
        trend = "acceptance-shape-is-irregular-or-uncertain"
    q50 = [float(counter["q50_demand"]) for candidate in ordered if isinstance((counter := candidate.get("conditional_rejection_counterproposal")), Mapping) and counter.get("q50_demand") is not None]
    q80 = [float(counter["adverse_opponent_demand"]) for candidate in ordered if isinstance((counter := candidate.get("conditional_rejection_counterproposal")), Mapping) and counter.get("adverse_opponent_demand") is not None]
    return {
        "representation": "qualitative-range-summary-without-price-knots",
        "status": "available",
        "acceptance_probability_range": {"minimum": _rounded(min(probabilities)), "maximum": _rounded(max(probabilities))},
        "acceptance_trend": trend,
        "conditional_rejection_counterproposal": {
            "q50_opponent_demand_range": {"minimum": _rounded(min(q50)), "maximum": _rounded(max(q50))} if q50 else None,
            "adverse_q80_opponent_demand_range": {"minimum": _rounded(min(q80)), "maximum": _rounded(max(q80))} if q80 else None,
        },
        "selection_boundary": "This is fallible behavioral evidence, not a finite offer menu or action recommendation; choose the numeric action independently.",
        "free_text_message_effects": "coarse-message-act-only",
        "exact_offer_evaluations": "shadow-receipt-only",
    }


def _visible_offer_prices(state: Mapping[str, object], player: str) -> list[float]:
    """Return one player's visible offers once each, including an uncommitted current offer."""
    history = state.get("history") if isinstance(state.get("history"), list) else []
    prices = [float(entry["offer"]["price"]) for entry in history if isinstance(entry, Mapping) and isinstance(entry.get("offer"), Mapping) and str(entry["offer"].get("from_player") or "") == player and isinstance(entry["offer"].get("price"), (int, float)) and not isinstance(entry["offer"].get("price"), bool)]
    current = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else None
    if current is not None and str(current.get("from_player") or "") == player and isinstance(current.get("price"), (int, float)) and not isinstance(current.get("price"), bool):
        current_key = (int(current.get("round") or state.get("round") or 0), float(current["price"]))
        represented = any(isinstance(entry, Mapping) and isinstance(entry.get("offer"), Mapping) and str(entry["offer"].get("from_player") or "") == player and (int(entry.get("round") or entry["offer"].get("round") or 0), float(entry["offer"].get("price") or 0.0)) == current_key for entry in history)
        if not represented:
            prices.append(current_key[1])
    return prices


def _one_round_seller_control(context: NegotiationContext, phase: str, config: NegotiationLiveConfig, *, incomplete_authority: str = "fixed-markup") -> dict[str, object]:
    """Select the frozen terminal seller price for a one-round opening."""
    base: dict[str, object] = {"status": "not-applicable", "authority": "deterministic-one-round-seller-price"}
    if phase != "offer" or context.our_role != "seller" or not context.horizon_known or context.max_rounds != 1 or context.round_number != 1:
        return base
    if context.complete_information and context.opponent_value is not None:
        tick = _positive_surplus_tick(context.opponent_value, config)
        target = context.opponent_value - tick
        if target <= context.our_value + config.consistency_value_tolerance:
            return {**base, "status": "no-mutually-positive-price", "mode": "complete-information-positive-surplus-boundary"}
        return {
            "status": "required",
            "authority": base["authority"],
            "mode": "complete-information-positive-surplus-boundary",
            "target_price": _rounded(target),
            "buyer_positive_surplus_tick": _rounded(tick),
            "rule": "This is the only seller offer. Quote the visible buyer reservation value minus one positive-surplus tick.",
        }
    target = context.our_value * (1.0 + config.one_round_incomplete_seller_markup)
    if incomplete_authority == "meta15-candidate-selection":
        return {
            "status": "advisory",
            "authority": "meta15-candidate-selection-with-fixed-markup-fallback",
            "mode": "incomplete-information-calibrated-markup",
            "target_price": _rounded(target),
            "markup_over_seller_value": config.one_round_incomplete_seller_markup,
            "rule": "This is the only seller offer and the buyer value is hidden. Include the calibrated markup as a reference candidate and deterministic fallback, then compare materially different feasible prices through the 1.5-round selector.",
        }
    return {
        "status": "required",
        "authority": base["authority"],
        "mode": "incomplete-information-calibrated-markup",
        "target_price": _rounded(target),
        "markup_over_seller_value": config.one_round_incomplete_seller_markup,
        "rule": "This is the only seller offer and the buyer value is hidden. Use the frozen pre-canary markup calibrated on one-round seller responses.",
    }


def _complete_information_offer_control(context: NegotiationContext, config: NegotiationLiveConfig) -> dict[str, object]:
    """Return the cumulative own-surplus boundary for every complete-information proposal."""
    base: dict[str, object] = {"status": "not-applicable", "authority": "complete-information-cumulative-surplus-floor"}
    if not context.complete_information or context.opponent_value is None:
        return base
    buyer_value = context.our_value if context.our_role == "buyer" else context.opponent_value
    seller_value = context.our_value if context.our_role == "seller" else context.opponent_value
    total = buyer_value - seller_value
    if total <= config.consistency_value_tolerance:
        return {**base, "status": "no-positive-joint-surplus", "total_surplus": _rounded(total)}
    own_share = config.complete_information_min_own_surplus_share
    boundary = seller_value + own_share * total if context.our_role == "seller" else buyer_value - own_share * total
    result: dict[str, object] = {
        "status": "available",
        "authority": base["authority"],
        "role": context.our_role,
        "minimum_own_surplus_share": own_share,
        "boundary_price": _rounded(boundary),
        "rule": "Do not make a binding proposal that gives us less than the configured share of visible joint surplus.",
    }
    result["minimum_next_offer" if context.our_role == "seller" else "maximum_next_offer"] = _rounded(boundary)
    return result


def _buyer_scale_aware_decision_control(game: dict[str, Any], context: NegotiationContext, phase: str, complete_information_offer_control: Mapping[str, object], config: NegotiationLiveConfig, *, mode: str) -> dict[str, object]:
    """Expose the narrow buyer counter-first boundary for visible positive-surplus decisions."""
    base: dict[str, object] = {"status": "not-enabled", "authority": "visible-scale-aware-buyer-decision", "mode": mode}
    if mode != "one-percent-counter-first":
        return base
    if phase != "decision" or context.our_role != "buyer" or not context.complete_information or context.opponent_value is None:
        return {**base, "status": "not-applicable"}
    total_surplus = context.our_value - context.opponent_value
    if total_surplus <= config.consistency_value_tolerance:
        return {**base, "status": "no-positive-joint-surplus", "total_surplus": _rounded(total_surplus)}
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else None
    current_price = float(current_offer["price"]) if current_offer is not None and isinstance(current_offer.get("price"), (int, float)) and not isinstance(current_offer.get("price"), bool) else None
    current_surplus = context.our_value - current_price if current_price is not None else None
    current_share = current_surplus / total_surplus if current_surplus is not None else None
    target = complete_information_offer_control.get("maximum_next_offer") if complete_information_offer_control.get("status") == "available" else None
    return {
        "status": "available",
        "authority": base["authority"],
        "mode": mode,
        "total_surplus": _rounded(total_surplus),
        "current_offer_own_surplus": _rounded(current_surplus) if current_surplus is not None else None,
        "current_offer_own_surplus_share": _rounded(current_share) if current_share is not None else None,
        "minimum_accept_surplus_share": config.buyer_minimum_accept_surplus_share,
        "counteroffer_available": _counteroffer_available(game),
        "counteroffer_target_price": _rounded(float(target)) if isinstance(target, (int, float)) and not isinstance(target, bool) else None,
        "counteroffer_target_own_surplus_share": config.complete_information_min_own_surplus_share,
        "rule": "Do not abandon visible positive joint surplus while a protected counteroffer remains available. Do not accept less than 1% of visible joint surplus; counter first at the existing complete-information proposal boundary, or exit if no counter is legal.",
    }


def _stalled_outside_reservation_control(state: Mapping[str, object], context: NegotiationContext, phase: str, current_surplus: float | None, config: NegotiationLiveConfig) -> dict[str, object]:
    """Require an early exit from exact no-overlap or empirically exhausted outside-reservation sequences."""
    base: dict[str, object] = {"status": "not-applicable", "authority": "stalled-outside-reservation-exit"}
    if phase != "decision" or current_surplus is None or current_surplus >= -config.consistency_value_tolerance:
        return base
    if context.complete_information and context.opponent_value is not None:
        buyer_value = context.our_value if context.our_role == "buyer" else context.opponent_value
        seller_value = context.our_value if context.our_role == "seller" else context.opponent_value
        if buyer_value <= seller_value + config.consistency_value_tolerance:
            return {**base, "status": "walkaway-required", "mode": "visible-no-positive-joint-surplus", "total_surplus": _rounded(buyer_value - seller_value), "rule": "The current offer loses value for us and no mutually nonnegative price exists."}
        return {**base, "status": "feasible-visible-overlap"}
    if context.complete_information or context.horizon_known:
        return base
    prices = _visible_offer_prices(state, context.opponent_player)
    offer_count = len(prices)
    gap = -float(current_surplus)
    gap_ratio = gap / context.our_value
    diagnostics: dict[str, object] = {
        **base,
        "mode": "unknown-horizon-incomplete-information-stall",
        "opponent_offer_count": offer_count,
        "reservation_gap": _rounded(gap),
        "reservation_gap_ratio": _rounded(gap_ratio),
    }
    if offer_count < config.stalled_exit_min_opponent_offers or gap_ratio < config.stalled_exit_min_reservation_gap_ratio:
        return {**diagnostics, "status": "insufficient-stall-evidence"}
    recent = prices[-config.stalled_exit_recent_offer_window :]
    direction = 1.0 if context.our_role == "seller" else -1.0
    progress = [max(0.0, direction * (current - previous)) for previous, current in zip(recent, recent[1:])]
    rate = _median(progress) if progress else 0.0
    projected = gap / rate if rate > config.consistency_value_tolerance else None
    diagnostics.update(
        {
            "recent_offer_window": len(recent),
            "median_recent_progress_toward_reservation": _rounded(rate),
            "projected_additional_opponent_offers_to_reservation": _rounded(projected) if projected is not None else None,
            "projection_status": "finite" if projected is not None else "unbounded-at-current-pace",
        }
    )
    if projected is None or projected >= config.stalled_exit_min_projected_opponent_offers:
        return {**diagnostics, "status": "walkaway-required", "rule": "The offer remains materially outside our reservation boundary after repeated opponent offers, and recent movement projects an exhausted continuation."}
    return {**diagnostics, "status": "continued-movement-plausible"}


def _hardening_budget_control(state: Mapping[str, object], context: NegotiationContext, phase: str, config: NegotiationLiveConfig) -> dict[str, object]:
    """Allow one information probe away from overlap, then require observed opponent progress before another."""
    base: dict[str, object] = {"status": "not-applicable", "authority": "one-probe-hardening-budget"}
    if phase != "decision":
        return base
    own_prices = _visible_offer_prices(state, context.our_player)
    opponent_prices = _visible_offer_prices(state, context.opponent_player)
    direction = 1.0 if context.our_role == "seller" else -1.0
    hardening_steps = [max(0.0, direction * (current - previous)) for previous, current in zip(own_prices, own_prices[1:])]
    hardening_count = sum(step > config.consistency_value_tolerance for step in hardening_steps)
    opponent_direction = direction
    opponent_progress = [max(0.0, opponent_direction * (current - previous)) for previous, current in zip(opponent_prices, opponent_prices[1:])]
    relative_progress = (opponent_progress[-1] / max(1.0, abs(context.our_value))) if opponent_progress else 0.0
    probe_changed_regime = relative_progress + 1e-12 >= config.hardening_response_minimum_relative_progress
    exhausted = hardening_count >= config.hardening_probe_budget and not probe_changed_regime
    result: dict[str, object] = {
        **base,
        "status": "exhausted" if exhausted else "available",
        "hardening_probe_budget": config.hardening_probe_budget,
        "observed_hardening_steps": hardening_count,
        "latest_hardening_magnitude": _rounded(hardening_steps[-1]) if hardening_steps else 0.0,
        "latest_opponent_progress_toward_us": _rounded(opponent_progress[-1]) if opponent_progress else 0.0,
        "latest_opponent_relative_progress": _rounded(relative_progress),
        "minimum_relative_progress_to_reopen_budget": _rounded(config.hardening_response_minimum_relative_progress),
        "probe_changed_opponent_regime": probe_changed_regime,
        "rule": "One movement away from feasible overlap is an information probe. Without a material opponent move toward us after that probe, later proposals may hold or soften but may not harden again.",
    }
    if exhausted and own_prices:
        result["maximum_next_offer" if context.our_role == "seller" else "minimum_next_offer"] = _rounded(own_prices[-1])
    return result


def _decision_facts(game: dict[str, Any], context: NegotiationContext, rating_surrogate: NegotiationRatingSurrogate | None, config: NegotiationLiveConfig, *, policy_features: Mapping[str, object] | None = None) -> dict[str, object]:
    """Expose authoritative arithmetic and compact observed movement without selecting an action."""
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    our_player = context.our_player
    opponent_player = context.opponent_player
    current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
    phase = str(game.get("phase") or state.get("phase") or "")

    def movement(player: str, role: str) -> dict[str, float | int | None]:
        prices = _visible_offer_prices(state, player)
        latest = prices[-1] if prices else None
        previous = prices[-2] if len(prices) >= 2 else None
        concession = None if latest is None or previous is None else previous - latest if role == "seller" else latest - previous
        return {"offer_count": len(prices), "previous_price": _rounded(previous) if previous is not None else None, "latest_price": _rounded(latest) if latest is not None else None, "latest_concession_toward_counterpart": _rounded(concession) if concession is not None else None}

    actionable_offer = current_offer if phase == "decision" and current_offer is not None and str(current_offer.get("from_player") or "") == opponent_player else None
    current_price = float(actionable_offer["price"]) if actionable_offer is not None and isinstance(actionable_offer.get("price"), (int, float)) and not isinstance(actionable_offer.get("price"), bool) else None
    own_payoff = _own_surplus(current_price, context) if current_price is not None else None
    maximum = context.max_rounds
    complete: dict[str, object] | None = None
    opponent_payoff = None
    if context.complete_information and context.opponent_value is not None:
        buyer_value = context.our_value if context.our_role == "buyer" else context.opponent_value
        seller_value = context.our_value if context.our_role == "seller" else context.opponent_value
        total_surplus = buyer_value - seller_value
        if current_price is not None:
            opponent_payoff = context.opponent_value - current_price if context.opponent_role == "buyer" else current_price - context.opponent_value
        complete = {
            "buyer_value": _rounded(buyer_value),
            "seller_value": _rounded(seller_value),
            "total_surplus": _rounded(total_surplus),
            "current_offer_own_surplus": _rounded(own_payoff) if own_payoff is not None else None,
            "current_offer_opponent_surplus": _rounded(opponent_payoff) if opponent_payoff is not None else None,
            "current_offer_own_surplus_share": _rounded(own_payoff / total_surplus) if own_payoff is not None and total_surplus > 0 else None,
            "current_offer_opponent_surplus_share": _rounded(opponent_payoff / total_surplus) if opponent_payoff is not None and total_surplus > 0 else None,
        }
    rating: dict[str, object]
    if rating_surrogate is None:
        rating = {"status": "unavailable"}
    else:
        no_deal = rating_surrogate.predict_scenario(game, outcome="no_deal", own_payoff=0.0, opponent_payoff=0.0, round_number=context.round_number)
        accepted = rating_surrogate.predict_scenario(game, outcome="agreement", own_payoff=own_payoff, opponent_payoff=opponent_payoff, round_number=context.round_number) if own_payoff is not None else None
        rating = {
            "status": "retrospective-surrogate",
            "current_offer_if_accepted": accepted,
            "no_deal": no_deal,
            "estimated_accept_minus_no_deal": _rounded(float(accepted["estimated_rating_delta"]) - float(no_deal["estimated_rating_delta"])) if accepted is not None else None,
            "boundary": "Fallible displayed-rating estimate; opponent rating and hidden reservation values are absent, and the estimate supplies no action authority.",
        }
    our_movement = movement(our_player, context.our_role)
    opponent_movement = movement(opponent_player, context.opponent_role)
    reciprocal_concession_control: dict[str, object] = {"status": "not-applicable", "authority": "bilateral-concession-bound"}
    if phase == "decision":
        previous_own_offer = our_movement.get("latest_price")
        opponent_concession = opponent_movement.get("latest_concession_toward_counterpart")
        opponent_offer_count = opponent_movement.get("offer_count")
        sufficient = isinstance(previous_own_offer, (int, float)) and not isinstance(previous_own_offer, bool) and isinstance(opponent_concession, (int, float)) and not isinstance(opponent_concession, bool) and isinstance(opponent_offer_count, int) and opponent_offer_count >= 2
        if sufficient:
            matched_concession = config.reciprocal_concession_match_ratio * max(0.0, float(opponent_concession))
            reciprocal_concession_control = {
                "status": "available",
                "authority": "bilateral-concession-bound",
                "role": context.our_role,
                "previous_own_offer": _rounded(float(previous_own_offer)),
                "latest_opponent_concession_toward_us": _rounded(float(opponent_concession)),
                "maximum_matched_concession": _rounded(matched_concession),
                "match_ratio": config.reciprocal_concession_match_ratio,
                "rule": "After at least 2 visible opponent offers, do not concede more than the configured fraction of the opponent's latest positive concession. A plateau or adverse move permits no further concession; a hold or harder move remains legal.",
            }
            if context.our_role == "buyer":
                reciprocal_concession_control["maximum_next_offer"] = _rounded(min(context.our_value, float(previous_own_offer) + matched_concession))
            else:
                reciprocal_concession_control["minimum_next_offer"] = _rounded(max(context.our_value, float(previous_own_offer) - matched_concession))
        else:
            reciprocal_concession_control = {"status": "insufficient-visible-movement", "authority": "bilateral-concession-bound", "role": context.our_role}
    incomplete_authority = str((policy_features or {}).get(NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE) or "fixed-markup")
    one_round_seller_control = _one_round_seller_control(context, phase, config, incomplete_authority=incomplete_authority)
    complete_information_offer_control = _complete_information_offer_control(context, config)
    buyer_scale_mode = str((policy_features or {}).get(NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE) or "legacy")
    buyer_scale_aware_decision_control = _buyer_scale_aware_decision_control(game, context, phase, complete_information_offer_control, config, mode=buyer_scale_mode)
    stalled_outside_reservation_control = _stalled_outside_reservation_control(state, context, phase, own_payoff, config)
    hardening_budget_control = _hardening_budget_control(state, context, phase, config)
    return {
        "authority": "arithmetic-and-observed-history-only",
        "objective": "Agreement has no independent value. Optimize expected competition result subject to legality and nonnegative own surplus.",
        "our_player": our_player,
        "opponent_player": opponent_player,
        "our_role": context.our_role,
        "opponent_role": context.opponent_role,
        "our_reservation_value": _rounded(context.our_value),
        "phase": phase,
        "current_round": context.round_number,
        "known_max_rounds": maximum,
        "later_rounds_after_current": max(0, maximum - context.round_number) if context.horizon_known and maximum is not None else None,
        "current_offer_price": _rounded(current_price) if current_price is not None else None,
        "current_offer_immediate_own_payoff": _rounded(own_payoff) if own_payoff is not None else None,
        "binding_offer_rule": "Every numeric offer is binding and may be accepted immediately. Use only terms that are acceptable as a terminal agreement.",
        "our_observed_offer_movement": our_movement,
        "opponent_observed_offer_movement": opponent_movement,
        "one_round_seller_control": one_round_seller_control,
        "complete_information_offer_control": complete_information_offer_control,
        "buyer_scale_aware_decision_control": buyer_scale_aware_decision_control,
        "stalled_outside_reservation_control": stalled_outside_reservation_control,
        "hardening_budget_control": hardening_budget_control,
        "reciprocal_concession_control": reciprocal_concession_control,
        "complete_information_surplus": complete,
        "rating_objective_surrogate": rating,
    }


def _material_tolerance(left: float, right: float, config: NegotiationLiveConfig) -> float:
    return max(config.consistency_value_tolerance, config.consistency_relative_tolerance * max(1.0, abs(left), abs(right)))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _opponent_concession_profile(evidence: _WithinGameEvidence, context: NegotiationContext, config: NegotiationLiveConfig) -> dict[str, object]:
    """Measure recent movement in price units toward our side and attenuate repeated plateaus."""
    prices = [float(row.proposal_price_ratio) * float(row.context.our_value) for row in evidence.rows if row.action_type == "proposal" and row.proposal_price_ratio is not None]
    direction = 1.0 if context.opponent_role == "buyer" else -1.0
    concessions = [direction * (current - previous) for previous, current in zip(prices, prices[1:])]
    recent = concessions[-config.continuation_concession_window :]
    raw_slope = _median(recent)
    positive_slope = max(0.0, raw_slope)
    plateau_length = 0
    if prices:
        plateau_length = 1
        for previous in reversed(prices[:-1]):
            if not math.isclose(previous, prices[-1], rel_tol=1e-9, abs_tol=max(1e-9, abs(prices[-1]) * 1e-9)):
                break
            plateau_length += 1
    plateau_multiplier = config.continuation_plateau_decay ** max(0, plateau_length - 1)
    effective_slope = positive_slope * plateau_multiplier
    return {
        "opponent_offer_count": len(prices),
        "latest_opponent_price": _rounded(prices[-1]) if prices else None,
        "recent_concessions_toward_us": [_rounded(value) for value in recent],
        "median_recent_concession_toward_us": _rounded(raw_slope),
        "latest_price_plateau_length": plateau_length,
        "plateau_multiplier": _rounded(plateau_multiplier),
        "effective_concession_slope": _rounded(effective_slope),
    }


def _unknown_horizon_lengths(bundle: _ForecastBundle, context: NegotiationContext) -> tuple[list[int], str]:
    observations = (*bundle.target_prior, *bundle.population_prior)
    selections = (
        ("matched-role-and-information", [observation for observation in observations if not observation.row.context.horizon_known and observation.row.context.opponent_role == context.opponent_role and observation.row.context.complete_information == context.complete_information]),
        ("matched-role", [observation for observation in observations if not observation.row.context.horizon_known and observation.row.context.opponent_role == context.opponent_role]),
        ("all-unknown-horizon", [observation for observation in observations if not observation.row.context.horizon_known]),
    )
    selected_label = selections[-1][0]
    selected = selections[-1][1]
    for label, candidates in selections:
        game_ids = {observation.row.context.game_id for observation in candidates}
        if len(game_ids) >= 5 or label == selections[-1][0]:
            selected_label = label
            selected = candidates
            break
    lengths: dict[str, int] = {}
    for observation in selected:
        game_id = observation.row.context.game_id
        lengths[game_id] = max(lengths.get(game_id, 0), int(observation.row.context.round_number))
    return list(lengths.values()), selected_label


def _unknown_horizon_hazard(lengths: Sequence[int], *, after_round: int, config: NegotiationLiveConfig) -> tuple[float, int, int]:
    at_risk = sum(length >= after_round for length in lengths)
    stopped = sum(length == after_round for length in lengths)
    prior = config.unknown_horizon_hazard_prior_games
    hazard = (stopped + prior * config.unknown_horizon_default_stop_hazard) / (at_risk + prior)
    return min(0.5, max(0.01, hazard)), at_risk, stopped


def _bounded_continuation_path(*, start_price: float, basis: str, context: NegotiationContext, evidence: _WithinGameEvidence, bundle: _ForecastBundle, config: NegotiationLiveConfig) -> tuple[dict[str, object], float]:
    concession = _opponent_concession_profile(evidence, context, config)
    effective_slope = float(concession["effective_concession_slope"])
    direction = 1.0 if context.opponent_role == "buyer" else -1.0
    first_counterproposal_round = context.round_number + 1
    if context.horizon_known and context.max_rounds is not None:
        available_steps = 0 if first_counterproposal_round > context.max_rounds else 1 + (context.max_rounds - first_counterproposal_round) // 2
        maximum_steps = min(config.continuation_max_opponent_offers, available_steps)
        lengths: list[int] = []
        hazard_source = "known-horizon-no-exogenous-stop"
    else:
        maximum_steps = config.continuation_max_opponent_offers
        lengths, hazard_source = _unknown_horizon_lengths(bundle, context)
    projected: list[dict[str, object]] = []
    projected_price = start_price
    survival = 1.0
    for step in range(1, maximum_steps + 1):
        round_number = first_counterproposal_round + 2 * (step - 1)
        interval_survival = 1.0
        hazard_steps: list[dict[str, object]] = []
        if step > 1:
            projected_price = max(1e-9, projected_price + direction * effective_slope * config.continuation_concession_decay ** (step - 2))
            if not context.horizon_known:
                previous_round = int(projected[-1]["projected_round"])
                for after_round in range(previous_round, round_number):
                    hazard, at_risk, stopped = _unknown_horizon_hazard(lengths, after_round=after_round, config=config)
                    interval_survival *= 1 - hazard
                    hazard_steps.append({"after_round": after_round, "hazard": _rounded(hazard), "at_risk_game_count": at_risk, "stopped_game_count": stopped})
                survival *= interval_survival
        projected_surplus = max(0.0, _own_surplus(projected_price, context))
        survival_weighted_value = survival * projected_surplus
        projected.append(
            {
                "opponent_offer_index": step,
                "projected_round": round_number,
                "projected_price": _rounded(projected_price),
                "projected_nonnegative_own_surplus": _rounded(projected_surplus),
                "survival_probability": _rounded(survival),
                "stop_hazard_before_offer": _rounded(1 - interval_survival),
                "hazard_steps_before_offer": hazard_steps,
                "survival_weighted_value": _rounded(survival_weighted_value),
            }
        )
    value = max((float(item["survival_weighted_value"]) for item in projected), default=0.0)
    return (
        {
            "mode": f"bounded-{basis}-counterproposal-plus-concession-path",
            "counterproposal_basis": basis,
            "maximum_projected_opponent_offers": config.continuation_max_opponent_offers,
            "projected_opponent_offer_count": len(projected),
            "concession_profile": concession,
            "horizon": {
                "known": context.horizon_known,
                "known_max_rounds": context.max_rounds,
                "unknown_horizon_hazard_source": hazard_source,
                "unknown_horizon_reference_game_count": len(lengths),
                "default_stop_hazard": config.unknown_horizon_default_stop_hazard if not context.horizon_known else None,
            },
            "projected_opponent_offers": projected,
            "bounded_continuation_value": _rounded(value),
            "boundary": "One conditional counterproposal followed by a bounded nonbranching concession path; no full game tree or equilibrium claim.",
        },
        value,
    )


@dataclass(frozen=True)
class _ShadowRollout:
    """Bounded continuation calculation kept outside the worker prompt."""

    game: dict[str, Any]
    evidence: _WithinGameEvidence
    bundle: _ForecastBundle
    opponent_id: str
    opponent_name: str
    config: NegotiationLiveConfig

    def evaluate_offer(self, price: float, *, message: str = "") -> dict[str, object]:
        response_row = _response_row(self.game, self.evidence, opponent_id=self.opponent_id, opponent_name=self.opponent_name, price=price, message=message)
        response = self.bundle.response(response_row)
        acceptance_probability = float(response["probability"])
        own_surplus = _own_surplus(price, response_row.context)
        continuation_value = 0.0
        counterproposal: dict[str, object] | None = None
        continuation: dict[str, object] | None = None
        risk_diagnostic: dict[str, object] | None = None
        if _later_round_available(response_row.context):
            rejected_row = replace(response_row, accepted=False, decision="RejectOffer")
            rejected_bundle = self.bundle.incorporate(rejected_row, precomputed_forecast=response)
            proposal_row = _proposal_row_after_rejection(rejected_row)
            proposal = rejected_bundle.proposal(proposal_row)
            distribution = proposal["distribution"]
            median_demand = _fast_quantile(distribution, 0.5)
            adverse_demand = _fast_quantile(distribution, self.config.adverse_counterproposal_quantile)
            median_price = price_from_opponent_demand(median_demand, opponent_role=response_row.context.opponent_role, our_value=response_row.context.our_value)
            adverse_price = price_from_opponent_demand(adverse_demand, opponent_role=response_row.context.opponent_role, our_value=response_row.context.our_value)
            central_initial_surplus = max(0.0, _own_surplus(median_price, response_row.context))
            adverse_initial_surplus = max(0.0, _own_surplus(adverse_price, response_row.context))
            counterproposal = {
                "mean_demand": _rounded(distribution.mean),
                "sigma_demand": _rounded(distribution.sigma),
                "q50_demand": _rounded(median_demand),
                "q50_price": _rounded(median_price),
                "q50_nonnegative_own_surplus": _rounded(central_initial_surplus),
                "adverse_quantile": self.config.adverse_counterproposal_quantile,
                "adverse_opponent_demand": _rounded(adverse_demand),
                "adverse_price": _rounded(adverse_price),
                "adverse_nonnegative_own_surplus": _rounded(adverse_initial_surplus),
                "expert_weights_after_conditional_rejection": {name: _rounded(float(value)) for name, value in dict(proposal["weights"]).items()},
            }
            continuation, continuation_value = _bounded_continuation_path(start_price=median_price, basis="central-q50", context=response_row.context, evidence=self.evidence, bundle=self.bundle, config=self.config)
            risk_diagnostic, _risk_value = _bounded_continuation_path(start_price=adverse_price, basis=f"adverse-q{int(round(100 * self.config.adverse_counterproposal_quantile))}", context=response_row.context, evidence=self.evidence, bundle=self.bundle, config=self.config)
            risk_diagnostic["authority"] = "risk-diagnostic-only"
        expected_value = acceptance_probability * own_surplus + (1 - acceptance_probability) * continuation_value
        return {
            "price": _rounded(price),
            "opponent_demand": _rounded(float(response_row.offered_demand)),
            "opponent_surplus_share": _rounded(float(response_row.offered_opponent_surplus_share)) if response_row.offered_opponent_surplus_share is not None else None,
            "own_surplus_if_accepted": _rounded(own_surplus),
            "opponent_response": _expert_probability_receipt(response),
            "conditional_rejection_counterproposal": counterproposal,
            "conditional_rejection_continuation": continuation,
            "conditional_rejection_risk_diagnostic": risk_diagnostic,
            "bounded_expected_value": _rounded(expected_value),
            "expected_continuation_basis": "central-q50",
            "boundary": "central single-path expected continuation with a separate adverse risk diagnostic; no full-tree or equilibrium claim",
        }

    def projection(self) -> dict[str, object]:
        state = self.game.get("game_state") if isinstance(self.game.get("game_state"), dict) else {}
        context_price = float(state.get("last_offer", {}).get("price")) if isinstance(state.get("last_offer"), dict) and state["last_offer"].get("price") is not None else None
        reference_context = _reference_context(self.game, self.evidence, opponent_id=self.opponent_id, opponent_name=self.opponent_name)
        phase = str(self.game.get("phase") or state.get("phase") or "")
        counteroffer_available = _counteroffer_available(self.game)
        offer_available = phase != "decision" or counteroffer_available
        candidates = [self.evaluate_offer(price) for price in _candidate_prices(reference_context, self.evidence, self.config)] if offer_available else []
        best = max(candidates, key=lambda value: float(value["bounded_expected_value"])) if candidates else None
        current_accept_value = _own_surplus(context_price, reference_context) if context_price is not None else None
        if phase == "decision" and current_accept_value is not None:
            best_counter_value = float(best["bounded_expected_value"]) if best is not None and counteroffer_available else 0.0
            if max(current_accept_value, best_counter_value) <= 0:
                recommendation = "walkaway"
            elif current_accept_value + self.config.consistency_value_tolerance >= best_counter_value:
                recommendation = "accept"
            else:
                recommendation = "reject-and-counter"
        else:
            recommendation = "offer" if best is not None else "walkaway"
        return {
            "mode": "shadow-only-not-in-worker-prompt",
            "recommendation": recommendation,
            "current_offer_accept_value": _rounded(current_accept_value) if current_accept_value is not None else None,
            "counteroffer_available": counteroffer_available,
            "best_candidate": copy.deepcopy(best),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "authority": "diagnostic-shadow-with-inherited-reject-only-v2.7-gate",
        }


def _positive_surplus_tick(value: float, config: NegotiationLiveConfig) -> float:
    return max(config.positive_surplus_minimum_tick, abs(value) * config.positive_surplus_relative_tick)


def _proposal_opponent_surplus(game: Mapping[str, object], price: float) -> float | None:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    if state.get("complete_information") is not True:
        return None
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    opponent_player = _other_player(our_player)
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    opponent_value = state.get(f"{opponent_player}_value")
    if isinstance(opponent_value, bool) or not isinstance(opponent_value, (int, float)):
        return None
    return float(opponent_value) - price if opponent_role == "buyer" else price - float(opponent_value)


def _guard_negotiation_action(*, game: dict[str, Any], action: dict[str, Any], shadow: Mapping[str, object], prompt_context: Mapping[str, object], config: NegotiationLiveConfig) -> tuple[dict[str, Any], list[str]]:
    """Apply v2.10's scale-aware buyer, terminal-price, cumulative-surplus, stalled-exit, and inherited controls."""
    facts = prompt_context.get("deterministic_decision_facts") if isinstance(prompt_context.get("deterministic_decision_facts"), Mapping) else {}
    role = str(facts.get("our_role") or "")
    raw_value = facts.get("our_reservation_value")
    if role not in {"buyer", "seller"} or isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return dict(action), []
    value = float(raw_value)
    zero_tolerance = max(config.consistency_value_tolerance, _positive_surplus_tick(value, config) * 1e-6)
    guarded = dict(action)
    applied: list[str] = []

    def own_surplus(price: float) -> float:
        return value - price if role == "buyer" else price - value

    def nudge_positive(price: float) -> float:
        del price
        tick = _positive_surplus_tick(value, config)
        return max(0.0, value - tick) if role == "buyer" else value + tick

    decision = str(guarded.get("decision") or "")
    current_surplus = facts.get("current_offer_immediate_own_payoff")
    buyer_scale = facts.get("buyer_scale_aware_decision_control") if isinstance(facts.get("buyer_scale_aware_decision_control"), Mapping) else {}
    if role == "buyer" and buyer_scale.get("status") == "available":
        counter_available = buyer_scale.get("counteroffer_available") is True
        counter_price = buyer_scale.get("counteroffer_target_price")
        counter_valid = counter_available and isinstance(counter_price, (int, float)) and not isinstance(counter_price, bool)
        current_share = buyer_scale.get("current_offer_own_surplus_share")
        minimum_share = buyer_scale.get("minimum_accept_surplus_share")
        if decision == "WalkAway" and counter_valid:
            guarded = {"decision": "RejectOffer", "product_price": float(counter_price)}
            applied.append("negotiation_v214_feasible_surplus_counter_before_walkaway")
            decision = "RejectOffer"
        elif decision == "AcceptOffer" and isinstance(current_share, (int, float)) and not isinstance(current_share, bool) and isinstance(minimum_share, (int, float)) and not isinstance(minimum_share, bool) and float(current_share) < float(minimum_share):
            if counter_valid:
                guarded = {"decision": "RejectOffer", "product_price": float(counter_price)}
                applied.append("negotiation_v214_minimum_buyer_surplus_counter")
                decision = "RejectOffer"
            else:
                return {"decision": "WalkAway"}, ["negotiation_v214_minimum_buyer_surplus_exit"]
    if decision == "AcceptOffer" and isinstance(current_surplus, (int, float)) and not isinstance(current_surplus, bool) and float(current_surplus) <= zero_tolerance:
        return {"decision": "WalkAway"}, ["negotiation_v27_no_zero_surplus_acceptance"]

    stalled = facts.get("stalled_outside_reservation_control") if isinstance(facts.get("stalled_outside_reservation_control"), Mapping) else {}
    if stalled.get("status") == "walkaway-required":
        return {"decision": "WalkAway"}, ["negotiation_v28_stalled_outside_reservation_exit"]

    hardening = facts.get("hardening_budget_control") if isinstance(facts.get("hardening_budget_control"), Mapping) else {}

    one_round = facts.get("one_round_seller_control") if isinstance(facts.get("one_round_seller_control"), Mapping) else {}
    if "product_price" in guarded and one_round.get("status") == "required" and isinstance(one_round.get("target_price"), (int, float)) and not isinstance(one_round.get("target_price"), bool):
        target = float(one_round["target_price"])
        if not math.isclose(float(guarded["product_price"]), target, rel_tol=1e-9, abs_tol=zero_tolerance):
            guarded["product_price"] = target
            mode = str(one_round.get("mode") or "")
            applied.append("negotiation_v28_one_round_seller_complete_boundary" if mode == "complete-information-positive-surplus-boundary" else "negotiation_v28_one_round_seller_incomplete_markup")

    complete_floor = facts.get("complete_information_offer_control") if isinstance(facts.get("complete_information_offer_control"), Mapping) else {}
    if "product_price" in guarded and complete_floor.get("status") == "available":
        if role == "seller":
            floor = complete_floor.get("minimum_next_offer")
            if isinstance(floor, (int, float)) and not isinstance(floor, bool) and float(guarded["product_price"]) < float(floor) - zero_tolerance:
                guarded["product_price"] = float(floor)
                applied.append("negotiation_v28_complete_information_surplus_floor")
        else:
            cap = complete_floor.get("maximum_next_offer")
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and float(guarded["product_price"]) > float(cap) + zero_tolerance:
                guarded["product_price"] = float(cap)
                applied.append("negotiation_v28_complete_information_surplus_floor")

    if "product_price" in guarded:
        price = float(guarded["product_price"])
        if own_surplus(price) <= zero_tolerance:
            opponent_surplus = _proposal_opponent_surplus(game, price)
            if decision == "RejectOffer" and opponent_surplus is not None and opponent_surplus > zero_tolerance:
                return {"decision": "WalkAway"}, ["negotiation_v27_no_zero_surplus_counter_transfer"]
            guarded["product_price"] = nudge_positive(price)
            applied.append("negotiation_v27_positive_surplus_offer_floor")

    if decision == "AcceptOffer" and shadow.get("recommendation") == "reject-and-counter" and shadow.get("counteroffer_available") is True:
        complete = facts.get("complete_information_surplus") if isinstance(facts.get("complete_information_surplus"), Mapping) else None
        evidence = prompt_context.get("evidence") if isinstance(prompt_context.get("evidence"), Mapping) else {}
        ood_flags = prompt_context.get("ood_flags") if isinstance(prompt_context.get("ood_flags"), list) else []
        best = shadow.get("best_candidate") if isinstance(shadow.get("best_candidate"), Mapping) else None
        if complete is not None and best is not None and not ood_flags:
            total = complete.get("total_surplus")
            current_share = complete.get("current_offer_own_surplus_share")
            current_value = shadow.get("current_offer_accept_value")
            best_value = best.get("bounded_expected_value")
            best_surplus = best.get("own_surplus_if_accepted")
            population_games = evidence.get("population_game_count")
            supported = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (total, current_share, current_value, best_value, best_surplus, population_games))
            if supported and float(total) > zero_tolerance:
                gain_share = (float(best_value) - float(current_value)) / float(total)
                candidate_share = float(best_surplus) / float(total)
                if float(current_share) <= config.shadow_reject_max_current_share and gain_share >= config.shadow_reject_min_expected_gain_share and candidate_share >= config.shadow_reject_min_candidate_share and int(float(population_games)) >= config.shadow_reject_min_population_games:
                    guarded = {"decision": "RejectOffer", "product_price": float(best["price"])}
                    applied.append("negotiation_v27_one_sided_shadow_reject")
                    decision = "RejectOffer"

    if "product_price" in guarded:
        control = facts.get("reciprocal_concession_control") if isinstance(facts.get("reciprocal_concession_control"), Mapping) else {}
        if control.get("status") == "available" and role == "buyer":
            cap = control.get("maximum_next_offer")
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and float(guarded["product_price"]) > float(cap) + zero_tolerance:
                guarded["product_price"] = float(cap)
                applied.append("negotiation_v27_reciprocal_concession_bound")
        elif control.get("status") == "available" and role == "seller":
            floor = control.get("minimum_next_offer")
            if isinstance(floor, (int, float)) and not isinstance(floor, bool) and float(guarded["product_price"]) < float(floor) - zero_tolerance:
                guarded["product_price"] = float(floor)
                applied.append("negotiation_v27_reciprocal_concession_bound")

    if "product_price" in guarded and hardening.get("status") == "exhausted":
        if role == "seller":
            cap = hardening.get("maximum_next_offer")
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and float(guarded["product_price"]) > float(cap) + zero_tolerance:
                guarded["product_price"] = float(cap)
                applied.append("negotiation_v210_hardening_probe_budget")
        else:
            floor = hardening.get("minimum_next_offer")
            if isinstance(floor, (int, float)) and not isinstance(floor, bool) and float(guarded["product_price"]) < float(floor) - zero_tolerance:
                guarded["product_price"] = float(floor)
                applied.append("negotiation_v210_hardening_probe_budget")

    if "product_price" in guarded and own_surplus(float(guarded["product_price"])) <= zero_tolerance:
        guarded["product_price"] = nudge_positive(float(guarded["product_price"]))
        if "negotiation_v27_positive_surplus_offer_floor" not in applied:
            applied.append("negotiation_v27_positive_surplus_offer_floor")
    return guarded, applied


@dataclass(frozen=True)
class NegotiationTurnForecast:
    """Prompt projection and shadow predictor frozen before model inference."""

    prompt_context: dict[str, object]
    forecast_receipt: dict[str, object]
    bundle: _ForecastBundle
    evidence: _WithinGameEvidence
    game: dict[str, Any]
    opponent_id: str
    opponent_name: str
    seed_sha256: str
    state_revision: int
    rollout: _ShadowRollout

    def guard_action(self, action: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        shadow = self.forecast_receipt.get("shadow_utility_rollout")
        return _guard_negotiation_action(game=self.game, action=action, shadow=shadow if isinstance(shadow, Mapping) else {}, prompt_context=self.prompt_context, config=self.rollout.config)

    def candidate_selection_evidence(self, action: Mapping[str, Any]) -> dict[str, object]:
        """Evaluate one frozen offer or terminal decision in common own-surplus units."""
        decision = str(action.get("decision") or "")
        price = action.get("product_price")
        if decision in {"AcceptOffer", "WalkAway"} or (decision == "RejectOffer" and price is None):
            state = self.game.get("game_state") if isinstance(self.game.get("game_state"), dict) else {}
            current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else {}
            current_price = current_offer.get("price")
            reference = _reference_context(self.game, self.evidence, opponent_id=self.opponent_id, opponent_name=self.opponent_name)
            terminal_value = _own_surplus(float(current_price), reference) if decision == "AcceptOffer" and isinstance(current_price, (int, float)) and not isinstance(current_price, bool) else 0.0
            return {
                "model_version": MODEL_VERSION,
                "frontier": "frozen-pre-inference-family-advisor",
                "evaluation": {"decision": decision, "terminal": True, "price": _rounded(float(current_price)) if decision == "AcceptOffer" and isinstance(current_price, (int, float)) and not isinstance(current_price, bool) else None, "own_surplus_if_accepted": _rounded(terminal_value), "bounded_expected_value": _rounded(terminal_value), "boundary": "exact terminal own surplus; no opponent response or continuation is fabricated"},
                "authority": "exact-terminal-candidate-value",
            }
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)):
            raise ValueError("Negotiation candidate evidence requires a finite product_price")
        return {
            "model_version": MODEL_VERSION,
            "frontier": "frozen-pre-inference-family-advisor",
            "evaluation": self.rollout.evaluate_offer(float(price), message=str(action.get("message") or "")),
            "authority": "candidate-specific-bounded-continuation-evidence",
        }

    def submission_prediction(self, action: dict[str, Any]) -> dict[str, object]:
        shadow = self.forecast_receipt["shadow_utility_rollout"]
        modeled = str(shadow.get("recommendation") or "unavailable") if isinstance(shadow, dict) else "unavailable"
        decision = str(action.get("decision") or "")
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "frontier": "computed-after-action-normalization-and-before-network-submission-from-the-frozen-pre-inference-bundle",
            "forecast_id": self.forecast_receipt["forecast_id"],
            "seed_sha256": self.seed_sha256,
            "state_revision": self.state_revision,
            "game_id": self.game.get("game_id"),
            "opponent": {"id": self.opponent_id, "name": self.opponent_name},
            "submitted_action": copy.deepcopy(action),
            "free_text_message_conditioning": "coarse-message-act-only",
            "policy_authority": f"one-round-seller-plus-scale-aware-buyer-plus-complete-surplus-floor-plus-stalled-exit-and-inherited-v2.7-controls-{LIVE_ENGINE_VERSION}",
        }
        state = self.game.get("game_state") if isinstance(self.game.get("game_state"), dict) else {}
        current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
        current_value = None
        if current_offer is not None:
            reference_context = _reference_context(self.game, self.evidence, opponent_id=self.opponent_id, opponent_name=self.opponent_name)
            current_value = _own_surplus(float(current_offer["price"]), reference_context)
        consistency: dict[str, object] = {"modeled_shadow_preference": modeled, "warnings": []}

        def add_grid_regret(evaluation: Mapping[str, object], warning: str) -> None:
            best = shadow.get("best_candidate") if isinstance(shadow, dict) and isinstance(shadow.get("best_candidate"), dict) else None
            if best is None:
                return
            evaluated_value = float(evaluation["bounded_expected_value"])
            best_value = float(best["bounded_expected_value"])
            regret = best_value - evaluated_value
            consistency["bounded_grid_regret"] = _rounded(max(0.0, regret))
            consistency["bounded_grid_regret_fraction"] = _rounded(max(0.0, regret) / max(1.0, abs(best_value)))
            consistency["bounded_grid_best_candidate_price"] = _rounded(float(best["price"]))
            if regret > _material_tolerance(best_value, evaluated_value, self.rollout.config):
                consistency["warnings"].append(warning)
        if decision == "AcceptOffer":
            consistency["submitted_decision"] = "accept"
            consistency["consistent_with_shadow"] = modeled == "accept"
            consistency["accepted_own_surplus"] = _rounded(current_value) if current_value is not None else None
            consistency["message_channel"] = "terminal-acceptance-exempt" if _message_available(self.game) and not str(action.get("message") or "").strip() else "provided-or-unavailable"
        elif decision == "WalkAway":
            consistency["submitted_decision"] = "walkaway"
            consistency["consistent_with_shadow"] = modeled == "walkaway"
            if current_value is not None and current_value >= -self.rollout.config.consistency_value_tolerance:
                consistency["warnings"].append("walkaway_from_nonnegative_current_offer")
        elif decision == "RejectOffer":
            if "product_price" in action:
                price = float(action["product_price"])
                evaluation = self.rollout.evaluate_offer(price, message=str(action.get("message") or ""))
                consistency["submitted_decision"] = "reject-and-counter"
                consistency["consistent_with_shadow"] = modeled == "reject-and-counter"
                consistency["submitted_counteroffer"] = evaluation
                add_grid_regret(evaluation, "submitted_counteroffer_below_best_shadow_grid_value")
                if not _counteroffer_available(self.game):
                    consistency["warnings"].append("counteroffer_submitted_when_current_action_schema_has_no_counteroffer")
                if current_value is not None and float(evaluation["bounded_expected_value"]) + _material_tolerance(float(evaluation["bounded_expected_value"]), current_value, self.rollout.config) < current_value:
                    consistency["warnings"].append("counteroffer_bounded_value_below_current_acceptance_value")
                if current_value is not None and float(evaluation["own_surplus_if_accepted"]) <= current_value + _material_tolerance(float(evaluation["own_surplus_if_accepted"]), current_value, self.rollout.config):
                    consistency["warnings"].append("counteroffer_does_not_improve_own_surplus_over_rejected_offer")
            else:
                consistency["submitted_decision"] = "reject-without-counter"
                consistency["consistent_with_shadow"] = modeled == "walkaway" and not _counteroffer_available(self.game)
                consistency["terminal_no_deal_equivalent"] = True
                if _counteroffer_available(self.game):
                    consistency["warnings"].append("counteroffer_price_missing_when_current_action_schema_requires_it")
                if current_value is not None and current_value > self.rollout.config.consistency_value_tolerance:
                    consistency["warnings"].append("terminal_rejection_of_positive_current_offer")
        elif "product_price" in action:
            evaluation = self.rollout.evaluate_offer(float(action["product_price"]), message=str(action.get("message") or ""))
            consistency["submitted_decision"] = "offer"
            consistency["consistent_with_shadow"] = modeled == "offer"
            consistency["submitted_offer"] = evaluation
            add_grid_regret(evaluation, "submitted_offer_below_best_shadow_grid_value")
        else:
            consistency["submitted_decision"] = "unsupported"
            consistency["consistent_with_shadow"] = False
            consistency["warnings"].append("unsupported_action_shape")
        if "product_price" in action and _message_available(self.game) and not str(action.get("message") or "").strip():
            consistency["warnings"].append("numeric_offer_omits_available_message_channel")
        if consistency.get("consistent_with_shadow") is False:
            consistency["status"] = "categorical-divergence"
        elif consistency["warnings"]:
            consistency["status"] = "material-warning"
        else:
            consistency["status"] = "consistent"
        receipt["local_dynamic_consistency"] = consistency
        return receipt


class NegotiationLiveAdvisorV2:
    """Serve frozen forecasts and update only from authenticated terminal games."""

    def __init__(self, *, seed_path: Path, journal_path: Path, project_root: Path) -> None:
        self.seed_path = seed_path
        self.journal_path = journal_path
        frontier_name = journal_path.name.replace("-completions", "-frontiers", 1) if "-completions" in journal_path.name else f"{journal_path.stem}-frontiers{journal_path.suffix}"
        self.frontier_journal_path = journal_path.with_name(frontier_name)
        self.project_root = project_root
        self._lock = threading.RLock()
        self.seed, self.seed_source_path = _load_seed_document(seed_path, project_root)
        self._verify_seed()
        self.seed_sha256 = str(self.seed["seed_sha256"])
        self.validation_config = NegotiationValidationConfig(**dict(self.seed["validation_config"]))
        model_values = {key: tuple(value) if isinstance(value, list) else value for key, value in dict(self.seed["model_config"]).items()}
        self.model_config = NegotiationModelConfigV21(**model_values)
        live_values = {key: tuple(value) if isinstance(value, list) else value for key, value in dict(self.seed["live_config"]).items()}
        self.live_config = NegotiationLiveConfig(**live_values)
        self.live_config.validate()
        rating_value = self.seed.get("rating_surrogate")
        self.rating_surrogate = NegotiationRatingSurrogate.from_dict(rating_value) if isinstance(rating_value, dict) else None
        self.model = NegotiationOpponentModelV21(self.model_config)
        self.paper_targets = {str(value["id"]) for value in self.seed["paper_targets"]}
        self.games: list[NegotiationGameEvidence] = []
        self.game_counts: dict[str, int] = defaultdict(int)
        self.completed_game_hashes: dict[str, str] = {}
        self._bundle_cache: dict[tuple[int, str], _ForecastBundle] = {}
        self._game_prefix_cache: dict[str, _GamePrefixCache] = {}
        self._active_frontiers: dict[str, _GameHistoricalFrontier] = {}
        self.state_revision = 0
        self.journal_sequence = 0
        self.frontier_sequence = 0
        for embedded in _seed_games(self.seed, seed_path=self.seed_source_path):
            self._apply_embedded(dict(embedded), source="seed")
        self.seed_game_count = self.state_revision
        self._replay_journal()
        self._replay_frontier_journal()

    def _verify_seed(self) -> None:
        if self.seed.get("schema_version") != SCHEMA_VERSION or self.seed.get("kind") != SEED_KIND or self.seed.get("model_version") != MODEL_VERSION:
            raise RuntimeError(f"unsupported negotiation advisor seed: {self.seed_path}")
        if self.seed.get("base_engine_version") != BASE_ENGINE_VERSION:
            raise RuntimeError(f"negotiation advisor seed has a different base engine: {self.seed_path}")
        if self.seed.get("seed_sha256") != _seed_digest(self.seed, seed_path=self.seed_source_path):
            raise RuntimeError(f"negotiation advisor seed SHA-256 mismatch: {self.seed_path}")
        for label, receipt in dict(self.seed.get("implementation_receipts") or {}).items():
            path = self.project_root / str(receipt["path"])
            if not path.is_file() or _sha_file(path) != receipt.get("sha256"):
                raise RuntimeError(f"negotiation advisor implementation receipt mismatch: {label}")

    def _apply_game(self, game: NegotiationGameEvidence, final_game_sha256: str) -> None:
        previous_hash = self.completed_game_hashes.get(game.game_id)
        if previous_hash is not None:
            if previous_hash != final_game_sha256:
                raise RuntimeError(f"conflicting negotiation terminal game: {game.game_id}")
            return
        self.games.append(game)
        self.game_counts[game.opponent_id] += 1
        self.completed_game_hashes[game.game_id] = final_game_sha256
        self.state_revision += 1
        self._prune_bundle_cache()

    def _apply_embedded(self, embedded: dict[str, Any], *, source: str) -> None:
        job = embedded.get("job")
        if not isinstance(job, dict) or embedded.get("job_object_sha256") != _sha(job):
            raise RuntimeError(f"invalid {source} negotiation job object")
        game = extract_negotiation_game(job, job_path=Path(f"{source}/{job.get('job_id', 'unknown')}.json"), job_sha256=str(embedded.get("job_sha256") or _sha(job)))
        self._apply_game(game, str(job.get("final_game_sha256") or _sha(job["final_game"])))

    def _replay_journal(self) -> None:
        if not self.journal_path.is_file():
            return
        for line_number, line in enumerate(self.journal_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != JOURNAL_KIND or record.get("model_version") != MODEL_VERSION or record.get("seed_sha256") != self.seed_sha256:
                raise RuntimeError(f"invalid negotiation advisor journal record at line {line_number}")
            self.journal_sequence = max(self.journal_sequence, int(record["journal_sequence"]))
            self._apply_embedded(dict(record["embedded_game"]), source="journal")

    def _history_prefix_sha256(self, state_revision: int) -> str:
        if not 0 <= state_revision <= self.state_revision:
            raise RuntimeError(f"negotiation historical frontier is outside the available corpus: {state_revision}")
        return _sha([{"game_id": game.game_id, "final_game_sha256": game.final_game_sha256} for game in self.games[:state_revision]])

    def _write_frontier_event(self, event: str, frontier: _GameHistoricalFrontier) -> None:
        self.frontier_sequence += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": FRONTIER_JOURNAL_KIND,
            "model_version": MODEL_VERSION,
            "frontier_sequence": self.frontier_sequence,
            "recorded_at": _now(),
            "seed_sha256": self.seed_sha256,
            "event": event,
            "frontier": asdict(frontier),
        }
        self.frontier_journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.frontier_journal_path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _replay_frontier_journal(self) -> None:
        if not self.frontier_journal_path.is_file():
            return
        for line_number, line in enumerate(self.frontier_journal_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != FRONTIER_JOURNAL_KIND or record.get("model_version") != MODEL_VERSION or record.get("seed_sha256") != self.seed_sha256:
                raise RuntimeError(f"invalid negotiation frontier journal record at line {line_number}")
            self.frontier_sequence = max(self.frontier_sequence, int(record["frontier_sequence"]))
            value = record.get("frontier")
            if not isinstance(value, dict):
                raise RuntimeError(f"negotiation frontier journal record has no frontier at line {line_number}")
            frontier = _GameHistoricalFrontier(**value)
            if frontier.history_prefix_sha256 != self._history_prefix_sha256(frontier.state_revision):
                raise RuntimeError(f"negotiation frontier historical prefix changed at line {line_number}")
            event = str(record.get("event") or "")
            if event == "started":
                prior = self._active_frontiers.get(frontier.game_id)
                if prior is not None and prior != frontier:
                    raise RuntimeError(f"conflicting negotiation game frontier at line {line_number}: {frontier.game_id}")
                self._active_frontiers[frontier.game_id] = frontier
            elif event == "closed":
                prior = self._active_frontiers.get(frontier.game_id)
                if prior is not None and prior != frontier:
                    raise RuntimeError(f"negotiation game frontier closure differs at line {line_number}: {frontier.game_id}")
                self._active_frontiers.pop(frontier.game_id, None)
            else:
                raise RuntimeError(f"unsupported negotiation frontier event at line {line_number}: {event!r}")
        for game_id in self.completed_game_hashes:
            self._active_frontiers.pop(game_id, None)
        self._prune_bundle_cache()

    def _frontier_for_game(self, *, game_id: str, opponent_id: str, opponent_name: str) -> _GameHistoricalFrontier:
        frontier = self._active_frontiers.get(game_id)
        if frontier is not None:
            if frontier.opponent_id != opponent_id:
                raise RuntimeError(f"negotiation opponent identity changed within game: {game_id}")
            return frontier
        frontier = _GameHistoricalFrontier(
            game_id=game_id,
            opponent_id=opponent_id,
            opponent_name=opponent_name,
            state_revision=self.state_revision,
            history_prefix_sha256=self._history_prefix_sha256(self.state_revision),
        )
        self._write_frontier_event("started", frontier)
        self._active_frontiers[game_id] = frontier
        return frontier

    def _retire_game_frontier(self, game_id: str) -> None:
        frontier = self._active_frontiers.get(game_id)
        if frontier is None:
            self._game_prefix_cache.pop(game_id, None)
            return
        self._write_frontier_event("closed", frontier)
        self._active_frontiers.pop(game_id, None)
        self._game_prefix_cache.pop(game_id, None)
        self._prune_bundle_cache()

    def _prune_bundle_cache(self) -> None:
        retained_revisions = {self.state_revision, *(frontier.state_revision for frontier in self._active_frontiers.values())}
        self._bundle_cache = {key: bundle for key, bundle in self._bundle_cache.items() if key[0] in retained_revisions}

    def _bundle(self, opponent_id: str, opponent_name: str, *, state_revision: int | None = None) -> _ForecastBundle:
        revision = self.state_revision if state_revision is None else state_revision
        if not 0 <= revision <= self.state_revision:
            raise RuntimeError(f"negotiation bundle revision is outside the available corpus: {revision}")
        observations, target_counts = prior_observations(self.games[:revision])
        population_only = target_counts.get(opponent_id, 0) == 0
        cache_identity = "__population_only__" if population_only else opponent_id
        cached = self._bundle_cache.get((revision, cache_identity))
        if cached is not None:
            return replace(cached, opponent_id=opponent_id, opponent_name=opponent_name) if population_only else cached
        target_prior = tuple(observation for observation in observations if observation.row.context.opponent_id == opponent_id)
        population_prior = tuple(observation for observation in observations if observation.row.context.opponent_id != opponent_id)
        complete_response_weights, complete_support = _support_adjusted_weights(RESPONSE_EXPERTS, self.live_config.complete_response_weights, target_rows=len(target_prior), full_support_rows=self.live_config.minimum_support_rows)
        incomplete_response_weights, incomplete_support = _support_adjusted_weights(RESPONSE_EXPERTS, self.live_config.incomplete_response_weights, target_rows=len(target_prior), full_support_rows=self.live_config.minimum_support_rows)
        proposal_weights, proposal_support = _support_adjusted_weights(PROPOSAL_EXPERTS, self.live_config.proposal_weights, target_rows=len(target_prior), full_support_rows=self.live_config.minimum_support_rows)
        if not math.isclose(complete_support, incomplete_support, abs_tol=1e-12) or not math.isclose(complete_support, proposal_support, abs_tol=1e-12):
            raise RuntimeError("target-support adjustment diverged across Negotiation experts")
        bundle = _ForecastBundle(
            opponent_id=opponent_id,
            opponent_name=opponent_name,
            global_game_index=revision,
            target_game_index=target_counts.get(opponent_id, 0),
            population_prior=population_prior,
            target_prior=target_prior,
            prefix=(),
            model=self.model,
            complete_response_weights=complete_response_weights,
            incomplete_response_weights=incomplete_response_weights,
            proposal_weights=proposal_weights,
            target_support_factor=complete_support,
            config=self.live_config,
        )
        self._bundle_cache[(revision, cache_identity)] = bundle
        return bundle

    def _bundle_for_visible_prefix(self, *, frontier: _GameHistoricalFrontier, evidence: _WithinGameEvidence) -> tuple[_ForecastBundle, dict[str, object]]:
        game_id = frontier.game_id
        opponent_id = frontier.opponent_id
        fingerprints = tuple(_row_fingerprint(row) for row in evidence.rows)
        cached = self._game_prefix_cache.get(game_id)
        reusable = cached is not None and cached.state_revision == frontier.state_revision and cached.opponent_id == opponent_id and len(cached.row_fingerprints) <= len(fingerprints) and fingerprints[: len(cached.row_fingerprints)] == cached.row_fingerprints
        if reusable and cached is not None:
            bundle = cached.bundle
            reused_rows = len(cached.row_fingerprints)
            mode = "incremental-visible-suffix"
        else:
            bundle = self._bundle(opponent_id, frontier.opponent_name, state_revision=frontier.state_revision)
            reused_rows = 0
            mode = "deterministic-full-prefix-rebuild"
        for row in evidence.rows[reused_rows:]:
            bundle = bundle.incorporate(row)
        self._game_prefix_cache[game_id] = _GamePrefixCache(state_revision=frontier.state_revision, opponent_id=opponent_id, row_fingerprints=fingerprints, bundle=bundle)
        return bundle, {
            "mode": mode,
            "state_revision": frontier.state_revision,
            "history_prefix_sha256": frontier.history_prefix_sha256,
            "reused_row_count": reused_rows,
            "incorporated_row_count": len(evidence.rows) - reused_rows,
            "visible_row_count": len(evidence.rows),
            "restart_behavior": "rebuild-from-the-persisted-game-frontier",
        }

    def _ood_flags(self, bundle: _ForecastBundle, context: NegotiationContext) -> list[str]:
        broad = [observation.row for observation in (*bundle.population_prior, *bundle.target_prior) if observation.row.context.opponent_role == context.opponent_role and observation.row.context.complete_information == context.complete_information]
        relevant = [row for row in broad if row.context.horizon_known == context.horizon_known]
        if context.horizon_known and context.max_rounds is not None:
            exact_horizon = [row for row in relevant if row.context.max_rounds == context.max_rounds]
            if len(exact_horizon) >= self.live_config.minimum_support_rows:
                relevant = exact_horizon
        flags: list[str] = []
        if len(relevant) < self.live_config.minimum_support_rows:
            flags.append("sparse-role-information-regime")
        if context.horizon_known:
            phases = [row.context.round_phase for row in relevant]
            if phases and context.round_phase > max(phases) + 0.05:
                flags.append("known-horizon-phase-above-matched-support")
        else:
            rounds = sorted(row.context.round_number for row in relevant)
            if rounds:
                index = min(len(rounds) - 1, max(0, math.ceil(self.live_config.unknown_horizon_round_quantile * len(rounds)) - 1))
                if context.round_number > rounds[index]:
                    flags.append("unknown-horizon-round-above-historical-quantile")
                if context.round_number > rounds[-1]:
                    flags.append("unknown-horizon-round-above-historical-maximum")
        if context.complete_information and context.opponent_value is not None and context.opponent_value <= 0:
            flags.append("invalid-visible-opponent-value")
        return flags

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "model_version": MODEL_VERSION,
            "base_engine_version": BASE_ENGINE_VERSION,
            "seed_sha256": self.seed_sha256,
            "seed_game_count": self.seed_game_count,
            "paper_target_count": len(self.paper_targets),
            "frontier": "pre-inference prompt forecast, shadow utility receipt, and pre-network exact-action receipt",
            "action_authority": "one-round seller price, complete-information cumulative-surplus floor, stalled outside-reservation exit, exact-zero-transfer prevention, reciprocal-concession bound, and reject-only calibrated shadow gate",
            "hot_policy_support": {"contract": NEGOTIATION_LIVE_POLICY_CONTRACT, "parameters": list(NEGOTIATION_HOT_PARAMETERS), "pinning_owner": "parallel supervisor"},
        }

    def status(self) -> dict[str, object]:
        with self._lock:
            return {**self.manifest_receipt(), "state_revision": self.state_revision, "journal_game_count": self.state_revision - self.seed_game_count, "journal_sequence": self.journal_sequence, "active_game_frontier_count": len(self._active_frontiers), "frontier_sequence": self.frontier_sequence}

    def forecast_turn(self, game: dict[str, Any], *, live_policy: Mapping[str, object] | None = None) -> NegotiationTurnForecast:
        with self._lock:
            if game.get("game_family") != "negotiation":
                raise ValueError("negotiation advisor received a non-negotiation turn")
            opponent_id, opponent_name, named = _identity(game)
            evidence = _live_evidence(game, opponent_id=opponent_id, opponent_name=opponent_name)
            game_id = str(game.get("game_id") or "")
            if not game_id:
                raise ValueError("live negotiation game has no game_id")
            frontier = self._frontier_for_game(game_id=game_id, opponent_id=opponent_id, opponent_name=opponent_name)
            initial_bundle = self._bundle(opponent_id, opponent_name, state_revision=frontier.state_revision)
            initial_weights = {
                "complete_response": copy.deepcopy(initial_bundle.complete_response_weights),
                "incomplete_response": copy.deepcopy(initial_bundle.incomplete_response_weights),
                "proposal": copy.deepcopy(initial_bundle.proposal_weights),
            }
            bundle, prefix_cache = self._bundle_for_visible_prefix(frontier=frontier, evidence=evidence)
            state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
            reference_context = _reference_context(game, evidence, opponent_id=opponent_id, opponent_name=opponent_name)
            target_games = bundle.target_game_index
            if opponent_id in self.paper_targets:
                tier = "sealed-paper-target"
            elif target_games >= self.validation_config.warmup_games:
                tier = "exploratory-target-supported"
            elif target_games:
                tier = "sparse-target-plus-population"
            else:
                tier = "population-only"
            final_weights = {
                "complete_response": copy.deepcopy(bundle.complete_response_weights),
                "incomplete_response": copy.deepcopy(bundle.incomplete_response_weights),
                "proposal": copy.deepcopy(bundle.proposal_weights),
            }
            ood_flags = self._ood_flags(bundle, reference_context)
            turn_config, live_policy_receipt = negotiation_turn_config(self.live_config, live_policy)
            rollout = _ShadowRollout(game=copy.deepcopy(game), evidence=evidence, bundle=bundle, opponent_id=opponent_id, opponent_name=opponent_name, config=turn_config)
            shadow = rollout.projection()
            response_profile = _prompt_response_profile(shadow["candidates"])
            policy_features = live_policy_receipt.get("features") if isinstance(live_policy_receipt, Mapping) and isinstance(live_policy_receipt.get("features"), Mapping) else {}
            decision_facts = _decision_facts(game, reference_context, self.rating_surrogate, turn_config, policy_features=policy_features)
            prompt_context: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION,
                "base_engine_version": BASE_ENGINE_VERSION,
                "status": "available",
                "seed_sha256": self.seed_sha256,
                "state_revision": frontier.state_revision,
                "historical_frontier": {"game_id": game_id, "history_prefix_sha256": frontier.history_prefix_sha256, "policy": "frozen-at-first-inference-until-terminal"},
                "opponent": {"id": opponent_id, "name": opponent_name, "named": named, "identity_scope": "stable-named" if named else "current-game-only"},
                "evidence": {"tier": tier, "target_game_count": target_games, "target_row_count": len(bundle.target_prior), "population_game_count": frontier.state_revision - target_games, "population_row_count": len(bundle.population_prior), "paper_target": opponent_id in self.paper_targets},
                "within_game_update": {**evidence.receipt(), "target_only_support_factor": _rounded(bundle.target_support_factor), "initial_expert_weights": initial_weights, "updated_expert_weights": final_weights},
                "deterministic_decision_facts": decision_facts,
                "opponent_response_profile": response_profile,
                "ood_flags": ood_flags,
                "model_boundary": {"authority": "fallible-statistical-evidence-with-explicit-narrow-policy-controls", "action_directive": "follow any required one_round_seller_control, buyer_scale_aware_decision_control, or stalled_outside_reservation_control and keep numeric proposals within complete_information_offer_control and reciprocal_concession_control", "finite_price_menu": False, "policy_guard": "scale-aware-buyer-plus-tri-control-plus-inherited-v2.7-v2.8-with-v2.9-shadow-grid-coherence", "policy_guard_never_forces_acceptance": True, "equilibrium_calculation": False, "full_game_tree": False, "hidden_reservation_value_inference": False, "shadow_utility_in_worker_prompt": False},
            }
            if live_policy_receipt is not None:
                prompt_context["live_policy"] = live_policy_receipt
            forecast_id = _sha({"seed_sha256": self.seed_sha256, "state_revision": frontier.state_revision, "game_id": game.get("game_id"), "visible_state": state, "prompt_context": prompt_context})
            forecast_receipt = {"schema_version": SCHEMA_VERSION, "kind": "glee-negotiation-v2-turn-forecast", "model_version": MODEL_VERSION, "frontier": "computed-before-model-inference", "forecast_id": forecast_id, "prompt_context": copy.deepcopy(prompt_context), "shadow_utility_rollout": shadow, "operational_reconstruction": prefix_cache, "current_terminal_state_revision": self.state_revision}
            return NegotiationTurnForecast(prompt_context=prompt_context, forecast_receipt=forecast_receipt, bundle=bundle, evidence=evidence, game=copy.deepcopy(game), opponent_id=opponent_id, opponent_name=opponent_name, seed_sha256=self.seed_sha256, state_revision=frontier.state_revision, rollout=rollout)

    def update_completed_game(self, final_game: dict[str, Any], *, completed_at: str, completion_order: int) -> dict[str, object]:
        with self._lock:
            if final_game.get("game_family") != "negotiation":
                return {"status": "ignored", "reason": "non-negotiation"}
            game_id = str(final_game.get("game_id") or "")
            result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
            if str(final_game.get("status") or "").casefold() in {"timeout", "cancelled", "abandoned"} or str(result.get("outcome") or "").casefold() in {"timeout", "cancelled", "abandoned"}:
                if game_id:
                    self._retire_game_frontier(game_id)
                return {"status": "ignored", "reason": "censored-terminal-state"}
            opponent_id, opponent_name, named = _identity(final_game)
            if not game_id:
                raise ValueError("completed negotiation game has no game_id")
            final_sha = _sha(final_game)
            previous = self.completed_game_hashes.get(game_id)
            if previous is not None:
                if previous != final_sha:
                    raise RuntimeError(f"conflicting completed negotiation game: {game_id}")
                self._retire_game_frontier(game_id)
                return {"status": "duplicate", "game_id": game_id, "state_revision": self.state_revision}
            job_id = _sha({"seed_sha256": self.seed_sha256, "game_id": game_id, "final_game_sha256": final_sha})
            job = {
                "schema_version": SCHEMA_VERSION,
                "kind": "negotiation-live-advisor-job",
                "job_id": job_id,
                "opponent": {"id": opponent_id, "name": opponent_name},
                "opponent_identity_scope": "named" if named else "hidden-game-only",
                "game_id": game_id,
                "game_family": "negotiation",
                "completed_at": completed_at,
                "completion_order": completion_order,
                "final_game_sha256": final_sha,
                "final_game": final_game,
            }
            embedded = {"job": job, "job_sha256": _sha(job), "job_object_sha256": _sha(job)}
            game = extract_negotiation_game(job, job_path=Path(f"journal/{job_id}.json"), job_sha256=str(embedded["job_sha256"]))
            self.journal_sequence += 1
            record = {"schema_version": SCHEMA_VERSION, "kind": JOURNAL_KIND, "model_version": MODEL_VERSION, "journal_sequence": self.journal_sequence, "recorded_at": _now(), "seed_sha256": self.seed_sha256, "embedded_game": embedded}
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(_canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._apply_game(game, final_sha)
            self._retire_game_frontier(game_id)
            return {"status": "updated", "game_id": game_id, "opponent": {"id": opponent_id, "name": opponent_name}, "state_revision": self.state_revision, "journal_sequence": self.journal_sequence}
