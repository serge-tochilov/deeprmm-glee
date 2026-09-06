"""Sealed, restart-safe live advisor for adaptive GLEE bargaining twin v2."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import struct
import threading
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glee_bargaining_twin import (
    BargainingContext,
    BargainingDecisionRow,
    BargainingGameEvidence,
    TwinConfig,
    _clamp,
    _finite,
    _gain,
    _history_context,
    _mixed_population_prior,
    _other_player,
    _ood_flags,
    _posterior_from_prior,
    _proposal_scores,
    _response_prediction,
    _response_scores,
    _support,
    classify_message_act,
    context_from_live_game,
    extract_bargaining_game,
    load_bargaining_corpus,
    proposal_particles,
    response_particles,
)
from .glee_bargaining_twin_v2 import ENGINE_VERSION as BASE_ENGINE_VERSION, AdaptiveConfig, AdaptiveOpponentState, GaussianComponent, GaussianMixture, PROPOSAL_EXPERTS, RESPONSE_EXPERTS, mode_kernel_distribution, target_response_probability
from .glee_bargaining_validation import LogisticResponseModel, PriorObservation, RecencyKernelModel, RidgeProposalModel, ValidationConfig, _ordinary_weights, _recency_weights
from .glee_named_dossier import named_opponent_id, normalize_opponent_name
from .immutable_pack import load_ordered_json_objects, seal_json_objects


SCHEMA_VERSION = 1
LIVE_ENGINE_VERSION = "v2.17"
MODEL_VERSION = f"bargaining-live-advisor-{LIVE_ENGINE_VERSION}"
SEED_KIND = "glee-bargaining-live-v2-seed"
SEED_REFERENCE_KIND = "glee-bargaining-live-v2-seed-reference"
JOURNAL_KIND = "glee-bargaining-live-v2-completion"
COMPILED_SEED_STATE_KIND = "glee-bargaining-live-v2-compiled-seed-state"
COMPILED_GAME_UPDATE_KIND = "glee-bargaining-live-v2-compiled-game-update"
COMPILED_STATE_VERSION = 1
CURVE_SHARES = (0.05, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.45, 0.5, 0.55, 0.6, 2 / 3, 0.75, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class LiveAdvisorConfig:
    """Frozen v2.17 settings for game-local learning, cap-aware continuation, and external bounded intervention."""

    within_game_expert_learning_rate: float = 0.75
    within_game_uniform_mix: float = 0.03
    within_game_response_prior_equivalent_rows: float = 1.0
    continuation_candidate_step: float = 0.025
    consistency_share_tolerance: float = 0.005
    consistency_min_rejected_offers: int = 1
    opening_minimum_opponent_share: float = 0.5
    later_minimum_opponent_share: float = 0.4
    post_rejection_concession_step: float = 0.05
    maximum_guarded_opponent_share: float = 0.65
    minimum_offer_accept_probability: float = 0.5
    response_probability_haircut: float = 0.1
    counterproposal_risk_quantile: float = 0.8
    first_probe_value_margin: float = 0.05
    repeated_probe_value_margin: float = 0.08
    discounted_acceptance_dominance_margin: float = 0.02
    unconditional_accept_our_share: float = 0.5
    minimum_material_own_share: float = 0.05
    reciprocal_concession_match_ratio: float = 1.0
    decision_override_additional_margin: float = 0.05
    offer_expected_value_regression_margin: float = 0.05
    exact_offer_match_share_tolerance: float = 0.000001
    patient_minimum_opponent_share: float = 0.25
    patient_maximum_opponent_share: float = 0.4
    patient_minimum_own_settlement_share: float = 0.6
    patient_reciprocal_window_minimum_own_share: float = 0.45
    patient_reciprocal_window_minimum_opponent_concessions: int = 2
    patient_reciprocal_window_minimum_bilateral_movement: float = 0.05
    patient_deadlock_minimum_repetitions_per_player: int = 10
    discounted_post_rejection_minimum_own_share: float = 0.5
    post_rejection_concession_minimum_value_gain: float = 0.02
    environmental_round_cap: int = 99
    environmental_terminal_window_rounds: int = 2
    environmental_minimum_positive_own_share: float = 0.000001
    loss_minimization_discount_retention: float = 0.5
    loss_minimization_minimum_bilateral_rejections: int = 2
    loss_minimization_bridge_fraction: float = 0.5
    economic_no_progress_minimum_plateau: int = 2
    economic_no_progress_minimum_discount_loss: float = 0.02

    def validate(self) -> None:
        if self.within_game_expert_learning_rate < 0 or not 0 <= self.within_game_uniform_mix < 1:
            raise ValueError("within-game expert adaptation settings are invalid")
        if self.within_game_response_prior_equivalent_rows <= 0:
            raise ValueError("within-game response prior strength must be positive")
        if not 0 < self.continuation_candidate_step <= 0.25:
            raise ValueError("continuation candidate step must lie in (0, 0.25]")
        if self.consistency_share_tolerance < 0:
            raise ValueError("consistency share tolerance cannot be negative")
        if self.consistency_min_rejected_offers < 1:
            raise ValueError("the repeated-probe threshold must be at least one rejected offer")
        for name in ("opening_minimum_opponent_share", "later_minimum_opponent_share", "maximum_guarded_opponent_share", "minimum_offer_accept_probability", "counterproposal_risk_quantile", "unconditional_accept_our_share", "minimum_material_own_share", "patient_minimum_opponent_share", "patient_maximum_opponent_share", "patient_minimum_own_settlement_share", "patient_reciprocal_window_minimum_own_share", "discounted_post_rejection_minimum_own_share"):
            value = float(getattr(self, name))
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0, 1)")
        if not 0 < self.reciprocal_concession_match_ratio <= 1:
            raise ValueError("reciprocal_concession_match_ratio must lie in (0, 1]")
        if not 0 <= self.decision_override_additional_margin < 1:
            raise ValueError("decision_override_additional_margin must lie in [0, 1)")
        if not 0 <= self.offer_expected_value_regression_margin < 1:
            raise ValueError("offer_expected_value_regression_margin must lie in [0, 1)")
        if not 0 <= self.exact_offer_match_share_tolerance < 1:
            raise ValueError("exact_offer_match_share_tolerance must lie in [0, 1)")
        if self.opening_minimum_opponent_share < self.later_minimum_opponent_share:
            raise ValueError("opening offer floor cannot be below the later-round floor")
        if self.maximum_guarded_opponent_share < self.opening_minimum_opponent_share:
            raise ValueError("maximum guarded share cannot be below the opening floor")
        if self.patient_minimum_opponent_share > self.patient_maximum_opponent_share:
            raise ValueError("patient opponent-share floor cannot exceed its ceiling")
        if not math.isclose(self.patient_maximum_opponent_share + self.patient_minimum_own_settlement_share, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("patient offer and settlement boundaries must describe the same share boundary")
        if not self.minimum_material_own_share < self.patient_reciprocal_window_minimum_own_share < self.patient_minimum_own_settlement_share:
            raise ValueError("patient reciprocal-window floor must lie between the material-payoff and ordinary settlement floors")
        if self.patient_reciprocal_window_minimum_opponent_concessions < 2:
            raise ValueError("patient reciprocal-window detection requires at least 2 opponent concessions")
        if not 0 < self.patient_reciprocal_window_minimum_bilateral_movement < 1:
            raise ValueError("patient reciprocal-window bilateral movement must lie in (0, 1)")
        if self.patient_deadlock_minimum_repetitions_per_player < 2:
            raise ValueError("patient deadlock detection requires at least 2 repeated offers per player")
        if not 0 <= self.post_rejection_concession_minimum_value_gain < 1:
            raise ValueError("post-rejection concession value gain must lie in [0, 1)")
        if self.environmental_round_cap < 2:
            raise ValueError("environmental round cap must be at least 2")
        if not 1 <= self.environmental_terminal_window_rounds < self.environmental_round_cap:
            raise ValueError("environmental terminal window must lie inside the environmental round cap")
        if not 0 < self.environmental_minimum_positive_own_share < self.minimum_material_own_share:
            raise ValueError("environmental minimum positive share must lie below the ordinary material-share floor")
        if not 0 < self.loss_minimization_discount_retention < 1:
            raise ValueError("loss-minimization discount retention must lie in (0, 1)")
        if self.loss_minimization_minimum_bilateral_rejections < 1:
            raise ValueError("loss minimization requires at least one bilateral rejection")
        if not 0 < self.loss_minimization_bridge_fraction <= 1:
            raise ValueError("loss-minimization bridge fraction must lie in (0, 1]")
        if self.economic_no_progress_minimum_plateau < 2:
            raise ValueError("economic no-progress detection requires at least 2 repeated offers per player")
        if not 0 <= self.economic_no_progress_minimum_discount_loss < 1:
            raise ValueError("economic no-progress discount loss must lie in [0, 1)")
        if self.post_rejection_concession_step < 0 or self.response_probability_haircut < 0 or self.first_probe_value_margin < 0 or self.repeated_probe_value_margin < 0 or self.discounted_acceptance_dominance_margin < 0:
            raise ValueError("conservative policy margins cannot be negative")


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


def _atomic_compact_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _encode_float_vector(values: Sequence[float]) -> dict[str, object]:
    numbers = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("compiled bargaining state contains a non-finite score")
    packed = struct.pack(f"<{len(numbers)}d", *numbers)
    return {"encoding": "base64-zlib-little-float64-v1", "count": len(numbers), "data": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii")}


def _decode_float_vector(value: object, *, expected_count: int) -> list[float]:
    if not isinstance(value, dict) or value.get("encoding") != "base64-zlib-little-float64-v1" or value.get("count") != expected_count:
        raise RuntimeError("compiled bargaining score vector has an invalid envelope")
    try:
        packed = zlib.decompress(base64.b64decode(str(value["data"]), validate=True))
        if len(packed) != expected_count * 8:
            raise RuntimeError("compiled bargaining score vector has an invalid byte length")
        numbers = list(struct.unpack(f"<{expected_count}d", packed))
    except (KeyError, ValueError, TypeError, zlib.error, struct.error) as error:
        raise RuntimeError("compiled bargaining score vector cannot be decoded") from error
    if any(not math.isfinite(number) for number in numbers):
        raise RuntimeError("compiled bargaining score vector contains a non-finite value")
    return numbers


def _serialize_observation(observation: PriorObservation) -> dict[str, object]:
    return {"row": asdict(observation.row), "global_game_index": observation.global_game_index, "opponent_game_index": observation.opponent_game_index}


def _restore_observation(value: object) -> PriorObservation:
    if not isinstance(value, dict) or not isinstance(value.get("row"), dict):
        raise RuntimeError("compiled bargaining observation has an invalid envelope")
    raw_row = copy.deepcopy(value["row"])
    raw_context = raw_row.pop("context", None)
    if not isinstance(raw_context, dict):
        raise RuntimeError("compiled bargaining observation has no context")
    row = BargainingDecisionRow(context=BargainingContext(**raw_context), **raw_row)
    return PriorObservation(row=row, global_game_index=int(value["global_game_index"]), opponent_game_index=int(value["opponent_game_index"]))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
            raise RuntimeError(f"bargaining seed reference escapes the project root: {path}") from error
        return resolved
    if reference.get("kind") == "absolute":
        return path.resolve()
    raise RuntimeError(f"unsupported bargaining seed source reference: {reference.get('kind')!r}")


def _load_seed_document(path: Path, project_root: Path) -> tuple[dict[str, Any], Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"bargaining advisor seed is not a JSON object: {path}")
    if value.get("kind") != SEED_REFERENCE_KIND:
        return value, path.resolve()
    source_reference = value.get("source")
    if not isinstance(source_reference, dict):
        raise RuntimeError(f"installed bargaining seed has no source reference: {path}")
    source = _resolve_seed_source(source_reference, project_root)
    if not source.is_file() or _sha_file(source) != value.get("source_file_sha256"):
        raise RuntimeError(f"installed bargaining seed source failed SHA-256 verification: {source}")
    seed = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(seed, dict) or seed.get("kind") != SEED_KIND:
        raise RuntimeError(f"installed bargaining seed source is unsupported: {source}")
    if seed.get("seed_sha256") != value.get("seed_sha256"):
        raise RuntimeError(f"installed bargaining seed logical identity changed: {source}")
    return seed, source


def install_seed(source: Path, destination: Path, *, project_root: Path) -> None:
    """Install a small immutable reference to one canonical seed and reject resume drift."""
    seed, seed_source = _load_seed_document(source.resolve(), project_root)
    if seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source):
        raise RuntimeError(f"invalid bargaining advisor seed: {seed_source}")
    reference = {
        "schema_version": 1,
        "kind": SEED_REFERENCE_KIND,
        "source": _seed_source_reference(seed_source, project_root),
        "source_file_sha256": _sha_file(seed_source),
        "seed_sha256": seed["seed_sha256"],
        "model_version": seed.get("model_version"),
        "corpus_sha256": seed.get("corpus_sha256"),
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != reference:
            raise RuntimeError(f"bargaining advisor seed reference differs on resume: {destination}")
        return
    _atomic_json(destination, reference)


def _implementation_paths(project_root: Path) -> dict[str, Path]:
    return {
        "live_advisor_module": project_root / "src" / "nommd_arena" / "glee_bargaining_live_v2.py",
        "intervention_module": project_root / "src" / "nommd_arena" / "glee_bargaining_intervention.py",
        "statistical_package_module": project_root / "src" / "nommd_arena" / "glee_statistical_package.py",
        "worker_module": project_root / "src" / "nommd_arena" / "glee_worker.py",
        "supervisor_module": project_root / "src" / "nommd_arena" / "glee_parallel.py",
        "transport_client_module": project_root / "src" / "nommd_arena" / "glee_transport.py",
        "dossier_broker_module": project_root / "src" / "nommd_arena" / "glee_dossier.py",
        "activity_scheduler_module": project_root / "src" / "nommd_arena" / "glee_activity_scheduler.py",
        "live_policy_module": project_root / "src" / "nommd_arena" / "glee_live_policy.py",
        "twin_module": project_root / "src" / "nommd_arena" / "glee_bargaining_twin.py",
        "adaptive_module": project_root / "src" / "nommd_arena" / "glee_bargaining_twin_v2.py",
        "validation_module": project_root / "src" / "nommd_arena" / "glee_bargaining_validation.py",
        "bargaining_prompt": project_root / "prompts" / "glee_nommd_bargaining.md",
        "meta_controller_module": project_root / "src" / "nommd_arena" / "glee_meta_controller_v2.py",
        "meta_prompt_transport_module": project_root / "src" / "nommd_arena" / "model_runner.py",
        "meta_common_prompt": project_root / "prompts" / "glee_meta_controller_common.md",
        "meta_planner_stage_prompt": project_root / "prompts" / "glee_meta_controller_planner.md",
        "meta_selector_stage_prompt": project_root / "prompts" / "glee_meta_controller_selector.md",
        "meta_planner_prompt": project_root / "prompts" / "glee_meta_controller_planner_bargaining.md",
        "meta_selector_prompt": project_root / "prompts" / "glee_meta_controller_selector_bargaining.md",
        "meta_controller_protocol": project_root / "protocols" / "glee-terra-meta-controller-live-v1.md",
        "live_protocol": project_root / "protocols" / "glee-bargaining-live-v2-17.md",
        "persistent_rmm_protocol": project_root / "protocols" / "glee-persistent-rmm-v1.md",
        "transport_protocol": project_root / "protocols" / "glee-transport-fault-containment-v2.md",
    }


def _seed_games(seed: Mapping[str, object], *, seed_path: Path) -> list[dict[str, Any]]:
    inline = seed.get("games")
    if isinstance(inline, list):
        if any(not isinstance(value, dict) for value in inline):
            raise RuntimeError(f"bargaining seed contains a non-object game: {seed_path}")
        return copy.deepcopy(inline)
    reference = seed.get("games_ref")
    if not isinstance(reference, dict):
        raise RuntimeError(f"bargaining seed has neither inline games nor a game-pack reference: {seed_path}")
    games = load_ordered_json_objects(root=seed_path.parent, reference=reference)
    if len(games) != int(seed.get("game_count") or -1):
        raise RuntimeError(f"bargaining seed game-pack count differs from its manifest: {seed_path}")
    return games


def _seed_digest(seed: Mapping[str, object], *, seed_path: Path) -> str:
    logical = {key: copy.deepcopy(value) for key, value in seed.items() if key not in {"seed_sha256", "games_ref"}}
    if "games_ref" in seed:
        logical["games"] = _seed_games(seed, seed_path=seed_path)
    return _sha(logical)


def reseal_live_seed_implementation(*, source_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Preserve one historical frontier exactly while sealing a new implementation epoch."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite bargaining advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path, project_root)
    parent_model_version = str(seed.get("model_version") or "")
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or not parent_model_version.startswith("bargaining-live-advisor-v2."):
        raise RuntimeError(f"unsupported parent bargaining advisor seed: {source_path}")
    if seed.get("base_engine_version") != BASE_ENGINE_VERSION or seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source_path):
        raise RuntimeError(f"invalid parent bargaining advisor seed: {source_path}")
    if "games_ref" in seed and output_path.parent.resolve() != seed_source_path.parent.resolve():
        raise ValueError("a referenced bargaining corpus can be resealed only beside its source manifest; move the manifest and corpus together first")
    parent_seed_sha256 = str(seed["seed_sha256"])
    seed["generated_at"] = _now()
    seed["parent_seed_sha256"] = parent_seed_sha256
    seed["parent_model_version"] = parent_model_version
    seed["model_version"] = MODEL_VERSION
    seed["live_config"] = {**asdict(LiveAdvisorConfig()), **dict(seed.get("live_config") or {})}
    seed["reseal_reason"] = "implementation-policy-epoch-with-identical-embedded-historical-frontier"
    seed["implementation_receipts"] = {
        label: {"path": str(path.relative_to(project_root)), "sha256": _sha_file(path)}
        for label, path in _implementation_paths(project_root).items()
    }
    seed["seed_sha256"] = _seed_digest(seed, seed_path=output_path)
    _atomic_json(output_path, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SEED_KIND,
        "path": str(output_path),
        "model_version": seed["model_version"],
        "base_engine_version": seed["base_engine_version"],
        "parent_seed_sha256": parent_seed_sha256,
        "seed_sha256": seed["seed_sha256"],
        "corpus_sha256": seed["corpus_sha256"],
        "game_count": seed["game_count"],
        "row_count": seed["row_count"],
        "paper_target_count": len(seed["paper_targets"]),
    }


def promote_live_seed_journal(*, source_path: Path, journal_path: Path, output_path: Path, project_root: Path) -> dict[str, object]:
    """Fold one verified completion journal into its parent frontier and seal the current policy epoch."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite bargaining advisor seed: {output_path}")
    seed, seed_source_path = _load_seed_document(source_path, project_root)
    parent_model_version = str(seed.get("model_version") or "")
    if seed.get("schema_version") != SCHEMA_VERSION or seed.get("kind") != SEED_KIND or not parent_model_version.startswith("bargaining-live-advisor-v2."):
        raise RuntimeError(f"unsupported parent bargaining advisor seed: {source_path}")
    if seed.get("base_engine_version") != BASE_ENGINE_VERSION or seed.get("seed_sha256") != _seed_digest(seed, seed_path=seed_source_path):
        raise RuntimeError(f"invalid parent bargaining advisor seed: {source_path}")
    if not journal_path.is_file():
        raise FileNotFoundError(f"bargaining advisor journal does not exist: {journal_path}")

    parent_seed_sha256 = str(seed["seed_sha256"])
    embedded_games = _seed_games(seed, seed_path=seed_source_path)
    parent_game_count = len(embedded_games)
    promoted_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        expected_sequence = len(promoted_records) + 1
        if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != JOURNAL_KIND or record.get("model_version") != parent_model_version or record.get("seed_sha256") != parent_seed_sha256:
            raise RuntimeError(f"invalid bargaining advisor journal record at line {line_number}")
        if record.get("journal_sequence") != expected_sequence:
            raise RuntimeError(f"non-contiguous bargaining advisor journal sequence at line {line_number}")
        embedded = record.get("embedded_game")
        if not isinstance(embedded, dict):
            raise RuntimeError(f"bargaining advisor journal record has no embedded game at line {line_number}")
        promoted_records.append(copy.deepcopy(embedded))
    if not promoted_records:
        raise RuntimeError(f"bargaining advisor journal has no completion records: {journal_path}")

    embedded_games.extend(promoted_records)
    games: list[BargainingGameEvidence] = []
    named_ids: set[str] = set()
    completed_hashes: dict[str, str] = {}
    source_counts = {"named": 0, "hidden": 0}
    for index, embedded in enumerate(embedded_games, start=1):
        job = embedded.get("job")
        if not isinstance(job, dict) or embedded.get("job_object_sha256") != _sha(job) or not str(embedded.get("job_sha256") or ""):
            raise RuntimeError(f"invalid embedded bargaining game at promoted position {index}")
        final_game = job.get("final_game")
        if not isinstance(final_game, dict) or job.get("final_game_sha256") != _sha(final_game):
            raise RuntimeError(f"invalid terminal bargaining game at promoted position {index}")
        game = extract_bargaining_game(job, job_path=Path(f"promoted/{job.get('job_id', index)}.json"), job_sha256=str(embedded["job_sha256"]))
        previous = completed_hashes.get(game.game_id)
        if previous is not None:
            if previous != game.final_game_sha256:
                raise RuntimeError(f"conflicting completed bargaining game in promoted frontier: {game.game_id}")
            raise RuntimeError(f"duplicate completed bargaining game in promoted frontier: {game.game_id}")
        completed_hashes[game.game_id] = game.final_game_sha256
        games.append(game)
        opponent = job.get("opponent") if isinstance(job.get("opponent"), dict) else {}
        opponent_name = normalize_opponent_name(opponent.get("name"))
        identity_scope = str(job.get("opponent_identity_scope") or "")
        named = identity_scope == "named" or (not identity_scope and bool(opponent_name) and opponent_name != "hidden opponent" and not game.opponent_id.startswith("hidden-"))
        if named:
            named_ids.add(game.opponent_id)
            source_counts["named"] += 1
        else:
            source_counts["hidden"] += 1

    validation_config = ValidationConfig(**dict(seed["validation_config"]))
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
        "generated_at": _now(),
        "parent_seed_sha256": parent_seed_sha256,
        "parent_model_version": parent_model_version,
        "frontier": "Only games embedded in this seed may initialize the live epoch; later completed games enter through the append-only run journal.",
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
        "promotion_reason": "prospective-canary-completions-folded-without-importing-other-development-archive-games",
        "validation_config": copy.deepcopy(seed["validation_config"]),
        "twin_config": copy.deepcopy(seed["twin_config"]),
        "adaptive_config": copy.deepcopy(seed["adaptive_config"]),
        "live_config": {**asdict(LiveAdvisorConfig()), **copy.deepcopy(seed["live_config"])},
        "base_engine_version": BASE_ENGINE_VERSION,
        "paper_targets": paper_targets,
        "game_count": len(games),
        "row_count": sum(len(game.rows) for game in games),
        "rejected": copy.deepcopy(seed.get("rejected") or []),
        "corpus_sha256": _sha([{"game_id": game.game_id, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256} for game in games]),
        "implementation_receipts": {
            label: {"path": str(path.relative_to(project_root)), "sha256": _sha_file(path)}
            for label, path in _implementation_paths(project_root).items()
        },
    }
    games_reference, _unique_games = seal_json_objects(root=output_path.parent, directory=output_path.parent / "corpora", prefix="bargaining-games", objects=embedded_games)
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


class BargainingLiveSeed:
    """Seal a self-contained historical frontier for a future live v2 epoch."""

    def __init__(
        self,
        *,
        dossier_root: Path,
        output_path: Path,
        project_root: Path,
        game_archive_root: Path | None = None,
        rating_history_path: Path | None = None,
        validation_config: ValidationConfig | None = None,
        twin_config: TwinConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
        live_config: LiveAdvisorConfig | None = None,
    ) -> None:
        self.dossier_root = dossier_root
        self.output_path = output_path
        self.project_root = project_root
        self.game_archive_root = game_archive_root
        self.rating_history_path = rating_history_path
        self.validation_config = validation_config or ValidationConfig()
        self.twin_config = twin_config or TwinConfig()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.live_config = live_config or LiveAdvisorConfig()
        self.validation_config.validate()
        self.adaptive_config.validate()
        self.live_config.validate()

    def _archive_games(self) -> tuple[list[BargainingGameEvidence], list[dict[str, object]], list[dict[str, str]], set[str], dict[str, int]]:
        if self.game_archive_root is None or self.rating_history_path is None:
            raise RuntimeError("both game_archive_root and rating_history_path are required for an archive seed")
        rating_history = json.loads(self.rating_history_path.read_text(encoding="utf-8"))
        deltas = rating_history.get("game_deltas")
        if not isinstance(deltas, dict):
            raise ValueError(f"rating history has no game_deltas map: {self.rating_history_path}")
        accepted: dict[str, tuple[str, Path, dict[str, Any], bool]] = {}
        rejected: list[dict[str, str]] = []
        paths = sorted(self.game_archive_root.glob("*/games/bargaining-*.json"))
        for path in paths:
            try:
                final_game = json.loads(path.read_text(encoding="utf-8"))
                if final_game.get("game_family") != "bargaining":
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
                        raise RuntimeError(f"conflicting archived bargaining games for {game_id}: {previous[1]} and {path}")
                    rejected.append({"path": str(path), "reason": f"duplicate_game:{previous[1]}"})
                    continue
                opponent = final_game.get("opponent") if isinstance(final_game.get("opponent"), dict) else {}
                name = normalize_opponent_name(opponent.get("name"))
                named = bool(name and str(opponent.get("type") or "agent") != "hidden")
                accepted[game_id] = (str(delta["completed_at"]), path, final_game, named)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
        games: list[BargainingGameEvidence] = []
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
                "schema_version": 3,
                "kind": "bargaining-archive-seed-job",
                "job_id": _sha({"game_id": game_id, "final_game_sha256": final_sha, "completed_at": completed_at}),
                "opponent": {"id": opponent_id, "name": opponent_name},
                "opponent_identity_scope": "named" if named else "hidden",
                "game_id": game_id,
                "game_family": "bargaining",
                "completed_at": completed_at,
                "completion_order": completion_order,
                "final_game_sha256": final_sha,
                "final_game": final_game,
                "source_game_path": str(path.resolve()),
            }
            job_sha = _sha(job)
            game = extract_bargaining_game(job, job_path=path, job_sha256=job_sha)
            games.append(game)
            embedded_games.append({"job": job, "job_sha256": job_sha, "job_object_sha256": job_sha})
        return games, embedded_games, rejected, named_ids, counts

    def run(self) -> dict[str, object]:
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite bargaining advisor seed: {self.output_path}")
        if self.game_archive_root is not None or self.rating_history_path is not None:
            games, embedded_games, rejected, named_ids, source_counts = self._archive_games()
            source = {
                "kind": "authenticated-run-game-archive",
                "game_archive_root": str(self.game_archive_root.resolve()) if self.game_archive_root is not None else None,
                "rating_history_path": str(self.rating_history_path.resolve()) if self.rating_history_path is not None else None,
                "counts": source_counts,
            }
        else:
            loaded_games, loaded_rejected = load_bargaining_corpus(self.dossier_root)
            games = list(loaded_games)
            rejected = list(loaded_rejected)
            embedded_games = []
            for game in games:
                job_path = Path(game.job_path)
                job = json.loads(job_path.read_text(encoding="utf-8"))
                embedded_games.append({"job": job, "job_sha256": game.job_sha256, "job_object_sha256": _sha(job)})
            named_ids = {game.opponent_id for game in games}
            source_counts = {"named": len(games), "hidden": 0}
            source = {"kind": "named-opponent-dossier-jobs", "dossier_root": str(self.dossier_root.resolve()), "counts": source_counts}
        if not games:
            raise RuntimeError("bargaining advisor seed has no accepted games")
        game_counts = Counter(game.opponent_id for game in games)
        paper_targets = [
            {"id": opponent_id, "name": next(game.opponent_name for game in games if game.opponent_id == opponent_id), "seed_game_count": count}
            for opponent_id, count in sorted(game_counts.items())
            if opponent_id in named_ids and count >= self.validation_config.min_games
        ]
        implementation_receipts = {
            label: {"path": str(path.relative_to(self.project_root)), "sha256": _sha_file(path)}
            for label, path in _implementation_paths(self.project_root).items()
        }
        seed: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": SEED_KIND,
            "model_version": MODEL_VERSION,
            "generated_at": _now(),
            "frontier": "Only games embedded in this seed may initialize the live epoch; later completed games enter through the append-only run journal.",
            "source": source,
            "validation_config": asdict(self.validation_config),
            "twin_config": asdict(self.twin_config),
            "adaptive_config": asdict(self.adaptive_config),
            "live_config": asdict(self.live_config),
            "base_engine_version": BASE_ENGINE_VERSION,
            "paper_targets": paper_targets,
            "game_count": len(games),
            "row_count": sum(len(game.rows) for game in games),
            "rejected": list(rejected),
            "corpus_sha256": _sha([{"game_id": game.game_id, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256} for game in games]),
            "implementation_receipts": implementation_receipts,
        }
        games_reference, _unique_games = seal_json_objects(root=self.output_path.parent, directory=self.output_path.parent / "corpora", prefix="bargaining-games", objects=embedded_games)
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


@dataclass(frozen=True)
class _FrozenForecastBundle:
    """One immutable fitted state used throughout a single live turn."""

    opponent_id: str
    opponent_name: str
    global_game_index: int
    target_game_index: int
    target_prior: tuple[PriorObservation, ...]
    population_prior: tuple[PriorObservation, ...]
    response_programs: tuple[Any, ...]
    proposal_programs: tuple[Any, ...]
    proposal_program_ids: tuple[str, ...]
    hierarchical_response_weights: tuple[float, ...]
    hierarchical_proposal_weights: tuple[float, ...]
    recency_model: RecencyKernelModel
    logistic_model: LogisticResponseModel
    ridge_model: RidgeProposalModel
    response_expert_weights: dict[str, float]
    proposal_expert_weights: dict[str, float]
    adaptive_config: AdaptiveConfig
    validation_config: ValidationConfig
    live_config: LiveAdvisorConfig
    within_game_rejected_shares: tuple[float, ...] = ()
    within_game_accepted_shares: tuple[float, ...] = ()
    _response_cache: dict[tuple[BargainingContext, float], dict[str, object]] = field(default_factory=dict, compare=False, repr=False)
    _proposal_cache: dict[BargainingContext, tuple[GaussianMixture, dict[str, GaussianMixture]]] = field(default_factory=dict, compare=False, repr=False)

    def without_target_recency_authority(self) -> tuple[_FrozenForecastBundle, dict[str, object]]:
        """Return a receipt-bearing response ensemble with target recency assigned zero authority."""
        original = dict(self.response_expert_weights)
        retained = {name: weight for name, weight in original.items() if name != "target_recency"}
        total = sum(retained.values())
        if total <= 0:
            raise RuntimeError("cannot suppress target recency from an empty response ensemble")
        effective = {name: (0.0 if name == "target_recency" else retained[name] / total) for name in RESPONSE_EXPERTS}
        receipt = {
            "status": "suppressed",
            "expert": "target_recency",
            "reason": "sparse target evidence is out of distribution for the current horizon regime",
            "original_weights": {name: _round(weight) for name, weight in original.items()},
            "effective_weights": {name: _round(weight) for name, weight in effective.items()},
        }
        return replace(self, response_expert_weights=effective, _response_cache={}), receipt

    def response(self, context: BargainingContext, offered_share: float) -> dict[str, object]:
        clamped_share = _clamp(offered_share, 0.0, 1.0)
        cache_key = (context, clamped_share)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            return cached
        row = BargainingDecisionRow(context=context, action_type="response", offered_share=clamped_share)
        base = {
            "hierarchical_program": _response_prediction(self.response_programs, self.hierarchical_response_weights, context, float(row.offered_share)),
            "recency_kernel": self.recency_model.response_probability(row),
            "regularized_tabular": self.logistic_model.probability(row),
        }
        probabilities = dict(base)
        probabilities["target_recency"] = target_response_probability(row, self.target_prior, current_target_game_index=self.target_game_index, base_probability=base["hierarchical_program"], config=self.adaptive_config)
        prior_probability = sum(self.response_expert_weights[name] * probabilities[name] for name in RESPONSE_EXPERTS)
        rejected_count = sum(offered_share <= share + 1e-9 for share in self.within_game_rejected_shares)
        accepted_count = sum(offered_share + 1e-9 >= share for share in self.within_game_accepted_shares)
        evidence_count = rejected_count + accepted_count
        if evidence_count:
            prior_strength = self.live_config.within_game_response_prior_equivalent_rows
            probability = (prior_strength * prior_probability + accepted_count) / (prior_strength + evidence_count)
        else:
            probability = prior_probability
        result = {
            "probability": _clamp(probability, 1e-6, 1 - 1e-6),
            "probability_before_within_game_monotone_overlay": _clamp(prior_probability, 1e-6, 1 - 1e-6),
            "within_game_monotone_evidence": {"accepted_lower_or_equal": accepted_count, "rejected_higher_or_equal": rejected_count},
            "expert_probabilities": probabilities,
            "expert_weights": dict(self.response_expert_weights),
        }
        self._response_cache[cache_key] = result
        return result

    def proposal(self, context: BargainingContext) -> tuple[GaussianMixture, dict[str, GaussianMixture]]:
        cached = self._proposal_cache.get(context)
        if cached is not None:
            return cached
        row = BargainingDecisionRow(context=context, action_type="proposal", proposal_share=0.5)
        hierarchical = GaussianMixture(
            tuple(
                GaussianComponent(particle.mean(context), particle.sigma, weight, identifier)
                for particle, weight, identifier in zip(self.proposal_programs, self.hierarchical_proposal_weights, self.proposal_program_ids, strict=True)
            )
        )
        recency = self.recency_model.proposal_prediction(row)
        tabular = self.ridge_model.prediction(row)
        distributions = {
            "hierarchical_program": hierarchical,
            "recency_kernel": GaussianMixture.single(mean=recency.mean, sigma=recency.sigma, source="recency-kernel"),
            "regularized_tabular": GaussianMixture.single(mean=tabular.mean, sigma=tabular.sigma, source="regularized-tabular"),
            "mode_kernel": mode_kernel_distribution(row, self.population_prior, self.target_prior, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index, config=self.adaptive_config),
        }
        result = (GaussianMixture.blend(distributions, self.proposal_expert_weights), distributions)
        self._proposal_cache[context] = result
        return result

    def incorporate(self, row: BargainingDecisionRow, config: LiveAdvisorConfig) -> _FrozenForecastBundle:
        """Return a new bundle after one visible opponent action without mutating terminal state."""
        response_weights = dict(self.response_expert_weights)
        proposal_weights = dict(self.proposal_expert_weights)
        hierarchical_response = self.hierarchical_response_weights
        hierarchical_proposal = self.hierarchical_proposal_weights
        rejected_shares = self.within_game_rejected_shares
        accepted_shares = self.within_game_accepted_shares
        if row.action_type == "response" and row.offered_share is not None and row.accepted is not None:
            forecast = self.response(row.context, float(row.offered_share))
            losses = {
                name: -math.log(float(probability) if row.accepted else 1 - float(probability))
                for name, probability in dict(forecast["expert_probabilities"]).items()
            }
            response_weights = _updated_expert_weights(response_weights, losses, adaptive_config=self.adaptive_config, live_config=config)
            hierarchical_response = tuple(_posterior_from_prior(hierarchical_response, _response_scores(self.response_programs, (row,))))
            if row.accepted:
                accepted_shares = (*accepted_shares, float(row.offered_share))
            else:
                rejected_shares = (*rejected_shares, float(row.offered_share))
        elif row.action_type == "proposal" and row.proposal_share is not None:
            _forecast, experts = self.proposal(row.context)
            losses = {name: distribution.nll(float(row.proposal_share)) for name, distribution in experts.items()}
            proposal_weights = _updated_expert_weights(proposal_weights, losses, adaptive_config=self.adaptive_config, live_config=config)
            hierarchical_proposal = tuple(_posterior_from_prior(hierarchical_proposal, _proposal_scores(self.proposal_programs, (row,))))
        observation = PriorObservation(row, self.global_game_index, self.target_game_index)
        target_prior = (*self.target_prior, observation)
        return replace(
            self,
            target_prior=target_prior,
            hierarchical_response_weights=hierarchical_response,
            hierarchical_proposal_weights=hierarchical_proposal,
            response_expert_weights=response_weights,
            proposal_expert_weights=proposal_weights,
            within_game_rejected_shares=rejected_shares,
            within_game_accepted_shares=accepted_shares,
            _response_cache={},
            _proposal_cache={},
        )

    def refit_local_baselines(self) -> _FrozenForecastBundle:
        """Refit tabular and recency baselines once after sequential local updates."""
        ordinary_population, ordinary_target = _ordinary_weights(self.population_prior, self.target_prior, self.validation_config)
        recency_population, recency_target = _recency_weights(self.population_prior, self.target_prior, current_global_game_index=self.global_game_index, current_target_game_index=self.target_game_index, config=self.validation_config)
        ordinary = ordinary_population + ordinary_target
        return replace(
            self,
            recency_model=RecencyKernelModel(recency_population + recency_target, self.validation_config),
            logistic_model=LogisticResponseModel.fit(ordinary, l2=self.validation_config.logistic_l2),
            ridge_model=RidgeProposalModel.fit(ordinary, l2=self.validation_config.ridge_l2),
            _response_cache={},
            _proposal_cache={},
        )


def _round(value: float) -> float:
    return round(float(value), 6)


def _distribution_modes(distribution: GaussianMixture, *, limit: int = 3) -> list[dict[str, float]]:
    candidates = sorted(((distribution.density(index / 100), index / 100) for index in range(101)), reverse=True)
    selected: list[dict[str, float]] = []
    for density, share in candidates:
        if all(abs(share - item["opponent_share"]) >= 0.05 for item in selected):
            selected.append({"opponent_share": _round(share), "relative_density": _round(density / candidates[0][0])})
            if len(selected) >= limit:
                break
    return selected


def _threshold(grid: Sequence[tuple[float, float]], probability: float) -> float | None:
    candidate = None
    suffix_minimum = 1.0
    for share, predicted in reversed(grid):
        suffix_minimum = min(suffix_minimum, predicted)
        if suffix_minimum >= probability:
            candidate = share
    return candidate


def _updated_expert_weights(prior: Mapping[str, float], losses: Mapping[str, float], *, adaptive_config: AdaptiveConfig, live_config: LiveAdvisorConfig) -> dict[str, float]:
    """Apply one bounded game-local generalized-Bayes update to fixed expert weights."""
    names = tuple(prior)
    if not names or any(name not in losses or not math.isfinite(float(losses[name])) for name in names):
        raise ValueError("within-game expert update requires one finite loss per expert")
    clipped = {name: _clamp(float(losses[name]), adaptive_config.expert_loss_floor, adaptive_config.expert_loss_ceiling) for name in names}
    minimum = min(clipped.values())
    raw = {name: max(float(prior[name]), 1e-300) * math.exp(-live_config.within_game_expert_learning_rate * (clipped[name] - minimum)) for name in names}
    total = sum(raw.values())
    uniform = 1.0 / len(names)
    return {name: (1 - live_config.within_game_uniform_mix) * raw[name] / total + live_config.within_game_uniform_mix * uniform for name in names}


@dataclass(frozen=True)
class _WithinGameEvidence:
    """Opponent-attributed actions recoverable from one authenticated live transcript."""

    rows: tuple[BargainingDecisionRow, ...]
    transcript_sha256: str
    current_offer_included: bool
    rejected_our_offer_shares: tuple[float, ...]
    accepted_our_offer_shares: tuple[float, ...]
    rejected_opponent_offer_our_shares: tuple[tuple[int, float], ...]
    opponent_proposal_shares: tuple[float, ...]
    current_offer_repetition_count: int
    our_latest_offer_plateau_length: int
    opponent_latest_offer_plateau_length: int

    def prompt_receipt(self) -> dict[str, object]:
        return {
            "method": "sequential-transcript-posterior",
            "source": "authenticated_history_plus_current_opponent_offer",
            "transcript_sha256": self.transcript_sha256,
            "turn_local_only": True,
            "restart_reconstruction": "deterministic-from-visible-transcript",
            "terminal_journal_mutated": False,
            "opponent_response_count": len(self.rejected_our_offer_shares) + len(self.accepted_our_offer_shares),
            "opponent_rejection_count": len(self.rejected_our_offer_shares),
            "our_rejection_count": len(self.rejected_opponent_offer_our_shares),
            "opponent_proposal_count": len(self.opponent_proposal_shares),
            "current_offer_included": self.current_offer_included,
            "current_offer_repetition_count": self.current_offer_repetition_count,
            "our_latest_offer_plateau_length": self.our_latest_offer_plateau_length,
            "opponent_latest_offer_plateau_length": self.opponent_latest_offer_plateau_length,
            "rejected_our_offer_opponent_shares": [_round(value) for value in self.rejected_our_offer_shares],
            "rejected_opponent_offer_our_shares": [{"round": round_number, "our_share": _round(share)} for round_number, share in self.rejected_opponent_offer_our_shares],
            "observed_opponent_proposal_shares": [_round(value) for value in self.opponent_proposal_shares],
        }


def _trailing_share_repetitions(values: Sequence[float]) -> int:
    """Count the exact-tolerance suffix run of one player's numeric offers."""
    if not values:
        return 0
    latest = float(values[-1])
    count = 0
    for value in reversed(values):
        if not math.isclose(float(value), latest, rel_tol=0.0, abs_tol=1e-6):
            break
        count += 1
    return count


def _live_opponent_evidence(game: dict[str, Any], *, opponent_id: str, opponent_name: str) -> _WithinGameEvidence:
    """Extract only already-visible opponent actions, including an unresolved current opponent offer."""
    state = game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("live bargaining game has no state")
    history = state.get("history")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ValueError("live bargaining history is not a list")
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    opponent_player = _other_player(our_player)
    rows: list[BargainingDecisionRow] = []
    prior: list[dict[str, Any]] = []

    def proposal_row(offer: dict[str, Any], context: BargainingContext, *, source: str) -> BargainingDecisionRow:
        message = str(offer.get("message") or "")
        return BargainingDecisionRow(context=context, action_type="proposal", proposal_share=_clamp(_gain(offer, opponent_player) / context.money_to_divide, 0.0, 1.0), message=message, message_act=classify_message_act(message, messages_allowed=context.messages_allowed), job_id=source, job_path="live-transcript", job_sha256="")

    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ValueError("live bargaining history contains a malformed entry")
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        if offer is None:
            raise ValueError("live bargaining history entry has no offer")
        proposer = str(entry.get("proposer") or offer.get("proposer") or "")
        if proposer not in {our_player, opponent_player}:
            raise ValueError("live bargaining history has an unknown proposer")
        round_number = int(entry.get("round") or offer.get("round") or index + 1)
        context = _history_context(game_id=str(game.get("game_id") or "live"), opponent_id=opponent_id, opponent_name=opponent_name, completed_at="", completion_order=0, state=state, our_player=our_player, opponent_player=opponent_player, round_number=round_number, prior=prior)
        decision = str(entry.get("decision") or "").casefold()
        if proposer == opponent_player:
            rows.append(proposal_row(offer, context, source="within-game-history-proposal"))
        elif decision in {"accept", "reject", "walkaway"}:
            rows.append(BargainingDecisionRow(context=context, action_type="response", offered_share=_clamp(_gain(offer, opponent_player) / context.money_to_divide, 0.0, 1.0), accepted=decision == "accept", decision=decision, response_time_ms=_finite(entry.get("response_time_ms")), job_id="within-game-history-response", job_path="live-transcript", job_sha256=""))
        prior.append(entry)

    current_offer_included = False
    current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
    if game.get("valid_actions", {}).get("type") == "decision" and current_offer is not None:
        proposer = str(current_offer.get("proposer") or state.get("proposer") or "")
        round_number = int(current_offer.get("round") or state.get("round") or len(history) + 1)
        represented = any(isinstance(entry, dict) and int(entry.get("round") or (entry.get("offer") or {}).get("round") or -1) == round_number for entry in history)
        if proposer == opponent_player and not represented:
            context = _history_context(game_id=str(game.get("game_id") or "live"), opponent_id=opponent_id, opponent_name=opponent_name, completed_at="", completion_order=0, state=state, our_player=our_player, opponent_player=opponent_player, round_number=round_number, prior=prior)
            rows.append(proposal_row(current_offer, context, source="within-game-current-proposal"))
            current_offer_included = True

    rejected = tuple(float(row.offered_share) for row in rows if row.action_type == "response" and row.accepted is False and row.offered_share is not None)
    accepted = tuple(float(row.offered_share) for row in rows if row.action_type == "response" and row.accepted is True and row.offered_share is not None)
    rejected_opponent_offers: list[tuple[int, float]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or str(entry.get("decision") or "").casefold() != "reject":
            continue
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        proposer = str(entry.get("proposer") or (offer or {}).get("proposer") or "")
        if offer is None or proposer != opponent_player:
            continue
        round_number = int(entry.get("round") or offer.get("round") or index + 1)
        rejected_opponent_offers.append((round_number, _clamp(_gain(offer, our_player) / float(state["money_to_divide"]), 0.0, 1.0)))
    proposals = tuple(float(row.proposal_share) for row in rows if row.action_type == "proposal" and row.proposal_share is not None)
    current_share = proposals[-1] if current_offer_included and proposals else None
    repetitions = sum(math.isclose(value, current_share, rel_tol=0.0, abs_tol=1e-6) for value in proposals) if current_share is not None else 0
    our_plateau = _trailing_share_repetitions(rejected)
    opponent_plateau = _trailing_share_repetitions(proposals)
    transcript = {"history": history, "current_opponent_offer": current_offer if current_offer_included else None}
    return _WithinGameEvidence(rows=tuple(rows), transcript_sha256=_sha(transcript), current_offer_included=current_offer_included, rejected_our_offer_shares=rejected, accepted_our_offer_shares=accepted, rejected_opponent_offer_our_shares=tuple(rejected_opponent_offers), opponent_proposal_shares=proposals, current_offer_repetition_count=repetitions, our_latest_offer_plateau_length=our_plateau, opponent_latest_offer_plateau_length=opponent_plateau)


def _identity(game: dict[str, Any]) -> tuple[str, str, bool]:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    name = normalize_opponent_name(opponent.get("name"))
    if name and str(opponent.get("type") or "agent") != "hidden":
        return named_opponent_id(name), name, True
    game_id = str(game.get("game_id") or "unknown")
    return f"hidden-{hashlib.sha256(game_id.encode('utf-8')).hexdigest()[:20]}", "hidden opponent", False


def _candidate_shares(config: LiveAdvisorConfig, evidence: _WithinGameEvidence) -> tuple[float, ...]:
    values = {0.5, *CURVE_SHARES, *evidence.rejected_our_offer_shares, *evidence.opponent_proposal_shares}
    value = config.continuation_candidate_step
    while value < 1.0 - 1e-12:
        values.add(value)
        value += config.continuation_candidate_step
    return tuple(sorted(_clamp(item, 0.001, 0.999) for item in values if 0 < item < 1))


@dataclass(frozen=True)
class _BehavioralRollout:
    """A conservative one-counteroffer calculation under one frozen game-local posterior."""

    bundle: _FrozenForecastBundle
    context: BargainingContext
    evidence: _WithinGameEvidence
    config: LiveAdvisorConfig

    def _discount(self, round_number: int) -> float:
        discount = self.context.our_discount
        if discount is None or not math.isfinite(discount) or not 0 <= discount <= 1:
            discount = 1.0
        return discount ** max(0, round_number - 1)

    def _effective_round_cap(self) -> tuple[int, str]:
        if self.context.horizon_known and self.context.max_rounds is not None:
            return self.context.max_rounds, "authenticated-game-horizon"
        return self.config.environmental_round_cap, "observed-platform-termination"

    def _round_available(self, round_number: int) -> bool:
        cap, _source = self._effective_round_cap()
        return round_number <= cap

    def _patient_self(self) -> bool:
        return self.context.our_discount is not None and math.isclose(self.context.our_discount, 1.0, rel_tol=0.0, abs_tol=1e-12)

    @staticmethod
    def _rubinstein_opening_opponent_share(offer_context: BargainingContext) -> float | None:
        """Return the stationary opponent share when the authenticated opening has a unique solution."""
        if offer_context.round_number != 1 or not offer_context.complete_information or offer_context.horizon_known:
            return None
        own_discount = offer_context.our_discount
        opponent_discount = offer_context.opponent_discount
        if own_discount is None or opponent_discount is None or not 0 <= own_discount <= 1 or not 0 <= opponent_discount <= 1:
            return None
        denominator = 1 - own_discount * opponent_discount
        if denominator <= 1e-12:
            return None
        return _clamp(opponent_discount * (1 - own_discount) / denominator, 0.0, 1.0)

    def _final_round(self, round_number: int) -> bool:
        return not self._round_available(round_number + 1)

    def _environmental_terminal_window(self, round_number: int) -> bool:
        cap, source = self._effective_round_cap()
        return source == "observed-platform-termination" and cap - round_number < self.config.environmental_terminal_window_rounds

    def _terminal_window(self, round_number: int) -> bool:
        cap, _source = self._effective_round_cap()
        return cap - round_number < self.config.environmental_terminal_window_rounds

    def _rejected_offer_reservation(self, round_number: int) -> dict[str, object]:
        if not self.evidence.rejected_opponent_offer_our_shares:
            return {"status": "not-applicable", "authority": "rejected-offer-regret-diagnostic", "active_constraint": False}
        candidates = [
            {
                "round": rejected_round,
                "our_share": share,
                "discounted_value": self._discount(rejected_round) * share,
            }
            for rejected_round, share in self.evidence.rejected_opponent_offer_our_shares
        ]
        best = max(candidates, key=lambda value: (float(value["discounted_value"]), -int(value["round"])))
        current_discount = self._discount(round_number)
        required_share = float(best["discounted_value"]) / current_discount if current_discount > 0 else math.inf
        recoverable = math.isfinite(required_share) and required_share <= 1 + self.config.consistency_share_tolerance
        return {
            "status": "available" if recoverable else "historical-sunk",
            "authority": "discounted-rejected-value-recovery-bound" if recoverable else "rejected-offer-regret-diagnostic",
            "source_round": int(best["round"]),
            "source_our_share": _round(float(best["our_share"])),
            "reserved_discounted_value": _round(float(best["discounted_value"])),
            "current_round_discount": _round(current_discount),
            "minimum_current_our_share": _round(required_share) if math.isfinite(required_share) else None,
            "rejected_offer_count": len(candidates),
            "recoverable_at_current_round": recoverable,
            "active_constraint": recoverable,
            "interpretation": "A previously rejected opponent offer supplies a discounted-value bound only while an equivalent current split remains feasible. Once discounting makes that value infeasible, it remains regret evidence rather than a fictitious reservation right.",
        }

    def _offer_floor(self, offer_context: BargainingContext) -> float:
        if self._patient_self():
            return self.config.patient_minimum_opponent_share
        floor = self.config.opening_minimum_opponent_share if offer_context.round_number <= 1 else self.config.later_minimum_opponent_share
        rubinstein_share = self._rubinstein_opening_opponent_share(offer_context)
        if rubinstein_share is not None:
            floor = min(floor, rubinstein_share)
        if self.evidence.rejected_our_offer_shares:
            floor = max(floor, max(self.evidence.rejected_our_offer_shares))
        return min(floor, self.config.maximum_guarded_opponent_share)

    def _reciprocal_concession_control(self, offer_context: BargainingContext) -> dict[str, object]:
        proposals = self.evidence.opponent_proposal_shares
        previous_own = offer_context.previous_our_offer_to_opponent_share
        if len(proposals) < 2 or previous_own is None:
            return {"status": "insufficient-visible-movement", "authority": "bilateral-concession-bound"}
        prior_opponent = float(proposals[-2])
        latest_opponent = float(proposals[-1])
        concession_toward_us = max(0.0, prior_opponent - latest_opponent)
        raw_cap = min(1.0, max(0.0, float(previous_own) + self.config.reciprocal_concession_match_ratio * concession_toward_us))
        floor = self._offer_floor(offer_context)
        return {
            "status": "available",
            "authority": "bilateral-concession-bound",
            "previous_own_offer_opponent_share": _round(float(previous_own)),
            "prior_opponent_demand_share": _round(prior_opponent),
            "latest_opponent_demand_share": _round(latest_opponent),
            "latest_opponent_concession_toward_us": _round(concession_toward_us),
            "match_ratio": _round(self.config.reciprocal_concession_match_ratio),
            "maximum_next_opponent_share": _round(raw_cap),
            "inherited_minimum_opponent_share": _round(floor),
            "effective_minimum_opponent_share": _round(min(floor, raw_cap)),
            "plateau_or_adverse_move": concession_toward_us <= 1e-12,
        }

    def _patient_reciprocal_concession_window_control(self, current_our_share: float) -> dict[str, object]:
        """Detect a live bilateral-concession window before either side retracts its latest movement."""

        tolerance = self.config.consistency_share_tolerance

        def trailing_movement(values: Sequence[float], *, direction: float) -> tuple[int, float, tuple[float, ...]]:
            if not values:
                return 0, 0.0, ()
            start = len(values) - 1
            material_steps = 0
            for index in range(len(values) - 1, 0, -1):
                movement = direction * (float(values[index]) - float(values[index - 1]))
                if movement < -tolerance:
                    break
                start = index - 1
                if movement > tolerance:
                    material_steps += 1
            window = tuple(float(value) for value in values[start:])
            total = max(0.0, direction * (window[-1] - window[0])) if window else 0.0
            return material_steps, total, window

        our_offers = self.evidence.rejected_our_offer_shares
        opponent_demands = self.evidence.opponent_proposal_shares
        our_steps, our_movement, our_window = trailing_movement(our_offers, direction=1.0)
        opponent_steps, opponent_movement, opponent_window = trailing_movement(opponent_demands, direction=-1.0)
        cap, cap_source = self._effective_round_cap()
        patient_open_horizon = self._patient_self() and cap_source == "observed-platform-termination" and self.context.round_number < cap
        known_opponent_discount = self.context.opponent_discount if self.context.complete_information else None
        opponent_discount_allows_window = known_opponent_discount is None or math.isclose(known_opponent_discount, 1.0, rel_tol=0.0, abs_tol=1e-12)
        below_ordinary_floor = current_our_share + tolerance < self.config.patient_minimum_own_settlement_share
        minimum_movement = self.config.patient_reciprocal_window_minimum_bilateral_movement
        checks = {
            "patient_open_horizon": patient_open_horizon,
            "opponent_discount_not_known_below_one": opponent_discount_allows_window,
            "current_offer_is_live": self.evidence.current_offer_included,
            "below_ordinary_patient_floor": below_ordinary_floor,
            "guarded_current_share": current_our_share + 1e-12 >= self.config.patient_reciprocal_window_minimum_own_share,
            "our_material_concession": our_steps >= 1 and our_movement + 1e-12 >= minimum_movement,
            "opponent_sustained_concession": opponent_steps >= self.config.patient_reciprocal_window_minimum_opponent_concessions and opponent_movement + 1e-12 >= minimum_movement,
        }
        active = all(checks.values())
        return {
            "status": "active" if active else "inactive",
            "authority": "patient-reciprocal-concession-window",
            "current_round": self.context.round_number,
            "round_cap_source": cap_source,
            "opponent_discount": _round(known_opponent_discount) if known_opponent_discount is not None else None,
            "current_offer_our_share": _round(current_our_share),
            "minimum_current_our_share": _round(self.config.patient_reciprocal_window_minimum_own_share),
            "ordinary_patient_settlement_floor": _round(self.config.patient_minimum_own_settlement_share),
            "minimum_bilateral_movement": _round(minimum_movement),
            "minimum_opponent_concessions": self.config.patient_reciprocal_window_minimum_opponent_concessions,
            "our_trailing_offer_opponent_shares": [_round(value) for value in our_window],
            "our_material_concession_steps": our_steps,
            "our_total_concession": _round(our_movement),
            "opponent_trailing_own_demand_shares": [_round(value) for value in opponent_window],
            "opponent_material_concession_steps": opponent_steps,
            "opponent_total_concession_toward_us": _round(opponent_movement),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "interpretation": "A patient open-horizon settlement below the ordinary 60% floor is authorized only when the opponent's discount is unknown or one, both parties' latest numeric paths remain non-retracting, each side has moved at least 5 percentage points, the opponent has made at least 2 material concessions, and the current offer gives us at least 45%. A known discounted opponent remains under time pressure until the separate terminal-cap policy applies. A plateau preserves the live window; a reversal closes it.",
        }

    def _rejected_value_recovery_control(self, offer_context: BargainingContext) -> dict[str, object]:
        reservation = self._rejected_offer_reservation(offer_context.round_number)
        required_share = reservation.get("minimum_current_our_share")
        active = reservation.get("status") == "available" and isinstance(required_share, (int, float))
        return {
            **reservation,
            "authority": "discounted-rejected-value-recovery-bound" if active else "rejected-offer-regret-diagnostic",
            "historical_current_share_equivalent": required_share,
            "minimum_current_offer_our_share": required_share if active else None,
            "maximum_current_offer_opponent_share": _round(1.0 - float(required_share)) if active else None,
        }

    def _loss_minimization_control(self, offer_context: BargainingContext) -> dict[str, object]:
        cap, cap_source = self._effective_round_cap()
        remaining_rounds = max(0, cap - offer_context.round_number)
        next_round_retention = self._discount(offer_context.round_number + 1)
        environmental_cap = cap_source == "observed-platform-termination"
        discounted_trigger = environmental_cap and not self._patient_self() and next_round_retention <= self.config.loss_minimization_discount_retention + 1e-12
        terminal_trigger = self._terminal_window(offer_context.round_number)
        minimum = self.config.loss_minimization_minimum_bilateral_rejections
        bilateral_rejections = len(self.evidence.rejected_our_offer_shares) >= minimum and len(self.evidence.opponent_proposal_shares) >= minimum
        latest_opponent_demand = float(self.evidence.opponent_proposal_shares[-1]) if self.evidence.opponent_proposal_shares else None
        deadlock_minimum = self.config.patient_deadlock_minimum_repetitions_per_player
        patient_deadlock_trigger = environmental_cap and self._patient_self() and latest_opponent_demand is not None and latest_opponent_demand <= 1 - self.config.minimum_material_own_share + 1e-12 and self.evidence.our_latest_offer_plateau_length >= deadlock_minimum and self.evidence.opponent_latest_offer_plateau_length >= deadlock_minimum
        current_retention = self._discount(offer_context.round_number)
        one_round_discount_loss = max(0.0, current_retention - next_round_retention)
        no_progress_minimum = self.config.economic_no_progress_minimum_plateau
        economic_no_progress_trigger = environmental_cap and not self._patient_self() and bilateral_rejections and latest_opponent_demand is not None and self.evidence.our_latest_offer_plateau_length >= no_progress_minimum and self.evidence.opponent_latest_offer_plateau_length >= no_progress_minimum and one_round_discount_loss + 1e-12 >= self.config.economic_no_progress_minimum_discount_loss
        active = (terminal_trigger and latest_opponent_demand is not None) or (discounted_trigger and bilateral_rejections and latest_opponent_demand is not None) or patient_deadlock_trigger or economic_no_progress_trigger
        result: dict[str, object] = {
            "status": "active" if active else "inactive",
            "authority": "cap-aware-revealed-settlement",
            "environmental_round_cap": cap,
            "round_cap_source": cap_source,
            "current_round": offer_context.round_number,
            "remaining_rounds_after_current": remaining_rounds,
            "next_round_discount_retention": _round(next_round_retention),
            "discount_retention_trigger": _round(self.config.loss_minimization_discount_retention),
            "discounted_trigger": discounted_trigger,
            "economic_no_progress_trigger": economic_no_progress_trigger,
            "one_round_discount_loss": _round(one_round_discount_loss),
            "economic_no_progress_minimum_discount_loss": _round(self.config.economic_no_progress_minimum_discount_loss),
            "economic_no_progress_minimum_plateau": no_progress_minimum,
            "patient_deadlock_trigger": patient_deadlock_trigger,
            "terminal_window_trigger": terminal_trigger,
            "terminal_window_source": cap_source if terminal_trigger else None,
            "minimum_bilateral_rejections": minimum,
            "bilateral_rejections_observed": bilateral_rejections,
            "patient_deadlock_minimum_repetitions_per_player": deadlock_minimum,
            "our_latest_offer_plateau_length": self.evidence.our_latest_offer_plateau_length,
            "opponent_latest_offer_plateau_length": self.evidence.opponent_latest_offer_plateau_length,
            "latest_opponent_revealed_own_share": _round(latest_opponent_demand) if latest_opponent_demand is not None else None,
        }
        if not active or latest_opponent_demand is None:
            return result
        trigger_round = cap - self.config.environmental_terminal_window_rounds + 1
        if discounted_trigger:
            trigger_round = next((round_number for round_number in range(1, cap + 1) if self._discount(round_number + 1) <= self.config.loss_minimization_discount_retention + 1e-12), trigger_round)
        prior_escape_offers = sum(
            1
            for row in self.evidence.rows
            if row.action_type == "response" and row.accepted is False and row.context.round_number >= trigger_round
        )
        previous_opponent_share = float(self.evidence.rejected_our_offer_shares[-1]) if self.evidence.rejected_our_offer_shares else self.config.opening_minimum_opponent_share
        minimum_own_share = self.config.environmental_minimum_positive_own_share if terminal_trigger else self.config.minimum_material_own_share
        revealed_target = min(1 - minimum_own_share, max(previous_opponent_share, latest_opponent_demand))
        if patient_deadlock_trigger and not terminal_trigger:
            stage = "patient-deadlock-settlement"
            target = latest_opponent_demand
        elif terminal_trigger or prior_escape_offers:
            stage = "revealed-settlement"
            target = revealed_target
        else:
            stage = "bridge"
            target = previous_opponent_share + self.config.loss_minimization_bridge_fraction * max(0.0, revealed_target - previous_opponent_share)
        result.update(
            {
                "stage": stage,
                "trigger_round": trigger_round,
                "prior_escape_offer_count": prior_escape_offers,
                "previous_rejected_offer_opponent_share": _round(previous_opponent_share),
                "bridge_fraction": _round(self.config.loss_minimization_bridge_fraction),
                "minimum_positive_own_share": _round(minimum_own_share),
                "target_opponent_share": _round(_clamp(target, 0.001, 1 - minimum_own_share)),
                "interpretation": "After bilateral rejection consumes half the discounted value under the observed platform cap, bridge the numeric gap once and then reproduce the opponent's revealed split. A patient environmental-horizon game settles after both players repeat unchanged offers for the evidence-derived deadlock threshold. Inside the final 2 executable rounds of either an authenticated or observed cap, reproduce the revealed split immediately rather than risk zero.",
            }
        )
        return result

    def _offer_controls(self, offer_context: BargainingContext) -> dict[str, object]:
        floor = self._offer_floor(offer_context)
        reciprocal = self._reciprocal_concession_control(offer_context)
        recovery = self._rejected_value_recovery_control(offer_context)
        loss_minimization = self._loss_minimization_control(offer_context)
        policy_regime = "patient" if self._patient_self() else "discounted"
        if loss_minimization.get("status") == "active":
            target = float(loss_minimization["target_opponent_share"])
            return {
                "policy_regime": policy_regime,
                "minimum_opponent_share": _round(target),
                "maximum_opponent_share": _round(target),
                "minimum_material_own_share": _round(self.config.minimum_material_own_share),
                "reciprocal_concession_control": reciprocal,
                "rejected_value_recovery_control": recovery,
                "loss_minimization_control": loss_minimization,
            }
        caps = [self.config.patient_maximum_opponent_share if self._patient_self() else 1 - self.config.minimum_material_own_share]
        if reciprocal.get("status") == "available":
            caps.append(float(reciprocal["maximum_next_opponent_share"]))
        if recovery.get("status") == "available" and isinstance(recovery.get("maximum_current_offer_opponent_share"), (int, float)):
            caps.append(float(recovery["maximum_current_offer_opponent_share"]))
        ceiling = max(0.001, min(caps))
        return {
            "policy_regime": policy_regime,
            "minimum_opponent_share": _round(min(floor, ceiling)),
            "maximum_opponent_share": _round(ceiling),
            "minimum_material_own_share": _round(self.config.minimum_material_own_share),
            "reciprocal_concession_control": reciprocal,
            "rejected_value_recovery_control": recovery,
            "loss_minimization_control": loss_minimization,
        }

    def evaluate_offer(self, offer_context: BargainingContext, opponent_share: float) -> dict[str, object]:
        share = _clamp(opponent_share, self.config.environmental_minimum_positive_own_share, 1 - self.config.environmental_minimum_positive_own_share)
        response = self.bundle.response(offer_context, share)
        raw_probability = float(response["probability"])
        probability = max(0.0, raw_probability - self.config.response_probability_haircut)
        accepted_value = self._discount(offer_context.round_number) * (1 - share)
        counterproposal_mean = None
        counterproposal_risk_share = None
        rejected_path_value = 0.0
        if self._round_available(offer_context.round_number + 1):
            rejection_context = replace(offer_context, round_number=offer_context.round_number + 1, previous_our_offer_to_opponent_share=share, previous_opponent_response="reject")
            proposal, _experts = self.bundle.proposal(rejection_context)
            counterproposal_mean = proposal.mean
            counterproposal_risk_share = proposal.quantile(self.config.counterproposal_risk_quantile)
            rejected_path_value = self._discount(rejection_context.round_number) * (1 - counterproposal_risk_share)
        expected_value = probability * accepted_value + (1 - probability) * rejected_path_value
        return {
            "opponent_share": _round(share),
            "our_nominal_share": _round(1 - share),
            "opponent_accept_probability_raw": _round(raw_probability),
            "opponent_accept_probability_conservative": _round(probability),
            "response_probability_haircut": _round(self.config.response_probability_haircut),
            "accepted_value": _round(accepted_value),
            "rejected_path_value": _round(rejected_path_value),
            "conditional_counterproposal_opponent_share_mean": _round(counterproposal_mean) if counterproposal_mean is not None else None,
            "conditional_counterproposal_opponent_share_risk_quantile": _round(counterproposal_risk_share) if counterproposal_risk_share is not None else None,
            "counterproposal_risk_quantile": _round(self.config.counterproposal_risk_quantile),
            "rejection_path_commitment": "accept-the-modeled-risk-quantile-counterproposal-immediately" if counterproposal_risk_share is not None else "no-later-round-available",
            "expected_value": _round(expected_value),
        }

    def best_offer(self, offer_context: BargainingContext) -> dict[str, object] | None:
        if not self._round_available(offer_context.round_number):
            return None
        controls = self._offer_controls(offer_context)
        floor = float(controls["minimum_opponent_share"])
        ceiling = float(controls["maximum_opponent_share"])
        recovery_share = controls["rejected_value_recovery_control"].get("maximum_current_offer_opponent_share")
        candidate_shares = tuple(sorted({*_candidate_shares(self.config, self.evidence), *(() if not isinstance(recovery_share, (int, float)) else (float(recovery_share),)), _clamp(floor, self.config.environmental_minimum_positive_own_share, 1 - self.config.environmental_minimum_positive_own_share), _clamp(ceiling, self.config.environmental_minimum_positive_own_share, 1 - self.config.environmental_minimum_positive_own_share)}))
        guarded = [self.evaluate_offer(offer_context, share) for share in candidate_shares if share + 1e-9 >= floor and share <= ceiling + 1e-9]
        credible = [candidate for candidate in guarded if float(candidate["opponent_accept_probability_raw"]) >= self.config.minimum_offer_accept_probability]
        eligible = credible or guarded
        best = max(eligible, key=lambda value: (float(value["expected_value"]), float(value["opponent_accept_probability_conservative"]), float(value["our_nominal_share"])))
        concession_control: dict[str, object] = {"status": "not-applicable", "reason": "the opponent has not rejected one of our offers"}
        loss_minimization = controls["loss_minimization_control"]
        if loss_minimization.get("status") == "active":
            concession_control = {
                "status": "authorized-by-loss-minimization",
                "candidate_opponent_share": best["opponent_share"],
                "concession_authorized": True,
                "reason": "the cap-aware bridge or revealed-settlement stage supersedes the ordinary concession hold",
            }
        elif self.evidence.rejected_our_offer_shares:
            standing_share = _clamp(float(self.evidence.rejected_our_offer_shares[-1]), floor, ceiling)
            standing = self.evaluate_offer(offer_context, standing_share)
            proposed_concession = float(best["opponent_share"]) - standing_share
            expected_value_gain = float(best["expected_value"]) - float(standing["expected_value"])
            authorized = proposed_concession <= self.config.consistency_share_tolerance or expected_value_gain >= self.config.post_rejection_concession_minimum_value_gain - 1e-12
            concession_control = {
                "status": "available",
                "latest_rejected_offer_opponent_share": _round(standing_share),
                "candidate_opponent_share": best["opponent_share"],
                "candidate_concession": _round(max(0.0, proposed_concession)),
                "candidate_expected_value_gain": _round(expected_value_gain),
                "minimum_expected_value_gain_for_concession": _round(self.config.post_rejection_concession_minimum_value_gain),
                "concession_authorized": authorized,
            }
            if not authorized:
                best = standing
        maximum_policy_share = min(ceiling, float(best["opponent_share"])) if self.evidence.rejected_our_offer_shares else ceiling
        return {
            **best,
            "round": offer_context.round_number,
            "candidate_count": len(candidate_shares),
            "guarded_candidate_count": len(guarded),
            "credible_candidate_count": len(credible),
            "minimum_opponent_share": _round(floor),
            "maximum_opponent_share": _round(ceiling),
            "maximum_policy_opponent_share": _round(maximum_policy_share),
            "post_rejection_concession_control": concession_control,
            "policy_regime": controls["policy_regime"],
            "minimum_material_own_share": controls["minimum_material_own_share"],
            "reciprocal_concession_control": controls["reciprocal_concession_control"],
            "rejected_value_recovery_control": controls["rejected_value_recovery_control"],
            "loss_minimization_control": loss_minimization,
            "minimum_raw_accept_probability": _round(self.config.minimum_offer_accept_probability),
            "accept_probability_constraint_relaxed": not bool(credible),
        }

    def projection(self, game: dict[str, Any], *, evidence_tier: str, ood_flags: Sequence[str]) -> dict[str, object]:
        state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
        action_type = str(game.get("valid_actions", {}).get("type") or "")
        discount = self.context.our_discount
        patient_continuation = discount is not None and math.isclose(discount, 1.0, rel_tol=0.0, abs_tol=1e-12) and self._round_available(self.context.round_number + 1)
        effective_cap, cap_source = self._effective_round_cap()
        result: dict[str, object] = {
            "method": "conservative-option-preserving-one-counteroffer-rollout" if action_type == "decision" and patient_continuation else "conservative-committed-one-counteroffer-rollout",
            "value_units": "fraction-of-initial-pool",
            "own_discount": _round(discount) if discount is not None else None,
            "missing_own_discount_assumption": "unit-discount" if discount is None else None,
            "known_horizon_respected": True,
            "effective_round_cap": effective_cap,
            "round_cap_source": cap_source,
            "remaining_rounds_after_current": max(0, effective_cap - self.context.round_number),
            "environmental_constraint": "The platform terminated a nominally unlimited observed game after round 99; v2.15 treats that reproducible boundary as an exogenous loss constraint without relabeling the official horizon as known, while authenticated finite horizons retain their stated caps.",
            "free_text_message_effects_modeled": False,
            "rejection_path_is_a_committed_acceptance": not (action_type == "decision" and patient_continuation),
        }
        if action_type == "offer":
            modeled_offer = self.best_offer(self.context)
            result["modeled_offer_policy"] = modeled_offer
            result["offer_expected_value_regression_control"] = {
                "status": "available" if modeled_offer is not None else "unavailable",
                "reference_opponent_share": modeled_offer.get("opponent_share") if modeled_offer is not None else None,
                "reference_expected_value": modeled_offer.get("expected_value") if modeled_offer is not None else None,
                "maximum_expected_value_regression": _round(self.config.offer_expected_value_regression_margin),
                "authority": "replace a non-analytic offer only when its advisor-estimated expected value falls more than the stated margin below the frozen behavioral candidate",
            }
            reciprocal_control = modeled_offer.get("reciprocal_concession_control") if modeled_offer is not None else {"status": "unavailable"}
            recovery_control = modeled_offer.get("rejected_value_recovery_control") if modeled_offer is not None else {"status": "unavailable"}
            loss_minimization_control = modeled_offer.get("loss_minimization_control") if modeled_offer is not None else {"status": "unavailable"}
            result["reciprocal_concession_control"] = reciprocal_control
            result["rejected_value_recovery_control"] = recovery_control
            result["loss_minimization_control"] = loss_minimization_control
            result["policy_guard"] = {
                "mode": "bounded-authoritative",
                "policy_regime": modeled_offer.get("policy_regime") if modeled_offer is not None else None,
                "minimum_opponent_share": modeled_offer.get("minimum_opponent_share") if modeled_offer is not None else None,
                "maximum_opponent_share": modeled_offer.get("maximum_policy_opponent_share") if modeled_offer is not None else None,
                "structural_maximum_opponent_share": modeled_offer.get("maximum_opponent_share") if modeled_offer is not None else None,
                "minimum_material_own_share": modeled_offer.get("minimum_material_own_share") if modeled_offer is not None else None,
                "recommended_opponent_share": modeled_offer.get("opponent_share") if modeled_offer is not None else None,
                "post_rejection_concession_control": modeled_offer.get("post_rejection_concession_control") if modeled_offer is not None else None,
                "loss_minimization_control": loss_minimization_control,
                "authority": "keep a non-analytic offer inside the regime-specific and reciprocal-concession bounds until cap-aware loss minimization selects a bridge or revealed-settlement split",
            }
            if recovery_control.get("status") == "available":
                result["consistency_check"] = {
                    "status": "recoverable-discounted-bound",
                    "failure_mode": None,
                    "reservation_source_round": recovery_control.get("source_round"),
                    "reserved_discounted_value": recovery_control.get("reserved_discounted_value"),
                    "minimum_current_offer_our_share": recovery_control.get("minimum_current_offer_our_share"),
                    "maximum_current_offer_opponent_share": recovery_control.get("maximum_current_offer_opponent_share"),
                    "active_constraint": True,
                    "interpretation": "The best rejected opponent offer remains economically recoverable, so the next offer preserves at least its discounted current-value equivalent.",
                }
            elif recovery_control.get("status") == "historical-sunk":
                result["consistency_check"] = {
                    "status": "historical-regret-only",
                    "failure_mode": None,
                    "reservation_source_round": recovery_control.get("source_round"),
                    "reserved_discounted_value": recovery_control.get("reserved_discounted_value"),
                    "historical_current_share_equivalent": recovery_control.get("historical_current_share_equivalent"),
                    "active_constraint": False,
                    "interpretation": "The strongest discounted value previously rejected is reported as regret evidence only. It is sunk and cannot force a less valuable rejection or reverse the current concession path.",
                }
            else:
                result["consistency_check"] = {"status": "not-applicable", "reason": "we have not rejected an opponent offer in this game"}
            return result
        if action_type != "decision":
            result["consistency_check"] = {"status": "not-applicable", "reason": "unsupported bargaining action type"}
            return result
        current_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
        if current_offer is None:
            result["consistency_check"] = {"status": "unavailable", "reason": "current opponent offer is absent"}
            return result
        opponent_share = _clamp(_gain(current_offer, self.context.opponent_player) / self.context.money_to_divide, 0.0, 1.0)
        our_share = 1 - opponent_share
        loss_minimization_control = self._loss_minimization_control(self.context)
        patient_deadlock_accept = loss_minimization_control.get("status") == "active" and loss_minimization_control.get("stage") == "patient-deadlock-settlement" and our_share + self.config.consistency_share_tolerance >= self.config.minimum_material_own_share
        reciprocal_window_control = self._patient_reciprocal_concession_window_control(our_share)
        patient_reciprocal_window_accept = reciprocal_window_control.get("status") == "active"
        result["loss_minimization_control"] = loss_minimization_control
        result["patient_reciprocal_concession_window_control"] = reciprocal_window_control
        accept_value = self._discount(self.context.round_number) * our_share
        next_context = replace(self.context, round_number=self.context.round_number + 1, previous_opponent_offer_share=opponent_share, previous_our_response="reject")
        next_offer = self.best_offer(next_context)
        continuation_options: list[dict[str, object]] = []
        if next_offer is not None:
            continuation_options.append({"kind": "optimized-modeled-offer", "value": float(next_offer["expected_value"]), "offer": next_offer})
        replication_offer = self.evaluate_offer(next_context, opponent_share) if patient_continuation else None
        if replication_offer is not None:
            continuation_options.append({"kind": "reproduce-current-deal", "value": float(replication_offer["expected_value"]), "offer": replication_offer})
        opponent_patient = self.context.opponent_discount is not None and math.isclose(self.context.opponent_discount, 1.0, rel_tol=0.0, abs_tol=1e-12)
        revealed_willingness_floor = accept_value if patient_continuation and opponent_patient else None
        if revealed_willingness_floor is not None:
            continuation_options.append({"kind": "revealed-willingness-current-deal-floor", "value": revealed_willingness_floor, "offer": replication_offer})
        selected_continuation = max(continuation_options, key=lambda value: float(value["value"])) if continuation_options else None
        reject_value = float(selected_continuation["value"]) if selected_continuation is not None else 0.0
        probe_count = len(self.evidence.rejected_our_offer_shares)
        margin = 0.0 if patient_continuation else self.config.repeated_probe_value_margin if probe_count >= self.config.consistency_min_rejected_offers else self.config.first_probe_value_margin
        final_round = self._final_round(self.context.round_number)
        environmental_terminal = self._environmental_terminal_window(self.context.round_number)
        reservation = self._rejected_offer_reservation(self.context.round_number)
        discounted_self = discount is not None and discount < 1 - 1e-12
        settlement_floor = self.config.minimum_material_own_share
        policy_regime = "patient" if self._patient_self() else "discounted" if discounted_self else "unknown"
        if self._patient_self() and not environmental_terminal and not patient_deadlock_accept:
            settlement_floor = self.config.patient_minimum_own_settlement_share
        if environmental_terminal and our_share > 0:
            preference = "accept"
            decision_basis = "positive-payoff-environmental-terminal-window"
        elif final_round and our_share > 0:
            preference = "accept"
            decision_basis = "positive-payoff-known-final-round"
        elif patient_deadlock_accept:
            preference = "accept"
            decision_basis = "evidence-derived-patient-deadlock-settlement"
        elif patient_reciprocal_window_accept:
            preference = "accept"
            decision_basis = "temporary-reciprocal-concession-window"
        elif patient_continuation and our_share + self.config.consistency_share_tolerance < settlement_floor:
            preference = "reject"
            decision_basis = "patient-settlement-boundary-not-met"
        elif patient_continuation:
            preference = "accept"
            decision_basis = "patient-settlement-boundary-met"
        elif our_share >= self.config.unconditional_accept_our_share:
            preference = "accept"
            decision_basis = "authenticated-offer-clears-unconditional-accept-share"
        elif reject_value > accept_value + margin:
            preference = "reject"
            decision_basis = "conservative-continuation-clears-required-margin"
        elif reject_value > accept_value + 1e-12:
            preference = "ambiguous"
            decision_basis = "positive-reject-advantage-below-robustness-margin"
        else:
            preference = "accept"
            decision_basis = "conservative-continuation-does-not-clear-required-margin"
        same_or_worse = next_offer is not None and float(next_offer["our_nominal_share"]) <= our_share + self.config.consistency_share_tolerance
        if preference in {"accept", "ambiguous"}:
            status = "warning"
            if same_or_worse:
                failure_mode = "reject-now-concede-next-round"
            elif preference == "ambiguous":
                failure_mode = "reject-advantage-below-robustness-margin"
            else:
                failure_mode = "accept-now-dominates-modeled-continuation"
        else:
            status = "conditional-rejection"
            failure_mode = None
        result["modeled_offer_policy_after_rejection"] = next_offer
        if patient_continuation:
            continuation_interpretation = "With unit self-discount and a later round available, rejection preserves the choice between an optimized offer and reproducing the current deal; it does not commit now to accepting the modeled 80th-percentile counterproposal."
        else:
            continuation_interpretation = "With non-unit self-discount, the conservative rollout values rejection through the selected committed next-offer path; no option-preservation claim applies."
        result["continuation_options_after_rejection"] = {
            "option_preserved": patient_continuation,
            "options": continuation_options,
            "selected_kind": selected_continuation.get("kind") if selected_continuation is not None else None,
            "revealed_willingness_floor": _round(revealed_willingness_floor) if revealed_willingness_floor is not None else None,
            "interpretation": continuation_interpretation,
        }
        result["decision_comparison"] = {
            "current_offer_opponent_share": _round(opponent_share),
            "current_offer_our_share": _round(our_share),
            "accept_now_value": _round(accept_value),
            "reject_value": _round(reject_value),
            "reject_advantage": _round(reject_value - accept_value),
            "required_reject_advantage": _round(margin),
            "required_reject_advantage_after_one_failed_probe": _round(self.config.repeated_probe_value_margin),
            "modeled_preference": preference,
            "decision_basis": decision_basis,
            "policy_regime": policy_regime,
            "minimum_current_own_settlement_share": _round(settlement_floor),
            "patient_reciprocal_concession_window_control": reciprocal_window_control,
            "rejected_offer_reservation": reservation,
            "environmental_terminal_window": environmental_terminal,
        }
        force_reject = not environmental_terminal and not final_round and not patient_deadlock_accept and not patient_reciprocal_window_accept and patient_continuation and our_share + self.config.consistency_share_tolerance < settlement_floor
        patient_boundary_accept = not environmental_terminal and not patient_deadlock_accept and not patient_reciprocal_window_accept and patient_continuation and our_share + self.config.consistency_share_tolerance >= settlement_floor
        accept_dominance = accept_value - reject_value
        required_accept_dominance = max(self.config.discounted_acceptance_dominance_margin, margin)
        post_rejection_exposure = bool(probe_count or self.evidence.rejected_opponent_offer_our_shares)
        discounted_current_utility_accept = discounted_self and post_rejection_exposure and our_share + self.config.consistency_share_tolerance >= self.config.minimum_material_own_share and preference == "accept" and accept_dominance > required_accept_dominance + 1e-12
        first_counterprobe_preserved = discounted_self and probe_count <= self.config.consistency_min_rejected_offers and preference in {"accept", "ambiguous"} and not discounted_current_utility_accept
        force_accept = (environmental_terminal and our_share > 0) or (final_round and our_share > 0) or patient_deadlock_accept or patient_reciprocal_window_accept or patient_boundary_accept or discounted_current_utility_accept
        result["policy_guard"] = {
            "mode": "thresholded-authority-eligible",
            "policy_regime": policy_regime,
            "minimum_current_own_settlement_share": _round(settlement_floor),
            "force_reject": force_reject,
            "force_reject_reason": "the current offer falls below the patient settlement boundary while another nonterminal round remains" if force_reject else None,
            "force_accept": force_accept,
            "force_accept_reason": "positive payoff inside the observed round-99 terminal window" if environmental_terminal and our_share > 0 else "positive known-final-round payoff" if final_round and our_share > 0 else "persistent patient deadlock settlement" if patient_deadlock_accept else "temporary reciprocal-concession window" if patient_reciprocal_window_accept else "patient settlement boundary met" if patient_boundary_accept else "current discounted utility dominates or nearly dominates the conservative post-rejection continuation" if discounted_current_utility_accept else None,
            "loss_minimization_control": loss_minimization_control,
            "patient_reciprocal_concession_window_control": reciprocal_window_control,
            "recommended_decision": preference if preference != "ambiguous" else "model-discretion",
            "behavioral_override_eligible": preference in {"accept", "reject"},
            "decision_override_additional_margin": _round(self.config.decision_override_additional_margin),
            "discounted_acceptance_dominance": _round(accept_dominance),
            "required_discounted_acceptance_dominance": _round(required_accept_dominance),
            "authority": "in capped complete-information states, a definite behavioral decision may replace the exact reference only when its quantified advantage clears one-round discount cost plus the stated additional margin",
            "authority_withheld_reason": None if preference in {"accept", "reject"} else "decision-value uncertainty leaves the action to the live model",
        }
        result["consistency_check"] = {
            "status": status,
            "failure_mode": failure_mode,
            "opponent_rejections_of_our_prior_offers": probe_count,
            "first_counterprobe_preserved": first_counterprobe_preserved,
            "modeled_next_offer_same_or_worse_for_us": same_or_worse,
            "rejection_is_coherent_only_if_next_offer_our_share": next_offer.get("our_nominal_share") if next_offer is not None and preference == "reject" else None,
            "active_settlement_floor": _round(settlement_floor),
            "rejected_offer_reservation": reservation,
            "interpretation": "Patient play retains its separate settlement boundary except during a narrowly evidenced bilateral-concession window or after both sides cross the persistent-deadlock threshold. Discounted play compares current utility with the conservative continuation and treats earlier rejected value as sunk. A known final round or the observed round-99 terminal window accepts any positive payoff rather than risking zero.",
        }
        return result


@dataclass(frozen=True)
class BargainingTurnForecast:
    """Prompt projection and exact-action predictor frozen before model inference."""

    prompt_context: dict[str, object]
    bundle: _FrozenForecastBundle
    context: BargainingContext
    seed_sha256: str
    state_revision: int
    rollout: _BehavioralRollout

    def rejection_path_offer_prediction(self, action: dict[str, Any]) -> dict[str, object] | None:
        """Evaluate one numeric offer in the next-round context created by rejecting the current offer."""
        continuation = self.prompt_context.get("behavioral_continuation") if isinstance(self.prompt_context.get("behavioral_continuation"), dict) else {}
        comparison = continuation.get("decision_comparison") if isinstance(continuation.get("decision_comparison"), dict) else {}
        current_opponent_share = _finite(comparison.get("current_offer_opponent_share"))
        if current_opponent_share is None or not self.rollout._round_available(self.context.round_number + 1):
            return None
        opponent_share = _clamp(_gain(action, self.context.opponent_player) / self.context.money_to_divide, 0.0, 1.0)
        next_context = replace(self.context, round_number=self.context.round_number + 1, previous_opponent_offer_share=current_opponent_share, previous_our_response="reject")
        return {
            "round": next_context.round_number,
            "current_offer_opponent_share": _round(current_opponent_share),
            "submitted_offer_opponent_share": _round(opponent_share),
            "behavioral_offer_evaluation": self.rollout.evaluate_offer(next_context, opponent_share),
        }

    def submission_prediction(self, action: dict[str, Any]) -> dict[str, object]:
        action_type = str(action.get("decision") or "offer").casefold()
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "frontier": "computed-before-network-submission",
            "seed_sha256": self.seed_sha256,
            "state_revision": self.state_revision,
            "game_id": self.context.game_id,
            "opponent": {"id": self.bundle.opponent_id, "name": self.bundle.opponent_name},
            "submitted_action": copy.deepcopy(action),
            "free_text_message_conditioning": "not-modeled",
        }
        if action_type in {"accept", "reject", "walkaway", "acceptoffer", "rejectoffer"}:
            receipt["prediction_scope"] = "No opponent action is predicted directly from a submitted decision; the turn prompt contains the reusable response curve and proposal distribution."
            continuation = self.prompt_context.get("behavioral_continuation") if isinstance(self.prompt_context.get("behavioral_continuation"), dict) else {}
            comparison = continuation.get("decision_comparison") if isinstance(continuation.get("decision_comparison"), dict) else {}
            modeled = str(comparison.get("modeled_preference") or "unavailable")
            submitted = {"acceptoffer": "accept", "rejectoffer": "reject"}.get(action_type, action_type)
            receipt["local_behavioral_consistency"] = {
                "submitted_decision": submitted,
                "modeled_preference": modeled,
                "consistent": modeled in {"ambiguous", "indifferent"} or modeled == submitted,
                "consistency_check": copy.deepcopy(continuation.get("consistency_check")),
            }
            return receipt
        opponent_share = _clamp(_gain(action, self.context.opponent_player) / self.context.money_to_divide, 0.0, 1.0)
        response = self.bundle.response(self.context, opponent_share)
        rejection_context = replace(self.context, round_number=self.context.round_number + 1, previous_our_offer_to_opponent_share=opponent_share, previous_opponent_response="reject")
        proposal, experts = self.bundle.proposal(rejection_context)
        continuation = self.prompt_context.get("behavioral_continuation") if isinstance(self.prompt_context.get("behavioral_continuation"), dict) else {}
        recovery = continuation.get("rejected_value_recovery_control") if isinstance(continuation.get("rejected_value_recovery_control"), dict) else {}
        interturn_consistency = None
        if recovery.get("status") == "available":
            our_share = 1 - opponent_share
            minimum_share = _finite(recovery.get("minimum_current_offer_our_share"))
            interturn_consistency = {
                "status": "recoverable-discounted-bound",
                "submitted_our_share": _round(our_share),
                "minimum_current_offer_our_share": recovery.get("minimum_current_offer_our_share"),
                "reserved_discounted_value": recovery.get("reserved_discounted_value"),
                "active_constraint": True,
                "consistent": minimum_share is not None and our_share + self.rollout.config.consistency_share_tolerance >= minimum_share,
                "interpretation": "The submitted offer is checked against the still-feasible discounted value of the best rejected opponent offer.",
            }
        elif recovery.get("status") == "historical-sunk":
            our_share = 1 - opponent_share
            interturn_consistency = {
                "status": "historical-regret-only",
                "submitted_our_share": _round(our_share),
                "historical_current_share_equivalent": recovery.get("historical_current_share_equivalent"),
                "reserved_discounted_value": recovery.get("reserved_discounted_value"),
                "active_constraint": False,
                "consistent": True,
                "interpretation": "The submitted offer may concede beyond a previously rejected payoff because that payoff is sunk rather than a present reservation right.",
            }
        receipt.update(
            {
                "prediction_scope": "Opponent response to this exact numeric offer and opponent's later proposal conditional on rejection; message content is excluded.",
                "offered_opponent_share": _round(opponent_share),
                "opponent_acceptance_probability_v2": _round(float(response["probability"])),
                "opponent_acceptance_probability_before_within_game_monotone_overlay": _round(float(response["probability_before_within_game_monotone_overlay"])),
                "within_game_monotone_evidence": copy.deepcopy(response["within_game_monotone_evidence"]),
                "response_expert_probabilities": {name: _round(float(value)) for name, value in dict(response["expert_probabilities"]).items()},
                "response_expert_weights": {name: _round(float(value)) for name, value in dict(response["expert_weights"]).items()},
                "local_interturn_consistency": interturn_consistency,
                "behavioral_offer_evaluation": self.rollout.evaluate_offer(self.context, opponent_share),
                "conditional_rejection_proposal": {**{key: _round(float(value)) if isinstance(value, (int, float)) else value for key, value in proposal.as_dict().items()}, "top_density_modes": _distribution_modes(proposal)},
                "conditional_rejection_proposal_experts": {name: {key: _round(float(value)) if isinstance(value, (int, float)) else value for key, value in distribution.as_dict().items()} for name, distribution in experts.items()},
                "proposal_expert_weights": {name: _round(float(value)) for name, value in self.bundle.proposal_expert_weights.items()},
            }
        )
        return receipt


class BargainingLiveAdvisorV2:
    """Serve frozen forecasts and update only from authenticated terminal games."""

    def __init__(self, *, seed_path: Path, journal_path: Path, project_root: Path) -> None:
        self.seed_path = seed_path
        self.journal_path = journal_path
        self.project_root = project_root
        self._lock = threading.RLock()
        self.seed, self.seed_source_path = _load_seed_document(seed_path, project_root)
        self.seed_sha256 = str(self.seed["seed_sha256"])
        self._verify_seed_metadata()
        self.validation_config = ValidationConfig(**dict(self.seed["validation_config"]))
        self.twin_config = TwinConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in dict(self.seed["twin_config"]).items()})
        self.adaptive_config = AdaptiveConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in dict(self.seed["adaptive_config"]).items()})
        self.live_config = LiveAdvisorConfig(**dict(self.seed["live_config"]))
        self.live_config.validate()
        self.response_programs = response_particles(self.twin_config)
        self.proposal_programs = proposal_particles(self.twin_config)
        self.proposal_program_ids = tuple(program.identifier for program in self.proposal_programs)
        self.response_complexities = [particle.complexity for particle in self.response_programs]
        self.proposal_complexities = [particle.complexity for particle in self.proposal_programs]
        self.paper_targets = {str(value["id"]) for value in self.seed["paper_targets"]}
        self.compiled_seed_state_path = self._compiled_seed_state_path()
        self.compiled_seed_state_status = "unavailable"
        self._initialize_empty_state()
        try:
            self._restore_compiled_seed_state()
            self.compiled_seed_state_status = "restored"
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._verify_seed()
            self._initialize_empty_state()
            for embedded in _seed_games(self.seed, seed_path=self.seed_source_path):
                self._apply_embedded(dict(embedded), source="seed")
            self._write_compiled_seed_state()
            self.compiled_seed_state_status = "compiled"
        self.seed_game_count = self.state_revision
        self._replay_journal()

    def _initialize_empty_state(self) -> None:
        self.global_response_scores = [0.0] * len(self.response_programs)
        self.global_proposal_scores = [0.0] * len(self.proposal_programs)
        self.global_response_count = 0
        self.global_proposal_count = 0
        self.target_response_scores: dict[str, list[float]] = {}
        self.target_proposal_scores: dict[str, list[float]] = {}
        self.target_response_count: dict[str, int] = defaultdict(int)
        self.target_proposal_count: dict[str, int] = defaultdict(int)
        self.observations: dict[str, list[PriorObservation]] = defaultdict(list)
        self.game_counts: dict[str, int] = defaultdict(int)
        self.adaptive_states: dict[str, AdaptiveOpponentState] = {}
        self.completed_game_hashes: dict[str, str] = {}
        self._bundle_cache: dict[tuple[int, str], _FrozenForecastBundle] = {}
        self.state_revision = 0
        self.journal_sequence = 0

    def _verify_seed_metadata(self) -> None:
        if self.seed.get("schema_version") != SCHEMA_VERSION or self.seed.get("kind") != SEED_KIND or self.seed.get("model_version") != MODEL_VERSION:
            raise RuntimeError(f"unsupported bargaining advisor seed: {self.seed_path}")
        if self.seed.get("base_engine_version") != BASE_ENGINE_VERSION:
            raise RuntimeError(f"bargaining advisor seed has a different base engine: {self.seed_path}")
        for label, receipt in dict(self.seed.get("implementation_receipts") or {}).items():
            path = self.project_root / str(receipt["path"])
            if not path.is_file() or _sha_file(path) != receipt.get("sha256"):
                raise RuntimeError(f"bargaining advisor implementation receipt mismatch: {label}")

    def _verify_seed(self) -> None:
        self._verify_seed_metadata()
        if self.seed.get("seed_sha256") != _seed_digest(self.seed, seed_path=self.seed_source_path):
            raise RuntimeError(f"bargaining advisor seed SHA-256 mismatch: {self.seed_path}")

    def _compiled_seed_state_path(self) -> Path:
        return self.seed_source_path.parent / ".bargaining-advisor-compiled" / f"{self.seed_sha256}.json"

    def _seed_storage_receipt(self) -> dict[str, object]:
        receipt: dict[str, object] = {"seed_source_file_sha256": _sha_file(self.seed_source_path)}
        reference = self.seed.get("games_ref")
        if not isinstance(reference, dict):
            receipt["game_storage"] = "inline"
            return receipt
        root = self.seed_source_path.parent.resolve()
        pack_path = (root / str(reference.get("path") or "")).resolve()
        try:
            pack_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("bargaining seed game-pack reference escapes its root") from error
        expected = str(reference.get("pack_sha256") or "")
        actual = _sha_file(pack_path)
        if len(expected) != 64 or actual != expected:
            raise RuntimeError(f"bargaining seed game pack failed SHA-256 verification: {pack_path}")
        receipt.update({"game_storage": "immutable-pack", "pack_path": str(pack_path), "pack_sha256": actual})
        return receipt

    @staticmethod
    def _adaptive_state_value(state: AdaptiveOpponentState) -> dict[str, object]:
        return {
            "response": {"scores": dict(state.response.scores), "update_count": state.response.update_count},
            "proposal": {"scores": dict(state.proposal.scores), "update_count": state.proposal.update_count},
        }

    def _restore_adaptive_state(self, value: object) -> AdaptiveOpponentState:
        if not isinstance(value, dict):
            raise RuntimeError("compiled bargaining adaptive state is not an object")
        state = AdaptiveOpponentState(self.adaptive_config)
        for name, expert_state in (("response", state.response), ("proposal", state.proposal)):
            raw = value.get(name)
            if not isinstance(raw, dict) or not isinstance(raw.get("scores"), dict):
                raise RuntimeError(f"compiled bargaining {name} adaptive state is invalid")
            scores = {str(key): float(score) for key, score in raw["scores"].items()}
            if set(scores) != set(expert_state.names) or any(not math.isfinite(score) for score in scores.values()):
                raise RuntimeError(f"compiled bargaining {name} adaptive scores are invalid")
            update_count = int(raw.get("update_count") or 0)
            if update_count < 0:
                raise RuntimeError(f"compiled bargaining {name} adaptive update count is invalid")
            expert_state.scores = scores
            expert_state.update_count = update_count
        return state

    def _compiled_state_value(self) -> dict[str, object]:
        return {
            "global_response_scores": _encode_float_vector(self.global_response_scores),
            "global_proposal_scores": _encode_float_vector(self.global_proposal_scores),
            "global_response_count": self.global_response_count,
            "global_proposal_count": self.global_proposal_count,
            "target_response_scores": {key: _encode_float_vector(value) for key, value in sorted(self.target_response_scores.items())},
            "target_proposal_scores": {key: _encode_float_vector(value) for key, value in sorted(self.target_proposal_scores.items())},
            "target_response_count": dict(sorted(self.target_response_count.items())),
            "target_proposal_count": dict(sorted(self.target_proposal_count.items())),
            "observations": {key: [_serialize_observation(observation) for observation in values] for key, values in sorted(self.observations.items())},
            "game_counts": dict(sorted(self.game_counts.items())),
            "adaptive_states": {key: self._adaptive_state_value(value) for key, value in sorted(self.adaptive_states.items())},
            "completed_game_hashes": dict(sorted(self.completed_game_hashes.items())),
            "state_revision": self.state_revision,
        }

    def _write_compiled_seed_state(self) -> None:
        state = self._compiled_state_value()
        document = {
            "schema_version": SCHEMA_VERSION,
            "kind": COMPILED_SEED_STATE_KIND,
            "compiled_state_version": COMPILED_STATE_VERSION,
            "model_version": MODEL_VERSION,
            "base_engine_version": BASE_ENGINE_VERSION,
            "seed_sha256": self.seed_sha256,
            "generated_at": _now(),
            "seed_storage": self._seed_storage_receipt(),
            "state": state,
            "state_sha256": _sha(state),
        }
        _atomic_compact_json(self.compiled_seed_state_path, document)

    def _restore_compiled_seed_state(self) -> None:
        document = json.loads(self.compiled_seed_state_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != COMPILED_SEED_STATE_KIND or document.get("compiled_state_version") != COMPILED_STATE_VERSION:
            raise RuntimeError("compiled bargaining seed state has an unsupported envelope")
        if document.get("model_version") != MODEL_VERSION or document.get("base_engine_version") != BASE_ENGINE_VERSION or document.get("seed_sha256") != self.seed_sha256:
            raise RuntimeError("compiled bargaining seed state belongs to a different model or seed")
        if document.get("seed_storage") != self._seed_storage_receipt():
            raise RuntimeError("compiled bargaining seed state storage receipt changed")
        state = document.get("state")
        if not isinstance(state, dict) or document.get("state_sha256") != _sha(state):
            raise RuntimeError("compiled bargaining seed state failed content verification")
        response_count = len(self.response_programs)
        proposal_count = len(self.proposal_programs)
        self.global_response_scores = _decode_float_vector(state.get("global_response_scores"), expected_count=response_count)
        self.global_proposal_scores = _decode_float_vector(state.get("global_proposal_scores"), expected_count=proposal_count)
        self.global_response_count = int(state.get("global_response_count") or 0)
        self.global_proposal_count = int(state.get("global_proposal_count") or 0)
        raw_target_response = state.get("target_response_scores")
        raw_target_proposal = state.get("target_proposal_scores")
        raw_observations = state.get("observations")
        if not isinstance(raw_target_response, dict) or not isinstance(raw_target_proposal, dict) or not isinstance(raw_observations, dict):
            raise RuntimeError("compiled bargaining seed state omits target state")
        self.target_response_scores = {str(key): _decode_float_vector(value, expected_count=response_count) for key, value in raw_target_response.items()}
        self.target_proposal_scores = {str(key): _decode_float_vector(value, expected_count=proposal_count) for key, value in raw_target_proposal.items()}
        self.target_response_count = defaultdict(int, {str(key): int(value) for key, value in dict(state.get("target_response_count") or {}).items()})
        self.target_proposal_count = defaultdict(int, {str(key): int(value) for key, value in dict(state.get("target_proposal_count") or {}).items()})
        self.observations = defaultdict(list, {str(key): [_restore_observation(value) for value in values] for key, values in raw_observations.items() if isinstance(values, list)})
        self.game_counts = defaultdict(int, {str(key): int(value) for key, value in dict(state.get("game_counts") or {}).items()})
        self.adaptive_states = {str(key): self._restore_adaptive_state(value) for key, value in dict(state.get("adaptive_states") or {}).items()}
        self.completed_game_hashes = {str(key): str(value) for key, value in dict(state.get("completed_game_hashes") or {}).items()}
        self.state_revision = int(state.get("state_revision") or 0)
        self.journal_sequence = 0
        self._bundle_cache = {}
        if self.state_revision != int(self.seed.get("game_count") or -1) or len(self.completed_game_hashes) != self.state_revision or sum(self.game_counts.values()) != self.state_revision:
            raise RuntimeError("compiled bargaining seed state has inconsistent game counts")
        expected_rows = int(self.seed.get("row_count") or -1)
        observed_rows = sum(len(values) for values in self.observations.values())
        if observed_rows != expected_rows or self.global_response_count + self.global_proposal_count != expected_rows:
            raise RuntimeError("compiled bargaining seed state has inconsistent row counts")
        if sum(self.target_response_count.values()) != self.global_response_count or sum(self.target_proposal_count.values()) != self.global_proposal_count:
            raise RuntimeError("compiled bargaining seed state has inconsistent target counts")

    def _target_scores(self, opponent_id: str) -> tuple[list[float], list[float]]:
        response = self.target_response_scores.setdefault(opponent_id, [0.0] * len(self.response_programs))
        proposal = self.target_proposal_scores.setdefault(opponent_id, [0.0] * len(self.proposal_programs))
        return response, proposal

    def _adaptive_state(self, opponent_id: str) -> AdaptiveOpponentState:
        return self.adaptive_states.setdefault(opponent_id, AdaptiveOpponentState(self.adaptive_config))

    def _bundle(self, opponent_id: str, opponent_name: str) -> _FrozenForecastBundle:
        population_only = self.game_counts.get(opponent_id, 0) == 0
        cache_identity = "__population_only__" if population_only else opponent_id
        cached = self._bundle_cache.get((self.state_revision, cache_identity))
        if cached is not None:
            return replace(cached, opponent_id=opponent_id, opponent_name=opponent_name) if population_only else cached
        target_response_scores, target_proposal_scores = self._target_scores(opponent_id)
        population_response_scores = [left - right for left, right in zip(self.global_response_scores, target_response_scores, strict=True)]
        population_proposal_scores = [left - right for left, right in zip(self.global_proposal_scores, target_proposal_scores, strict=True)]
        population_response_prior = _mixed_population_prior(population_response_scores, row_count=self.global_response_count - self.target_response_count[opponent_id], complexities=self.response_complexities, config=self.twin_config)
        population_proposal_prior = _mixed_population_prior(population_proposal_scores, row_count=self.global_proposal_count - self.target_proposal_count[opponent_id], complexities=self.proposal_complexities, config=self.twin_config)
        hierarchical_response = _posterior_from_prior(population_response_prior, target_response_scores)
        hierarchical_proposal = _posterior_from_prior(population_proposal_prior, target_proposal_scores)
        target_prior = tuple(self.observations[opponent_id])
        population_prior = tuple(observation for other_id, values in self.observations.items() if other_id != opponent_id for observation in values)
        ordinary_population, ordinary_target = _ordinary_weights(population_prior, target_prior, self.validation_config)
        recency_population, recency_target = _recency_weights(population_prior, target_prior, current_global_game_index=self.state_revision, current_target_game_index=self.game_counts[opponent_id], config=self.validation_config)
        ordinary = ordinary_population + ordinary_target
        state = self._adaptive_state(opponent_id)
        bundle = _FrozenForecastBundle(
            opponent_id=opponent_id,
            opponent_name=opponent_name,
            global_game_index=self.state_revision,
            target_game_index=self.game_counts[opponent_id],
            target_prior=target_prior,
            population_prior=population_prior,
            response_programs=self.response_programs,
            proposal_programs=self.proposal_programs,
            proposal_program_ids=self.proposal_program_ids,
            hierarchical_response_weights=tuple(hierarchical_response),
            hierarchical_proposal_weights=tuple(hierarchical_proposal),
            recency_model=RecencyKernelModel(recency_population + recency_target, self.validation_config),
            logistic_model=LogisticResponseModel.fit(ordinary, l2=self.validation_config.logistic_l2),
            ridge_model=RidgeProposalModel.fit(ordinary, l2=self.validation_config.ridge_l2),
            response_expert_weights=dict(state.response.weights()),
            proposal_expert_weights=dict(state.proposal.weights()),
            adaptive_config=self.adaptive_config,
            validation_config=self.validation_config,
            live_config=self.live_config,
        )
        self._bundle_cache[(self.state_revision, cache_identity)] = bundle
        return bundle

    def _compile_game_update(self, game: BargainingGameEvidence, final_game_sha256: str) -> dict[str, object]:
        opponent_id = game.opponent_id
        target_game_index = self.game_counts[opponent_id]
        adaptive_state_after = None
        if target_game_index >= self.validation_config.warmup_games:
            bundle = self._bundle(opponent_id, game.opponent_name)
            response_losses: dict[str, list[float]] = defaultdict(list)
            proposal_losses: dict[str, list[float]] = defaultdict(list)
            for row in game.rows:
                if row.action_type == "response" and row.offered_share is not None and row.accepted is not None:
                    forecast = bundle.response(row.context, float(row.offered_share))
                    for name, probability in dict(forecast["expert_probabilities"]).items():
                        response_losses[name].append(-math.log(float(probability) if row.accepted else 1 - float(probability)))
                elif row.action_type == "proposal" and row.proposal_share is not None:
                    _forecast, experts = bundle.proposal(row.context)
                    for name, distribution in experts.items():
                        proposal_losses[name].append(distribution.nll(float(row.proposal_share)))
            state = copy.deepcopy(self._adaptive_state(opponent_id))
            if response_losses:
                state.response.update({name: sum(response_losses[name]) / len(response_losses[name]) for name in RESPONSE_EXPERTS})
            if proposal_losses:
                state.proposal.update({name: sum(proposal_losses[name]) / len(proposal_losses[name]) for name in PROPOSAL_EXPERTS})
            adaptive_state_after = self._adaptive_state_value(state)
        response_update = _response_scores(self.response_programs, game.rows)
        proposal_update = _proposal_scores(self.proposal_programs, game.rows)
        return {
            "kind": COMPILED_GAME_UPDATE_KIND,
            "compiled_state_version": COMPILED_STATE_VERSION,
            "model_version": MODEL_VERSION,
            "seed_sha256": self.seed_sha256,
            "game_id": game.game_id,
            "opponent_id": opponent_id,
            "expected_state_revision": self.state_revision,
            "expected_target_game_index": target_game_index,
            "final_game_sha256": final_game_sha256,
            "row_sha256": _sha([asdict(row) for row in game.rows]),
            "response_score_delta": response_update,
            "proposal_score_delta": proposal_update,
            "response_count": sum(row.action_type == "response" for row in game.rows),
            "proposal_count": sum(row.action_type == "proposal" for row in game.rows),
            "adaptive_state_after": adaptive_state_after,
        }

    def _stored_compiled_game_update(self, update: Mapping[str, object]) -> dict[str, object]:
        stored = {key: copy.deepcopy(value) for key, value in update.items() if key not in {"response_score_delta", "proposal_score_delta"}}
        stored["response_score_delta"] = _encode_float_vector(update["response_score_delta"] if isinstance(update["response_score_delta"], Sequence) else ())
        stored["proposal_score_delta"] = _encode_float_vector(update["proposal_score_delta"] if isinstance(update["proposal_score_delta"], Sequence) else ())
        stored["update_sha256"] = _sha(stored)
        return stored

    def _restore_compiled_game_update(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise RuntimeError("compiled bargaining game update is not an object")
        stored = copy.deepcopy(value)
        expected_sha256 = stored.pop("update_sha256", None)
        if expected_sha256 != _sha(stored):
            raise RuntimeError("compiled bargaining game update failed content verification")
        stored["response_score_delta"] = _decode_float_vector(stored.get("response_score_delta"), expected_count=len(self.response_programs))
        stored["proposal_score_delta"] = _decode_float_vector(stored.get("proposal_score_delta"), expected_count=len(self.proposal_programs))
        return stored

    def _apply_compiled_game_update(self, game: BargainingGameEvidence, final_game_sha256: str, update: Mapping[str, object]) -> None:
        if update.get("kind") != COMPILED_GAME_UPDATE_KIND or update.get("compiled_state_version") != COMPILED_STATE_VERSION or update.get("model_version") != MODEL_VERSION or update.get("seed_sha256") != self.seed_sha256:
            raise RuntimeError(f"invalid compiled bargaining update for game {game.game_id}")
        if update.get("game_id") != game.game_id or update.get("opponent_id") != game.opponent_id or update.get("final_game_sha256") != final_game_sha256 or update.get("row_sha256") != _sha([asdict(row) for row in game.rows]):
            raise RuntimeError(f"compiled bargaining update evidence mismatch for game {game.game_id}")
        if int(update["expected_state_revision"]) != self.state_revision or int(update["expected_target_game_index"]) != self.game_counts[game.opponent_id]:
            raise RuntimeError(f"compiled bargaining update frontier mismatch for game {game.game_id}")
        response_update = update.get("response_score_delta")
        proposal_update = update.get("proposal_score_delta")
        if not isinstance(response_update, Sequence) or len(response_update) != len(self.response_programs) or not isinstance(proposal_update, Sequence) or len(proposal_update) != len(self.proposal_programs):
            raise RuntimeError(f"compiled bargaining update score dimensions changed for game {game.game_id}")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in (*response_update, *proposal_update)):
            raise RuntimeError(f"compiled bargaining update has a non-finite score for game {game.game_id}")
        response_count = int(update.get("response_count") or 0)
        proposal_count = int(update.get("proposal_count") or 0)
        if response_count != sum(row.action_type == "response" for row in game.rows) or proposal_count != sum(row.action_type == "proposal" for row in game.rows):
            raise RuntimeError(f"compiled bargaining update row counts changed for game {game.game_id}")
        adaptive_state_after = update.get("adaptive_state_after")
        if adaptive_state_after is not None:
            self.adaptive_states[game.opponent_id] = self._restore_adaptive_state(adaptive_state_after)
        target_response_scores, target_proposal_scores = self._target_scores(game.opponent_id)
        for total, delta in ((self.global_response_scores, response_update), (self.global_proposal_scores, proposal_update), (target_response_scores, response_update), (target_proposal_scores, proposal_update)):
            for index, value in enumerate(delta):
                total[index] += float(value)
        self.global_response_count += response_count
        self.global_proposal_count += proposal_count
        self.target_response_count[game.opponent_id] += response_count
        self.target_proposal_count[game.opponent_id] += proposal_count
        self.observations[game.opponent_id].extend(PriorObservation(row, self.state_revision, self.game_counts[game.opponent_id]) for row in game.rows)
        self.game_counts[game.opponent_id] += 1
        self.completed_game_hashes[game.game_id] = final_game_sha256
        self.state_revision += 1
        self._bundle_cache.clear()

    def _apply_game(self, game: BargainingGameEvidence, final_game_sha256: str) -> None:
        previous_hash = self.completed_game_hashes.get(game.game_id)
        if previous_hash is not None:
            if previous_hash != final_game_sha256:
                raise RuntimeError(f"conflicting bargaining terminal game: {game.game_id}")
            return
        update = self._compile_game_update(game, final_game_sha256)
        self._apply_compiled_game_update(game, final_game_sha256, update)

    def _apply_embedded(self, embedded: dict[str, Any], *, source: str) -> None:
        job = embedded.get("job")
        if not isinstance(job, dict) or embedded.get("job_object_sha256") != _sha(job):
            raise RuntimeError(f"invalid {source} bargaining job object")
        game = extract_bargaining_game(job, job_path=Path(f"{source}/{job.get('job_id', 'unknown')}.json"), job_sha256=str(embedded.get("job_sha256") or _sha(job)))
        self._apply_game(game, str(job.get("final_game_sha256") or _sha(job["final_game"])))

    def _replay_journal(self) -> None:
        if not self.journal_path.is_file():
            return
        for line_number, line in enumerate(self.journal_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != JOURNAL_KIND or record.get("model_version") != MODEL_VERSION or record.get("seed_sha256") != self.seed_sha256:
                raise RuntimeError(f"invalid bargaining advisor journal record at line {line_number}")
            self.journal_sequence = max(self.journal_sequence, int(record["journal_sequence"]))
            embedded = dict(record["embedded_game"])
            if record.get("compiled_update") is None:
                self._apply_embedded(embedded, source="journal")
                continue
            job = embedded.get("job")
            if not isinstance(job, dict) or embedded.get("job_object_sha256") != _sha(job):
                raise RuntimeError(f"invalid journal bargaining job object at line {line_number}")
            game = extract_bargaining_game(job, job_path=Path(f"journal/{job.get('job_id', 'unknown')}.json"), job_sha256=str(embedded.get("job_sha256") or _sha(job)))
            final_sha256 = str(job.get("final_game_sha256") or _sha(job["final_game"]))
            previous_hash = self.completed_game_hashes.get(game.game_id)
            if previous_hash is not None:
                if previous_hash != final_sha256:
                    raise RuntimeError(f"conflicting bargaining terminal game: {game.game_id}")
                continue
            update = self._restore_compiled_game_update(record["compiled_update"])
            self._apply_compiled_game_update(game, final_sha256, update)

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "model_version": MODEL_VERSION,
            "base_engine_version": BASE_ENGINE_VERSION,
            "seed_sha256": self.seed_sha256,
            "seed_game_count": self.seed_game_count,
            "compiled_seed_state": {"contract": COMPILED_SEED_STATE_KIND, "version": COMPILED_STATE_VERSION},
            "paper_target_count": len(self.paper_targets),
            "frontier": "pre-inference prompt forecast and pre-network exact-action receipt",
        }

    def status(self) -> dict[str, object]:
        with self._lock:
            return {**self.manifest_receipt(), "state_revision": self.state_revision, "journal_game_count": self.state_revision - self.seed_game_count, "journal_sequence": self.journal_sequence}

    def forecast_turn(self, game: dict[str, Any]) -> BargainingTurnForecast:
        with self._lock:
            if game.get("game_family") != "bargaining":
                raise ValueError("bargaining advisor received a non-bargaining turn")
            opponent_id, opponent_name, named = _identity(game)
            context = context_from_live_game(game, opponent_id=opponent_id, opponent_name=opponent_name)
            raw_base_bundle = self._bundle(opponent_id, opponent_name)
            target_rows = [observation.row for observation in raw_base_bundle.target_prior]
            target_games = self.game_counts[opponent_id]
            ood_flags = _ood_flags(context, _support(target_rows))
            target_recency_suppressed = 0 < target_games < self.validation_config.warmup_games and "unseen_horizon_regime" in ood_flags
            if target_recency_suppressed:
                base_bundle, target_recency_gate = raw_base_bundle.without_target_recency_authority()
                target_recency_gate.update({"target_game_count": target_games, "warmup_game_count": self.validation_config.warmup_games, "ood_flags": list(ood_flags)})
            else:
                base_bundle = raw_base_bundle
                target_recency_gate = {
                    "status": "retained",
                    "expert": "target_recency",
                    "reason": "the sparse-target incompatible-horizon gate did not fire",
                    "target_game_count": target_games,
                    "warmup_game_count": self.validation_config.warmup_games,
                    "ood_flags": list(ood_flags),
                    "effective_weights": {name: _round(weight) for name, weight in base_bundle.response_expert_weights.items()},
                }
            within_game = _live_opponent_evidence(game, opponent_id=opponent_id, opponent_name=opponent_name)
            bundle = base_bundle
            for row in within_game.rows:
                bundle = bundle.incorporate(row, self.live_config)
                if target_recency_suppressed:
                    bundle, _gate_receipt = bundle.without_target_recency_authority()
            if within_game.rows:
                bundle = bundle.refit_local_baselines()
            response_grid = [(index / 200, float(bundle.response(context, index / 200)["probability"])) for index in range(201)]
            curve = [{"opponent_share": _round(share), "accept_probability_v2": _round(float(bundle.response(context, share)["probability"]))} for share in CURVE_SHARES]
            dense = response_grid[1:-1]
            myopic_share, myopic_probability = max(dense, key=lambda value: value[1] * (1 - value[0]))
            proposal, _experts = bundle.proposal(context)
            within_game_receipt = within_game.prompt_receipt()
            if within_game.rejected_our_offer_shares:
                latest_rejected_share = within_game.rejected_our_offer_shares[-1]
                within_game_receipt["latest_rejected_offer_probability_shift"] = {
                    "opponent_share": _round(latest_rejected_share),
                    "prior_probability": _round(float(base_bundle.response(context, latest_rejected_share)["probability"])),
                    "posterior_probability": _round(float(bundle.response(context, latest_rejected_share)["probability"])),
                }
            if within_game.opponent_proposal_shares:
                base_proposal, _base_experts = base_bundle.proposal(context)
                within_game_receipt["proposal_mean_shift"] = {"prior_mean": _round(base_proposal.mean), "posterior_mean": _round(proposal.mean)}
            if opponent_id in self.paper_targets:
                tier = "sealed-paper-target"
            elif target_games >= self.validation_config.warmup_games:
                tier = "exploratory-opponent-adapted"
            elif target_games:
                tier = "sparse-opponent-plus-population"
            else:
                tier = "population-only"
            rollout = _BehavioralRollout(bundle=bundle, context=context, evidence=within_game, config=self.live_config)
            behavioral_continuation = rollout.projection(game, evidence_tier=tier, ood_flags=ood_flags)
            guard_can_force = ["bound-nonanalytic-offer", "reject-zero-own-payoff", "patient-settlement-boundary", "temporary-reciprocal-concession-window", "persistent-patient-deadlock-settlement", "current-discounted-utility-settlement", "terminal-cap-loss-minimization", "thresholded-capped-decision-override", "gross-offer-expected-value-regression"]
            guard_cannot_force = ["behavioral-decision-before-first-failed-probe"]
            prompt_context: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION,
                "base_engine_version": BASE_ENGINE_VERSION,
                "status": "available",
                "seed_sha256": self.seed_sha256,
                "state_revision": self.state_revision,
                "opponent": {"id": opponent_id, "name": opponent_name, "named": named},
                "evidence": {"tier": tier, "target_game_count": target_games, "target_row_count": len(target_rows), "population_game_count": self.state_revision - target_games, "paper_target": opponent_id in self.paper_targets, "target_recency_gate": target_recency_gate},
                "within_game_posterior": within_game_receipt,
                "control_parameters": {
                    "decision_override_additional_margin": _round(self.live_config.decision_override_additional_margin),
                    "offer_expected_value_regression_margin": _round(self.live_config.offer_expected_value_regression_margin),
                    "exact_offer_match_share_tolerance": _round(self.live_config.exact_offer_match_share_tolerance),
                    "patient_minimum_own_settlement_share": _round(self.live_config.patient_minimum_own_settlement_share),
                    "patient_minimum_opponent_share": _round(self.live_config.patient_minimum_opponent_share),
                    "patient_maximum_opponent_share": _round(self.live_config.patient_maximum_opponent_share),
                    "patient_reciprocal_window_minimum_own_share": _round(self.live_config.patient_reciprocal_window_minimum_own_share),
                    "patient_reciprocal_window_minimum_opponent_concessions": self.live_config.patient_reciprocal_window_minimum_opponent_concessions,
                    "patient_reciprocal_window_minimum_bilateral_movement": _round(self.live_config.patient_reciprocal_window_minimum_bilateral_movement),
                    "patient_deadlock_minimum_repetitions_per_player": self.live_config.patient_deadlock_minimum_repetitions_per_player,
                    "post_rejection_concession_minimum_value_gain": _round(self.live_config.post_rejection_concession_minimum_value_gain),
                    "environmental_round_cap": self.live_config.environmental_round_cap,
                    "environmental_terminal_window_rounds": self.live_config.environmental_terminal_window_rounds,
                    "environmental_minimum_positive_own_share": _round(self.live_config.environmental_minimum_positive_own_share),
                    "loss_minimization_discount_retention": _round(self.live_config.loss_minimization_discount_retention),
                    "loss_minimization_minimum_bilateral_rejections": self.live_config.loss_minimization_minimum_bilateral_rejections,
                    "loss_minimization_bridge_fraction": _round(self.live_config.loss_minimization_bridge_fraction),
                },
                "response_to_our_numeric_offer": {
                    "curve_by_opponent_share": curve,
                    "minimum_share_sustaining_at_least_50_percent_acceptance": _threshold(response_grid, 0.5),
                    "minimum_share_sustaining_at_least_80_percent_acceptance": _threshold(response_grid, 0.8),
                    "myopic_no_continuation_candidate": {"opponent_share": _round(myopic_share), "accept_probability_v2": _round(myopic_probability), "objective": "accept_probability * our_immediate_share"},
                    "expert_weights": {name: _round(weight) for name, weight in bundle.response_expert_weights.items()},
                },
                "opponent_proposal_share": {
                    **{key: _round(float(value)) if isinstance(value, (int, float)) else value for key, value in proposal.as_dict().items()},
                    "top_density_modes": _distribution_modes(proposal),
                    "expert_weights": {name: _round(weight) for name, weight in bundle.proposal_expert_weights.items()},
                },
                "behavioral_continuation": behavioral_continuation,
                "ood_flags": ood_flags,
                "model_boundary": {
                    "authority": "fallible-statistical-evidence-plus-bounded-offer-and-zero-payoff-controls",
                    "engine_authenticated": False,
                    "action_directive": "bounded offer controls, gross expected-value regression control, zero-payoff rejection, and thresholded capped complete-information decision overrides",
                    "policy_guard_can_force": guard_can_force,
                    "policy_guard_cannot_force": guard_cannot_force,
                    "equilibrium_calculation": False,
                    "continuation_value_included_in_myopic_candidate": False,
                    "behavioral_continuation_is_bounded": True,
                    "free_text_message_effects_modeled": False,
                },
            }
            state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
            last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else None
            if last_offer is not None:
                observed_share = _clamp(_gain(last_offer, context.opponent_player) / context.money_to_divide, 0.0, 1.0)
                prompt_context["current_opponent_offer_diagnostic"] = {"observed_opponent_share": _round(observed_share), "relative_density_at_observation": _round(proposal.density(observed_share) / max(proposal.density(index / 100) for index in range(101))), "post_observation_only": True}
            return BargainingTurnForecast(prompt_context=prompt_context, bundle=bundle, context=context, seed_sha256=self.seed_sha256, state_revision=self.state_revision, rollout=rollout)

    def update_completed_game(self, final_game: dict[str, Any], *, completed_at: str, completion_order: int) -> dict[str, object]:
        with self._lock:
            if final_game.get("game_family") != "bargaining":
                return {"status": "ignored", "reason": "non-bargaining"}
            result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
            if str(final_game.get("status") or "").casefold() in {"timeout", "cancelled", "abandoned"} or str(result.get("outcome") or "").casefold() in {"timeout", "cancelled", "abandoned"}:
                return {"status": "ignored", "reason": "censored-terminal-state"}
            opponent_id, opponent_name, _named = _identity(final_game)
            game_id = str(final_game.get("game_id") or "")
            if not game_id:
                raise ValueError("completed bargaining game has no game_id")
            final_sha = _sha(final_game)
            previous = self.completed_game_hashes.get(game_id)
            if previous is not None:
                if previous != final_sha:
                    raise RuntimeError(f"conflicting completed bargaining game: {game_id}")
                return {"status": "duplicate", "game_id": game_id, "state_revision": self.state_revision}
            job_id = _sha({"seed_sha256": self.seed_sha256, "game_id": game_id, "final_game_sha256": final_sha})
            job = {
                "schema_version": SCHEMA_VERSION,
                "kind": "bargaining-live-advisor-job",
                "job_id": job_id,
                "opponent": {"id": opponent_id, "name": opponent_name},
                "game_id": game_id,
                "game_family": "bargaining",
                "completed_at": completed_at,
                "completion_order": completion_order,
                "final_game_sha256": final_sha,
                "final_game": final_game,
            }
            embedded = {"job": job, "job_sha256": _sha(job), "job_object_sha256": _sha(job)}
            game = extract_bargaining_game(job, job_path=Path(f"journal/{job_id}.json"), job_sha256=str(embedded["job_sha256"]))
            compiled_update = self._compile_game_update(game, final_sha)
            next_sequence = self.journal_sequence + 1
            record = {"schema_version": SCHEMA_VERSION, "kind": JOURNAL_KIND, "model_version": MODEL_VERSION, "journal_sequence": next_sequence, "recorded_at": _now(), "seed_sha256": self.seed_sha256, "embedded_game": embedded, "compiled_update": self._stored_compiled_game_update(compiled_update)}
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(_canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.journal_sequence = next_sequence
            self._apply_compiled_game_update(game, final_sha, compiled_update)
            return {"status": "updated", "game_id": game_id, "opponent": {"id": opponent_id, "name": opponent_name}, "state_revision": self.state_revision, "journal_sequence": self.journal_sequence}
