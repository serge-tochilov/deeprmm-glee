"""Incremental named-opponent dossiers with asynchronous opponent-family synthesis lanes."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, Field, field_validator

from .glee_named_dossier import _event_linkage, _iter_jsonl, _parse_worker_request, _read_jsonl, named_opponent_id, normalize_opponent_name
from .immutable_blob import hydrate_call_request, load_referenced_text_blob, reference_text_blob
from .immutable_pack import load_json_object
from .glee_policy import GLEE_FAMILIES
from .glee_semantics import PERSUASION_INFORMATION_SEMANTICS_VERSION, model_game_semantics, model_visible_game_state, persuasion_information_semantics
from .glee_synopsis import live_dossier_projection, validate_future_synopsis
from .model_runner import ArenaCodexRunner

_SCHEMA_VERSION = 3
_UPDATE_ROLE = "glee_named_opponent_incremental"


def _state_contract(game_family: str) -> dict[str, object]:
    contract: dict[str, object] = {
        "current_family_dossier": "Fallible recursively compressed state from earlier direct games in this exact opponent-family lane; null when the first such game starts the dossier.",
        "sibling_family_dossiers": "Atomic snapshots of direct-evidence projections from the other family lanes. They are contextual hypotheses, never direct evidence for the target family, and recursively transferred hypotheses are excluded.",
        "new_game_evidence_batch": "The only new direct evidence in this transition, ordered by game completion and restricted to the target family. Every supplied game must be incorporated. Authenticated state and outcomes outrank model hypotheses.",
        "output": "A complete cumulative replacement dossier for the target family, with direct findings and cross-family transfers kept explicitly separate.",
        "game_semantics": persuasion_information_semantics() if game_family == "persuasion" else None,
    }
    if game_family == "negotiation":
        contract["price_transfer"] = "Make every reusable price comparison in stable_direct_tendencies, opponent_model_of_us, uncertainties, executive_model, prompt_synopsis, and recommended_counterpolicy dimensionless. Divide by the authenticated acting player's reservation value when visible, or use own/opponent share of visible total surplus under complete information, and state the denominator and role. If no authenticated denominator exists, describe direction, repetition, or relative movement and omit the nominal amount. Raw nominal prices may remain only in archival direct-evidence fields that are not projected into a live turn."
    return contract


_DEFAULT_EFFORT = "max"
DEFAULT_FAMILY_MODELS = {
    "bargaining": "gpt-5.6-luna",
    "negotiation": "gpt-5.6-terra",
    "persuasion": "gpt-5.6-sol",
}
DEFAULT_MAX_BATCH_GAMES = 5
_DEFAULT_MAX_BATCH_CHARS = 700_000
_SYNTHESIS_PROJECTION_VERSION = "compact-behavioral-v3"
_COMPACT_EVIDENCE_KIND = "named-opponent-game-evidence-compact-v1"
_PACKED_EVIDENCE_KIND = "named-opponent-game-evidence-packed-ref-v1"
_INPUT_REF_KIND = "incremental-dossier-input-blob-ref-v1"
_COMPLETION_KINDS = {"game_completed", "game_completed_during_opponent_turn", "late_game_reconciled"}
_CENSORED_DOSSIER_OUTCOMES = {"timeout"}


def _dossier_semantics_compatible(game_family: str, dossier: dict[str, Any]) -> bool:
    """Reject persuasion memory synthesized under the ambiguous legacy field interpretation."""
    return game_family != "persuasion" or dossier.get("game_semantics_version") == PERSUASION_INFORMATION_SEMANTICS_VERSION


def _persuasion_call_semantics_status(source_call: dict[str, Any]) -> dict[str, str]:
    """Mark legacy model hypotheses so offline synthesis cannot mistake them for game mechanics."""
    request = source_call.get("request") if isinstance(source_call.get("request"), dict) else {}
    user = request.get("user")
    try:
        payload = json.loads(user) if isinstance(user, str) else user
    except json.JSONDecodeError:
        payload = None
    semantics = payload.get("game_semantics") if isinstance(payload, dict) and isinstance(payload.get("game_semantics"), dict) else {}
    if semantics.get("version") == PERSUASION_INFORMATION_SEMANTICS_VERSION:
        return {"status": "explicit-corrected-semantics", "version": PERSUASION_INFORMATION_SEMANTICS_VERSION}
    return {
        "status": "legacy-ambiguous-semantics",
        "constraint": "Do not inherit any claim that is_seller_know_cv=false means the seller lacks current-product-quality information; the seller always observes current quality.",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _input_receipt(*, root: Path, revision_dir: Path, input_packet: dict[str, Any]) -> dict[str, object]:
    text = _canonical(input_packet)
    reference = reference_text_blob(root=root, receipt_dir=revision_dir, value=text)
    digest = _sha(input_packet)
    if reference.get("sha256") != digest:
        raise RuntimeError("incremental dossier input hash disagrees with its immutable text blob")
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": _INPUT_REF_KIND,
        "input_sha256": digest,
        "input_ref": reference,
    }


def _load_input_packet(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("kind") != _INPUT_REF_KIND:
        return value
    reference = value.get("input_ref")
    if not isinstance(reference, dict):
        raise RuntimeError(f"incremental dossier input receipt has no immutable reference: {path}")
    text = load_referenced_text_blob(receipt_dir=path.parent, reference=reference)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != value.get("input_sha256"):
        raise RuntimeError(f"incremental dossier input receipt failed SHA-256 verification: {path}")
    packet = json.loads(text)
    if not isinstance(packet, dict) or _canonical(packet) != text:
        raise RuntimeError(f"incremental dossier input blob is not canonical JSON: {path}")
    return packet


def _metadata_value(metadata: object) -> object:
    if hasattr(metadata, "__dict__"):
        return metadata.__dict__
    if isinstance(metadata, dict):
        return metadata
    return {"value": str(metadata)}


def resolve_dossier_family_models(
    family_models: Mapping[str, str] | None = None,
    *,
    model: str | None = None,
) -> dict[str, str]:
    """Resolve complete family-specific synthesis-model settings, with an optional global override."""
    resolved = dict(DEFAULT_FAMILY_MODELS)
    if family_models is not None:
        unknown = set(family_models) - set(GLEE_FAMILIES)
        if unknown:
            raise ValueError(f"unsupported dossier model families: {sorted(unknown)}")
        resolved.update(family_models)
    if model is not None:
        resolved = {family: model for family in GLEE_FAMILIES}
    for family, selected in resolved.items():
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"dossier model for {family} must be a non-empty string")
    return resolved


def _opponent_from_game(game: dict[str, Any]) -> tuple[str, str] | None:
    opponent = game.get("opponent")
    if not isinstance(opponent, dict) or opponent.get("type") == "hidden":
        return None
    name = normalize_opponent_name(opponent.get("name"))
    if not name:
        return None
    return named_opponent_id(name), name


class ConsumedGameUpdate(BaseModel):
    """Explicit evidence-accounting receipt for one game consumed in the latest batch."""

    game_id: str = Field(min_length=1)
    game_family: Literal["bargaining", "negotiation", "persuasion"]
    evidence_update: str = Field(min_length=1, max_length=4000)


class TransferredHypothesis(BaseModel):
    """One explicitly cross-family hypothesis that never becomes direct evidence."""

    source_family: Literal["bargaining", "negotiation", "persuasion"]
    hypothesis: str = Field(min_length=1, max_length=3000)
    confidence: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=3000)


class DossierConfidenceProfile(BaseModel):
    """Separate local descriptive confidence from broader generalization claims."""

    within_observed_context: int = Field(ge=0, le=100)
    cross_game_generalization: int = Field(ge=0, le=100)
    cross_family_transfer: int = Field(ge=0, le=100)


class IncrementalDossierDraft(BaseModel):
    """Model-authored complete replacement state for one opponent-family lane."""

    game_family: Literal["bargaining", "negotiation", "persuasion"]
    confidence: int = Field(ge=0, le=100)
    confidence_profile: DossierConfidenceProfile
    executive_model: str = Field(min_length=1, max_length=8000)
    latest_evidence_update: str = Field(min_length=1, max_length=6000)
    latest_game_updates: list[ConsumedGameUpdate] = Field(min_length=1, max_length=64)
    direct_evidence_model: str = Field(min_length=1, max_length=8000)
    stable_direct_tendencies: list[str] = Field(max_length=24)
    role_conditioning: str = Field(min_length=1, max_length=5000)
    phase_conditioning: str = Field(min_length=1, max_length=5000)
    response_to_pressure: str = Field(min_length=1, max_length=5000)
    adaptation_and_learning: str = Field(min_length=1, max_length=6000)
    communication_and_truthfulness: str = Field(min_length=1, max_length=6000)
    timing_and_resource_policy: str = Field(min_length=1, max_length=5000)
    opponent_model_of_us: str = Field(min_length=1, max_length=6000)
    exploitable_regularities: list[str] = Field(max_length=20)
    contradictions_and_counterexamples: list[str] = Field(max_length=20)
    sibling_context_assessment: str = Field(min_length=1, max_length=6000)
    transferred_hypotheses: list[TransferredHypothesis] = Field(max_length=20)
    uncertainties: list[str] = Field(max_length=20)
    recommended_counterpolicy: str = Field(min_length=1, max_length=6000)
    prompt_synopsis: str = Field(min_length=1, max_length=3600)

    @field_validator("prompt_synopsis")
    @classmethod
    def _clean_prompt_synopsis(cls, value: str) -> str:
        return validate_future_synopsis(value)


class IncrementalNamedOpponentStore:
    """Own the durable inbox, evidence bundles, revision chains, and atomic current pointers."""

    def __init__(self, *, root: Path, project_root: Path, prompts_dir: Path | None = None, sibling_context_root: Path | None = None) -> None:
        self.root = root
        self.project_root = project_root.resolve()
        self.prompts_dir = prompts_dir or self.project_root / "prompts"
        self.jobs_root = self.root / "jobs"
        self.opponents_root = self.root / "opponents"
        self.sibling_context_root = sibling_context_root or self.root

    def _source_reference(self, source_run: Path) -> dict[str, str]:
        source = source_run.resolve()
        try:
            relative = source.relative_to(self.project_root)
        except ValueError:
            return {"kind": "absolute", "path": str(source), "label": source.name}
        return {"kind": "project-relative", "path": str(relative), "label": source.name}

    def resolve_source_run(self, reference: dict[str, Any]) -> Path:
        """Resolve one job source-run locator against this store's project root."""
        path = Path(str(reference["path"]))
        return (self.project_root / path).resolve() if reference.get("kind") == "project-relative" else path.resolve()

    def _resolve_source(self, reference: dict[str, Any]) -> Path:
        """Compatibility alias for callers that predate the public source-reference contract."""
        return self.resolve_source_run(reference)

    def enqueue_completed_game(
        self,
        *,
        source_run: Path,
        final_game: dict[str, Any],
        completed_at: str | None = None,
        completion_order: int | None = None,
    ) -> dict[str, object] | None:
        """Publish a cheap immutable job reference; this method never invokes a model."""
        identity = _opponent_from_game(final_game)
        if identity is None:
            return None
        opponent_key, opponent_name = identity
        game_id = normalize_opponent_name(final_game.get("game_id"))
        family = str(final_game.get("game_family") or "")
        if not game_id or family not in GLEE_FAMILIES:
            raise ValueError("completed named-opponent game must identify a supported family and game_id")
        source = self._source_reference(source_run)
        game_sha = _sha(final_game)
        source_path = source_run.resolve()
        preferred_game_path = source_path / "games" / f"{family}-{game_id}.json"
        game_path = preferred_game_path if preferred_game_path.is_file() else None
        if game_path is None:
            for candidate in sorted((source_path / "games").glob("*.json")):
                candidate_game = _read_json(candidate)
                if candidate_game.get("game_id") == game_id and candidate_game.get("game_family") == family and _sha(candidate_game) == game_sha:
                    game_path = candidate
                    break
        if game_path is None or _sha(_read_json(game_path)) != game_sha:
            raise RuntimeError(f"completed game is not durably stored under its source run: {source_path}/{family}/{game_id}")
        game_relative_path = str(game_path.relative_to(source_path))
        job_id = _sha({"source": source, "game_id": game_id, "final_game_sha256": game_sha})
        job = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "named-opponent-update-job",
            "job_id": job_id,
            "opponent": {"id": opponent_key, "name": opponent_name},
            "game_id": game_id,
            "game_family": family,
            "completed_at": completed_at or _now(),
            "completion_order": completion_order,
            "source_run": source,
            "final_game_sha256": game_sha,
            "final_game_ref": {"kind": "source-run-relative", "path": game_relative_path, "object_sha256": game_sha},
        }
        path = self.jobs_root / opponent_key / f"{job_id}.ref.json"
        if path.is_file():
            existing = _read_json(path)
            if existing.get("job_id") != job_id or existing.get("final_game_sha256") != game_sha:
                raise RuntimeError(f"named-opponent job collision: {path}")
            return existing
        _atomic_json(path, job)
        return job

    def enqueue_source_run(self, source_run: Path, *, require_complete: bool = True, families: set[str] | None = None, exclude_outcomes: set[str] | None = None) -> dict[str, object]:
        """Backfill one run as the same per-game jobs emitted during future live play."""
        selected_families = set(GLEE_FAMILIES) if families is None else set(families)
        unknown = selected_families - set(GLEE_FAMILIES)
        if unknown:
            raise ValueError(f"unsupported GLEE families: {sorted(unknown)}")
        excluded_outcomes = {str(value) for value in (exclude_outcomes or set())}
        source_run = source_run.resolve()
        if require_complete and not (source_run / "complete.json").is_file():
            raise RuntimeError(f"source run is not complete: {source_run}")
        events_path = source_run / "events.jsonl"
        events = _read_jsonl(events_path) if events_path.is_file() else []
        completions: dict[str, tuple[str | None, int]] = {}
        for line_number, event in enumerate(events, start=1):
            if event.get("kind") not in _COMPLETION_KINDS or event.get("game_id") is None:
                continue
            completions[str(event["game_id"])] = (str(event.get("ts")) if event.get("ts") else None, line_number)
        jobs: list[dict[str, object]] = []
        excluded = 0
        games_root = source_run / "games"
        if games_root.is_dir():
            for path in sorted(games_root.glob("*.json")):
                game = _read_json(path)
                if game.get("game_family") not in selected_families:
                    excluded += 1
                    continue
                result = game.get("result") if isinstance(game.get("result"), dict) else {}
                if str(result.get("outcome")) in excluded_outcomes:
                    excluded += 1
                    continue
                game_id = str(game.get("game_id") or "")
                completed_at, completion_order = completions.get(game_id, (None, 0))
                job = self.enqueue_completed_game(source_run=source_run, final_game=game, completed_at=completed_at, completion_order=completion_order)
                if job is not None:
                    jobs.append(job)
        return {"schema_version": _SCHEMA_VERSION, "source_run": str(source_run), "named_game_count": len(jobs), "excluded_game_count": excluded, "families": sorted(selected_families), "excluded_outcomes": sorted(excluded_outcomes), "job_ids": [job["job_id"] for job in jobs]}

    def lane_keys(self) -> list[tuple[str, str]]:
        if not self.jobs_root.is_dir():
            return []
        lanes: set[tuple[str, str]] = set()
        for directory in self.jobs_root.iterdir():
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                job = _read_json(path)
                family = str(job.get("game_family") or "")
                if family in GLEE_FAMILIES:
                    lanes.add((directory.name, family))
        return sorted(lanes)

    def _ordered_jobs(self, opponent_key: str, game_family: str) -> list[dict[str, Any]]:
        directory = self.jobs_root / opponent_key
        if not directory.is_dir():
            return []
        jobs = [job for path in directory.glob("*.json") if (job := _read_json(path)).get("game_family") == game_family]
        jobs.sort(key=lambda job: (str(job.get("completed_at") or ""), int(job.get("completion_order") or 0), str(job["job_id"])))
        return jobs

    def _current_pointer_path(self, opponent_key: str, game_family: str) -> Path:
        return self.opponents_root / opponent_key / game_family / "current.json"

    @staticmethod
    def _load_current_from_root(root: Path, opponent_key: str, game_family: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        pointer_path = root / "opponents" / opponent_key / game_family / "current.json"
        if not pointer_path.is_file():
            return None, None
        pointer = _read_json(pointer_path)
        if pointer.get("schema_version") != _SCHEMA_VERSION or pointer.get("opponent_id") != opponent_key or pointer.get("game_family") != game_family:
            raise RuntimeError(f"unsupported incremental dossier pointer: {pointer_path}")
        dossier_path = root / str(pointer["path"])
        if _sha_file(dossier_path) != pointer.get("sha256"):
            raise RuntimeError(f"incremental dossier failed SHA-256 verification: {dossier_path}")
        dossier = _read_json(dossier_path)
        return pointer, dossier

    def _load_current(self, opponent_key: str, game_family: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return self._load_current_from_root(self.root, opponent_key, game_family)

    def pending_jobs(self, opponent_key: str, game_family: str) -> list[dict[str, Any]]:
        _pointer, dossier = self._load_current(opponent_key, game_family)
        revision = dossier.get("revision") if isinstance(dossier, dict) and isinstance(dossier.get("revision"), dict) else {}
        processed = {str(value) for value in revision.get("processed_job_ids") or []}
        return [job for job in self._ordered_jobs(opponent_key, game_family) if str(job["job_id"]) not in processed and self._job_is_modeling_eligible(job)]

    def load_job_final_game(self, job: dict[str, Any]) -> dict[str, Any]:
        """Load and verify a job's canonical final game through either storage representation."""
        legacy = job.get("final_game")
        if isinstance(legacy, dict):
            final_game = copy.deepcopy(legacy)
        else:
            reference = job.get("final_game_ref")
            if not isinstance(reference, dict) or reference.get("kind") != "source-run-relative":
                raise RuntimeError(f"named-opponent job has no supported final-game reference: {job.get('job_id')}")
            source_run = self.resolve_source_run(job["source_run"])
            path = (source_run / str(reference.get("path") or "")).resolve()
            try:
                path.relative_to(source_run)
            except ValueError as error:
                raise RuntimeError(f"named-opponent final-game reference escapes its source run: {path}") from error
            final_game = _read_json(path)
            if _sha(final_game) != reference.get("object_sha256"):
                raise RuntimeError(f"named-opponent final-game reference failed SHA-256 verification: {path}")
        if _sha(final_game) != job.get("final_game_sha256"):
            raise RuntimeError(f"named-opponent final game changed: {job.get('job_id')}")
        return final_game

    def _job_final_game(self, job: dict[str, Any]) -> dict[str, Any]:
        """Compatibility alias for callers that predate the public immutable-reference contract."""
        return self.load_job_final_game(job)

    def _job_is_modeling_eligible(self, job: dict[str, Any]) -> bool:
        final_game = self.load_job_final_game(job)
        result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
        outcome = str(result.get("outcome") or "").strip().casefold()
        status = str(final_game.get("status") or "").strip().casefold()
        return outcome not in _CENSORED_DOSSIER_OUTCOMES and status not in _CENSORED_DOSSIER_OUTCOMES

    def has_pending(self, opponent_key: str, game_family: str) -> bool:
        return bool(self.pending_jobs(opponent_key, game_family))

    @staticmethod
    def _direct_sibling_projection(dossier: dict[str, Any]) -> dict[str, object]:
        return {
            "game_family": dossier.get("game_family"),
            "confidence": dossier.get("confidence"),
            "confidence_profile": dossier.get("confidence_profile"),
            "direct_evidence_model": dossier.get("direct_evidence_model"),
            "stable_direct_tendencies": dossier.get("stable_direct_tendencies") or [],
            "role_conditioning": dossier.get("role_conditioning"),
            "phase_conditioning": dossier.get("phase_conditioning"),
            "response_to_pressure": dossier.get("response_to_pressure"),
            "adaptation_and_learning": dossier.get("adaptation_and_learning"),
            "communication_and_truthfulness": dossier.get("communication_and_truthfulness"),
            "timing_and_resource_policy": dossier.get("timing_and_resource_policy"),
            "opponent_model_of_us": dossier.get("opponent_model_of_us"),
            "contradictions_and_counterexamples": dossier.get("contradictions_and_counterexamples") or [],
            "uncertainties": dossier.get("uncertainties") or [],
            "direct_game_ids": (dossier.get("revision") or {}).get("direct_game_ids") or [],
            "direct_game_count": (dossier.get("revision") or {}).get("direct_game_count") or 0,
            "recursive_transfer_excluded": True,
        }

    def _sibling_family_inputs(self, opponent_key: str, target_family: str) -> list[dict[str, object]]:
        siblings: list[dict[str, object]] = []
        for family in GLEE_FAMILIES:
            if family == target_family:
                continue
            pointer, dossier = self._load_current_from_root(self.sibling_context_root, opponent_key, family)
            if pointer is not None and dossier is not None and _dossier_semantics_compatible(family, dossier):
                siblings.append(
                    {
                        "game_family": family,
                        "availability": "incremental-family-dossier",
                        "provenance": {
                            "revision_number": pointer.get("revision_number"),
                            "revision_sha256": pointer.get("sha256"),
                            "generated_at": dossier.get("generated_at"),
                        },
                        "direct_dossier_projection": self._direct_sibling_projection(dossier),
                    }
                )
                continue
            siblings.append({"game_family": family, "availability": "none", "provenance": None, "direct_dossier_projection": None})
        return siblings

    @staticmethod
    def _event_game_id(event: dict[str, Any], turn_games: dict[str, str]) -> str:
        if event.get("game_id") is not None:
            return str(event["game_id"])
        game = event.get("game")
        if isinstance(game, dict) and game.get("game_id") is not None:
            return str(game["game_id"])
        turn_id = normalize_opponent_name(event.get("turn_id"))
        return turn_games.get(turn_id, "")

    def _build_game_bundle(self, job: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
        opponent_key = str(job["opponent"]["id"])
        game_family = str(job["game_family"])
        job_id = str(job["job_id"])
        legacy_path = self.root / Path("opponents") / opponent_key / game_family / "games" / f"{job_id}.json"
        reference_path = self.root / Path("opponents") / opponent_key / game_family / "games" / f"{job_id}.ref.json"
        for path in (legacy_path, reference_path):
            if path.is_file():
                return _read_json(path), path, _sha_file(path)
        path = reference_path
        source_run = self.resolve_source_run(job["source_run"])
        events_path = source_run / "events.jsonl"
        calls_path = source_run / "llm_calls.jsonl"
        if not events_path.is_file() or not calls_path.is_file():
            raise FileNotFoundError(f"named-opponent job source is incomplete: {source_run}")
        turn_games, worker_receipts, call_receipts = _event_linkage(event for _line_number, event in _iter_jsonl(events_path))
        target_game_id = str(job["game_id"])
        target_turn_ids = {turn_id for turn_id, game_id in turn_games.items() if game_id == target_game_id}
        model_calls: list[dict[str, object]] = []
        for line_number, call in _iter_jsonl(calls_path):
            hydrated_call = hydrate_call_request(call, log_path=calls_path)
            payload = _parse_worker_request(hydrated_call)
            if payload is None:
                continue
            payload_identity = _opponent_from_game({"opponent": payload.get("opponent")})
            if payload_identity is None or payload_identity[0] != opponent_key:
                continue
            turn_receipt = payload.get("turn_receipt") if isinstance(payload.get("turn_receipt"), dict) else {}
            turn_id = normalize_opponent_name(turn_receipt.get("turn_id"))
            call_id = str(call.get("call_id") or "")
            call_receipt = call_receipts.get(call_id, {})
            if not turn_id:
                turn_id = str(call_receipt.get("turn_id") or "")
            game_id = turn_games.get(turn_id) or call_receipt.get("game_id")
            if game_id is None and ":r" in turn_id:
                game_id = turn_id.split(":r", 1)[0]
            if str(game_id or "") != target_game_id:
                continue
            target_turn_ids.add(turn_id)
            worker = worker_receipts.get(turn_id, call_receipt)
            model_calls.append(
                {
                    "source_log_line": line_number,
                    "source_call_sha256": _sha(call),
                    "game_id": target_game_id,
                    "turn_id": turn_id,
                    "game_family": payload.get("game_family"),
                    "phase": payload.get("phase"),
                    "branch": (worker.get("call_branches") or {}).get(call_id) or call_receipt.get("branch"),
                    "selected": bool(call_id and call_id == worker.get("selected_call_id")),
                    "selected_action": worker.get("selected_action") if call_id and call_id == worker.get("selected_call_id") else None,
                    "call": hydrated_call,
                }
            )
        relevant_events = []
        for line_number, event in _iter_jsonl(events_path):
            event_game_id = self._event_game_id(event, turn_games)
            turn_id = normalize_opponent_name(event.get("turn_id"))
            if event_game_id == target_game_id or turn_id in target_turn_ids:
                relevant_events.append({"source_log_line": line_number, "source_event_sha256": _sha(event), "event": copy.deepcopy(event)})
        full_bundle = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "named-opponent-game-evidence",
            "job_id": job_id,
            "opponent": copy.deepcopy(job["opponent"]),
            "game_id": target_game_id,
            "game_family": job["game_family"],
            "completed_at": job.get("completed_at"),
            "source_run": copy.deepcopy(job["source_run"]),
            "evidence_contract": {
                "authentication": "The final game and attributed server events are authenticated evidence. Model calls are hypotheses and cognitive traces, not facts about the opponent.",
                "coverage": "Every source model call linked to this game and normalized named identity is included, regardless of branch selection, success, failure, or partial output.",
                "identity": "The join uses a normalized display name because the competition interface does not expose an immutable opponent identifier.",
            },
            "final_game": self.load_job_final_game(job),
            "events": relevant_events,
            "model_calls": model_calls,
        }
        if _sha(full_bundle["final_game"]) != job["final_game_sha256"]:
            raise RuntimeError(f"final game changed while resolving named-opponent job {job_id}")
        compact_evidence = {
            "schema_version": _SCHEMA_VERSION,
            "kind": _COMPACT_EVIDENCE_KIND,
            "job_id": job_id,
            "opponent": copy.deepcopy(job["opponent"]),
            "game_id": target_game_id,
            "game_family": job["game_family"],
            "completed_at": job.get("completed_at"),
            "source_run": copy.deepcopy(job["source_run"]),
            "source_logs": {
                "events": {"path": "events.jsonl", "line_receipts": len(relevant_events)},
                "model_calls": {"path": "llm_calls.jsonl", "line_receipts": len(model_calls)},
            },
            "final_game_sha256": job["final_game_sha256"],
            "projection": self._synthesis_projection(full_bundle),
        }
        _atomic_json(path, compact_evidence)
        return compact_evidence, path, _sha_file(path)

    def _synthesis_projection(self, bundle: dict[str, Any]) -> dict[str, object]:
        """Keep exact behavioral evidence while removing cumulative transport duplication."""
        if bundle.get("kind") == _COMPACT_EVIDENCE_KIND:
            projection = bundle.get("projection")
            if not isinstance(projection, dict):
                raise RuntimeError(f"compact game evidence has no projection: {bundle.get('job_id')}")
            return copy.deepcopy(projection)
        if bundle.get("kind") == _PACKED_EVIDENCE_KIND:
            reference = bundle.get("projection_ref")
            if not isinstance(reference, dict):
                raise RuntimeError(f"packed game evidence has no projection reference: {bundle.get('job_id')}")
            return load_json_object(root=self.root, reference=reference, object_sha256_value=str(reference.get("object_sha256") or ""))
        calls: list[dict[str, object]] = []
        for record in bundle.get("model_calls") or []:
            source_call = record.get("call") if isinstance(record.get("call"), dict) else {}
            call = {
                key: copy.deepcopy(source_call[key])
                for key in (
                    "schema_version",
                    "attempt",
                    "backend",
                    "call_id",
                    "effort",
                    "elapsed_s",
                    "error",
                    "input_chars",
                    "model",
                    "ok",
                    "output_chars",
                    "prompt_sha256",
                    "prompt_version",
                    "reasoning_tokens",
                    "response",
                    "role",
                    "thinking",
                    "tokens_in",
                    "tokens_out",
                    "transient",
                    "ts",
                )
                if key in source_call
            }
            request = source_call.get("request") if isinstance(source_call.get("request"), dict) else {}
            call["request_receipt"] = {
                "system_sha256": request.get("system_sha256"),
                "user_sha256": request.get("user_sha256"),
            }
            if bundle.get("game_family") == "persuasion":
                call["persuasion_semantics"] = _persuasion_call_semantics_status(source_call)
            calls.append(
                {
                    "source_log_line": record.get("source_log_line"),
                    "source_call_sha256": record.get("source_call_sha256"),
                    "turn_id": record.get("turn_id"),
                    "game_family": record.get("game_family"),
                    "phase": record.get("phase"),
                    "branch": record.get("branch"),
                    "selected": record.get("selected"),
                    "selected_action": record.get("selected_action"),
                    "call": call,
                }
            )
        events: list[dict[str, object]] = []
        for record in bundle.get("events") or []:
            source_event = record.get("event") if isinstance(record.get("event"), dict) else {}
            event = {
                key: copy.deepcopy(source_event[key])
                for key in ("schema_version", "ts", "kind", "game_id", "turn_id", "family", "status", "result")
                if key in source_event
            }
            decision = source_event.get("decision") if isinstance(source_event.get("decision"), dict) else None
            if decision is not None:
                event["decision_receipt"] = {
                    key: copy.deepcopy(decision[key])
                    for key in ("action", "proposal", "fallback", "fallback_reason", "selection_branch", "tetrad_update")
                    if key in decision
                }
            events.append(
                {
                    "source_log_line": record.get("source_log_line"),
                    "source_event_sha256": record.get("source_event_sha256"),
                    "event": event,
                }
            )
        final_game = copy.deepcopy(bundle["final_game"])
        semantics = model_game_semantics(final_game)
        if semantics is not None:
            final_game["game_state"] = model_visible_game_state(final_game)
        return {
            "schema_version": _SCHEMA_VERSION,
            "projection_contract": {
                "version": _SYNTHESIS_PROJECTION_VERSION,
                "source_evidence": "Immutable source-run JSONL records remain canonical; the game evidence receipt pins every cited event and model call by source path, line number, and canonical SHA-256.",
                "projection": "This cloud-facing view retains a semantically normalized copy of the authenticated final game, compact event receipts, every model response, branch selection, selected action, and call metrics while removing repeated cumulative request histories, schemas, commands, provider streams, and activation memory.",
            },
            "game_semantics": semantics,
            "opponent": bundle["opponent"],
            "game_id": bundle["game_id"],
            "game_family": bundle["game_family"],
            "completed_at": bundle.get("completed_at"),
            "evidence_contract": bundle["evidence_contract"],
            "final_game": final_game,
            "events": events,
            "model_calls": calls,
        }

    def _lane_lock(self, opponent_key: str, game_family: str) -> tuple[int, Path]:
        path = self.opponents_root / opponent_key / game_family / ".update.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor, path

    @staticmethod
    def _unlock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def process_next(
        self,
        opponent_key: str,
        game_family: str,
        *,
        model: str | None = None,
        effort: str = _DEFAULT_EFFORT,
        timeout_s: int = 3600,
        max_batch_games: int = DEFAULT_MAX_BATCH_GAMES,
        max_batch_chars: int = _DEFAULT_MAX_BATCH_CHARS,
    ) -> dict[str, object] | None:
        """Apply one bounded ordered batch to one opponent-family dossier under its serial lane lock."""
        if game_family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported GLEE family: {game_family}")
        selected_model = resolve_dossier_family_models(model=model)[game_family]
        if max_batch_games < 1:
            raise ValueError("max_batch_games must be positive")
        if max_batch_chars < 1:
            raise ValueError("max_batch_chars must be positive")
        descriptor, _lock_path = self._lane_lock(opponent_key, game_family)
        try:
            pending = self.pending_jobs(opponent_key, game_family)
            if not pending:
                return None
            pointer, current = self._load_current(opponent_key, game_family)
            if current is not None and not _dossier_semantics_compatible(game_family, current):
                raise RuntimeError(f"legacy {game_family} dossier cannot be continued under corrected game semantics; regenerate this lane from raw game evidence in a fresh root")
            expected_pointer_sha = str(pointer.get("sha256")) if pointer is not None else None
            parent_sha = expected_pointer_sha
            parent_revision = int(pointer.get("revision_number") or 0) if pointer is not None else 0
            revision_number = parent_revision + 1
            sibling_inputs = self._sibling_family_inputs(opponent_key, game_family)
            sibling_provenance = [
                {"game_family": sibling["game_family"], "availability": sibling["availability"], "provenance": sibling["provenance"]}
                for sibling in sibling_inputs
            ]
            batch: list[tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, object]]] = []
            batch_limit = 1 if pointer is None else max_batch_games
            for job in pending[:batch_limit]:
                if job.get("game_family") != game_family:
                    raise RuntimeError(f"foreign-family job entered {opponent_key}/{game_family}: {job.get('job_id')}")
                bundle, bundle_path, bundle_sha = self._build_game_bundle(job)
                projection = self._synthesis_projection(bundle)
                candidate = [*batch, (job, bundle, bundle_path, bundle_sha, projection)]
                candidate_packet = {
                    "schema_version": _SCHEMA_VERSION,
                    "task": "Replace one opponent-family dossier after incorporating an ordered batch of newly completed games from that same family.",
                    "transition": {
                        "opponent_id": opponent_key,
                        "game_family": game_family,
                        "revision_number": revision_number,
                        "parent_revision_sha256": parent_sha,
                        "sibling_inputs": sibling_provenance,
                        "new_jobs": [
                            {
                                "job_id": item[0]["job_id"],
                                "game_id": item[0]["game_id"],
                                "game_family": item[0]["game_family"],
                                "completed_at": item[0].get("completed_at"),
                                "game_bundle_sha256": item[3],
                            }
                            for item in candidate
                        ],
                    },
                    "state_contract": _state_contract(game_family),
                    "current_family_dossier": current,
                    "sibling_family_dossiers": sibling_inputs,
                    "new_game_evidence_batch": [item[4] for item in candidate],
                }
                candidate_chars = len(_canonical(candidate_packet))
                if candidate_chars > max_batch_chars:
                    if batch:
                        break
                    raise RuntimeError(
                        f"single-game dossier projection exceeds max_batch_chars for {opponent_key}/{game_family}: "
                        f"{candidate_chars} > {max_batch_chars}"
                    )
                batch = candidate
            if not batch:
                raise RuntimeError(f"failed to select a pending dossier game for {opponent_key}/{game_family}")
            jobs = [item[0] for item in batch]
            transition_jobs = [
                {
                    "job_id": item[0]["job_id"],
                    "game_id": item[0]["game_id"],
                    "game_family": item[0]["game_family"],
                    "completed_at": item[0].get("completed_at"),
                    "game_bundle_sha256": item[3],
                }
                for item in batch
            ]
            transition_id = _sha(
                {
                    "opponent_id": opponent_key,
                    "game_family": game_family,
                    "parent_sha256": parent_sha,
                    "sibling_inputs": sibling_provenance,
                    "jobs": transition_jobs,
                    "model": selected_model,
                    "effort": effort,
                    "synthesis_projection_version": _SYNTHESIS_PROJECTION_VERSION,
                }
            )
            relative_dir = Path("opponents") / opponent_key / game_family / "revisions" / f"{revision_number:06d}-{transition_id[:20]}"
            revision_dir = self.root / relative_dir
            legacy_input_path = revision_dir / "input.json"
            input_path = revision_dir / "input.ref.json"
            dossier_path = revision_dir / "dossier.json"
            input_packet = {
                "schema_version": _SCHEMA_VERSION,
                "task": "Replace one opponent-family dossier after incorporating an ordered batch of newly completed games from that same family.",
                "transition": {
                    "opponent_id": opponent_key,
                    "game_family": game_family,
                    "revision_number": revision_number,
                    "parent_revision_sha256": parent_sha,
                    "sibling_inputs": sibling_provenance,
                    "new_jobs": transition_jobs,
                },
                "state_contract": _state_contract(game_family),
                "current_family_dossier": current,
                "sibling_family_dossiers": sibling_inputs,
                "new_game_evidence_batch": [item[4] for item in batch],
            }
            existing_input_paths = [path for path in (legacy_input_path, input_path) if path.is_file()]
            if len(existing_input_paths) > 1:
                raise RuntimeError(f"incremental dossier revision has competing input receipts: {revision_dir}")
            if existing_input_paths:
                if _load_input_packet(existing_input_paths[0]) != input_packet:
                    raise RuntimeError(f"incremental dossier transition input changed: {existing_input_paths[0]}")
            else:
                _atomic_json(input_path, _input_receipt(root=self.root, revision_dir=revision_dir, input_packet=input_packet))
            if dossier_path.is_file():
                dossier = _read_json(dossier_path)
                revision = dossier.get("revision") if isinstance(dossier.get("revision"), dict) else {}
                if dossier.get("schema_version") != _SCHEMA_VERSION or dossier.get("game_family") != game_family or revision.get("transition_id") != transition_id:
                    raise RuntimeError(f"invalid resumable incremental dossier revision: {dossier_path}")
                IncrementalDossierDraft.model_validate({name: dossier.get(name) for name in IncrementalDossierDraft.model_fields})
            else:
                runner = ArenaCodexRunner(
                    prompts_dir=self.prompts_dir,
                    log_path=revision_dir / "synthesis-calls.jsonl",
                    session_dir=self.opponents_root / opponent_key / game_family / ".cli-session",
                    timeout_s=timeout_s,
                    validation_retries=2,
                    blob_root=self.root,
                )
                parsed, metadata = runner.call_structured(_UPDATE_ROLE, _canonical(input_packet), IncrementalDossierDraft, model=selected_model, effort=effort)
                draft = parsed.model_dump(mode="json")
                if draft["game_family"] != game_family:
                    raise RuntimeError(f"dossier synthesis returned family {draft['game_family']!r} for {game_family!r}")
                expected_games = [(str(job["game_id"]), str(job["game_family"])) for job in jobs]
                actual_games = [(str(item["game_id"]), str(item["game_family"])) for item in draft["latest_game_updates"]]
                if actual_games != expected_games:
                    raise RuntimeError(f"dossier synthesis did not account for the exact ordered game batch: {actual_games!r} != {expected_games!r}")
                previous_revision = current.get("revision") if isinstance(current, dict) and isinstance(current.get("revision"), dict) else {}
                processed_job_ids = [str(value) for value in previous_revision.get("processed_job_ids") or []]
                processed_job_ids.extend(str(job["job_id"]) for job in jobs)
                direct_game_ids = [str(value) for value in previous_revision.get("direct_game_ids") or []]
                for job in jobs:
                    game_id = str(job["game_id"])
                    if game_id not in direct_game_ids:
                        direct_game_ids.append(game_id)
                dossier = {
                    "schema_version": _SCHEMA_VERSION,
                    "game_semantics_version": PERSUASION_INFORMATION_SEMANTICS_VERSION if game_family == "persuasion" else None,
                    "opponent": {"id": opponent_key, "name": jobs[-1]["opponent"]["name"]},
                    "game_family": game_family,
                    "generated_at": _now(),
                    "revision": {
                        "number": revision_number,
                        "transition_id": transition_id,
                        "parent_revision_sha256": parent_sha,
                        "initialization": "first-completed-game-of-this-opponent-family" if pointer is None else None,
                        "sibling_inputs": sibling_provenance,
                        "new_jobs": [
                            {
                                **transition_job,
                                "game_bundle_path": str(item[2].relative_to(self.root)),
                            }
                            for transition_job, item in zip(transition_jobs, batch, strict=True)
                        ],
                        "processed_job_ids": processed_job_ids,
                        "direct_game_ids": direct_game_ids,
                        "direct_game_count": len(direct_game_ids),
                    },
                    "synthesis": {
                        "model": selected_model,
                        "effort": effort,
                        "role": _UPDATE_ROLE,
                        "mode": "current-family-dossier-plus-sibling-direct-projections-plus-ordered-direct-game-batch",
                        "batch_size": len(batch),
                        "call_metadata": _metadata_value(metadata),
                    },
                    **draft,
                }
                _atomic_json(dossier_path, dossier)
            dossier_sha = _sha_file(dossier_path)
            current_pointer, _current_dossier = self._load_current(opponent_key, game_family)
            actual_parent = str(current_pointer.get("sha256")) if current_pointer is not None else None
            if actual_parent != expected_pointer_sha:
                raise RuntimeError(f"incremental dossier parent changed during transition for {opponent_key}/{game_family}")
            pointer_value = {
                "schema_version": _SCHEMA_VERSION,
                "opponent_id": opponent_key,
                "opponent_name": jobs[-1]["opponent"]["name"],
                "game_family": game_family,
                "model": selected_model,
                "game_semantics_version": dossier.get("game_semantics_version"),
                "revision_number": revision_number,
                "transition_id": transition_id,
                "path": str(relative_dir / "dossier.json"),
                "sha256": dossier_sha,
                "updated_at": _now(),
            }
            _atomic_json(self._current_pointer_path(opponent_key, game_family), pointer_value)
            return {
                "schema_version": _SCHEMA_VERSION,
                "opponent_id": opponent_key,
                "opponent_name": jobs[-1]["opponent"]["name"],
                "game_family": game_family,
                "model": selected_model,
                "job_ids": [job["job_id"] for job in jobs],
                "game_ids": [job["game_id"] for job in jobs],
                "batch_size": len(batch),
                "revision_number": revision_number,
                "revision_sha256": dossier_sha,
                "path": pointer_value["path"],
            }
        finally:
            self._unlock(descriptor)

    def process_until_idle(
        self,
        *,
        max_workers: int = 4,
        model: str | None = None,
        family_models: Mapping[str, str] | None = None,
        families: set[str] | None = None,
        effort: str = _DEFAULT_EFFORT,
        timeout_s: int = 3600,
        max_batch_games: int = DEFAULT_MAX_BATCH_GAMES,
        max_batch_chars: int = _DEFAULT_MAX_BATCH_CHARS,
        event: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """Drain the current durable inbox with one active transition per opponent-family lane."""
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        selected_families = set(GLEE_FAMILIES) if families is None else set(families)
        unsupported_families = selected_families - set(GLEE_FAMILIES)
        if unsupported_families:
            raise ValueError(f"unsupported dossier families: {sorted(unsupported_families)}")
        resolved_models = resolve_dossier_family_models(family_models, model=model)
        completed: list[dict[str, object]] = []
        active: dict[Future[dict[str, object] | None], tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="glee-dossier-model") as pool:
            while True:
                for future in [future for future in active if future.done()]:
                    active.pop(future)
                    result = future.result()
                    if result is not None:
                        completed.append(result)
                        if event is not None:
                            event({"kind": "revision_published", **result})
                active_lanes = set(active.values())
                selected_lanes = [lane for lane in self.lane_keys() if lane[1] in selected_families]
                candidates = [lane for lane in selected_lanes if lane not in active_lanes and self.has_pending(*lane)]
                for opponent_key, game_family in candidates[: max_workers - len(active)]:
                    selected_model = resolved_models[game_family]
                    active[
                        pool.submit(
                            self.process_next,
                            opponent_key,
                            game_family,
                            model=selected_model,
                            effort=effort,
                            timeout_s=timeout_s,
                            max_batch_games=max_batch_games,
                            max_batch_chars=max_batch_chars,
                        )
                    ] = (opponent_key, game_family)
                if not active and not any(self.has_pending(*lane) for lane in selected_lanes):
                    break
                time.sleep(0.05)
        return {"schema_version": _SCHEMA_VERSION, "published_count": len(completed), "revisions": completed}

    def watch(
        self,
        *,
        max_workers: int = 4,
        model: str | None = None,
        family_models: Mapping[str, str] | None = None,
        families: set[str] | None = None,
        effort: str = _DEFAULT_EFFORT,
        timeout_s: int = 3600,
        max_batch_games: int = DEFAULT_MAX_BATCH_GAMES,
        max_batch_chars: int = _DEFAULT_MAX_BATCH_CHARS,
        poll_interval_s: float = 2.0,
        retry_delay_s: float = 30.0,
        stop_when_idle: Callable[[], bool] | None = None,
        event: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """Schedule independent opponent-family lanes until stopped or a supplied feeder is done and drained."""
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        selected_families = set(GLEE_FAMILIES) if families is None else set(families)
        unsupported_families = selected_families - set(GLEE_FAMILIES)
        if unsupported_families:
            raise ValueError(f"unsupported dossier families: {sorted(unsupported_families)}")
        resolved_models = resolve_dossier_family_models(family_models, model=model)
        active: dict[Future[dict[str, object] | None], tuple[str, str]] = {}
        retry_after: dict[tuple[str, str], float] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="glee-dossier-model") as pool:
            while True:
                now = time.monotonic()
                for future in [future for future in active if future.done()]:
                    lane = active.pop(future)
                    opponent_key, game_family = lane
                    try:
                        result = future.result()
                    except Exception as error:
                        retry_after[lane] = now + retry_delay_s
                        if event is not None:
                            event({"kind": "revision_failed", "opponent_id": opponent_key, "game_family": game_family, "model": resolved_models[game_family], "error": f"{type(error).__name__}: {error}", "retry_after_s": retry_delay_s})
                    else:
                        retry_after.pop(lane, None)
                        if result is not None and event is not None:
                            event({"kind": "revision_published", **result})
                active_lanes = set(active.values())
                selected_lanes = [lane for lane in self.lane_keys() if lane[1] in selected_families]
                candidates = [
                    lane
                    for lane in selected_lanes
                    if lane not in active_lanes and retry_after.get(lane, 0.0) <= now and self.has_pending(*lane)
                ]
                for opponent_key, game_family in candidates[: max_workers - len(active)]:
                    selected_model = resolved_models[game_family]
                    active[
                        pool.submit(
                            self.process_next,
                            opponent_key,
                            game_family,
                            model=selected_model,
                            effort=effort,
                            timeout_s=timeout_s,
                            max_batch_games=max_batch_games,
                            max_batch_chars=max_batch_chars,
                        )
                    ] = (opponent_key, game_family)
                    if event is not None:
                        event({"kind": "revision_started", "opponent_id": opponent_key, "game_family": game_family, "model": selected_model})
                if stop_when_idle is not None and stop_when_idle() and not active and not any(self.has_pending(*lane) for lane in selected_lanes):
                    break
                time.sleep(max(0.05, poll_interval_s))


class IncrementalNamedDossierReader:
    """Read atomic latest pointers without waiting for an updater lock or cloud call."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _load_latest(self, opponent_key: str, game_family: str) -> tuple[str, dict[str, Any]] | None:
        lane = (opponent_key, game_family)
        pointer_path = self.root / "opponents" / opponent_key / game_family / "current.json"
        if not pointer_path.is_file():
            with self._lock:
                return self._cache.get(lane)
        try:
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != _SCHEMA_VERSION or pointer.get("opponent_id") != opponent_key or pointer.get("game_family") != game_family:
                raise RuntimeError("pointer identity mismatch")
            sha = str(pointer["sha256"])
            with self._lock:
                cached = self._cache.get(lane)
                if cached is not None and cached[0] == sha:
                    if _dossier_semantics_compatible(game_family, cached[1]):
                        return cached
                    self._cache.pop(lane, None)
                    return None
            dossier_path = self.root / str(pointer["path"])
            if _sha_file(dossier_path) != sha:
                raise RuntimeError("revision hash mismatch")
            dossier = _read_json(dossier_path)
            if dossier.get("schema_version") != _SCHEMA_VERSION or dossier.get("game_family") != game_family:
                raise RuntimeError("revision schema mismatch")
            if not _dossier_semantics_compatible(game_family, dossier):
                with self._lock:
                    self._cache.pop(lane, None)
                return None
        except (OSError, KeyError, ValueError, json.JSONDecodeError, RuntimeError):
            with self._lock:
                return self._cache.get(lane)
        with self._lock:
            self._cache[lane] = (sha, dossier)
        return sha, dossier

    def view(self, opponent_name: str, game_family: str) -> dict[str, object] | None:
        if game_family not in GLEE_FAMILIES:
            return None
        opponent_key = named_opponent_id(opponent_name)
        latest = self._load_latest(opponent_key, game_family)
        if latest is None:
            return None
        sha, dossier = latest
        revision = dossier.get("revision") if isinstance(dossier.get("revision"), dict) else {}
        synthesis = dossier.get("synthesis") if isinstance(dossier.get("synthesis"), dict) else {}
        live_projection = live_dossier_projection(dossier)
        return {
            "provenance": {
                "kind": "incremental-opponent-family-synthesis",
                "opponent_id": opponent_key,
                "game_family": game_family,
                "revision_number": revision.get("number"),
                "revision_sha256": sha,
                "parent_revision_sha256": revision.get("parent_revision_sha256"),
                "generated_at": dossier.get("generated_at"),
                "model": synthesis.get("model"),
                "effort": synthesis.get("effort"),
            },
            "family_synopsis": {
                "game_family": game_family,
                "confidence": dossier.get("confidence"),
                "confidence_profile": dossier.get("confidence_profile"),
                "evidence_basis": "direct with explicitly labeled cross-family hypotheses",
                "direct_game_count": revision.get("direct_game_count") or 0,
                "live_projection": live_projection,
            },
        }
