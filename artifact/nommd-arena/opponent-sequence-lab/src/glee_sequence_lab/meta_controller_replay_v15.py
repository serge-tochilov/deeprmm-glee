"""Resumable chronological replay of the learned-forecast-independent 1.5-round Terra controller."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from nommd_arena.glee_meta_controller_v2 import CONDITIONAL_SURFACE_CONTRACT, ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT, FrozenCandidateSet, build_conditional_surface, build_planner_payload_v15, build_selector_payload_v15, freeze_planner_candidates, select_frozen_candidate
from nommd_arena.glee_nommd import nommd_candidate_plan_model, nommd_candidate_selection_model
from nommd_arena.glee_policy import action_model, apply_deterministic_safeguards, normalize_action
from nommd_arena.immutable_blob import load_referenced_text_blob
from nommd_arena.model_runner import ArenaCodexRunner

from .conditional_release import ConditionalTwinRelease
from .corpus import GLEE_FAMILIES, file_sha256, object_sha256
from .pre_terra_conditional_v2 import CandidateAction


REPLAY_CONTRACT = "glee-terra-meta-controller-v2.1.5-chronological-replay-v1"
SAMPLE_PROTOCOL = "glee-terra-meta-controller-v2.1.5-balanced-old-test-sample-v1"
DEFAULT_CASES_PER_FAMILY = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _metadata(value: object) -> dict[str, object]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


def submitted_action_from_event(row: Mapping[str, object]) -> dict[str, Any] | None:
    """Read either the current direct-action receipt or the older nested-decision receipt."""
    direct = row.get("action")
    if isinstance(direct, Mapping):
        return dict(direct)
    decision = row.get("decision")
    nested = decision.get("action") if isinstance(decision, Mapping) else None
    return dict(nested) if isinstance(nested, Mapping) else None


def collapse_candidate_projections(*, candidates: Sequence[Mapping[str, object]], family: str, phase: str) -> tuple[list[dict[str, object]], list[int]]:
    """Collapse predictor-equivalent legal actions while retaining an index for every frozen planner candidate."""
    unique: list[dict[str, object]] = []
    unique_index: dict[str, int] = {}
    projection_indices: list[int] = []
    for candidate in candidates:
        normalized = CandidateAction.from_mapping(candidate, family=family, phase=phase).receipt()
        digest = object_sha256(normalized)
        if digest not in unique_index:
            unique_index[digest] = len(unique)
            unique.append(normalized)
        projection_indices.append(unique_index[digest])
    return unique, projection_indices


def _median_unused(rows: Sequence[dict[str, object]], used_games: set[str]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: (str(row["source_call_ts"]), str(row["sample_id"])))
    if not ordered:
        raise ValueError("replay stratum has no eligible cases")
    center = (len(ordered) - 1) // 2
    offsets = [0]
    for distance in range(1, len(ordered)):
        offsets.extend((distance, -distance))
    for offset in offsets:
        index = center + offset
        if 0 <= index < len(ordered) and str(ordered[index]["game_id"]) not in used_games:
            return ordered[index]
    raise ValueError("replay stratum cannot supply a distinct game")


def _case_strata(family: str) -> tuple[tuple[str, Any], ...]:
    if family == "bargaining":
        return (
            ("hidden-player-1", lambda row: row["identity_scope"] == "hidden" and row["our_player"] == "player_1"),
            ("hidden-player-2", lambda row: row["identity_scope"] == "hidden" and row["our_player"] == "player_2"),
            ("known-player-1", lambda row: row["identity_scope"] == "known" and row["our_player"] == "player_1"),
            ("known-player-2", lambda row: row["identity_scope"] == "known" and row["our_player"] == "player_2"),
        )
    if family == "negotiation":
        return (
            ("hidden-accepted", lambda row: row["identity_scope"] == "hidden" and row["target_label"] == "accept"),
            ("hidden-rejected-known-horizon", lambda row: row["identity_scope"] == "hidden" and row["target_label"] == "reject" and row["horizon_known"] is True),
            ("known-rejected-complete-information", lambda row: row["identity_scope"] == "known" and row["target_label"] == "reject" and row["complete_information"] is True),
            ("known-rejected-incomplete-information", lambda row: row["identity_scope"] == "known" and row["target_label"] == "reject" and row["complete_information"] is False),
        )
    if family == "persuasion":
        return (
            ("hidden-buy", lambda row: row["identity_scope"] == "hidden" and row["target_label"] == "buy"),
            ("hidden-pass", lambda row: row["identity_scope"] == "hidden" and row["target_label"] == "pass"),
            ("known-buy", lambda row: row["identity_scope"] == "known" and row["target_label"] == "buy"),
            ("known-pass", lambda row: row["identity_scope"] == "known" and row["target_label"] == "pass"),
        )
    raise ValueError(f"unsupported replay family: {family!r}")


@dataclass(frozen=True)
class ReplayCase:
    ordinal: int
    stratum: str
    sample_id: str
    game_id: str
    turn_id: str
    family: str
    identity_scope: str
    target_label: str
    phase: str
    round_number: int
    source_call_ts: str
    source_run: Path
    source_llm_call_line: int
    source_user_sha256: str

    def receipt(self, *, project_root: Path) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "stratum": self.stratum,
            "sample_id": self.sample_id,
            "game_id": self.game_id,
            "turn_id": self.turn_id,
            "family": self.family,
            "identity_scope": self.identity_scope,
            "historical_direct_response": self.target_label,
            "phase": self.phase,
            "round_number": self.round_number,
            "source_call_ts": self.source_call_ts,
            "source_run": str(self.source_run.relative_to(project_root)),
            "source_llm_call_line": self.source_llm_call_line,
            "source_user_sha256": self.source_user_sha256,
        }


def select_replay_cases(corpus_dir: Path, *, cases_per_family: int = DEFAULT_CASES_PER_FAMILY) -> tuple[ReplayCase, ...]:
    """Select one fixed old-test case for each preregistered family stratum, then restore global chronology."""
    if cases_per_family != DEFAULT_CASES_PER_FAMILY:
        raise ValueError(f"the frozen balanced replay requires exactly {DEFAULT_CASES_PER_FAMILY} cases per family")
    root = corpus_dir.resolve()
    games = {str(row["game_id"]): row for row in pl.read_parquet(root / "games.parquet").to_dicts()}
    features = {str(row["sample_id"]): row for row in pl.read_parquet(root / "features.parquet").to_dicts()}
    rows: list[dict[str, object]] = []
    for target in pl.read_parquet(root / "targets.parquet").to_dicts():
        if target.get("chronological_split") != "test":
            continue
        feature = features.get(str(target["sample_id"]))
        game = games.get(str(target["game_id"]))
        if feature is None or game is None or feature.get("game_id") != game.get("game_id") or feature.get("family") != game.get("family"):
            raise RuntimeError("conditional replay corpus lost sample alignment")
        rows.append({**game, **target, **feature})
    selected: list[tuple[str, dict[str, object]]] = []
    for family in GLEE_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        used_games: set[str] = set()
        for label, predicate in _case_strata(family):
            chosen = _median_unused([row for row in family_rows if predicate(row)], used_games)
            used_games.add(str(chosen["game_id"]))
            selected.append((label, chosen))
    selected.sort(key=lambda item: (str(item[1]["source_call_ts"]), str(item[1]["sample_id"])))
    return tuple(
        ReplayCase(
            ordinal=index,
            stratum=label,
            sample_id=str(row["sample_id"]),
            game_id=str(row["game_id"]),
            turn_id=str(row["turn_id"]),
            family=str(row["family"]),
            identity_scope=str(row["identity_scope"]),
            target_label=str(row["target_label"]),
            phase=str(row["phase"]),
            round_number=int(row["round_number"]),
            source_call_ts=str(row["source_call_ts"]),
            source_run=Path(str(row["source_run"])).resolve(),
            source_llm_call_line=int(row["source_llm_call_line"]),
            source_user_sha256=str(row["source_user_sha256"]),
        )
        for index, (label, row) in enumerate(selected, start=1)
    )


class MetaControllerV15Replay:
    """Run planners, one warm local conditional pass, and selectors over a frozen balanced sample."""

    def __init__(self, *, project_root: Path, corpus_dir: Path, release_dir: Path, run_dir: Path, model: str = "gpt-5.6-terra", effort: str = "high", timeout_s: int = 600, max_workers: int = 3, runner_factory: Any | None = None, conditional_release: Any | None = None) -> None:
        if timeout_s <= 0 or max_workers <= 0:
            raise ValueError("replay timeout and worker count must be positive")
        self.project_root = project_root.resolve()
        self.corpus_dir = corpus_dir.resolve()
        self.release_dir = release_dir.resolve()
        self.run_dir = run_dir.resolve()
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_workers = max_workers
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cases = select_replay_cases(self.corpus_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.complete_path = self.run_dir / "complete.json"
        self.events_path = self.run_dir / "events.jsonl"
        self._event_lock = threading.Lock()
        self._runner_factory = runner_factory or self._default_runner
        self._conditional_release = conditional_release
        self._games = {str(row["game_id"]): row for row in pl.read_parquet(self.corpus_dir / "games.parquet").to_dicts()}
        self._targets = {str(row["sample_id"]): row for row in pl.read_parquet(self.corpus_dir / "targets.parquet").to_dicts()}
        self._features = {str(row["sample_id"]): row for row in pl.read_parquet(self.corpus_dir / "features.parquet").to_dicts()}
        self._events: dict[str, list[dict[str, object]]] = {}
        for row in pl.read_parquet(self.corpus_dir / "events.parquet").sort(["game_id", "event_index"]).to_dicts():
            self._events.setdefault(str(row["game_id"]), []).append(row)
        self._source_call_rows: dict[Path, list[str]] = {}
        self._source_lifecycle: dict[Path, tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]] = {}
        self._write_or_validate_manifest()

    def _default_runner(self, case: ReplayCase, stage: str) -> ArenaCodexRunner:
        case_dir = self._case_dir(case)
        return ArenaCodexRunner(prompts_dir=self.project_root / "prompts", log_path=case_dir / f"{stage}-calls.jsonl", session_dir=case_dir / f".{stage}-cli-session", timeout_s=self.timeout_s, validation_retries=0, blob_root=self.run_dir)

    def _prompt_receipts(self) -> dict[str, dict[str, str]]:
        runner = ArenaCodexRunner(prompts_dir=self.project_root / "prompts", log_path=self.run_dir / ".preflight-calls.jsonl", session_dir=self.run_dir / ".preflight-session", timeout_s=self.timeout_s, validation_retries=0, blob_root=self.run_dir)
        receipts: dict[str, dict[str, str]] = {}
        for stage in ("planner", "selector"):
            for family in GLEE_FAMILIES:
                role = f"glee_meta_controller_v2_15_{stage}_{family}"
                _text, version, digest = runner._prompt(role)
                receipts[role] = {"version": version, "sha256": digest}
        return receipts

    def _configuration(self) -> dict[str, object]:
        source_runs = sorted({case.source_run for case in self.cases})
        return {
            "schema_version": 1,
            "contract": REPLAY_CONTRACT,
            "controller": ONE_AND_HALF_ROUND_CONTROLLER_CONTRACT,
            "sample_protocol": SAMPLE_PROTOCOL,
            "model": self.model,
            "effort": self.effort,
            "timeout_s": self.timeout_s,
            "max_workers": self.max_workers,
            "validation_retries": 0,
            "provider_call_budget": {"planner_calls": len(self.cases), "selector_calls": len(self.cases), "maximum_initial_calls": 2 * len(self.cases)},
            "corpus": {"path": str(self.corpus_dir.relative_to(self.project_root)), "manifest_sha256": file_sha256(self.corpus_dir / "manifest.json")},
            "conditional_release": {"path": str(self.release_dir.relative_to(self.project_root)), "manifest_sha256": file_sha256(self.release_dir / "manifest.json")},
            "source_runs": [{"path": str(path.relative_to(self.project_root)), "events_sha256": file_sha256(path / "events.jsonl"), "llm_calls_sha256": file_sha256(path / "llm_calls.jsonl")} for path in source_runs],
            "prompts": self._prompt_receipts(),
            "cases": [case.receipt(project_root=self.project_root) for case in self.cases],
            "interpretation_boundary": "Adaptive 12-turn transport and decision-formation replay over an already inspected old-test suffix. Historical one-shot actions are baselines, but downstream opponent responses are valid only for the recorded actions; counterfactual selected actions are mechanically and qualitatively auditable rather than outcome-scored.",
        }

    def _write_or_validate_manifest(self) -> None:
        configuration = self._configuration()
        if self.manifest_path.is_file():
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            comparable = {key: actual.get(key) for key in configuration}
            if comparable != configuration:
                raise RuntimeError("replay resume configuration differs from its immutable manifest")
            return
        _atomic_json(self.manifest_path, {**configuration, "status": "prepared", "prepared_at": _now()})

    def _event(self, kind: str, **values: object) -> None:
        record = {"schema_version": 1, "ts": _now(), "kind": kind, **values}
        with self._event_lock:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(_canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _case_dir(self, case: ReplayCase) -> Path:
        return self.run_dir / "cases" / f"{case.ordinal:02d}-{case.family}-{case.sample_id[:12]}"

    def _source_payload(self, case: ReplayCase) -> dict[str, object]:
        rows = self._source_call_rows.get(case.source_run)
        if rows is None:
            rows = (case.source_run / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            self._source_call_rows[case.source_run] = rows
        if not 1 <= case.source_llm_call_line <= len(rows):
            raise RuntimeError("source LLM call line is outside its sealed receipt")
        record = json.loads(rows[case.source_llm_call_line - 1])
        if record.get("role") != f"glee_nommd_{case.family}":
            raise RuntimeError("source LLM role does not match the replay case")
        request = record.get("request")
        if not isinstance(request, Mapping) or not isinstance(request.get("user_ref"), Mapping):
            raise RuntimeError("source LLM request has no immutable user reference")
        text = load_referenced_text_blob(receipt_dir=case.source_run, reference=dict(request["user_ref"]))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != case.source_user_sha256 or request.get("user_sha256") != case.source_user_sha256:
            raise RuntimeError("source LLM user payload hash changed")
        payload = json.loads(text)
        receipt = payload.get("turn_receipt") if isinstance(payload, Mapping) else None
        if not isinstance(payload, Mapping) or not isinstance(receipt, Mapping) or receipt.get("turn_id") != case.turn_id or payload.get("game_family") != case.family:
            raise RuntimeError("source LLM payload does not match the replay turn")
        return dict(payload)

    def _source_turn(self, case: ReplayCase) -> tuple[dict[str, Any], dict[str, Any]]:
        lifecycle = self._source_lifecycle.get(case.source_run)
        if lifecycle is None:
            observed: dict[str, dict[str, object]] = {}
            moved: dict[str, dict[str, object]] = {}
            for line in (case.source_run / "events.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                turn_id = str(row.get("turn_id") or "")
                if row.get("kind") == "turn_observed" and isinstance(row.get("game"), Mapping):
                    observed[turn_id] = row
                elif row.get("kind") == "move_submitted" and submitted_action_from_event(row) is not None:
                    moved[turn_id] = row
            lifecycle = observed, moved
            self._source_lifecycle[case.source_run] = lifecycle
        observed, moved = lifecycle
        observation = observed.get(case.turn_id)
        submission = moved.get(case.turn_id)
        if observation is None or submission is None or not isinstance(observation.get("game"), Mapping):
            raise RuntimeError("source turn lacks an observed game or submitted move")
        action = submitted_action_from_event(submission)
        if action is None:
            raise RuntimeError("source submitted move has no action")
        game = dict(observation["game"])
        if game.get("game_id") != case.game_id or game.get("game_family") != case.family:
            raise RuntimeError("source observed game does not match the replay case")
        return game, normalize_action(game, action)

    def _planner_path(self, case: ReplayCase) -> Path:
        return self._case_dir(case) / "planner.json"

    def _conditional_path(self, case: ReplayCase) -> Path:
        return self._case_dir(case) / "conditional.json"

    def _selector_path(self, case: ReplayCase) -> Path:
        return self._case_dir(case) / "selector.json"

    def _candidate_guard(self, game: dict[str, Any]) -> Any:
        return lambda action: apply_deterministic_safeguards(game, action)

    def _load_planner(self, case: ReplayCase) -> tuple[FrozenCandidateSet, dict[str, object]]:
        receipt = json.loads(self._planner_path(case).read_text(encoding="utf-8"))
        game, _recorded = self._source_turn(case)
        parsed = nommd_candidate_plan_model(action_model(game)).model_validate(receipt["planner_output"])
        frozen = freeze_planner_candidates(game=game, parsed=parsed, guard=self._candidate_guard(game))
        if frozen.receipt() != receipt.get("candidate_set"):
            raise RuntimeError("saved planner output no longer reconstructs its frozen candidate set")
        return frozen, receipt

    def _plan_one(self, case: ReplayCase) -> dict[str, object]:
        path = self._planner_path(case)
        if path.is_file():
            _frozen, receipt = self._load_planner(case)
            return receipt
        game, recorded_action = self._source_turn(case)
        worker_payload = self._source_payload(case)
        payload = build_planner_payload_v15(worker_payload=worker_payload)
        runner = self._runner_factory(case, "planner")
        started = time.monotonic()
        parsed, metadata = runner.call_structured(f"glee_meta_controller_v2_15_planner_{case.family}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")), nommd_candidate_plan_model(action_model(game)), model=self.model, effort=self.effort)
        frozen = freeze_planner_candidates(game=game, parsed=parsed, guard=self._candidate_guard(game))
        receipt = {
            "schema_version": 1,
            "contract": REPLAY_CONTRACT,
            "stage": "planner",
            "created_at": _now(),
            "sample_id": case.sample_id,
            "turn_id": case.turn_id,
            "planner_payload_sha256": _sha(payload),
            "authenticated_turn_sha256": payload["authenticated_turn_sha256"],
            "learned_model_boundary": payload["learned_model_boundary"],
            "planner_output": parsed.model_dump(mode="json", exclude_none=True),
            "candidate_set": frozen.receipt(),
            "recorded_action": recorded_action,
            "call": _metadata(metadata),
            "elapsed_s": round(time.monotonic() - started, 6),
        }
        _atomic_json(path, receipt)
        self._event("planner_committed", ordinal=case.ordinal, family=case.family, sample_id=case.sample_id, candidate_set_sha256=frozen.candidate_set_sha256, candidate_count=len(frozen.candidates))
        print(_canonical({"stage": "planner", "ordinal": case.ordinal, "family": case.family, "status": "committed"}), flush=True)
        return receipt

    def _run_parallel_stage(self, stage: str, method: Any) -> None:
        pending = [case for case in self.cases if not (self._planner_path(case) if stage == "planner" else self._selector_path(case)).is_file()]
        if not pending:
            return
        errors: list[tuple[ReplayCase, Exception]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending)), thread_name_prefix=f"glee-v15-{stage}") as executor:
            future_cases = {executor.submit(method, case): case for case in pending}
            for future in as_completed(future_cases):
                case = future_cases[future]
                try:
                    future.result()
                except Exception as error:
                    errors.append((case, error))
                    _atomic_json(self._case_dir(case) / f"{stage}-error.json", {"schema_version": 1, "failed_at": _now(), "stage": stage, "sample_id": case.sample_id, "error": f"{type(error).__name__}: {error}"})
                    self._event("stage_failed", stage=stage, ordinal=case.ordinal, family=case.family, sample_id=case.sample_id, error=f"{type(error).__name__}: {error}")
        if errors:
            case, error = errors[0]
            raise RuntimeError(f"{stage} failed for replay case {case.ordinal} ({case.sample_id}): {type(error).__name__}: {error}") from error

    def _conditional_one(self, release: Any, case: ReplayCase) -> dict[str, object]:
        path = self._conditional_path(case)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        frozen, _planner = self._load_planner(case)
        game, _recorded = self._source_turn(case)
        target = self._targets[case.sample_id]
        feature = self._features[case.sample_id]
        static_game = self._games[case.game_id]
        prefix_length = int(target["pre_candidate_prefix_length"])
        sample = {"game": static_game, "events": self._events[case.game_id][:prefix_length], "target": target}
        candidate_receipts = [CandidateAction.from_live_action(game=game, action=candidate.action).receipt() for candidate in frozen.candidates]
        unique_candidate_receipts, projection_indices = collapse_candidate_projections(candidates=candidate_receipts, family=case.family, phase=case.phase)
        sample_before = object_sha256(sample)
        started = time.monotonic()
        unique_forecasts = release.predict_candidates(sample=sample, family=case.family, phase=case.phase, base_feature_indices=feature["feature_indices"], base_feature_values=feature["feature_values"], base_feature_vector_sha256=str(feature["feature_vector_sha256"]), candidates=unique_candidate_receipts)
        forecasts = [copy.deepcopy(unique_forecasts[index]) for index in projection_indices]
        if object_sha256(sample) != sample_before:
            raise RuntimeError("conditional inference mutated the authenticated historical sample")
        surface = build_conditional_surface(candidate_set=frozen, forecasts=forecasts)
        receipt = {
            "schema_version": 1,
            "contract": REPLAY_CONTRACT,
            "stage": "conditional",
            "created_at": _now(),
            "sample_id": case.sample_id,
            "turn_id": case.turn_id,
            "candidate_set_sha256": frozen.candidate_set_sha256,
            "candidate_action_projections": candidate_receipts,
            "unique_candidate_action_projections": unique_candidate_receipts,
            "projection_index_by_candidate": projection_indices,
            "conditional_surface": surface,
            "historical_direct_response": case.target_label,
            "elapsed_s": round(time.monotonic() - started, 6),
        }
        _atomic_json(path, receipt)
        self._event("conditional_surface_committed", ordinal=case.ordinal, family=case.family, sample_id=case.sample_id, candidate_set_sha256=frozen.candidate_set_sha256)
        print(_canonical({"stage": "conditional", "ordinal": case.ordinal, "family": case.family, "status": "committed"}), flush=True)
        return receipt

    def _run_conditionals(self) -> None:
        release = self._conditional_release or ConditionalTwinRelease(self.release_dir)
        for case in self.cases:
            self._conditional_one(release, case)

    def _load_surface(self, case: ReplayCase, frozen: FrozenCandidateSet) -> dict[str, object]:
        receipt = json.loads(self._conditional_path(case).read_text(encoding="utf-8"))
        surface = receipt.get("conditional_surface")
        if not isinstance(surface, Mapping) or surface.get("contract") != CONDITIONAL_SURFACE_CONTRACT or surface.get("candidate_set_sha256") != frozen.candidate_set_sha256:
            raise RuntimeError("saved conditional surface does not match its candidate set")
        rebuilt = build_conditional_surface(candidate_set=frozen, forecasts=[dict(row["forecast"]) for row in surface["rows"]])
        if rebuilt != surface:
            raise RuntimeError("saved conditional surface no longer validates")
        return dict(surface)

    def _select_one(self, case: ReplayCase) -> dict[str, object]:
        path = self._selector_path(case)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        frozen, planner = self._load_planner(case)
        surface = self._load_surface(case, frozen)
        worker_payload = self._source_payload(case)
        payload = build_selector_payload_v15(worker_payload=worker_payload, candidate_set=frozen, conditional_surface=surface)
        runner = self._runner_factory(case, "selector")
        started = time.monotonic()
        parsed, metadata = runner.call_structured(f"glee_meta_controller_v2_15_selector_{case.family}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")), nommd_candidate_selection_model(), model=self.model, effort=self.effort)
        selected = select_frozen_candidate(candidate_set=frozen, parsed=parsed)
        recorded_action = dict(planner["recorded_action"])
        receipt = {
            "schema_version": 1,
            "contract": REPLAY_CONTRACT,
            "stage": "selector",
            "created_at": _now(),
            "sample_id": case.sample_id,
            "turn_id": case.turn_id,
            "selector_payload_sha256": _sha(payload),
            "candidate_set_sha256": frozen.candidate_set_sha256,
            "selector_output": parsed.model_dump(mode="json", exclude_none=True),
            "selected_candidate": selected.candidate.receipt(),
            "selected_action": dict(selected.candidate.action),
            "recorded_action": recorded_action,
            "recorded_action_match": dict(selected.candidate.action) == recorded_action,
            "recorded_action_in_candidate_set": any(dict(candidate.action) == recorded_action for candidate in frozen.candidates),
            "tetrad_update": selected.update.model_dump(mode="json") if selected.update is not None else None,
            "tetrad_transport_issues": list(selected.transport_issues),
            "call": _metadata(metadata),
            "elapsed_s": round(time.monotonic() - started, 6),
        }
        _atomic_json(path, receipt)
        self._event("selector_committed", ordinal=case.ordinal, family=case.family, sample_id=case.sample_id, candidate_index=selected.candidate.index, recorded_action_match=receipt["recorded_action_match"])
        print(_canonical({"stage": "selector", "ordinal": case.ordinal, "family": case.family, "status": "committed"}), flush=True)
        return receipt

    @staticmethod
    def _surface_spread(surface: Mapping[str, object], family: str) -> float:
        desired = "buy" if family == "persuasion" else "accept"
        values: list[float] = []
        for row in surface["rows"]:
            forecast = row["forecast"]
            labels = list(forecast["labels"])
            values.append(float(forecast["response_probabilities"][labels.index(desired)]))
        return max(values) - min(values) if values else 0.0

    def _summarize(self, elapsed_s: float) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for case in self.cases:
            planner = json.loads(self._planner_path(case).read_text(encoding="utf-8"))
            conditional = json.loads(self._conditional_path(case).read_text(encoding="utf-8"))
            selector = json.loads(self._selector_path(case).read_text(encoding="utf-8"))
            planner_call = planner["call"]
            selector_call = selector["call"]
            candidates = planner["candidate_set"]["candidates"]
            recorded_indices = [index for index, candidate in enumerate(candidates) if candidate["action"] == planner["recorded_action"]]
            recorded_forecast = conditional["conditional_surface"]["rows"][recorded_indices[0]]["forecast"] if recorded_indices else None
            rows.append(
                {
                    "ordinal": case.ordinal,
                    "family": case.family,
                    "sample_id": case.sample_id,
                    "identity_scope": case.identity_scope,
                    "stratum": case.stratum,
                    "candidate_count": len(candidates),
                    "unique_predictor_projection_count": len(conditional.get("unique_candidate_action_projections", conditional["candidate_action_projections"])),
                    "conditional_accept_or_buy_spread": self._surface_spread(conditional["conditional_surface"], case.family),
                    "recorded_action_in_candidate_set": bool(selector["recorded_action_in_candidate_set"]),
                    "recorded_action_match": bool(selector["recorded_action_match"]),
                    "recorded_action_forecast_correct": recorded_forecast.get("predicted_response") == case.target_label if isinstance(recorded_forecast, Mapping) else None,
                    "planner_safeguards": sum(len(candidate.get("planner_candidate_safeguards") or []) for candidate in planner["candidate_set"]["candidates"]),
                    "planner_elapsed_s": float(planner["elapsed_s"]),
                    "selector_elapsed_s": float(selector["elapsed_s"]),
                    "tokens_in": int(planner_call.get("tokens_in") or 0) + int(selector_call.get("tokens_in") or 0),
                    "tokens_out": int(planner_call.get("tokens_out") or 0) + int(selector_call.get("tokens_out") or 0),
                    "reasoning_tokens": int(planner_call.get("reasoning_tokens") or 0) + int(selector_call.get("reasoning_tokens") or 0),
                }
            )
        families: dict[str, object] = {}
        for family in GLEE_FAMILIES:
            selected = [row for row in rows if row["family"] == family]
            families[family] = {
                "turns": len(selected),
                "known_identity_turns": sum(row["identity_scope"] == "known" for row in selected),
                "recorded_action_covered": sum(bool(row["recorded_action_in_candidate_set"]) for row in selected),
                "recorded_action_selected": sum(bool(row["recorded_action_match"]) for row in selected),
                "candidate_count_mean": statistics.mean(float(row["candidate_count"]) for row in selected),
                "unique_predictor_projections": sum(int(row["unique_predictor_projection_count"]) for row in selected),
                "planner_candidates": sum(int(row["candidate_count"]) for row in selected),
                "covered_recorded_forecasts_correct": sum(row["recorded_action_forecast_correct"] is True for row in selected),
                "conditional_accept_or_buy_spread_mean": statistics.mean(float(row["conditional_accept_or_buy_spread"]) for row in selected),
                "planner_safeguards": sum(int(row["planner_safeguards"]) for row in selected),
                "planner_elapsed_s_median": statistics.median(float(row["planner_elapsed_s"]) for row in selected),
                "selector_elapsed_s_median": statistics.median(float(row["selector_elapsed_s"]) for row in selected),
                "tokens_in": sum(int(row["tokens_in"]) for row in selected),
                "tokens_out": sum(int(row["tokens_out"]) for row in selected),
                "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in selected),
            }
        return {
            "schema_version": 1,
            "contract": REPLAY_CONTRACT,
            "status": "complete",
            "completed_at": _now(),
            "elapsed_s": round(elapsed_s, 6),
            "turns": len(rows),
            "provider_calls": 2 * len(rows),
            "model": self.model,
            "effort": self.effort,
            "families": families,
            "rows": rows,
            "interpretation_boundary": json.loads(self.manifest_path.read_text(encoding="utf-8"))["interpretation_boundary"],
        }

    def _write_report(self, summary: Mapping[str, object]) -> None:
        rows = []
        for family, values in summary["families"].items():
            rows.append(f"| {family} | {values['turns']} | {values['recorded_action_covered']} | {values['recorded_action_selected']} | {values['candidate_count_mean']:.2f} | {values['conditional_accept_or_buy_spread_mean']:.3f} | {values['planner_safeguards']} | {values['planner_elapsed_s_median']:.1f} | {values['selector_elapsed_s_median']:.1f} |")
        report = "\n".join(
            [
                "# GLEE 1.5-round Terra chronological replay",
                "",
                f"The replay completed {summary['turns']} frozen historical turns with {summary['model']} at {summary['effort']} effort, using 2 new Terra calls per turn and no GLEE API calls.",
                "",
                "| Family | Turns | Recorded action covered | Recorded action selected | Mean candidates | Mean accept/buy spread | Planner safeguards | Median planner s | Median selector s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *rows,
                "",
                str(summary["interpretation_boundary"]),
                "",
            ]
        )
        (self.run_dir / "report.md").write_text(report, encoding="utf-8")

    def run(self, *, prepare_only: bool = False) -> dict[str, object]:
        if prepare_only:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.complete_path.is_file():
            return json.loads(self.complete_path.read_text(encoding="utf-8"))
        started = time.monotonic()
        self._event("replay_entered", turns=len(self.cases), maximum_initial_provider_calls=2 * len(self.cases), model=self.model, effort=self.effort)
        self._run_parallel_stage("planner", self._plan_one)
        self._run_conditionals()
        self._run_parallel_stage("selector", self._select_one)
        summary = self._summarize(time.monotonic() - started)
        _atomic_json(self.complete_path, summary)
        self._write_report(summary)
        self._event("replay_completed", summary_sha256=_sha(summary), turns=len(self.cases))
        return summary
