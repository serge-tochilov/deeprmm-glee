"""Backend-neutral selection over one immutable, fully evaluated candidate set."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .glee_meta_controller_v2 import PERSUASION_BUYER_CONTINUATION_SELECTOR_PAYLOAD_CONTRACT, FrozenCandidateSet, SelectedCandidate, select_frozen_candidate
from .glee_nommd import nommd_candidate_selection_model


SELECTOR_BACKEND_REQUEST_CONTRACT = "glee-selector-backend-request-v2"
SELECTOR_BACKEND_RESULT_CONTRACT = "glee-selector-backend-result-v2"
SELECTOR_WORDING_CONTRACT = "glee-selector-committed-wording-v1"
SELECTOR_POSITION_PRESENTATION_CONTRACT = "glee-selector-position-presentation-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _turn_id(payload: Mapping[str, object]) -> str:
    authenticated = payload.get("authenticated_turn")
    if not isinstance(authenticated, Mapping):
        raise ValueError("selector payload has no authenticated turn")
    receipt = authenticated.get("turn_receipt")
    value = receipt.get("turn_id") if isinstance(receipt, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("selector payload has no stable turn identifier")
    return value


def _wording_fields(candidate_set: FrozenCandidateSet) -> tuple[str, ...]:
    fields = {key for candidate in candidate_set.candidates for key, value in candidate.action.items() if key == "message" and isinstance(value, str)}
    return tuple(sorted(fields))


@dataclass(frozen=True)
class SelectorBackendRequest:
    """One exact selector task shared by cloud and local backends."""

    family: str
    action_type: str
    turn_id: str
    candidate_set: FrozenCandidateSet
    selector_payload: Mapping[str, object]
    fallback_candidate_id: str
    wording_fields: tuple[str, ...]
    timeout_s: float
    request_sha256: str
    presentation_to_canonical: tuple[int, ...]
    canonical_to_presentation: tuple[int, ...]
    presentation_seed_sha256: str

    @property
    def presentation_candidate_ids(self) -> tuple[str, ...]:
        """Return immutable candidate identifiers in the order shown to the selector."""
        return tuple(self.candidate_set.candidates[index].action_sha256 for index in self.presentation_to_canonical)

    def resolve_presentation_index(self, index: object) -> SelectedCandidate:
        """Resolve one model-visible position back to the canonical frozen candidate."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.presentation_to_canonical):
            raise ValueError("selector index does not identify a presented candidate")
        canonical_index = self.presentation_to_canonical[index]
        return SelectedCandidate(candidate=self.candidate_set.candidates[canonical_index])

    def presentation_index(self, candidate_id: str) -> int:
        """Return the model-visible position of one canonical candidate identifier."""
        matches = [candidate.index for candidate in self.candidate_set.candidates if candidate.action_sha256 == candidate_id]
        if len(matches) != 1:
            raise ValueError("candidate identifier does not resolve uniquely inside the frozen set")
        return self.canonical_to_presentation[matches[0]]

    def position_presentation_receipt(self) -> dict[str, object]:
        """Expose the sealed inverse mapping for exact replay without showing it to the selector."""
        rows = []
        for presentation_index, canonical_index in enumerate(self.presentation_to_canonical):
            candidate = self.candidate_set.candidates[canonical_index]
            rows.append({"presentation_index": presentation_index, "canonical_index": canonical_index, "candidate_id": candidate.action_sha256})
        return {
            "contract": SELECTOR_POSITION_PRESENTATION_CONTRACT,
            "mode": "seed-stable-per-turn-permutation",
            "seed_sha256": self.presentation_seed_sha256,
            "canonical_candidate_set_sha256": self.candidate_set.candidate_set_sha256,
            "presentation_to_canonical": rows,
        }

    def wire_payload(self) -> dict[str, object]:
        """Return the backend-neutral wire object without mutable internal references."""
        body = {
            "contract": SELECTOR_BACKEND_REQUEST_CONTRACT,
            "family": self.family,
            "action_type": self.action_type,
            "turn_id": self.turn_id,
            "selector_payload": copy.deepcopy(dict(self.selector_payload)),
            "candidate_ids": list(self.presentation_candidate_ids),
            "fallback_candidate_id": self.fallback_candidate_id,
            "wording_contract": {
                "contract": SELECTOR_WORDING_CONTRACT,
                "fields": list(self.wording_fields),
                "mode": "exact-precommitted-candidate-wording",
                "rule": "A backend may omit wording or repeat the selected candidate's exact wording fields; it cannot invent or edit wording after the conditional forecasts were committed.",
            },
            "deadline_budget_s": self.timeout_s,
        }
        return {**body, "request_sha256": self.request_sha256}


def _position_permutation(*, candidate_set: FrozenCandidateSet, turn_id: str) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Derive one runtime-independent pseudorandom order from already committed turn material."""
    canonical_indexes = tuple(candidate.index for candidate in candidate_set.candidates)
    if canonical_indexes != tuple(range(len(candidate_set.candidates))):
        raise ValueError("frozen candidates must retain contiguous canonical indexes")
    seed_sha256 = _sha({"contract": SELECTOR_POSITION_PRESENTATION_CONTRACT, "turn_id": turn_id, "candidate_set_sha256": candidate_set.candidate_set_sha256})
    presentation_to_canonical = tuple(sorted(canonical_indexes, key=lambda index: hashlib.sha256(f"{seed_sha256}|{candidate_set.candidates[index].action_sha256}".encode("utf-8")).hexdigest()))
    canonical_to_presentation_values = [-1] * len(presentation_to_canonical)
    for presentation_index, canonical_index in enumerate(presentation_to_canonical):
        canonical_to_presentation_values[canonical_index] = presentation_index
    canonical_to_presentation = tuple(canonical_to_presentation_values)
    if sorted(presentation_to_canonical) != list(canonical_indexes) or sorted(canonical_to_presentation) != list(canonical_indexes):
        raise ValueError("selector position permutation is not bijective")
    return presentation_to_canonical, canonical_to_presentation, seed_sha256


def _present_rows(*, surface: object, candidate_set: FrozenCandidateSet, presentation_to_canonical: tuple[int, ...], name: str) -> object:
    if surface is None:
        return None
    if not isinstance(surface, Mapping):
        raise ValueError(f"selector payload {name} is not an object")
    presented = copy.deepcopy(dict(surface))
    rows = presented.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
        raise ValueError(f"selector payload {name} does not cover the frozen candidates")
    by_canonical_index: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or isinstance(row.get("candidate_index"), bool) or not isinstance(row.get("candidate_index"), int):
            raise ValueError(f"selector payload {name} has an invalid candidate index")
        canonical_index = int(row["candidate_index"])
        if canonical_index in by_canonical_index or not 0 <= canonical_index < len(candidate_set.candidates):
            raise ValueError(f"selector payload {name} has duplicate or out-of-range candidate indexes")
        if row.get("action_sha256") != candidate_set.candidates[canonical_index].action_sha256:
            raise ValueError(f"selector payload {name} changed candidate alignment")
        by_canonical_index[canonical_index] = copy.deepcopy(dict(row))
    if set(by_canonical_index) != set(range(len(candidate_set.candidates))):
        raise ValueError(f"selector payload {name} has incomplete candidate coverage")
    presented_rows: list[dict[str, object]] = []
    for presentation_index, canonical_index in enumerate(presentation_to_canonical):
        row = by_canonical_index[canonical_index]
        row["candidate_index"] = presentation_index
        presented_rows.append(row)
    presented["rows"] = presented_rows
    return presented


def _present_selector_payload(*, payload: Mapping[str, object], candidate_set: FrozenCandidateSet, presentation_to_canonical: tuple[int, ...]) -> dict[str, object]:
    """Relabel and reorder every aligned selector-visible row while preserving canonical candidate IDs."""
    presented = copy.deepcopy(dict(payload))
    receipt = presented.get("candidate_set")
    if not isinstance(receipt, Mapping) or receipt != candidate_set.receipt():
        raise ValueError("selector backend candidate set differs from the selector payload")
    raw_candidates = receipt.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(candidate_set.candidates):
        raise ValueError("selector backend candidate receipt does not cover the frozen set")
    candidate_rows: list[dict[str, object]] = []
    for presentation_index, canonical_index in enumerate(presentation_to_canonical):
        raw = raw_candidates[canonical_index]
        if not isinstance(raw, Mapping) or raw.get("candidate_index") != canonical_index or raw.get("action_sha256") != candidate_set.candidates[canonical_index].action_sha256:
            raise ValueError("selector backend canonical candidate receipt is misaligned")
        row = copy.deepcopy(dict(raw))
        row["candidate_index"] = presentation_index
        candidate_rows.append(row)
    presented_receipt = copy.deepcopy(dict(receipt))
    presented_receipt["candidates"] = candidate_rows
    presented["candidate_set"] = presented_receipt
    presented["conditional_opponent_response_surface"] = _present_rows(surface=presented.get("conditional_opponent_response_surface"), candidate_set=candidate_set, presentation_to_canonical=presentation_to_canonical, name="conditional opponent response surface")
    if "family_candidate_decision_evidence" in presented:
        presented["family_candidate_decision_evidence"] = _present_rows(surface=presented.get("family_candidate_decision_evidence"), candidate_set=candidate_set, presentation_to_canonical=presentation_to_canonical, name="family candidate decision evidence")
    return presented


def build_selector_backend_request(
    *,
    candidate_set: FrozenCandidateSet,
    selector_payload: Mapping[str, object],
    fallback_candidate_id: str,
    timeout_s: float,
) -> SelectorBackendRequest:
    """Validate and freeze the common selector input before choosing a backend."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("selector backend timeout must be finite and positive")
    canonical_payload = copy.deepcopy(dict(selector_payload))
    receipt = canonical_payload.get("candidate_set")
    if not isinstance(receipt, Mapping) or receipt != candidate_set.receipt():
        raise ValueError("selector backend candidate set differs from the selector payload")
    if fallback_candidate_id not in {candidate.action_sha256 for candidate in candidate_set.candidates}:
        raise ValueError("selector backend fallback does not identify a frozen candidate")
    family = str(canonical_payload.get("authenticated_turn", {}).get("game_family") or "") if isinstance(canonical_payload.get("authenticated_turn"), Mapping) else ""
    if family != candidate_set.family:
        raise ValueError("selector backend family differs from the frozen candidate set")
    turn_id = _turn_id(canonical_payload)
    presentation_to_canonical, canonical_to_presentation, presentation_seed_sha256 = _position_permutation(candidate_set=candidate_set, turn_id=turn_id)
    payload = _present_selector_payload(payload=canonical_payload, candidate_set=candidate_set, presentation_to_canonical=presentation_to_canonical)
    presentation_candidate_ids = [candidate_set.candidates[index].action_sha256 for index in presentation_to_canonical]
    body = {
        "contract": SELECTOR_BACKEND_REQUEST_CONTRACT,
        "family": candidate_set.family,
        "action_type": candidate_set.action_type,
        "turn_id": turn_id,
        "selector_payload": payload,
        "candidate_ids": presentation_candidate_ids,
        "fallback_candidate_id": fallback_candidate_id,
        "wording_contract": {
            "contract": SELECTOR_WORDING_CONTRACT,
            "fields": list(_wording_fields(candidate_set)),
            "mode": "exact-precommitted-candidate-wording",
            "rule": "A backend may omit wording or repeat the selected candidate's exact wording fields; it cannot invent or edit wording after the conditional forecasts were committed.",
        },
        "deadline_budget_s": float(timeout_s),
    }
    return SelectorBackendRequest(
        family=candidate_set.family,
        action_type=candidate_set.action_type,
        turn_id=str(body["turn_id"]),
        candidate_set=candidate_set,
        selector_payload=payload,
        fallback_candidate_id=fallback_candidate_id,
        wording_fields=_wording_fields(candidate_set),
        timeout_s=float(timeout_s),
        request_sha256=_sha(body),
        presentation_to_canonical=presentation_to_canonical,
        canonical_to_presentation=canonical_to_presentation,
        presentation_seed_sha256=presentation_seed_sha256,
    )


@dataclass(frozen=True)
class SelectorBackendOutcome:
    """One validated backend choice resolved to an immutable candidate."""

    selected: SelectedCandidate
    backend_id: str
    role: str
    elapsed_s: float
    metadata: object
    receipt: Mapping[str, object]


class SelectorBackend(Protocol):
    """Select one precommitted candidate without owning mechanics or safeguards."""

    @property
    def manifest_receipt(self) -> Mapping[str, object]: ...

    def select(self, request: SelectorBackendRequest) -> SelectorBackendOutcome: ...


def _validate_local_result(request: SelectorBackendRequest, value: object) -> tuple[SelectedCandidate, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("local selector result must be an object")
    result = copy.deepcopy(dict(value))
    if "update" in result:
        raise ValueError("local selector result contains the retired tetrad update field")
    if result.get("contract") != SELECTOR_BACKEND_RESULT_CONTRACT:
        raise ValueError("local selector result has the wrong contract")
    if result.get("request_sha256") != request.request_sha256:
        raise ValueError("local selector result does not match its request")
    candidate_id = result.get("candidate_id")
    matches = [candidate for candidate in request.candidate_set.candidates if candidate.action_sha256 == candidate_id]
    if len(matches) != 1:
        raise ValueError("local selector result does not identify one supplied candidate")
    candidate = matches[0]
    wording = result.get("wording")
    if wording is not None:
        if not isinstance(wording, Mapping) or set(wording) - set(request.wording_fields):
            raise ValueError("local selector result contains an unbounded wording field")
        expected = {field: candidate.action.get(field) for field in wording}
        if dict(wording) != expected:
            raise ValueError("local selector result edited wording after candidate commitment")
    parsed = nommd_candidate_selection_model().model_validate({"candidate_index": candidate.index})
    return select_frozen_candidate(candidate_set=request.candidate_set, parsed=parsed), result


class TerraSelectorBackend:
    """Adapt one structured cloud selector call to the backend-neutral contract."""

    def __init__(self, *, runner_factory: Callable[[int], Any], model: str, effort: str, backend_id: str = "terra-cli-v2-position-permuted", clock: Callable[[], float] = time.monotonic) -> None:
        if not backend_id.strip():
            raise ValueError("cloud selector backend identifier must be nonempty")
        self.runner_factory = runner_factory
        self.model = model
        self.effort = effort
        self.backend_id = backend_id
        self.clock = clock

    @property
    def manifest_receipt(self) -> Mapping[str, object]:
        return {"contract": SELECTOR_BACKEND_REQUEST_CONTRACT, "backend_id": self.backend_id, "model": self.model, "effort": self.effort, "output": "seed-stable-presentation-index-inverted-to-canonical-candidate-id"}

    def select(self, request: SelectorBackendRequest) -> SelectorBackendOutcome:
        started = self.clock()
        role = "glee_persuasion_buyer_continuation_selector" if request.selector_payload.get("contract") == PERSUASION_BUYER_CONTINUATION_SELECTOR_PAYLOAD_CONTRACT else f"glee_meta_controller_v2_15_selector_{request.family}"
        metadata: object = None
        runner = self.runner_factory(max(1, math.floor(request.timeout_s)))
        parsed, metadata = runner.call_structured(
            role,
            json.dumps(request.selector_payload, ensure_ascii=False, separators=(",", ":")),
            nommd_candidate_selection_model(),
            model=self.model,
            effort=self.effort,
        )
        presentation_index = getattr(parsed, "candidate_index", None)
        selected = request.resolve_presentation_index(presentation_index)
        elapsed = self.clock() - started
        receipt = {
            "contract": SELECTOR_BACKEND_RESULT_CONTRACT,
            "request_sha256": request.request_sha256,
            "backend_id": self.backend_id,
            "candidate_id": selected.candidate.action_sha256,
            "candidate_index": selected.candidate.index,
            "presentation_candidate_index": presentation_index,
            "position_presentation": request.position_presentation_receipt(),
            "wording": {field: selected.candidate.action.get(field) for field in request.wording_fields},
        }
        return SelectorBackendOutcome(selected=selected, backend_id=self.backend_id, role=role, elapsed_s=round(elapsed, 6), metadata=metadata, receipt=receipt)


class LocalCallableSelectorBackend:
    """Call one local selector exactly once and fail closed on any malformed result."""

    def __init__(self, *, selector: Callable[[Mapping[str, object]], object], backend_id: str = "local-callable-v1", clock: Callable[[], float] = time.monotonic) -> None:
        if not backend_id.strip():
            raise ValueError("local selector backend identifier must be nonempty")
        self.selector = selector
        self.backend_id = backend_id
        self.clock = clock

    @property
    def manifest_receipt(self) -> Mapping[str, object]:
        return {"contract": SELECTOR_BACKEND_REQUEST_CONTRACT, "backend_id": self.backend_id, "attempts_per_turn": 1, "failure": "caller-owned deterministic fallback"}

    def select(self, request: SelectorBackendRequest) -> SelectorBackendOutcome:
        started = self.clock()
        raw = self.selector(request.wire_payload())
        selected, result = _validate_local_result(request, raw)
        elapsed = self.clock() - started
        if elapsed > request.timeout_s:
            raise TimeoutError(f"local selector exceeded its {request.timeout_s:.3f}s deadline budget")
        receipt = {**result, "backend_id": self.backend_id, "candidate_index": selected.candidate.index, "presentation_candidate_index": request.presentation_index(selected.candidate.action_sha256), "position_presentation": request.position_presentation_receipt()}
        return SelectorBackendOutcome(selected=selected, backend_id=self.backend_id, role=f"glee_local_selector_{request.family}", elapsed_s=round(elapsed, 6), metadata=None, receipt=receipt)


def _aligned_row(payload: Mapping[str, object], surface_name: str, candidate_index: int, value_name: str) -> Mapping[str, object] | None:
    surface = payload.get(surface_name)
    rows = surface.get("rows") if isinstance(surface, Mapping) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("candidate_index") == candidate_index and isinstance(row.get(value_name), Mapping):
            return row[value_name]
    return None


def _probability(forecast: Mapping[str, object] | None, label: str) -> float | None:
    labels = forecast.get("labels") if isinstance(forecast, Mapping) else None
    values = forecast.get("response_probabilities") if isinstance(forecast, Mapping) else None
    if not isinstance(labels, list) or not isinstance(values, list) or label not in labels or len(labels) != len(values):
        return None
    value = values[labels.index(label)]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None


def reference_expected_value(*, family: str, payload: Mapping[str, object], candidate_index: int) -> tuple[float | None, str]:
    """Return the established bounded response-weighted value for one aligned candidate."""
    forecast = _aligned_row(payload, "conditional_opponent_response_surface", candidate_index, "forecast")
    evidence = _aligned_row(payload, "family_candidate_decision_evidence", candidate_index, "evidence")
    if family == "persuasion":
        return _probability(forecast, "buy"), "conditional-buy-probability"
    accept = _probability(forecast, "accept")
    reject = _probability(forecast, "reject")
    if accept is None or reject is None:
        return None, "missing-conditional-response-probability"
    if family == "bargaining":
        evaluation = evidence.get("behavioral_offer_evaluation") if isinstance(evidence, Mapping) and isinstance(evidence.get("behavioral_offer_evaluation"), Mapping) else None
        accepted = _finite(evaluation.get("accepted_value")) if isinstance(evaluation, Mapping) else None
        rejected = _finite(evaluation.get("rejected_path_value")) if isinstance(evaluation, Mapping) else None
        return (accept * accepted + reject * rejected, "conditional-accept-plus-rejection-value") if accepted is not None and rejected is not None else (None, "missing-bargaining-value-evidence")
    if family == "negotiation":
        evaluation = evidence.get("evaluation") if isinstance(evidence, Mapping) and isinstance(evidence.get("evaluation"), Mapping) else None
        accepted = _finite(evaluation.get("own_surplus_if_accepted")) if isinstance(evaluation, Mapping) else None
        continuation = evaluation.get("conditional_rejection_continuation") if isinstance(evaluation, Mapping) and isinstance(evaluation.get("conditional_rejection_continuation"), Mapping) else None
        rejected = _finite(continuation.get("bounded_continuation_value")) if isinstance(continuation, Mapping) else 0.0
        return (accept * accepted + reject * rejected, "conditional-accept-plus-bounded-continuation") if accepted is not None and rejected is not None else (None, "missing-negotiation-value-evidence")
    return None, "unsupported-family"


def reference_candidate_scores(request: SelectorBackendRequest) -> list[dict[str, object]]:
    """Return aligned reference values without selecting or mutating a candidate."""
    rows: list[dict[str, object]] = []
    for candidate in request.candidate_set.candidates:
        presentation_index = request.presentation_index(candidate.action_sha256)
        score, basis = reference_expected_value(family=request.family, payload=request.selector_payload, candidate_index=presentation_index)
        rows.append({"candidate_id": candidate.action_sha256, "candidate_index": candidate.index, "presentation_candidate_index": presentation_index, "score": score, "basis": basis})
    return rows


class LocalExpectedValueSelectorBackend:
    """Deterministic reference backend for replay and local failure-control tests."""

    backend_id = "local-reference-expected-value-v1"

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock

    @property
    def manifest_receipt(self) -> Mapping[str, object]:
        return {"contract": SELECTOR_BACKEND_REQUEST_CONTRACT, "backend_id": self.backend_id, "policy": "maximum-direct-response-weighted-bounded-value", "tie_break": "fallback-then-candidate-id", "learned_selector": False}

    def select(self, request: SelectorBackendRequest) -> SelectorBackendOutcome:
        started = self.clock()
        rows = reference_candidate_scores(request)
        scored = [row for row in rows if isinstance(row["score"], float)]
        if not scored:
            candidate_id = request.fallback_candidate_id
            status = "deterministic-fallback-no-comparable-score"
        else:
            ranked = sorted(scored, key=lambda row: (-float(row["score"]), row["candidate_id"] != request.fallback_candidate_id, str(row["candidate_id"])))
            candidate_id = str(ranked[0]["candidate_id"])
            status = "selected-maximum-reference-value"
        wire = request.wire_payload()
        raw = local_result(request=wire, candidate_id=candidate_id)
        selected, result = _validate_local_result(request, raw)
        elapsed = self.clock() - started
        if elapsed > request.timeout_s:
            raise TimeoutError(f"local reference selector exceeded its {request.timeout_s:.3f}s deadline budget")
        receipt = {**result, "backend_id": self.backend_id, "candidate_index": selected.candidate.index, "presentation_candidate_index": request.presentation_index(selected.candidate.action_sha256), "position_presentation": request.position_presentation_receipt(), "status": status, "scores": rows}
        return SelectorBackendOutcome(selected=selected, backend_id=self.backend_id, role=f"glee_local_reference_selector_{request.family}", elapsed_s=round(elapsed, 6), metadata=None, receipt=receipt)


def local_result(*, request: Mapping[str, object], candidate_id: str, wording: Mapping[str, str] | None = None) -> dict[str, object]:
    """Build the strict result envelope expected from a local selector implementation."""
    return {
        "contract": SELECTOR_BACKEND_RESULT_CONTRACT,
        "request_sha256": request.get("request_sha256"),
        "candidate_id": candidate_id,
        "wording": copy.deepcopy(dict(wording)) if wording is not None else None,
    }
