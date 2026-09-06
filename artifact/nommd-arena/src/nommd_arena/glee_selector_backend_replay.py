"""Replay the backend-neutral reference selector over immutable event-log prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .glee_meta_controller_v2 import FrozenCandidateSet, PlannedCandidate
from .glee_selector_backend import LocalExpectedValueSelectorBackend, SELECTOR_BACKEND_REQUEST_CONTRACT, SelectorBackendRequest, build_selector_backend_request


REPLAY_CONTRACT = "glee-selector-backend-chronological-replay-v1"


@dataclass(frozen=True)
class SelectorReplayExample:
    """One historical selector request paired with the successful cloud choice."""

    request: SelectorBackendRequest
    cloud_candidate_index: int
    ts: str


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def read_event_prefix(path: Path, *, byte_limit: int | None = None, expected_sha256: str | None = None) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Read exactly one immutable byte prefix and verify it before parsing complete events."""
    available = path.stat().st_size
    byte_limit = available if byte_limit is None else byte_limit
    if byte_limit < 0 or byte_limit > available:
        raise ValueError(f"event prefix byte limit {byte_limit} is outside the available range 0..{available}: {path}")
    with path.open("rb") as stream:
        raw = stream.read(byte_limit)
    prefix_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and prefix_sha256 != expected_sha256:
        raise ValueError(f"event prefix digest mismatch: {path}")
    complete = raw if raw.endswith(b"\n") else raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return rows, {"path": str(path), "byte_limit": byte_limit, "consumed_bytes": len(complete), "prefix_sha256": prefix_sha256, "event_count": len(rows)}


def _read_prefix(path: Path) -> tuple[list[dict[str, Any]], dict[str, object]]:
    return read_event_prefix(path)


def _candidate_set(value: Mapping[str, object]) -> FrozenCandidateSet:
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("planner receipt has no candidates")
    candidates: list[PlannedCandidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("action"), Mapping):
            raise ValueError("planner candidate receipt is malformed")
        action = dict(raw["action"])
        if _sha(action) != raw.get("action_sha256"):
            raise ValueError("planner candidate action hash changed")
        candidates.append(
            PlannedCandidate(
                index=int(raw["candidate_index"]),
                action=MappingProxyType(action),
                purpose=str(raw["purpose"]),
                action_sha256=str(raw["action_sha256"]),
                safeguards=tuple(str(item) for item in raw.get("planner_candidate_safeguards") or ()),
            )
        )
    frozen = FrozenCandidateSet(family=str(value["family"]), action_type=str(value["action_type"]), candidates=tuple(candidates), candidate_set_sha256=str(value["candidate_set_sha256"]))
    expected = _sha({"family": frozen.family, "action_type": frozen.action_type, "candidates": [candidate.receipt() for candidate in frozen.candidates]})
    if frozen.candidate_set_sha256 != expected:
        raise ValueError("planner candidate-set hash changed")
    return frozen


def _branches(event: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    decision = event.get("decision")
    receipts = decision.get("branch_receipts") if isinstance(decision, Mapping) else None
    return {str(receipt.get("branch")): receipt for receipt in receipts if isinstance(receipt, Mapping)} if isinstance(receipts, list) else {}


def extract_selector_example(event: Mapping[str, object], *, authenticated_turn: Mapping[str, object] | None = None) -> SelectorReplayExample | None:
    """Recover one exact backend request and its successful cloud-selected candidate index."""
    branches = _branches(event)
    planner = branches.get("meta15-planner")
    conditional = branches.get("conditional-twin")
    selector = branches.get("meta15-selector")
    if not all(isinstance(receipt, Mapping) and receipt.get("status") == "succeeded" for receipt in (planner, conditional, selector)):
        return None
    assert planner is not None and conditional is not None and selector is not None
    raw_set = planner.get("candidate_set")
    surface = conditional.get("surface")
    selected_receipt = selector.get("selected_candidate")
    if not isinstance(raw_set, Mapping) or not isinstance(surface, Mapping) or not isinstance(selected_receipt, Mapping):
        raise ValueError("successful staged selector receipt is incomplete")
    candidates = _candidate_set(raw_set)
    turn = dict(authenticated_turn) if authenticated_turn is not None else {"turn_receipt": {"turn_id": str(event.get("turn_id") or "")}, "game_family": candidates.family, "valid_actions": {"type": candidates.action_type}}
    payload: dict[str, object] = {
        "authenticated_turn": turn,
        "candidate_set": candidates.receipt(),
        "conditional_opponent_response_surface": dict(surface),
    }
    family_evidence = branches.get("family-candidate-evidence")
    if isinstance(family_evidence, Mapping) and isinstance(family_evidence.get("evidence"), Mapping):
        payload["family_candidate_decision_evidence"] = dict(family_evidence["evidence"])
    request = build_selector_backend_request(candidate_set=candidates, selector_payload=payload, fallback_candidate_id=candidates.candidates[0].action_sha256, timeout_s=12.0)
    selected_index = int(selected_receipt["candidate_index"])
    if selected_index not in {candidate.index for candidate in candidates.candidates}:
        raise ValueError("successful cloud selector chose a candidate outside the frozen set")
    return SelectorReplayExample(request=request, cloud_candidate_index=selected_index, ts=str(event.get("ts") or ""))


def selector_examples(events: Sequence[Mapping[str, object]]) -> list[SelectorReplayExample]:
    """Join successful worker receipts to the authenticated turn observed earlier in the same log."""
    observed: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("kind") != "turn_observed" or not isinstance(event.get("game"), Mapping):
            continue
        turn_id = str(event.get("turn_id") or "")
        game = dict(event["game"])
        observed[turn_id] = {
            "turn_receipt": {"turn_id": turn_id},
            "game_family": game.get("game_family"),
            "your_player": game.get("your_player"),
            "phase": game.get("phase"),
            "opponent": game.get("opponent"),
            "official_prompt": game.get("prompt"),
            "visible_game_state": game.get("game_state"),
            "valid_actions": game.get("valid_actions"),
        }
    examples: list[SelectorReplayExample] = []
    for event in events:
        turn_id = str(event.get("turn_id") or "")
        example = extract_selector_example(event, authenticated_turn=observed.get(turn_id))
        if example is not None:
            examples.append(example)
    return examples


def _record(example: SelectorReplayExample) -> dict[str, object]:
    request = example.request
    selected_index = example.cloud_candidate_index
    backend = LocalExpectedValueSelectorBackend()
    started = time.perf_counter()
    first = backend.select(request)
    elapsed_ms = 1000 * (time.perf_counter() - started)
    second = backend.select(request)
    if second.selected.candidate.action_sha256 != first.selected.candidate.action_sha256 or second.receipt.get("scores") != first.receipt.get("scores"):
        raise RuntimeError("reference selector replay is not deterministic")
    local_index = first.selected.candidate.index
    scores = {int(row["candidate_index"]): row.get("score") for row in first.receipt["scores"] if isinstance(row, Mapping)}
    return {
        "ts": example.ts,
        "turn_id": request.turn_id,
        "family": request.candidate_set.family,
        "candidate_count": len(request.candidate_set.candidates),
        "cloud_candidate_index": selected_index,
        "local_candidate_index": local_index,
        "fallback_candidate_index": 0,
        "agreement": selected_index == local_index,
        "local_is_fallback": local_index == 0,
        "cloud_score": scores.get(selected_index),
        "local_score": scores.get(local_index),
        "fallback_score": scores.get(0),
        "elapsed_ms": elapsed_ms,
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 9) if values else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 9)


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparable = [row for row in rows if isinstance(row.get("cloud_score"), (int, float)) and isinstance(row.get("local_score"), (int, float)) and isinstance(row.get("fallback_score"), (int, float))]
    return {
        "turn_count": len(rows),
        "candidate_count_mean": _mean([float(row["candidate_count"]) for row in rows]),
        "cloud_local_agreement": round(sum(bool(row["agreement"]) for row in rows) / len(rows), 9) if rows else None,
        "local_fallback_rate": round(sum(bool(row["local_is_fallback"]) for row in rows) / len(rows), 9) if rows else None,
        "comparable_score_count": len(comparable),
        "local_minus_cloud_proxy_value_mean": _mean([float(row["local_score"]) - float(row["cloud_score"]) for row in comparable]),
        "local_minus_fallback_proxy_value_mean": _mean([float(row["local_score"]) - float(row["fallback_score"]) for row in comparable]),
        "runtime_ms_mean": _mean([float(row["elapsed_ms"]) for row in rows]),
        "runtime_ms_p95": _p95([float(row["elapsed_ms"]) for row in rows]),
        "all_choices_identify_frozen_candidates": all(0 <= int(row["local_candidate_index"]) < int(row["candidate_count"]) for row in rows),
        "deterministic_replay": True,
    }


def replay(paths: Sequence[Path]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for path in paths:
        events, receipt = _read_prefix(path)
        inputs.append(receipt)
        records.extend(_record(example) for example in selector_examples(events))
    records.sort(key=lambda row: (str(row["ts"]), str(row["turn_id"])))
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_family[str(record["family"])].append(record)
    families: dict[str, object] = {}
    for family, rows in sorted(by_family.items()):
        boundary = max(1, math.floor(0.8 * len(rows))) if rows else 0
        families[family] = {"all": _summary(rows), "chronological_first_80_percent": _summary(rows[:boundary]), "chronological_last_20_percent_retrospective": _summary(rows[boundary:])}
    result = {
        "contract": REPLAY_CONTRACT,
        "selector_backend_contract": SELECTOR_BACKEND_REQUEST_CONTRACT,
        "inputs": inputs,
        "turn_count": len(records),
        "families": families,
        "interpretation": {
            "status": "retrospective-development-replay",
            "counterfactual_boundary": "Proxy values come from already-frozen conditional and family models; they are not observed counterfactual game payoffs.",
            "suffix_boundary": "The last 20 percent was inspected during development and is not an untouched promotion set.",
            "future_gate": "Freeze this implementation and require nonnegative local-minus-fallback proxy value with zero illegal actions, zero nondeterministic fallbacks, and zero deadline-reserve violations on the next untouched chronological block before live collector use.",
        },
    }
    return {**result, "result_sha256": _sha(result)}


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = replay(arguments.events)
    _atomic_write(arguments.output, result)
    print(_canonical({"output": str(arguments.output), "turn_count": result["turn_count"], "result_sha256": result["result_sha256"]}))


if __name__ == "__main__":
    main()
