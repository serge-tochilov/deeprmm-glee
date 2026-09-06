"""Append-only functional-tetrad memory with deterministic activation retrieval."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import CognitiveTrace, TetradDisposition, TetradLedgerRecord, TetradRecordKind, TetradUpdate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unique(values: Iterable[str], limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
        if len(result) == limit:
            break
    return result


class ActivationTetradLedger:
    """Preserve every ABDE trace and expose a bounded activation-ranked working set."""

    def __init__(self, *, root: Path, participants: list[str], main_desire: str, retrieval_limit: int, decay: float, known_minds: Iterable[str] = ()) -> None:
        self.root = root
        self.participants = list(participants)
        self.participant_set = set(participants)
        self.known_minds = self._load_known_minds(known_minds)
        self.mental_entity_set = self.participant_set | self.known_minds
        self.retrieval_limit = retrieval_limit
        self.decay = decay
        self._lock = threading.RLock()
        self._records: dict[str, list[TetradLedgerRecord]] = {participant: [] for participant in participants}
        self._record_ids: dict[str, set[str]] = {participant: set() for participant in participants}
        self._processed: dict[str, set[str]] = {participant: set() for participant in participants}
        self._published_events = self._load_published_events()
        self._correction_ids = self._load_correction_ids()
        self._stage_ticks = self._load_stage_ticks()
        for participant in participants:
            self._load_participant(participant)
            self._seed(participant, main_desire)

    def _load_known_minds(self, supplied: Iterable[str]) -> set[str]:
        path = self.root / "known-minds.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        result = {str(value).strip() for value in [*stored, *supplied] if str(value).strip()}
        invalid = sorted(value for value in result if len(value) > 40)
        if invalid:
            raise ValueError(f"tetrad-ledger mind labels exceed 40 characters: {invalid}")
        if sorted(result) != sorted(str(value) for value in stored):
            _write_json(path, sorted(result))
        return result

    def register_minds(self, minds: Iterable[str]) -> None:
        with self._lock:
            additions = {str(value).strip() for value in minds if str(value).strip()}
            invalid = sorted(value for value in additions if len(value) > 40)
            if invalid:
                raise ValueError(f"tetrad-ledger mind labels exceed 40 characters: {invalid}")
            if additions <= self.mental_entity_set:
                return
            self.known_minds.update(additions - self.participant_set)
            self.mental_entity_set.update(additions)
            _write_json(self.root / "known-minds.json", sorted(self.known_minds))

    def _load_stage_ticks(self) -> dict[str, int]:
        path = self.root / "stage-ticks.json"
        if not path.exists():
            return {"seed": 0}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(stage): int(tick) for stage, tick in payload.items()}

    def _load_published_events(self) -> dict[str, dict[str, object]]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return {}
        events: dict[str, dict[str, object]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            events[str(event["event_id"])] = event
        return events

    def _load_correction_ids(self) -> set[str]:
        path = self.root / "cognitive-corrections.jsonl"
        if not path.exists():
            return set()
        return {
            str(json.loads(line)["correction_id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def _load_participant(self, participant: str) -> None:
        records_path = self.root / participant / "records.jsonl"
        if records_path.exists():
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = TetradLedgerRecord.model_validate_json(line)
                self._records[participant].append(record)
                self._record_ids[participant].add(record.record_id)
        processed_path = self.root / participant / "processed.json"
        if processed_path.exists():
            self._processed[participant] = set(json.loads(processed_path.read_text(encoding="utf-8")))

    def _seed(self, participant: str, main_desire: str) -> None:
        desire = TetradLedgerRecord(
            record_id=f"{participant}:seed:main-desire",
            owner=participant,
            tick=0,
            stage_id="seed",
            kind=TetradRecordKind.DESIRE,
            disposition=TetradDisposition.ACTIVE,
            actor=participant,
            content=main_desire,
            strength=100,
            salience=100,
            mental_path=[],
            source_record_ids=[],
            tags=["self", "main-desire", "persistent"],
            visibility="internal",
        )
        emotion = TetradLedgerRecord(
            record_id=f"{participant}:seed:emotion",
            owner=participant,
            tick=0,
            stage_id="seed",
            kind=TetradRecordKind.EMOTION,
            disposition=TetradDisposition.FELT,
            actor=participant,
            content="Initial neutral state: moderate confidence, low urgency, and no frustration.",
            strength=35,
            salience=25,
            mental_path=[],
            source_record_ids=[],
            tags=["self", "emotion", "initial"],
            visibility="internal",
        )
        self._append(desire)
        self._append(emotion)
        self.mark_processed(participant, [desire.record_id, emotion.record_id])

    def stage_tick(self, stage_id: str) -> int:
        with self._lock:
            if stage_id in self._stage_ticks:
                return self._stage_ticks[stage_id]
            tick = max(self._stage_ticks.values(), default=-1) + 1
            self._stage_ticks[stage_id] = tick
            _write_json(self.root / "stage-ticks.json", self._stage_ticks)
            return tick

    def _append(self, record: TetradLedgerRecord) -> None:
        with self._lock:
            if record.record_id in self._record_ids[record.owner]:
                return
            path = self.root / record.owner / "records.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
            self._records[record.owner].append(record)
            self._record_ids[record.owner].add(record.record_id)

    def mark_processed(self, participant: str, record_ids: Iterable[str]) -> None:
        with self._lock:
            before = len(self._processed[participant])
            self._processed[participant].update(record_ids)
            if len(self._processed[participant]) != before:
                _write_json(self.root / participant / "processed.json", sorted(self._processed[participant]))

    def publish(self, *, event_id: str, stage_id: str, actor: str, content: str, viewers: Iterable[str], visibility: str, tags: Iterable[str], salience: int = 70, strength: int = 100) -> None:
        tick = self.stage_tick(stage_id)
        normalized_tags = _unique(["action", actor, *tags], 10)
        normalized_viewers = _unique(viewers, len(self.participants))
        event = {
            "event_id": event_id,
            "stage_id": stage_id,
            "tick": tick,
            "actor": actor,
            "content": content,
            "viewers": normalized_viewers,
            "visibility": visibility,
            "tags": normalized_tags,
            "salience": salience,
            "strength": strength,
        }
        with self._lock:
            prior = self._published_events.get(event_id)
            if prior is not None and prior != event:
                raise ValueError(f"tetrad-ledger event id changed on replay: {event_id}")
            if prior is None:
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                self._published_events[event_id] = event
        for viewer in normalized_viewers:
            if viewer not in self.participant_set:
                raise ValueError(f"unknown tetrad-ledger viewer: {viewer}")
            record = TetradLedgerRecord(
                record_id=f"{viewer}:{event_id}",
                owner=viewer,
                tick=tick,
                stage_id=stage_id,
                kind=TetradRecordKind.ACTION,
                disposition=TetradDisposition.OBSERVED,
                actor=actor,
                content=content,
                strength=strength,
                salience=salience,
                mental_path=[],
                source_record_ids=[],
                tags=normalized_tags,
                visibility=visibility,
            )
            self._append(record)

    def pending(self, participant: str) -> list[TetradLedgerRecord]:
        with self._lock:
            return [record for record in self._records[participant] if record.record_id not in self._processed[participant]]

    def _activation(self, record: TetradLedgerRecord, current_tick: int, cues: set[str], tag_counts: Counter[str]) -> tuple[float, int]:
        age = max(1, current_tick - record.tick + 1)
        base_level = -self.decay * math.log(age)
        salience = 1.25 * record.salience / 100.0
        strength = 0.75 * record.strength / 100.0
        recurrence_count = min(4, max((tag_counts[tag] for tag in record.tags), default=0))
        recurrence = 0.25 * math.log1p(recurrence_count)
        overlap = set(record.tags) & cues
        spreading = 0.8 * len(overlap) / max(1.0, math.sqrt(max(1, len(record.tags)) * max(1, len(cues))))
        return base_level + salience + strength + recurrence + spreading, age

    def retrieve(self, participant: str, *, stage_id: str, cues: Iterable[str]) -> list[dict[str, object]]:
        current_tick = self.stage_tick(stage_id)
        cue_set = set(_unique(cues, 32))
        with self._lock:
            records = [record for record in self._records[participant] if record.record_id in self._processed[participant]]
            tag_counts = Counter(tag for record in records if record.kind == TetradRecordKind.ACTION for tag in record.tags)
            ranked = [(self._activation(record, current_tick, cue_set, tag_counts), record) for record in records]
            ranked.sort(key=lambda item: (item[0][0], item[1].tick, item[1].record_id), reverse=True)
            companion_budget = min(3, max(0, self.retrieval_limit - 1))
            primary_count = self.retrieval_limit - companion_budget
            selected = ranked[:primary_count]
            selected_ids = {record.record_id for _score, record in selected}
            generic_tags = {"action", "belief", "desire", "emotion", "self", "persistent", "initial"}
            companions: list[tuple[tuple[float, int], TetradLedgerRecord]] = []
            for _score, anchor in selected:
                if anchor.kind == TetradRecordKind.ACTION:
                    continue
                anchor_tags = set(anchor.tags) - generic_tags
                if not anchor_tags:
                    continue
                for candidate_score, candidate in ranked:
                    if candidate.record_id in selected_ids or candidate.kind != anchor.kind or candidate.mental_path != anchor.mental_path:
                        continue
                    if anchor_tags & (set(candidate.tags) - generic_tags):
                        companions.append((candidate_score, candidate))
                        selected_ids.add(candidate.record_id)
                        break
                if len(companions) == companion_budget:
                    break
            selected.extend(companions)
            for candidate in ranked:
                if len(selected) == self.retrieval_limit:
                    break
                if candidate[1].record_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate[1].record_id)
            companion_ids = {record.record_id for _score, record in companions}
        result = [
            {
                **record.model_dump(mode="json"),
                "activation": round(score, 6),
                "age": age,
                "rmm_order": len(record.mental_path),
                "selection_reason": "linked-history" if record.record_id in companion_ids else "activation",
            }
            for (score, age), record in selected
        ]
        receipt = {
            "participant": participant,
            "stage_id": stage_id,
            "tick": current_tick,
            "cues": sorted(cue_set),
            "selected": [{"record_id": item["record_id"], "activation": item["activation"], "selection_reason": item["selection_reason"]} for item in result],
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "retrievals.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
        return result

    def context(self, participant: str, *, stage_id: str, cues: Iterable[str]) -> tuple[dict[str, object], dict[str, object]]:
        tick = self.stage_tick(stage_id)
        new_records = self.pending(participant)
        activated = self.retrieve(participant, stage_id=stage_id, cues=cues)
        slot_to_record_id: dict[int, str] = {}

        def expose(record: dict[str, object], slot: int) -> dict[str, object]:
            exposed = dict(record)
            exposed.pop("owner", None)
            exposed.pop("record_id", None)
            exposed.pop("source_record_ids", None)
            exposed["source_slot"] = slot
            return exposed

        new_context: list[dict[str, object]] = []
        for record in new_records:
            slot = len(slot_to_record_id)
            slot_to_record_id[slot] = record.record_id
            new_context.append(expose(record.model_dump(mode="json"), slot))
        activated_context: list[dict[str, object]] = []
        for record in activated:
            slot = len(slot_to_record_id)
            slot_to_record_id[slot] = str(record["record_id"])
            activated_context.append(expose(record, slot))
        context = {
            "contract": {
                "append_only": True,
                "new_information_priority": "Every new observation is supplied directly. Older traces are selected by bounded activation retrieval, with a small linked-history reserve for earlier records about the same subject.",
                "conflicts": "Never erase or silently replace a prior belief, desire, or emotion. Append a new trace; incompatible traces may coexist.",
                "mental_path": f"The current mind is implicit. [] is direct; [AU] models AU; [AU, VI] models AU's model of VI. Path length is the recursive modeling depth and may be at most 5. Allowed mind labels: {', '.join(sorted(self.mental_entity_set))}.",
                "desires": "A desire may later be satisfied or abandoned, but its earlier trace remains in the ledger.",
                "sources": "Cite evidence using integer source_slots shown on supplied records. Never copy or construct internal record identifiers.",
                "sparse_update": "Return 0 to 3 new cognitive traces. Use an empty update when the evidence does not warrant a meaningful belief, desire, or emotion change.",
            },
            "stage_id": stage_id,
            "tick": tick,
            "record_count": len(self._records[participant]),
            "new_observations": new_context,
            "activated_records": activated_context,
        }
        metadata = {
            "stage_id": stage_id,
            "tick": tick,
            "pending_record_ids": [record.record_id for record in new_records],
            "source_record_ids_by_slot": slot_to_record_id,
        }
        return context, metadata

    def validate_update(self, participant: str, update: TetradUpdate | None, metadata: dict[str, object]) -> list[str]:
        if update is None:
            return ["model returned no cognitive update"]
        slot_map = {int(slot): str(record_id) for slot, record_id in dict(metadata.get("source_record_ids_by_slot", {})).items()}
        issues: list[str] = []
        for index, trace in enumerate(update.updates, start=1):
            invalid_slots = [slot for slot in trace.source_slots if slot not in slot_map]
            if invalid_slots:
                issues.append(f"trace {index} cited unavailable source slots {invalid_slots}")
            unknown_minds = [mind for mind in trace.mental_path if mind not in self.mental_entity_set]
            if unknown_minds:
                issues.append(f"trace {index} used unknown mental-path minds {unknown_minds}")
        return issues

    def commit_update(self, participant: str, update: TetradUpdate | None, metadata: dict[str, object], external_issues: Iterable[str] = ()) -> dict[str, object]:
        stage_id = str(metadata["stage_id"])
        tick = int(metadata["tick"])
        slot_map = {int(slot): str(record_id) for slot, record_id in dict(metadata.get("source_record_ids_by_slot", {})).items()}
        issues = [*_unique(external_issues, 16), *self.validate_update(participant, update, metadata)]
        committed_ids: list[str] = []
        if update is not None:
            for index, trace in enumerate(update.updates, start=1):
                if any(mind not in self.mental_entity_set for mind in trace.mental_path):
                    continue
                source_record_ids = _unique((slot_map[slot] for slot in trace.source_slots if slot in slot_map), 8)
                record = self._record_from_trace(participant, stage_id, tick, index, trace, source_record_ids)
                self._append(record)
                committed_ids.append(record.record_id)
        self.mark_processed(participant, [*metadata.get("pending_record_ids", []), *committed_ids])
        receipt = {
            "correction_id": f"{participant}:{stage_id}",
            "participant": participant,
            "stage_id": stage_id,
            "tick": tick,
            "submitted_updates": len(update.updates) if update is not None else 0,
            "committed_record_ids": committed_ids,
            "issues": issues,
        }
        if issues:
            with self._lock:
                if receipt["correction_id"] not in self._correction_ids:
                    self.root.mkdir(parents=True, exist_ok=True)
                    with (self.root / "cognitive-corrections.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
                    self._correction_ids.add(str(receipt["correction_id"]))
        return receipt

    @staticmethod
    def _record_from_trace(participant: str, stage_id: str, tick: int, index: int, trace: CognitiveTrace, source_record_ids: list[str]) -> TetradLedgerRecord:
        return TetradLedgerRecord(
            record_id=f"{participant}:{stage_id}:bde:{index:02d}",
            owner=participant,
            tick=tick,
            stage_id=stage_id,
            kind=trace.kind,
            disposition=trace.disposition,
            actor=participant,
            content=trace.content,
            strength=trace.strength,
            salience=trace.salience,
            mental_path=trace.mental_path,
            source_record_ids=source_record_ids,
            tags=_unique(trace.tags, 10),
            visibility="internal",
        )

    def summary(self) -> dict[str, object]:
        participants: dict[str, object] = {}
        for participant in self.participants:
            records = self._records[participant]
            participants[participant] = {
                "records": len(records),
                "by_kind": dict(sorted(Counter(record.kind.value for record in records).items())),
                "by_rmm_order": dict(sorted(Counter(str(len(record.mental_path)) for record in records).items())),
                "max_rmm_order": max((len(record.mental_path) for record in records), default=0),
                "unprocessed": len(self.pending(participant)),
            }
        all_records = [record for participant in self.participants for record in self._records[participant]]
        return {
            "mode": "activation_ledger",
            "append_only": True,
            "retrieval_limit": self.retrieval_limit,
            "decay": self.decay,
            "known_minds": sorted(self.known_minds),
            "stages": len(self._stage_ticks),
            "records": len(all_records),
            "authenticated_events": len(self._published_events),
            "cognitive_correction_receipts": len(self._correction_ids),
            "by_rmm_order": dict(sorted(Counter(str(len(record.mental_path)) for record in all_records).items())),
            "max_rmm_order": max((len(record.mental_path) for record in all_records), default=0),
            "participants": participants,
        }

    def write_summary(self) -> dict[str, object]:
        summary = self.summary()
        _write_json(self.root / "summary.json", summary)
        return summary
