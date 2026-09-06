"""Immutable named-opponent evidence and offline Sol dossier synthesis."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .glee_policy import GLEE_FAMILIES
from .immutable_blob import hydrate_call_request
from .model_runner import ArenaCodexRunner

_SCHEMA_VERSION = 1
_SYNTHESIS_ROLE = "glee_named_opponent_synthesis"
_DEFAULT_MODEL = "gpt-5.6-sol"
_DEFAULT_EFFORT = "max"
_MAX_DIRECT_SYNTHESIS_CHARS = 350_000
_MAX_DIGEST_CHUNK_CHARS = 300_000


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def normalize_opponent_name(value: object) -> str:
    """Normalize only presentation whitespace while preserving server-supplied spelling."""
    return " ".join(str(value or "").split())


def named_opponent_id(name: str) -> str:
    """Return the same case-insensitive identity key used by the live broker."""
    normalized = normalize_opponent_name(name)
    if not normalized:
        raise ValueError("named opponent identity cannot be empty")
    return hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()[:20]


def _safe_label(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return (normalized or "source")[:80]


class FamilySynopsisDraft(BaseModel):
    """One model-authored, prompt-ready family projection."""

    game_family: Literal["bargaining", "negotiation", "persuasion"]
    confidence: int = Field(ge=0, le=100)
    prompt_synopsis: str = Field(min_length=1, max_length=2800)


class NamedOpponentSynthesisDraft(BaseModel):
    """Sol's derived opponent model before deterministic provenance is attached."""

    common_evidence_summary: str = Field(min_length=1, max_length=4000)
    opponent_model_of_us: str = Field(min_length=1, max_length=2400)
    recurring_tendencies: list[str] = Field(max_length=10)
    uncertainties: list[str] = Field(max_length=10)
    family_synopses: list[FamilySynopsisDraft] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def exactly_one_synopsis_per_family(self) -> NamedOpponentSynthesisDraft:
        actual = [item.game_family for item in self.family_synopses]
        if len(set(actual)) != 3 or set(actual) != set(GLEE_FAMILIES):
            raise ValueError(f"family_synopses must contain exactly one entry for each of {GLEE_FAMILIES}")
        return self


class FamilyEvidenceDigest(BaseModel):
    """Family-specific evidence retained from one complete chronological chunk."""

    game_family: Literal["bargaining", "negotiation", "persuasion"]
    evidence_notes: list[str] = Field(max_length=14)


class NamedOpponentChunkDigest(BaseModel):
    """Lossless-coverage semantic digest used only when one corpus exceeds direct context."""

    chunk_summary: str = Field(min_length=1, max_length=5000)
    authenticated_observations: list[str] = Field(max_length=24)
    model_hypotheses_and_errors: list[str] = Field(max_length=24)
    opponent_model_of_us: list[str] = Field(max_length=16)
    uncertainties: list[str] = Field(max_length=16)
    family_evidence: list[FamilyEvidenceDigest] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def exactly_one_digest_per_family(self) -> NamedOpponentChunkDigest:
        actual = [item.game_family for item in self.family_evidence]
        if len(set(actual)) != 3 or set(actual) != set(GLEE_FAMILIES):
            raise ValueError(f"family_evidence must contain exactly one entry for each of {GLEE_FAMILIES}")
        return self


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield line_number, value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [value for _line_number, value in _iter_jsonl(path)]


def _source_inventory(source_run: Path, *, require_complete: bool) -> tuple[list[dict[str, object]], str]:
    required = (source_run / "llm_calls.jsonl", source_run / "events.jsonl", source_run / "manifest.json")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source run is missing {missing[0]}")
    if require_complete and not (source_run / "complete.json").is_file():
        raise RuntimeError(f"source run is not complete: {source_run}")
    paths = [*required]
    complete = source_run / "complete.json"
    if complete.is_file():
        paths.append(complete)
    games_dir = source_run / "games"
    if games_dir.is_dir():
        paths.extend(sorted(games_dir.glob("*.json")))
    inventory = [
        {
            "path": str(path.relative_to(source_run)),
            "bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in paths
    ]
    return inventory, _sha(inventory)


def _event_linkage(events: Iterable[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    turn_games: dict[str, str] = {}
    worker_receipts: dict[str, dict[str, object]] = {}
    call_receipts: dict[str, dict[str, object]] = {}
    latest_turn_by_game: dict[str, str] = {}
    observed_counts: dict[str, int] = {}
    for event in events:
        turn_id = normalize_opponent_name(event.get("turn_id"))
        if event.get("kind") == "turn_observed":
            game = event.get("game")
            if isinstance(game, dict) and game.get("game_id") is not None:
                game_id = str(game["game_id"])
                observed_counts[game_id] = observed_counts.get(game_id, 0) + 1
                if not turn_id:
                    turn_id = f"{game_id}:observed:{event.get('turn') or observed_counts[game_id]}"
                turn_games[turn_id] = game_id
                latest_turn_by_game[game_id] = turn_id
        elif event.get("kind") in {"worker_finished", "move_submitted"} and isinstance(event.get("decision"), dict):
            decision = event["decision"]
            game_id = str(event.get("game_id") or turn_games.get(turn_id) or "")
            if not turn_id and game_id:
                turn_id = latest_turn_by_game.get(game_id, f"{game_id}:move:{len(worker_receipts) + 1}")
            selected_call_id = None
            if isinstance(decision.get("call_metadata"), dict):
                selected_call_id = decision["call_metadata"].get("call_id")
            branches: dict[str, str] = {}
            for receipt in decision.get("branch_receipts") or []:
                if not isinstance(receipt, dict) or not isinstance(receipt.get("call_metadata"), dict):
                    continue
                call_id = receipt["call_metadata"].get("call_id")
                if call_id:
                    branches[str(call_id)] = str(receipt.get("branch") or receipt.get("effort") or "unknown")
            receipt = {
                "selection_branch": decision.get("selection_branch"),
                "selected_call_id": selected_call_id,
                "call_branches": branches,
                "selected_action": decision.get("action"),
                "fallback": bool(decision.get("fallback")),
                "game_id": game_id or None,
                "turn_id": turn_id or None,
            }
            if turn_id:
                worker_receipts[turn_id] = receipt
            if selected_call_id:
                call_receipts[str(selected_call_id)] = receipt
            for call_id, branch in branches.items():
                call_receipts[call_id] = {**receipt, "branch": branch}
    return turn_games, worker_receipts, call_receipts


def _parse_worker_request(call: dict[str, Any], *, log_path: Path | None = None) -> dict[str, Any] | None:
    hydrated = hydrate_call_request(call, log_path=log_path) if log_path is not None else call
    request = hydrated.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("user"), str):
        return None
    try:
        payload = json.loads(request["user"])
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("game_family") not in GLEE_FAMILIES:
        return None
    opponent = payload.get("opponent")
    if not isinstance(opponent, dict) or opponent.get("type") == "hidden":
        return None
    name = normalize_opponent_name(opponent.get("name"))
    if not name:
        return None
    return payload


def _load_games(source_run: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    games_dir = source_run / "games"
    if not games_dir.is_dir():
        return result
    for path in sorted(games_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("game_id") is not None:
            result[str(value["game_id"])] = value
    return result


class NamedOpponentCorpusStore:
    """Extract exact named-opponent calls into immutable, content-verified shards."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "index.json"

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema_version": _SCHEMA_VERSION, "sources": {}, "opponents": {}}
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != _SCHEMA_VERSION or not isinstance(value.get("opponents"), dict):
            raise RuntimeError(f"unsupported named-dossier index: {self.index_path}")
        value.setdefault("sources", {})
        return value

    def extract_run(self, source_run: Path, *, require_complete: bool = True) -> dict[str, object]:
        source_run = source_run.resolve()
        inventory, source_sha = _source_inventory(source_run, require_complete=require_complete)
        events = _read_jsonl(source_run / "events.jsonl")
        turn_games, worker_receipts, call_receipts = _event_linkage(events)
        games = _load_games(source_run)
        calls = _read_jsonl(source_run / "llm_calls.jsonl")
        groups: dict[str, dict[str, Any]] = {}
        for line_index, call in enumerate(calls, start=1):
            hydrated_call = hydrate_call_request(call, log_path=source_run / "llm_calls.jsonl")
            payload = _parse_worker_request(hydrated_call)
            if payload is None:
                continue
            opponent_name = normalize_opponent_name(payload["opponent"]["name"])
            opponent_key = named_opponent_id(opponent_name)
            group = groups.setdefault(opponent_key, {"name": opponent_name, "aliases": set(), "calls": [], "prompts": {}, "game_ids": set()})
            group["aliases"].add(opponent_name)
            turn_receipt = payload.get("turn_receipt") if isinstance(payload.get("turn_receipt"), dict) else {}
            turn_id = normalize_opponent_name(turn_receipt.get("turn_id"))
            call_id = str(call.get("call_id") or "")
            call_receipt = call_receipts.get(call_id, {})
            if not turn_id:
                turn_id = str(call_receipt.get("turn_id") or "")
            game_id = turn_games.get(turn_id) or call_receipt.get("game_id")
            if game_id is None and ":r" in turn_id:
                game_id = turn_id.split(":r", 1)[0]
            if game_id:
                group["game_ids"].add(game_id)
            raw_call = copy.deepcopy(hydrated_call)
            request = raw_call.get("request") if isinstance(raw_call.get("request"), dict) else {}
            system_text = request.pop("system", None)
            system_sha = str(request.get("system_sha256") or hashlib.sha256(str(system_text or "").encode("utf-8")).hexdigest())
            if system_text is not None:
                previous = group["prompts"].setdefault(system_sha, {"prompt_version": call.get("prompt_version"), "system": system_text})
                if previous["system"] != system_text:
                    raise RuntimeError(f"system prompt hash collision in {source_run}: {system_sha}")
            worker = worker_receipts.get(turn_id, call_receipt)
            group["calls"].append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "model_call",
                    "evidence_id": _sha({"source_sha256": source_sha, "line": line_index, "call": call}),
                    "source_log_line": line_index,
                    "source_call_sha256": _sha(call),
                    "game_id": game_id,
                    "turn_id": turn_id,
                    "game_family": payload.get("game_family"),
                    "phase": payload.get("phase"),
                    "branch": (worker.get("call_branches") or {}).get(call_id) or call_receipt.get("branch"),
                    "selected": bool(call_id and call_id == worker.get("selected_call_id")),
                    "selected_action": worker.get("selected_action") if call_id and call_id == worker.get("selected_call_id") else None,
                    "call": raw_call,
                }
            )
        index = self._load_index()
        report: dict[str, object] = {"schema_version": _SCHEMA_VERSION, "source_run": str(source_run), "source_sha256": source_sha, "opponents": []}
        for opponent_key, group in sorted(groups.items(), key=lambda item: item[1]["name"].casefold()):
            source_label = _safe_label(source_run.name)
            relative_path = Path("opponents") / opponent_key / "raw" / f"{source_label}--{source_sha[:16]}.jsonl"
            path = self.root / relative_path
            records: list[dict[str, object]] = [
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "corpus_manifest",
                    "opponent": {"id": opponent_key, "name": group["name"], "aliases": sorted(group["aliases"], key=str.casefold)},
                    "source": {"label": source_run.name, "sha256": source_sha, "inventory": inventory},
                    "call_count": len(group["calls"]),
                    "game_ids": sorted(group["game_ids"]),
                    "contract": "Every named-opponent model-call row is preserved exactly except that repeated system text is factored into prompt records by SHA-256. Provider event streams, final responses, errors, partial failures, visible user context, selected and unselected branches, and transport metadata remain intact.",
                }
            ]
            records.extend(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "system_prompt",
                    "system_sha256": digest,
                    **prompt,
                }
                for digest, prompt in sorted(group["prompts"].items())
            )
            records.extend(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "final_game",
                    "game_id": game_id,
                    "game": games[game_id],
                }
                for game_id in sorted(group["game_ids"])
                if game_id in games
            )
            records.extend(group["calls"])
            text = "".join(_canonical(record) + "\n" for record in records)
            if path.is_file() and path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"immutable named-opponent shard changed: {path}")
            if not path.is_file():
                _atomic_text(path, text)
            shard_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            opponent_entry = index["opponents"].setdefault(opponent_key, {"name": group["name"], "aliases": [], "raw_shards": [], "current_summary": None})
            opponent_entry["aliases"] = sorted(set(opponent_entry.get("aliases") or []) | set(group["aliases"]), key=str.casefold)
            shard_entry = {
                "path": str(relative_path),
                "sha256": shard_sha,
                "source_run_sha256": source_sha,
                "call_count": len(group["calls"]),
                "game_ids": sorted(group["game_ids"]),
            }
            existing_paths = {item["path"] for item in opponent_entry["raw_shards"]}
            if shard_entry["path"] not in existing_paths:
                opponent_entry["raw_shards"].append(shard_entry)
                opponent_entry["raw_shards"].sort(key=lambda item: item["path"])
            report["opponents"].append({"id": opponent_key, "name": group["name"], **shard_entry})
        source_entry = {
            "label": source_run.name,
            "sha256": source_sha,
            "complete": (source_run / "complete.json").is_file(),
            "inventory": inventory,
            "named_opponent_ids": sorted(groups),
            "named_call_count": sum(len(group["calls"]) for group in groups.values()),
        }
        existing_source = index["sources"].get(source_sha)
        if existing_source is not None and existing_source != source_entry:
            raise RuntimeError(f"sealed source-run registry entry changed: {source_run}")
        index["sources"][source_sha] = source_entry
        index["updated_at"] = _now()
        _atomic_json(self.index_path, index)
        return report

    def _projection(self, opponent_key: str, opponent_entry: dict[str, Any]) -> tuple[dict[str, object], str, list[dict[str, Any]]]:
        shard_refs = sorted(opponent_entry.get("raw_shards") or [], key=lambda item: item["path"])
        if not shard_refs:
            raise RuntimeError(f"named opponent has no raw evidence: {opponent_key}")
        corpus_sha = _sha([{"path": item["path"], "sha256": item["sha256"]} for item in shard_refs])
        games: dict[str, dict[str, Any]] = {}
        turns: dict[tuple[str, str], dict[str, Any]] = {}
        for shard in shard_refs:
            path = self.root / shard["path"]
            if _sha_file(path) != shard["sha256"]:
                raise RuntimeError(f"named-opponent shard failed SHA-256 verification: {path}")
            for record in _read_jsonl(path):
                if record.get("kind") == "final_game" and isinstance(record.get("game"), dict):
                    games[str(record["game_id"])] = record["game"]
                elif record.get("kind") == "model_call":
                    call = record.get("call") if isinstance(record.get("call"), dict) else {}
                    request = call.get("request") if isinstance(call.get("request"), dict) else {}
                    try:
                        user_payload = json.loads(request.get("user") or "{}")
                    except ValueError:
                        user_payload = {}
                    if not isinstance(user_payload, dict):
                        user_payload = {}
                    visible_context = {
                        key: user_payload.get(key)
                        for key in ("objective", "turn_receipt", "game_family", "your_player", "phase", "opponent", "official_prompt", "visible_game_state", "valid_actions", "analytic_bargaining_reference")
                        if key in user_payload
                    }
                    provider = call.get("provider") if isinstance(call.get("provider"), dict) else {}
                    game_id = str(record.get("game_id") or "")
                    turn_id = str(record.get("turn_id") or "")
                    turn = turns.setdefault(
                        (game_id, turn_id),
                        {
                            "game_id": game_id,
                            "turn_id": turn_id,
                            "game_family": record.get("game_family"),
                            "phase": record.get("phase"),
                            "visible_contexts": {},
                            "model_calls": [],
                        },
                    )
                    context_sha = _sha(visible_context)
                    turn["visible_contexts"].setdefault(context_sha, visible_context)
                    turn["model_calls"].append(
                        {
                            "evidence_id": record.get("evidence_id"),
                            "source_shard": shard["path"],
                            "selected": record.get("selected"),
                            "selected_action": record.get("selected_action"),
                            "branch": record.get("branch"),
                            "model": call.get("model"),
                            "effort": call.get("effort"),
                            "ok": call.get("ok"),
                            "error": call.get("error"),
                            "model_output": {
                                "response": call.get("response"),
                                "provider_event_stream": provider.get("event_stream"),
                                "provider_reasoning_items": provider.get("reasoning_items"),
                                "provider_stderr": provider.get("stderr"),
                            },
                        }
                    )
        projected_turns = []
        for turn in turns.values():
            projected_turns.append(
                {
                    **{key: value for key, value in turn.items() if key != "visible_contexts"},
                    "visible_contexts": [turn["visible_contexts"][digest] for digest in sorted(turn["visible_contexts"])],
                }
            )
        packet: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "task": "Derive a compact named-opponent dossier from all preserved model branches and authenticated game outcomes.",
            "opponent": {"id": opponent_key, "name": opponent_entry["name"], "aliases": opponent_entry.get("aliases") or []},
            "corpus_sha256": corpus_sha,
            "evidence_contract": {
                "complete_model_output": "Every extracted selected, unselected, successful, failed, and partial model output is included; nothing is sampled or truncated.",
                "authentication": "Model outputs are hypotheses and contemplated actions, not facts about the opponent. Visible game states, attributed game history, legal server results, and final game records are authenticated evidence.",
                "identity": "The competition interface currently supplies a display name rather than an immutable opponent identifier. Evidence is joined only by normalized display name, so name reuse or renaming remains an explicit identity uncertainty.",
                "memory_exclusion": "The raw user prompts are preserved in immutable shards, but old tetrad_memory projections are excluded here because they can contain unrelated Self records and cannot authenticate a named opponent.",
                "transfer": "Cross-family transfer is allowed only as an explicit uncertain inference. Absence of direct evidence must be stated, not filled with generic strategy advice.",
            },
            "final_games": [games[game_id] for game_id in sorted(games)],
            "turns": projected_turns,
        }
        return packet, corpus_sha, list(games.values())

    @staticmethod
    def _final_results(packet: dict[str, object]) -> list[dict[str, object]]:
        results = []
        for game in packet["final_games"]:
            state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
            results.append(
                {
                    "game_id": game.get("game_id"),
                    "game_family": game.get("game_family"),
                    "your_player": game.get("your_player"),
                    "opponent": game.get("opponent"),
                    "status": game.get("status"),
                    "result": game.get("result"),
                    "final_round": state.get("round"),
                }
            )
        return results

    def _digest_packets(self, packet: dict[str, object]) -> list[dict[str, object]]:
        base = {
            "schema_version": _SCHEMA_VERSION,
            "task": "Digest this complete chronological corpus chunk for a later named-opponent synthesis.",
            "opponent": packet["opponent"],
            "corpus_sha256": packet["corpus_sha256"],
            "evidence_contract": packet["evidence_contract"],
            "final_results": self._final_results(packet),
        }
        chunks: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        for turn in packet["turns"]:
            candidate = [*current, turn]
            candidate_packet = {**base, "turns": candidate}
            if current and len(_canonical(candidate_packet)) > _MAX_DIGEST_CHUNK_CHARS:
                chunks.append(current)
                current = [turn]
            else:
                current = candidate
            if len(_canonical({**base, "turns": current})) > _MAX_DIGEST_CHUNK_CHARS:
                raise RuntimeError(f"one named-opponent turn exceeds the lossless digest limit of {_MAX_DIGEST_CHUNK_CHARS} characters")
        if current:
            chunks.append(current)
        return [
            {
                **base,
                "chunk": {"index": index, "count": len(chunks), "turn_count": len(turns)},
                "turns": turns,
            }
            for index, turns in enumerate(chunks, start=1)
        ]

    def _hierarchical_synthesis_packet(
        self,
        *,
        packet: dict[str, object],
        summary_dir: Path,
        runner: ArenaCodexRunner,
        model: str,
        effort: str,
    ) -> dict[str, object]:
        digest_envelopes: list[dict[str, object]] = []
        chunks = self._digest_packets(packet)
        for chunk in chunks:
            index = int(chunk["chunk"]["index"])
            chunk_dir = summary_dir / "chunks"
            input_path = chunk_dir / f"chunk-{index:03d}-input.json"
            digest_path = chunk_dir / f"chunk-{index:03d}-digest.json"
            if digest_path.is_file():
                envelope = json.loads(digest_path.read_text(encoding="utf-8"))
            else:
                _atomic_json(input_path, chunk)
                parsed, metadata = runner.call_structured("glee_named_opponent_digest", _canonical(chunk), NamedOpponentChunkDigest, model=model, effort=effort)
                metadata_value = metadata.__dict__ if hasattr(metadata, "__dict__") else metadata if isinstance(metadata, dict) else {"value": str(metadata)}
                evidence_ids = [
                    str(call["evidence_id"])
                    for turn in chunk["turns"]
                    for call in turn["model_calls"]
                    if call.get("evidence_id") is not None
                ]
                envelope = {
                    "schema_version": _SCHEMA_VERSION,
                    "chunk": chunk["chunk"],
                    "covered_evidence_ids": evidence_ids,
                    "call_metadata": metadata_value,
                    "digest": parsed.model_dump(mode="json"),
                }
                _atomic_json(digest_path, envelope)
            digest_envelopes.append(envelope)
        return {
            "schema_version": _SCHEMA_VERSION,
            "task": "Produce the final named-opponent dossier from lossless-coverage chronological Sol Max digests.",
            "opponent": packet["opponent"],
            "corpus_sha256": packet["corpus_sha256"],
            "evidence_contract": {
                **packet["evidence_contract"],
                "hierarchical_coverage": "Every raw model output was assigned to exactly one chronological digest chunk. The final synthesis sees every validated chunk digest; no turn was sampled or truncated.",
            },
            "final_results": self._final_results(packet),
            "chunk_digests": digest_envelopes,
        }

    def synthesize(
        self,
        *,
        prompts_dir: Path,
        model: str = _DEFAULT_MODEL,
        effort: str = _DEFAULT_EFFORT,
        timeout_s: int = 3600,
        opponent_names: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        index = self._load_index()
        selected_ids = {named_opponent_id(name) for name in opponent_names or ()}
        unknown = selected_ids - set(index["opponents"])
        if unknown:
            raise ValueError(f"requested named opponents are absent from the corpus: {sorted(unknown)}")
        report: dict[str, object] = {"schema_version": _SCHEMA_VERSION, "model": model, "effort": effort, "opponents": []}
        for opponent_key, opponent_entry in sorted(index["opponents"].items(), key=lambda item: item[1]["name"].casefold()):
            if selected_ids and opponent_key not in selected_ids:
                continue
            packet, corpus_sha, games = self._projection(opponent_key, opponent_entry)
            relative_dir = Path("opponents") / opponent_key / "summaries" / corpus_sha
            summary_dir = self.root / relative_dir
            dossier_path = summary_dir / "dossier.json"
            input_path = summary_dir / "input.json"
            log_path = summary_dir / "sol-max-calls.jsonl"
            if dossier_path.is_file():
                dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
                summary_sha = _sha_file(dossier_path)
            else:
                _atomic_json(input_path, packet)
                runner = ArenaCodexRunner(prompts_dir=prompts_dir, log_path=log_path, session_dir=summary_dir / ".cli-session", timeout_s=timeout_s, validation_retries=2)
                direct_body = _canonical(packet)
                if len(direct_body) <= _MAX_DIRECT_SYNTHESIS_CHARS:
                    synthesis_packet = packet
                    synthesis_mode = "direct-complete-corpus"
                    digest_count = 0
                else:
                    synthesis_packet = self._hierarchical_synthesis_packet(packet=packet, summary_dir=summary_dir, runner=runner, model=model, effort=effort)
                    synthesis_mode = "chronological-complete-coverage-digests"
                    digest_count = len(synthesis_packet["chunk_digests"])
                body = _canonical(synthesis_packet)
                parsed, metadata = runner.call_structured(_SYNTHESIS_ROLE, body, NamedOpponentSynthesisDraft, model=model, effort=effort)
                draft = parsed.model_dump(mode="json")
                family_drafts = {item["game_family"]: item for item in draft["family_synopses"]}
                game_ids_by_family = {
                    family: sorted(str(game["game_id"]) for game in games if game.get("game_family") == family and game.get("game_id") is not None)
                    for family in GLEE_FAMILIES
                }
                any_games = sorted(str(game["game_id"]) for game in games if game.get("game_id") is not None)
                family_synopses = []
                for family in GLEE_FAMILIES:
                    direct_ids = game_ids_by_family[family]
                    basis = "direct" if direct_ids else "cross-family-only" if any_games else "none"
                    family_synopses.append(
                        {
                            **family_drafts[family],
                            "evidence_basis": basis,
                            "direct_game_count": len(direct_ids),
                            "source_game_ids": direct_ids if direct_ids else any_games,
                        }
                    )
                metadata_value = metadata.__dict__ if hasattr(metadata, "__dict__") else metadata if isinstance(metadata, dict) else {"value": str(metadata)}
                dossier = {
                    "schema_version": _SCHEMA_VERSION,
                    "opponent": {"id": opponent_key, "name": opponent_entry["name"], "aliases": opponent_entry.get("aliases") or []},
                    "corpus_sha256": corpus_sha,
                    "generated_at": _now(),
                    "synthesis": {
                        "model": model,
                        "effort": effort,
                        "role": _SYNTHESIS_ROLE,
                        "mode": synthesis_mode,
                        "digest_count": digest_count,
                        "call_metadata": metadata_value,
                    },
                    "common_evidence_summary": draft["common_evidence_summary"],
                    "opponent_model_of_us": draft["opponent_model_of_us"],
                    "recurring_tendencies": draft["recurring_tendencies"],
                    "uncertainties": draft["uncertainties"],
                    "family_synopses": family_synopses,
                }
                _atomic_json(dossier_path, dossier)
                summary_sha = _sha_file(dossier_path)
            relative_dossier = str(relative_dir / "dossier.json")
            opponent_entry["current_summary"] = {"path": relative_dossier, "sha256": summary_sha, "corpus_sha256": corpus_sha}
            report["opponents"].append({"id": opponent_key, "name": opponent_entry["name"], **opponent_entry["current_summary"]})
        index["updated_at"] = _now()
        _atomic_json(self.index_path, index)
        return report


def freeze_named_dossier_snapshot(root: Path, destination: Path) -> dict[str, Any]:
    """Freeze current derived dossiers once per run so later synthesis cannot alter play."""
    if destination.is_file():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError(f"unsupported frozen named-dossier snapshot: {destination}")
        return value
    index_path = root / "index.json"
    opponents: dict[str, object] = {}
    index_sha: str | None = None
    if index_path.is_file():
        index_sha = _sha_file(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for opponent_key, entry in sorted((index.get("opponents") or {}).items()):
            summary = entry.get("current_summary") if isinstance(entry, dict) else None
            if not isinstance(summary, dict) or not summary.get("path"):
                continue
            path = root / str(summary["path"])
            if _sha_file(path) != summary.get("sha256"):
                raise RuntimeError(f"named-opponent summary failed SHA-256 verification: {path}")
            opponents[opponent_key] = json.loads(path.read_text(encoding="utf-8"))
    snapshot = {"schema_version": _SCHEMA_VERSION, "frozen_at": _now(), "source_index_sha256": index_sha, "opponents": opponents}
    _atomic_json(destination, snapshot)
    return snapshot


def named_dossier_view(snapshot: dict[str, Any] | None, opponent_name: str, game_family: str) -> dict[str, object] | None:
    """Return only the authenticated synthesis provenance and current family synopsis."""
    if snapshot is None or game_family not in GLEE_FAMILIES:
        return None
    opponent_key = named_opponent_id(opponent_name)
    dossier = (snapshot.get("opponents") or {}).get(opponent_key)
    if not isinstance(dossier, dict):
        return None
    family = next((item for item in dossier.get("family_synopses") or [] if item.get("game_family") == game_family), None)
    if not isinstance(family, dict):
        return None
    synthesis = dossier.get("synthesis") if isinstance(dossier.get("synthesis"), dict) else {}
    return {
        "provenance": {
            "kind": "offline-named-opponent-synthesis",
            "opponent_id": opponent_key,
            "corpus_sha256": dossier.get("corpus_sha256"),
            "generated_at": dossier.get("generated_at"),
            "model": synthesis.get("model"),
            "effort": synthesis.get("effort"),
        },
        "common_evidence_summary": dossier.get("common_evidence_summary"),
        "opponent_model_of_us": dossier.get("opponent_model_of_us"),
        "recurring_tendencies": dossier.get("recurring_tendencies") or [],
        "uncertainties": dossier.get("uncertainties") or [],
        "family_synopsis": family,
    }


def named_dossier_snapshot_sha(snapshot: dict[str, Any]) -> str:
    """Hash a frozen snapshot for run-manifest provenance."""
    return _sha(snapshot)


class NamedOpponentDossierPipeline:
    """Extract one or more complete runs, then synthesize selected or all opponents."""

    def __init__(self, *, project_root: Path, output_root: Path) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.store = NamedOpponentCorpusStore(output_root)

    def run(
        self,
        *,
        source_runs: tuple[Path, ...],
        model: str = _DEFAULT_MODEL,
        effort: str = _DEFAULT_EFFORT,
        timeout_s: int = 3600,
        opponent_names: tuple[str, ...] | None = None,
        extract_only: bool = False,
        require_complete: bool = True,
    ) -> dict[str, object]:
        if not source_runs and not self.store.index_path.is_file():
            raise ValueError("at least one source run is required before synthesis")
        extraction = [self.store.extract_run(path, require_complete=require_complete) for path in source_runs]
        synthesis = None
        if not extract_only:
            os.environ.pop("GLEE_API_KEY", None)
            synthesis = self.store.synthesize(prompts_dir=self.project_root / "prompts", model=model, effort=effort, timeout_s=timeout_s, opponent_names=opponent_names)
        return {"schema_version": _SCHEMA_VERSION, "output_root": str(self.output_root), "extraction": extraction, "synthesis": synthesis}
