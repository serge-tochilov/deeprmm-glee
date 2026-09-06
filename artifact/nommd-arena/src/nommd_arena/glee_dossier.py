"""Single-writer, versioned ABDE dossier broker for parallel GLEE play."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .glee_nommd import GLEE_MAIN_DESIRE
from .glee_semantics import model_static_game_context, model_visible_game
from .glee_statistical_package import OpponentStatisticalPackageReader
from .glee_tactics import GlobalTacticLedger
from .models import CognitiveTrace, TetradUpdate

_CONTENT_LIMIT = 1200
_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bounded(label: str, value: object) -> str:
    content = f"{label}: {_canonical(value)}"
    if len(content) <= _CONTENT_LIMIT:
        return content
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    marker = f" ... [projection truncated; sha256={digest}] ... "
    remaining = _CONTENT_LIMIT - len(marker)
    head = remaining * 3 // 5
    return content[:head] + marker + content[-(remaining - head) :]


def _clean_label(value: object) -> str:
    return " ".join(str(value or "").split())


def _hidden_label(game_id: str) -> str:
    candidate = f"HIDDEN:{game_id}"
    if len(candidate) <= 40:
        return candidate
    return f"HIDDEN:{hashlib.sha256(game_id.encode()).hexdigest()[:16]}"


def _configuration(game: dict[str, Any]) -> dict[str, object]:
    return model_static_game_context(game)


@dataclass(frozen=True)
class DossierScopes:
    self_scope: str
    game_scope: str
    opponent_scope: str
    population_scope: str
    agent_label: str
    opponent_label: str
    opponent_persistent: bool

    def keys(self) -> tuple[str, str, str, str]:
        return (self.self_scope, self.game_scope, self.opponent_scope, self.population_scope)

    def as_dict(self) -> dict[str, object]:
        return {
            "self": self.self_scope,
            "game": self.game_scope,
            "opponent": self.opponent_scope,
            "population": self.population_scope,
            "agent_label": self.agent_label,
            "opponent_label": self.opponent_label,
            "opponent_persistent": self.opponent_persistent,
        }


@dataclass(frozen=True)
class DossierSnapshot:
    turn_id: str
    state_hash: str
    scopes: DossierScopes
    versions: dict[str, int]
    memory_context: dict[str, object]
    source_record_ids_by_slot: dict[int, str]
    snapshot_id: str

    def metadata(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "state_hash": self.state_hash,
            "snapshot_id": self.snapshot_id,
            "scopes": self.scopes.as_dict(),
            "versions": self.versions,
            "source_record_ids_by_slot": self.source_record_ids_by_slot,
        }


class DossierConflictError(RuntimeError):
    """Report a stable receipt identifier reused for different evidence."""


class DossierBroker:
    """Own all durable dossier writes and issue immutable version-pinned snapshots."""

    def __init__(
        self,
        *,
        root: Path,
        agent_name: str,
        retrieval_limit: int = 12,
        decay: float = 0.5,
        opponent_statistical_package_reader: OpponentStatisticalPackageReader | None = None,
        global_tactic_ledger: GlobalTacticLedger | None = None,
    ) -> None:
        if retrieval_limit < 1:
            raise ValueError("retrieval_limit must be positive")
        if decay < 0:
            raise ValueError("decay must be nonnegative")
        self.root = root
        self.agent_name = agent_name
        self.retrieval_limit = retrieval_limit
        self.decay = decay
        self.opponent_statistical_package_reader = opponent_statistical_package_reader
        self.global_tactic_ledger = global_tactic_ledger
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "dossiers.sqlite3"
        self.receipts_path = self.root / "broker-receipts.jsonl"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()
        self._seed_self()

    def close(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scope_versions (
                scope_key TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                event_time TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                game_id TEXT,
                turn_id TEXT,
                record_type TEXT NOT NULL,
                kind TEXT NOT NULL,
                disposition TEXT NOT NULL,
                actor TEXT NOT NULL,
                content TEXT NOT NULL,
                strength INTEGER NOT NULL,
                salience INTEGER NOT NULL,
                mental_path_json TEXT NOT NULL,
                source_record_ids_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                visibility TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS record_scopes (
                record_id TEXT NOT NULL REFERENCES records(record_id),
                scope_key TEXT NOT NULL REFERENCES scope_versions(scope_key),
                scope_version INTEGER NOT NULL,
                PRIMARY KEY (record_id, scope_key)
            );
            CREATE INDEX IF NOT EXISTS record_scopes_scope_idx ON record_scopes(scope_key, scope_version);
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                opponent_label TEXT NOT NULL,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                first_seen_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id),
                family TEXT NOT NULL,
                phase TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                snapshot_id TEXT,
                base_versions_json TEXT,
                worker_decision_json TEXT,
                prepared_action_json TEXT,
                accepted_action_json TEXT,
                server_result_json TEXT,
                issues_json TEXT,
                stale_scopes_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS turns_game_idx ON turns(game_id, created_at);
            """
        )
        existing = self._connection.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
        if existing is not None and int(existing["value"]) != _SCHEMA_VERSION:
            raise RuntimeError(f"unsupported dossier schema version: {existing['value']}")
        self._connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', ?)", (str(_SCHEMA_VERSION),))

    def _receipt(self, kind: str, **values: object) -> None:
        record = {"schema_version": 1, "ts": _now(), "kind": kind, **values}
        with self.receipts_path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _transaction(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def scopes_for_game(game: dict[str, Any], agent_name: str) -> DossierScopes:
        game_id = str(game["game_id"])
        opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
        name = _clean_label(opponent.get("name"))
        persistent = bool(name and opponent.get("type") != "hidden")
        if persistent:
            opponent_label = name[:40]
            identity_hash = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:20]
            opponent_scope = f"opponent:named:{identity_hash}"
        else:
            opponent_label = _hidden_label(game_id)
            opponent_scope = f"opponent:hidden:{game_id}"
        family = str(game["game_family"])
        fingerprint = _sha(_configuration(game))[:20]
        return DossierScopes(
            self_scope=f"self:{agent_name}",
            game_scope=f"game:{game_id}",
            opponent_scope=opponent_scope,
            population_scope=f"population:{family}:{fingerprint}",
            agent_label=agent_name,
            opponent_label=opponent_label,
            opponent_persistent=persistent,
        )

    @staticmethod
    def state_hash(game: dict[str, Any]) -> str:
        return _sha(game)

    @classmethod
    def turn_id(cls, game: dict[str, Any]) -> str:
        state = game["game_state"]
        action_type = str(game["valid_actions"]["type"])
        return f"{game['game_id']}:r{state.get('round', 'na')}:{action_type}:{cls.state_hash(game)[:16]}"

    def _record_core(self, record: dict[str, object]) -> tuple[object, ...]:
        return tuple(record[name] for name in (
            "game_id",
            "turn_id",
            "record_type",
            "kind",
            "disposition",
            "actor",
            "content",
            "strength",
            "salience",
            "mental_path_json",
            "source_record_ids_json",
            "tags_json",
            "visibility",
            "payload_json",
        ))

    def _stored_core(self, row: sqlite3.Row) -> tuple[object, ...]:
        return tuple(row[name] for name in (
            "game_id",
            "turn_id",
            "record_type",
            "kind",
            "disposition",
            "actor",
            "content",
            "strength",
            "salience",
            "mental_path_json",
            "source_record_ids_json",
            "tags_json",
            "visibility",
            "payload_json",
        ))

    def _append_records_locked(self, records: Iterable[dict[str, object]]) -> dict[str, object]:
        prepared = list(records)
        missing_pairs: list[tuple[str, str]] = []
        new_records: list[dict[str, object]] = []
        touched_scopes: set[str] = set()
        for record in prepared:
            record_id = str(record["record_id"])
            existing = self._connection.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
            if existing is not None:
                if self._stored_core(existing) != self._record_core(record):
                    raise DossierConflictError(f"dossier record id changed on replay: {record_id}")
            else:
                new_records.append(record)
            for scope in record["scopes"]:
                pair = self._connection.execute("SELECT 1 FROM record_scopes WHERE record_id = ? AND scope_key = ?", (record_id, scope)).fetchone()
                if pair is None:
                    missing_pairs.append((record_id, str(scope)))
                    touched_scopes.add(str(scope))
        now = _now()
        for scope in sorted(touched_scopes):
            self._connection.execute("INSERT OR IGNORE INTO scope_versions(scope_key, version, updated_at) VALUES(?, 0, ?)", (scope, now))
            self._connection.execute("UPDATE scope_versions SET version = version + 1, updated_at = ? WHERE scope_key = ?", (now, scope))
        for record in new_records:
            self._connection.execute(
                """
                INSERT INTO records(record_id, event_time, committed_at, game_id, turn_id, record_type, kind, disposition, actor, content, strength, salience, mental_path_json, source_record_ids_json, tags_json, visibility, payload_json)
                VALUES(:record_id, :event_time, :committed_at, :game_id, :turn_id, :record_type, :kind, :disposition, :actor, :content, :strength, :salience, :mental_path_json, :source_record_ids_json, :tags_json, :visibility, :payload_json)
                """,
                {**record, "committed_at": now},
            )
        for record_id, scope in missing_pairs:
            version = int(self._connection.execute("SELECT version FROM scope_versions WHERE scope_key = ?", (scope,)).fetchone()["version"])
            self._connection.execute("INSERT INTO record_scopes(record_id, scope_key, scope_version) VALUES(?, ?, ?)", (record_id, scope, version))
        versions = self._scope_versions_locked(touched_scopes)
        return {"record_ids": [str(record["record_id"]) for record in new_records], "touched_scopes": sorted(touched_scopes), "versions": versions}

    def _scope_versions_locked(self, scopes: Iterable[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for scope in scopes:
            row = self._connection.execute("SELECT version FROM scope_versions WHERE scope_key = ?", (scope,)).fetchone()
            result[str(scope)] = int(row["version"]) if row is not None else 0
        return result

    def _seed_self(self) -> None:
        scope = f"self:{self.agent_name}"
        records = [
            self._record(
                record_id=f"{scope}:seed:main-desire",
                scopes=[scope],
                record_type="seed",
                kind="desire",
                disposition="active",
                actor=self.agent_name,
                content=GLEE_MAIN_DESIRE,
                strength=100,
                salience=100,
                tags=["self", "main-desire", "persistent"],
                visibility="internal",
                payload={"main_desire": GLEE_MAIN_DESIRE},
            ),
            self._record(
                record_id=f"{scope}:seed:functional-emotion",
                scopes=[scope],
                record_type="seed",
                kind="emotion",
                disposition="felt",
                actor=self.agent_name,
                content="Initial neutral control state: moderate confidence, low urgency, and no frustration.",
                strength=35,
                salience=25,
                tags=["self", "emotion", "initial"],
                visibility="internal",
                payload={"confidence": 50, "urgency": 25, "frustration": 0},
            ),
        ]
        with self._lock:
            self._transaction()
            try:
                result = self._append_records_locked(records)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if result["record_ids"]:
            self._receipt("self_seeded", record_ids=result["record_ids"], versions=result["versions"])

    @staticmethod
    def _record(
        *,
        record_id: str,
        scopes: Iterable[str],
        record_type: str,
        kind: str,
        disposition: str,
        actor: str,
        content: str,
        strength: int,
        salience: int,
        tags: Iterable[str],
        visibility: str,
        payload: object,
        game_id: str | None = None,
        turn_id: str | None = None,
        mental_path: Iterable[str] = (),
        source_record_ids: Iterable[str] = (),
        event_time: str | None = None,
    ) -> dict[str, object]:
        return {
            "record_id": record_id,
            "event_time": event_time or _now(),
            "game_id": game_id,
            "turn_id": turn_id,
            "record_type": record_type,
            "kind": kind,
            "disposition": disposition,
            "actor": actor,
            "content": content,
            "strength": strength,
            "salience": salience,
            "mental_path_json": _canonical(list(mental_path)),
            "source_record_ids_json": _canonical(list(source_record_ids)),
            "tags_json": _canonical(list(dict.fromkeys(str(tag) for tag in tags))[:10]),
            "visibility": visibility,
            "payload_json": _canonical(payload),
            "scopes": tuple(dict.fromkeys(str(scope) for scope in scopes)),
        }

    def observe_turn(self, game: dict[str, Any]) -> DossierSnapshot:
        turn_id = self.turn_id(game)
        state_hash = self.state_hash(game)
        scopes = self.scopes_for_game(game, self.agent_name)
        family = str(game["game_family"])
        visible_game = model_visible_game(game)
        now = _now()
        context_record = self._record(
            record_id=f"engine:{game['game_id']}:context",
            scopes=[scopes.game_scope, scopes.opponent_scope, scopes.population_scope],
            record_type="engine-context",
            kind="action",
            disposition="observed",
            actor="ENGINE",
            content=_bounded("Game configuration", _configuration(game)),
            strength=100,
            salience=90,
            tags=["glee", family, "game-context", scopes.opponent_label],
            visibility="engine",
            payload=_configuration(game),
            game_id=str(game["game_id"]),
        )
        turn_record = self._record(
            record_id=f"engine:{turn_id}",
            scopes=[scopes.game_scope, scopes.opponent_scope, scopes.population_scope],
            record_type="engine-turn",
            kind="action",
            disposition="observed",
            actor="ENGINE",
            content=_bounded(f"Current visible {family} turn", visible_game),
            strength=100,
            salience=100,
            tags=["glee", family, "current-turn", scopes.opponent_label, str(game["valid_actions"]["type"]), f"round-{game['game_state'].get('round', 0)}"],
            visibility="engine",
            payload=visible_game,
            game_id=str(game["game_id"]),
            turn_id=turn_id,
        )
        with self._lock:
            self._transaction()
            try:
                existing = self._connection.execute("SELECT state_hash FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                if existing is not None and existing["state_hash"] != state_hash:
                    raise DossierConflictError(f"turn id changed on replay: {turn_id}")
                game_row = self._connection.execute("SELECT family, scopes_json FROM games WHERE game_id = ?", (str(game["game_id"]),)).fetchone()
                scopes_json = _canonical(scopes.as_dict())
                if game_row is not None and (game_row["family"] != family or game_row["scopes_json"] != scopes_json):
                    raise DossierConflictError(f"game identity changed on replay: {game['game_id']}")
                self._connection.execute(
                    "INSERT OR IGNORE INTO games(game_id, family, opponent_label, scopes_json, status, first_seen_at) VALUES(?, ?, ?, ?, 'active', ?)",
                    (str(game["game_id"]), family, scopes.opponent_label, scopes_json, now),
                )
                self._connection.execute(
                    "INSERT OR IGNORE INTO turns(turn_id, game_id, family, phase, state_hash, scopes_json, status, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, 'observed', ?, ?)",
                    (turn_id, str(game["game_id"]), family, str(game.get("phase") or game["valid_actions"]["type"]), state_hash, scopes_json, now, now),
                )
                dangling = self._connection.execute("SELECT turn_id, issues_json FROM turns WHERE game_id = ? AND turn_id != ? AND status IN ('observed', 'prepared', 'submitting', 'transport-suspended')", (str(game["game_id"]), turn_id)).fetchall()
                for prior in dangling:
                    issues = json.loads(prior["issues_json"]) if prior["issues_json"] else []
                    issues.append("server advanced to a later visible turn without a locally acknowledged submission receipt")
                    self._connection.execute("UPDATE turns SET status = 'reconciled', issues_json = ?, updated_at = ? WHERE turn_id = ?", (_canonical(list(dict.fromkeys(issues))), now, prior["turn_id"]))
                appended = self._append_records_locked([context_record, turn_record])
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        self._receipt("turn_observed", turn_id=turn_id, game_id=game["game_id"], state_hash=state_hash, reconciled_turn_ids=[str(row["turn_id"]) for row in dangling], appended=appended)
        return self.snapshot(game)

    def _eligible_snapshot_records_locked(self, scopes: DossierScopes) -> list[dict[str, object]]:
        """Materialize only current-game records and basal Self seeds; broad-scope statistics are aggregated separately."""
        placeholders = ",".join("?" for _ in scopes.keys())
        rows = self._connection.execute(
            f"""
            WITH eligible_record_ids(record_id) AS (
                SELECT record_id
                FROM record_scopes
                WHERE scope_key = ?
                UNION
                SELECT rs.record_id
                FROM record_scopes rs
                JOIN records r ON r.record_id = rs.record_id
                WHERE rs.scope_key = ? AND r.record_type = 'seed'
            )
            SELECT r.*, rs.scope_key
            FROM eligible_record_ids eligible
            JOIN records r ON r.record_id = eligible.record_id
            JOIN record_scopes rs ON rs.record_id = r.record_id
            WHERE rs.scope_key IN ({placeholders})
            ORDER BY r.seq
            """,
            (scopes.game_scope, scopes.self_scope, *scopes.keys()),
        ).fetchall()
        records: dict[str, dict[str, object]] = {}
        for row in rows:
            record_id = str(row["record_id"])
            record = records.setdefault(record_id, {name: row[name] for name in row.keys() if name != "scope_key"})
            record.setdefault("scopes", []).append(str(row["scope_key"]))
        return list(records.values())

    def _snapshot_scope_stats_locked(self, scopes: DossierScopes, current_game_id: str) -> tuple[int, int]:
        """Compute broad-scope counts without loading record payloads."""
        placeholders = ",".join("?" for _ in scopes.keys())
        scope_stats = self._connection.execute(
            f"""
            SELECT COUNT(DISTINCT r.record_id) AS record_count
            FROM records r
            JOIN record_scopes rs ON rs.record_id = r.record_id
            WHERE rs.scope_key IN ({placeholders})
            """,
            scopes.keys(),
        ).fetchone()
        opponent_prior_game_count = self._connection.execute(
            """
            SELECT COUNT(DISTINCT r.game_id) AS game_count
            FROM records r
            JOIN record_scopes rs ON rs.record_id = r.record_id
            WHERE rs.scope_key = ? AND r.game_id IS NOT NULL AND r.game_id != ?
            """,
            (scopes.opponent_scope, current_game_id),
        ).fetchone()
        return int(scope_stats["record_count"]), int(opponent_prior_game_count["game_count"])

    def snapshot(self, game: dict[str, Any]) -> DossierSnapshot:
        """Issue an operational snapshot without tetrad retrieval or activation ranking."""
        turn_id = self.turn_id(game)
        state_hash = self.state_hash(game)
        scopes = self.scopes_for_game(game, self.agent_name)
        current_game_id = str(game["game_id"])
        with self._lock:
            record_count, opponent_prior_game_count = self._snapshot_scope_stats_locked(scopes, current_game_id)
            versions = self._scope_versions_locked(scopes.keys())
        opponent_statistical_package = self.opponent_statistical_package_reader.view(game) if self.opponent_statistical_package_reader is not None else None
        global_tactic_memory = self.global_tactic_ledger.view(game) if self.global_tactic_ledger is not None else None
        context: dict[str, object] = {
            "contract": "glee-operational-context-v1",
            "identity": self.agent_name,
            "opponent_mind_label": scopes.opponent_label,
            "opponent_identity_scope": "persistent-named" if scopes.opponent_persistent else "game-local-hidden",
            "opponent_prior_game_count": opponent_prior_game_count,
            "scope_versions": {"self": versions[scopes.self_scope], "game": versions[scopes.game_scope], "opponent": versions[scopes.opponent_scope], "population": versions[scopes.population_scope]},
            "record_count": record_count,
        }
        if opponent_statistical_package is not None:
            context["opponent_statistical_package"] = opponent_statistical_package
        if global_tactic_memory is not None:
            context["global_tactic_memory"] = global_tactic_memory
        snapshot_id = _sha({"contract": context["contract"], "turn_id": turn_id, "versions": versions, "opponent_statistical_package": opponent_statistical_package, "global_tactic_memory": global_tactic_memory})
        with self._lock:
            self._connection.execute("UPDATE turns SET snapshot_id = ?, base_versions_json = ?, updated_at = ? WHERE turn_id = ?", (snapshot_id, _canonical(versions), _now(), turn_id))
        tactic_sha256 = self.global_tactic_ledger.sha256 if self.global_tactic_ledger is not None else None
        package_receipt = self.opponent_statistical_package_reader.receipt if self.opponent_statistical_package_reader is not None else None
        package_resolution = opponent_statistical_package.get("identity_resolution", {}).get("status") if isinstance(opponent_statistical_package, dict) else None
        package_overlay = opponent_statistical_package.get("live_overlay") if isinstance(opponent_statistical_package, dict) and isinstance(opponent_statistical_package.get("live_overlay"), dict) else None
        self._receipt("snapshot_issued", turn_id=turn_id, snapshot_id=snapshot_id, versions=versions, source_slots=0, tetrad_retrieval="disabled", global_tactic_ledger_sha256=tactic_sha256, global_tactic_trigger_count=len(global_tactic_memory.get("entries", [])) if isinstance(global_tactic_memory, dict) else 0, opponent_statistical_package_manifest_sha256=package_receipt.get("manifest_sha256") if isinstance(package_receipt, dict) else None, opponent_statistical_package_resolution=package_resolution, opponent_statistical_overlay_revision=package_overlay.get("revision") if isinstance(package_overlay, dict) else None, opponent_statistical_overlay_state_sha256=package_overlay.get("state_sha256") if isinstance(package_overlay, dict) else None)
        return DossierSnapshot(turn_id=turn_id, state_hash=state_hash, scopes=scopes, versions=versions, memory_context=context, source_record_ids_by_slot={}, snapshot_id=snapshot_id)

    def turn_receipt(self, turn_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        for name in ("scopes_json", "base_versions_json", "worker_decision_json", "prepared_action_json", "accepted_action_json", "server_result_json", "issues_json", "stale_scopes_json"):
            raw = result.pop(name)
            result[name.removesuffix("_json")] = json.loads(raw) if raw is not None else None
        return result

    def prepare_turn(self, snapshot: DossierSnapshot, decision: dict[str, object], action: dict[str, Any]) -> None:
        now = _now()
        with self._lock:
            self._transaction()
            try:
                row = self._connection.execute("SELECT status, state_hash, prepared_action_json FROM turns WHERE turn_id = ?", (snapshot.turn_id,)).fetchone()
                if row is None or row["state_hash"] != snapshot.state_hash:
                    raise DossierConflictError(f"cannot prepare unknown or changed turn: {snapshot.turn_id}")
                action_json = _canonical(action)
                if row["status"] == "accepted":
                    accepted = self._connection.execute("SELECT accepted_action_json FROM turns WHERE turn_id = ?", (snapshot.turn_id,)).fetchone()["accepted_action_json"]
                    if accepted != action_json:
                        raise DossierConflictError(f"accepted turn cannot be prepared with a different action: {snapshot.turn_id}")
                elif row["prepared_action_json"] is not None and row["prepared_action_json"] != action_json:
                    raise DossierConflictError(f"prepared action changed on replay: {snapshot.turn_id}")
                else:
                    self._connection.execute(
                        "UPDATE turns SET status = 'prepared', snapshot_id = ?, base_versions_json = ?, worker_decision_json = ?, prepared_action_json = ?, updated_at = ? WHERE turn_id = ?",
                        (snapshot.snapshot_id, _canonical(snapshot.versions), _canonical(decision), action_json, now, snapshot.turn_id),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        self._receipt("turn_prepared", turn_id=snapshot.turn_id, snapshot_id=snapshot.snapshot_id, action=action)

    def mark_submitting(self, turn_id: str) -> None:
        with self._lock:
            row = self._connection.execute("SELECT status FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            if row is None or row["status"] not in {"prepared", "submitting", "accepted"}:
                raise RuntimeError(f"turn is not prepared for submission: {turn_id}")
            if row["status"] != "accepted":
                self._connection.execute("UPDATE turns SET status = 'submitting', updated_at = ? WHERE turn_id = ?", (_now(), turn_id))
        self._receipt("submission_started", turn_id=turn_id)

    def reconcile_terminal_submission(self, turn_id: str, *, issue: str) -> None:
        """Close one prepared turn when the server has already made it terminal."""
        now = _now()
        with self._lock:
            self._transaction()
            try:
                row = self._connection.execute("SELECT status, issues_json FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                if row is None or row["status"] not in {"prepared", "submitting", "accepted", "reconciled"}:
                    raise RuntimeError(f"turn is not eligible for terminal reconciliation: {turn_id}")
                status = str(row["status"])
                if status not in {"accepted", "reconciled"}:
                    issues = json.loads(row["issues_json"]) if row["issues_json"] else []
                    issues.append(issue)
                    self._connection.execute("UPDATE turns SET status = 'reconciled', issues_json = ?, updated_at = ? WHERE turn_id = ?", (_canonical(list(dict.fromkeys(issues))), now, turn_id))
                    status = "reconciled"
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        self._receipt("submission_terminal_reconciled", turn_id=turn_id, issue=issue, status=status)

    def suspend_transport_submission(self, turn_id: str, *, issue: str) -> None:
        now = _now()
        with self._lock:
            self._transaction()
            try:
                row = self._connection.execute("SELECT status, issues_json FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                if row is None or row["status"] not in {"prepared", "submitting", "transport-suspended"}:
                    raise RuntimeError(f"turn is not eligible for transport suspension: {turn_id}")
                issues = json.loads(row["issues_json"]) if row["issues_json"] else []
                issues.append(issue)
                self._connection.execute("UPDATE turns SET status = 'transport-suspended', issues_json = ?, updated_at = ? WHERE turn_id = ?", (_canonical(list(dict.fromkeys(issues))), now, turn_id))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        self._receipt("submission_transport_suspended", turn_id=turn_id, issue=issue)

    def transport_suspended_turn_ids(self) -> set[str]:
        with self._lock:
            rows = self._connection.execute("SELECT turn_id FROM turns WHERE status = 'transport-suspended'").fetchall()
        return {str(row["turn_id"]) for row in rows}

    def record_rejection(self, turn_id: str, action: dict[str, Any], result: dict[str, Any]) -> None:
        self._receipt("submission_rejected", turn_id=turn_id, action=action, result=result)

    def commit_accepted(
        self,
        *,
        snapshot: DossierSnapshot,
        action: dict[str, Any],
        result: dict[str, Any],
        tetrad_update: TetradUpdate | None,
        issues: Iterable[str] = (),
    ) -> dict[str, object]:
        all_issues = list(dict.fromkeys(str(issue) for issue in issues))
        allowed_minds = {snapshot.scopes.agent_label, snapshot.scopes.opponent_label}
        records = [
            self._record(
                record_id=f"accepted:{snapshot.turn_id}",
                scopes=snapshot.scopes.keys(),
                record_type="accepted-action",
                kind="action",
                disposition="observed",
                actor=self.agent_name,
                content=_bounded("Accepted own action", {"action": action, "server_result": result}),
                strength=100,
                salience=95,
                tags=["glee", "own-action", str(action.get("decision") or "offer")],
                visibility="engine",
                payload={"action": action, "server_result": result},
                game_id=snapshot.scopes.game_scope.removeprefix("game:"),
                turn_id=snapshot.turn_id,
            )
        ]
        if tetrad_update is not None:
            for index, trace in enumerate(tetrad_update.updates, start=1):
                invalid_slots = [slot for slot in trace.source_slots if slot not in snapshot.source_record_ids_by_slot]
                unknown_minds = [mind for mind in trace.mental_path if mind not in allowed_minds]
                if invalid_slots:
                    all_issues.append(f"trace {index} discarded: unavailable source slots {invalid_slots}")
                if unknown_minds:
                    all_issues.append(f"trace {index} discarded: unknown mental-path minds {unknown_minds}")
                if invalid_slots or unknown_minds:
                    continue
                source_ids = [snapshot.source_record_ids_by_slot[slot] for slot in trace.source_slots]
                records.append(self._trace_record(snapshot, index, trace, source_ids))
        if result.get("game_over") or result.get("result") is not None:
            records.append(
                self._record(
                    record_id=f"result:{snapshot.scopes.game_scope}",
                    scopes=snapshot.scopes.keys(),
                    record_type="game-result",
                    kind="action",
                    disposition="observed",
                    actor="ENGINE",
                    content=_bounded("Final game result", result.get("result") or result),
                    strength=100,
                    salience=100,
                    tags=["glee", "game-result"],
                    visibility="engine",
                    payload=result.get("result") or result,
                    game_id=snapshot.scopes.game_scope.removeprefix("game:"),
                    turn_id=snapshot.turn_id,
                )
            )
        with self._lock:
            self._transaction()
            try:
                current_versions = self._scope_versions_locked(snapshot.scopes.keys())
                stale_scopes = sorted(scope for scope, expected in snapshot.versions.items() if current_versions.get(scope, 0) != expected)
                appended = self._append_records_locked(records)
                now = _now()
                row = self._connection.execute("SELECT accepted_action_json, server_result_json FROM turns WHERE turn_id = ?", (snapshot.turn_id,)).fetchone()
                if row is None:
                    raise DossierConflictError(f"accepted action has no observed turn: {snapshot.turn_id}")
                action_json = _canonical(action)
                result_json = _canonical(result)
                if row["accepted_action_json"] is not None and (row["accepted_action_json"] != action_json or row["server_result_json"] != result_json):
                    raise DossierConflictError(f"accepted receipt changed on replay: {snapshot.turn_id}")
                self._connection.execute(
                    "UPDATE turns SET status = 'accepted', accepted_action_json = ?, server_result_json = ?, issues_json = ?, stale_scopes_json = ?, updated_at = ? WHERE turn_id = ?",
                    (action_json, result_json, _canonical(all_issues), _canonical(stale_scopes), now, snapshot.turn_id),
                )
                if result.get("game_over"):
                    self._connection.execute("UPDATE games SET status = 'completed', result_json = ?, completed_at = ? WHERE game_id = ?", (_canonical(result.get("result") or result), now, snapshot.scopes.game_scope.removeprefix("game:")))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        receipt = {"turn_id": snapshot.turn_id, "appended": appended, "stale_scopes": stale_scopes, "issues": all_issues, "game_over": bool(result.get("game_over"))}
        self._receipt("turn_committed", **receipt)
        return receipt

    def _trace_record(self, snapshot: DossierSnapshot, index: int, trace: CognitiveTrace, source_ids: list[str]) -> dict[str, object]:
        scopes = [snapshot.scopes.self_scope, snapshot.scopes.game_scope, snapshot.scopes.opponent_scope]
        return self._record(
            record_id=f"cognitive:{snapshot.turn_id}:{index:02d}",
            scopes=scopes,
            record_type="cognitive",
            kind=trace.kind.value,
            disposition=trace.disposition.value,
            actor=self.agent_name,
            content=trace.content,
            strength=trace.strength,
            salience=trace.salience,
            tags=trace.tags,
            visibility="internal",
            payload=trace.model_dump(mode="json"),
            game_id=snapshot.scopes.game_scope.removeprefix("game:"),
            turn_id=snapshot.turn_id,
            mental_path=trace.mental_path,
            source_record_ids=source_ids,
        )

    def mark_game_completed(self, game_id: str, final_state: dict[str, Any]) -> None:
        with self._lock:
            row = self._connection.execute("SELECT scopes_json, status, result_json FROM games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                return
            result = final_state.get("result") or final_state
            result_json = _canonical(result)
            if row["status"] == "completed" and row["result_json"] not in (None, result_json):
                raise DossierConflictError(f"game result changed on replay: {game_id}")
            raw_scopes = json.loads(str(row["scopes_json"]))
            scopes = DossierScopes(
                self_scope=str(raw_scopes["self"]),
                game_scope=str(raw_scopes["game"]),
                opponent_scope=str(raw_scopes["opponent"]),
                population_scope=str(raw_scopes["population"]),
                agent_label=str(raw_scopes["agent_label"]),
                opponent_label=str(raw_scopes["opponent_label"]),
                opponent_persistent=bool(raw_scopes["opponent_persistent"]),
            )
            result_record = self._record(
                record_id=f"result:{scopes.game_scope}",
                scopes=scopes.keys(),
                record_type="game-result",
                kind="action",
                disposition="observed",
                actor="ENGINE",
                content=_bounded("Final game result", result),
                strength=100,
                salience=100,
                tags=["glee", "game-result"],
                visibility="engine",
                payload=result,
                game_id=game_id,
            )
            existing_result = self._connection.execute("SELECT turn_id FROM records WHERE record_id = ?", (f"result:{scopes.game_scope}",)).fetchone()
            if existing_result is not None:
                result_record["turn_id"] = existing_result["turn_id"]
            self._transaction()
            try:
                appended = self._append_records_locked([result_record])
                now = _now()
                self._connection.execute("UPDATE games SET status = 'completed', result_json = ?, completed_at = COALESCE(completed_at, ?) WHERE game_id = ?", (result_json, now, game_id))
                dangling = self._connection.execute("SELECT turn_id, issues_json FROM turns WHERE game_id = ? AND status IN ('prepared', 'submitting', 'transport-suspended')", (game_id,)).fetchall()
                for turn in dangling:
                    issues = json.loads(turn["issues_json"]) if turn["issues_json"] else []
                    issues.append("server advanced or completed without a locally acknowledged submission receipt")
                    self._connection.execute("UPDATE turns SET status = 'reconciled', issues_json = ?, updated_at = ? WHERE turn_id = ?", (_canonical(list(dict.fromkeys(issues))), now, turn["turn_id"]))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        self._receipt("game_completed_external", game_id=game_id, result=result, appended=appended)

    def known_active_game_ids(self) -> set[str]:
        with self._lock:
            rows = self._connection.execute("SELECT game_id FROM games WHERE status = 'active'").fetchall()
        return {str(row["game_id"]) for row in rows}

    def completed_game_ids(self) -> set[str]:
        with self._lock:
            rows = self._connection.execute("SELECT game_id FROM games WHERE status = 'completed'").fetchall()
        return {str(row["game_id"]) for row in rows}

    def game_counts_by_family(self, status: str) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute("SELECT family, COUNT(*) AS count FROM games WHERE status = ? GROUP BY family", (status,)).fetchall()
        return {str(row["family"]): int(row["count"]) for row in rows}

    def summary(self) -> dict[str, object]:
        with self._lock:
            record_count = int(self._connection.execute("SELECT COUNT(*) AS count FROM records").fetchone()["count"])
            scope_count = int(self._connection.execute("SELECT COUNT(*) AS count FROM scope_versions").fetchone()["count"])
            turn_counts = {str(row["status"]): int(row["count"]) for row in self._connection.execute("SELECT status, COUNT(*) AS count FROM turns GROUP BY status")}
            game_counts = {str(row["status"]): int(row["count"]) for row in self._connection.execute("SELECT status, COUNT(*) AS count FROM games GROUP BY status")}
        return {"schema_version": _SCHEMA_VERSION, "append_only": True, "single_writer": True, "records": record_count, "scopes": scope_count, "turns": turn_counts, "games": game_counts, "retrieval_limit": self.retrieval_limit, "decay": self.decay}
