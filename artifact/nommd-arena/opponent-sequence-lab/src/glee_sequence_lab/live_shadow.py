"""Pre-Terra visible-prefix shadow inference and append-only maturation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import socket
import socketserver
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import GLEE_FAMILIES, canonical_json, extract_events, object_sha256
from .shadow import ShadowCandidate


LIVE_SHADOW_IPC_CONTRACT = "glee-sequence-live-shadow-ipc-v1"
LIVE_SHADOW_CONTEXT_CONTRACT = "glee-sequence-pre-terra-context-v1"
LIVE_SHADOW_REGISTRY_CONTRACT = "glee-sequence-pre-terra-registry-v1"
LIVE_SHADOW_SERVICE_CONTRACT = "glee-sequence-live-shadow-service-v1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(value: object, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player: {player!r}")


def _identity_scope(game: Mapping[str, object]) -> tuple[str, str]:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
    name = str(opponent.get("name") or "").strip()
    hidden = opponent.get("type") == "hidden" or not name
    return ("hidden", "") if hidden else ("known", name.casefold())


def _static_live_game(game: Mapping[str, Any]) -> dict[str, object]:
    family = str(game.get("game_family") or "")
    if family not in GLEE_FAMILIES:
        raise ValueError(f"unsupported game family: {family!r}")
    state = game.get("game_state")
    if not isinstance(state, Mapping):
        raise ValueError("live game has no game_state object")
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    identity_scope, normalized_name = _identity_scope(game)
    complete_information = state.get("complete_information") is True
    horizon_known = state.get("horizon_known") is True or family == "persuasion"
    maximum = state.get("max_rounds") if family != "persuasion" else state.get("total_rounds")
    max_rounds = int(maximum) if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 else None
    static_scale_log = 0.0
    static_self_value = None
    static_visible_opponent_value = None
    static_environment_probability = None
    static_aux_value = None
    seller_knows_quality = False
    our_role = our_player
    opponent_role = opponent_player
    messages_allowed = state.get("messages_allowed") is True
    if family == "bargaining":
        pool = _number(state.get("money_to_divide"), 0.0) or 0.0
        static_scale_log = math.log1p(max(0.0, pool))
        static_self_value = _number(state.get("delta_1" if our_player == "player_1" else "delta_2"))
        static_visible_opponent_value = _number(state.get("delta_2" if our_player == "player_1" else "delta_1")) if complete_information else None
    elif family == "negotiation":
        our_role = str(state.get(f"{our_player}_role") or "")
        opponent_role = str(state.get(f"{opponent_player}_role") or "")
        static_self_value = _number(state.get(f"{our_player}_value"))
        static_visible_opponent_value = _number(state.get(f"{opponent_player}_value")) if complete_information else None
        static_scale_log = math.log1p(max(0.0, static_self_value or 0.0))
        messages_allowed = state.get("messages_allowed") is not False
    else:
        our_role = str(state.get(f"{our_player}_role") or "")
        opponent_role = str(state.get(f"{opponent_player}_role") or "")
        static_environment_probability = _number(state.get("p"))
        product_price = _number(state.get("product_price"), 0.0) or 0.0
        static_aux_value = math.log1p(max(0.0, product_price))
        static_scale_log = static_aux_value
        seller_knows_quality = state.get("is_seller_know_cv") is True
        messages_allowed = str(state.get("seller_message_type") or "text").casefold() in {"binary", "text"}
    return {
        "game_id": str(game.get("game_id") or ""),
        "family": family,
        "source_type": "prospective-live",
        "generator_id": None,
        "started_at": None,
        "completed_at": None,
        "chronological_split": "prospective",
        "identity_scope": identity_scope,
        "opponent_name_hash": hashlib.sha256((normalized_name or "hidden").encode("utf-8")).hexdigest(),
        "account_key": None,
        "account_confidence": None,
        "account_fold": -1,
        "our_player": our_player,
        "our_role": our_role,
        "opponent_role": opponent_role,
        "complete_information": complete_information,
        "horizon_known": horizon_known,
        "messages_allowed": messages_allowed,
        "max_rounds": max_rounds,
        "static_scale_log": static_scale_log,
        "static_self_value": static_self_value,
        "static_visible_opponent_value": static_visible_opponent_value,
        "static_environment_probability": static_environment_probability,
        "static_aux_value": static_aux_value,
        "static_seller_knows_quality": seller_knows_quality,
        "engine_version": "live-visible-prefix",
        "advisor_version": "",
        "policy_revision": "",
        "archive_path": None,
        "archive_sha256": None,
    }


@dataclass(frozen=True)
class LiveOpportunity:
    family: str
    game_id: str
    prefix_events: tuple[dict[str, object], ...]
    sample: dict[str, object]
    context: dict[str, object]
    prefix_sha256: str
    target_event_index: int
    target_kind: str
    causal_bridge_event_count: int


def build_pre_terra_opportunity(game: Mapping[str, Any], *, turn_id: str, synthetic_features: Mapping[str, object]) -> LiveOpportunity | None:
    """Build one direct-response forecast before the still-unknown self proposal or signal."""
    family = str(game.get("game_family") or "")
    valid_actions = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    action_type = str(valid_actions.get("type") or game.get("phase") or "")
    eligible = (family in {"bargaining", "negotiation"} and action_type == "offer") or (family == "persuasion" and action_type in {"seller_message", "seller_recommendation"})
    if not eligible:
        return None
    static_game = _static_live_game(game)
    if family == "persuasion" and static_game["our_role"] != "seller":
        return None
    events = extract_events(game)
    prefix_count = len(events)
    bridge_count = 1
    target_index = prefix_count + bridge_count
    placeholder = "pass" if family == "persuasion" else "reject"
    game_id = str(game.get("game_id") or "")
    target = {
        "sample_id": hashlib.sha256(f"pre-terra:{game_id}:{target_index}:response".encode("utf-8")).hexdigest(),
        "game_id": game_id,
        "source_type": "prospective-live",
        "target_event_index": target_index,
        "prefix_length": prefix_count,
        "target_kind": "response",
        "target_label": placeholder,
        "target_value": None,
        "target_value_present": False,
        "target_message_act": None,
        "target_message_present": False,
        "target_delay_log_ms": None,
        "target_delay_present": False,
        "chronological_split": "prospective",
        "identity_scope": static_game["identity_scope"],
        "account_key": None,
        "account_confidence": None,
        "account_fold": -1,
    }
    prefix = {"static_game": static_game, "events": events}
    context = {
        "contract": LIVE_SHADOW_CONTEXT_CONTRACT,
        "frontier": "authenticated-visible-prefix-before-terra",
        "turn_id": turn_id,
        "game_id": game_id,
        "family": family,
        "action_type": action_type,
        "visible_game_sha256": object_sha256(game),
        "static_game": static_game,
        "prefix_events": events,
        "target": {"event_index": target_index, "kind": "response", "causal_bridge_event_count": bridge_count, "bridge": "still-unobserved-self-proposal-or-signal"},
        "synthetic_features": dict(synthetic_features),
        "synthetic_features_sha256": object_sha256(dict(synthetic_features)),
        "model_consumed_synthetic_features": False,
    }
    return LiveOpportunity(
        family=family,
        game_id=game_id,
        prefix_events=tuple(events),
        sample={"game": static_game, "events": events, "target": target},
        context=context,
        prefix_sha256=object_sha256(prefix),
        target_event_index=target_index,
        target_kind="response",
        causal_bridge_event_count=bridge_count,
    )


def _warmup_opportunities() -> tuple[LiveOpportunity, ...]:
    """Build one registry-free startup sample for every frozen family head."""
    games: tuple[dict[str, object], ...] = (
        {
            "game_id": "__shadow_warmup_bargaining__",
            "game_family": "bargaining",
            "your_player": "player_1",
            "phase": "offer",
            "opponent": {"type": "hidden", "name": None},
            "valid_actions": {"type": "offer", "fields": {}},
            "game_state": {"history": [], "round": 1, "money_to_divide": 100.0, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "delta_1": 0.9, "delta_2": 0.8},
        },
        {
            "game_id": "__shadow_warmup_negotiation__",
            "game_family": "negotiation",
            "your_player": "player_1",
            "phase": "offer",
            "opponent": {"type": "hidden", "name": None},
            "valid_actions": {"type": "offer", "fields": {}},
            "game_state": {"history": [], "round": 1, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0},
        },
        {
            "game_id": "__shadow_warmup_persuasion__",
            "game_family": "persuasion",
            "your_player": "player_1",
            "phase": "seller_message",
            "opponent": {"type": "hidden", "name": None},
            "valid_actions": {"type": "seller_message", "fields": {}},
            "game_state": {"history": [], "round": 1, "total_rounds": 4, "p": 0.6, "product_price": 10.0, "seller_message_type": "text", "is_seller_know_cv": True, "player_1_role": "seller", "player_2_role": "buyer"},
        },
    )
    opportunities = tuple(build_pre_terra_opportunity(game, turn_id=f"warmup-{index}", synthetic_features={}) for index, game in enumerate(games, start=1))
    if any(opportunity is None for opportunity in opportunities):
        raise RuntimeError("failed to build a sequence-shadow startup sample")
    return tuple(opportunity for opportunity in opportunities if opportunity is not None)


class PreTerraShadowRegistry:
    """Append-only registry for bridged forecasts, exact pre-Terra contexts, and later outcomes."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS contexts (
                candidate_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_event_index INTEGER NOT NULL,
                target_kind TEXT NOT NULL,
                context_sha256 TEXT NOT NULL,
                context_json TEXT NOT NULL,
                PRIMARY KEY (candidate_id, game_id, target_event_index, target_kind)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                candidate_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_event_index INTEGER NOT NULL,
                target_kind TEXT NOT NULL,
                family TEXT NOT NULL,
                source_turn_id TEXT NOT NULL,
                prefix_event_count INTEGER NOT NULL,
                causal_bridge_event_count INTEGER NOT NULL,
                prefix_sha256 TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                prediction_sha256 TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                PRIMARY KEY (candidate_id, game_id, target_event_index, target_kind),
                FOREIGN KEY (candidate_id, game_id, target_event_index, target_kind) REFERENCES contexts(candidate_id, game_id, target_event_index, target_kind)
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                candidate_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_event_index INTEGER NOT NULL,
                target_kind TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                outcome_sha256 TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                PRIMARY KEY (candidate_id, game_id, target_event_index, target_kind),
                FOREIGN KEY (candidate_id, game_id, target_event_index, target_kind) REFERENCES predictions(candidate_id, game_id, target_event_index, target_kind)
            );
            """
        )
        self.connection.execute("INSERT OR IGNORE INTO registry_metadata(key, value) VALUES (?, ?)", ("contract", LIVE_SHADOW_REGISTRY_CONTRACT))
        contract = self.connection.execute("SELECT value FROM registry_metadata WHERE key = ?", ("contract",)).fetchone()
        if contract is None or contract[0] != LIVE_SHADOW_REGISTRY_CONTRACT:
            raise ValueError("pre-Terra shadow registry contract mismatch")

    def close(self) -> None:
        self.connection.close()

    def register(self, prediction: Mapping[str, object], *, context: Mapping[str, object], prefix_event_count: int, causal_bridge_event_count: int, prefix_sha256: str, source_turn_id: str, registered_at: str | None = None) -> dict[str, object]:
        if prefix_event_count < 0 or causal_bridge_event_count < 1 or prefix_event_count + causal_bridge_event_count != prediction.get("target_event_index"):
            raise ValueError("pre-Terra prediction must declare every unobserved bridge event before its target")
        if len(prefix_sha256) != 64 or any(character not in "0123456789abcdef" for character in prefix_sha256):
            raise ValueError("prefix SHA-256 must be lowercase hexadecimal")
        required = ("candidate_id", "game_id", "target_event_index", "target_kind", "family", "authority")
        if any(key not in prediction for key in required) or prediction.get("authority") != "prospective-shadow-only":
            raise ValueError("prediction is missing its prospective shadow identity or authority")
        if context.get("contract") != LIVE_SHADOW_CONTEXT_CONTRACT or context.get("model_consumed_synthetic_features") is not False:
            raise ValueError("pre-Terra context contract or feature-consumption boundary is invalid")
        key = (str(prediction["candidate_id"]), str(prediction["game_id"]), int(prediction["target_event_index"]), str(prediction["target_kind"]))
        prediction_payload = canonical_json(dict(prediction))
        prediction_sha256 = object_sha256(dict(prediction))
        context_payload = canonical_json(dict(context))
        context_sha256 = object_sha256(dict(context))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT p.prefix_sha256, p.prediction_sha256, c.context_sha256 FROM predictions p JOIN contexts c USING(candidate_id, game_id, target_event_index, target_kind) WHERE p.candidate_id = ? AND p.game_id = ? AND p.target_event_index = ? AND p.target_kind = ?", key).fetchone()
            if existing is not None:
                if tuple(existing) != (prefix_sha256, prediction_sha256, context_sha256):
                    raise ValueError("pre-Terra prediction key already has different immutable evidence")
                self.connection.execute("COMMIT")
                return {"status": "already-registered", "prediction_sha256": prediction_sha256, "context_sha256": context_sha256, "key": key}
            self.connection.execute("INSERT INTO contexts(candidate_id, game_id, target_event_index, target_kind, context_sha256, context_json) VALUES (?, ?, ?, ?, ?, ?)", (*key, context_sha256, context_payload))
            self.connection.execute("INSERT INTO predictions(candidate_id, game_id, target_event_index, target_kind, family, source_turn_id, prefix_event_count, causal_bridge_event_count, prefix_sha256, registered_at, prediction_sha256, prediction_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*key, str(prediction["family"]), source_turn_id, prefix_event_count, causal_bridge_event_count, prefix_sha256, registered_at or _utc_now(), prediction_sha256, prediction_payload))
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "registered", "prediction_sha256": prediction_sha256, "context_sha256": context_sha256, "key": key}

    def pending_for_game(self, *, candidate_id: str, game_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute("SELECT p.* FROM predictions p LEFT JOIN outcomes o USING(candidate_id, game_id, target_event_index, target_kind) WHERE p.candidate_id = ? AND p.game_id = ? AND o.candidate_id IS NULL ORDER BY p.target_event_index, p.target_kind", (candidate_id, game_id)).fetchall()
        return [dict(row) for row in rows]

    def record_outcome(self, *, candidate_id: str, game_id: str, target_event_index: int, target_kind: str, outcome: Mapping[str, object], observed_at: str | None = None) -> dict[str, object]:
        key = (candidate_id, game_id, target_event_index, target_kind)
        payload = canonical_json(dict(outcome))
        payload_sha256 = object_sha256(dict(outcome))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.connection.execute("SELECT 1 FROM predictions WHERE candidate_id = ? AND game_id = ? AND target_event_index = ? AND target_kind = ?", key).fetchone() is None:
                raise ValueError("cannot mature an outcome without a preregistered prediction")
            existing = self.connection.execute("SELECT outcome_sha256 FROM outcomes WHERE candidate_id = ? AND game_id = ? AND target_event_index = ? AND target_kind = ?", key).fetchone()
            if existing is not None:
                if existing[0] != payload_sha256:
                    raise ValueError("pre-Terra outcome key already has a different immutable outcome")
                self.connection.execute("COMMIT")
                return {"status": "already-recorded", "outcome_sha256": payload_sha256, "key": key}
            self.connection.execute("INSERT INTO outcomes(candidate_id, game_id, target_event_index, target_kind, observed_at, outcome_sha256, outcome_json) VALUES (?, ?, ?, ?, ?, ?, ?)", (*key, observed_at or _utc_now(), payload_sha256, payload))
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "recorded", "outcome_sha256": payload_sha256, "key": key}

    def summary(self) -> dict[str, object]:
        predictions = int(self.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        outcomes = int(self.connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])
        by_family = {str(row[0]): {"predictions": int(row[1]), "outcomes": int(row[2])} for row in self.connection.execute("SELECT p.family, COUNT(*), COUNT(o.candidate_id) FROM predictions p LEFT JOIN outcomes o USING(candidate_id, game_id, target_event_index, target_kind) GROUP BY p.family ORDER BY p.family")}
        return {"contract": LIVE_SHADOW_REGISTRY_CONTRACT, "predictions": predictions, "outcomes": outcomes, "pending": predictions - outcomes, "by_family": by_family}


class LiveShadowAdapter:
    """Own the frozen candidate and serialize every GPU call and registry transaction."""

    def __init__(self, *, release_dir: Path, registry_path: Path, device: str | None = None) -> None:
        self.candidate = ShadowCandidate(release_dir, device=device)
        self.registry = PreTerraShadowRegistry(registry_path)
        self.lock = threading.Lock()
        self.warmup_receipt = self._warmup()

    def _warmup(self) -> dict[str, object]:
        started = time.perf_counter()
        families: dict[str, float] = {}
        for opportunity in _warmup_opportunities():
            family_started = time.perf_counter()
            self.candidate.predict([opportunity.sample], family=opportunity.family)
            families[opportunity.family] = round((time.perf_counter() - family_started) * 1_000, 3)
        return {"status": "ready", "families": families, "elapsed_ms": round((time.perf_counter() - started) * 1_000, 3), "predictions_registered": 0}

    def close(self) -> None:
        self.registry.close()

    def forecast(self, *, game: Mapping[str, Any], turn_id: str, synthetic_features: Mapping[str, object], observed_at: str | None = None) -> dict[str, object]:
        with self.lock:
            opportunity = build_pre_terra_opportunity(game, turn_id=turn_id, synthetic_features=synthetic_features)
            if opportunity is None:
                return {"status": "ineligible", "reason": "no direct opponent-response target before this Terra decision"}
            started = time.perf_counter()
            prediction = self.candidate.predict([opportunity.sample], family=opportunity.family)[0]
            prediction.update({"forecast_frontier": "authenticated-visible-prefix-before-terra", "causal_bridge_event_count": opportunity.causal_bridge_event_count, "synthetic_features_sha256": opportunity.context["synthetic_features_sha256"], "synthetic_features_consumed": False})
            receipt = self.registry.register(prediction, context=opportunity.context, prefix_event_count=len(opportunity.prefix_events), causal_bridge_event_count=opportunity.causal_bridge_event_count, prefix_sha256=opportunity.prefix_sha256, source_turn_id=turn_id, registered_at=observed_at)
            return {
                "status": receipt["status"],
                "candidate_id": self.candidate.candidate_id,
                "game_id": opportunity.game_id,
                "target_event_index": opportunity.target_event_index,
                "target_kind": opportunity.target_kind,
                "prediction_sha256": receipt["prediction_sha256"],
                "context_sha256": receipt["context_sha256"],
                "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                "forecast_exposed_to_terra": False,
            }

    def observe(self, *, game: Mapping[str, Any], observed_at: str | None = None, source: str = "live-supervisor") -> dict[str, object]:
        with self.lock:
            game_id = str(game.get("game_id") or "")
            events = extract_events(game)
            pending = self.registry.pending_for_game(candidate_id=self.candidate.candidate_id, game_id=game_id)
            matured: list[dict[str, object]] = []
            mismatches: list[dict[str, object]] = []
            for row in pending:
                target_index = int(row["target_event_index"])
                if target_index >= len(events):
                    continue
                event = events[target_index]
                if event.get("actor") != "opponent" or event.get("kind") != row["target_kind"]:
                    mismatches.append({"target_event_index": target_index, "expected_kind": row["target_kind"], "actual_actor": event.get("actor"), "actual_kind": event.get("kind")})
                    continue
                outcome = {
                    "contract": "glee-sequence-pre-terra-outcome-v1",
                    "source": source,
                    "actual_action": event.get("action_label"),
                    "actual_event_sha256": object_sha256(event),
                    "observed_game_sha256": object_sha256(game),
                    "event": event,
                }
                matured.append(self.registry.record_outcome(candidate_id=self.candidate.candidate_id, game_id=game_id, target_event_index=target_index, target_kind=str(row["target_kind"]), outcome=outcome, observed_at=observed_at))
            return {"status": "observed", "game_id": game_id, "events": len(events), "pending_before": len(pending), "matured": len(matured), "mismatches": mismatches}

    def status(self) -> dict[str, object]:
        return {"contract": LIVE_SHADOW_SERVICE_CONTRACT, "candidate_id": self.candidate.candidate_id, "device": str(self.candidate.device), "warmup": self.warmup_receipt, "registry": self.registry.summary(), "authority": "prospective-shadow-only"}


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            self._respond({"status": "error", "error": "request is empty, oversized, or not newline terminated"})
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, Mapping) or request.get("contract") != LIVE_SHADOW_IPC_CONTRACT:
                raise ValueError("IPC contract mismatch")
            operation = str(request.get("operation") or "")
            adapter: LiveShadowAdapter = self.server.adapter  # type: ignore[attr-defined]
            if operation == "forecast":
                game = request.get("game")
                features = request.get("synthetic_features")
                if not isinstance(game, Mapping) or not isinstance(features, Mapping):
                    raise ValueError("forecast requires game and synthetic_features objects")
                response = adapter.forecast(game=game, turn_id=str(request.get("turn_id") or ""), synthetic_features=features, observed_at=str(request.get("observed_at") or "") or None)
            elif operation == "observe":
                game = request.get("game")
                if not isinstance(game, Mapping):
                    raise ValueError("observe requires a game object")
                response = adapter.observe(game=game, observed_at=str(request.get("observed_at") or "") or None, source=str(request.get("source") or "live-supervisor"))
            elif operation == "status":
                response = adapter.status()
            else:
                raise ValueError(f"unsupported operation: {operation!r}")
        except Exception as error:
            response = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        self._respond(response)
        try:
            self.server.publish_status()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _respond(self, value: Mapping[str, object]) -> None:
        self.wfile.write(canonical_json(dict(value)).encode("utf-8") + b"\n")


class _UnixShadowServer(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def __init__(self, socket_path: str, adapter: LiveShadowAdapter, status_path: Path) -> None:
        self.adapter = adapter
        self.status_path = status_path
        super().__init__(socket_path, _RequestHandler)

    def publish_status(self) -> None:
        _atomic_json(self.status_path, {**self.adapter.status(), "socket": self.server_address, "pid": os.getpid(), "updated_at": _utc_now()})


def _prepare_socket_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        return
    if not stat.S_ISSOCK(path.stat().st_mode):
        raise FileExistsError(f"refusing to replace a non-socket path: {path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(str(path))
    except OSError:
        path.unlink()
    else:
        raise RuntimeError(f"shadow service socket is already active: {path}")
    finally:
        probe.close()


def serve_live_shadow(*, release_dir: Path, registry_path: Path, socket_path: Path, status_path: Path, device: str | None = None) -> None:
    """Run the credential-free, behaviorally inert shadow sidecar until terminated."""
    resolved_socket = socket_path.resolve()
    _prepare_socket_path(resolved_socket)
    adapter = LiveShadowAdapter(release_dir=release_dir, registry_path=registry_path, device=device)
    server = _UnixShadowServer(str(resolved_socket), adapter, status_path.resolve())
    os.chmod(resolved_socket, 0o600)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def stop(_signal: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.publish_status()
        server.serve_forever(poll_interval=0.25)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        server.server_close()
        adapter.close()
        if resolved_socket.exists() and stat.S_ISSOCK(resolved_socket.stat().st_mode):
            resolved_socket.unlink()


def request_live_shadow(socket_path: Path, request: Mapping[str, object], *, timeout_s: float = 1.0) -> dict[str, object]:
    """Issue one bounded newline-delimited request to a local shadow sidecar."""
    if timeout_s <= 0:
        raise ValueError("shadow timeout must be positive")
    payload = canonical_json(dict(request)).encode("utf-8") + b"\n"
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("shadow request exceeds the IPC size limit")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_s)
    try:
        client.connect(str(socket_path))
        client.sendall(payload)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = client.recv(65_536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise ValueError("shadow response exceeds the IPC size limit")
            if b"\n" in chunk:
                break
    finally:
        client.close()
    raw = b"".join(chunks).partition(b"\n")[0]
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise ValueError("shadow response is not an object")
    return response
