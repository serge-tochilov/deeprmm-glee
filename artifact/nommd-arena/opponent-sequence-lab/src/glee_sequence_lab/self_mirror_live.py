"""Credential-free live public self-mirror scoring and prospective receipts."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import _event, canonical_json, classify_persuasion_signal, extract_events, file_sha256, object_sha256
from .live_shadow import _static_live_game
from .self_mirror import PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR, _sanitize_event, _sanitize_game
from .self_mirror_release import PublicSelfMirrorRelease, SELF_MIRROR_AUTHORITY


SELF_MIRROR_LIVE_IPC_CONTRACT = "glee-public-self-mirror-live-ipc-v1"
SELF_MIRROR_LIVE_SERVICE_CONTRACT = "glee-public-self-mirror-live-service-v1"
SELF_MIRROR_SURFACE_CONTRACT = "glee-public-self-mirror-candidate-surface-v1"
SELF_MIRROR_REGISTRY_CONTRACT = "glee-public-self-mirror-prospective-registry-v1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("candidate numeric value is absent or non-finite")
    return float(value)


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player: {player!r}")


def _bargaining_gain(action: Mapping[str, object], player: str) -> float:
    aliases = {
        "player_1": ("player_1_gain", "alice_gain"),
        "player_2": ("player_2_gain", "bob_gain"),
    }
    if player not in aliases:
        raise ValueError(f"unsupported player: {player!r}")
    for key in aliases[player]:
        if action.get(key) is not None:
            return _number(action[key])
    raise ValueError(f"Bargaining candidate lacks a gain for {player}")


def _target(*, game: Mapping[str, object], event_count: int, kind: str, label: str) -> dict[str, object]:
    game_id = str(game["game_id"])
    return {
        "sample_id": hashlib.sha256(f"public-self-mirror-live:{game_id}:{event_count}:{kind}".encode("utf-8")).hexdigest(),
        "game_id": game_id,
        "source_type": "prospective-live",
        "target_event_index": event_count,
        "prefix_length": event_count,
        "target_kind": kind,
        "target_label": label,
        "target_value": None,
        "target_value_present": False,
        "target_message_act": None,
        "target_message_present": False,
        "target_delay_log_ms": None,
        "target_delay_present": False,
        "chronological_split": "prospective",
        "identity_scope": str(game["identity_scope"]),
        "account_key": None,
        "account_confidence": None,
        "account_fold": -1,
    }


def _public_sample(game: Mapping[str, Any], *, extra_current_signal: bool = False, events_override: Sequence[Mapping[str, object]] | None = None, target_kind: str, target_label: str) -> tuple[dict[str, object], str]:
    original_static = _static_live_game(game)
    raw_events = [dict(event) for event in events_override] if events_override is not None else extract_events(game)
    if extra_current_signal:
        state = _mapping(game.get("game_state"), name="Persuasion state")
        round_number = state.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise ValueError("Persuasion buyer turn has no current round")
        channel = str(state.get("seller_message_type") or "text").casefold()
        message = state.get("seller_message")
        polarity, family_act, _fingerprint = classify_persuasion_signal(message, channel=channel)
        raw_events.append(
            _event(
                game_id=str(game.get("game_id") or ""),
                event_index=len(raw_events),
                round_number=round_number,
                state=state,
                actor="opponent",
                kind="signal",
                action_label=f"signal_{polarity}",
                action_value=1.0 if polarity == "positive" else -1.0 if polarity == "negative" else 0.0,
                message=message,
                family_act=family_act,
            )
        )
    public_static = _sanitize_game(original_static)
    public_events = [_sanitize_event(event, game=original_static) for event in raw_events]
    target = _target(game=public_static, event_count=len(public_events), kind=target_kind, label=target_label)
    sample = {"game": public_static, "events": public_events, "target": target}
    prefix_sha256 = object_sha256({"game": public_static, "events": public_events})
    return sample, prefix_sha256


def _candidate_label_and_value(*, game: Mapping[str, Any], action: Mapping[str, object], target_kind: str) -> tuple[str, float | None]:
    family = str(game.get("game_family") or "")
    if target_kind == "proposal":
        if family == "bargaining":
            state = _mapping(game.get("game_state"), name="Bargaining state")
            pool = _number(state.get("money_to_divide"))
            if pool <= 0:
                raise ValueError("Bargaining pool must be positive")
            opponent = _other_player(str(game.get("your_player") or ""))
            return "proposal", _bargaining_gain(action, opponent) / pool
        if family == "negotiation":
            return "proposal", math.log1p(_number(action.get("product_price"))) / PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR
        raise ValueError("Persuasion has no numeric proposal target")
    if family in {"bargaining", "negotiation"}:
        decision = str(action.get("decision") or "").casefold()
        labels = {"accept": "accept", "acceptoffer": "accept", "reject": "reject", "rejectoffer": "reject", "walkaway": "walkaway", "walk_away": "walkaway"}
        if decision not in labels:
            raise ValueError(f"unsupported response candidate: {decision!r}")
        return labels[decision], None
    state = _mapping(game.get("game_state"), name="Persuasion state")
    our_player = str(game.get("your_player") or "")
    role = str(state.get(f"{our_player}_role") or "").casefold()
    if role == "buyer":
        decision = str(action.get("decision") or "").casefold()
        if decision not in {"yes", "no"}:
            raise ValueError(f"unsupported Persuasion buyer candidate: {decision!r}")
        return ("buy" if decision == "yes" else "pass"), None
    action_type = str(_mapping(game.get("valid_actions"), name="valid actions").get("type") or game.get("phase") or "")
    if action_type == "seller_recommendation":
        decision = str(action.get("decision") or "").casefold()
        if decision not in {"yes", "no"}:
            raise ValueError(f"unsupported Persuasion recommendation: {decision!r}")
        return ("signal_positive" if decision == "yes" else "signal_negative"), None
    polarity, _family_act, _fingerprint = classify_persuasion_signal(action.get("message"), channel="text")
    return f"signal_{polarity}", None


def _gaussian_log_density(value: float, location: float, log_scale: float) -> float:
    bounded_log_scale = min(2.0, max(-4.0, log_scale))
    z = (value - location) * math.exp(-bounded_log_scale)
    return -0.5 * z * z - bounded_log_scale - 0.5 * math.log(2.0 * math.pi)


def _log_mean_exp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot ensemble an empty component score set")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values) / len(values))


def _component_scores(prediction: Mapping[str, object], *, label: str, value: float | None) -> dict[int, float]:
    labels = prediction.get("labels")
    components = prediction.get("components")
    if not isinstance(labels, list) or not isinstance(components, list):
        raise ValueError("self-mirror prediction is malformed")
    scores: dict[int, float] = {}
    for raw in components:
        component = _mapping(raw, name="self-mirror component prediction")
        seed = int(component["seed"])
        if value is None:
            if label not in labels:
                raise ValueError(f"self-mirror label is absent from the frozen vocabulary: {label!r}")
            probabilities = component.get("action_probabilities")
            if not isinstance(probabilities, list) or len(probabilities) != len(labels):
                raise ValueError("self-mirror action probabilities are malformed")
            scores[seed] = math.log(max(1e-12, float(probabilities[labels.index(label)])))
        else:
            scores[seed] = _gaussian_log_density(value, float(component["value_location"]), float(component["value_log_scale"]))
    return scores


class PublicSelfMirrorRegistry:
    """Store immutable candidate surfaces before selection and submitted actions afterward."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS forecasts (
                release_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                family TEXT NOT NULL,
                candidate_set_sha256 TEXT NOT NULL,
                visible_prefix_sha256 TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                forecast_sha256 TEXT NOT NULL,
                forecast_json TEXT NOT NULL,
                PRIMARY KEY (release_id, turn_id)
            );
            CREATE TABLE IF NOT EXISTS selections (
                release_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                forecast_sha256 TEXT NOT NULL,
                selected_candidate_index INTEGER,
                selected_action_sha256 TEXT,
                submitted_action_sha256 TEXT NOT NULL,
                selection_sha256 TEXT NOT NULL,
                selection_json TEXT NOT NULL,
                PRIMARY KEY (release_id, turn_id),
                FOREIGN KEY (release_id, turn_id) REFERENCES forecasts(release_id, turn_id)
            );
            """
        )
        self.connection.execute("INSERT OR IGNORE INTO registry_metadata(key, value) VALUES (?, ?)", ("contract", SELF_MIRROR_REGISTRY_CONTRACT))
        if self.connection.execute("SELECT value FROM registry_metadata WHERE key = ?", ("contract",)).fetchone() != (SELF_MIRROR_REGISTRY_CONTRACT,):
            raise ValueError("public self-mirror registry contract mismatch")

    def close(self) -> None:
        self.connection.close()

    def register_forecast(self, surface: Mapping[str, object], *, visible_prefix_sha256: str) -> dict[str, object]:
        if surface.get("contract") != SELF_MIRROR_SURFACE_CONTRACT or surface.get("authority") != SELF_MIRROR_AUTHORITY:
            raise ValueError("public self-mirror surface has the wrong contract or authority")
        payload = canonical_json(dict(surface))
        forecast_sha256 = object_sha256(dict(surface))
        key = (str(surface["release_id"]), str(surface["turn_id"]))
        identity = (str(surface["game_id"]), str(surface["family"]), str(surface["candidate_set_sha256"]), visible_prefix_sha256, forecast_sha256)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT game_id, family, candidate_set_sha256, visible_prefix_sha256, forecast_sha256 FROM forecasts WHERE release_id = ? AND turn_id = ?", key).fetchone()
            if existing is not None:
                if existing != identity:
                    raise ValueError("public self-mirror turn already has different immutable evidence")
                self.connection.execute("COMMIT")
                return {"status": "already-registered", "forecast_sha256": forecast_sha256}
            self.connection.execute("INSERT INTO forecasts(release_id, turn_id, game_id, family, candidate_set_sha256, visible_prefix_sha256, registered_at, forecast_sha256, forecast_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (*key, *identity[:4], _utc_now(), forecast_sha256, payload))
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "registered", "forecast_sha256": forecast_sha256}

    def record_selection(self, selection: Mapping[str, object]) -> dict[str, object]:
        key = (str(selection["release_id"]), str(selection["turn_id"]))
        forecast = self.connection.execute("SELECT forecast_sha256 FROM forecasts WHERE release_id = ? AND turn_id = ?", key).fetchone()
        if forecast is None or forecast[0] != selection.get("forecast_sha256"):
            raise ValueError("cannot record a self-mirror selection without its immutable forecast")
        payload = canonical_json(dict(selection))
        selection_sha256 = object_sha256(dict(selection))
        identity = (str(selection["forecast_sha256"]), selection.get("selected_candidate_index"), selection.get("selected_action_sha256"), str(selection["submitted_action_sha256"]), selection_sha256)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT forecast_sha256, selected_candidate_index, selected_action_sha256, submitted_action_sha256, selection_sha256 FROM selections WHERE release_id = ? AND turn_id = ?", key).fetchone()
            if existing is not None:
                if existing != identity:
                    raise ValueError("public self-mirror turn already has a different immutable selection")
                self.connection.execute("COMMIT")
                return {"status": "already-recorded", "selection_sha256": selection_sha256}
            self.connection.execute("INSERT INTO selections(release_id, turn_id, recorded_at, forecast_sha256, selected_candidate_index, selected_action_sha256, submitted_action_sha256, selection_sha256, selection_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (*key, _utc_now(), *identity[:-1], selection_sha256, payload))
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "recorded", "selection_sha256": selection_sha256}

    def summary(self) -> dict[str, object]:
        forecasts = int(self.connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0])
        selections = int(self.connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0])
        return {"contract": SELF_MIRROR_REGISTRY_CONTRACT, "forecasts": forecasts, "selections": selections, "pending": forecasts - selections, "path": str(self.path)}


class PublicSelfMirrorAdapter:
    """Score immutable candidates from public information and serialize GPU and registry access."""

    def __init__(self, *, release_dir: Path, registry_path: Path, device: str | None = None) -> None:
        self.release = PublicSelfMirrorRelease(release_dir, device=device)
        self.registry = PublicSelfMirrorRegistry(registry_path)
        self.lock = threading.Lock()
        self.started_at = _utc_now()

    def close(self) -> None:
        self.registry.close()

    def _prediction(self, sample: Mapping[str, object], *, family: str) -> Mapping[str, object]:
        return self.release.predict_components([sample], family=family)[0]

    def forecast(
        self,
        *,
        game: Mapping[str, Any],
        turn_id: str,
        candidate_set_sha256: str,
        candidates: Sequence[Mapping[str, object]],
        projected_game: Mapping[str, Any] | None = None,
        projected_candidates: Sequence[Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        with self.lock:
            if not turn_id.strip() or len(candidate_set_sha256) != 64 or not 1 <= len(candidates) <= 8:
                raise ValueError("public self-mirror forecast requires an identified immutable candidate set")
            family = str(game.get("game_family") or "")
            action_type = str(_mapping(game.get("valid_actions"), name="valid actions").get("type") or game.get("phase") or "")
            started = time.perf_counter()
            component_scores: list[dict[int, float]] = []
            prefix_hashes: list[str] = []
            if projected_game is not None:
                if family != "negotiation" or action_type != "decision" or not isinstance(projected_candidates, Sequence) or len(projected_candidates) != len(candidates):
                    raise ValueError("compound self-mirror projection is malformed")
                projected_events = extract_events(projected_game)
                if not projected_events or projected_events[-1].get("actor") != "self" or projected_events[-1].get("kind") != "response" or projected_events[-1].get("action_label") != "reject":
                    raise ValueError("compound self-mirror projection lacks the fixed self rejection")
                source_sample, source_hash = _public_sample(projected_game, events_override=projected_events[:-1], target_kind="response", target_label="reject")
                proposal_sample, proposal_hash = _public_sample(projected_game, events_override=projected_events, target_kind="proposal", target_label="proposal")
                source_scores = _component_scores(self._prediction(source_sample, family=family), label="reject", value=None)
                proposal_prediction = self._prediction(proposal_sample, family=family)
                for raw in projected_candidates:
                    projected = _mapping(raw, name="projected candidate")
                    action = _mapping(projected.get("action"), name="projected action")
                    label, value = _candidate_label_and_value(game=projected_game, action=action, target_kind="proposal")
                    proposal_scores = _component_scores(proposal_prediction, label=label, value=value)
                    if proposal_scores.keys() != source_scores.keys():
                        raise ValueError("compound self-mirror components lost seed alignment")
                    component_scores.append({seed: source_scores[seed] + proposal_scores[seed] for seed in source_scores})
                prefix_hashes.extend((source_hash, proposal_hash))
                frontier = "self-reject-current-offer-then-self-counteroffer"
            else:
                extra_signal = family == "persuasion" and action_type == "buyer_decision"
                target_kind = "proposal" if family in {"bargaining", "negotiation"} and action_type == "offer" else "signal" if family == "persuasion" and action_type in {"seller_message", "seller_recommendation"} else "response"
                default_label = "proposal" if target_kind == "proposal" else "signal_unknown" if target_kind == "signal" else "pass"
                sample, prefix_hash = _public_sample(game, extra_current_signal=extra_signal, target_kind=target_kind, target_label=default_label)
                prediction = self._prediction(sample, family=family)
                for raw in candidates:
                    candidate = _mapping(raw, name="candidate")
                    action = _mapping(candidate.get("action"), name="candidate action")
                    label, value = _candidate_label_and_value(game=game, action=action, target_kind=target_kind)
                    component_scores.append(_component_scores(prediction, label=label, value=value))
                prefix_hashes.append(prefix_hash)
                frontier = f"self-{target_kind}"
            if len(component_scores) != len(candidates):
                raise ValueError("public self-mirror did not score every candidate")
            ensemble_scores = [_log_mean_exp(list(scores.values())) for scores in component_scores]
            displayed_scores = [round(score, 8) for score in ensemble_scores]
            maximum = max(displayed_scores)
            normalizer = sum(math.exp(score - maximum) for score in displayed_scores)
            relative = [math.exp(score - maximum) / normalizer for score in displayed_scores]
            ranks = {candidate_index: 1 + sum(other > score for other in displayed_scores) for candidate_index, score in enumerate(displayed_scores)}
            percentiles = {
                candidate_index: 1.0
                if len(candidates) == 1
                else (sum(other < score for other in displayed_scores) + 0.5 * (sum(other == score for other in displayed_scores) - 1)) / (len(candidates) - 1)
                for candidate_index, score in enumerate(displayed_scores)
            }
            seed_choices: dict[int, frozenset[int]] = {}
            for seed in component_scores[0]:
                best = max(component_scores[index][seed] for index in range(len(component_scores)))
                seed_choices[seed] = frozenset(index for index in range(len(component_scores)) if math.isclose(component_scores[index][seed], best, rel_tol=0.0, abs_tol=1e-8))
            rows: list[dict[str, object]] = []
            for index, (raw, scores) in enumerate(zip(candidates, component_scores, strict=True)):
                candidate = _mapping(raw, name="candidate")
                values = list(scores.values())
                mean = sum(values) / len(values)
                disagreement = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
                rank = ranks[index]
                rows.append(
                    {
                        "candidate_index": int(candidate["candidate_index"]),
                        "action_sha256": str(candidate["action_sha256"]),
                        "forecast": {
                            "ensemble_log_expectedness": displayed_scores[index],
                            "relative_expectedness": round(relative[index], 8),
                            "expectedness_rank": rank,
                            "expectedness_percentile": round(percentiles[index], 8),
                            "component_log_expectedness": {str(seed): round(value, 8) for seed, value in sorted(scores.items())},
                            "component_log_score_stddev": round(disagreement, 8),
                            "component_top_choice_disagreement": len(set(seed_choices.values())) > 1,
                            "population_prediction": True,
                            "account_prediction": None,
                            "message_wording_scored": False,
                        },
                    }
                )
            visible_prefix_sha256 = object_sha256(prefix_hashes)
            surface = {
                "contract": SELF_MIRROR_SURFACE_CONTRACT,
                "service_contract": SELF_MIRROR_LIVE_SERVICE_CONTRACT,
                "authority": SELF_MIRROR_AUTHORITY,
                "release_id": self.release.release_id,
                "release_manifest_sha256": file_sha256(self.release.manifest_path),
                "turn_id": turn_id,
                "game_id": str(game.get("game_id") or ""),
                "family": family,
                "frontier": frontier,
                "candidate_set_sha256": candidate_set_sha256,
                "rows": rows,
                "selection_boundary": "Expectedness can distinguish candidates only inside the caller-supplied economically near-equivalent admissible set; it has no candidate-generation or hard-control authority.",
            }
            registration = self.registry.register_forecast(surface, visible_prefix_sha256=visible_prefix_sha256)
            return {**surface, "registration": registration, "forecast_sha256": registration["forecast_sha256"], "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3)}

    def record_selection(self, *, turn_id: str, forecast_sha256: str, selected_candidate_index: int | None, selected_action_sha256: str | None, submitted_action: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            selection = {
                "contract": "glee-public-self-mirror-selection-v1",
                "release_id": self.release.release_id,
                "turn_id": turn_id,
                "forecast_sha256": forecast_sha256,
                "selected_candidate_index": selected_candidate_index,
                "selected_action_sha256": selected_action_sha256,
                "submitted_action_sha256": object_sha256(dict(submitted_action)),
                "submitted_action": dict(submitted_action),
            }
            return {**self.registry.record_selection(selection), "selection": selection}

    def status(self) -> dict[str, object]:
        with self.lock:
            return {"contract": SELF_MIRROR_LIVE_SERVICE_CONTRACT, "release_id": self.release.release_id, "device": str(self.release.device), "started_at": self.started_at, "registry": self.registry.summary(), "authority": SELF_MIRROR_AUTHORITY}


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            self._respond({"status": "error", "error": "request is empty, oversized, or not newline terminated"})
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, Mapping) or request.get("contract") != SELF_MIRROR_LIVE_IPC_CONTRACT:
                raise ValueError("public self-mirror IPC contract mismatch")
            adapter: PublicSelfMirrorAdapter = self.server.adapter  # type: ignore[attr-defined]
            operation = str(request.get("operation") or "")
            if operation == "forecast":
                game = _mapping(request.get("game"), name="game")
                candidates = request.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError("forecast requires candidates")
                projected_game = request.get("projected_game")
                projected_candidates = request.get("projected_candidates")
                response = adapter.forecast(game=game, turn_id=str(request.get("turn_id") or ""), candidate_set_sha256=str(request.get("candidate_set_sha256") or ""), candidates=candidates, projected_game=projected_game if isinstance(projected_game, Mapping) else None, projected_candidates=projected_candidates if isinstance(projected_candidates, list) else None)
            elif operation == "record_selection":
                submitted = _mapping(request.get("submitted_action"), name="submitted action")
                selected_index = request.get("selected_candidate_index")
                response = adapter.record_selection(turn_id=str(request.get("turn_id") or ""), forecast_sha256=str(request.get("forecast_sha256") or ""), selected_candidate_index=int(selected_index) if isinstance(selected_index, int) and not isinstance(selected_index, bool) else None, selected_action_sha256=str(request.get("selected_action_sha256") or "") or None, submitted_action=submitted)
            elif operation == "status":
                response = adapter.status()
            else:
                raise ValueError(f"unsupported public self-mirror operation: {operation!r}")
        except Exception as error:
            response = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        self._respond(response)
        try:
            self.server.publish_status()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _respond(self, value: Mapping[str, object]) -> None:
        self.wfile.write(canonical_json(dict(value)).encode("utf-8") + b"\n")


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = True
    request_queue_size = 64

    def __init__(self, socket_path: str, adapter: PublicSelfMirrorAdapter, status_path: Path) -> None:
        self.adapter = adapter
        self.status_path = status_path
        super().__init__(socket_path, _RequestHandler)

    def publish_status(self) -> None:
        _atomic_json(self.status_path, {**self.adapter.status(), "socket": self.server_address, "pid": os.getpid(), "updated_at": _utc_now()})


def _prepare_socket(path: Path) -> None:
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
        raise RuntimeError(f"public self-mirror socket is already active: {path}")
    finally:
        probe.close()


def serve_public_self_mirror(*, release_dir: Path, registry_path: Path, socket_path: Path, status_path: Path, device: str | None = None) -> None:
    resolved_socket = socket_path.resolve()
    _prepare_socket(resolved_socket)
    adapter = PublicSelfMirrorAdapter(release_dir=release_dir, registry_path=registry_path, device=device)
    server = _UnixServer(str(resolved_socket), adapter, status_path.resolve())
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


def run_public_self_mirror_smoke(*, release_dir: Path, corpus_dir: Path, output_path: Path, device: str | None = None) -> dict[str, object]:
    from .data import CorpusIndex

    if output_path.exists():
        raise FileExistsError(f"public self-mirror smoke output already exists: {output_path}")
    release = PublicSelfMirrorRelease(release_dir, device=device)
    index = CorpusIndex([corpus_dir.resolve()])
    families: dict[str, object] = {}
    for family in ("bargaining", "negotiation", "persuasion"):
        dataset = index.subset(family=family, split="test", source_types={"real"})
        categorical = next((dataset[position] for position in range(len(dataset)) if dataset[position]["target"]["target_kind"] != "proposal"), None)
        proposal = next((dataset[position] for position in range(len(dataset)) if dataset[position]["target"]["target_kind"] == "proposal"), None)
        samples = [sample for sample in (categorical, proposal) if sample is not None]
        if not samples:
            raise ValueError(f"public self-mirror corpus has no test sample for {family}")
        families[family] = {"predictions": release.predict_components(samples, family=family), "latency": release.benchmark(samples, family=family, warmup=1, repetitions=3)}
    result = {"contract": "glee-public-self-mirror-transport-smoke-v1", "status": "passed", "release_id": release.release_id, "families": families, "authority": "transport-only"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    return result
