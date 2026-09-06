"""Credential-free live service for post-planner candidate-response inference."""

from __future__ import annotations

import json
import os
import signal
import socket
import socketserver
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .buyer_continuation_live import BUYER_CONTINUATION_AUTHORITY, BUYER_CONTINUATION_LIVE_CONTRACT, PersuasionBuyerContinuationRelease
from .conditional_release import ConditionalTwinRelease
from .corpus import canonical_json, file_sha256, object_sha256
from .live_shadow import build_pre_terra_opportunity
from .pre_terra_conditional_v2 import CandidateAction
from .pre_terra_v3 import SEQUENCE_SHADOW_FEATURE_CONTRACT, EngineeredFeatureHasher


LIVE_CONDITIONAL_IPC_CONTRACT = "glee-post-planner-live-conditional-ipc-v1"
LIVE_CONDITIONAL_SERVICE_CONTRACT = "glee-post-planner-live-conditional-service-v1"
LIVE_CONDITIONAL_ACTIVATION_CONTRACT = "glee-terra-meta-controller-live-activation-v1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_activation(path: Path, release_dir: Path, buyer_continuation_release_dir: Path | None = None) -> dict[str, object]:
    activation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(activation, dict) or activation.get("contract") != LIVE_CONDITIONAL_ACTIVATION_CONTRACT or activation.get("status") != "active-controlled-online-evaluation":
        raise ValueError("conditional-twin activation is absent or inactive")
    model = activation.get("conditional_twin")
    if not isinstance(model, Mapping):
        raise ValueError("conditional-twin activation has no model receipt")
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model.get("candidate_id") != manifest.get("candidate_id") or model.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("conditional-twin activation does not identify the supplied frozen release")
    if activation.get("decision_authority") != "advisory-candidate-response-evidence-only":
        raise ValueError("conditional-twin activation grants unsupported decision authority")
    continuation = activation.get("persuasion_buyer_continuation")
    if buyer_continuation_release_dir is None:
        if continuation is not None:
            raise ValueError("activation requires an absent Persuasion buyer-continuation release")
    else:
        if not isinstance(continuation, Mapping):
            raise ValueError("activation does not authorize the supplied Persuasion buyer-continuation release")
        continuation_manifest_path = buyer_continuation_release_dir / "manifest.json"
        continuation_manifest = json.loads(continuation_manifest_path.read_text(encoding="utf-8"))
        if continuation.get("release_id") != continuation_manifest.get("release_id") or continuation.get("manifest_sha256") != file_sha256(continuation_manifest_path):
            raise ValueError("Persuasion buyer-continuation activation does not identify the supplied frozen release")
        if continuation.get("authority") != BUYER_CONTINUATION_AUTHORITY:
            raise ValueError("Persuasion buyer-continuation activation grants unsupported decision authority")
    return activation


def _collapse_candidate_projections(*, candidates: Sequence[Mapping[str, object]], family: str, phase: str) -> tuple[list[dict[str, object]], list[int]]:
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


class LiveConditionalAdapter:
    """Validate one immutable candidate set and evaluate it in one bounded local batch."""

    def __init__(self, *, release_dir: Path, activation_path: Path, buyer_continuation_release_dir: Path | None = None, buyer_continuation_registry_path: Path | None = None, device: str | None = None, release: Any | None = None, execution_runtime: Mapping[str, object] | None = None) -> None:
        self.release_dir = release_dir.resolve()
        self.activation_path = activation_path.resolve()
        self.buyer_continuation_release_dir = buyer_continuation_release_dir.resolve() if buyer_continuation_release_dir is not None else None
        self.activation = _load_activation(self.activation_path, self.release_dir, self.buyer_continuation_release_dir)
        self.release = release or ConditionalTwinRelease(self.release_dir, device=device)
        if self.release.candidate_id != self.activation["conditional_twin"]["candidate_id"]:
            raise ValueError("loaded conditional twin disagrees with its activation")
        self.hasher = EngineeredFeatureHasher()
        self.execution_runtime = dict(execution_runtime or {"backend": "frozen-native", "device": str(getattr(self.release, "device", device or "injected"))})
        self.buyer_continuation = PersuasionBuyerContinuationRelease(self.buyer_continuation_release_dir, sequence=self.release.sequence, registry_path=buyer_continuation_registry_path) if self.buyer_continuation_release_dir is not None else None
        self.started_at = _utc_now()
        self.request_count = 0
        self.failure_count = 0
        self.warmup_count = 0
        self.warmup_latency_s: float | None = None
        self.last_request_at: str | None = None
        self.last_latency_s: float | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self.buyer_continuation is not None:
            self.buyer_continuation.close()

    def forecast_candidates(self, *, game: Mapping[str, Any], turn_id: str, synthetic_features: Mapping[str, object], candidate_set_sha256: str, candidates: Sequence[Mapping[str, object]], _record_request: bool = True) -> dict[str, object]:
        started = time.monotonic()
        if not turn_id.strip() or len(candidate_set_sha256) != 64:
            raise ValueError("conditional request lacks a turn or candidate-set identity")
        if not 2 <= len(candidates) <= 5:
            raise ValueError("conditional request must contain 2 to 5 candidates")
        opportunity = build_pre_terra_opportunity(game, turn_id=turn_id, synthetic_features=synthetic_features)
        if opportunity is None:
            raise ValueError("turn is outside the direct-response conditional frontier")
        vector = self.hasher.project(synthetic_features)
        action_projections: list[dict[str, object]] = []
        candidate_receipts: list[tuple[int, str]] = []
        for expected_index, row in enumerate(candidates):
            if not isinstance(row, Mapping) or row.get("candidate_index") != expected_index:
                raise ValueError("conditional candidates are not in immutable index order")
            action_sha256 = str(row.get("action_sha256") or "")
            action = row.get("action")
            if len(action_sha256) != 64 or not isinstance(action, Mapping) or object_sha256(dict(action)) != action_sha256:
                raise ValueError("conditional candidate action hash is invalid")
            candidate_receipts.append((expected_index, action_sha256))
            action_projections.append(CandidateAction.from_live_action(game=game, action=action).receipt())
        unique, projection_indices = _collapse_candidate_projections(candidates=action_projections, family=opportunity.family, phase=str(game.get("phase") or game.get("valid_actions", {}).get("type") or ""))
        with self._lock:
            unique_forecasts = self.release.predict_candidates(
                sample=opportunity.sample,
                family=opportunity.family,
                phase=str(game.get("phase") or game.get("valid_actions", {}).get("type") or ""),
                base_feature_indices=vector.indices,
                base_feature_values=vector.values,
                base_feature_vector_sha256=vector.vector_sha256,
                candidates=unique,
            )
            forecasts = [dict(unique_forecasts[index]) for index in projection_indices]
            elapsed_s = round(time.monotonic() - started, 6)
            if _record_request:
                self.request_count += 1
                self.last_request_at = _utc_now()
                self.last_latency_s = elapsed_s
        rows = [
            {"candidate_index": index, "action_sha256": action_sha256, "forecast": forecast}
            for (index, action_sha256), forecast in zip(candidate_receipts, forecasts, strict=True)
        ]
        return {
            "status": "predicted",
            "contract": LIVE_CONDITIONAL_IPC_CONTRACT,
            "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT,
            "activation_id": self.activation.get("activation_id"),
            "controller": self.activation.get("controller"),
            "decision_authority": self.activation.get("decision_authority"),
            "candidate_id": self.release.candidate_id,
            "candidate_manifest_sha256": file_sha256(self.release_dir / "manifest.json"),
            "turn_id": turn_id,
            "game_id": opportunity.game_id,
            "family": opportunity.family,
            "candidate_set_sha256": candidate_set_sha256,
            "candidate_count": len(candidates),
            "unique_projection_count": len(unique),
            "projection_index_by_candidate": projection_indices,
            "base_feature_vector_sha256": vector.vector_sha256,
            "prefix_sha256": opportunity.prefix_sha256,
            "rows": rows,
            "elapsed_s": elapsed_s,
            "execution_backend": self.execution_runtime.get("backend"),
            "execution_device": self.execution_runtime.get("device"),
        }

    def forecast_buyer_continuation(self, *, game: Mapping[str, Any], turn_id: str, candidate_set_sha256: str, candidates: Sequence[Mapping[str, object]], register: bool = True, _record_request: bool = True) -> dict[str, object]:
        """Forecast the next seller signal after each immutable nonterminal buyer action."""
        if self.buyer_continuation is None:
            raise ValueError("Persuasion buyer-continuation inference is not activated")
        started = time.monotonic()
        with self._lock:
            result = self.buyer_continuation.predict(game=game, turn_id=turn_id, candidate_set_sha256=candidate_set_sha256, candidates=candidates, register=register)
            elapsed_s = round(time.monotonic() - started, 6)
            if _record_request:
                self.request_count += 1
                self.last_request_at = _utc_now()
                self.last_latency_s = elapsed_s
        return {
            **result,
            "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT,
            "activation_id": self.activation.get("activation_id"),
            "controller": self.activation.get("controller"),
            "elapsed_s": elapsed_s,
            "execution_backend": self.execution_runtime.get("backend"),
            "execution_device": self.execution_runtime.get("device"),
        }

    def warmup(self) -> dict[str, object]:
        """Exercise every frozen family head before the service advertises readiness."""
        common: dict[str, object] = {"your_player": "player_1", "opponent": {"type": "hidden", "name": None}}
        cases: tuple[tuple[dict[str, object], tuple[dict[str, object], ...]], ...] = (
            (
                {**common, "game_id": "__conditional_warmup_bargaining__", "game_family": "bargaining", "phase": "offer", "valid_actions": {"type": "offer", "fields": {}}, "game_state": {"current_player": "player_1", "history": [], "round": 1, "money_to_divide": 100.0, "complete_information": False, "horizon_known": False, "messages_allowed": True}},
                ({"alice_gain": 55.0, "bob_gain": 45.0, "message": "balanced"}, {"alice_gain": 60.0, "bob_gain": 40.0, "message": "firm"}),
            ),
            (
                {**common, "game_id": "__conditional_warmup_negotiation__", "game_family": "negotiation", "phase": "offer", "valid_actions": {"type": "offer", "fields": {}}, "game_state": {"current_player": "player_1", "history": [], "round": 1, "complete_information": False, "horizon_known": False, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0}},
                ({"product_price": 50.0, "message": "balanced"}, {"product_price": 60.0, "message": "firm"}),
            ),
            (
                {**common, "game_id": "__conditional_warmup_persuasion__", "game_family": "persuasion", "phase": "seller_message", "valid_actions": {"type": "seller_message", "fields": {}}, "game_state": {"current_player": "player_1", "player_1_role": "seller", "player_2_role": "buyer", "history": [], "round": 1, "total_rounds": 4, "current_quality": "high", "product_price": 10.0, "p": 0.6, "u": 0.0, "v": 20.0, "seller_message_type": "text", "is_seller_know_cv": True}},
                ({"message": "I recommend buying this product."}, {"message": "This is high quality and worth buying."}),
            ),
        )
        started = time.monotonic()
        for game, actions in cases:
            family = str(game["game_family"])
            candidates = [{"candidate_index": index, "action_sha256": object_sha256(action), "action": action} for index, action in enumerate(actions)]
            features = {"contract": SEQUENCE_SHADOW_FEATURE_CONTRACT, "game_family": family, "phase": game["phase"], "producer_keys": [], "features": {}}
            self.forecast_candidates(game=game, turn_id=f"warmup-{family}", synthetic_features=features, candidate_set_sha256=object_sha256(candidates), candidates=candidates, _record_request=False)
            self.warmup_count += 1
        if self.buyer_continuation is not None:
            buyer_game = {
                **common,
                "game_id": "__conditional_warmup_persuasion_buyer__",
                "game_family": "persuasion",
                "phase": "buyer_decision",
                "valid_actions": {"type": "buyer_decision", "fields": {}},
                "game_state": {"current_player": "player_1", "player_1_role": "buyer", "player_2_role": "seller", "history": [], "round": 1, "total_rounds": 4, "seller_message": "I recommend buying this product.", "seller_message_type": "text", "product_price": 10.0, "p": 0.6, "u": 0.0, "v": 20.0},
            }
            buyer_actions = ({"decision": "yes"}, {"decision": "no"})
            buyer_candidates = [{"candidate_index": index, "action_sha256": object_sha256(action), "action": action} for index, action in enumerate(buyer_actions)]
            self.forecast_buyer_continuation(game=buyer_game, turn_id="warmup-persuasion-buyer", candidate_set_sha256=object_sha256(buyer_candidates), candidates=buyer_candidates, register=False, _record_request=False)
            self.warmup_count += 1
        self.warmup_latency_s = round(time.monotonic() - started, 6)
        return {"status": "complete", "path_count": self.warmup_count, "elapsed_s": self.warmup_latency_s}

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": "ready",
                "contract": LIVE_CONDITIONAL_SERVICE_CONTRACT,
                "activation_id": self.activation.get("activation_id"),
                "controller": self.activation.get("controller"),
                "decision_authority": self.activation.get("decision_authority"),
                "candidate_id": self.release.candidate_id,
                "candidate_manifest_sha256": file_sha256(self.release_dir / "manifest.json"),
                "activation_sha256": file_sha256(self.activation_path),
                "started_at": self.started_at,
                "request_count": self.request_count,
                "failure_count": self.failure_count,
                "warmup_count": self.warmup_count,
                "warmup_latency_s": self.warmup_latency_s,
                "last_request_at": self.last_request_at,
                "last_latency_s": self.last_latency_s,
                "execution_runtime": self.execution_runtime,
                "persuasion_buyer_continuation": self.buyer_continuation.status() if self.buyer_continuation is not None else None,
            }

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            self._respond({"status": "error", "error": "request is empty, oversized, or not newline terminated"})
            return
        adapter: LiveConditionalAdapter = self.server.adapter  # type: ignore[attr-defined]
        try:
            request = json.loads(raw)
            if not isinstance(request, Mapping) or request.get("contract") != LIVE_CONDITIONAL_IPC_CONTRACT:
                raise ValueError("conditional IPC contract mismatch")
            operation = str(request.get("operation") or "")
            if operation == "forecast_candidates":
                game = request.get("game")
                features = request.get("synthetic_features")
                candidates = request.get("candidates")
                if not isinstance(game, Mapping) or not isinstance(features, Mapping) or not isinstance(candidates, list) or not all(isinstance(value, Mapping) for value in candidates):
                    raise ValueError("candidate forecast requires game, synthetic features, and candidate rows")
                response = adapter.forecast_candidates(game=game, turn_id=str(request.get("turn_id") or ""), synthetic_features=features, candidate_set_sha256=str(request.get("candidate_set_sha256") or ""), candidates=candidates)
            elif operation == "forecast_buyer_continuation":
                game = request.get("game")
                candidates = request.get("candidates")
                if not isinstance(game, Mapping) or not isinstance(candidates, list) or not all(isinstance(value, Mapping) for value in candidates):
                    raise ValueError("buyer-continuation forecast requires game and candidate rows")
                response = adapter.forecast_buyer_continuation(game=game, turn_id=str(request.get("turn_id") or ""), candidate_set_sha256=str(request.get("candidate_set_sha256") or ""), candidates=candidates)
            elif operation == "status":
                response = adapter.status()
            else:
                raise ValueError(f"unsupported conditional operation: {operation!r}")
        except Exception as error:
            adapter.record_failure()
            response = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        self._respond(response)
        try:
            self.server.publish_status()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _respond(self, value: Mapping[str, object]) -> None:
        try:
            self.wfile.write(canonical_json(dict(value)).encode("utf-8") + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            return


class _UnixConditionalServer(socketserver.UnixStreamServer):
    allow_reuse_address = False
    request_queue_size = 64

    def __init__(self, socket_path: str, adapter: LiveConditionalAdapter, status_path: Path) -> None:
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
        raise RuntimeError(f"conditional service socket is already active: {path}")
    finally:
        probe.close()


def serve_live_conditional(*, release_dir: Path, activation_path: Path, socket_path: Path, status_path: Path, buyer_continuation_release_dir: Path | None = None, buyer_continuation_registry_path: Path | None = None, device: str | None = None, runtime_manifest_path: Path | None = None, require_transport_evidence: bool = True) -> None:
    """Run the credential-free post-planner conditional service until terminated."""
    resolved_socket = socket_path.resolve()
    _prepare_socket_path(resolved_socket)
    release = None
    execution_runtime = None
    if device == "cpu":
        if runtime_manifest_path is None:
            raise ValueError("CPU conditional service requires a promoted runtime manifest")
        if buyer_continuation_release_dir is None:
            raise ValueError("the promoted CPU conditional service requires its activated buyer-continuation release")
        from .cpu_reference import load_validated_cpu_reference_release

        package_root = Path(__file__).resolve().parents[2]
        release, execution_runtime = load_validated_cpu_reference_release(
            release_dir=release_dir,
            buyer_continuation_release_dir=buyer_continuation_release_dir,
            runtime_manifest_path=runtime_manifest_path,
            live_conditional_path=Path(__file__),
            buyer_continuation_live_path=Path(__file__).with_name("buyer_continuation_live.py"),
            cli_path=Path(__file__).with_name("cli.py"),
            uv_lock_path=package_root / "uv.lock",
            require_transport_evidence=require_transport_evidence,
        )
    elif runtime_manifest_path is not None:
        raise ValueError("a conditional runtime manifest is valid only for the explicit CPU backend")
    adapter = LiveConditionalAdapter(release_dir=release_dir, activation_path=activation_path, buyer_continuation_release_dir=buyer_continuation_release_dir, buyer_continuation_registry_path=buyer_continuation_registry_path, device=device, release=release, execution_runtime=execution_runtime)
    adapter.warmup()
    server = _UnixConditionalServer(str(resolved_socket), adapter, status_path.resolve())
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
