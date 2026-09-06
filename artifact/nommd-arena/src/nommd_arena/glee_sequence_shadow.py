"""Credential-free client and feature boundary for the pre-Terra sequence shadow."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Mapping


SEQUENCE_SHADOW_IPC_CONTRACT = "glee-sequence-live-shadow-ipc-v1"
SEQUENCE_SHADOW_FEATURE_CONTRACT = "glee-terra-synthetic-feature-bundle-v1"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


SYNTHETIC_FEATURE_KEYS = (
    "bargaining_decision_control",
    "canonical_bargaining_facts",
    "game_semantics",
    "analytic_bargaining_reference",
    "bargaining_opponent_model_v2",
    "opponent_statistical_decision_forecast",
    "opponent_account_hypothesis",
    "rating_v3_advisory",
    "negotiation_opponent_model_v2",
    "persuasion_decision_facts",
    "bargaining_analytic_authority",
    "message_realization_profile",
    "opponent_message_policy_signal",
)


def terra_synthetic_feature_bundle(worker_payload: Mapping[str, object]) -> dict[str, object]:
    """Select compact precomputed reasoning features from the exact pre-Terra payload."""
    features = {key: worker_payload[key] for key in SYNTHETIC_FEATURE_KEYS if key in worker_payload}
    return {
        "contract": SEQUENCE_SHADOW_FEATURE_CONTRACT,
        "frontier": "exact-worker-payload-before-terra",
        "game_family": worker_payload.get("game_family"),
        "phase": worker_payload.get("phase"),
        "producer_keys": sorted(features),
        "features": features,
    }


class GleeSequenceShadowClient:
    """Make bounded Unix-socket requests without loading PyTorch into a family process."""

    def __init__(self, socket_path: Path, *, timeout_s: float = 1.0) -> None:
        if timeout_s <= 0:
            raise ValueError("sequence-shadow timeout must be positive")
        self.socket_path = socket_path.resolve()
        self.timeout_s = float(timeout_s)

    @property
    def receipt(self) -> dict[str, object]:
        return {"contract": SEQUENCE_SHADOW_IPC_CONTRACT, "socket": str(self.socket_path), "timeout_s": self.timeout_s, "authority": "prospective-shadow-only", "forecast_exposed_to_worker": False}

    def _request(self, operation: str, **values: object) -> dict[str, object]:
        request = {"contract": SEQUENCE_SHADOW_IPC_CONTRACT, "operation": operation, **values}
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("sequence-shadow request exceeds the IPC size limit")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.timeout_s)
        try:
            client.connect(str(self.socket_path))
            client.sendall(payload)
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = client.recv(65_536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_MESSAGE_BYTES:
                    raise ValueError("sequence-shadow response exceeds the IPC size limit")
                if b"\n" in chunk:
                    break
        finally:
            client.close()
        response = json.loads(b"".join(chunks).partition(b"\n")[0])
        if not isinstance(response, dict):
            raise ValueError("sequence-shadow response is not an object")
        if response.get("status") == "error":
            raise RuntimeError(f"sequence-shadow sidecar failed: {response.get('error_type')}: {response.get('error')}")
        return response

    def forecast(self, *, game: Mapping[str, Any], turn_id: str, synthetic_features: Mapping[str, object], observed_at: str) -> dict[str, object]:
        return self._request("forecast", game=dict(game), turn_id=turn_id, synthetic_features=dict(synthetic_features), observed_at=observed_at)

    def observe(self, *, game: Mapping[str, Any], observed_at: str, source: str) -> dict[str, object]:
        return self._request("observe", game=dict(game), observed_at=observed_at, source=source)

    def status(self) -> dict[str, object]:
        return self._request("status")
