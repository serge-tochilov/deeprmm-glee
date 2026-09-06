"""Credential-free client for the public-information self-mirror sidecar."""

from __future__ import annotations

import copy
import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .glee_conditional_twin import GleeConditionalTwinClient
from .glee_meta_controller_v2 import FrozenCandidateSet, negotiation_reject_counteroffer_action
from .immutable_pack import object_sha256


SELF_MIRROR_LIVE_IPC_CONTRACT = "glee-public-self-mirror-live-ipc-v1"
SELF_MIRROR_LIVE_SERVICE_CONTRACT = "glee-public-self-mirror-live-service-v1"
SELF_MIRROR_SURFACE_CONTRACT = "glee-public-self-mirror-candidate-surface-v1"
SELF_MIRROR_AUTHORITY = "bounded-public-expectedness-selector-evidence-only"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class GleePublicSelfMirrorClient:
    """Score a frozen candidate set without exposing private or planner state to the self-mirror."""

    def __init__(self, socket_path: Path, *, timeout_s: float = 3.0) -> None:
        if timeout_s <= 0:
            raise ValueError("public self-mirror timeout must be positive")
        self.socket_path = socket_path.resolve()
        self.timeout_s = float(timeout_s)

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "contract": SELF_MIRROR_LIVE_IPC_CONTRACT,
            "service_contract": SELF_MIRROR_LIVE_SERVICE_CONTRACT,
            "surface_contract": SELF_MIRROR_SURFACE_CONTRACT,
            "socket": str(self.socket_path),
            "timeout_s": self.timeout_s,
            "authority": SELF_MIRROR_AUTHORITY,
            "planner_visibility": False,
            "candidate_generation_authority": False,
            "hard_control_authority": False,
        }

    def _request(self, operation: str, **values: object) -> dict[str, object]:
        request = {"contract": SELF_MIRROR_LIVE_IPC_CONTRACT, "operation": operation, **values}
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("public self-mirror request exceeds the IPC size limit")
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
                    raise ValueError("public self-mirror response exceeds the IPC size limit")
                if b"\n" in chunk:
                    break
        finally:
            client.close()
        response = json.loads(b"".join(chunks).partition(b"\n")[0])
        if not isinstance(response, dict):
            raise ValueError("public self-mirror response is not an object")
        if response.get("status") == "error":
            raise RuntimeError(f"public self-mirror sidecar failed: {response.get('error_type')}: {response.get('error')}")
        return response

    @staticmethod
    def _project_negotiation_counteroffer_subset(*, game: Mapping[str, Any], candidate_set: FrozenCandidateSet) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, object]] | None:
        """Project only compound counteroffers while preserving terminal candidates on a mixed decision frontier."""
        counters = tuple(candidate for candidate in candidate_set.candidates if negotiation_reject_counteroffer_action(candidate.action))
        if not counters:
            return None
        if len(counters) == len(candidate_set.candidates):
            projection = GleeConditionalTwinClient._project_negotiation_reject_counteroffers(game=game, candidate_set=candidate_set)
            if projection is None:
                return None
            projected_game, projected_candidates, _projected_set_sha256, bridge_receipt = projection
            return projected_game, projected_candidates, bridge_receipt
        reindexed = tuple(replace(candidate, index=index) for index, candidate in enumerate(counters))
        subset_identity = {"family": candidate_set.family, "action_type": candidate_set.action_type, "candidates": [candidate.receipt() for candidate in reindexed]}
        subset = FrozenCandidateSet(family=candidate_set.family, action_type=candidate_set.action_type, candidates=reindexed, candidate_set_sha256=object_sha256(subset_identity))
        projection = GleeConditionalTwinClient._project_negotiation_reject_counteroffers(game=game, candidate_set=subset)
        if projection is None:
            return None
        projected_game, projected_candidates, _projected_set_sha256, raw_receipt = projection
        remapped_candidates: list[dict[str, object]] = []
        identity_map: list[dict[str, object]] = []
        for source, projected in zip(counters, projected_candidates, strict=True):
            remapped = copy.deepcopy(projected)
            remapped["candidate_index"] = source.index
            remapped_candidates.append(remapped)
            identity_map.append(
                {
                    "candidate_index": source.index,
                    "source_action_sha256": source.action_sha256,
                    "projected_action_sha256": remapped["action_sha256"],
                }
            )
        bridge_receipt = copy.deepcopy(raw_receipt)
        bridge_receipt.update(
            {
                "status": "projected-subset",
                "source_candidate_set_sha256": candidate_set.candidate_set_sha256,
                "counteroffer_subset_candidate_set_sha256": subset.candidate_set_sha256,
                "projected_candidate_set_sha256": object_sha256(
                    {
                        "source_candidate_set_sha256": candidate_set.candidate_set_sha256,
                        "projected_candidates": remapped_candidates,
                    }
                ),
                "candidate_identity_map": identity_map,
                "source_candidate_count": len(candidate_set.candidates),
                "projected_candidate_count": len(counters),
                "terminal_candidate_count": len(candidate_set.candidates) - len(counters),
            }
        )
        return projected_game, remapped_candidates, bridge_receipt

    def forecast_candidates(self, *, game: Mapping[str, Any], turn_id: str, candidate_set: FrozenCandidateSet) -> dict[str, object]:
        candidates = [{"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "action": dict(candidate.action)} for candidate in candidate_set.candidates]
        projection = self._project_negotiation_counteroffer_subset(game=game, candidate_set=candidate_set)
        extras: dict[str, object] = {}
        if projection is not None:
            projected_game, projected_candidates, bridge_receipt = projection
            extras = {"projected_game": projected_game, "projected_candidates": projected_candidates, "negotiation_reject_counteroffer_bridge": bridge_receipt}
        response = self._request("forecast", game=dict(game), turn_id=turn_id, candidate_set_sha256=candidate_set.candidate_set_sha256, candidates=candidates, **extras)
        if response.get("contract") != SELF_MIRROR_SURFACE_CONTRACT or response.get("service_contract") != SELF_MIRROR_LIVE_SERVICE_CONTRACT or response.get("authority") != SELF_MIRROR_AUTHORITY:
            raise ValueError("public self-mirror response has the wrong contract or authority")
        if response.get("candidate_set_sha256") != candidate_set.candidate_set_sha256 or response.get("turn_id") != turn_id:
            raise ValueError("public self-mirror response changed the turn or candidate set")
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
            raise ValueError("public self-mirror response does not cover the candidate set")
        for candidate, row in zip(candidate_set.candidates, rows, strict=True):
            if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(row.get("forecast"), Mapping):
                raise ValueError("public self-mirror response lost immutable candidate alignment")
        forecast_sha256 = response.get("forecast_sha256")
        if not isinstance(forecast_sha256, str) or len(forecast_sha256) != 64:
            raise ValueError("public self-mirror response lacks its prospective forecast identity")
        if projection is None:
            return response
        surface = copy.deepcopy(response)
        surface["negotiation_reject_counteroffer_bridge"] = bridge_receipt
        return surface

    def record_selection(self, *, turn_id: str, forecast_sha256: str, selected_candidate_index: int | None, selected_action_sha256: str | None, submitted_action: Mapping[str, object]) -> dict[str, object]:
        response = self._request(
            "record_selection",
            turn_id=turn_id,
            forecast_sha256=forecast_sha256,
            selected_candidate_index=selected_candidate_index,
            selected_action_sha256=selected_action_sha256,
            submitted_action=dict(submitted_action),
        )
        selection = response.get("selection")
        if not isinstance(selection, Mapping) or selection.get("turn_id") != turn_id or selection.get("submitted_action_sha256") != object_sha256(dict(submitted_action)):
            raise ValueError("public self-mirror selection receipt changed the submitted action")
        return response

    def status(self) -> dict[str, object]:
        response = self._request("status")
        if response.get("contract") != SELF_MIRROR_LIVE_SERVICE_CONTRACT or response.get("authority") != SELF_MIRROR_AUTHORITY:
            raise ValueError("public self-mirror service status has the wrong contract or authority")
        return response
