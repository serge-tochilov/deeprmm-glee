"""Credential-free client for the activated post-planner conditional twin."""

from __future__ import annotations

import copy
import json
import math
import socket
from pathlib import Path
from typing import Any, Mapping

from .glee_meta_controller_v2 import NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT, FrozenCandidateSet, negotiation_reject_counteroffer_action, negotiation_reject_counteroffer_frontier
from .glee_policy import normalize_action
from .immutable_pack import object_sha256


LIVE_CONDITIONAL_IPC_CONTRACT = "glee-post-planner-live-conditional-ipc-v1"
LIVE_CONDITIONAL_SERVICE_CONTRACT = "glee-post-planner-live-conditional-service-v1"
BUYER_CONTINUATION_LIVE_CONTRACT = "glee-persuasion-buyer-continuation-live-v1"
BUYER_CONTINUATION_AUTHORITY = "advisory-next-seller-signal-evidence-only"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class GleeConditionalTwinClient:
    """Request one aligned candidate-response surface without loading PyTorch into family workers."""

    def __init__(self, socket_path: Path, *, timeout_s: float = 3.0) -> None:
        if timeout_s <= 0:
            raise ValueError("conditional-twin timeout must be positive")
        self.socket_path = socket_path.resolve()
        self.timeout_s = float(timeout_s)

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "contract": LIVE_CONDITIONAL_IPC_CONTRACT,
            "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT,
            "socket": str(self.socket_path),
            "timeout_s": self.timeout_s,
            "authority": "advisory-candidate-response-evidence-only",
            "forecast_exposed_to_selector": True,
            "final_action_authority": False,
            "negotiation_compound_action_bridge": NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT,
        }

    @staticmethod
    def _project_negotiation_reject_counteroffers(*, game: Mapping[str, Any], candidate_set: FrozenCandidateSet) -> tuple[dict[str, Any], list[dict[str, object]], str, dict[str, object]] | None:
        """Project a fixed rejection plus variable counteroffer into the frozen offer-response twin."""
        if not negotiation_reject_counteroffer_frontier(game):
            return None
        if candidate_set.family != "negotiation" or candidate_set.action_type != "decision" or not all(negotiation_reject_counteroffer_action(candidate.action) for candidate in candidate_set.candidates):
            raise ValueError("Negotiation bridge requires only reject-plus-counteroffer candidates")
        projected_game = copy.deepcopy(dict(game))
        state = projected_game.get("game_state")
        if not isinstance(state, dict):
            raise ValueError("Negotiation bridge has no mutable game state")
        history = state.get("history")
        last_offer = state.get("last_offer")
        round_number = state.get("round")
        our_player = str(projected_game.get("your_player") or state.get("current_player") or "")
        if not isinstance(history, list) or not isinstance(last_offer, Mapping) or isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1 or not our_player:
            raise ValueError("Negotiation bridge lacks a complete current offer context")
        price = last_offer.get("price")
        proposer = str(last_offer.get("from_player") or "")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) < 0.0 or not proposer:
            raise ValueError("Negotiation bridge current offer is malformed")
        offer_round = last_offer.get("round")
        if isinstance(offer_round, bool) or not isinstance(offer_round, int) or offer_round < 1:
            offer_round = round_number
        fixed_offer = {"round": offer_round, "from_player": proposer, "price": float(price)}
        message = last_offer.get("message")
        if isinstance(message, str) and message:
            fixed_offer["message"] = message
        history.append({"round": offer_round, "offer": fixed_offer, "decided_by": our_player, "decision": "RejectOffer"})
        state["round"] = round_number + 1
        state["phase"] = "offer"
        state["current_player"] = our_player
        state.pop("last_offer", None)
        projected_game["phase"] = "offer"
        original_fields = game.get("valid_actions", {}).get("fields") if isinstance(game.get("valid_actions"), Mapping) else None
        projected_fields: dict[str, object] = {"product_price": original_fields.get("product_price", "number") if isinstance(original_fields, Mapping) else "number"}
        if isinstance(original_fields, Mapping) and "message" in original_fields:
            projected_fields["message"] = original_fields["message"]
        projected_game["valid_actions"] = {"type": "offer", "fields": projected_fields}
        projected_candidates: list[dict[str, object]] = []
        for candidate in candidate_set.candidates:
            source_action = dict(candidate.action)
            projected_action = normalize_action(projected_game, {key: value for key, value in source_action.items() if key in {"product_price", "message"}})
            projected_candidates.append({"candidate_index": candidate.index, "action_sha256": object_sha256(projected_action), "action": projected_action})
        identity = {
            "contract": NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT,
            "source_game_sha256": object_sha256(game),
            "source_candidate_set_sha256": candidate_set.candidate_set_sha256,
            "projected_game_sha256": object_sha256(projected_game),
            "projected_candidates": projected_candidates,
        }
        projected_candidate_set_sha256 = object_sha256(identity)
        receipt = {
            "contract": NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT,
            "status": "projected",
            "source_phase": str(game.get("phase") or "decision"),
            "projected_phase": "offer",
            "fixed_event": "self-reject-current-opponent-offer",
            "variable_event": "self-counteroffer-proposal",
            "target_event": "direct-opponent-response",
            "causal_bridge_event_count": 2,
            "source_game_sha256": identity["source_game_sha256"],
            "source_candidate_set_sha256": candidate_set.candidate_set_sha256,
            "projected_candidate_set_sha256": projected_candidate_set_sha256,
            "projected_game_sha256": identity["projected_game_sha256"],
            "candidate_identity_map": [
                {
                    "candidate_index": source.index,
                    "source_action_sha256": source.action_sha256,
                    "projected_action_sha256": projected["action_sha256"],
                }
                for source, projected in zip(candidate_set.candidates, projected_candidates, strict=True)
            ],
        }
        return projected_game, projected_candidates, projected_candidate_set_sha256, receipt

    def _request(self, operation: str, **values: object) -> dict[str, object]:
        request = {"contract": LIVE_CONDITIONAL_IPC_CONTRACT, "operation": operation, **values}
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("conditional-twin request exceeds the IPC size limit")
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
                    raise ValueError("conditional-twin response exceeds the IPC size limit")
                if b"\n" in chunk:
                    break
        finally:
            client.close()
        response = json.loads(b"".join(chunks).partition(b"\n")[0])
        if not isinstance(response, dict):
            raise ValueError("conditional-twin response is not an object")
        if response.get("status") == "error":
            raise RuntimeError(f"conditional-twin sidecar failed: {response.get('error_type')}: {response.get('error')}")
        return response

    def forecast_candidates(self, *, game: Mapping[str, Any], turn_id: str, synthetic_features: Mapping[str, object], candidate_set: FrozenCandidateSet) -> dict[str, object]:
        source_candidates = [
            {"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "action": dict(candidate.action)}
            for candidate in candidate_set.candidates
        ]
        projection = self._project_negotiation_reject_counteroffers(game=game, candidate_set=candidate_set)
        request_game: Mapping[str, Any] = game
        request_candidates = source_candidates
        request_candidate_set_sha256 = candidate_set.candidate_set_sha256
        bridge_receipt: dict[str, object] | None = None
        if projection is not None:
            request_game, request_candidates, request_candidate_set_sha256, bridge_receipt = projection
        response = self._request(
            "forecast_candidates",
            game=dict(request_game),
            turn_id=turn_id,
            synthetic_features=dict(synthetic_features),
            candidate_set_sha256=request_candidate_set_sha256,
            candidates=request_candidates,
        )
        if response.get("contract") != LIVE_CONDITIONAL_IPC_CONTRACT or response.get("service_contract") != LIVE_CONDITIONAL_SERVICE_CONTRACT:
            raise ValueError("conditional-twin response has the wrong contract")
        if response.get("decision_authority") != "advisory-candidate-response-evidence-only" or response.get("candidate_set_sha256") != request_candidate_set_sha256:
            raise ValueError("conditional-twin response exceeded its authority or changed the candidate set")
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
            raise ValueError("conditional-twin response does not cover the candidate set")
        for candidate, request_candidate, row in zip(candidate_set.candidates, request_candidates, rows, strict=True):
            if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != request_candidate["action_sha256"] or not isinstance(row.get("forecast"), Mapping):
                raise ValueError("conditional-twin response lost candidate alignment")
        if bridge_receipt is None:
            return response
        remapped = copy.deepcopy(response)
        remapped["candidate_set_sha256"] = candidate_set.candidate_set_sha256
        remapped["rows"] = [{**dict(row), "candidate_index": candidate.index, "action_sha256": candidate.action_sha256} for candidate, row in zip(candidate_set.candidates, rows, strict=True)]
        remapped["negotiation_reject_counteroffer_bridge"] = bridge_receipt
        return remapped

    def forecast_buyer_continuation(self, *, game: Mapping[str, Any], turn_id: str, candidate_set: FrozenCandidateSet) -> dict[str, object]:
        candidates = [
            {"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "action": dict(candidate.action)}
            for candidate in candidate_set.candidates
        ]
        response = self._request(
            "forecast_buyer_continuation",
            game=dict(game),
            turn_id=turn_id,
            candidate_set_sha256=candidate_set.candidate_set_sha256,
            candidates=candidates,
        )
        if response.get("contract") != BUYER_CONTINUATION_LIVE_CONTRACT or response.get("service_contract") != LIVE_CONDITIONAL_SERVICE_CONTRACT:
            raise ValueError("buyer-continuation response has the wrong contract")
        if response.get("authority") != BUYER_CONTINUATION_AUTHORITY or response.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
            raise ValueError("buyer-continuation response exceeded its authority or changed the candidate set")
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) != len(candidate_set.candidates):
            raise ValueError("buyer-continuation response does not cover the candidate set")
        for candidate, row in zip(candidate_set.candidates, rows, strict=True):
            if not isinstance(row, Mapping) or row.get("candidate_index") != candidate.index or row.get("action_sha256") != candidate.action_sha256 or not isinstance(row.get("forecast"), Mapping):
                raise ValueError("buyer-continuation response lost candidate alignment")
        return response

    def status(self) -> dict[str, object]:
        response = self._request("status")
        if response.get("contract") != LIVE_CONDITIONAL_SERVICE_CONTRACT:
            raise ValueError("conditional-twin service status has the wrong contract")
        return response
