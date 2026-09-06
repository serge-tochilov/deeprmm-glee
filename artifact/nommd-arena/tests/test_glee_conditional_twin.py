from __future__ import annotations

import copy
import json
import socket
import threading

from nommd_arena.glee_conditional_twin import BUYER_CONTINUATION_AUTHORITY, BUYER_CONTINUATION_LIVE_CONTRACT, GleeConditionalTwinClient, LIVE_CONDITIONAL_IPC_CONTRACT, LIVE_CONDITIONAL_SERVICE_CONTRACT
from nommd_arena.glee_meta_controller_v2 import NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT
from nommd_arena.glee_meta_controller_v2 import freeze_fixed_candidates, freeze_planner_candidates
from nommd_arena.glee_nommd import nommd_candidate_plan_model
from nommd_arena.glee_policy import action_model


def _candidate_set():
    game = {"game_id": "conditional-client", "game_family": "negotiation", "your_player": "player_1", "phase": "offer", "valid_actions": {"type": "offer", "fields": {"product_price": "number", "message": "string"}}, "game_state": {"current_player": "player_1", "history": [], "round": 1, "complete_information": False, "horizon_known": False, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0}}
    model = nommd_candidate_plan_model(action_model(game))
    plan = model.model_validate({"candidates": [{"action": {"product_price": 50.0, "message": "first"}, "purpose": "first"}, {"action": {"product_price": 60.0, "message": "second"}, "purpose": "second"}]})
    return game, freeze_planner_candidates(game=game, parsed=plan)


def test_conditional_twin_client_preserves_candidate_hash_alignment(tmp_path) -> None:
    game, candidate_set = _candidate_set()
    path = tmp_path / "conditional.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    captured: dict[str, object] = {}

    def serve() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while not raw.endswith(b"\n"):
                raw += connection.recv(65_536)
            captured.update(json.loads(raw))
            rows = [{"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.45, 0.05]}} for candidate in candidate_set.candidates]
            response = {"status": "predicted", "contract": LIVE_CONDITIONAL_IPC_CONTRACT, "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT, "decision_authority": "advisory-candidate-response-evidence-only", "candidate_set_sha256": candidate_set.candidate_set_sha256, "rows": rows}
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve)
    thread.start()
    client = GleeConditionalTwinClient(path, timeout_s=1.0)
    response = client.forecast_candidates(game=game, turn_id="turn-1", synthetic_features={"contract": "features"}, candidate_set=candidate_set)
    thread.join(timeout=2.0)
    server.close()
    assert response["candidate_set_sha256"] == candidate_set.candidate_set_sha256
    assert captured["operation"] == "forecast_candidates"
    assert [row["action_sha256"] for row in captured["candidates"]] == [candidate.action_sha256 for candidate in candidate_set.candidates]


def test_conditional_twin_client_projects_compound_negotiation_action_and_remaps_identity(tmp_path) -> None:
    game = {
        "game_id": "conditional-counteroffer-client",
        "game_family": "negotiation",
        "your_player": "player_1",
        "phase": "decision",
        "valid_actions": {"type": "decision", "fields": {"decision": "choice", "product_price": "number", "message": "string"}},
        "game_state": {
            "current_player": "player_1",
            "history": [{"round": 1, "offer": {"round": 1, "from_player": "player_1", "price": 70.0, "message": "opening"}, "decided_by": "player_2", "decision": "RejectOffer"}],
            "last_offer": {"round": 2, "from_player": "player_2", "price": 100.0, "message": "counter"},
            "round": 2,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 10,
            "messages_allowed": True,
            "player_1_role": "buyer",
            "player_2_role": "seller",
            "player_1_value": 80.0,
        },
    }
    original_game = copy.deepcopy(game)
    candidate_set = freeze_fixed_candidates(
        game=game,
        candidate_specs=(
            {"action": {"decision": "RejectOffer", "product_price": 60.0, "message": "first"}, "purpose": "first"},
            {"action": {"decision": "RejectOffer", "product_price": 70.0, "message": "second"}, "purpose": "second"},
        ),
    )
    path = tmp_path / "conditional-counteroffer.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    captured: dict[str, object] = {}

    def serve() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while not raw.endswith(b"\n"):
                raw += connection.recv(65_536)
            captured.update(json.loads(raw))
            rows = [{"candidate_index": row["candidate_index"], "action_sha256": row["action_sha256"], "forecast": {"authority": "prospective-shadow-only", "labels": ["accept", "reject", "walkaway"], "response_probabilities": [0.5, 0.45, 0.05]}} for row in captured["candidates"]]
            response = {"status": "predicted", "contract": LIVE_CONDITIONAL_IPC_CONTRACT, "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT, "decision_authority": "advisory-candidate-response-evidence-only", "candidate_set_sha256": captured["candidate_set_sha256"], "rows": rows}
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve)
    thread.start()
    client = GleeConditionalTwinClient(path, timeout_s=1.0)
    response = client.forecast_candidates(game=game, turn_id="counteroffer-turn-1", synthetic_features={"contract": "features"}, candidate_set=candidate_set)
    thread.join(timeout=2.0)
    server.close()
    projected_game = captured["game"]
    projected_state = projected_game["game_state"]
    assert game == original_game
    assert projected_game["phase"] == "offer"
    assert projected_game["valid_actions"]["type"] == "offer"
    assert projected_state["round"] == 3
    assert "last_offer" not in projected_state
    assert projected_state["history"][-1] == {"round": 2, "offer": {"round": 2, "from_player": "player_2", "price": 100.0, "message": "counter"}, "decided_by": "player_1", "decision": "RejectOffer"}
    assert [row["action"] for row in captured["candidates"]] == [{"product_price": 60.0, "message": "first"}, {"product_price": 70.0, "message": "second"}]
    assert captured["candidate_set_sha256"] != candidate_set.candidate_set_sha256
    assert response["candidate_set_sha256"] == candidate_set.candidate_set_sha256
    assert [row["action_sha256"] for row in response["rows"]] == [candidate.action_sha256 for candidate in candidate_set.candidates]
    bridge = response["negotiation_reject_counteroffer_bridge"]
    assert bridge["contract"] == NEGOTIATION_REJECT_COUNTEROFFER_BRIDGE_CONTRACT
    assert bridge["causal_bridge_event_count"] == 2
    assert bridge["projected_candidate_set_sha256"] == captured["candidate_set_sha256"]
    assert [row["source_action_sha256"] for row in bridge["candidate_identity_map"]] == [candidate.action_sha256 for candidate in candidate_set.candidates]
    assert [row["projected_action_sha256"] for row in bridge["candidate_identity_map"]] == [row["action_sha256"] for row in captured["candidates"]]


def test_conditional_twin_client_preserves_buyer_continuation_alignment_and_authority(tmp_path) -> None:
    game = {
        "game_id": "buyer-continuation-client",
        "game_family": "persuasion",
        "your_player": "player_2",
        "phase": "buyer_decision",
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}},
        "game_state": {"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "history": [], "round": 2, "total_rounds": 10},
    }
    candidate_set = freeze_fixed_candidates(game=game, candidate_specs=({"action": {"decision": "yes"}, "purpose": "buy"}, {"action": {"decision": "no"}, "purpose": "pass"}))
    path = tmp_path / "buyer-continuation.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    captured: dict[str, object] = {}

    def serve() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while not raw.endswith(b"\n"):
                raw += connection.recv(65_536)
            captured.update(json.loads(raw))
            rows = [{"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"labels": ["signal_positive", "signal_negative", "signal_unknown"], "response_probabilities": [0.6, 0.3, 0.1]}} for candidate in candidate_set.candidates]
            response = {"status": "predicted", "contract": BUYER_CONTINUATION_LIVE_CONTRACT, "service_contract": LIVE_CONDITIONAL_SERVICE_CONTRACT, "authority": BUYER_CONTINUATION_AUTHORITY, "candidate_set_sha256": candidate_set.candidate_set_sha256, "rows": rows}
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve)
    thread.start()
    client = GleeConditionalTwinClient(path, timeout_s=1.0)
    response = client.forecast_buyer_continuation(game=game, turn_id="buyer-turn-1", candidate_set=candidate_set)
    thread.join(timeout=2.0)
    server.close()
    assert response["authority"] == BUYER_CONTINUATION_AUTHORITY
    assert captured["operation"] == "forecast_buyer_continuation"
    assert [row["action"]["decision"] for row in captured["candidates"]] == ["yes", "no"]
