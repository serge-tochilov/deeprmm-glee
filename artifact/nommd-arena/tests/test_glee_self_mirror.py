from __future__ import annotations

import json
import socket
import threading

from nommd_arena.glee_meta_controller_v2 import freeze_fixed_candidates
from nommd_arena.glee_self_mirror import GleePublicSelfMirrorClient, SELF_MIRROR_AUTHORITY, SELF_MIRROR_LIVE_SERVICE_CONTRACT, SELF_MIRROR_SURFACE_CONTRACT
from nommd_arena.immutable_pack import object_sha256


def _game_and_candidates():
    game = {
        "game_id": "self-mirror-client",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}},
        "game_state": {"current_player": "player_1", "history": [], "round": 1, "money_to_divide": 100.0, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "delta_1": 0.9, "delta_2": 0.8},
    }
    candidates = freeze_fixed_candidates(
        game=game,
        candidate_specs=(
            {"action": {"alice_gain": 60.0, "bob_gain": 40.0, "message": "first"}, "purpose": "first"},
            {"action": {"alice_gain": 55.0, "bob_gain": 45.0, "message": "second"}, "purpose": "second"},
        ),
    )
    return game, candidates


def test_public_self_mirror_client_preserves_candidate_alignment_and_records_selection(tmp_path) -> None:
    game, candidate_set = _game_and_candidates()
    path = tmp_path / "self-mirror.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(2)
    captured: list[dict[str, object]] = []

    def serve() -> None:
        for call in range(2):
            connection, _address = server.accept()
            with connection:
                raw = b""
                while not raw.endswith(b"\n"):
                    raw += connection.recv(65_536)
                request = json.loads(raw)
                captured.append(request)
                if call == 0:
                    rows = []
                    for index, candidate in enumerate(candidate_set.candidates):
                        rows.append({"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"ensemble_log_expectedness": -0.2 - index, "relative_expectedness": 0.75 if index == 0 else 0.25, "expectedness_rank": index + 1, "expectedness_percentile": 1.0 - index, "component_log_expectedness": {"1729": -0.1 - index, "2718": -0.3 - index}, "component_log_score_stddev": 0.1, "component_top_choice_disagreement": False, "population_prediction": True, "account_prediction": None, "message_wording_scored": False}})
                    response = {"contract": SELF_MIRROR_SURFACE_CONTRACT, "service_contract": SELF_MIRROR_LIVE_SERVICE_CONTRACT, "authority": SELF_MIRROR_AUTHORITY, "release_id": "self-v1", "turn_id": "turn-1", "game_id": game["game_id"], "family": "bargaining", "candidate_set_sha256": candidate_set.candidate_set_sha256, "rows": rows, "forecast_sha256": "a" * 64}
                else:
                    submitted = request["submitted_action"]
                    response = {"status": "recorded", "selection": {"turn_id": "turn-1", "submitted_action_sha256": object_sha256(submitted)}}
                connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve)
    thread.start()
    client = GleePublicSelfMirrorClient(path, timeout_s=1.0)
    surface = client.forecast_candidates(game=game, turn_id="turn-1", candidate_set=candidate_set)
    receipt = client.record_selection(turn_id="turn-1", forecast_sha256=surface["forecast_sha256"], selected_candidate_index=1, selected_action_sha256=candidate_set.candidates[1].action_sha256, submitted_action=candidate_set.candidates[1].action)
    thread.join(timeout=2.0)
    server.close()
    assert surface["rows"][1]["action_sha256"] == candidate_set.candidates[1].action_sha256
    assert receipt["status"] == "recorded"
    assert captured[0]["operation"] == "forecast"
    assert "synthetic_features" not in captured[0]
    assert captured[1]["operation"] == "record_selection"


def test_public_self_mirror_client_projects_only_mixed_negotiation_counteroffers(tmp_path) -> None:
    game = {
        "game_id": "self-mirror-mixed-negotiation",
        "game_family": "negotiation",
        "your_player": "player_1",
        "phase": "decision",
        "valid_actions": {"type": "decision", "fields": {"decision": "choice", "product_price": "number", "message": "string"}},
        "game_state": {
            "current_player": "player_1",
            "history": [],
            "last_offer": {"round": 2, "from_player": "player_2", "price": 100.0, "message": "Current offer."},
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
    candidate_set = freeze_fixed_candidates(
        game=game,
        candidate_specs=(
            {"action": {"decision": "AcceptOffer"}, "purpose": "accept"},
            {"action": {"decision": "WalkAway"}, "purpose": "walk away"},
            {"action": {"decision": "RejectOffer", "product_price": 70.0, "message": "First counter."}, "purpose": "first counter"},
            {"action": {"decision": "RejectOffer", "product_price": 75.0, "message": "Second counter."}, "purpose": "second counter"},
        ),
    )
    path = tmp_path / "mixed-self-mirror.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    captured: list[dict[str, object]] = []

    def serve() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while not raw.endswith(b"\n"):
                raw += connection.recv(65_536)
            request = json.loads(raw)
            captured.append(request)
            rows = [
                {
                    "candidate_index": candidate.index,
                    "action_sha256": candidate.action_sha256,
                    "forecast": {
                        "ensemble_log_expectedness": -0.1 - candidate.index,
                        "relative_expectedness": 0.25,
                        "expectedness_rank": candidate.index + 1,
                        "expectedness_percentile": 1.0 - candidate.index / 3.0,
                        "component_log_expectedness": {"1729": -0.1 - candidate.index, "2718": -0.1 - candidate.index},
                        "component_log_score_stddev": 0.0,
                        "component_top_choice_disagreement": False,
                        "population_prediction": True,
                        "account_prediction": None,
                        "message_wording_scored": False,
                    },
                }
                for candidate in candidate_set.candidates
            ]
            response = {
                "contract": SELF_MIRROR_SURFACE_CONTRACT,
                "service_contract": SELF_MIRROR_LIVE_SERVICE_CONTRACT,
                "authority": SELF_MIRROR_AUTHORITY,
                "release_id": "self-v1",
                "turn_id": "mixed-turn",
                "game_id": game["game_id"],
                "family": "negotiation",
                "candidate_set_sha256": candidate_set.candidate_set_sha256,
                "rows": rows,
                "forecast_sha256": "b" * 64,
            }
            connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve)
    thread.start()
    surface = GleePublicSelfMirrorClient(path, timeout_s=1.0).forecast_candidates(game=game, turn_id="mixed-turn", candidate_set=candidate_set)
    thread.join(timeout=2.0)
    server.close()
    request = captured[0]
    assert [row["candidate_index"] for row in request["candidates"]] == [0, 1, 2, 3]
    assert [row["candidate_index"] for row in request["projected_candidates"]] == [2, 3]
    assert [row["action"]["product_price"] for row in request["projected_candidates"]] == [70.0, 75.0]
    bridge = request["negotiation_reject_counteroffer_bridge"]
    assert bridge["status"] == "projected-subset"
    assert bridge["source_candidate_count"] == 4
    assert bridge["projected_candidate_count"] == 2
    assert bridge["terminal_candidate_count"] == 2
    assert surface["negotiation_reject_counteroffer_bridge"]["candidate_identity_map"] == bridge["candidate_identity_map"]
