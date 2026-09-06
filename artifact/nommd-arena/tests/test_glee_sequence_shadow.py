from __future__ import annotations

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

from nommd_arena.glee_parallel import ParallelGleeRun
from nommd_arena.glee_sequence_shadow import GleeSequenceShadowClient, SEQUENCE_SHADOW_FEATURE_CONTRACT, terra_synthetic_feature_bundle
from nommd_arena.glee_tactics import GlobalTacticLedger


def test_terra_feature_bundle_selects_synthetic_context_without_raw_prompt() -> None:
    payload = {
        "game_family": "bargaining",
        "phase": "offer",
        "official_prompt": "large official prose",
        "visible_game_state": {"history": [1, 2]},
        "tetrad_memory": {"private": "memory"},
        "canonical_bargaining_facts": {"round": 3},
        "message_realization_profile": {"profile_id": "anti-aligned"},
        "opponent_statistical_decision_forecast": {"accept": 0.7},
        "rating_v3_advisory": {"delta": 1.2},
    }
    bundle = terra_synthetic_feature_bundle(payload)
    assert bundle["contract"] == SEQUENCE_SHADOW_FEATURE_CONTRACT
    assert bundle["producer_keys"] == ["canonical_bargaining_facts", "message_realization_profile", "opponent_statistical_decision_forecast", "rating_v3_advisory"]
    assert "official_prompt" not in bundle["features"]
    assert "visible_game_state" not in bundle["features"]
    assert "tetrad_memory" not in bundle["features"]


def test_sequence_shadow_client_uses_opaque_newline_ipc(tmp_path) -> None:
    path = tmp_path / "shadow.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    captured = {}

    def serve() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while not raw.endswith(b"\n"):
                raw += connection.recv(65_536)
            captured.update(json.loads(raw))
            connection.sendall(b'{"status":"registered","prediction_sha256":"abc"}\n')

    thread = threading.Thread(target=serve)
    thread.start()
    client = GleeSequenceShadowClient(path, timeout_s=1.0)
    response = client.forecast(game={"game_id": "g"}, turn_id="t", synthetic_features={"x": 1}, observed_at="now")
    thread.join(timeout=2.0)
    server.close()
    assert response == {"status": "registered", "prediction_sha256": "abc"}
    assert captured["operation"] == "forecast"
    assert captured["turn_id"] == "t"


def test_supervisor_registers_exact_feature_snapshot_before_worker_solve(tmp_path) -> None:
    order: list[str] = []
    captured: dict[str, object] = {}

    class Client:
        @staticmethod
        def stats() -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 1, "scores": {}}

    class Shadow:
        receipt = {"contract": "glee-sequence-live-shadow-ipc-v1", "authority": "prospective-shadow-only"}

        @staticmethod
        def observe(**_values: object) -> dict[str, object]:
            order.append("observe")
            return {"status": "observed", "matured": 0, "mismatches": []}

        @staticmethod
        def forecast(**values: object) -> dict[str, object]:
            order.append("shadow")
            captured.update(values)
            return {"status": "registered", "prediction_sha256": "prediction", "context_sha256": "context"}

    class Worker:
        minimum_start_budget_s = 0.0

        @staticmethod
        def solve(_envelope: object) -> None:
            order.append("terra")

    game = {
        "game_id": "pre-terra-ordering",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "hidden", "name": None},
        "prompt": "Official bargaining rules",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}},
        "game_state": {"current_player": "player_1", "history": [], "round": 1, "money_to_divide": 100.0, "complete_information": True, "horizon_known": True, "max_rounds": 5, "messages_allowed": True, "delta_1": 0.9, "delta_2": 0.8},
    }
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=100, max_parallel=1, max_games=1, families=("bargaining",), client=Client(), worker=Worker(), global_tactic_ledger=GlobalTacticLedger(), sequence_shadow_client=Shadow())
    try:
        run._discover_pending([game])
        assert order == ["observe", "shadow"]
        assert captured["game"] == game
        feature_bundle = captured["synthetic_features"]
        assert isinstance(feature_bundle, dict)
        assert "canonical_bargaining_facts" in feature_bundle["features"]
        assert "official_prompt" not in feature_bundle["features"]
        queued = next(iter(run._waiting.values()))
        assert queued.bargaining_analytic_authority_prepared is True
        with ThreadPoolExecutor(max_workers=1) as pool:
            run._dispatch_waiting(pool)
            next(iter(run._inflight.values()))[0].result(timeout=2.0)
        assert order == ["observe", "shadow", "terra"]
    finally:
        run.broker.close()
