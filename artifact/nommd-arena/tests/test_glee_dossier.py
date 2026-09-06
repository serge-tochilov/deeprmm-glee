import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from glee_sdk import GleeAPIError

from nommd_arena.glee_dossier import DossierBroker
from nommd_arena.glee_parallel import ParallelGleeRun
from nommd_arena.glee_tactics import GlobalTacticLedger
from nommd_arena.glee_worker import CAPACITY_MODEL_CHAIN, CapacityFallbackGleeTurnWorker, WorkerDecision
from nommd_arena.models import CognitiveTrace, TetradDisposition, TetradRecordKind, TetradUpdate


def _game(game_id: str, *, opponent_name: str | None = None, game_family: str = "bargaining") -> dict[str, Any]:
    return {
        "game_id": game_id,
        "game_family": game_family,
        "your_player": "player_1",
        "phase": "decision",
        "opponent": {"type": "agent" if opponent_name else "hidden", "name": opponent_name},
        "prompt": "Official rules",
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 100,
            "last_offer": {"player_1_gain": 50, "player_2_gain": 50},
            "round": 1,
            "history": [],
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 3,
            "messages_allowed": True,
            "delta_1": 0.95,
            "delta_2": 0.95,
        },
        "valid_actions": {"type": "decision", "fields": {"decision": ["accept", "reject", "walkaway"]}},
    }


def test_hidden_opponents_are_game_local_but_named_opponents_recur(tmp_path: Path) -> None:
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01")
    hidden_1 = broker.observe_turn(_game("hidden-1"))
    hidden_2 = broker.observe_turn(_game("hidden-2"))
    named_1 = broker.observe_turn(_game("named-1", opponent_name="Aster"))
    named_2 = broker.observe_turn(_game("named-2", opponent_name="Aster", game_family="negotiation"))
    assert hidden_1.scopes.opponent_scope != hidden_2.scopes.opponent_scope
    assert hidden_1.scopes.opponent_label != hidden_2.scopes.opponent_label
    assert not hidden_1.scopes.opponent_persistent
    assert named_1.scopes.opponent_scope == named_2.scopes.opponent_scope
    assert named_1.scopes.population_scope != named_2.scopes.population_scope
    assert named_1.scopes.opponent_label == "Aster"
    assert named_1.scopes.opponent_persistent
    assert hidden_1.memory_context["opponent_prior_game_count"] == 0
    assert hidden_2.memory_context["opponent_prior_game_count"] == 0
    assert named_1.memory_context["opponent_prior_game_count"] == 0
    assert named_2.memory_context["opponent_prior_game_count"] == 1
    broker.close()


def test_global_tactic_memory_reaches_hidden_and_named_games_without_identity_conflation(tmp_path: Path) -> None:
    ledger = GlobalTacticLedger(Path(__file__).resolve().parents[1] / "tactics" / "glee-global-tactics.json")
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01", global_tactic_ledger=ledger)
    hidden_game = _game("hidden-with-tactic")
    hidden_game["your_player"] = "player_2"
    hidden_game["game_state"]["current_player"] = "player_2"
    hidden_game["game_state"]["last_offer"] = {"player_1_gain": 95, "player_2_gain": 5, "message": "Rubinstein equilibrium 5/95", "proposer": "player_1"}
    named_game = {**hidden_game, "game_id": "named-with-tactic", "opponent": {"type": "agent", "name": "Aster"}}
    hidden = broker.observe_turn(hidden_game)
    named = broker.observe_turn(named_game)
    hidden_memory = hidden.memory_context["global_tactic_memory"]
    named_memory = named.memory_context["global_tactic_memory"]
    assert hidden_memory == named_memory
    assert hidden_memory["provenance"]["sha256"] == ledger.sha256
    assert [entry["id"] for entry in hidden_memory["entries"]] == ["bargaining-authority-ratio-ambiguity-v1", "bargaining-token-share-probe-v1"]
    assert "unlabelled_ratio_order" in hidden_memory["entries"][0]["observed"]
    assert "merely positive" in hidden_memory["entries"][1]["avoid"]
    assert len(json.dumps(hidden_memory, ensure_ascii=False, separators=(",", ":"))) < 1000
    assert hidden.memory_context["opponent_identity_scope"] == "game-local-hidden"
    assert named.memory_context["opponent_identity_scope"] == "persistent-named"
    assert hidden_memory["scope"] == {"activation": "current-state-triggered", "identity": "neutral", "authority": "advisory"}
    receipts = [json.loads(line) for line in broker.receipts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshot_receipts = [receipt for receipt in receipts if receipt["kind"] == "snapshot_issued"]
    assert len(snapshot_receipts) == 2
    assert {receipt["global_tactic_ledger_sha256"] for receipt in snapshot_receipts} == {ledger.sha256}
    assert {receipt["global_tactic_trigger_count"] for receipt in snapshot_receipts} == {2}
    broker.close()


def test_global_tactic_memory_is_absent_when_no_current_trigger_fires(tmp_path: Path) -> None:
    ledger = GlobalTacticLedger(Path(__file__).resolve().parents[1] / "tactics" / "glee-global-tactics.json")
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01", global_tactic_ledger=ledger)
    snapshot = broker.observe_turn(_game("neutral"))
    assert "global_tactic_memory" not in snapshot.memory_context
    receipt = json.loads(broker.receipts_path.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["global_tactic_ledger_sha256"] == ledger.sha256
    assert receipt["global_tactic_trigger_count"] == 0
    broker.close()


def test_repository_global_tactic_views_stay_within_live_prompt_budget() -> None:
    ledger = GlobalTacticLedger(Path(__file__).resolve().parents[1] / "tactics" / "glee-global-tactics.json")
    for family in ("bargaining", "negotiation", "persuasion"):
        view = ledger.view(family)
        assert view is not None
        serialized = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
        assert len(serialized) <= 6000


def test_explicit_global_tactic_ledger_path_fails_closed_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="global tactic ledger is missing"):
        GlobalTacticLedger(tmp_path / "missing.json")


def test_snapshot_does_not_materialize_unrelated_global_self_records(tmp_path: Path) -> None:
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01")
    game = _game("bounded-snapshot")
    initial = broker.observe_turn(game)
    unrelated = [
        broker._record(
            record_id=f"unrelated-self:{index:04d}",
            scopes=[initial.scopes.self_scope],
            record_type="cognitive",
            kind="belief",
            disposition="affirmed",
            actor="DeepRMM-01",
            content=f"Unrelated prior-game trace {index}",
            strength=50,
            salience=50,
            tags=["unrelated", "prior-game"],
            visibility="internal",
            payload={"index": index},
            game_id=f"unrelated-game-{index}",
        )
        for index in range(256)
    ]
    with broker._lock:
        broker._transaction()
        try:
            broker._append_records_locked(unrelated)
            broker._connection.commit()
        except Exception:
            broker._connection.rollback()
            raise
        materialized = broker._eligible_snapshot_records_locked(initial.scopes)
    assert len(materialized) == 4
    assert {record["record_type"] for record in materialized} == {"seed", "engine-context", "engine-turn"}
    snapshot = broker.snapshot(game)
    assert snapshot.memory_context["record_count"] == 260
    assert snapshot.memory_context["contract"] == "glee-operational-context-v1"
    assert "retrievable_record_count" not in snapshot.memory_context
    assert "new_observations" not in snapshot.memory_context
    assert "activated_records" not in snapshot.memory_context
    assert snapshot.source_record_ids_by_slot == {}
    broker.close()


def test_raw_prior_game_evidence_is_replaced_by_family_statistical_package(tmp_path: Path) -> None:
    class StaticPackageReader:
        receipt = {"manifest_sha256": "a" * 64}

        def view(self, game: dict[str, Any]) -> dict[str, object]:
            family = str(game["game_family"])
            return {"contract": "glee-opponent-statistical-package-v1", "family": family, "identity_resolution": {"status": "exact-current-label", "public_player_id": "aster-id"}, "evidence": {"games": 4, "observations": 8, "tier": "direct-moderate"}, "live_overlay": {"revision": 17, "state_sha256": "c" * 64}, "action_model": {"authority": "advisory", "contexts": [{"condition": f"kind=proposal|family={family}", "distribution": {"outcomes": [{"outcome": "value=25", "probability": 0.7}]}}]}}

    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01", opponent_statistical_package_reader=StaticPackageReader())
    first = broker.observe_turn(_game("bargaining-with-aster", opponent_name="Aster"))
    update = TetradUpdate(
        updates=[
            CognitiveTrace(
                kind=TetradRecordKind.BELIEF,
                disposition=TetradDisposition.AFFIRMED,
                content="Aster prolongs decisions when the visible state has many credible continuations.",
                strength=70,
                salience=85,
                mental_path=["Aster"],
                source_slots=[0],
                tags=["latency", "complexity-pressure"],
            )
        ]
    )
    action = {"decision": "accept"}
    broker.prepare_turn(first, {"action": action}, action)
    broker.mark_submitting(first.turn_id)
    broker.commit_accepted(snapshot=first, action=action, result={"valid": True, "game_over": False}, tetrad_update=update)
    second = broker.observe_turn(_game("negotiation-with-aster", opponent_name="Aster", game_family="negotiation"))
    assert "activated_records" not in second.memory_context
    package = second.memory_context["opponent_statistical_package"]
    assert package["family"] == "negotiation"
    assert package["identity_resolution"]["public_player_id"] == "aster-id"
    assert package["action_model"]["authority"] == "advisory"
    receipt = json.loads(broker.receipts_path.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["opponent_statistical_package_manifest_sha256"] == "a" * 64
    assert receipt["opponent_statistical_package_resolution"] == "exact-current-label"
    assert receipt["opponent_statistical_overlay_revision"] == 17
    assert receipt["opponent_statistical_overlay_state_sha256"] == "c" * 64
    broker.close()


def test_collision_package_stays_population_only_without_prose_fallback(tmp_path: Path) -> None:
    class CollisionPackageReader:
        receipt = {"manifest_sha256": "b" * 64}

        def view(self, _game: dict[str, Any]) -> dict[str, object]:
            return {"contract": "glee-opponent-statistical-package-v1", "family": "bargaining", "identity_resolution": {"status": "current-label-collision", "public_player_id": None, "candidate_public_player_ids": ["reserve-old", "reserve-new"]}, "evidence": {"games": 0, "observations": 0, "tier": "population-only"}, "action_model": {"authority": "advisory", "contexts": []}}

    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01", opponent_statistical_package_reader=CollisionPackageReader())
    snapshot = broker.observe_turn(_game("collision-game", opponent_name="RESERVE"))
    package = snapshot.memory_context["opponent_statistical_package"]
    assert package["identity_resolution"]["status"] == "current-label-collision"
    assert package["identity_resolution"]["public_player_id"] is None
    assert package["evidence"]["tier"] == "population-only"
    assert "named_opponent_dossier" not in snapshot.memory_context
    broker.close()


def test_same_opponent_concurrent_commits_append_and_receipt_stale_versions(tmp_path: Path) -> None:
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01")
    first = broker.observe_turn(_game("game-1", opponent_name="Aster"))
    second = broker.observe_turn(_game("game-2", opponent_name="Aster"))
    update = TetradUpdate(
        updates=[
            CognitiveTrace(
                kind=TetradRecordKind.BELIEF,
                disposition=TetradDisposition.AFFIRMED,
                content="Aster appears willing to accept an equal split.",
                strength=70,
                salience=70,
                mental_path=["Aster"],
                source_slots=[0],
                tags=["acceptance"],
            )
        ]
    )
    action = {"decision": "accept"}
    broker.prepare_turn(first, {"action": action}, action)
    broker.mark_submitting(first.turn_id)
    first_receipt = broker.commit_accepted(snapshot=first, action=action, result={"valid": True, "game_over": True, "result": {"payoff": 50}}, tetrad_update=update)
    broker.prepare_turn(second, {"action": action}, action)
    broker.mark_submitting(second.turn_id)
    second_receipt = broker.commit_accepted(snapshot=second, action=action, result={"valid": True, "game_over": True, "result": {"payoff": 50}}, tetrad_update=update)
    assert first.scopes.opponent_scope in first_receipt["stale_scopes"]
    assert second.scopes.opponent_scope in second_receipt["stale_scopes"]
    summary = broker.summary()
    assert summary["games"] == {"completed": 2}
    assert summary["records"] >= 10
    broker.close()


def test_observation_and_accepted_commit_are_idempotent(tmp_path: Path) -> None:
    broker = DossierBroker(root=tmp_path / "broker", agent_name="DeepRMM-01")
    game = _game("replay-1")
    first = broker.observe_turn(game)
    count_after_first = broker.summary()["records"]
    second = broker.observe_turn(game)
    assert first.turn_id == second.turn_id
    assert broker.summary()["records"] == count_after_first
    action = {"decision": "accept"}
    result = {"valid": True, "game_over": True, "result": {"payoff": 50}}
    broker.prepare_turn(second, {"action": action}, action)
    broker.mark_submitting(second.turn_id)
    broker.commit_accepted(snapshot=second, action=action, result=result, tetrad_update=TetradUpdate(updates=[]))
    count_after_commit = broker.summary()["records"]
    broker.prepare_turn(second, {"action": action}, action)
    broker.mark_submitting(second.turn_id)
    broker.commit_accepted(snapshot=second, action=action, result=result, tetrad_update=TetradUpdate(updates=[]))
    assert broker.summary()["records"] == count_after_commit
    broker.close()


def test_transport_suspension_survives_restart_and_reconciles_when_server_advances(tmp_path: Path) -> None:
    root = tmp_path / "broker"
    game = _game("transport-reconcile")
    broker = DossierBroker(root=root, agent_name="DeepRMM-01")
    first = broker.observe_turn(game)
    action = {"decision": "accept"}
    broker.prepare_turn(first, {"action": action}, action)
    broker.mark_submitting(first.turn_id)
    broker.suspend_transport_submission(first.turn_id, issue="ambiguous move POST")
    assert broker.transport_suspended_turn_ids() == {first.turn_id}
    broker.close()

    broker = DossierBroker(root=root, agent_name="DeepRMM-01")
    assert broker.transport_suspended_turn_ids() == {first.turn_id}
    later = _game("transport-reconcile")
    later["game_state"]["round"] = 2
    later["game_state"]["history"] = [{"round": 1, "decision": "accept"}]
    second = broker.observe_turn(later)
    assert second.turn_id != first.turn_id
    assert broker.turn_receipt(first.turn_id)["status"] == "reconciled"
    assert broker.transport_suspended_turn_ids() == set()
    broker.close()


def test_parallel_supervisor_keeps_api_on_owner_thread_and_bounds_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    games = [_game(f"game-{index}") for index in range(3)]
    owner_thread = threading.get_ident()

    class FakeClient:
        def __init__(self) -> None:
            self.matched = False
            self.moved: set[str] = set()
            self.api_threads: list[int] = []

        def _thread(self) -> None:
            self.api_threads.append(threading.get_ident())

        def stats(self) -> dict[str, object]:
            self._thread()
            active = len(games) - len(self.moved) if self.matched else 0
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": active, "scores": {}}

        def queue(self, _family: str) -> dict[str, object]:
            self._thread()
            self.matched = True
            return {"status": "queued"}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            self._thread()
            return {"status": "left"}

        def pending_games(self) -> list[dict[str, Any]]:
            self._thread()
            return [game for game in games if self.matched and game["game_id"] not in self.moved]

        def move(self, game_id: str, action: dict[str, Any]) -> dict[str, object]:
            self._thread()
            assert action == {"decision": "accept"}
            self.moved.add(game_id)
            return {"valid": True, "game_over": True, "result": {"payoff": 50}}

        def game_state(self, game_id: str) -> dict[str, Any]:
            self._thread()
            game = next(game for game in games if game["game_id"] == game_id)
            return {**game, "status": "completed", "result": {"payoff": 50}}

    class FakeWorker:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0
            self.worker_threads: list[int] = []

        def solve(self, _envelope: object) -> WorkerDecision:
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                self.worker_threads.append(threading.get_ident())
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return WorkerDecision(action={"decision": "accept"}, proposal={"decision": "accept"}, tetrad_update=TetradUpdate(updates=[]), tetrad_transport_issues=[], deterministic_safeguards=[], fallback=False, fallback_reason=None, role="glee_nommd_bargaining", elapsed_s=0.02, call_metadata={"call_id": "fake"})

        def fallback(self, _envelope: object, reason: str) -> WorkerDecision:
            return WorkerDecision(action={"decision": "accept"}, proposal=None, tetrad_update=None, tetrad_transport_issues=[], deterministic_safeguards=[], fallback=True, fallback_reason=reason, role="glee_nommd_bargaining", elapsed_s=0.0, call_metadata=None)

    client = FakeClient()
    worker = FakeWorker()
    monkeypatch.setenv("GLEE_API_KEY", "glee_must_not_reach_workers")
    result = ParallelGleeRun(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        env_file=None,
        model="test-model",
        effort="max",
        poll_interval_s=0,
        max_parallel=2,
        max_games=3,
        families=("bargaining",),
        client=client,
        worker=worker,
        global_tactic_ledger=GlobalTacticLedger(),
    ).run()
    assert result["mode"] == "glee-parallel-v17"
    assert result["worker_policy"] == "capacity-chain"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_timeout_s"] == 108
    assert manifest["emergency_margin_s"] == 12.0
    assert manifest["high_effort"] == "high"
    assert result["completed_game_ids"] == ["game-0", "game-1", "game-2"]
    assert client.moved == {"game-0", "game-1", "game-2"}
    assert set(client.api_threads) == {owner_thread}
    assert "GLEE_API_KEY" not in os.environ
    assert worker.maximum_active == 2
    assert all(thread_id != owner_thread for thread_id in worker.worker_threads)
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(event.get("kind") == "turn_observed" for event in events) == 3


def test_parallel_default_wires_every_family_to_the_capacity_chain(tmp_path: Path) -> None:
    class StatsClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

    run = ParallelGleeRun(project_root=Path(__file__).resolve().parents[1], run_dir=tmp_path / "run", env_file=None, model="gpt-5.6-terra", effort="high", max_parallel=1, max_games=1, families=("negotiation",), opponent_timing_root=tmp_path / "timing", client=StatsClient(), global_tactic_ledger=GlobalTacticLedger())
    assert run.worker_policy == "capacity-chain"
    assert isinstance(run.worker, CapacityFallbackGleeTurnWorker)
    assert run.worker.model_chain == CAPACITY_MODEL_CHAIN
    assert run._mode() == "glee-parallel-v17"
    run.broker.close()


def test_parallel_legacy_bargaining_capacity_policy_preserves_the_v16_chain(tmp_path: Path) -> None:
    class StatsClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

    project_root = Path(__file__).resolve().parents[1]
    run = ParallelGleeRun(
        project_root=project_root,
        run_dir=tmp_path / "run",
        env_file=None,
        model="gpt-5.6-terra",
        effort="high",
        high_effort="high",
        worker_policy="bargaining-capacity-chain",
        max_parallel=1,
        max_games=1,
        families=("bargaining",),
        client=StatsClient(),
        global_tactic_ledger=GlobalTacticLedger(),
    )
    assert isinstance(run.worker, CapacityFallbackGleeTurnWorker)
    assert run.worker.model_chain == CAPACITY_MODEL_CHAIN
    assert run._mode() == "glee-parallel-v16"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capacity_model_chain"] == run.worker.manifest_chain
    run.broker.close()


def test_fixed_family_slots_prioritize_persuasion_and_cap_stop_family_admission(tmp_path: Path) -> None:
    class QueueClient:
        def __init__(self) -> None:
            self.queued: list[str] = []

        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def queue(self, family: str) -> dict[str, object]:
            self.queued.append(family)
            return {"status": "queued", "game_family": family}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

    client = QueueClient()
    run = ParallelGleeRun(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        env_file=None,
        model="test-model",
        effort="max",
        max_parallel=5,
        max_games=None,
        family_slots={"bargaining": 1, "negotiation": 1, "persuasion": 3},
        stop_after_family=("persuasion", 6),
        client=client,
        worker=object(),
        global_tactic_ledger=GlobalTacticLedger(),
    )
    run._last_topup = float("-inf")
    run._top_up(active_games=0)
    assert client.queued == ["persuasion", "bargaining", "negotiation"]
    run._queued_families.clear()
    client.queued.clear()
    for index in range(6):
        game = _game(f"persuasion-{index}", game_family="persuasion")
        run.broker.observe_turn(game)
        if index < 4:
            run.broker.mark_game_completed(game["game_id"], {**game, "status": "completed", "result": {"payoff": index}})
    run._last_topup = float("-inf")
    run._top_up(active_games=2)
    assert client.queued == ["bargaining", "negotiation"]
    assert run._limit_reason(time.monotonic()) is None
    for index in range(4, 6):
        game = _game(f"persuasion-{index}", game_family="persuasion")
        run.broker.mark_game_completed(game["game_id"], {**game, "status": "completed", "result": {"payoff": index}})
    assert "reaching the configured target of 6" in str(run._limit_reason(time.monotonic()))
    assert run.broker.game_counts_by_family("completed") == {"persuasion": 6}
    run.broker.close()


def test_fixed_family_slots_reserve_active_games_against_global_completion_cap(tmp_path: Path) -> None:
    class QueueClient:
        def __init__(self) -> None:
            self.queued: list[str] = []

        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def queue(self, family: str) -> dict[str, object]:
            self.queued.append(family)
            return {"status": "queued", "game_family": family}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

    client = QueueClient()
    run = ParallelGleeRun(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        env_file=None,
        model="test-model",
        effort="max",
        max_parallel=5,
        max_games=5,
        families=("bargaining",),
        family_slots={"bargaining": 5},
        client=client,
        worker=object(),
        global_tactic_ledger=GlobalTacticLedger(),
    )
    for index in range(5):
        game = _game(f"bargaining-{index}", game_family="bargaining")
        run.broker.observe_turn(game)
        if index < 3:
            run.broker.mark_game_completed(game["game_id"], {**game, "status": "completed", "result": {"payoff": index}})
    run._last_topup = float("-inf")
    run._top_up(active_games=2)
    assert client.queued == []
    run.broker.close()


def test_persistent_graceful_drain_request_prevents_new_admission(tmp_path: Path) -> None:
    class QueueClient:
        def __init__(self) -> None:
            self.queued: list[str] = []
            self.left = 0

        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 2, "scores": {}}

        def queue(self, family: str) -> dict[str, object]:
            self.queued.append(family)
            return {"status": "queued", "game_family": family}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            self.left += 1
            return {"status": "left"}

    client = QueueClient()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test-model", effort="max", max_parallel=2, max_games=100, families=("persuasion",), family_slots={"persuasion": 2}, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    run.drain_request_path.write_text("requested\n", encoding="utf-8")
    reason = run._limit_reason(time.monotonic())
    assert reason == "persistent graceful-drain request"
    run._enter_drain(reason)
    run._last_topup = float("-inf")
    run._top_up(active_games=2)
    assert client.left == 1
    assert client.queued == []
    run.broker.close()


def test_rss_ceiling_requests_graceful_drain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MemoryClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

    monkeypatch.setattr("nommd_arena.glee_parallel._process_memory_mib", lambda: {"rss_mib": 1024.5, "peak_rss_mib": 1024.5})
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test-model", effort="max", max_parallel=1, max_games=1, families=("bargaining",), max_rss_mib=1024.0, client=MemoryClient(), worker=object(), global_tactic_ledger=GlobalTacticLedger())
    reason = run._limit_reason(time.monotonic())
    assert reason == "process RSS 1024.500 MiB reached the configured ceiling of 1024.000 MiB"
    memory_events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    assert memory_events[-1]["kind"] == "memory_sample"
    assert memory_events[-1]["rss_mib"] == 1024.5
    run.broker.close()


def test_memory_receipt_preserves_run_wide_peak_when_proc_report_decreases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MemoryClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

    samples = iter(({"rss_mib": 243.043, "peak_rss_mib": 243.043}, {"rss_mib": 242.934, "peak_rss_mib": 242.934}))
    monkeypatch.setattr("nommd_arena.glee_parallel._process_memory_mib", lambda: next(samples))
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test-model", effort="max", max_parallel=1, max_games=1, families=("bargaining",), client=MemoryClient(), worker=object(), global_tactic_ledger=GlobalTacticLedger())
    first = run._sample_memory(force=True)
    second = run._sample_memory(force=True)
    assert first["peak_rss_mib"] == 243.043
    assert second == {"rss_mib": 242.934, "process_reported_peak_rss_mib": 242.934, "peak_rss_mib": 243.043}
    run.broker.close()


def test_memory_receipt_restores_run_wide_peak_after_supervisor_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MemoryClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(json.dumps({"kind": "memory_sample", "rss_mib": 240.0, "peak_rss_mib": 300.0}) + "\n", encoding="utf-8")
    monkeypatch.setattr("nommd_arena.glee_parallel._process_memory_mib", lambda: {"rss_mib": 245.0, "peak_rss_mib": 250.0})
    run = ParallelGleeRun(project_root=tmp_path, run_dir=run_dir, env_file=None, model="test-model", effort="max", max_parallel=1, max_games=1, families=("bargaining",), client=MemoryClient(), worker=object(), global_tactic_ledger=GlobalTacticLedger())
    assert run._sample_memory(force=True) == {"rss_mib": 245.0, "process_reported_peak_rss_mib": 250.0, "peak_rss_mib": 300.0}
    run.broker.close()


def test_forced_refresh_reconciles_game_that_completes_during_shutdown_window(tmp_path: Path) -> None:
    game = _game("late-completion", opponent_name="Aster", game_family="negotiation")
    final_state = {**game, "status": "completed", "result": {"outcome": "no_deal", "player_1_payoff": 0, "player_2_payoff": 0}}

    class LateCompletionClient:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

        def game_state(self, game_id: str) -> dict[str, Any]:
            assert game_id == "late-completion"
            return final_state

    run = ParallelGleeRun(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        env_file=None,
        model="test-model",
        effort="max",
        max_parallel=1,
        max_games=1,
        families=("negotiation",),
        client=LateCompletionClient(),
        worker=object(),
        global_tactic_ledger=GlobalTacticLedger(),
    )
    run.broker.observe_turn(game)
    run._last_refresh = time.monotonic()
    run._refresh_known_games(set())
    assert run.broker.known_active_game_ids() == {"late-completion"}
    run._last_refresh = float("-inf")
    run._refresh_known_games(set(), force=True)
    assert run.broker.known_active_game_ids() == set()
    assert run.broker.completed_game_ids() == {"late-completion"}
    assert (tmp_path / "run" / "games" / "negotiation-late-completion.json").is_file()
    assert not (tmp_path / "opponent-dossiers").exists()
    run.broker.close()


def test_known_game_refresh_is_round_robin_and_rate_limit_is_nonfatal(tmp_path: Path) -> None:
    games = [_game(f"game-{index}", game_family="negotiation") for index in range(3)]

    class RefreshClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 3, "scores": {}}

        def game_state(self, game_id: str) -> dict[str, Any]:
            self.calls.append(game_id)
            if len(self.calls) == 1:
                raise GleeAPIError(429, "Rate limit exceeded: 60 requests per 60s")
            game = next(item for item in games if item["game_id"] == game_id)
            return {**game, "status": "completed", "result": {"outcome": "no_deal"}}

    client = RefreshClient()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test-model", effort="max", max_parallel=1, max_games=3, families=("negotiation",), client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    for game in games:
        run.broker.observe_turn(game)
    run._last_refresh = float("-inf")
    run._refresh_known_games(set())
    assert client.calls == ["game-0"]
    assert run.broker.known_active_game_ids() == {"game-0", "game-1", "game-2"}
    run._last_refresh = float("-inf")
    run._refresh_known_games(set())
    assert client.calls == ["game-0", "game-1"]
    assert run.broker.known_active_game_ids() == {"game-0", "game-2"}
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    failed = next(event for event in events if event["kind"] == "known_game_refresh_failed")
    assert failed["rate_limited"] is True
    assert failed["candidate_count"] == 3
    run.broker.close()


def test_submitted_terminal_move_uses_accepted_result_without_a_redundant_state_get(tmp_path: Path) -> None:
    game = _game("submitted-terminal")

    class Client:
        def __init__(self) -> None:
            self.matched = False
            self.moved = False

        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": int(self.matched and not self.moved), "scores": {}}

        def leave_queue(self, _family: str | None = None) -> dict[str, object]:
            return {"status": "left"}

        def queue(self, _family: str) -> dict[str, object]:
            self.matched = True
            return {"status": "queued"}

        def pending_games(self) -> list[dict[str, Any]]:
            return [game] if self.matched and not self.moved else []

        def move(self, _game_id: str, _action: dict[str, Any]) -> dict[str, object]:
            self.moved = True
            return {"valid": True, "game_over": True, "result": {"outcome": "agreement", "player_1_payoff": 50, "player_2_payoff": 50}}

        def game_state(self, _game_id: str) -> dict[str, Any]:
            raise AssertionError("a game-over move already supplies the terminal result")

    class Worker:
        minimum_start_budget_s = 1.0

        def solve(self, _envelope: object) -> WorkerDecision:
            return WorkerDecision(action={"decision": "accept"}, proposal=None, tetrad_update=TetradUpdate(updates=[]), tetrad_transport_issues=[], deterministic_safeguards=[], fallback=False, fallback_reason=None, role="glee_nommd_bargaining", elapsed_s=0.01, call_metadata={"call_id": "test"})

        def fallback(self, _envelope: object, reason: str) -> WorkerDecision:
            raise AssertionError(reason)

    class PackageReader:
        receipt = {"contract": "test-package", "manifest_sha256": "package-sha", "live_overlay_contract": "test-overlay"}

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.updates: dict[str, dict[str, object]] = {}
            self.closed = False

        def view(self, _game: dict[str, object]) -> dict[str, object]:
            return {"identity_resolution": {"status": "hidden-population"}, "live_overlay": {"revision": len(self.updates), "state_sha256": f"state-{len(self.updates)}"}}

        def update_completed_game(self, final_game: dict[str, object], **_values: object) -> dict[str, object]:
            self.calls.append(final_game)
            game_id = str(final_game["game_id"])
            status = "duplicate" if game_id in self.updates else "updated"
            self.updates[game_id] = final_game
            return {"status": status, "revision": len(self.updates)}

        def status(self) -> dict[str, object]:
            return {"revision": len(self.updates)}

        def close(self) -> None:
            self.closed = True

    package_reader = PackageReader()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test-model", effort="max", poll_interval_s=0, max_parallel=1, max_games=1, families=("bargaining",), client=Client(), worker=Worker(), opponent_statistical_package_reader=package_reader, global_tactic_ledger=GlobalTacticLedger())
    result = run.run()
    assert result["completed_game_ids"] == ["submitted-terminal"]
    final_state = json.loads((tmp_path / "run" / "games" / "bargaining-submitted-terminal.json").read_text(encoding="utf-8"))
    assert final_state["result"]["outcome"] == "agreement"
    assert final_state["game_state"]["history"][-1]["decision"] == "accept"
    completion = next(json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "game_completed")
    assert completion["final_state_source"] == "accepted-move-result"
    assert [update["game_id"] for update in package_reader.calls] == ["submitted-terminal", "submitted-terminal"]
    assert list(package_reader.updates) == ["submitted-terminal"]
    assert package_reader.closed is True
