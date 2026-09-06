import json
from pathlib import Path
from typing import Any

import pytest
from glee_sdk import GleeAPIError
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout as RequestsTimeout

import nommd_arena.glee_parallel as parallel_module
from nommd_arena.glee_parallel import ParallelGleeRun
from nommd_arena.glee_tactics import GlobalTacticLedger
from nommd_arena.glee_timing import OpponentTimingStore, bootstrap_timing_store
from nommd_arena.glee_transport import NonReplayingGleeClient
from nommd_arena.glee_worker import TurnEnvelope, WorkerDecision
from nommd_arena.models import TetradUpdate


def _bargaining_game(game_id: str, name: str | None, delay_ms: float, *, your_player: str = "player_1") -> dict[str, Any]:
    opponent_type = "agent" if name else "hidden"
    proposer = your_player
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": your_player,
        "phase": "offer",
        "opponent": {"type": opponent_type, "name": name},
        "prompt": "Official bargaining rules",
        "valid_actions": {"type": "offer", "fields": {"player_1_gain": "number", "player_2_gain": "number"}},
        "game_state": {
            "phase": "offer",
            "round": 2,
            "history": [{"round": 1, "proposer": proposer, "offer": {"round": 1, "proposer": proposer, "player_1_gain": 50, "player_2_gain": 50}, "decision": "reject", "response_time_ms": delay_ms}],
            "messages_allowed": False,
            "complete_information": True,
            "horizon_known": False,
        },
    }


def _observe(store: OpponentTimingStore, game: dict[str, Any], index: int) -> dict[str, object]:
    return store.observe_turn(game=game, turn_id=f"{game['game_id']}:turn", observed_at=f"2026-08-12T00:00:{index:02d}+00:00", source_run="test-run", source_event_sequence=index)


def test_exact_server_delay_is_attributed_only_to_the_opponent_and_deduplicated(tmp_path: Path) -> None:
    store = OpponentTimingStore(tmp_path / "timing")
    game = _bargaining_game("named-one", "SlowMind", 12_000)
    first = _observe(store, game, 1)
    second = _observe(store, game, 2)
    assert first["inserted_exact"] == 1
    assert second["inserted_exact"] == 0
    assert first["profile"]["exact"]["median_ms"] == 12_000
    own_response = _bargaining_game("own-response", "SlowMind", 500, your_player="player_2")
    own_response["game_state"]["history"][0]["proposer"] = "player_1"
    own_response["game_state"]["history"][0]["offer"]["proposer"] = "player_1"
    _observe(store, own_response, 3)
    assert store.counts()["observations"] == 1
    store.close()


def test_timing_fingerprint_separates_fast_and_slow_profiles_and_estimates_hardness(tmp_path: Path) -> None:
    store = OpponentTimingStore(tmp_path / "timing")
    for index, delay in enumerate((500, 700, 900, 1_100), start=1):
        _observe(store, _bargaining_game(f"fast-{index}", "FastMind", delay), index)
    slow_receipt: dict[str, object] | None = None
    for index, delay in enumerate((9_000, 10_000, 11_000, 30_000), start=10):
        slow_receipt = _observe(store, _bargaining_game(f"slow-{index}", "SlowMind", delay), index)
    assert store.profile(opponent_id=slow_receipt["profile"]["opponent_id"], family="bargaining")["engine_hint"] == "model-call-like"
    fast_profile = next(store.profile(opponent_id=row[0], family="bargaining") for row in store.connection.execute("SELECT DISTINCT opponent_id FROM observations WHERE opponent_name = 'FastMind'"))
    assert fast_profile["engine_hint"] == "deterministic-like"
    assert slow_receipt["latest_hardness"]["hardness_hint"] == "hard-state-like"
    hidden = _observe(store, _bargaining_game("hidden-slow", None, 10_500), 20)
    assert hidden["timing_candidates"][0]["opponent_name"] == "SlowMind"
    assert hidden["timing_candidates"][0]["timing_similarity"] > hidden["timing_candidates"][1]["timing_similarity"]
    store.close()


def test_local_causal_wall_delay_fills_moves_without_server_response_timing(tmp_path: Path) -> None:
    store = OpponentTimingStore(tmp_path / "timing", poll_resolution_s=4.0)
    initial = _bargaining_game("wall-game", "SignalMind", 1_000)
    initial["game_state"]["history"] = []
    store.record_submission(game=initial, turn_id="wall-game:r1", submitted_at="2026-08-12T00:00:00+00:00", source_run="test-run", source_event_sequence=1)
    following = {**initial, "phase": "decision", "game_state": {**initial["game_state"], "phase": "decision", "round": 2}}
    receipt = store.observe_turn(game=following, turn_id="wall-game:r2", observed_at="2026-08-12T00:00:05+00:00", source_run="test-run", source_event_sequence=2)
    assert receipt["inserted_wall"] is True
    assert receipt["profile"]["causal_wall"]["median_ms"] == 5_000
    row = store.connection.execute("SELECT move_kind, resolution_ms FROM observations").fetchone()
    assert dict(row) == {"move_kind": "opponent-proposal", "resolution_ms": 4_000.0}
    duplicate = store.observe_turn(game=following, turn_id="wall-game:r2", observed_at="2026-08-12T00:00:06+00:00", source_run="test-run", source_event_sequence=3)
    assert duplicate["inserted_wall"] is False
    store.close()


def test_bootstrap_uses_event_and_game_references_without_copying_full_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source-run"
    (run_dir / "games").mkdir(parents=True)
    first = _bargaining_game("bootstrap", "ArchiveMind", 1_000)
    first["game_state"]["history"] = []
    second = _bargaining_game("bootstrap", "ArchiveMind", 9_000)
    events = [
        {"schema_version": 1, "event_sequence": 1, "ts": "2026-08-12T00:00:00+00:00", "kind": "turn_observed", "turn_id": "bootstrap:r1", "game_id": "bootstrap", "family": "bargaining", "game": first},
        {"schema_version": 1, "event_sequence": 2, "ts": "2026-08-12T00:00:02+00:00", "kind": "move_submitted", "turn_id": "bootstrap:r1", "game_id": "bootstrap", "action": {"decision": "reject"}, "result": {"valid": True, "game_over": False}},
        {"schema_version": 1, "event_sequence": 3, "ts": "2026-08-12T00:00:08+00:00", "kind": "turn_observed", "turn_id": "bootstrap:r2", "game_id": "bootstrap", "family": "bargaining", "game": second},
    ]
    (run_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    receipt = bootstrap_timing_store(source_root=tmp_path / "runs", output_root=tmp_path / "timing")
    assert receipt["store"] == {"observations": 2, "games": 1, "named_opponents": 1}
    assert not list((tmp_path / "timing").glob("*.json"))
    store = OpponentTimingStore(tmp_path / "timing")
    sources = {row[0] for row in store.connection.execute("SELECT timing_source FROM observations")}
    assert sources == {"server-response-time", "local-causal-wall"}
    store.close()


class _Client:
    def __init__(self) -> None:
        self.moves: list[tuple[str, dict[str, Any]]] = []

    def stats(self) -> dict[str, object]:
        return {"agent_id": "agent", "agent_name": "DeepRMM-01", "active_games": 1, "scores": {}}

    def move(self, game_id: str, action: dict[str, Any]) -> dict[str, object]:
        self.moves.append((game_id, action))
        return {"valid": True, "game_over": True, "result": {"outcome": "agreement", "player_1_payoff": 50, "player_2_payoff": 50}}


class _AmbiguousMoveClient(_Client):
    timeout = 10

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error
        self.attempts = 0

    def move(self, game_id: str, action: dict[str, Any]) -> dict[str, object]:
        self.attempts += 1
        raise self.error


class _FixedRandom:
    def betavariate(self, _alpha: float, _beta: float) -> float:
        return 0.5

    def uniform(self, _lower: float, _upper: float) -> float:
        return 0.0


def _decision() -> WorkerDecision:
    return WorkerDecision(action={"decision": "accept"}, proposal=None, tetrad_update=TetradUpdate(updates=[]), tetrad_transport_issues=[], deterministic_safeguards=[], fallback=False, fallback_reason=None, role="glee_nommd_bargaining", elapsed_s=5.0, call_metadata={"model": "test"}, selection_branch="terra-high")


def test_concealment_schedules_total_latency_without_blocking_and_releases_at_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=20, move_delay_max_s=20, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("concealed", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._delay_random = _FixedRandom()
    run._first_seen[snapshot.turn_id] = 100.0
    run._stage_submission(envelope, _decision())
    assert not client.moves
    assert snapshot.turn_id in run._ready
    assert run.broker.turn_receipt(snapshot.turn_id)["status"] == "prepared"
    clock[0] = 119.9
    run._release_ready()
    assert not client.moves
    clock[0] = 120.0
    run._release_ready()
    assert client.moves == [("concealed", {"decision": "accept"})]
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    released = next(event for event in events if event["kind"] == "move_delay_released")
    assert released["concealment"]["release_reason"] == "target-reached"
    assert released["concealment"]["applied_delay_s"] == 15.0
    run.broker.close()
    run.opponent_timing.close()


def test_ready_move_submissions_are_spaced_without_serializing_model_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=2, max_games=2, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_submission_min_spacing_s=5.0, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    first = _bargaining_game("first", "Other", 1_000)
    first["phase"] = "decision"
    first["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    second = _bargaining_game("second", "Other", 1_000)
    second["phase"] = "decision"
    second["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    first_snapshot = run.broker.observe_turn(first)
    second_snapshot = run.broker.observe_turn(second)
    first_envelope = TurnEnvelope(game=first, snapshot=first_snapshot, deadline_at_monotonic=200.0)
    second_envelope = TurnEnvelope(game=second, snapshot=second_snapshot, deadline_at_monotonic=220.0)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._first_seen[first_snapshot.turn_id] = 100.0
    run._first_seen[second_snapshot.turn_id] = 100.0

    run._stage_submission(first_envelope, _decision())
    assert client.moves == [("first", {"decision": "accept"})]
    clock[0] = 106.0
    run._stage_submission(second_envelope, _decision())
    assert client.moves == [("first", {"decision": "accept"})]
    assert second_snapshot.turn_id in run._ready
    clock[0] = 109.9
    run._release_ready()
    assert len(client.moves) == 1
    clock[0] = 110.0
    run._release_ready()
    assert client.moves[-1] == ("second", {"decision": "accept"})
    released = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "move_delay_released"]
    assert released[-1]["concealment"]["queued_beyond_preferred_release_s"] == 4.0
    assert json.loads(run.manifest_path.read_text(encoding="utf-8"))["api_dispatch_smoothing"]["model_call_serialization"] is False
    run.broker.close()
    run.opponent_timing.close()


def test_move_submission_spacing_yields_to_transport_wait_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=2, max_games=2, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_submission_min_spacing_s=5.0, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    first = _bargaining_game("first", "Other", 1_000)
    first["phase"] = "decision"
    first["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    urgent = _bargaining_game("urgent", "Other", 1_000)
    urgent["phase"] = "decision"
    urgent["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    first_snapshot = run.broker.observe_turn(first)
    urgent_snapshot = run.broker.observe_turn(urgent)
    first_envelope = TurnEnvelope(game=first, snapshot=first_snapshot, deadline_at_monotonic=200.0)
    urgent_envelope = TurnEnvelope(game=urgent, snapshot=urgent_snapshot, deadline_at_monotonic=130.0)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._first_seen[first_snapshot.turn_id] = 100.0
    run._first_seen[urgent_snapshot.turn_id] = 100.0

    run._stage_submission(first_envelope, _decision())
    clock[0] = 106.0
    run._stage_submission(urgent_envelope, _decision())
    assert client.moves == [("first", {"decision": "accept"}), ("urgent", {"decision": "accept"})]
    released = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "move_delay_released"]
    assert released[-1]["concealment"]["release_reason"] == "submission-spacing-transport-reserve"
    assert released[-1]["concealment"]["remaining_deadline_s"] == 24.0
    run.broker.close()
    run.opponent_timing.close()


@pytest.mark.parametrize("operation", ["move", "queue"])
def test_production_client_never_replays_a_post_after_connection_failure(operation: str) -> None:
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, method: str, url: str, **kwargs: Any) -> None:
            self.calls += 1
            raise RequestsConnectionError("peer closed after receiving request")

    client = NonReplayingGleeClient(api_key="test", timeout=10)
    session = Session()
    client.session = session
    with pytest.raises(RequestsConnectionError):
        if operation == "move":
            client.move("game-1", {"decision": "accept"})
        else:
            client.queue("bargaining")
    assert session.calls == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RequestsConnectionError("peer closed after receiving POST"), "ambiguous-post-connection-error"),
        (RequestsTimeout("response timed out after POST"), "ambiguous-post-timeout"),
        (GleeAPIError(500, "internal move failure"), "ambiguous-server-error"),
    ],
)
def test_ambiguous_move_transport_failure_is_durably_isolated_to_one_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str) -> None:
    client = _AmbiguousMoveClient(error)
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("ambiguous-transport", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 105.0)
    run._first_seen[snapshot.turn_id] = 100.0
    run._stage_submission(envelope, _decision())
    assert client.attempts == 1
    assert run.broker.turn_receipt(snapshot.turn_id)["status"] == "transport-suspended"
    assert run.broker.transport_suspended_turn_ids() == {snapshot.turn_id}
    assert snapshot.turn_id in run._transport_blocked_turns
    run._discover_pending([game])
    assert client.attempts == 1
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    ambiguous = next(event for event in events if event["kind"] == "move_submission_transport_ambiguous")
    assert ambiguous["reason"] == reason
    suspended = next(event for event in events if event["kind"] == "move_submission_suspended")
    assert suspended["broker_status"] == "transport-suspended"
    assert suspended["durable"] is True
    run.broker.close()
    run.opponent_timing.close()


def test_inactive_game_submission_race_reconciles_one_turn_without_killing_supervisor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _AmbiguousMoveClient(GleeAPIError(400, "Game is not active"))
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("inactive-race", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 105.0)
    run._first_seen[snapshot.turn_id] = 100.0

    run._stage_submission(envelope, _decision())

    assert client.attempts == 1
    assert run.broker.turn_receipt(snapshot.turn_id)["status"] == "reconciled"
    assert snapshot.turn_id not in run._transport_blocked_turns
    assert not run._draining
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["kind"] == "move_submission_terminal_race" for event in events)
    assert not any(event["kind"] == "move_submission_turn_failure_contained" for event in events)
    run.broker.close()
    run.opponent_timing.close()


def test_unexpected_release_failure_is_isolated_before_supervisor_drains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _AmbiguousMoveClient(RuntimeError("unexpected submission defect"))
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("unexpected-release", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 105.0)
    run._first_seen[snapshot.turn_id] = 100.0

    run._stage_submission(envelope, _decision())

    assert client.attempts == 1
    assert run.broker.turn_receipt(snapshot.turn_id)["status"] == "transport-suspended"
    assert snapshot.turn_id in run._transport_blocked_turns
    assert run._draining
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    contained = next(event for event in events if event["kind"] == "move_submission_turn_failure_contained")
    assert contained["supervisor_survived"] is True
    assert contained["broker_status"] == "transport-suspended"
    run.broker.close()
    run.opponent_timing.close()


def test_reconciled_delayed_submission_is_discarded_before_network_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=20, move_delay_max_s=20, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("reconciled-ready", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._delay_random = _FixedRandom()
    run._first_seen[snapshot.turn_id] = 100.0
    run._stage_submission(envelope, _decision())
    assert snapshot.turn_id in run._ready
    run.broker.reconcile_terminal_submission(snapshot.turn_id, issue="test server-side advancement")

    clock[0] = 120.0
    run._release_ready()

    assert client.moves == []
    assert snapshot.turn_id not in run._ready
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    discarded = next(event for event in events if event["kind"] == "stale_ready_submission_discarded")
    assert discarded["broker_status"] == "reconciled"
    run.broker.close()
    run.opponent_timing.close()


def test_recovered_prepared_turn_is_never_submitted_after_its_original_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("expired-transport", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=100.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 105.0)
    run._stage_submission(envelope, _decision())
    assert client.moves == []
    assert snapshot.turn_id in run._transport_blocked_turns
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    exhausted = next(event for event in events if event["kind"] == "move_submission_transport_exhausted")
    assert exhausted["reason"] == "original-turn-deadline-elapsed"
    run.broker.close()
    run.opponent_timing.close()


def test_hidden_timing_persona_uses_the_pinned_game_quantile_instead_of_ki_beta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=14, move_delay_max_s=46, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("persona", None, 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    profile = {
        "profile_id": "terse-lower",
        "timing_persona": {
            "contract": "glee-hi-timing-persona-v1",
            "activation": "authoritative",
            "timing_profile_id": "test-mixed",
            "game_speed_quantile": 0.5,
            "residual_quantile_half_width": 0.1,
            "quantile_seconds": [{"q": 0.0, "seconds": 14.0}, {"q": 0.5, "seconds": 30.0}, {"q": 1.0, "seconds": 46.0}],
            "move_multipliers": {"proposal": 1.0, "decision": 1.0},
        },
    }
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0, message_style_profile=profile)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._delay_random = _FixedRandom()
    run._first_seen[snapshot.turn_id] = 100.0
    run._stage_submission(envelope, _decision())
    assert snapshot.turn_id in run._ready
    scheduled = next(json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "move_delay_scheduled")
    assert scheduled["concealment"]["target_elapsed_s"] == 30.0
    assert scheduled["concealment"]["timing_persona"]["authority"] == "hi-timing-persona"
    assert scheduled["concealment"]["timing_persona"]["game_speed_quantile"] == 0.5
    run.broker.close()
    run.opponent_timing.close()


def test_concealment_never_spends_the_deadline_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=46, move_delay_max_s=46, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("deadline", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    clock = [187.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._first_seen[snapshot.turn_id] = 170.0
    run._stage_submission(envelope, _decision())
    assert client.moves == [("deadline", {"decision": "accept"})]
    released = next(json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "move_delay_released")
    assert released["concealment"]["release_reason"] == "deadline-reserve"
    assert released["concealment"]["applied_delay_s"] == 0.0
    assert released["concealment"]["remaining_deadline_s"] == 13.0
    run.broker.close()
    run.opponent_timing.close()


def test_natural_long_latency_receives_no_additional_delay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=20, move_delay_max_s=20, client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    game = _bargaining_game("naturally-slow", "Other", 1_000)
    game["phase"] = "decision"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    snapshot = run.broker.observe_turn(game)
    envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=200.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 130.0)
    run._first_seen[snapshot.turn_id] = 100.0
    run._stage_submission(envelope, _decision())
    assert client.moves == [("naturally-slow", {"decision": "accept"})]
    released = next(json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "move_delay_released")
    assert released["concealment"]["release_reason"] == "natural-latency-reached-target"
    assert released["concealment"]["applied_delay_s"] == 0.0
    run.broker.close()
    run.opponent_timing.close()


def test_known_only_concealment_skips_hidden_identity_but_retains_named_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=20, move_delay_max_s=20, move_delay_scope="known-only", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    hidden = _bargaining_game("hidden", None, 1_000)
    hidden["phase"] = "decision"
    hidden["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    hidden_snapshot = run.broker.observe_turn(hidden)
    hidden_envelope = TurnEnvelope(game=hidden, snapshot=hidden_snapshot, deadline_at_monotonic=200.0)
    named = _bargaining_game("named", "Other", 1_000)
    named["phase"] = "decision"
    named["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    named_snapshot = run.broker.observe_turn(named)
    named_envelope = TurnEnvelope(game=named, snapshot=named_snapshot, deadline_at_monotonic=200.0)
    clock = [105.0]
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: clock[0])
    run._first_seen[hidden_snapshot.turn_id] = 100.0
    run._first_seen[named_snapshot.turn_id] = 100.0

    run._stage_submission(hidden_envelope, _decision())
    run._stage_submission(named_envelope, _decision())

    assert client.moves == [("hidden", {"decision": "accept"})]
    assert named_snapshot.turn_id in run._ready
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    hidden_release = next(event for event in events if event["kind"] == "move_delay_released" and event["game_id"] == "hidden")
    assert hidden_release["concealment"]["release_reason"] == "identity-scope-disabled"
    assert hidden_release["concealment"]["scope_applied"] is False
    run.broker.close()
    run.opponent_timing.close()


def test_hidden_only_concealment_skips_named_identity_but_retains_hidden_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    run = ParallelGleeRun(project_root=tmp_path, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", worker_policy="single", model_timeout_s=20, turn_deadline_s=100, emergency_margin_s=12, poll_interval_s=2, max_parallel=1, max_games=1, families=("bargaining",), opponent_timing_root=tmp_path / "timing", move_delay_min_s=20, move_delay_max_s=20, move_delay_scope="hidden-only", client=client, worker=object(), global_tactic_ledger=GlobalTacticLedger())
    named = _bargaining_game("named", "Other", 1_000)
    named["phase"] = "decision"
    named["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    named_snapshot = run.broker.observe_turn(named)
    named_envelope = TurnEnvelope(game=named, snapshot=named_snapshot, deadline_at_monotonic=200.0)
    hidden = _bargaining_game("hidden", None, 1_000)
    hidden["phase"] = "decision"
    hidden["valid_actions"] = {"type": "decision", "fields": {"decision": ["accept", "reject"]}}
    hidden_snapshot = run.broker.observe_turn(hidden)
    hidden_envelope = TurnEnvelope(game=hidden, snapshot=hidden_snapshot, deadline_at_monotonic=200.0)
    monkeypatch.setattr(parallel_module.time, "monotonic", lambda: 105.0)
    run._first_seen[named_snapshot.turn_id] = 100.0
    run._first_seen[hidden_snapshot.turn_id] = 100.0

    run._stage_submission(named_envelope, _decision())
    run._stage_submission(hidden_envelope, _decision())

    assert client.moves == [("named", {"decision": "accept"})]
    assert hidden_snapshot.turn_id in run._ready
    events = [json.loads(line) for line in run.events_path.read_text(encoding="utf-8").splitlines()]
    named_release = next(event for event in events if event["kind"] == "move_delay_released" and event["game_id"] == "named")
    assert named_release["concealment"]["release_reason"] == "identity-scope-disabled"
    assert named_release["concealment"]["scope_applied"] is False
    run.broker.close()
    run.opponent_timing.close()
