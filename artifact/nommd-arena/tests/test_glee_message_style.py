import hashlib
import json
from pathlib import Path

from nommd_arena.glee_message_style import MessageStylePolicyStore, opponent_message_policy_signal, realize_message_style, sample_timing_persona_target
from nommd_arena.glee_parallel import ParallelGleeRun
from nommd_arena.glee_persuasion_twin_v2 import classify_persuasion_signal
from nommd_arena.glee_tactics import GlobalTacticLedger
from nommd_arena.glee_worker import TurnEnvelope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = PROJECT_ROOT / "policies" / "message-style"
CANDIDATE_RELEASE = POLICY_ROOT / "releases" / "v2-joint-hi-persona.json"


class _FixedRandom:
    def betavariate(self, _alpha: float, _beta: float) -> float:
        return 0.7

    def uniform(self, _lower: float, _upper: float) -> float:
        return 0.0


def _candidate_policy_root(tmp_path: Path) -> Path:
    root = tmp_path / "message-style-policy"
    release = root / "releases" / CANDIDATE_RELEASE.name
    release.parent.mkdir(parents=True)
    contents = CANDIDATE_RELEASE.read_bytes()
    release.write_bytes(contents)
    pointer = {"schema_version": 1, "contract": "glee-message-style-policy-pointer-v1", "release": f"releases/{release.name}", "release_sha256": hashlib.sha256(contents).hexdigest()}
    (root / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    return root


def _bargaining_game(game_id: str, *, hidden: bool = True) -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "proposal",
        "prompt": "test",
        "opponent": {"type": "hidden" if hidden else "agent", "name": None if hidden else "Known"},
        "game_state": {"current_player": "player_1", "messages_allowed": True, "money_to_divide": 100.0, "round": 1},
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}},
    }


def _persuasion_game(game_id: str) -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "player_1",
        "opponent": {"type": "hidden", "name": None},
        "game_state": {"current_player": "player_1", "player_1_role": "seller", "product_price": 12.0, "round": 1, "seller_message_type": "text"},
        "valid_actions": {"type": "seller_message", "fields": {"message": "string"}},
    }


def test_hidden_profile_is_anti_aligned_with_first_move_and_pinned_for_the_game(tmp_path: Path) -> None:
    store = MessageStylePolicyStore(root=POLICY_ROOT, assignments_path=tmp_path / "assignments.jsonl")
    game = _bargaining_game("hidden-bargaining")
    hard = {"alice_gain": 80.0, "bob_gain": 20.0, "message": "This is my firm position."}
    profile, created, error = store.profile_for_action(game, hard)
    assert created is True and error is None and profile is not None
    assert profile["economic_style"] == "hard"
    assert profile["profile_id"] == "conversational-comma"
    styled, receipt = realize_message_style(game, hard, profile)
    assert {key: value for key, value in styled.items() if key != "message"} == {key: value for key, value in hard.items() if key != "message"}
    assert styled["message"] != hard["message"]
    assert receipt["nonmessage_action_unchanged"] is True
    soft = {"alice_gain": 20.0, "bob_gain": 80.0, "message": "This is fair to both sides."}
    pinned, created, error = store.profile_for_action(game, soft)
    assert created is False and error is None and pinned == profile
    restarted = MessageStylePolicyStore(root=POLICY_ROOT, assignments_path=tmp_path / "assignments.jsonl")
    assert restarted.assigned_profile(game) == profile


def test_known_game_keeps_native_strategic_message(tmp_path: Path) -> None:
    store = MessageStylePolicyStore(root=POLICY_ROOT, assignments_path=tmp_path / "assignments.jsonl")
    game = _bargaining_game("known-bargaining", hidden=False)
    action = {"alice_gain": 80.0, "bob_gain": 20.0, "message": "This is my firm position."}
    profile, created, error = store.profile_for_action(game, action)
    assert created is True and error is None and profile is not None and profile["profile_id"] == "native-strategic"
    styled, receipt = realize_message_style(game, action, profile)
    assert styled == action
    assert receipt["message_changed"] is False


def test_absent_optional_message_pins_the_game_profile_without_injecting_prose(tmp_path: Path) -> None:
    store = MessageStylePolicyStore(root=POLICY_ROOT, assignments_path=tmp_path / "assignments.jsonl")
    game = _bargaining_game("silent-bargaining")
    action = {"alice_gain": 60.0, "bob_gain": 40.0}
    profile, created, error = store.profile_for_action(game, action)
    assert profile is not None and created is True and error is None
    styled, receipt = realize_message_style(game, action, profile)
    assert styled == action
    assert receipt["status"] == "not-applicable"
    assert (tmp_path / "assignments.jsonl").is_file()


def test_hidden_joint_persona_pins_one_game_speed_quantile_and_lexical_profile(tmp_path: Path) -> None:
    random_source = _FixedRandom()
    store = MessageStylePolicyStore(root=_candidate_policy_root(tmp_path), assignments_path=tmp_path / "assignments.jsonl", random_source=random_source)
    game = _bargaining_game("joint-hidden")
    action = {"alice_gain": 80.0, "bob_gain": 20.0}
    profile, created, error = store.profile_for_action(game, action)
    assert created is True and error is None and profile is not None
    assert profile["profile_id"] == "conversational-comma"
    persona = profile["timing_persona"]
    assert persona["activation"] == "authoritative"
    assert persona["timing_profile_id"] == "hi-conversational-mixed"
    assert persona["game_speed_quantile"] == 0.7
    target, receipt = sample_timing_persona_target(game, profile, random_source)
    assert target is not None and 28.0 < target < 38.0
    assert receipt["game_speed_quantile"] == 0.7
    pinned, created, error = store.profile_for_action(game, {"alice_gain": 20.0, "bob_gain": 80.0, "message": "changed"})
    assert created is False and error is None and pinned == profile


def test_candidate_joint_policy_keeps_known_identity_on_ki_timing_and_native_lexical_style(tmp_path: Path) -> None:
    random_source = _FixedRandom()
    store = MessageStylePolicyStore(root=_candidate_policy_root(tmp_path), assignments_path=tmp_path / "known-assignments.jsonl", random_source=random_source)
    game = _bargaining_game("joint-known", hidden=False)
    action = {"alice_gain": 80.0, "bob_gain": 20.0, "message": "This is my firm position."}
    profile, created, error = store.profile_for_action(game, action)
    assert created is True and error is None and profile is not None
    assert profile["profile_id"] == "native-strategic"
    assert profile["timing_persona"] == {"contract": "glee-hi-timing-persona-v1", "activation": "ki-stable", "timing_profile_id": "ki-beta-2-2-v1"}
    target, receipt = sample_timing_persona_target(game, profile, random_source)
    assert target is None and receipt["status"] == "ki-stable"


def test_persuasion_profile_stays_pinned_without_freezing_recommendation_polarity(tmp_path: Path) -> None:
    store = MessageStylePolicyStore(root=POLICY_ROOT, assignments_path=tmp_path / "assignments.jsonl")
    game = _persuasion_game("persuasion-polarity")
    positive = {"message": "I recommend accepting this offer."}
    profile, created, error = store.profile_for_action(game, positive)
    assert created is True and error is None and profile is not None and profile["economic_style"] == "positive"
    positive_styled, _receipt = realize_message_style(game, positive, profile)
    assert classify_persuasion_signal(positive_styled["message"], channel="text")[0] == "positive"
    game["game_state"]["round"] = 2
    negative = {"message": "I recommend declining this offer."}
    pinned, created, error = store.profile_for_action(game, negative)
    assert created is False and error is None and pinned == profile
    negative_styled, receipt = realize_message_style(game, negative, pinned)
    assert classify_persuasion_signal(negative_styled["message"], channel="text")[0] == "negative"
    assert receipt["assigned_economic_style"] == "positive"
    assert receipt["current_economic_style"] == "negative"


def test_identity_free_message_signal_uses_opponent_language_without_claiming_identity() -> None:
    game = _bargaining_game("message-signal")
    game["game_state"]["history"] = [
        {"round": 1, "offer": {"round": 1, "proposer": "player_2", "player_1_gain": 30.0, "player_2_gain": 70.0, "message": "This is a detailed proposal, and I believe it gives us a reasonable basis for agreement."}, "decision": "reject"}
    ]
    signal = opponent_message_policy_signal(game)
    assert signal is not None
    assert signal["contract"] == "glee-identity-free-message-policy-signal-v1"
    assert signal["lexical_style"]["comma"] == 1.0
    assert set(signal["forecast_shifts"]) == {"next_proposal_hardness", "next_response_acceptance"}
    assert "identity" in signal["authority"]
    assert "detailed proposal" not in str(signal)


def test_parallel_supervisor_realizes_style_before_submission_preparation(tmp_path: Path) -> None:
    class Client:
        def stats(self) -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

    run = ParallelGleeRun(project_root=PROJECT_ROOT, run_dir=tmp_path / "run", env_file=None, model="test", effort="high", max_parallel=1, max_games=1, families=("bargaining",), client=Client(), worker=object(), global_tactic_ledger=GlobalTacticLedger(), message_style_policy_root=POLICY_ROOT)
    try:
        game = _bargaining_game("supervisor-style")
        snapshot = run.broker.observe_turn(game)
        envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=10**20)
        action = {"alice_gain": 80.0, "bob_gain": 20.0, "message": "This is my firm position."}
        styled, profile = run._realize_message_style(envelope, action, stage="test")
        assert profile is not None and profile["profile_id"] == "conversational-comma"
        assert styled["alice_gain"] == action["alice_gain"] and styled["bob_gain"] == action["bob_gain"]
        assert styled["message"] != action["message"]
        assert run._mode() == "glee-parallel-v32"
    finally:
        run.broker.close()
