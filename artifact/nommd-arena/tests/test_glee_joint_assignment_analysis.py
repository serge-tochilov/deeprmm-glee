from dataclasses import replace

from nommd_arena.glee_activity_eda import LocalGame
from nommd_arena.glee_joint_assignment_analysis import CandidateEdge, GameCandidates, PublicEvent, SelfAlignment, align_self_games, assign_with_marginals, evaluate_assignment


def _game(game_id: str, completed_at: float) -> LocalGame:
    return LocalGame(game_id, "bargaining", completed_at - 1.0, completed_at, 1.0, "hidden", None, f"run/games/bargaining-{game_id}.json", game_id, False)


def _event(event_id: str, *, player_id: str, frontier: int, start: float, end: float, capacity: int = 1) -> PublicEvent:
    return PublicEvent(event_id, int(event_id.removeprefix("e")), frontier, "bargaining", player_id, start, end, capacity, -1.0, "clean-single", 10.0)


def test_self_alignment_expands_capacity_without_reusing_slots() -> None:
    games = [_game("g1", 11.0), _game("g2", 12.0), _game("g3", 40.0)]
    events = [_event("e1", player_id="self", frontier=2, start=10.0, end=20.0, capacity=2)]

    alignments, summary = align_self_games(games, events, alignment_slack_s=5.0, observation_quantum_s=10.0)

    assert [(row.self_event.event_id if row.self_event else None, row.self_slot_index) for row in alignments] == [("e1", 0), ("e1", 1), (None, None)]
    assert summary["aligned_games"] == 2
    assert summary["unmatched_games"] == 1
    assert summary["capacity_violations"] == 0


def test_self_alignment_does_not_impose_false_publication_order() -> None:
    games = [_game("g1", 11.0), _game("g2", 19.0)]
    events = [
        _event("e1", player_id="self", frontier=2, start=18.0, end=20.0),
        _event("e2", player_id="self", frontier=3, start=10.0, end=12.0),
    ]

    alignments, summary = align_self_games(games, events, alignment_slack_s=2.0, observation_quantum_s=10.0)

    assert [(row.game.game_id, row.self_event.event_id if row.self_event else None) for row in alignments] == [("g1", "e2"), ("g2", "e1")]
    assert summary["aligned_games"] == 2
    assert summary["unused_public_capacity"] == 0


def test_self_alignment_uses_authenticated_rating_delta_to_resolve_overlap() -> None:
    games = [replace(_game("g1", 11.0), rating_delta=5.0), replace(_game("g2", 12.0), rating_delta=-3.0)]
    events = [replace(_event("e1", player_id="self", frontier=2, start=10.0, end=20.0), clean_rating_delta=-3.0), replace(_event("e2", player_id="self", frontier=2, start=10.0, end=20.0), clean_rating_delta=5.0)]

    alignments, summary = align_self_games(games, events, alignment_slack_s=5.0, observation_quantum_s=10.0)

    assert [(row.game.game_id, row.self_event.event_id if row.self_event else None) for row in alignments] == [("g1", "e2"), ("g2", "e1")]
    assert summary["rating_delta_mismatch_max"] == 0.0
    assert summary["rating_delta_mismatch_gt_0_11"] == 0


def test_assignment_analysis_resolves_shared_capacity_globally() -> None:
    event_a = _event("e1", player_id="id-a", frontier=2, start=10.0, end=20.0)
    event_b = _event("e2", player_id="id-b", frontier=2, start=10.0, end=20.0)
    first_alignment = SelfAlignment(_game("g1", 12.0), _event("e3", player_id="self", frontier=2, start=10.0, end=20.0), 0, "interval-contained", 0.0)
    second_alignment = SelfAlignment(_game("g2", 13.0), replace(first_alignment.self_event, event_id="e4", source_change_sequence=4), 0, "interval-contained", 0.0)
    first = GameCandidates(first_alignment, 1, ("id-b",), (CandidateEdge(event_a, 0.0, 0, 0.0, 10.0, 0.0), CandidateEdge(event_b, 0.0, 0, 0.0, 9.0, 0.0)), 0.01, 0)
    second = GameCandidates(second_alignment, 1, ("id-a",), (CandidateEdge(event_a, 0.0, 0, 0.0, 8.0, 0.0),), 0.01, 0)

    marginal, assignment_summary = assign_with_marginals([first, second], model="activity-only", samples=16, temperature=0.35, seed="fixture")
    evaluation, _rows = evaluate_assignment([first, second], marginal, {"e1": event_a, "e2": event_b})

    assert marginal.map_assignment.selected == {"g1": "e2", "g2": "e1"}
    assert assignment_summary["map_capacity_violations"] == 0
    assert evaluation["candidate_coverage"] == 1.0
    assert evaluation["top1_accuracy"] == 1.0


def test_evidence_conditioning_preserves_a_collision_set() -> None:
    event_a = _event("e1", player_id="id-a", frontier=2, start=10.0, end=20.0)
    event_b = _event("e2", player_id="id-b", frontier=2, start=10.0, end=20.0)
    game = replace(_game("g1", 12.0), identity_scope="known", opponent_name="RESERVE")
    alignment = SelfAlignment(game, _event("e3", player_id="self", frontier=2, start=10.0, end=20.0), 0, "interval-contained", 0.0)
    row = GameCandidates(alignment, 1, ("id-a", "id-b"), (CandidateEdge(event_a, 0.0, 0, 0.0, 1.0, 0.0), CandidateEdge(event_b, 0.0, 0, 0.0, 0.5, 0.0)), 0.01, 0)

    marginal, _summary = assign_with_marginals([row], model="evidence-conditioned", samples=8, temperature=0.35, seed="fixture")
    evaluation, _rows = evaluate_assignment([row], marginal, {"e1": event_a, "e2": event_b})

    assert set(marginal.probabilities["g1"]) == {"e1", "e2", None}
    assert evaluation["collision_set_games"] == 1
    assert evaluation["collision_candidate_coverage"] == 1.0
