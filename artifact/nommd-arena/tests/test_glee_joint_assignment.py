from dataclasses import dataclass

import pytest

from nommd_arena.glee_joint_assignment import AssignmentDemand, AssignmentOption, align_monotone, sample_capacity_marginals, solve_capacity_assignment, solve_capacity_assignment_auction


def test_joint_assignment_uses_global_capacity_instead_of_greedy_reuse() -> None:
    demands = [
        AssignmentDemand("g-1", (AssignmentOption("event-a", 10.0), AssignmentOption("event-b", 9.0)), -10.0),
        AssignmentDemand("g-2", (AssignmentOption("event-a", 8.0),), -10.0),
    ]

    result = solve_capacity_assignment(demands, {"event-a": 1, "event-b": 1})

    assert result.selected == {"g-1": "event-b", "g-2": "event-a"}
    assert result.used_capacity == {"event-a": 1, "event-b": 1}
    assert result.total_utility == pytest.approx(17.0)


def test_joint_assignment_preserves_explicit_unmatched_mass() -> None:
    demands = [
        AssignmentDemand("g-1", (AssignmentOption("event-a", 1.0),), -2.0),
        AssignmentDemand("g-2", (AssignmentOption("event-a", 0.5),), 0.0),
    ]

    result = solve_capacity_assignment(demands, {"event-a": 1})

    assert result.selected == {"g-1": "event-a", "g-2": None}
    assert result.unmatched == ("g-2",)


def test_joint_assignment_supports_aggregate_public_capacity() -> None:
    demands = [AssignmentDemand(f"g-{index}", (AssignmentOption("aggregate", 1.0),), -5.0) for index in range(3)]

    result = solve_capacity_assignment(demands, {"aggregate": 2})

    assert sum(resource == "aggregate" for resource in result.selected.values()) == 2
    assert len(result.unmatched) == 1


def test_auction_matches_exact_solver_on_small_conflict() -> None:
    demands = [
        AssignmentDemand("g-1", (AssignmentOption("event-a", 10.0), AssignmentOption("event-b", 9.0)), -10.0),
        AssignmentDemand("g-2", (AssignmentOption("event-a", 8.0), AssignmentOption("event-b", 1.0)), -10.0),
        AssignmentDemand("g-3", (AssignmentOption("event-b", 7.0),), 0.0),
    ]
    capacities = {"event-a": 1, "event-b": 1}

    exact = solve_capacity_assignment(demands, capacities)
    auction = solve_capacity_assignment_auction(demands, capacities)

    assert auction.selected == exact.selected
    assert auction.total_utility == pytest.approx(exact.total_utility)
    assert auction.used_capacity == exact.used_capacity


def test_perturbed_marginals_are_deterministic_and_capacity_feasible() -> None:
    demands = [
        AssignmentDemand("g-1", (AssignmentOption("event-a", 1.0), AssignmentOption("event-b", 1.0)), -4.0),
        AssignmentDemand("g-2", (AssignmentOption("event-a", 1.0), AssignmentOption("event-b", 1.0)), -4.0),
    ]

    first = sample_capacity_marginals(demands, {"event-a": 1, "event-b": 1}, samples=24, seed="fixture")
    second = sample_capacity_marginals(demands, {"event-a": 1, "event-b": 1}, samples=24, seed="fixture")

    assert first == second
    assert first.map_assignment.used_capacity == {"event-a": 1, "event-b": 1}
    assert all(sum(probabilities.values()) == pytest.approx(1.0) for probabilities in first.probabilities.values())


@dataclass(frozen=True)
class _Timed:
    timestamp: int


def test_monotone_alignment_recovers_a_non_greedy_capacity_assignment() -> None:
    demands = [_Timed(10), _Timed(11)]
    slots = [_Timed(9), _Timed(10)]

    result = align_monotone(demands, slots, lambda demand, slot: abs(demand.timestamp - slot.timestamp), unmatched_demand_cost=5.0, unused_slot_cost=5.0)

    assert result.demand_to_slot == (0, 1)
    assert result.unused_slots == ()
    assert result.total_cost == pytest.approx(2.0)


def test_monotone_alignment_never_forces_an_incompatible_match() -> None:
    demands = [_Timed(10), _Timed(100)]
    slots = [_Timed(9)]

    result = align_monotone(demands, slots, lambda demand, slot: abs(demand.timestamp - slot.timestamp) if abs(demand.timestamp - slot.timestamp) <= 5 else None, unmatched_demand_cost=3.0, unused_slot_cost=2.0)

    assert result.demand_to_slot == (0, None)
    assert result.unused_slots == ()
