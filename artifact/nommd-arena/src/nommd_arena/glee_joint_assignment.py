"""Deterministic capacity-aware assignment primitives for offline GLEE reconstruction."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence, TypeVar


JOINT_ASSIGNMENT_SOLVER_CONTRACT = "glee-capacity-assignment-solver-v1"

DemandKey = TypeVar("DemandKey", bound=Hashable)
ResourceKey = TypeVar("ResourceKey", bound=Hashable)
SlotValue = TypeVar("SlotValue")
DemandValue = TypeVar("DemandValue")


@dataclass(frozen=True)
class AssignmentOption:
    """One resource available to one demand with a finite log-utility-like score."""

    resource_id: str
    utility: float


@dataclass(frozen=True)
class AssignmentDemand:
    """One unit demand, its candidate resources, and its private unmatched option."""

    demand_id: str
    options: tuple[AssignmentOption, ...]
    unmatched_utility: float


@dataclass(frozen=True)
class CapacityAssignment:
    """One globally feasible assignment with each public resource used at most to capacity."""

    selected: dict[str, str | None]
    used_capacity: dict[str, int]
    total_utility: float

    @property
    def unmatched(self) -> tuple[str, ...]:
        return tuple(sorted(demand_id for demand_id, resource_id in self.selected.items() if resource_id is None))


@dataclass(frozen=True)
class AssignmentMarginals:
    """Perturb-and-MAP assignment frequencies plus the unperturbed optimum."""

    map_assignment: CapacityAssignment
    probabilities: dict[str, dict[str | None, float]]
    samples: int
    temperature: float


@dataclass
class _Edge:
    destination: int
    reverse: int
    capacity: int
    cost: int
    demand_id: str | None = None
    resource_id: str | None = None


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validated_demands(demands: Iterable[AssignmentDemand], capacities: dict[str, int]) -> tuple[AssignmentDemand, ...]:
    normalized: list[AssignmentDemand] = []
    demand_ids: set[str] = set()
    for demand in demands:
        if not demand.demand_id or demand.demand_id in demand_ids:
            raise ValueError(f"demand IDs must be nonempty and unique: {demand.demand_id!r}")
        demand_ids.add(demand.demand_id)
        options: list[AssignmentOption] = []
        option_ids: set[str] = set()
        for option in demand.options:
            if option.resource_id not in capacities:
                raise ValueError(f"unknown assignment resource: {option.resource_id}")
            if option.resource_id in option_ids:
                raise ValueError(f"duplicate resource {option.resource_id!r} for demand {demand.demand_id!r}")
            option_ids.add(option.resource_id)
            options.append(AssignmentOption(option.resource_id, _finite(option.utility, label="option utility")))
        normalized.append(AssignmentDemand(demand.demand_id, tuple(sorted(options, key=lambda option: option.resource_id)), _finite(demand.unmatched_utility, label="unmatched utility")))
    for resource_id, capacity in capacities.items():
        if not resource_id or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError(f"capacity must be a nonnegative integer: {resource_id!r}={capacity!r}")
    return tuple(sorted(normalized, key=lambda demand: demand.demand_id))


def _add_edge(graph: list[list[_Edge]], source: int, destination: int, capacity: int, cost: int, *, demand_id: str | None = None, resource_id: str | None = None) -> int:
    forward_index = len(graph[source])
    reverse_index = len(graph[destination])
    graph[source].append(_Edge(destination, reverse_index, capacity, cost, demand_id, resource_id))
    graph[destination].append(_Edge(source, forward_index, 0, -cost))
    return forward_index


def solve_capacity_assignment(demands: Iterable[AssignmentDemand], capacities: dict[str, int], *, cost_scale: int = 1_000_000) -> CapacityAssignment:
    """Return the maximum-utility feasible assignment, with a private unmatched edge for every demand."""

    if isinstance(cost_scale, bool) or not isinstance(cost_scale, int) or cost_scale < 1:
        raise ValueError("cost_scale must be a positive integer")
    normalized = _validated_demands(demands, capacities)
    if not normalized:
        return CapacityAssignment({}, {}, 0.0)
    resource_ids = sorted({option.resource_id for demand in normalized for option in demand.options if capacities[option.resource_id] > 0})
    source = 0
    demand_offset = 1
    resource_offset = demand_offset + len(normalized)
    sink = resource_offset + len(resource_ids)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    resource_nodes = {resource_id: resource_offset + index for index, resource_id in enumerate(resource_ids)}
    selection_edges: dict[tuple[str, str | None], tuple[int, int]] = {}
    utilities: dict[tuple[str, str | None], float] = {}
    for index, demand in enumerate(normalized):
        node = demand_offset + index
        _add_edge(graph, source, node, 1, 0)
        maximum = max([demand.unmatched_utility, *(option.utility for option in demand.options if capacities[option.resource_id] > 0)])
        unmatched_cost = round((maximum - demand.unmatched_utility) * cost_scale)
        edge_index = _add_edge(graph, node, sink, 1, unmatched_cost, demand_id=demand.demand_id)
        selection_edges[(demand.demand_id, None)] = (node, edge_index)
        utilities[(demand.demand_id, None)] = demand.unmatched_utility
        for option in demand.options:
            if capacities[option.resource_id] == 0:
                continue
            option_cost = round((maximum - option.utility) * cost_scale)
            edge_index = _add_edge(graph, node, resource_nodes[option.resource_id], 1, option_cost, demand_id=demand.demand_id, resource_id=option.resource_id)
            selection_edges[(demand.demand_id, option.resource_id)] = (node, edge_index)
            utilities[(demand.demand_id, option.resource_id)] = option.utility
    for resource_id, node in resource_nodes.items():
        _add_edge(graph, node, sink, capacities[resource_id], 0)
    potential = [0] * len(graph)
    flow = 0
    while flow < len(normalized):
        distance = [math.inf] * len(graph)
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        queue: list[tuple[float, int]] = [(0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.destination]
                candidate = current_distance + reduced
                if candidate < distance[edge.destination]:
                    distance[edge.destination] = candidate
                    parent[edge.destination] = (node, edge_index)
                    heapq.heappush(queue, (candidate, edge.destination))
        if parent[sink] is None:
            raise RuntimeError("private unmatched edges should make every demand assignable")
        for node, value in enumerate(distance):
            if math.isfinite(value):
                potential[node] += int(value)
        node = sink
        while node != source:
            prior, edge_index = parent[node] or (-1, -1)
            if prior < 0:
                raise RuntimeError("broken min-cost-flow predecessor chain")
            edge = graph[prior][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior
        flow += 1
    selected: dict[str, str | None] = {}
    for key, (node, edge_index) in selection_edges.items():
        if graph[node][edge_index].capacity == 0:
            demand_id, resource_id = key
            if demand_id in selected:
                raise RuntimeError(f"demand assigned more than once: {demand_id}")
            selected[demand_id] = resource_id
    if len(selected) != len(normalized):
        raise RuntimeError("assignment result does not cover every demand")
    used = Counter(resource_id for resource_id in selected.values() if resource_id is not None)
    for resource_id, count in used.items():
        if count > capacities[resource_id]:
            raise RuntimeError(f"assignment capacity exceeded: {resource_id}")
    total_utility = sum(utilities[(demand_id, resource_id)] for demand_id, resource_id in selected.items())
    return CapacityAssignment(dict(sorted(selected.items())), dict(sorted(used.items())), total_utility)


def solve_capacity_assignment_auction(demands: Iterable[AssignmentDemand], capacities: dict[str, int], *, epsilon: float = 0.0025, maximum_bids: int = 10_000_000) -> CapacityAssignment:
    """Return a deterministic epsilon-auction assignment for a large sparse capacitated component."""

    epsilon = _finite(epsilon, label="auction epsilon")
    if epsilon <= 0 or isinstance(maximum_bids, bool) or not isinstance(maximum_bids, int) or maximum_bids < 1:
        raise ValueError("auction controls are invalid")
    normalized = _validated_demands(demands, capacities)
    if not normalized:
        return CapacityAssignment({}, {}, 0.0)
    demand_index = {demand.demand_id: index for index, demand in enumerate(normalized)}
    demand_options = [{option.resource_id: option.utility for option in demand.options if capacities[option.resource_id] > 0} for demand in normalized]
    interested: Counter[str] = Counter(option.resource_id for demand in normalized for option in demand.options if capacities[option.resource_id] > 0)
    slots: list[str] = []
    resource_slots: dict[str, list[int]] = {}
    for resource_id in sorted(interested):
        count = min(capacities[resource_id], interested[resource_id])
        resource_slots[resource_id] = list(range(len(slots), len(slots) + count))
        slots.extend([resource_id] * count)
    prices = [0.0] * len(slots)
    owners: list[int | None] = [None] * len(slots)
    selected_slot: list[int | None] = [None] * len(normalized)
    selected_unknown = [False] * len(normalized)
    queue = list(reversed(range(len(normalized))))
    bids = 0
    while queue:
        index = queue.pop()
        demand = normalized[index]
        choices: list[tuple[float, int, int | None]] = [(demand.unmatched_utility, 1, None)]
        for resource_id, utility in demand_options[index].items():
            for slot in resource_slots[resource_id]:
                choices.append((utility - prices[slot], 0, slot))
        choices.sort(key=lambda choice: (-choice[0], choice[1], choice[2] if choice[2] is not None else len(slots)))
        best_net, _kind, best_slot = choices[0]
        if best_slot is None:
            selected_unknown[index] = True
            continue
        second_net = choices[1][0] if len(choices) > 1 else demand.unmatched_utility
        prices[best_slot] += max(epsilon, best_net - second_net + epsilon)
        displaced = owners[best_slot]
        owners[best_slot] = index
        selected_slot[index] = best_slot
        selected_unknown[index] = False
        if displaced is not None:
            selected_slot[displaced] = None
            selected_unknown[displaced] = False
            queue.append(displaced)
        bids += 1
        if bids > maximum_bids:
            raise RuntimeError(f"capacity auction exceeded {maximum_bids} bids")
    selected: dict[str, str | None] = {}
    used: Counter[str] = Counter()
    total_utility = 0.0
    for demand in normalized:
        index = demand_index[demand.demand_id]
        slot = selected_slot[index]
        if slot is None:
            if not selected_unknown[index]:
                raise RuntimeError(f"auction left demand unassigned: {demand.demand_id}")
            resource_id = None
            utility = demand.unmatched_utility
        else:
            resource_id = slots[slot]
            utility = demand_options[index][resource_id]
            used[resource_id] += 1
        selected[demand.demand_id] = resource_id
        total_utility += utility
    for resource_id, count in used.items():
        if count > capacities[resource_id]:
            raise RuntimeError(f"auction capacity exceeded: {resource_id}")
    return CapacityAssignment(dict(sorted(selected.items())), dict(sorted(used.items())), total_utility)


def _gumbel(seed: str, sample: int, demand_id: str, resource_id: str | None) -> float:
    payload = f"{seed}\0{sample}\0{demand_id}\0{resource_id or '<unknown>'}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    uniform = (integer + 0.5) / (2**64)
    return -math.log(-math.log(uniform))


def sample_capacity_marginals(demands: Iterable[AssignmentDemand], capacities: dict[str, int], *, samples: int = 32, temperature: float = 0.35, seed: str = JOINT_ASSIGNMENT_SOLVER_CONTRACT, solver: Callable[[Iterable[AssignmentDemand], dict[str, int]], CapacityAssignment] = solve_capacity_assignment) -> AssignmentMarginals:
    """Approximate structured assignment marginals with deterministic edge-perturbed MAP solves."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    temperature = _finite(temperature, label="temperature")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = _validated_demands(demands, capacities)
    map_assignment = solver(normalized, capacities)
    counts: dict[str, Counter[str | None]] = {demand.demand_id: Counter() for demand in normalized}
    for sample in range(samples):
        perturbed = [
            AssignmentDemand(
                demand.demand_id,
                tuple(AssignmentOption(option.resource_id, option.utility + temperature * _gumbel(seed, sample, demand.demand_id, option.resource_id)) for option in demand.options),
                demand.unmatched_utility + temperature * _gumbel(seed, sample, demand.demand_id, None),
            )
            for demand in normalized
        ]
        assignment = solver(perturbed, capacities)
        for demand_id, resource_id in assignment.selected.items():
            counts[demand_id][resource_id] += 1
    probabilities = {
        demand_id: dict(sorted(((resource_id, count / samples) for resource_id, count in values.items()), key=lambda row: (row[0] is None, row[0] or "")))
        for demand_id, values in sorted(counts.items())
    }
    return AssignmentMarginals(map_assignment, probabilities, samples, temperature)


@dataclass(frozen=True)
class MonotoneAlignment:
    """Order-preserving demand-to-slot alignment with explicit unmatched demands and unused slots."""

    demand_to_slot: tuple[int | None, ...]
    unused_slots: tuple[int, ...]
    total_cost: float


def align_monotone(demands: Sequence[DemandValue], slots: Sequence[SlotValue], match_cost: Callable[[DemandValue, SlotValue], float | None], *, unmatched_demand_cost: float, unused_slot_cost: float) -> MonotoneAlignment:
    """Compute a minimum-cost sequence alignment without reusing or reordering capacity slots."""

    unmatched_demand_cost = _finite(unmatched_demand_cost, label="unmatched demand cost")
    unused_slot_cost = _finite(unused_slot_cost, label="unused slot cost")
    if unmatched_demand_cost < 0 or unused_slot_cost < 0:
        raise ValueError("sequence-alignment gap costs must be nonnegative")
    demand_count = len(demands)
    slot_count = len(slots)
    decisions = [bytearray(slot_count + 1) for _ in range(demand_count + 1)]
    previous = [index * unused_slot_cost for index in range(slot_count + 1)]
    for slot_index in range(1, slot_count + 1):
        decisions[0][slot_index] = 3
    for demand_index in range(1, demand_count + 1):
        current = [demand_index * unmatched_demand_cost] + [math.inf] * slot_count
        decisions[demand_index][0] = 2
        for slot_index in range(1, slot_count + 1):
            cost = match_cost(demands[demand_index - 1], slots[slot_index - 1])
            choices = [
                (previous[slot_index] + unmatched_demand_cost, 1, 2),
                (current[slot_index - 1] + unused_slot_cost, 2, 3),
            ]
            if cost is not None:
                cost = _finite(cost, label="match cost")
                if cost < 0:
                    raise ValueError("sequence-alignment match costs must be nonnegative")
                choices.append((previous[slot_index - 1] + cost, 0, 1))
            selected_cost, _priority, decision = min(choices)
            current[slot_index] = selected_cost
            decisions[demand_index][slot_index] = decision
        previous = current
    demand_to_slot: list[int | None] = [None] * demand_count
    unused_slots: list[int] = []
    demand_index = demand_count
    slot_index = slot_count
    while demand_index > 0 or slot_index > 0:
        decision = decisions[demand_index][slot_index]
        if decision == 1:
            demand_to_slot[demand_index - 1] = slot_index - 1
            demand_index -= 1
            slot_index -= 1
        elif decision == 2:
            demand_index -= 1
        elif decision == 3:
            unused_slots.append(slot_index - 1)
            slot_index -= 1
        else:
            raise RuntimeError(f"broken monotone-alignment traceback at ({demand_index}, {slot_index})")
    return MonotoneAlignment(tuple(demand_to_slot), tuple(reversed(unused_slots)), previous[-1])
