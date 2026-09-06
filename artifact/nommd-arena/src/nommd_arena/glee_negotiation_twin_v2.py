"""Causal, shadow-only executable opponent model for GLEE Negotiation v2.0."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


SCHEMA_VERSION = 1
ENGINE_VERSION = "v2.0"
MODEL_VERSION = f"negotiation-twin-{ENGINE_VERSION}-development"
MESSAGE_ACTS = ("none", "price", "urgency", "fairness", "conditional", "walkaway", "commitment", "other")
_SQRT_TWO = math.sqrt(2.0)
_PRICE_NUMBER = re.compile(r"(?:\$\s*)?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _other_player(player: str) -> str:
    try:
        return {"player_1": "player_2", "player_2": "player_1"}[player]
    except KeyError as error:
        raise ValueError(f"unsupported player identity: {player}") from error


def classify_negotiation_message(message: str, *, messages_allowed: bool) -> str:
    """Map visible prose to a small deterministic act vocabulary without interpreting truth."""
    if not messages_allowed or not message.strip():
        return "none"
    lowered = message.casefold()
    if any(term in lowered for term in ("walk away", "walkaway", "no deal", "leave", "withdraw")):
        return "walkaway"
    if any(term in lowered for term in ("if you", "provided", "conditional", "in return", "then i")):
        return "conditional"
    if any(term in lowered for term in ("final", "last", "now", "immediately", "deadline", "time", "today")):
        return "urgency"
    if any(term in lowered for term in ("fair", "equitable", "reasonable", "balanced")):
        return "fairness"
    if any(term in lowered for term in ("accept", "agree", "close", "settle", "deal", "commit")):
        return "commitment"
    if _PRICE_NUMBER.search(message):
        return "price"
    return "other"


def opponent_demand(price: float, *, opponent_role: str, our_value: float) -> float:
    """Return a role-invariant log price coordinate where larger values favor the opponent."""
    if opponent_role not in {"buyer", "seller"}:
        raise ValueError(f"unsupported negotiation role: {opponent_role}")
    checked_price = _finite_nonnegative(price, "price")
    checked_value = _finite_positive(our_value, "our_value")
    ratio = max(checked_price / checked_value, 1e-12)
    direction = 1.0 if opponent_role == "seller" else -1.0
    return direction * math.log(ratio)


def price_from_opponent_demand(demand: float, *, opponent_role: str, our_value: float) -> float:
    """Invert the role-invariant opponent-demand coordinate."""
    if not math.isfinite(float(demand)):
        raise ValueError("demand must be finite")
    if opponent_role not in {"buyer", "seller"}:
        raise ValueError(f"unsupported negotiation role: {opponent_role}")
    direction = 1.0 if opponent_role == "seller" else -1.0
    return _finite_positive(our_value, "our_value") * math.exp(direction * float(demand))


def opponent_surplus_share(price: float, *, opponent_role: str, our_role: str, our_value: float, opponent_value: float | None) -> float | None:
    """Return the opponent's complete-information surplus share, including out-of-range offers."""
    if opponent_value is None:
        return None
    checked_price = _finite_nonnegative(price, "price")
    checked_our_value = _finite_positive(our_value, "our_value")
    checked_opponent_value = _finite_positive(opponent_value, "opponent_value")
    if {opponent_role, our_role} != {"buyer", "seller"}:
        raise ValueError("negotiation roles must contain one buyer and one seller")
    buyer_value = checked_opponent_value if opponent_role == "buyer" else checked_our_value
    seller_value = checked_opponent_value if opponent_role == "seller" else checked_our_value
    total_surplus = buyer_value - seller_value
    if total_surplus <= 0:
        return None
    opponent_surplus = buyer_value - checked_price if opponent_role == "buyer" else checked_price - seller_value
    return opponent_surplus / total_surplus


@dataclass(frozen=True)
class NegotiationContext:
    """Authenticated state visible immediately before one opponent action."""

    game_id: str
    opponent_id: str
    opponent_name: str
    completed_at: str
    completion_order: int
    our_player: str
    opponent_player: str
    our_role: str
    opponent_role: str
    our_value: float
    opponent_value: float | None
    complete_information: bool
    horizon_known: bool
    max_rounds: int | None
    messages_allowed: bool
    round_number: int
    observed_opponent_actions: int
    previous_opponent_demand: float | None
    previous_our_demand: float | None
    previous_opponent_response: str | None
    previous_our_response: str | None
    current_offer_demand: float | None
    current_offer_opponent_surplus_share: float | None
    current_offer_message_act: str

    @property
    def round_phase(self) -> float:
        if self.horizon_known and self.max_rounds is not None and self.max_rounds > 1:
            return min(1.0, max(0.0, (self.round_number - 1) / (self.max_rounds - 1)))
        if self.horizon_known and self.max_rounds == 1:
            return 1.0
        return 1.0 - math.exp(-max(0, self.round_number - 1) / 12.0)

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.completed_at, self.completion_order, self.game_id)


@dataclass(frozen=True)
class NegotiationDecisionRow:
    """One authenticated action attributed to the modeled opponent."""

    context: NegotiationContext
    action_type: Literal["proposal", "response"]
    proposal_demand: float | None = None
    proposal_price_ratio: float | None = None
    proposal_opponent_surplus_share: float | None = None
    offered_demand: float | None = None
    offered_opponent_surplus_share: float | None = None
    accepted: bool | None = None
    decision: str | None = None
    message: str = ""
    message_act: str = "none"
    response_time_ms: float | None = None
    job_id: str = ""
    job_path: str = ""
    job_sha256: str = ""

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["context"]["round_phase"] = self.context.round_phase
        return value


@dataclass(frozen=True)
class NegotiationGameEvidence:
    """One immutable completed game and its opponent-attributed actions."""

    game_id: str
    opponent_id: str
    opponent_name: str
    completed_at: str
    completion_order: int
    job_id: str
    job_path: str
    job_sha256: str
    final_game_sha256: str
    outcome: str
    rows: tuple[NegotiationDecisionRow, ...]

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.completed_at, self.completion_order, self.game_id)


def _history_context(
    *,
    game_id: str,
    opponent_id: str,
    opponent_name: str,
    completed_at: str,
    completion_order: int,
    state: dict[str, Any],
    our_player: str,
    opponent_player: str,
    round_number: int,
    prior: Sequence[dict[str, Any]],
    current_offer: dict[str, Any] | None,
) -> NegotiationContext:
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    if {our_role, opponent_role} != {"buyer", "seller"}:
        raise ValueError("negotiation state must contain one buyer and one seller")
    our_value = _finite_positive(state.get(f"{our_player}_value"), f"{our_player}_value")
    complete_information = state.get("complete_information") is True
    raw_opponent_value = state.get(f"{opponent_player}_value") if complete_information else None
    opponent_value = _finite_positive(raw_opponent_value, f"{opponent_player}_value") if raw_opponent_value is not None else None
    previous_opponent_demand = None
    previous_our_demand = None
    previous_opponent_response = None
    previous_our_response = None
    for entry in reversed(prior):
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else {}
        from_player = str(offer.get("from_player") or "")
        decision = str(entry.get("decision") or "") or None
        price = offer.get("price")
        if price is None:
            continue
        demand = opponent_demand(float(price), opponent_role=opponent_role, our_value=our_value)
        if from_player == opponent_player and previous_opponent_demand is None:
            previous_opponent_demand = demand
            previous_our_response = decision
        elif from_player == our_player and previous_our_demand is None:
            previous_our_demand = demand
            previous_opponent_response = decision
        if previous_opponent_demand is not None and previous_our_demand is not None:
            break
    maximum = state.get("max_rounds")
    max_rounds = maximum if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 else None
    current_offer_demand = None
    current_offer_share = None
    current_offer_message_act = "none"
    if current_offer is not None:
        current_price = _finite_nonnegative(current_offer.get("price"), "current_offer.price")
        current_offer_demand = opponent_demand(current_price, opponent_role=opponent_role, our_value=our_value)
        current_offer_share = opponent_surplus_share(current_price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
        current_offer_message_act = classify_negotiation_message(str(current_offer.get("message") or ""), messages_allowed=state.get("messages_allowed") is not False)
    return NegotiationContext(
        game_id=game_id,
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        completed_at=completed_at,
        completion_order=completion_order,
        our_player=our_player,
        opponent_player=opponent_player,
        our_role=our_role,
        opponent_role=opponent_role,
        our_value=our_value,
        opponent_value=opponent_value,
        complete_information=complete_information,
        horizon_known=state.get("horizon_known") is True,
        max_rounds=max_rounds,
        messages_allowed=state.get("messages_allowed") is not False,
        round_number=round_number,
        observed_opponent_actions=len(prior),
        previous_opponent_demand=previous_opponent_demand,
        previous_our_demand=previous_our_demand,
        previous_opponent_response=previous_opponent_response,
        previous_our_response=previous_our_response,
        current_offer_demand=current_offer_demand,
        current_offer_opponent_surplus_share=current_offer_share,
        current_offer_message_act=current_offer_message_act,
    )


def extract_negotiation_game(job: dict[str, Any], *, job_path: Path, job_sha256: str | None = None) -> NegotiationGameEvidence:
    """Extract causally ordered opponent actions from one immutable named-opponent job."""
    if job.get("game_family") != "negotiation":
        raise ValueError("job is not a negotiation game")
    final_game = job.get("final_game")
    if not isinstance(final_game, dict):
        raise ValueError("job has no final_game object")
    state = final_game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("final game has no game_state object")
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("final negotiation state has no history list")
    our_player = str(final_game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    opponent = job.get("opponent") if isinstance(job.get("opponent"), dict) else final_game.get("opponent")
    if not isinstance(opponent, dict):
        raise ValueError("job has no named opponent")
    opponent_id = str(opponent.get("id") or "")
    opponent_name = str(opponent.get("name") or "")
    if not opponent_id or not opponent_name:
        raise ValueError("job has incomplete opponent identity")
    game_id = str(job.get("game_id") or final_game.get("game_id") or "")
    completed_at = str(job.get("completed_at") or "")
    completion_order = int(job.get("completion_order") or 0)
    job_id = str(job.get("job_id") or "")
    actual_job_sha = job_sha256 or _sha_file(job_path)
    expected_final_sha = str(job.get("final_game_sha256") or "")
    if expected_final_sha and _sha(final_game) != expected_final_sha:
        raise ValueError(f"final game SHA-256 mismatch in {job_path}")
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    our_value = _finite_positive(state.get(f"{our_player}_value"), f"{our_player}_value")
    complete_information = state.get("complete_information") is True
    raw_opponent_value = state.get(f"{opponent_player}_value") if complete_information else None
    visible_opponent_value = _finite_positive(raw_opponent_value, f"{opponent_player}_value") if raw_opponent_value is not None else None
    rows: list[NegotiationDecisionRow] = []
    prior: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError(f"malformed negotiation history entry in {job_path}")
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        if offer is None:
            raise ValueError(f"negotiation history entry has no offer in {job_path}")
        from_player = str(offer.get("from_player") or "")
        decided_by = str(entry.get("decided_by") or "")
        if from_player not in {our_player, opponent_player} or decided_by != _other_player(from_player):
            raise ValueError(f"negotiation history has inconsistent actors in {job_path}")
        round_number = int(entry.get("round") or offer.get("round") or len(prior) + 1)
        current_offer = offer if decided_by == opponent_player else None
        context = _history_context(
            game_id=game_id,
            opponent_id=opponent_id,
            opponent_name=opponent_name,
            completed_at=completed_at,
            completion_order=completion_order,
            state=state,
            our_player=our_player,
            opponent_player=opponent_player,
            round_number=round_number,
            prior=prior,
            current_offer=current_offer,
        )
        price = _finite_nonnegative(offer.get("price"), "offer.price")
        demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
        share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=visible_opponent_value)
        decision = str(entry.get("decision") or "")
        message = str(offer.get("message") or "")
        if from_player == opponent_player:
            rows.append(
                NegotiationDecisionRow(
                    context=context,
                    action_type="proposal",
                    proposal_demand=demand,
                    proposal_price_ratio=price / our_value,
                    proposal_opponent_surplus_share=share,
                    message=message,
                    message_act=classify_negotiation_message(message, messages_allowed=context.messages_allowed),
                    job_id=job_id,
                    job_path=str(job_path),
                    job_sha256=actual_job_sha,
                )
            )
        else:
            if decision not in {"AcceptOffer", "RejectOffer", "WalkAway"}:
                raise ValueError(f"unsupported negotiation response {decision!r} in {job_path}")
            response_time = entry.get("response_time_ms")
            rows.append(
                NegotiationDecisionRow(
                    context=context,
                    action_type="response",
                    offered_demand=demand,
                    offered_opponent_surplus_share=share,
                    accepted=decision == "AcceptOffer",
                    decision=decision,
                    response_time_ms=float(response_time) if isinstance(response_time, (int, float)) and not isinstance(response_time, bool) else None,
                    job_id=job_id,
                    job_path=str(job_path),
                    job_sha256=actual_job_sha,
                )
            )
        prior.append(entry)
    result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
    return NegotiationGameEvidence(
        game_id=game_id,
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        completed_at=completed_at,
        completion_order=completion_order,
        job_id=job_id,
        job_path=str(job_path),
        job_sha256=actual_job_sha,
        final_game_sha256=expected_final_sha,
        outcome=str(result.get("outcome") or ""),
        rows=tuple(rows),
    )


def load_negotiation_corpus(dossier_root: Path, *, project_root: Path | None = None) -> tuple[tuple[NegotiationGameEvidence, ...], tuple[dict[str, str], ...]]:
    """Load clean named Negotiation jobs without reading model-authored dossier prose."""
    from .glee_incremental_dossier import IncrementalNamedOpponentStore

    resolved_dossier_root = dossier_root.resolve()
    inferred_project_root = resolved_dossier_root.parent.parent if resolved_dossier_root.parent.name == "opponent-dossiers" else resolved_dossier_root.parent
    store = IncrementalNamedOpponentStore(root=resolved_dossier_root, project_root=(project_root or inferred_project_root).resolve())
    jobs_root = dossier_root / "jobs"
    paths = sorted(jobs_root.glob("*/*.json")) if jobs_root.is_dir() else []
    games: list[NegotiationGameEvidence] = []
    rejected: list[dict[str, str]] = []
    seen: dict[str, NegotiationGameEvidence] = {}
    for path in paths:
        try:
            job = _read_json(path)
            if job.get("game_family") != "negotiation":
                continue
            final_game = job.get("final_game") if isinstance(job.get("final_game"), dict) else {}
            if isinstance(job.get("final_game_ref"), dict):
                final_game = store.load_job_final_game(job)
            result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
            terminal = str(final_game.get("status") or "").casefold()
            outcome = str(result.get("outcome") or "").casefold()
            if terminal in {"timeout", "cancelled", "abandoned"} or outcome in {"timeout", "cancelled", "abandoned"}:
                rejected.append({"path": str(path), "reason": "censored_terminal_state"})
                continue
            hydrated_job = {**job, "final_game": final_game}
            game = extract_negotiation_game(hydrated_job, job_path=path)
            previous = seen.get(game.game_id)
            if previous is not None:
                if previous.final_game_sha256 != game.final_game_sha256 or previous.opponent_id != game.opponent_id:
                    raise RuntimeError(f"conflicting immutable jobs for game {game.game_id}: {previous.job_path} and {path}")
                rejected.append({"path": str(path), "reason": f"duplicate_game:{previous.job_path}"})
                continue
            seen[game.game_id] = game
            games.append(game)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
    games.sort(key=lambda game: game.sort_key)
    return tuple(games), tuple(rejected)


@dataclass(frozen=True)
class PriorObservation:
    """One causally prior opponent action with global and target game indices."""

    row: NegotiationDecisionRow
    global_game_index: int
    opponent_game_index: int


@dataclass(frozen=True)
class NegotiationModelConfig:
    """Frozen initial settings for partial pooling, local modes, and prefix adaptation."""

    population_game_decay: float = 0.997
    target_game_decay: float = 0.9
    same_game_decay: float = 0.92
    population_equivalent_rows: float = 4.0
    target_population_equivalent_rows: float = 2.0
    same_game_strength: float = 2.0
    response_base_probability: float = 0.15
    response_base_equivalent_rows: float = 2.0
    response_demand_bandwidth: float = 0.22
    proposal_kernel_sigma: float = 0.07
    proposal_floor_sigma: float = 0.7
    proposal_floor_equivalent_rows: float = 0.35

    def validate(self) -> None:
        for name in ("population_game_decay", "target_game_decay", "same_game_decay"):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("population_equivalent_rows", "target_population_equivalent_rows", "same_game_strength", "response_base_equivalent_rows", "response_demand_bandwidth", "proposal_kernel_sigma", "proposal_floor_sigma", "proposal_floor_equivalent_rows"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.response_base_probability < 1:
            raise ValueError("response_base_probability must lie in (0, 1)")


@dataclass(frozen=True)
class GaussianComponent:
    """One weighted component in opponent-demand space."""

    mean: float
    sigma: float
    weight: float
    source: str


class GaussianMixture:
    """Dependency-free Gaussian mixture retaining repeated negotiation price modes."""

    def __init__(self, components: Sequence[GaussianComponent]) -> None:
        selected = [component for component in components if component.weight > 0 and component.sigma > 0 and math.isfinite(component.mean) and math.isfinite(component.sigma) and math.isfinite(component.weight)]
        if not selected:
            raise ValueError("Gaussian mixture has no valid components")
        total = sum(component.weight for component in selected)
        coalesced: dict[tuple[float, float], float] = {}
        for component in selected:
            key = (round(component.mean, 12), component.sigma)
            coalesced[key] = coalesced.get(key, 0.0) + component.weight
        self.components = tuple(GaussianComponent(mean, sigma, weight / total, "coalesced") for (mean, sigma), weight in coalesced.items())
        self.raw_component_count = len(selected)

    @property
    def mean(self) -> float:
        return sum(component.weight * component.mean for component in self.components)

    @property
    def sigma(self) -> float:
        center = self.mean
        variance = sum(component.weight * (component.sigma**2 + component.mean**2) for component in self.components) - center**2
        return max(1e-6, math.sqrt(max(0.0, variance)))

    def density(self, observed: float) -> float:
        return sum(component.weight * math.exp(-0.5 * ((observed - component.mean) / component.sigma) ** 2) / (component.sigma * math.sqrt(2 * math.pi)) for component in self.components)

    def nll(self, observed: float) -> float:
        return -math.log(max(self.density(observed), 1e-300))

    def cdf(self, value: float) -> float:
        return sum(component.weight * 0.5 * (1.0 + math.erf((value - component.mean) / (component.sigma * _SQRT_TWO))) for component in self.components)

    def quantile(self, probability: float) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("quantile probability must lie in [0, 1]")
        low = min(component.mean - 8 * component.sigma for component in self.components)
        high = max(component.mean + 8 * component.sigma for component in self.components)
        for _iteration in range(56):
            middle = (low + high) / 2
            if self.cdf(middle) < probability:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "sigma": self.sigma, "q10": self.quantile(0.1), "q50": self.quantile(0.5), "q90": self.quantile(0.9), "component_count": len(self.components), "raw_component_count": self.raw_component_count}


def _optional_distance(current: float | None, previous: float | None, scale: float, missing_factor: float) -> float:
    if current is None and previous is None:
        return 1.0
    if current is None or previous is None:
        return missing_factor
    return math.exp(-abs(current - previous) / scale)


def context_similarity(current: NegotiationContext, previous: NegotiationContext, *, action_type: str) -> float:
    """Compare only state visible before both actions; the current opponent message is excluded."""
    value = 1.0
    value *= 1.0 if current.opponent_role == previous.opponent_role else 0.2
    value *= 1.0 if current.complete_information == previous.complete_information else 0.48
    value *= 1.0 if current.horizon_known == previous.horizon_known else 0.62
    value *= 1.0 if current.messages_allowed == previous.messages_allowed else 0.85
    value *= math.exp(-abs(current.round_phase - previous.round_phase) / 0.38)
    value *= math.exp(-0.08 * abs(math.log(current.our_value / previous.our_value)))
    if current.horizon_known and previous.horizon_known:
        if current.max_rounds is None or previous.max_rounds is None:
            value *= 0.7
        else:
            value *= math.exp(-abs(math.log(current.max_rounds / previous.max_rounds)) / 1.2)
    value *= _optional_distance(current.previous_opponent_demand, previous.previous_opponent_demand, 0.4, 0.72)
    value *= _optional_distance(current.previous_our_demand, previous.previous_our_demand, 0.4, 0.72)
    if current.previous_opponent_response != previous.previous_opponent_response:
        value *= 0.78
    if action_type == "response":
        value *= _optional_distance(current.current_offer_demand, previous.current_offer_demand, 0.22, 0.3)
        if current.current_offer_message_act != previous.current_offer_message_act:
            value *= 0.88
    return max(value, 1e-12)


def _scaled_weights(values: Sequence[tuple[NegotiationDecisionRow, float]], mass: float) -> list[tuple[NegotiationDecisionRow, float]]:
    total = sum(weight for _row, weight in values)
    if total <= 0:
        return []
    return [(row, weight * mass / total) for row, weight in values]


class NegotiationOpponentModelV2:
    """Forecast responses and proposals from structured prior behavior and the visible game prefix."""

    def __init__(self, config: NegotiationModelConfig | None = None) -> None:
        self.config = config or NegotiationModelConfig()
        self.config.validate()

    def _prior_weights(
        self,
        current: NegotiationDecisionRow,
        observations: Sequence[PriorObservation],
        *,
        current_global_game_index: int,
        current_target_game_index: int,
        scope: Literal["population", "target"],
    ) -> list[tuple[NegotiationDecisionRow, float]]:
        selected: list[tuple[NegotiationDecisionRow, float]] = []
        for observation in observations:
            previous = observation.row
            if previous.action_type != current.action_type:
                continue
            if scope == "population":
                age = max(0, current_global_game_index - observation.global_game_index - 1)
                recency = self.config.population_game_decay**age
            else:
                age = max(0, current_target_game_index - observation.opponent_game_index - 1)
                recency = self.config.target_game_decay**age
            similarity = context_similarity(current.context, previous.context, action_type=current.action_type)
            selected.append((previous, recency * similarity))
        return selected

    def _prefix_weights(self, current: NegotiationDecisionRow, prefix: Sequence[NegotiationDecisionRow]) -> list[tuple[NegotiationDecisionRow, float]]:
        selected: list[tuple[NegotiationDecisionRow, float]] = []
        matching = [row for row in prefix if row.action_type == current.action_type]
        for index, previous in enumerate(reversed(matching)):
            recency = self.config.same_game_decay**index
            similarity = context_similarity(current.context, previous.context, action_type=current.action_type)
            selected.append((previous, self.config.same_game_strength * recency * similarity))
        return selected

    def response_forecast(
        self,
        row: NegotiationDecisionRow,
        *,
        population_prior: Sequence[PriorObservation],
        target_prior: Sequence[PriorObservation],
        prefix: Sequence[NegotiationDecisionRow],
        current_global_game_index: int,
        current_target_game_index: int,
    ) -> dict[str, dict[str, float]]:
        if row.action_type != "response" or row.accepted is None:
            raise ValueError("response forecast requires a response row")
        population = _scaled_weights(
            self._prior_weights(row, population_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="population"),
            self.config.population_equivalent_rows,
        )
        target = self._prior_weights(row, target_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="target")
        prefix_rows = self._prefix_weights(row, prefix)

        def probability(weighted: Sequence[tuple[NegotiationDecisionRow, float]], *, prior_probability: float, prior_mass: float) -> tuple[float, float]:
            numerator = prior_mass * prior_probability + sum(weight * float(bool(previous.accepted)) for previous, weight in weighted)
            denominator = prior_mass + sum(weight for _previous, weight in weighted)
            return min(1 - 1e-6, max(1e-6, numerator / denominator)), denominator

        population_probability, population_mass = probability(population, prior_probability=self.config.response_base_probability, prior_mass=self.config.response_base_equivalent_rows)
        target_probability, target_mass = probability(target, prior_probability=population_probability, prior_mass=self.config.target_population_equivalent_rows)
        adaptive_probability, adaptive_mass = probability(prefix_rows, prior_probability=target_probability, prior_mass=max(self.config.target_population_equivalent_rows, target_mass))
        return {
            "population_kernel": {"probability": population_probability, "effective_mass": population_mass},
            "target_kernel": {"probability": target_probability, "effective_mass": target_mass},
            "adaptive_v2": {"probability": adaptive_probability, "effective_mass": adaptive_mass},
        }

    def _proposal_components(self, weighted: Sequence[tuple[NegotiationDecisionRow, float]], source: str) -> list[GaussianComponent]:
        return [GaussianComponent(float(row.proposal_demand), self.config.proposal_kernel_sigma, weight, source) for row, weight in weighted if row.proposal_demand is not None and weight > 0]

    def _proposal_mixture(self, components: Sequence[GaussianComponent]) -> GaussianMixture:
        evidence = sum(component.weight for component in components)
        center = sum(component.weight * component.mean for component in components) / evidence if evidence else 0.0
        floor_mass = self.config.proposal_floor_equivalent_rows * max(1.0, math.sqrt(evidence))
        return GaussianMixture((*components, GaussianComponent(center, self.config.proposal_floor_sigma, floor_mass, "broad-floor")))

    def proposal_forecast(
        self,
        row: NegotiationDecisionRow,
        *,
        population_prior: Sequence[PriorObservation],
        target_prior: Sequence[PriorObservation],
        prefix: Sequence[NegotiationDecisionRow],
        current_global_game_index: int,
        current_target_game_index: int,
    ) -> dict[str, GaussianMixture]:
        if row.action_type != "proposal" or row.proposal_demand is None:
            raise ValueError("proposal forecast requires a proposal row")
        population = _scaled_weights(
            self._prior_weights(row, population_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="population"),
            self.config.population_equivalent_rows,
        )
        target = self._prior_weights(row, target_prior, current_global_game_index=current_global_game_index, current_target_game_index=current_target_game_index, scope="target")
        prefix_rows = self._prefix_weights(row, prefix)
        population_components = self._proposal_components(population, "population")
        target_components = self._proposal_components(target, "target")
        prefix_components = self._proposal_components(prefix_rows, "same-game-prefix")
        return {
            "population_kernel": self._proposal_mixture(population_components),
            "target_kernel": self._proposal_mixture((*population_components, *target_components)),
            "adaptive_v2": self._proposal_mixture((*population_components, *target_components, *prefix_components)),
        }


def rows(games: Iterable[NegotiationGameEvidence]) -> list[NegotiationDecisionRow]:
    """Flatten complete games while preserving their existing order."""
    return [row for game in games for row in game.rows]


def prior_observations(games: Sequence[NegotiationGameEvidence]) -> tuple[list[PriorObservation], dict[str, int]]:
    """Build deterministic row provenance for a chronologically ordered game prefix."""
    target_indices: dict[str, int] = {}
    observations: list[PriorObservation] = []
    for global_index, game in enumerate(games):
        opponent_index = target_indices.get(game.opponent_id, 0)
        observations.extend(PriorObservation(row=row, global_game_index=global_index, opponent_game_index=opponent_index) for row in game.rows)
        target_indices[game.opponent_id] = opponent_index + 1
    return observations, target_indices


def model_receipt(config: NegotiationModelConfig) -> dict[str, object]:
    """Return the stable model identity embedded in validation and future seeds."""
    return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "model_version": MODEL_VERSION, "message_vocabulary": list(MESSAGE_ACTS), "config": asdict(config), "config_sha256": _sha(asdict(config))}
