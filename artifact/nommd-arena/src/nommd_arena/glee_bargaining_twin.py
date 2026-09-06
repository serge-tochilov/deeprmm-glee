"""Shadow-only executable opponent models for GLEE bargaining."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


SCHEMA_VERSION = 1
MODEL_VERSION = "bargaining-twin-v0"
MESSAGE_ACTS = ("none", "authority", "allocation", "fairness-urgency", "fairness", "urgency", "commitment", "other")
RESPONSE_CURVE_SHARES = (0.25, 0.3, 1 / 3, 0.4, 0.45, 0.5, 0.55, 0.6, 2 / 3, 0.7, 0.75)
_EPSILON = 1e-9
_SQRT_TWO = math.sqrt(2.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float = 0.001, high: float = 0.999) -> float:
    return min(high, max(low, value))


def _other_player(player: str) -> str:
    try:
        return {"player_1": "player_2", "player_2": "player_1"}[player]
    except KeyError as error:
        raise ValueError(f"unsupported player identity: {player}") from error


def _gain(offer: dict[str, Any], player: str) -> float:
    aliases = {"player_1": ("player_1_gain", "alice_gain"), "player_2": ("player_2_gain", "bob_gain")}
    for key in aliases[player]:
        number = _finite(offer.get(key))
        if number is not None:
            return number
    raise ValueError(f"offer does not identify {player}'s gain")


def classify_message_act(message: str, *, messages_allowed: bool) -> str:
    """Map free text to a small, deterministic strategic-act vocabulary."""
    if not messages_allowed or not message.strip():
        return "none"
    lowered = message.casefold()
    if any(term in lowered for term in ("rubinstein", "subgame", "equilibrium")):
        return "authority"
    allocation = any(character.isdigit() for character in message) and any(term in lowered for term in ("/", "%", "alice", "bob", "split", "share"))
    if allocation:
        return "allocation"
    fairness = any(term in lowered for term in ("fair", "equal", "equitable", "balanced"))
    urgency = any(term in lowered for term in ("now", "close", "final", "time", "round", "today", "quick"))
    if fairness and urgency:
        return "fairness-urgency"
    if fairness:
        return "fairness"
    if urgency:
        return "urgency"
    if any(term in lowered for term in ("accept", "ready", "settle", "commit", "deal")):
        return "commitment"
    return "other"


@dataclass(frozen=True)
class BargainingContext:
    """Visible state immediately before one opponent action."""

    game_id: str
    opponent_id: str
    opponent_name: str
    completed_at: str
    completion_order: int
    our_player: str
    opponent_player: str
    round_number: int
    money_to_divide: float
    complete_information: bool
    horizon_known: bool
    max_rounds: int | None
    messages_allowed: bool
    our_discount: float | None
    opponent_discount: float | None
    previous_opponent_offer_share: float | None
    previous_our_offer_to_opponent_share: float | None
    previous_opponent_response: str | None
    previous_our_response: str | None

    @property
    def progress(self) -> float:
        if self.horizon_known and self.max_rounds is not None and self.max_rounds > 1:
            return _clamp((self.round_number - 1) / (self.max_rounds - 1), 0.0, 1.0)
        return min(1.0, max(0.0, (self.round_number - 1) / 5.0))

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.completed_at, self.completion_order, self.game_id)


@dataclass(frozen=True)
class BargainingDecisionRow:
    """One authenticated action attributed to the modeled opponent."""

    context: BargainingContext
    action_type: Literal["proposal", "response"]
    proposal_share: float | None = None
    offered_share: float | None = None
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
        value["context"]["progress"] = self.context.progress
        return value


@dataclass(frozen=True)
class BargainingGameEvidence:
    """One immutable game job and its opponent-attributed decisions."""

    game_id: str
    opponent_id: str
    opponent_name: str
    completed_at: str
    completion_order: int
    job_id: str
    job_path: str
    job_sha256: str
    final_game_sha256: str
    rows: tuple[BargainingDecisionRow, ...]

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.completed_at, self.completion_order, self.game_id)


@dataclass(frozen=True)
class ResponseParticle:
    """A probabilistic acceptance script with bounded contextual adjustments."""

    threshold: float
    temperature: float
    deadline_slope: float
    complete_offset: float
    player_1_offset: float
    equality_logit: float

    @property
    def identifier(self) -> str:
        return _sha({"kind": "response", **asdict(self)})[:16]

    @property
    def complexity(self) -> int:
        return sum(abs(value) > _EPSILON for value in (self.deadline_slope, self.complete_offset, self.player_1_offset, self.equality_logit))

    def probability(self, context: BargainingContext, offered_share: float) -> float:
        threshold = self.threshold + self.deadline_slope * context.progress
        if context.complete_information:
            threshold += self.complete_offset
        if context.opponent_player == "player_1":
            threshold += self.player_1_offset
        logit = (offered_share - threshold) / self.temperature
        if math.isclose(offered_share, 0.5, rel_tol=0.0, abs_tol=0.005):
            logit += self.equality_logit
        if logit >= 0:
            probability = 1.0 / (1.0 + math.exp(-min(logit, 40.0)))
        else:
            exponential = math.exp(max(logit, -40.0))
            probability = exponential / (1.0 + exponential)
        return _clamp(probability, 1e-6, 1 - 1e-6)


@dataclass(frozen=True)
class ProposalParticle:
    """A stochastic offer script combining an anchor, reaction, and concession rule."""

    anchor: float
    reaction: float
    concession_to_parity: float
    complete_offset: float
    player_1_offset: float
    sigma: float

    @property
    def identifier(self) -> str:
        return _sha({"kind": "proposal", **asdict(self)})[:16]

    @property
    def complexity(self) -> int:
        return sum(abs(value) > _EPSILON for value in (self.reaction, self.concession_to_parity, self.complete_offset, self.player_1_offset))

    def mean(self, context: BargainingContext) -> float:
        value = self.anchor
        if context.complete_information:
            value += self.complete_offset
        if context.opponent_player == "player_1":
            value += self.player_1_offset
        if context.previous_our_offer_to_opponent_share is not None:
            value += self.reaction * (context.previous_our_offer_to_opponent_share - value)
        value += self.concession_to_parity * context.progress * (0.5 - value)
        return _clamp(value)

    def log_likelihood(self, context: BargainingContext, observed_share: float) -> float:
        residual = (observed_share - self.mean(context)) / self.sigma
        return -0.5 * residual * residual - math.log(self.sigma * math.sqrt(2 * math.pi))


@dataclass(frozen=True)
class TwinConfig:
    """Finite candidate grammar and Bayesian regularization settings."""

    response_thresholds: tuple[float, ...] = (0.2, 0.25, 0.3, 1 / 3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 2 / 3, 0.7, 0.75, 0.8)
    response_temperatures: tuple[float, ...] = (0.03, 0.08, 0.16)
    deadline_slopes: tuple[float, ...] = (-0.12, 0.0, 0.12)
    context_offsets: tuple[float, ...] = (-0.05, 0.0, 0.05)
    equality_logits: tuple[float, ...] = (0.0, 1.5)
    proposal_anchors: tuple[float, ...] = (0.2, 0.25, 0.3, 1 / 3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 2 / 3, 0.7, 0.75, 0.8)
    proposal_reactions: tuple[float, ...] = (0.0, 0.5, 1.0)
    proposal_concessions: tuple[float, ...] = (0.0, 0.5, 1.0)
    proposal_sigmas: tuple[float, ...] = (0.03, 0.08, 0.16)
    complexity_penalty: float = 0.35
    population_equivalent_rows: float = 12.0
    uniform_prior_mix: float = 0.05
    artifact_particles: int = 64
    message_prior_equivalent_rows: float = 8.0

    @classmethod
    def small_test_config(cls) -> TwinConfig:
        return cls(
            response_thresholds=(0.35, 0.5, 0.65),
            response_temperatures=(0.04, 0.12),
            deadline_slopes=(0.0,),
            context_offsets=(0.0,),
            equality_logits=(0.0,),
            proposal_anchors=(0.3, 0.5, 0.7),
            proposal_reactions=(0.0, 1.0),
            proposal_concessions=(0.0,),
            proposal_sigmas=(0.04, 0.12),
            artifact_particles=8,
        )


def response_particles(config: TwinConfig) -> tuple[ResponseParticle, ...]:
    return tuple(
        ResponseParticle(threshold, temperature, deadline, complete, player_1, equality)
        for threshold in config.response_thresholds
        for temperature in config.response_temperatures
        for deadline in config.deadline_slopes
        for complete in config.context_offsets
        for player_1 in config.context_offsets
        for equality in config.equality_logits
    )


def proposal_particles(config: TwinConfig) -> tuple[ProposalParticle, ...]:
    return tuple(
        ProposalParticle(anchor, reaction, concession, complete, player_1, sigma)
        for anchor in config.proposal_anchors
        for reaction in config.proposal_reactions
        for concession in config.proposal_concessions
        for complete in config.context_offsets
        for player_1 in config.context_offsets
        for sigma in config.proposal_sigmas
    )


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
) -> BargainingContext:
    previous_opponent_offer_share = None
    previous_our_offer_to_opponent_share = None
    previous_opponent_response = None
    previous_our_response = None
    money = _finite(state.get("money_to_divide"))
    if money is None or money <= 0:
        raise ValueError("bargaining game has invalid money_to_divide")
    for entry in reversed(prior):
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else {}
        proposer = str(entry.get("proposer") or offer.get("proposer") or "")
        decision = str(entry.get("decision") or "").casefold() or None
        if previous_opponent_offer_share is None and proposer == opponent_player:
            previous_opponent_offer_share = _gain(offer, opponent_player) / money
            previous_our_response = decision
        if previous_our_offer_to_opponent_share is None and proposer == our_player:
            previous_our_offer_to_opponent_share = _gain(offer, opponent_player) / money
            previous_opponent_response = decision
        if previous_opponent_offer_share is not None and previous_our_offer_to_opponent_share is not None:
            break
    maximum = state.get("max_rounds")
    max_rounds = maximum if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 else None
    return BargainingContext(
        game_id=game_id,
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        completed_at=completed_at,
        completion_order=completion_order,
        our_player=our_player,
        opponent_player=opponent_player,
        round_number=round_number,
        money_to_divide=money,
        complete_information=state.get("complete_information") is True,
        horizon_known=state.get("horizon_known") is True,
        max_rounds=max_rounds,
        messages_allowed=state.get("messages_allowed") is not False,
        our_discount=_finite(state.get("delta_1" if our_player == "player_1" else "delta_2")),
        opponent_discount=_finite(state.get("delta_1" if opponent_player == "player_1" else "delta_2")),
        previous_opponent_offer_share=previous_opponent_offer_share,
        previous_our_offer_to_opponent_share=previous_our_offer_to_opponent_share,
        previous_opponent_response=previous_opponent_response,
        previous_our_response=previous_our_response,
    )


def extract_bargaining_game(job: dict[str, Any], *, job_path: Path, job_sha256: str | None = None) -> BargainingGameEvidence:
    """Extract exact opponent actions from one immutable named-opponent job."""
    if job.get("game_family") != "bargaining":
        raise ValueError("job is not a bargaining game")
    final_game = job.get("final_game")
    if not isinstance(final_game, dict):
        raise ValueError("job has no final_game object")
    state = final_game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("final game has no game_state object")
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("final bargaining state has no history list")
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
    rows: list[BargainingDecisionRow] = []
    prior: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError(f"malformed bargaining history entry in {job_path}")
        offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else None
        if offer is None:
            raise ValueError(f"bargaining history entry has no offer in {job_path}")
        proposer = str(entry.get("proposer") or offer.get("proposer") or "")
        if proposer not in {our_player, opponent_player}:
            raise ValueError(f"bargaining history has unknown proposer in {job_path}")
        round_number = int(entry.get("round") or offer.get("round") or len(prior) + 1)
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
        )
        message = str(offer.get("message") or "")
        response_time = _finite(entry.get("response_time_ms"))
        decision = str(entry.get("decision") or "").casefold()
        if proposer == opponent_player:
            rows.append(
                BargainingDecisionRow(
                    context=context,
                    action_type="proposal",
                    proposal_share=_clamp(_gain(offer, opponent_player) / context.money_to_divide, 0.0, 1.0),
                    message=message,
                    message_act=classify_message_act(message, messages_allowed=context.messages_allowed),
                    job_id=job_id,
                    job_path=str(job_path),
                    job_sha256=actual_job_sha,
                )
            )
        else:
            if decision not in {"accept", "reject", "walkaway"}:
                raise ValueError(f"unsupported bargaining response {decision!r} in {job_path}")
            rows.append(
                BargainingDecisionRow(
                    context=context,
                    action_type="response",
                    offered_share=_clamp(_gain(offer, opponent_player) / context.money_to_divide),
                    accepted=decision == "accept",
                    decision=decision,
                    response_time_ms=response_time,
                    job_id=job_id,
                    job_path=str(job_path),
                    job_sha256=actual_job_sha,
                )
            )
        prior.append(entry)
    return BargainingGameEvidence(
        game_id=game_id,
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        completed_at=completed_at,
        completion_order=completion_order,
        job_id=job_id,
        job_path=str(job_path),
        job_sha256=actual_job_sha,
        final_game_sha256=expected_final_sha,
        rows=tuple(rows),
    )


def load_bargaining_corpus(dossier_root: Path, *, project_root: Path | None = None) -> tuple[tuple[BargainingGameEvidence, ...], tuple[dict[str, str], ...]]:
    """Load an immutable snapshot of current bargaining jobs without reading prose dossiers."""
    from .glee_incremental_dossier import IncrementalNamedOpponentStore

    resolved_dossier_root = dossier_root.resolve()
    inferred_project_root = resolved_dossier_root.parent.parent if resolved_dossier_root.parent.name == "opponent-dossiers" else resolved_dossier_root.parent
    store = IncrementalNamedOpponentStore(root=resolved_dossier_root, project_root=(project_root or inferred_project_root).resolve())
    jobs_root = dossier_root / "jobs"
    paths = sorted(jobs_root.glob("*/*.json")) if jobs_root.is_dir() else []
    games: list[BargainingGameEvidence] = []
    rejected: list[dict[str, str]] = []
    seen_games: dict[str, BargainingGameEvidence] = {}
    for path in paths:
        try:
            job = _read_json(path)
            if job.get("game_family") != "bargaining":
                continue
            final_game = job.get("final_game") if isinstance(job.get("final_game"), dict) else {}
            if isinstance(job.get("final_game_ref"), dict):
                final_game = store.load_job_final_game(job)
            result = final_game.get("result") if isinstance(final_game.get("result"), dict) else {}
            if str(final_game.get("status") or "").casefold() in {"timeout", "cancelled", "abandoned"} or str(result.get("outcome") or "").casefold() in {"timeout", "cancelled", "abandoned"}:
                rejected.append({"path": str(path), "reason": "censored_terminal_state"})
                continue
            hydrated_job = {**job, "final_game": final_game}
            game = extract_bargaining_game(hydrated_job, job_path=path)
            previous = seen_games.get(game.game_id)
            if previous is not None:
                if previous.final_game_sha256 != game.final_game_sha256 or previous.opponent_id != game.opponent_id:
                    raise RuntimeError(f"conflicting immutable jobs for game {game.game_id}: {previous.job_path} and {path}")
                rejected.append({"path": str(path), "reason": f"duplicate_game:{previous.job_path}"})
                continue
            seen_games[game.game_id] = game
            games.append(game)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
    games.sort(key=lambda game: game.sort_key)
    return tuple(games), tuple(rejected)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot normalize an empty posterior")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _normalize_log_weights(values: Sequence[float]) -> list[float]:
    normalizer = _logsumexp(values)
    return [math.exp(value - normalizer) for value in values]


def _mixed_population_prior(log_likelihoods: Sequence[float], *, row_count: int, complexities: Sequence[int], config: TwinConfig) -> list[float]:
    scale = max(1.0, row_count / config.population_equivalent_rows)
    scores = [likelihood / scale - config.complexity_penalty * complexity for likelihood, complexity in zip(log_likelihoods, complexities, strict=True)]
    learned = _normalize_log_weights(scores)
    uniform = 1.0 / len(learned)
    return [(1 - config.uniform_prior_mix) * value + config.uniform_prior_mix * uniform for value in learned]


def _response_scores(particles: Sequence[ResponseParticle], rows: Sequence[BargainingDecisionRow]) -> list[float]:
    responses = [row for row in rows if row.action_type == "response" and row.offered_share is not None and row.accepted is not None]
    scores: list[float] = []
    for particle in particles:
        total = 0.0
        for row in responses:
            probability = particle.probability(row.context, float(row.offered_share))
            total += math.log(probability if row.accepted else 1 - probability)
        scores.append(total)
    return scores


def _proposal_scores(particles: Sequence[ProposalParticle], rows: Sequence[BargainingDecisionRow]) -> list[float]:
    proposals = [row for row in rows if row.action_type == "proposal" and row.proposal_share is not None]
    scores: list[float] = []
    for particle in particles:
        scores.append(sum(particle.log_likelihood(row.context, float(row.proposal_share)) for row in proposals))
    return scores


def _posterior_from_prior(prior: Sequence[float], log_likelihoods: Sequence[float]) -> list[float]:
    return _normalize_log_weights([math.log(max(weight, 1e-300)) + likelihood for weight, likelihood in zip(prior, log_likelihoods, strict=True)])


def _weighted_particle_records(particles: Sequence[ResponseParticle | ProposalParticle], weights: Sequence[float], *, limit: int) -> tuple[list[dict[str, object]], float]:
    ranked = sorted(zip(particles, weights, strict=True), key=lambda pair: pair[1], reverse=True)
    selected = ranked[:limit]
    retained_mass = sum(weight for _particle, weight in selected)
    records = [{"id": particle.identifier, "weight": weight / retained_mass, "program": asdict(particle)} for particle, weight in selected]
    return records, retained_mass


def _response_prediction(particles: Sequence[ResponseParticle], weights: Sequence[float], context: BargainingContext, offered_share: float) -> float:
    return sum(weight * particle.probability(context, offered_share) for particle, weight in zip(particles, weights, strict=True))


def _proposal_mixture_mean(particles: Sequence[ProposalParticle], weights: Sequence[float], context: BargainingContext) -> float:
    return sum(weight * particle.mean(context) for particle, weight in zip(particles, weights, strict=True))


def _normal_cdf(value: float, mean: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((value - mean) / (sigma * _SQRT_TWO)))


def _proposal_quantile(particles: Sequence[ProposalParticle], weights: Sequence[float], context: BargainingContext, quantile: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(48):
        middle = (low + high) / 2
        probability = sum(weight * _normal_cdf(middle, particle.mean(context), particle.sigma) for particle, weight in zip(particles, weights, strict=True))
        if probability < quantile:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _proposal_log_likelihood(particles: Sequence[ProposalParticle], weights: Sequence[float], row: BargainingDecisionRow) -> float:
    terms = [math.log(max(weight, 1e-300)) + particle.log_likelihood(row.context, float(row.proposal_share)) for particle, weight in zip(particles, weights, strict=True)]
    return _logsumexp(terms)


def _response_nll(particles: Sequence[ResponseParticle], weights: Sequence[float], rows: Sequence[BargainingDecisionRow]) -> float | None:
    selected = [row for row in rows if row.action_type == "response" and row.offered_share is not None and row.accepted is not None]
    if not selected:
        return None
    total = 0.0
    for row in selected:
        probability = _response_prediction(particles, weights, row.context, float(row.offered_share))
        total += math.log(probability if row.accepted else 1 - probability)
    return -total / len(selected)


def _proposal_nll(particles: Sequence[ProposalParticle], weights: Sequence[float], rows: Sequence[BargainingDecisionRow]) -> float | None:
    selected = [row for row in rows if row.action_type == "proposal" and row.proposal_share is not None]
    return -sum(_proposal_log_likelihood(particles, weights, row) for row in selected) / len(selected) if selected else None


def _response_metrics(particles: Sequence[ResponseParticle], weights: Sequence[float], rows: Sequence[BargainingDecisionRow]) -> dict[str, float | int | None]:
    selected = [row for row in rows if row.action_type == "response" and row.offered_share is not None and row.accepted is not None]
    if not selected:
        return {"count": 0, "nll": None, "brier": None, "accuracy": None}
    probabilities = [_response_prediction(particles, weights, row.context, float(row.offered_share)) for row in selected]
    labels = [1.0 if row.accepted else 0.0 for row in selected]
    nll = -sum(math.log(probability if label else 1 - probability) for probability, label in zip(probabilities, labels, strict=True)) / len(selected)
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(selected)
    accuracy = sum((probability >= 0.5) == bool(label) for probability, label in zip(probabilities, labels, strict=True)) / len(selected)
    return {"count": len(selected), "nll": nll, "brier": brier, "accuracy": accuracy}


def _proposal_metrics(particles: Sequence[ProposalParticle], weights: Sequence[float], rows: Sequence[BargainingDecisionRow]) -> dict[str, float | int | None]:
    selected = [row for row in rows if row.action_type == "proposal" and row.proposal_share is not None]
    if not selected:
        return {"count": 0, "nll": None, "mae": None, "rmse": None, "interval_80_coverage": None}
    means = [_proposal_mixture_mean(particles, weights, row.context) for row in selected]
    errors = [mean - float(row.proposal_share) for mean, row in zip(means, selected, strict=True)]
    likelihood = _proposal_nll(particles, weights, selected)
    covered = 0
    for row in selected:
        lower = _proposal_quantile(particles, weights, row.context, 0.1)
        upper = _proposal_quantile(particles, weights, row.context, 0.9)
        covered += lower <= float(row.proposal_share) <= upper
    return {
        "count": len(selected),
        "nll": likelihood,
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "interval_80_coverage": covered / len(selected),
    }


def _empirical_baseline(train_rows: Sequence[BargainingDecisionRow], test_rows: Sequence[BargainingDecisionRow]) -> dict[str, dict[str, float | int | None]]:
    train_responses = [row for row in train_rows if row.action_type == "response" and row.accepted is not None]
    test_responses = [row for row in test_rows if row.action_type == "response" and row.accepted is not None]
    acceptance = (sum(bool(row.accepted) for row in train_responses) + 1) / (len(train_responses) + 2)
    if test_responses:
        labels = [1.0 if row.accepted else 0.0 for row in test_responses]
        response = {
            "count": len(labels),
            "nll": -sum(math.log(acceptance if label else 1 - acceptance) for label in labels) / len(labels),
            "brier": sum((acceptance - label) ** 2 for label in labels) / len(labels),
            "accuracy": sum((acceptance >= 0.5) == bool(label) for label in labels) / len(labels),
        }
    else:
        response = {"count": 0, "nll": None, "brier": None, "accuracy": None}
    train_proposals = [float(row.proposal_share) for row in train_rows if row.action_type == "proposal" and row.proposal_share is not None]
    test_proposals = [float(row.proposal_share) for row in test_rows if row.action_type == "proposal" and row.proposal_share is not None]
    mean = sum(train_proposals) / len(train_proposals) if train_proposals else 0.5
    if test_proposals:
        errors = [mean - value for value in test_proposals]
        proposal = {"count": len(errors), "mae": sum(abs(error) for error in errors) / len(errors), "rmse": math.sqrt(sum(error * error for error in errors) / len(errors))}
    else:
        proposal = {"count": 0, "mae": None, "rmse": None}
    return {"response": response, "proposal": proposal}


def _message_model(population_rows: Sequence[BargainingDecisionRow], target_rows: Sequence[BargainingDecisionRow], config: TwinConfig) -> dict[str, object]:
    population_counts = {act: 0 for act in MESSAGE_ACTS}
    target_counts = {act: 0 for act in MESSAGE_ACTS}
    for row in population_rows:
        if row.action_type == "proposal":
            population_counts[row.message_act] += 1
    for row in target_rows:
        if row.action_type == "proposal":
            target_counts[row.message_act] += 1
    population_total = sum(population_counts.values())
    population_probabilities = {act: (population_counts[act] + 1) / (population_total + len(MESSAGE_ACTS)) for act in MESSAGE_ACTS}
    posterior_counts = {act: target_counts[act] + config.message_prior_equivalent_rows * population_probabilities[act] for act in MESSAGE_ACTS}
    denominator = sum(posterior_counts.values())
    probabilities = {act: posterior_counts[act] / denominator for act in MESSAGE_ACTS}
    return {"vocabulary": list(MESSAGE_ACTS), "population_counts": population_counts, "target_counts": target_counts, "probabilities": probabilities}


def _rows(games: Iterable[BargainingGameEvidence]) -> list[BargainingDecisionRow]:
    return [row for game in games for row in game.rows]


def _support(rows: Sequence[BargainingDecisionRow]) -> dict[str, object]:
    contexts = [row.context for row in rows]
    if not contexts:
        return {"row_count": 0}
    return {
        "row_count": len(rows),
        "action_counts": {kind: sum(row.action_type == kind for row in rows) for kind in ("proposal", "response")},
        "complete_information": sorted({context.complete_information for context in contexts}),
        "horizon_known": sorted({context.horizon_known for context in contexts}),
        "opponent_players": sorted({context.opponent_player for context in contexts}),
        "money_to_divide": sorted({context.money_to_divide for context in contexts}),
        "round_range": [min(context.round_number for context in contexts), max(context.round_number for context in contexts)],
        "opponent_discount_range": [
            min((context.opponent_discount for context in contexts if context.opponent_discount is not None), default=None),
            max((context.opponent_discount for context in contexts if context.opponent_discount is not None), default=None),
        ],
    }


def _ood_flags(context: BargainingContext, support: dict[str, Any]) -> list[str]:
    if not support or support.get("row_count") == 0:
        return ["no_target_support"]
    flags: list[str] = []
    if context.complete_information not in support.get("complete_information", []):
        flags.append("unseen_information_regime")
    if context.horizon_known not in support.get("horizon_known", []):
        flags.append("unseen_horizon_regime")
    if context.opponent_player not in support.get("opponent_players", []):
        flags.append("unseen_opponent_role")
    round_range = support.get("round_range")
    if isinstance(round_range, list) and len(round_range) == 2 and context.round_number > int(round_range[1]):
        flags.append("later_round_than_observed")
    money = support.get("money_to_divide", [])
    if money and context.money_to_divide not in money:
        flags.append("unseen_pool_scale")
    return flags


def _drift_status(
    *,
    games: Sequence[BargainingGameEvidence],
    response_programs: Sequence[ResponseParticle],
    proposal_programs: Sequence[ProposalParticle],
    response_population_prior: Sequence[float],
    proposal_population_prior: Sequence[float],
) -> dict[str, object]:
    if len(games) < 6:
        return {"status": "insufficient-evidence", "recent_game_count": 0, "response_excess_nll": None, "proposal_excess_nll": None}
    recent_count = min(3, max(1, len(games) // 4))
    earlier_rows = _rows(games[:-recent_count])
    recent_rows = _rows(games[-recent_count:])
    response_weights = _posterior_from_prior(response_population_prior, _response_scores(response_programs, earlier_rows))
    proposal_weights = _posterior_from_prior(proposal_population_prior, _proposal_scores(proposal_programs, earlier_rows))
    earlier_response_nll = _response_nll(response_programs, response_weights, earlier_rows)
    recent_response_nll = _response_nll(response_programs, response_weights, recent_rows)
    earlier_proposal_nll = _proposal_nll(proposal_programs, proposal_weights, earlier_rows)
    recent_proposal_nll = _proposal_nll(proposal_programs, proposal_weights, recent_rows)

    def excess(recent: float | None, earlier: float | None) -> float | None:
        return recent - earlier if recent is not None and earlier is not None else None

    response_excess = excess(recent_response_nll, earlier_response_nll)
    proposal_excess = excess(recent_proposal_nll, earlier_proposal_nll)
    available = [value for value in (response_excess, proposal_excess) if value is not None]
    maximum = max(available, default=0.0)
    status = "candidate-change" if maximum >= 1.0 else "watch" if maximum >= 0.35 else "stable"
    return {
        "status": status,
        "recent_game_count": recent_count,
        "response_excess_nll": response_excess,
        "proposal_excess_nll": proposal_excess,
        "interpretation": "Heuristic predictive-surprise signal, not a posterior probability of a policy change.",
    }


class BargainingTwin:
    """Load, predict with, and sample from one fitted opponent artifact."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("model_version") != MODEL_VERSION:
            raise ValueError(f"unsupported bargaining twin version: {artifact.get('model_version')}")
        expected_sha = str(artifact.get("artifact_sha256") or "")
        unhashed = dict(artifact)
        unhashed.pop("artifact_sha256", None)
        if not expected_sha or _sha(unhashed) != expected_sha:
            raise ValueError("bargaining twin artifact SHA-256 mismatch")
        self.artifact = artifact
        self.response_programs = tuple(ResponseParticle(**record["program"]) for record in artifact["response_model"]["particles"])
        self.response_weights = tuple(float(record["weight"]) for record in artifact["response_model"]["particles"])
        self.proposal_programs = tuple(ProposalParticle(**record["program"]) for record in artifact["proposal_model"]["particles"])
        self.proposal_weights = tuple(float(record["weight"]) for record in artifact["proposal_model"]["particles"])

    @classmethod
    def from_path(cls, path: Path) -> BargainingTwin:
        return cls(_read_json(path))

    def response_probability(self, context: BargainingContext, offered_share: float) -> float:
        return _response_prediction(self.response_programs, self.response_weights, context, offered_share)

    def proposal_distribution(self, context: BargainingContext) -> dict[str, float]:
        return {
            "mean": _proposal_mixture_mean(self.proposal_programs, self.proposal_weights, context),
            "q10": _proposal_quantile(self.proposal_programs, self.proposal_weights, context, 0.1),
            "q50": _proposal_quantile(self.proposal_programs, self.proposal_weights, context, 0.5),
            "q90": _proposal_quantile(self.proposal_programs, self.proposal_weights, context, 0.9),
        }

    def sample_response(self, context: BargainingContext, offered_share: float, *, seed: int) -> str:
        generator = random.Random(seed)
        return "accept" if generator.random() < self.response_probability(context, offered_share) else "reject"

    def sample_proposal(self, context: BargainingContext, *, seed: int) -> float:
        generator = random.Random(seed)
        index = generator.choices(range(len(self.proposal_programs)), weights=self.proposal_weights, k=1)[0]
        particle = self.proposal_programs[index]
        return _clamp(generator.gauss(particle.mean(context), particle.sigma), 0.0, 1.0)

    def compact_projection(self, context: BargainingContext) -> dict[str, object]:
        response_curve = {f"{share:.6f}": self.response_probability(context, share) for share in RESPONSE_CURVE_SHARES}
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shadow-bargaining-twin-projection",
            "model_version": MODEL_VERSION,
            "opponent": self.artifact["opponent"],
            "training_game_count": self.artifact["training"]["game_count"],
            "response_curve_by_opponent_share": response_curve,
            "proposal_opponent_share": self.proposal_distribution(context),
            "message_act_probabilities": self.artifact["message_model"]["probabilities"],
            "drift": self.artifact["drift"],
            "ood_flags": _ood_flags(context, self.artifact["support"]),
            "use_boundary": "Shadow prediction only; deterministic legality and payoff guards remain authoritative.",
        }


def context_from_live_game(game: dict[str, Any], *, opponent_id: str, opponent_name: str) -> BargainingContext:
    """Construct a model context from one current bargaining turn without mutating it."""
    if game.get("game_family") != "bargaining":
        raise ValueError("live game is not bargaining")
    state = game.get("game_state")
    if not isinstance(state, dict):
        raise ValueError("live game has no state")
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    opponent_player = _other_player(our_player)
    history = state.get("history") if isinstance(state.get("history"), list) else []
    return _history_context(
        game_id=str(game.get("game_id") or "live-shadow"),
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        completed_at="",
        completion_order=0,
        state=state,
        our_player=our_player,
        opponent_player=opponent_player,
        round_number=int(state.get("round") or 1),
        prior=history,
    )


class BargainingTwinExperiment:
    """Fit and evaluate shadow bargaining twins from immutable named-opponent jobs."""

    def __init__(
        self,
        *,
        dossier_root: Path,
        output_dir: Path,
        min_games: int = 10,
        holdout_fraction: float = 0.2,
        min_train_games: int = 4,
        config: TwinConfig | None = None,
        opponents: set[str] | None = None,
    ) -> None:
        if min_games < 2:
            raise ValueError("min_games must be at least 2")
        if min_train_games < 1:
            raise ValueError("min_train_games must be positive")
        if min_games <= min_train_games:
            raise ValueError("min_games must exceed min_train_games")
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between zero and one")
        self.dossier_root = dossier_root
        self.output_dir = output_dir
        self.min_games = min_games
        self.holdout_fraction = holdout_fraction
        self.min_train_games = min_train_games
        self.config = config or TwinConfig()
        self.opponents = opponents
        self.response_programs = response_particles(self.config)
        self.proposal_programs = proposal_particles(self.config)

    def _eligible(self, grouped: dict[str, list[BargainingGameEvidence]]) -> dict[str, list[BargainingGameEvidence]]:
        return {
            opponent_id: games
            for opponent_id, games in grouped.items()
            if len(games) >= self.min_games and (self.opponents is None or games[0].opponent_name in self.opponents or opponent_id in self.opponents)
        }

    def _split(self, games: Sequence[BargainingGameEvidence]) -> tuple[list[BargainingGameEvidence], list[BargainingGameEvidence]]:
        holdout_count = max(1, int(math.ceil(len(games) * self.holdout_fraction)))
        holdout_count = min(holdout_count, len(games) - self.min_train_games)
        if holdout_count < 1:
            raise ValueError("not enough games for the requested chronological split")
        return list(games[:-holdout_count]), list(games[-holdout_count:])

    def _fit_weights(
        self,
        *,
        population_rows: Sequence[BargainingDecisionRow],
        target_rows: Sequence[BargainingDecisionRow],
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        population_response_scores = _response_scores(self.response_programs, population_rows)
        response_prior = _mixed_population_prior(
            population_response_scores,
            row_count=sum(row.action_type == "response" for row in population_rows),
            complexities=[particle.complexity for particle in self.response_programs],
            config=self.config,
        )
        response_weights = _posterior_from_prior(response_prior, _response_scores(self.response_programs, target_rows))
        population_proposal_scores = _proposal_scores(self.proposal_programs, population_rows)
        proposal_prior = _mixed_population_prior(
            population_proposal_scores,
            row_count=sum(row.action_type == "proposal" for row in population_rows),
            complexities=[particle.complexity for particle in self.proposal_programs],
            config=self.config,
        )
        proposal_weights = _posterior_from_prior(proposal_prior, _proposal_scores(self.proposal_programs, target_rows))
        return response_prior, response_weights, proposal_prior, proposal_weights

    def _artifact(
        self,
        *,
        games: Sequence[BargainingGameEvidence],
        population_games: Sequence[BargainingGameEvidence],
    ) -> dict[str, object]:
        target_rows = _rows(games)
        population_rows = _rows(population_games)
        response_prior, response_weights, proposal_prior, proposal_weights = self._fit_weights(population_rows=population_rows, target_rows=target_rows)
        response_records, response_mass = _weighted_particle_records(self.response_programs, response_weights, limit=self.config.artifact_particles)
        proposal_records, proposal_mass = _weighted_particle_records(self.proposal_programs, proposal_weights, limit=self.config.artifact_particles)
        drift = _drift_status(
            games=games,
            response_programs=self.response_programs,
            proposal_programs=self.proposal_programs,
            response_population_prior=response_prior,
            proposal_population_prior=proposal_prior,
        )
        artifact: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "glee-executable-opponent-model",
            "status": "shadow-only",
            "model_version": MODEL_VERSION,
            "generated_at": _now(),
            "opponent": {"id": games[0].opponent_id, "name": games[0].opponent_name},
            "family": "bargaining",
            "training": {
                "game_count": len(games),
                "row_count": len(target_rows),
                "proposal_count": sum(row.action_type == "proposal" for row in target_rows),
                "response_count": sum(row.action_type == "response" for row in target_rows),
                "game_ids": [game.game_id for game in games],
                "job_receipts": [{"job_id": game.job_id, "job_path": game.job_path, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256} for game in games],
                "population_game_count": len(population_games),
                "first_completed_at": games[0].completed_at,
                "last_completed_at": games[-1].completed_at,
            },
            "response_model": {"particle_count": len(response_records), "candidate_count": len(self.response_programs), "retained_posterior_mass": response_mass, "particles": response_records},
            "proposal_model": {"particle_count": len(proposal_records), "candidate_count": len(self.proposal_programs), "retained_posterior_mass": proposal_mass, "particles": proposal_records},
            "message_model": _message_model(population_rows, target_rows, self.config),
            "support": _support(target_rows),
            "drift": drift,
            "epistemic_boundary": {
                "target": "A predictive behavioral program, not a reconstruction of the opponent's hidden prompt, model, operator, or subjective state.",
                "counterfactuals": "Off-support simulations are model implications and are not authenticated observations of the opponent.",
                "rmm": "Nested mental-state variables enter only through program components that change observable predictions.",
            },
        }
        artifact["artifact_sha256"] = _sha(artifact)
        return artifact

    def _evaluate_target(
        self,
        *,
        target_games: Sequence[BargainingGameEvidence],
        all_games: Sequence[BargainingGameEvidence],
    ) -> dict[str, object]:
        train_games, holdout_games = self._split(target_games)
        cutoff = holdout_games[0].sort_key
        population_games = [game for game in all_games if game.opponent_id != target_games[0].opponent_id and game.sort_key < cutoff]
        target_train_rows = _rows(train_games)
        population_rows = _rows(population_games)
        _response_prior, response_weights, _proposal_prior, proposal_weights = self._fit_weights(population_rows=population_rows, target_rows=target_train_rows)
        holdout_rows = _rows(holdout_games)
        return {
            "opponent": {"id": target_games[0].opponent_id, "name": target_games[0].opponent_name},
            "split": {
                "train_game_ids": [game.game_id for game in train_games],
                "holdout_game_ids": [game.game_id for game in holdout_games],
                "cutoff": {"completed_at": cutoff[0], "completion_order": cutoff[1], "game_id": cutoff[2]},
                "population_prior_game_count": len(population_games),
            },
            "twin": {
                "response": _response_metrics(self.response_programs, response_weights, holdout_rows),
                "proposal": _proposal_metrics(self.proposal_programs, proposal_weights, holdout_rows),
            },
            "empirical_baseline": _empirical_baseline(population_rows + target_train_rows, holdout_rows),
        }

    @staticmethod
    def _aggregate_evaluation(targets: Sequence[dict[str, Any]]) -> dict[str, object]:
        def weighted(model: str, task: str, metric: str) -> float | None:
            pairs = [(int(target[model][task]["count"]), target[model][task].get(metric)) for target in targets]
            pairs = [(count, float(value)) for count, value in pairs if count > 0 and value is not None]
            denominator = sum(count for count, _value in pairs)
            return sum(count * value for count, value in pairs) / denominator if denominator else None

        return {
            "opponent_count": len(targets),
            "twin": {
                "response_nll": weighted("twin", "response", "nll"),
                "response_brier": weighted("twin", "response", "brier"),
                "response_accuracy": weighted("twin", "response", "accuracy"),
                "proposal_nll": weighted("twin", "proposal", "nll"),
                "proposal_mae": weighted("twin", "proposal", "mae"),
                "proposal_rmse": weighted("twin", "proposal", "rmse"),
                "proposal_interval_80_coverage": weighted("twin", "proposal", "interval_80_coverage"),
            },
            "empirical_baseline": {
                "response_nll": weighted("empirical_baseline", "response", "nll"),
                "response_brier": weighted("empirical_baseline", "response", "brier"),
                "response_accuracy": weighted("empirical_baseline", "response", "accuracy"),
                "proposal_mae": weighted("empirical_baseline", "proposal", "mae"),
                "proposal_rmse": weighted("empirical_baseline", "proposal", "rmse"),
            },
        }

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty shadow artifact directory: {self.output_dir}")
        games, rejected = load_bargaining_corpus(self.dossier_root)
        grouped: dict[str, list[BargainingGameEvidence]] = {}
        for game in games:
            grouped.setdefault(game.opponent_id, []).append(game)
        eligible = self._eligible(grouped)
        if not eligible:
            raise RuntimeError(f"no opponent has at least {self.min_games} bargaining games")
        corpus_receipts = [
            {"game_id": game.game_id, "opponent_id": game.opponent_id, "opponent_name": game.opponent_name, "job_id": game.job_id, "job_path": game.job_path, "job_sha256": game.job_sha256, "final_game_sha256": game.final_game_sha256}
            for game in games
        ]
        corpus = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bargaining-twin-corpus",
            "source_root": str(self.dossier_root.resolve()),
            "game_count": len(games),
            "opponent_count": len(grouped),
            "row_count": sum(len(game.rows) for game in games),
            "rejected": list(rejected),
            "receipts": corpus_receipts,
        }
        corpus["corpus_sha256"] = _sha(corpus_receipts)
        evaluations = [self._evaluate_target(target_games=target_games, all_games=games) for _opponent_id, target_games in sorted(eligible.items())]
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bargaining-twin-chronological-evaluation",
            "model_version": MODEL_VERSION,
            "split": "Per-opponent last-game holdout with population evidence restricted to games completed before the target's first holdout game.",
            "min_games": self.min_games,
            "min_train_games": self.min_train_games,
            "holdout_fraction": self.holdout_fraction,
            "targets": evaluations,
            "aggregate": self._aggregate_evaluation(evaluations),
        }
        model_records = []
        for opponent_id, target_games in sorted(eligible.items()):
            population_games = [game for game in games if game.opponent_id != opponent_id]
            artifact = self._artifact(games=target_games, population_games=population_games)
            relative = Path("opponents") / opponent_id / "bargaining" / "model.json"
            _atomic_json(self.output_dir / relative, artifact)
            model_records.append({"opponent": artifact["opponent"], "path": str(relative), "artifact_sha256": artifact["artifact_sha256"], "game_count": len(target_games)})
        _atomic_json(self.output_dir / "corpus.json", corpus)
        _atomic_json(self.output_dir / "evaluation.json", evaluation)
        module_path = Path(__file__).resolve()
        protocol_path = module_path.parents[2] / "protocols" / "glee-executable-opponent-models-v1.md"
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "glee-executable-opponent-model-run",
            "status": "shadow-only",
            "model_version": MODEL_VERSION,
            "generated_at": _now(),
            "corpus_sha256": corpus["corpus_sha256"],
            "config": asdict(self.config),
            "selection": {
                "min_games": self.min_games,
                "min_train_games": self.min_train_games,
                "holdout_fraction": self.holdout_fraction,
                "opponents": sorted(self.opponents) if self.opponents is not None else None,
            },
            "implementation_receipts": {
                "module": {"path": "src/nommd_arena/glee_bargaining_twin.py", "sha256": _sha_file(module_path)},
                "protocol": {"path": "protocols/glee-executable-opponent-models-v1.md", "sha256": _sha_file(protocol_path)},
            },
            "model_count": len(model_records),
            "models": model_records,
            "evaluation_path": "evaluation.json",
            "corpus_path": "corpus.json",
            "promotion": "No live-policy effect; promotion requires an explicit protocol revision after chronological predictive and decision-value review.",
        }
        manifest["manifest_sha256"] = _sha(manifest)
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return {"manifest": manifest, "evaluation": evaluation["aggregate"], "corpus": {key: corpus[key] for key in ("game_count", "opponent_count", "row_count", "corpus_sha256")}}
