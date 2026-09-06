"""Causal role-specific opponent models for repeated GLEE Persuasion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .glee_named_dossier import named_opponent_id, normalize_opponent_name


SCHEMA_VERSION = 1
ENGINE_VERSION = "v2.4"
MODEL_VERSION = f"persuasion-twin-{ENGINE_VERSION}"
POLICIES = ("truthful", "inverting", "pooling-positive", "pooling-negative", "noisy-selective", "terminal-cashout")
SIGNAL_POLARITIES = ("positive", "negative", "unknown")
_SPACE = re.compile(r"\s+")
_NEGATIVE_RECOMMENDATION = re.compile(r"\b(?:do\s+not|don't|not|cannot|can't|avoid|skip|pass|decline)\b.{0,36}\b(?:recommend|buy|purchase|worth)\b|\brecommend(?:ed|ing)?\b.{0,18}\b(?:pass(?:ing)?|skip(?:ping)?|declin(?:e|ing)|not\s+(?:buy|purchase))\b|\bnot\s+(?:a\s+)?(?:high|good|excellent)\s+quality\b|\b(?:low|poor|bad|inferior)\s+quality\b", re.IGNORECASE)
_NEGATIVE_DIRECTIVE = re.compile(r"\b(?:(?:please|consider)\s+|i\s+(?:would|will|recommend)\s+)?(?:pass(?:ing)?|skip(?:ping)?|declin(?:e|ing))(?:\s+(?:on\s+)?(?:this|it|one|today|the\s+product))?\b", re.IGNORECASE)
_POSITIVE_RECOMMENDATION = re.compile(r"\b(?:recommend|buy|purchase|worthwhile|excellent|good|high\s+quality|strong\s+value)\b", re.IGNORECASE)
_NEGATIVE_STANDALONE = re.compile(r"^(?:no|false|negative|pass)$", re.IGNORECASE)
_POSITIVE_STANDALONE = re.compile(r"^(?:yes|true|positive|buy)$", re.IGNORECASE)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _probability(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _optional_finite(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name)


def _clip_probability(value: float, floor: float = 1e-6) -> float:
    return min(1 - floor, max(floor, float(value)))


def _normalized_message(message: object) -> str:
    value = str(message or "").replace("\u2018", "'").replace("\u2019", "'")
    return _SPACE.sub(" ", value.strip()).casefold()


def classify_persuasion_signal(message: object, *, channel: str) -> tuple[str, str, str]:
    """Return polarity, semantic act, and a stable exact-message fingerprint."""
    normalized = _normalized_message(message)
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16] if normalized else "empty"
    if not normalized:
        return "unknown", "silence", fingerprint
    if _NEGATIVE_STANDALONE.fullmatch(normalized):
        return "negative", "binary-recommendation", fingerprint
    if _POSITIVE_STANDALONE.fullmatch(normalized):
        return "positive", "binary-recommendation", fingerprint
    negative = min((match for pattern in (_NEGATIVE_RECOMMENDATION, _NEGATIVE_DIRECTIVE) if (match := pattern.search(normalized)) is not None), key=lambda match: match.start(), default=None)
    positive = _POSITIVE_RECOMMENDATION.search(normalized)
    if negative is not None and (positive is None or negative.start() <= positive.start()):
        act = "quality-claim-low" if re.search(r"\b(?:low|poor|bad|inferior)\s+quality\b|\bnot\s+(?:a\s+)?(?:high|good|excellent)\s+quality\b", negative.group(0), re.IGNORECASE) else "discourage"
        return "negative", act, fingerprint
    if positive is not None:
        act = "quality-claim-high" if re.search(r"\b(?:high|good|excellent)\s+quality\b", normalized[: positive.end() + 24], re.IGNORECASE) else "recommend"
        return "positive", act, fingerprint
    if channel == "binary":
        return "unknown", "binary-other", fingerprint
    if any(term in normalized for term in ("price", "cost", "value", "$")):
        return "unknown", "price-value", fingerprint
    if any(term in normalized for term in ("trust", "honest", "truth", "lied", "deceive")):
        return "unknown", "trust-framing", fingerprint
    return "unknown", "other", fingerprint


def _player_roles(game: Mapping[str, Any]) -> tuple[str, str, str, str]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    our_player = str(game.get("your_player") or state.get("current_player") or "")
    if our_player not in {"player_1", "player_2"}:
        raise ValueError("persuasion game has no supported self player")
    opponent_player = "player_2" if our_player == "player_1" else "player_1"
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    action_type = str((game.get("valid_actions") or {}).get("type") or game.get("phase") or "")
    if {our_role, opponent_role} != {"seller", "buyer"}:
        our_role = "buyer" if action_type == "buyer_decision" else "seller"
        opponent_role = "seller" if our_role == "buyer" else "buyer"
    return our_player, opponent_player, our_role, opponent_role


def _opponent_identity(game: Mapping[str, Any]) -> tuple[str, str, bool]:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
    name = normalize_opponent_name(opponent.get("name"))
    named = bool(name and str(opponent.get("type") or "agent") != "hidden")
    game_id = str(game.get("game_id") or "")
    if named and name:
        return named_opponent_id(name), name, True
    suffix = hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:20]
    return f"hidden-game-{suffix}", "hidden opponent", False


@dataclass(frozen=True)
class PersuasionContext:
    """Authenticated state visible immediately before one round's buyer decision."""

    game_id: str
    opponent_id: str
    opponent_name: str
    opponent_named: bool
    completed_at: str
    completion_order: int
    our_role: str
    opponent_role: str
    channel: str
    product_price: float
    prior_high_probability: float
    low_value: float | None
    high_value: float | None
    seller_knows_buyer_values: bool | None
    round_number: int
    total_rounds: int
    prior_buys: int
    prior_passes: int
    positive_buys: int
    positive_passes: int
    negative_buys: int
    negative_passes: int
    unknown_buys: int
    unknown_passes: int
    observed_high: int
    observed_low: int
    positive_observed_high: int
    positive_observed_low: int
    negative_observed_high: int
    negative_observed_low: int
    revealed_signal_quality_phases: tuple[tuple[str, str, float], ...]

    @property
    def round_phase(self) -> float:
        if self.total_rounds <= 1:
            return 1.0
        return min(1.0, max(0.0, (self.round_number - 1) / (self.total_rounds - 1)))

    @property
    def normalized_low_surplus(self) -> float | None:
        return (self.low_value - self.product_price) / max(1.0, abs(self.product_price)) if self.low_value is not None else None

    @property
    def normalized_high_surplus(self) -> float | None:
        return (self.high_value - self.product_price) / max(1.0, abs(self.product_price)) if self.high_value is not None else None

    @property
    def normalized_prior_surplus(self) -> float | None:
        if self.high_value is None or self.low_value is None:
            return None
        expected = self.prior_high_probability * self.high_value + (1 - self.prior_high_probability) * self.low_value
        return (expected - self.product_price) / max(1.0, abs(self.product_price))


@dataclass(frozen=True)
class PersuasionRoundEvidence:
    """One causally projected round attributed to the modeled opponent role."""

    context: PersuasionContext
    signal_polarity: str
    signal_act: str
    message_fingerprint: str
    message: str
    bought: bool
    current_quality: str | None
    observed_quality: str | None
    buyer_payoff: float
    seller_payoff: float
    response_time_ms: float | None

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["context"]["round_phase"] = self.context.round_phase
        value["context"]["normalized_low_surplus"] = self.context.normalized_low_surplus
        value["context"]["normalized_high_surplus"] = self.context.normalized_high_surplus
        value["context"]["normalized_prior_surplus"] = self.context.normalized_prior_surplus
        return value


@dataclass(frozen=True)
class PersuasionGameEvidence:
    """One immutable terminal game with pass outcomes censored from buyer knowledge."""

    game_id: str
    opponent_id: str
    opponent_name: str
    opponent_named: bool
    completed_at: str
    completion_order: int
    source_path: str
    final_game_sha256: str
    history_projection: str
    source_history_rounds: int
    projected_history_rounds: int
    outcome: str
    our_role: str
    opponent_role: str
    rows: tuple[PersuasionRoundEvidence, ...]

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return self.completed_at, self.completion_order, self.game_id


def _persuasion_player_for_role(state: Mapping[str, Any], role: str) -> str:
    players = [player for player in ("player_1", "player_2") if str(state.get(f"{player}_role") or "").casefold() == role]
    if len(players) != 1:
        raise ValueError(f"terminal Persuasion state does not identify exactly one {role}")
    return players[0]


def _persuasion_history_bought(entry: Mapping[str, Any]) -> bool:
    return entry.get("bought") is True or str(entry.get("buyer_decision") or "").casefold() == "yes"


def project_persuasion_terminal_state(final_game: Mapping[str, Any], *, submitted_action: Mapping[str, Any] | None = None, response_time_ms: int | None = None) -> tuple[dict[str, Any], dict[str, object]]:
    """Project one omitted terminal buyer decision from authenticated aggregate results without mutating the source object."""
    if final_game.get("game_family") != "persuasion":
        raise ValueError("terminal object is not Persuasion")
    projected = deepcopy(dict(final_game))
    state = projected.get("game_state") if isinstance(projected.get("game_state"), dict) else None
    if state is None:
        raise ValueError("terminal Persuasion state is missing")
    history = state.get("history")
    if not isinstance(history, list) or any(not isinstance(entry, Mapping) for entry in history):
        raise ValueError("terminal Persuasion state has no valid history list")
    result = projected.get("result") if isinstance(projected.get("result"), Mapping) else state.get("result") if isinstance(state.get("result"), Mapping) else None
    if result is None:
        raise ValueError("terminal Persuasion state has no aggregate result")
    rounds_played = int(result.get("rounds_played") or 0)
    if rounds_played <= 0:
        raise ValueError("terminal Persuasion result has no positive rounds_played")
    source_history_rounds = len(history)
    if source_history_rounds == rounds_played:
        return projected, {"status": "source-complete", "source_history_rounds": source_history_rounds, "projected_history_rounds": source_history_rounds, "rounds_played": rounds_played}
    if rounds_played - source_history_rounds != 1:
        raise ValueError(f"terminal Persuasion history differs from rounds_played by {rounds_played - source_history_rounds}, not one")
    _our_player, _opponent_player, our_role, _opponent_role = _player_roles(projected)
    if our_role != "buyer":
        raise ValueError("only a submitted terminal buyer decision may be reconstructed")
    buyer_player = _persuasion_player_for_role(state, "buyer")
    seller_player = _persuasion_player_for_role(state, "seller")
    bought_before = sum(_persuasion_history_bought(entry) for entry in history)
    high_before = sum(_persuasion_history_bought(entry) and str(entry.get("quality") or "").casefold() == "high" for entry in history)
    low_before = sum(_persuasion_history_bought(entry) and str(entry.get("quality") or "").casefold() == "low" for entry in history)
    bought_delta = int(result.get("rounds_bought")) - bought_before
    high_delta = int(result.get("bought_high")) - high_before
    low_delta = int(result.get("bought_low")) - low_before
    if bought_delta not in {0, 1} or high_delta not in {0, 1} or low_delta not in {0, 1} or high_delta + low_delta != bought_delta:
        raise ValueError(f"terminal Persuasion aggregate counts cannot identify one buyer decision: bought={bought_delta}, high={high_delta}, low={low_delta}")
    inferred_decision = "yes" if bought_delta else "no"
    if submitted_action is not None and str(submitted_action.get("decision") or "").casefold() != inferred_decision:
        raise ValueError("submitted terminal buyer decision conflicts with authenticated aggregate counts")
    seller_message = state.get("seller_message")
    if seller_message is None or not str(seller_message).strip():
        raise ValueError("terminal Persuasion state omits the final seller message")
    buyer_total = _finite(result.get(f"{buyer_player}_payoff"), f"{buyer_player}_payoff")
    seller_total = _finite(result.get(f"{seller_player}_payoff"), f"{seller_player}_payoff")
    buyer_before = sum(_finite(entry.get("buyer_payoff", 0), "buyer_payoff") for entry in history)
    seller_before = sum(_finite(entry.get("seller_payoff", 0), "seller_payoff") for entry in history)
    buyer_payoff = buyer_total - buyer_before
    seller_payoff = seller_total - seller_before
    price = _finite(state.get("product_price"), "product_price")
    expected_seller_payoff = price if bought_delta else 0.0
    if not math.isclose(seller_payoff, expected_seller_payoff, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"terminal Persuasion seller payoff delta {seller_payoff} conflicts with inferred decision")
    quality = "high" if high_delta else "low" if low_delta else None
    expected_buyer_payoff = _finite(state.get("v" if quality == "high" else "u"), f"{quality}_value") - price if quality is not None else 0.0
    if not math.isclose(buyer_payoff, expected_buyer_payoff, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"terminal Persuasion buyer payoff delta {buyer_payoff} conflicts with inferred decision")
    terminal_entry: dict[str, Any] = {"round": rounds_played, "seller_message": deepcopy(seller_message), "buyer_decision": inferred_decision, "bought": bool(bought_delta), "buyer_payoff": buyer_payoff, "seller_payoff": seller_payoff}
    if quality is not None:
        terminal_entry["quality"] = quality
    if response_time_ms is not None:
        terminal_entry["response_time_ms"] = response_time_ms
    state["history"] = [deepcopy(dict(entry)) for entry in history] + [terminal_entry]
    state["buyer_total_payoff"] = buyer_total
    state["seller_total_payoff"] = seller_total
    return projected, {"status": "reconstructed-final-buyer-round", "source_history_rounds": source_history_rounds, "projected_history_rounds": len(state["history"]), "rounds_played": rounds_played, "inferred_decision": inferred_decision, "inferred_quality": quality}


def _history_counts(history: Sequence[Mapping[str, Any]], *, channel: str, total_rounds: int) -> tuple[dict[str, int], tuple[tuple[str, str, float], ...]]:
    counts = Counter({"prior_buys": 0, "prior_passes": 0, "positive_buys": 0, "positive_passes": 0, "negative_buys": 0, "negative_passes": 0, "unknown_buys": 0, "unknown_passes": 0, "observed_high": 0, "observed_low": 0, "positive_observed_high": 0, "positive_observed_low": 0, "negative_observed_high": 0, "negative_observed_low": 0})
    revealed: list[tuple[str, str, float]] = []
    for index, entry in enumerate(history):
        bought = entry.get("bought") is True or str(entry.get("buyer_decision") or "").casefold() == "yes"
        counts["prior_buys" if bought else "prior_passes"] += 1
        polarity, _act, _fingerprint = classify_persuasion_signal(entry.get("seller_message"), channel=channel)
        counts[f"{polarity}_{'buys' if bought else 'passes'}"] += 1
        if not bought:
            continue
        quality = str(entry.get("quality") or "").casefold()
        if quality not in {"high", "low"}:
            continue
        counts[f"observed_{quality}"] += 1
        if polarity in {"positive", "negative"}:
            counts[f"{polarity}_observed_{quality}"] += 1
            round_number = int(entry.get("round") or index + 1)
            phase = 1.0 if total_rounds <= 1 else min(1.0, max(0.0, (round_number - 1) / (total_rounds - 1)))
            revealed.append((polarity, quality, phase))
    return dict(counts), tuple(revealed)


def context_from_game(game: Mapping[str, Any], *, completed_at: str = "", completion_order: int = 0, prior_history: Sequence[Mapping[str, Any]] | None = None, round_number: int | None = None) -> PersuasionContext:
    """Project one live or archived game into scale-safe causal facts."""
    if game.get("game_family") != "persuasion":
        raise ValueError("game is not Persuasion")
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    _our_player, _opponent_player, our_role, opponent_role = _player_roles(game)
    opponent_id, opponent_name, opponent_named = _opponent_identity(game)
    channel = str(state.get("seller_message_type") or "text").casefold()
    if channel not in {"binary", "text"}:
        channel = "text"
    history = list(prior_history if prior_history is not None else state.get("history") or [])
    if any(not isinstance(entry, Mapping) for entry in history):
        raise ValueError("persuasion history contains a non-object entry")
    total_rounds = int(state.get("total_rounds") or 20)
    counts, revealed = _history_counts(history, channel=channel, total_rounds=total_rounds)
    raw_knows = state.get("is_seller_know_cv", state.get("seller_knows_buyer_values"))
    return PersuasionContext(
        game_id=str(game.get("game_id") or ""),
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        opponent_named=opponent_named,
        completed_at=completed_at,
        completion_order=completion_order,
        our_role=our_role,
        opponent_role=opponent_role,
        channel=channel,
        product_price=_finite(state.get("product_price"), "product_price"),
        prior_high_probability=_probability(state.get("p"), "p"),
        low_value=_optional_finite(state.get("u"), "u"),
        high_value=_optional_finite(state.get("v"), "v"),
        seller_knows_buyer_values=raw_knows if isinstance(raw_knows, bool) else None,
        round_number=int(round_number if round_number is not None else state.get("round") or len(history) + 1),
        total_rounds=total_rounds,
        revealed_signal_quality_phases=revealed,
        **counts,
    )


def extract_persuasion_game(final_game: Mapping[str, Any], *, completed_at: str, completion_order: int, source_path: Path | str = "") -> PersuasionGameEvidence:
    """Extract opponent-role evidence while censoring unpurchased quality from buyer knowledge."""
    projected_game, projection = project_persuasion_terminal_state(final_game)
    state = projected_game.get("game_state") if isinstance(projected_game.get("game_state"), Mapping) else {}
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("terminal Persuasion state has no history list")
    _our_player, _opponent_player, our_role, opponent_role = _player_roles(projected_game)
    opponent_id, opponent_name, opponent_named = _opponent_identity(projected_game)
    rows: list[PersuasionRoundEvidence] = []
    channel = str(state.get("seller_message_type") or "text").casefold()
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise ValueError("terminal Persuasion history contains a non-object entry")
        round_number = int(raw.get("round") or index + 1)
        context = context_from_game(projected_game, completed_at=completed_at, completion_order=completion_order, prior_history=history[:index], round_number=round_number)
        message = str(raw.get("seller_message") or "")
        polarity, act, fingerprint = classify_persuasion_signal(message, channel=channel)
        bought = raw.get("bought") is True or str(raw.get("buyer_decision") or "").casefold() == "yes"
        quality = str(raw.get("quality") or "").casefold()
        current_quality = quality if quality in {"high", "low"} and our_role == "seller" else None
        observed_quality = quality if quality in {"high", "low"} and bought else None
        response_time = raw.get("response_time_ms")
        rows.append(
            PersuasionRoundEvidence(
                context=context,
                signal_polarity=polarity,
                signal_act=act,
                message_fingerprint=fingerprint,
                message=message,
                bought=bought,
                current_quality=current_quality,
                observed_quality=observed_quality,
                buyer_payoff=_finite(raw.get("buyer_payoff", 0), "buyer_payoff"),
                seller_payoff=_finite(raw.get("seller_payoff", 0), "seller_payoff"),
                response_time_ms=float(response_time) if isinstance(response_time, (int, float)) and not isinstance(response_time, bool) else None,
            )
        )
    result = projected_game.get("result") if isinstance(projected_game.get("result"), Mapping) else state.get("result") if isinstance(state.get("result"), Mapping) else {}
    return PersuasionGameEvidence(
        game_id=str(projected_game.get("game_id") or ""),
        opponent_id=opponent_id,
        opponent_name=opponent_name,
        opponent_named=opponent_named,
        completed_at=completed_at,
        completion_order=completion_order,
        source_path=str(source_path),
        final_game_sha256=object_sha256(final_game),
        history_projection=str(projection["status"]),
        source_history_rounds=int(projection["source_history_rounds"]),
        projected_history_rounds=int(projection["projected_history_rounds"]),
        outcome=str(result.get("outcome") or final_game.get("status") or ""),
        our_role=our_role,
        opponent_role=opponent_role,
        rows=tuple(rows),
    )


def load_persuasion_archive(game_archive_root: Path, *, rating_history_path: Path | None = None) -> tuple[tuple[PersuasionGameEvidence, ...], tuple[dict[str, str], ...]]:
    """Load unique completed Persuasion games from immutable run outputs in causal order."""
    deltas: Mapping[str, Any] = {}
    if rating_history_path is not None and rating_history_path.is_file():
        rating = json.loads(rating_history_path.read_text(encoding="utf-8"))
        if isinstance(rating, Mapping) and isinstance(rating.get("game_deltas"), Mapping):
            deltas = rating["game_deltas"]
    accepted: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    rejected: list[dict[str, str]] = []
    for path in sorted(game_archive_root.glob("*/games/persuasion-*.json")):
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(game, dict) or game.get("game_family") != "persuasion":
                continue
            game_id = str(game.get("game_id") or "")
            if not game_id:
                raise ValueError("missing game_id")
            prior = accepted.get(game_id)
            if prior is not None:
                if object_sha256(prior[2]) != object_sha256(game):
                    raise RuntimeError(f"conflicting terminal copies: {prior[1]} and {path}")
                rejected.append({"path": str(path), "reason": f"duplicate_game:{prior[1]}"})
                continue
            result = game.get("result") if isinstance(game.get("result"), Mapping) else {}
            outcome = str(result.get("outcome") or game.get("status") or "").casefold()
            if outcome != "completed":
                rejected.append({"path": str(path), "reason": f"censored_terminal_state:{outcome or 'unknown'}"})
                continue
            delta = deltas.get(game_id) if isinstance(deltas, Mapping) else None
            completed_at = str(delta.get("completed_at") or "") if isinstance(delta, Mapping) else ""
            if not completed_at:
                completed_at = f"path-order:{path.as_posix()}"
            accepted[game_id] = completed_at, path, game
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
    games: list[PersuasionGameEvidence] = []
    ordered = sorted(accepted.items(), key=lambda item: (item[1][0], item[0]))
    for completion_order, (_game_id, (completed_at, path, game)) in enumerate(ordered, start=1):
        games.append(extract_persuasion_game(game, completed_at=completed_at, completion_order=completion_order, source_path=path))
    return tuple(games), tuple(rejected)


@dataclass(frozen=True)
class PersuasionModelConfig:
    """Regularization and narrow-authority thresholds fixed before evaluation."""

    buyer_response_alpha: float = 2.0
    buyer_response_beta: float = 2.0
    target_multiplier: float = 3.0
    same_channel_multiplier: float = 2.0
    same_polarity_multiplier: float = 2.5
    same_act_multiplier: float = 1.35
    exact_message_multiplier: float = 2.0
    within_game_response_multiplier: float = 4.0
    within_game_signal_prior_strength: float = 3.0
    within_game_global_prior_strength: float = 4.0
    within_game_global_mix: float = 0.2
    within_game_adaptation_scale: float = 3.0
    phase_decay: float = 2.0
    configuration_decay: float = 1.5
    game_row_discount: float = 0.5
    seller_target_game_limit: int = 12
    seller_target_game_multiplier: float = 1.0
    seller_target_row_discount: float = 0.5
    within_game_reliability_multiplier: float = 3.0
    population_empirical_max_weight: float = 0.3
    target_exact_max_weight: float = 0.35
    authority_min_effective_support: float = 20.0
    buyer_buy_margin_ratio: float = 0.05
    buyer_pass_margin_ratio: float = 0.05
    seller_signal_probability_gap: float = 0.15

    def validate(self) -> None:
        positives = (self.buyer_response_alpha, self.buyer_response_beta, self.target_multiplier, self.same_channel_multiplier, self.same_polarity_multiplier, self.same_act_multiplier, self.exact_message_multiplier, self.within_game_response_multiplier, self.within_game_signal_prior_strength, self.within_game_global_prior_strength, self.within_game_adaptation_scale, self.phase_decay, self.configuration_decay, self.seller_target_game_multiplier, self.within_game_reliability_multiplier, self.authority_min_effective_support)
        if any(value <= 0 or not math.isfinite(value) for value in positives):
            raise ValueError("Persuasion model weights and supports must be finite and positive")
        if not 0 < self.game_row_discount <= 1 or not 0 < self.seller_target_row_discount <= 1:
            raise ValueError("Persuasion row discounts must lie in (0, 1]")
        if isinstance(self.seller_target_game_limit, bool) or not isinstance(self.seller_target_game_limit, int) or self.seller_target_game_limit < 1:
            raise ValueError("seller_target_game_limit must be a positive integer")
        if not 0 <= self.population_empirical_max_weight < 1 or not 0 <= self.target_exact_max_weight < 1:
            raise ValueError("Persuasion empirical mixture weights must lie in [0, 1)")
        if not 0 <= self.within_game_global_mix <= 1:
            raise ValueError("within_game_global_mix must lie in [0, 1]")
        if min(self.buyer_buy_margin_ratio, self.buyer_pass_margin_ratio, self.seller_signal_probability_gap) < 0:
            raise ValueError("Persuasion authority margins must be nonnegative")


def _beta_summary(alpha: float, beta: float) -> dict[str, float]:
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1))
    radius = 1.645 * math.sqrt(max(0.0, variance))
    return {"mean": mean, "lower_90": max(0.0, mean - radius), "upper_90": min(1.0, mean + radius), "effective_support": max(0.0, total - 4.0)}


class BuyerResponseModel:
    """Estimate how a buyer responds to a candidate seller signal."""

    def __init__(self, games: Sequence[PersuasionGameEvidence], config: PersuasionModelConfig | None = None) -> None:
        self.games = tuple(games)
        self.config = config or PersuasionModelConfig()
        self.config.validate()

    def predict(self, context: PersuasionContext, *, signal_polarity: str, signal_act: str, message_fingerprint: str) -> dict[str, object]:
        alpha = self.config.buyer_response_alpha
        beta = self.config.buyer_response_beta
        population_rows = 0
        target_rows = 0
        exact_rows = 0
        contributing_games: set[str] = set()
        latest_order = max((game.completion_order for game in self.games), default=0)
        for game in self.games:
            if game.opponent_role != "buyer":
                continue
            rows = game.rows
            if not rows:
                continue
            per_row = len(rows) ** (-self.config.game_row_discount)
            recency = math.exp(-max(0, latest_order - game.completion_order) / max(25.0, len(self.games) / 4 or 25.0))
            for row in rows:
                population_rows += 1
                weight = per_row * recency
                if game.opponent_id == context.opponent_id and context.opponent_named:
                    weight *= self.config.target_multiplier
                    target_rows += 1
                if row.context.channel == context.channel:
                    weight *= self.config.same_channel_multiplier
                else:
                    weight *= 0.35
                if row.signal_polarity == signal_polarity:
                    weight *= self.config.same_polarity_multiplier
                elif signal_polarity != "unknown" and row.signal_polarity != "unknown":
                    weight *= 0.35
                if row.signal_act == signal_act:
                    weight *= self.config.same_act_multiplier
                if row.message_fingerprint == message_fingerprint and message_fingerprint != "empty":
                    weight *= self.config.exact_message_multiplier
                    exact_rows += 1
                weight *= math.exp(-self.config.phase_decay * abs(row.context.round_phase - context.round_phase))
                config_distance = abs(row.context.prior_high_probability - context.prior_high_probability)
                for historical_value, current_value in ((row.context.normalized_low_surplus, context.normalized_low_surplus), (row.context.normalized_high_surplus, context.normalized_high_surplus)):
                    if historical_value is not None and current_value is not None:
                        config_distance += abs(historical_value - current_value)
                weight *= math.exp(-self.config.configuration_decay * min(3.0, config_distance))
                if weight <= 1e-12:
                    continue
                contributing_games.add(game.game_id)
                if row.bought:
                    alpha += weight
                else:
                    beta += weight
        historical = _beta_summary(alpha, beta)
        within_buys = int(getattr(context, f"{signal_polarity}_buys")) if signal_polarity in SIGNAL_POLARITIES else 0
        within_passes = int(getattr(context, f"{signal_polarity}_passes")) if signal_polarity in SIGNAL_POLARITIES else 0
        within_signal_count = within_buys + within_passes
        within_total_count = context.prior_buys + context.prior_passes
        historical_mean = float(historical["mean"])
        signal_strength = self.config.within_game_signal_prior_strength
        global_strength = self.config.within_game_global_prior_strength
        signal_mean = (signal_strength * historical_mean + within_buys) / (signal_strength + within_signal_count)
        global_mean = (global_strength * historical_mean + context.prior_buys) / (global_strength + within_total_count)
        current_mean = (1 - self.config.within_game_global_mix) * signal_mean + self.config.within_game_global_mix * global_mean
        adaptation_weight = 1 - math.exp(-within_total_count / self.config.within_game_adaptation_scale)
        mean = (1 - adaptation_weight) * historical_mean + adaptation_weight * current_mean
        adaptive_support = signal_strength + within_signal_count + self.config.within_game_global_mix * (global_strength + within_total_count)
        variance = mean * (1 - mean) / (adaptive_support + 1)
        radius = 1.645 * math.sqrt(max(0.0, variance))
        return {
            "model": "adaptive-hierarchical-buyer-response-v2.4",
            "buy_probability": mean,
            "lower_90": max(0.0, mean - radius),
            "upper_90": min(1.0, mean + radius),
            "effective_support": adaptive_support,
            "historical_buy_probability": historical_mean,
            "historical_effective_support": historical["effective_support"],
            "current_signal_posterior": signal_mean,
            "current_global_posterior": global_mean,
            "current_game_adaptation_weight": adaptation_weight,
            "current_game_total_buys": context.prior_buys,
            "current_game_total_passes": context.prior_passes,
            "source_game_count": len(contributing_games),
            "population_row_count": population_rows,
            "target_row_count": target_rows,
            "exact_message_row_count": exact_rows,
            "within_game_matching_buys": within_buys,
            "within_game_matching_passes": within_passes,
            "within_game_response_multiplier": self.config.within_game_response_multiplier,
            "within_game_signal_prior_strength": self.config.within_game_signal_prior_strength,
            "within_game_global_prior_strength": self.config.within_game_global_prior_strength,
            "within_game_global_mix": self.config.within_game_global_mix,
            "within_game_adaptation_scale": self.config.within_game_adaptation_scale,
            "candidate": {"signal_polarity": signal_polarity, "signal_act": signal_act, "message_fingerprint": message_fingerprint},
        }


def _policy_positive_probability(policy: str, quality: str, round_phase: float) -> float:
    if policy == "truthful":
        return 0.92 if quality == "high" else 0.08
    if policy == "inverting":
        return 0.08 if quality == "high" else 0.92
    if policy == "pooling-positive":
        return 0.9
    if policy == "pooling-negative":
        return 0.1
    if policy == "noisy-selective":
        return 0.75 if quality == "high" else 0.35
    if policy == "terminal-cashout":
        if quality == "high":
            return 0.9
        return 0.12 if round_phase < 0.75 else 0.78
    raise ValueError(f"unsupported Persuasion seller policy: {policy}")


def _signal_likelihood(polarity: str, *, positive_probability: float) -> float:
    if polarity == "positive":
        return positive_probability
    if polarity == "negative":
        return 1 - positive_probability
    return 0.5


class SellerReliabilityModel:
    """Infer a seller policy mixture only from qualities revealed by purchases."""

    def __init__(self, games: Sequence[PersuasionGameEvidence], config: PersuasionModelConfig | None = None) -> None:
        self.games = tuple(games)
        self.config = config or PersuasionModelConfig()
        self.config.validate()
        self._cached_population_prior: dict[str, float] | None = None

    def _population_prior(self) -> dict[str, float]:
        if self._cached_population_prior is not None:
            return dict(self._cached_population_prior)
        accumulated = {policy: 1.0 for policy in POLICIES}
        by_opponent: dict[str, list[PersuasionRoundEvidence]] = defaultdict(list)
        for game in self.games:
            if game.opponent_role == "seller":
                by_opponent[game.opponent_id].extend(row for row in game.rows if row.observed_quality in {"high", "low"})
        for rows in by_opponent.values():
            logs = {policy: 0.0 for policy in POLICIES}
            for row in rows:
                if row.signal_polarity == "unknown" or row.observed_quality is None:
                    continue
                for policy in POLICIES:
                    positive = _policy_positive_probability(policy, row.observed_quality, row.context.round_phase)
                    logs[policy] += math.log(_clip_probability(_signal_likelihood(row.signal_polarity, positive_probability=positive)))
            maximum = max(logs.values())
            weights = {policy: math.exp(value - maximum) for policy, value in logs.items()}
            total = sum(weights.values())
            for policy in POLICIES:
                accumulated[policy] += weights[policy] / total
        total = sum(accumulated.values())
        self._cached_population_prior = {policy: accumulated[policy] / total for policy in POLICIES}
        return dict(self._cached_population_prior)

    def predict(self, context: PersuasionContext, *, signal_polarity: str, message_fingerprint: str) -> dict[str, object]:
        population_weights = self._population_prior()
        logs = {policy: math.log(_clip_probability(weight)) for policy, weight in population_weights.items()}
        target_rows: list[PersuasionRoundEvidence] = []
        population_matching: list[PersuasionRoundEvidence] = []
        exact_weighted_high = 1.0
        exact_weighted_low = 1.0
        exact_rows = 0
        exact_effective_support = 0.0
        for game in self.games:
            if game.opponent_role != "seller":
                continue
            for row in game.rows:
                if row.observed_quality not in {"high", "low"}:
                    continue
                if row.context.channel == context.channel and row.signal_polarity == signal_polarity:
                    population_matching.append(row)
        target_games = []
        if context.opponent_named:
            target_games = sorted((game for game in self.games if game.opponent_role == "seller" and game.opponent_id == context.opponent_id), key=lambda game: game.sort_key)[-self.config.seller_target_game_limit :]
        for game_index, game in enumerate(target_games):
            informative = [row for row in game.rows if row.observed_quality in {"high", "low"}]
            if not informative:
                continue
            game_recency = math.exp(-(len(target_games) - game_index - 1) / max(1.0, self.config.seller_target_game_limit / 2))
            row_weight = self.config.seller_target_game_multiplier * game_recency * len(informative) ** (-self.config.seller_target_row_discount)
            for row in informative:
                target_rows.append(row)
                if row.message_fingerprint == message_fingerprint and message_fingerprint != "empty":
                    exact_rows += 1
                    exact_effective_support += row_weight
                    if row.observed_quality == "high":
                        exact_weighted_high += row_weight
                    else:
                        exact_weighted_low += row_weight
                if row.signal_polarity == "unknown":
                    continue
                for policy in POLICIES:
                    positive = _policy_positive_probability(policy, str(row.observed_quality), row.context.round_phase)
                    logs[policy] += row_weight * math.log(_clip_probability(_signal_likelihood(row.signal_polarity, positive_probability=positive)))
        historical_maximum = max(logs.values())
        historical_weights = {policy: math.exp(value - historical_maximum) for policy, value in logs.items()}
        historical_total = sum(historical_weights.values())
        historical_weights = {policy: value / historical_total for policy, value in historical_weights.items()}
        for polarity, quality, phase in context.revealed_signal_quality_phases:
            for policy in POLICIES:
                positive = _policy_positive_probability(policy, quality, phase)
                logs[policy] += self.config.within_game_reliability_multiplier * math.log(_clip_probability(_signal_likelihood(polarity, positive_probability=positive)))
        maximum = max(logs.values())
        policy_weights = {policy: math.exp(value - maximum) for policy, value in logs.items()}
        weight_total = sum(policy_weights.values())
        policy_weights = {policy: value / weight_total for policy, value in policy_weights.items()}
        component_posteriors: dict[str, float] = {}
        prior = context.prior_high_probability
        for policy in POLICIES:
            high_like = _signal_likelihood(signal_polarity, positive_probability=_policy_positive_probability(policy, "high", context.round_phase))
            low_like = _signal_likelihood(signal_polarity, positive_probability=_policy_positive_probability(policy, "low", context.round_phase))
            denominator = prior * high_like + (1 - prior) * low_like
            component_posteriors[policy] = prior if denominator <= 0 else prior * high_like / denominator
        policy_mean = sum(policy_weights[policy] * component_posteriors[policy] for policy in POLICIES)
        empirical_high = 1.0 + sum(row.observed_quality == "high" for row in population_matching)
        empirical_low = 1.0 + sum(row.observed_quality == "low" for row in population_matching)
        empirical = _beta_summary(empirical_high, empirical_low)
        exact = _beta_summary(exact_weighted_high, exact_weighted_low)
        history_scale = 1 / (1 + self.config.within_game_reliability_multiplier * len(context.revealed_signal_quality_phases))
        empirical_weight = history_scale * min(self.config.population_empirical_max_weight, empirical["effective_support"] / 100)
        exact_weight = history_scale * min(self.config.target_exact_max_weight, exact["effective_support"] / 8)
        posterior = (1 - empirical_weight) * policy_mean + empirical_weight * empirical["mean"]
        posterior = (1 - exact_weight) * posterior + exact_weight * exact["mean"]
        supported_components = [component_posteriors[policy] for policy in POLICIES if policy_weights[policy] >= 0.03]
        lower = min([posterior, empirical["lower_90"], *supported_components])
        upper = max([posterior, empirical["upper_90"], *supported_components])
        return {
            "model": "censored-seller-policy-mixture-v2.2",
            "posterior_high_probability": posterior,
            "lower_90": max(0.0, lower),
            "upper_90": min(1.0, upper),
            "policy_weights": policy_weights,
            "historical_policy_weights": historical_weights,
            "policy_component_posteriors": component_posteriors,
            "dominant_policy_before_current_game": max(historical_weights, key=historical_weights.get),
            "dominant_policy_after_current_game": max(policy_weights, key=policy_weights.get),
            "current_game_policy_shift_total_variation": 0.5 * sum(abs(policy_weights[policy] - historical_weights[policy]) for policy in POLICIES),
            "population_matching_purchased_rows": len(population_matching),
            "target_purchased_rows": len(target_rows),
            "target_game_count": len(target_games),
            "exact_message_purchased_rows": exact_rows,
            "exact_message_effective_support": exact_effective_support,
            "within_game_revealed_rows": len(context.revealed_signal_quality_phases),
            "within_game_reliability_multiplier": self.config.within_game_reliability_multiplier,
            "censoring_contract": "quality from every pass is excluded even if a terminal archive contains it",
        }


def model_receipt(config: PersuasionModelConfig | None = None) -> dict[str, object]:
    selected = config or PersuasionModelConfig()
    selected.validate()
    return {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION, "engine_version": ENGINE_VERSION, "config": asdict(selected), "seller_side": "hierarchical buyer-response model with complete buy/pass labels and bounded current-game adaptation", "buyer_side": "censored interpretable seller-policy mixture", "causal_boundary": "only prior games and current-game observations available before the decision may contribute"}


def iter_rows(games: Iterable[PersuasionGameEvidence], *, opponent_role: str | None = None) -> Iterable[PersuasionRoundEvidence]:
    for game in games:
        if opponent_role is not None and game.opponent_role != opponent_role:
            continue
        yield from game.rows
