"""Distinct outward identity policy for the local GLEE collector candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glee_bargaining_twin import classify_message_act
from .glee_negotiation_twin_v2 import classify_negotiation_message
from .glee_persuasion_twin_v2 import classify_persuasion_signal


COLLECTOR_IDENTITY_POLICY_CONTRACT = "glee-collector-identity-policy-v1"
COLLECTOR_IDENTITY_POINTER_CONTRACT = "glee-collector-identity-policy-pointer-v1"
COLLECTOR_IDENTITY_ASSIGNMENT_CONTRACT = "glee-collector-identity-assignment-v1"
COLLECTOR_TIMING_TARGET_CONTRACT = "glee-collector-timing-target-v1"

_FAMILIES = ("bargaining", "negotiation", "persuasion")
_IDENTITY_MODES = ("known", "hidden")
_RENDERERS = {"measured-plain", "compact-fragment", "dialogue-question", "neutral-sentence", "amount-forward"}
_MOVE_KINDS = {
    "bargaining": {"proposal", "decision"},
    "negotiation": {"proposal", "decision"},
    "persuasion": {"seller-signal", "buyer-decision"},
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _identity_mode(game: Mapping[str, object]) -> str:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
    return "hidden" if opponent.get("type") == "hidden" or not str(opponent.get("name") or "").strip() else "known"


def _messages_allowed(game: Mapping[str, object]) -> bool:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    if state.get("messages_allowed") is False:
        return False
    valid = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    fields = valid.get("fields")
    return "message" in fields if isinstance(fields, Mapping) else state.get("messages_allowed") is True


def _message_action(game: Mapping[str, object], action: Mapping[str, object]) -> bool:
    if not _messages_allowed(game):
        return False
    family = str(game.get("game_family") or "")
    valid = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    action_type = str(valid.get("type") or game.get("phase") or "")
    if family == "bargaining":
        return action_type == "offer"
    if family == "negotiation":
        return action_type == "offer" or (action_type == "decision" and str(action.get("decision") or "") == "RejectOffer" and _finite(action.get("product_price")) is not None)
    return family == "persuasion" and action_type == "seller_message"


def _move_kind(game: Mapping[str, object]) -> str | None:
    family = str(game.get("game_family") or "")
    valid = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    action_type = str(valid.get("type") or game.get("phase") or "")
    if family in {"bargaining", "negotiation"}:
        return "proposal" if action_type == "offer" else "decision" if action_type == "decision" else None
    if family == "persuasion":
        return "seller-signal" if action_type in {"seller_message", "seller_recommendation"} else "buyer-decision" if action_type == "buyer_decision" else None
    return None


def _format_amount(value: object) -> str:
    number = _finite(value)
    if number is None:
        return "the amount shown"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _pick(game: Mapping[str, object], profile_id: str, semantic: str, values: Sequence[str]) -> str:
    if not values:
        raise ValueError("collector identity renderer has no variants")
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    key = "|".join((str(game.get("game_id") or ""), profile_id, str(state.get("round") or ""), semantic))
    index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(values)
    return values[index]


def _bargaining_message(game: Mapping[str, object], action: Mapping[str, object], renderer: str, semantic: str, profile_id: str) -> str:
    alice = _format_amount(action.get("alice_gain") if "alice_gain" in action else action.get("player_1_gain"))
    bob = _format_amount(action.get("bob_gain") if "bob_gain" in action else action.get("player_2_gain"))
    measured = {
        "allocation": (f"Offer: Alice {alice}, Bob {bob}.", f"The division is Alice {alice} and Bob {bob}."),
        "authority": ("I am keeping this position.", "I am holding this division."),
        "commitment": ("I can agree on this division.", "I am ready to proceed on these figures."),
        "fairness": ("I consider this division workable for both sides.", "This division leaves both sides a workable share."),
        "fairness-urgency": ("This division is workable now, before more value is lost.", "The shares are workable, and another round only reduces them."),
        "urgency": ("Closing this round avoids another loss in value.", "A decision now preserves more of the pool."),
        "other": ("Please evaluate the division shown.", "The figures above are my proposal."),
    }
    compact = {
        "allocation": (f"alice {alice} bob {bob}", "figures are the offer"),
        "authority": ("holding this line", "staying with this split"),
        "commitment": ("can agree here", "ready on these figures"),
        "fairness": ("workable for both", "both sides get value"),
        "fairness-urgency": ("workable before decay", "settle before more loss"),
        "urgency": ("close this round", "avoid another decay"),
        "other": ("review the division", "consider these figures"),
    }
    dialogue = {
        "allocation": ("Would this division work for you?", "Can you work with these figures?"),
        "authority": ("Can we proceed from this position?", "Can you decide against this line?"),
        "commitment": ("Can we agree on this division?", "Shall we settle on these figures?"),
        "fairness": ("Does this leave both sides enough?", "Would you call this workable for both sides?"),
        "fairness-urgency": ("Can we take this workable split before more value is lost?", "Shall we settle before another round reduces the pool?"),
        "urgency": ("Can we close before another round?", "Shall we avoid another loss in value?"),
        "other": ("Would you take this division?", "Can you evaluate these figures?"),
    }
    neutral = {
        "allocation": (f"The division assigns {alice} to Alice and {bob} to Bob.", "The displayed figures define the division."),
        "authority": ("This position remains unchanged.", "The current division remains my position."),
        "commitment": ("This division is available for agreement.", "I am prepared to proceed on the displayed figures."),
        "fairness": ("The division gives each side a workable outcome.", "Both sides retain value under this division."),
        "fairness-urgency": ("The division is workable and avoids another round of erosion.", "These shares remain workable before further loss."),
        "urgency": ("Agreement this round avoids further erosion.", "Another round would reduce the available value."),
        "other": ("The division is open for consideration.", "The displayed figures are the current proposal."),
    }
    ledger = {
        "allocation": (f"Alice {alice}; Bob {bob}. Review the allocation.", f"{alice} to Alice; {bob} to Bob. Those are the figures."),
        "authority": (f"Alice {alice}; Bob {bob}. I am retaining this line.", f"{alice} and {bob}. The position remains fixed."),
        "commitment": (f"Alice {alice}; Bob {bob}. I can proceed on that basis.", f"{alice} and {bob}. Agreement is available."),
        "fairness": (f"Alice {alice}; Bob {bob}. Both sides retain value.", f"{alice} and {bob}. The allocation is workable."),
        "fairness-urgency": (f"Alice {alice}; Bob {bob}. Delay reduces both figures.", f"{alice} and {bob}. This round preserves more value."),
        "urgency": (f"Alice {alice}; Bob {bob}. Another round reduces value.", f"{alice} and {bob}. A decision now avoids more loss."),
        "other": (f"Alice {alice}; Bob {bob}. Please review.", f"{alice} and {bob}. Consider the allocation."),
    }
    table = measured if renderer == "measured-plain" else compact if renderer == "compact-fragment" else dialogue if renderer == "dialogue-question" else neutral if renderer == "neutral-sentence" else ledger
    return _pick(game, profile_id, semantic, table.get(semantic, table["other"]))


def _negotiation_message(game: Mapping[str, object], action: Mapping[str, object], renderer: str, semantic: str, profile_id: str) -> str:
    amount = _format_amount(action.get("product_price"))
    measured = {
        "walkaway": ("I will withdraw rather than cross this amount.", "I would leave the trade beyond this point."),
        "conditional": ("If this amount works, I can conclude the trade.", "If you can use this amount, we can finish."),
        "urgency": ("This amount is available for agreement now.", "A decision now can complete the trade."),
        "fairness": ("I consider this a reasonable trading point.", "This amount gives us a workable trading point."),
        "commitment": ("I can agree at this amount.", "I am ready to transact at this amount."),
        "price": (f"My amount is {amount}.", f"I am proposing {amount}."),
        "other": ("Please evaluate the amount shown.", "The displayed amount is my proposal."),
    }
    compact = {
        "walkaway": ("leave beyond this", "otherwise I withdraw"),
        "conditional": ("works if you agree", "use this and finish"),
        "urgency": ("close on this round", "decide on this amount"),
        "fairness": ("reasonable trade point", "workable for both"),
        "commitment": ("can transact here", "ready at this amount"),
        "price": (f"amount {amount}", f"counter {amount}"),
        "other": ("review the amount", "consider this counter"),
    }
    dialogue = {
        "walkaway": ("Should we stop if this point cannot work?", "Can we agree before I withdraw?"),
        "conditional": ("If this amount works for you, shall we finish?", "Can we conclude if you use this amount?"),
        "urgency": ("Can we close on this amount now?", "Shall we finish the trade this round?"),
        "fairness": ("Does this look like a reasonable trading point?", "Can we use this as a workable point for both sides?"),
        "commitment": ("Can we agree at this amount?", "Shall we transact on this figure?"),
        "price": (f"Would {amount} work for you?", f"Can you transact at {amount}?"),
        "other": ("Can you evaluate the amount shown?", "Would you consider this counter?"),
    }
    neutral = {
        "walkaway": ("The trade ends if this boundary cannot be met.", "I will withdraw beyond the displayed boundary."),
        "conditional": ("Agreement at this amount would complete the trade.", "The trade can conclude if this amount is accepted."),
        "urgency": ("The trade can be completed at this amount now.", "A decision this round can close the trade."),
        "fairness": ("This amount provides a reasonable trading point.", "The displayed amount is workable for both sides."),
        "commitment": ("This amount is available for agreement.", "I am prepared to transact on this figure."),
        "price": (f"The current amount is {amount}.", f"The counter is {amount}."),
        "other": ("The displayed amount is available for consideration.", "This counter is open for review."),
    }
    ledger = {
        "walkaway": (f"{amount}. Beyond it, I withdraw.", f"{amount}. That is the exit boundary."),
        "conditional": (f"{amount}. Agreement there completes the trade.", f"{amount}. Accepting it finishes the exchange."),
        "urgency": (f"{amount}. We can close this round.", f"{amount}. A decision now completes the trade."),
        "fairness": (f"{amount}. It is a workable trading point.", f"{amount}. Both sides can use this point."),
        "commitment": (f"{amount}. I can transact there.", f"{amount}. Agreement is available."),
        "price": (f"{amount}. That is my counter.", f"{amount}. Please review the figure."),
        "other": (f"{amount}. Please evaluate it.", f"{amount}. The figure is open for review."),
    }
    table = measured if renderer == "measured-plain" else compact if renderer == "compact-fragment" else dialogue if renderer == "dialogue-question" else neutral if renderer == "neutral-sentence" else ledger
    return _pick(game, profile_id, semantic, table.get(semantic, table["other"]))


def _persuasion_message(game: Mapping[str, object], renderer: str, polarity: str, profile_id: str) -> str:
    measured = {
        "positive": ("I recommend that you buy this item.", "I recommend taking this item."),
        "negative": ("I recommend that you pass on this item.", "I do not recommend this purchase."),
        "unknown": ("My signal remains inconclusive.", "The available signal is inconclusive."),
    }
    compact = {"positive": ("worth a buy", "recommend taking it"), "negative": ("recommend passing", "not worth buying"), "unknown": ("signal stays unclear", "no firm view")}
    dialogue = {"positive": ("I would buy this one; would you?", "This one looks like a worthwhile purchase."), "negative": ("I would not buy this one; would you?", "I recommend passing on this one; do you agree?"), "unknown": ("Would you decide from your own estimate?", "Can you treat this signal as uncertain?")}
    neutral = {"positive": ("This item is a worthwhile buy.", "I recommend purchasing this item."), "negative": ("This item is not worth buying.", "I recommend passing on this item."), "unknown": ("The signal does not favor either choice.", "The recommendation remains uncertain.")}
    ledger = {"positive": ("Recommendation: buy.", "Signal: buy the item."), "negative": ("I recommend: pass.", "Signal: do not buy."), "unknown": ("Recommendation: undecided.", "Signal: inconclusive.")}
    table = measured if renderer == "measured-plain" else compact if renderer == "compact-fragment" else dialogue if renderer == "dialogue-question" else neutral if renderer == "neutral-sentence" else ledger
    return _pick(game, profile_id, polarity, table[polarity])


def _semantic(game: Mapping[str, object], action: Mapping[str, object]) -> tuple[str, str | None]:
    family = str(game.get("game_family") or "")
    original = str(action.get("message") or "").strip()
    if family == "bargaining":
        act = classify_message_act(original, messages_allowed=True) if original else "allocation"
        return act if act != "none" else "allocation", None
    if family == "negotiation":
        act = classify_negotiation_message(original, messages_allowed=True) if original else "price"
        return act if act != "none" else "price", None
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    polarity, act, _fingerprint = classify_persuasion_signal(original, channel=str(state.get("seller_message_type") or "text").casefold())
    return act, polarity


def realize_collector_identity(game: Mapping[str, object], action: Mapping[str, object], assignment: Mapping[str, object]) -> tuple[dict[str, Any], dict[str, object]]:
    """Render only the outward message while preserving every non-message action field."""
    styled = copy.deepcopy(dict(action))
    if not _message_action(game, styled):
        return styled, {"contract": COLLECTOR_IDENTITY_POLICY_CONTRACT, "status": "not-applicable", "profile_id": assignment.get("profile_id"), "action_unchanged": True}
    family = str(game.get("game_family") or "")
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    original = str(styled.get("message") or "").strip()
    channel = str(state.get("seller_message_type") or "text").casefold()
    if family == "persuasion" and channel == "binary":
        return styled, {"contract": COLLECTOR_IDENTITY_POLICY_CONTRACT, "status": "protocol-bound-binary", "profile_id": assignment.get("profile_id"), "action_unchanged": True}
    semantic, polarity = _semantic(game, styled)
    profile_id = str(assignment.get("profile_id") or "")
    renderer = str(assignment.get("renderer") or "")
    if renderer not in _RENDERERS:
        raise ValueError("collector identity assignment has an unsupported renderer")
    if family == "bargaining":
        rendered = _bargaining_message(game, styled, renderer, semantic, profile_id)
    elif family == "negotiation":
        rendered = _negotiation_message(game, styled, renderer, semantic, profile_id)
    elif family == "persuasion":
        rendered = _persuasion_message(game, renderer, polarity or "unknown", profile_id)
    else:
        raise ValueError(f"unsupported collector identity family: {family}")
    rendered = rendered.strip()
    if not rendered or len(rendered) > 500:
        raise RuntimeError("collector identity renderer produced an invalid message")
    if family == "persuasion":
        rendered_polarity, _act, _fingerprint = classify_persuasion_signal(rendered, channel=channel)
        if rendered_polarity != polarity:
            raise RuntimeError("collector identity renderer changed Persuasion signal polarity")
    styled["message"] = rendered
    before = {key: value for key, value in action.items() if key != "message"}
    after = {key: value for key, value in styled.items() if key != "message"}
    if before != after:
        raise RuntimeError("collector identity renderer changed a non-message action field")
    return styled, {
        "contract": COLLECTOR_IDENTITY_POLICY_CONTRACT,
        "status": "styled",
        "profile_id": profile_id,
        "renderer": renderer,
        "semantic": semantic,
        "polarity": polarity,
        "original_message_sha256": _sha256_text(original),
        "rendered_message_sha256": _sha256_text(rendered),
        "message_changed": rendered != original,
        "nonmessage_action_unchanged": True,
    }


def _log_quantile(points: Sequence[Mapping[str, object]], quantile: float) -> float:
    bounded = min(1.0, max(0.0, quantile))
    for left, right in zip(points, points[1:]):
        left_q = float(left["q"])
        right_q = float(right["q"])
        if bounded > right_q:
            continue
        fraction = 0.0 if left_q == right_q else (bounded - left_q) / (right_q - left_q)
        return math.exp(math.log(float(left["seconds"])) + fraction * (math.log(float(right["seconds"])) - math.log(float(left["seconds"]))))
    return float(points[-1]["seconds"])


def sample_collector_timing_target(game: Mapping[str, object], assignment: Mapping[str, object], random_source: object) -> tuple[float | None, dict[str, object]]:
    """Sample a continuous target around the game-pinned collector speed quantile."""
    family = str(game.get("game_family") or "")
    move_kind = _move_kind(game)
    timing = assignment.get("timing") if isinstance(assignment.get("timing"), Mapping) else {}
    points = timing.get("quantile_seconds") if isinstance(timing.get("quantile_seconds"), list) else None
    game_quantile = _finite(assignment.get("game_speed_quantile"))
    residual = _finite(timing.get("residual_quantile_half_width"))
    multiplier = _finite((timing.get("move_multipliers") or {}).get(move_kind)) if isinstance(timing.get("move_multipliers"), Mapping) else None
    if family not in _FAMILIES or move_kind not in _MOVE_KINDS.get(family, set()) or points is None or game_quantile is None or residual is None or multiplier is None:
        return None, {"contract": COLLECTOR_TIMING_TARGET_CONTRACT, "status": "inapplicable-or-invalid"}
    jitter = float(random_source.uniform(-residual, residual))
    sampled_quantile = game_quantile + jitter
    if sampled_quantile < 0.0:
        sampled_quantile = -sampled_quantile
    if sampled_quantile > 1.0:
        sampled_quantile = 2.0 - sampled_quantile
    target = _log_quantile(points, sampled_quantile) * multiplier
    return target, {
        "contract": COLLECTOR_TIMING_TARGET_CONTRACT,
        "status": "sampled",
        "profile_id": assignment.get("profile_id"),
        "timing_profile_id": assignment.get("timing_profile_id"),
        "game_speed_quantile": round(game_quantile, 6),
        "residual_quantile_jitter": round(jitter, 6),
        "sampled_quantile": round(sampled_quantile, 6),
        "move_kind": move_kind,
        "move_multiplier": multiplier,
        "target_elapsed_s": round(target, 6),
    }


class CollectorIdentityPolicyStore:
    """Load one isolated collector release and durably pin one outward profile per game."""

    def __init__(self, *, root: Path, assignments_path: Path, random_source: object | None = None) -> None:
        self.root = root.resolve()
        self.assignments_path = assignments_path.resolve()
        self._random = random_source or secrets.SystemRandom()
        self._lock = threading.Lock()
        self.release, self.release_sha256, self.release_path = self._load_release()
        self._assignments: dict[str, dict[str, object]] = {}
        if self.assignments_path.is_file():
            for line in self.assignments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_assignment(record)
                self._validate_assignment_release(record)
                game_id = str(record["game_id"])
                prior = self._assignments.get(game_id)
                if prior is not None and prior != record:
                    raise RuntimeError(f"collector identity game {game_id} has conflicting assignments")
                self._assignments[game_id] = record

    @staticmethod
    def _valid_quantiles(points: object) -> bool:
        if not isinstance(points, list) or len(points) < 3:
            return False
        quantiles = [_finite(point.get("q")) if isinstance(point, Mapping) else None for point in points]
        seconds = [_finite(point.get("seconds")) if isinstance(point, Mapping) else None for point in points]
        return not any(value is None for value in quantiles + seconds) and quantiles[0] == 0.0 and quantiles[-1] == 1.0 and all(left < right for left, right in zip(quantiles, quantiles[1:])) and all(value > 0 for value in seconds) and all(left <= right for left, right in zip(seconds, seconds[1:]))

    @classmethod
    def _validate_release(cls, release: object) -> dict[str, object]:
        if not isinstance(release, dict) or release.get("schema_version") != 1 or release.get("contract") != COLLECTOR_IDENTITY_POLICY_CONTRACT:
            raise RuntimeError("collector identity release has an incompatible contract")
        profiles = release.get("profiles")
        routing = release.get("routing")
        timings = release.get("timing_profiles")
        multipliers = release.get("move_multipliers")
        fallbacks = release.get("fallback_messages")
        if not isinstance(release.get("revision"), str) or not isinstance(profiles, dict) or not profiles or not isinstance(routing, dict) or set(routing) != set(_IDENTITY_MODES) or not isinstance(timings, dict) or not timings or not isinstance(multipliers, dict) or set(multipliers) != set(_FAMILIES) or not isinstance(fallbacks, dict):
            raise RuntimeError("collector identity release is incomplete")
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not isinstance(profile, Mapping) or profile.get("renderer") not in _RENDERERS or profile.get("timing_profile") not in timings or not str(profile.get("prompt_contract") or "").strip():
                raise RuntimeError(f"collector identity profile {profile_id!r} is invalid")
        for identity_mode in _IDENTITY_MODES:
            routes = routing[identity_mode]
            if not isinstance(routes, Mapping) or set(routes) != set(_FAMILIES):
                raise RuntimeError(f"collector identity {identity_mode} routing is incomplete")
            for family, values in routes.items():
                if not isinstance(values, list) or not values or any(value not in profiles for value in values):
                    raise RuntimeError(f"collector identity route {identity_mode}:{family} is invalid")
        for timing_id, timing in timings.items():
            distribution = timing.get("game_quantile_distribution") if isinstance(timing, Mapping) else None
            residual = _finite(timing.get("residual_quantile_half_width")) if isinstance(timing, Mapping) else None
            if not isinstance(timing_id, str) or not isinstance(distribution, Mapping) or distribution.get("kind") != "triangular" or not cls._valid_quantiles(timing.get("quantile_seconds")) or residual is None or not 0 <= residual <= 0.5:
                raise RuntimeError(f"collector timing profile {timing_id!r} is invalid")
            low, mode, high = (_finite(distribution.get(name)) for name in ("low", "mode", "high"))
            if low is None or mode is None or high is None or not 0 <= low <= mode <= high <= 1 or low == high:
                raise RuntimeError(f"collector timing distribution {timing_id!r} is invalid")
        for family, values in multipliers.items():
            if not isinstance(values, Mapping) or set(values) != _MOVE_KINDS[family] or any(_finite(value) is None or not 0.5 <= float(value) <= 2.0 for value in values.values()):
                raise RuntimeError(f"collector move multipliers for {family} are invalid")
        expected_fallbacks = {"bargaining", "negotiation", "persuasion_positive", "persuasion_negative", "persuasion_unknown"}
        if set(fallbacks) != expected_fallbacks or any(not isinstance(values, list) or len(values) < 2 or any(not isinstance(value, str) or not value.strip() for value in values) for values in fallbacks.values()):
            raise RuntimeError("collector fallback wording set is invalid")
        for expected in ("positive", "negative", "unknown"):
            for message in fallbacks[f"persuasion_{expected}"]:
                polarity, _act, _fingerprint = classify_persuasion_signal(message, channel="text")
                if polarity != expected:
                    raise RuntimeError(f"collector Persuasion fallback changes {expected} polarity")
        return copy.deepcopy(release)

    def _load_release(self) -> tuple[dict[str, object], str, Path]:
        pointer_path = self.root / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("schema_version") != 1 or pointer.get("contract") != COLLECTOR_IDENTITY_POINTER_CONTRACT or not isinstance(pointer.get("release"), str) or not isinstance(pointer.get("release_sha256"), str):
            raise RuntimeError("collector identity pointer is invalid")
        release_path = (self.root / str(pointer["release"])).resolve()
        try:
            release_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("collector identity release escapes its policy root") from error
        digest = _sha256_file(release_path)
        if digest != pointer["release_sha256"]:
            raise RuntimeError("collector identity release hash differs from its pointer")
        return self._validate_release(json.loads(release_path.read_text(encoding="utf-8"))), digest, release_path

    @staticmethod
    def _validate_assignment(record: object) -> None:
        quantile = _finite(record.get("game_speed_quantile")) if isinstance(record, Mapping) else None
        if not isinstance(record, Mapping) or record.get("schema_version") != 1 or record.get("contract") != COLLECTOR_IDENTITY_ASSIGNMENT_CONTRACT or not all(isinstance(record.get(name), str) and str(record[name]) for name in ("game_id", "family", "identity_mode", "profile_id", "renderer", "timing_profile_id", "policy_revision", "release_sha256")) or record.get("family") not in _FAMILIES or record.get("identity_mode") not in _IDENTITY_MODES or record.get("renderer") not in _RENDERERS or quantile is None or not 0 <= quantile <= 1:
            raise RuntimeError("collector identity assignment is invalid")

    def _validate_assignment_release(self, record: Mapping[str, object]) -> None:
        profile = self.release["profiles"].get(record["profile_id"])
        if not isinstance(profile, Mapping) or record.get("policy_revision") != self.release["revision"] or record.get("release_sha256") != self.release_sha256 or record.get("renderer") != profile.get("renderer") or record.get("timing_profile_id") != profile.get("timing_profile"):
            raise RuntimeError("collector identity assignment does not match the active release")

    def _sample_triangular(self, distribution: Mapping[str, object]) -> float:
        low = float(distribution["low"])
        mode = float(distribution["mode"])
        high = float(distribution["high"])
        unit = float(self._random.random())
        breakpoint = (mode - low) / (high - low)
        if unit <= breakpoint:
            return low + math.sqrt(unit * (high - low) * (mode - low))
        return high - math.sqrt((1.0 - unit) * (high - low) * (high - mode))

    def _append(self, record: Mapping[str, object]) -> None:
        self.assignments_path.parent.mkdir(parents=True, exist_ok=True)
        with self.assignments_path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def profile_for_game(self, game: Mapping[str, object]) -> tuple[dict[str, object], bool]:
        """Return the immutable profile for a game, creating and fsyncing it once when absent."""
        game_id = str(game.get("game_id") or "")
        family = str(game.get("game_family") or "")
        if not game_id or family not in _FAMILIES:
            raise ValueError("collector identity assignment requires a supported game")
        with self._lock:
            existing = self._assignments.get(game_id)
            if existing is not None:
                if existing["family"] != family or existing["identity_mode"] != _identity_mode(game):
                    raise RuntimeError(f"collector identity game {game_id} conflicts with its durable assignment")
                return copy.deepcopy(existing), False
            identity_mode = _identity_mode(game)
            routes = self.release["routing"][identity_mode][family]
            profile_id = routes[min(len(routes) - 1, int(float(self._random.random()) * len(routes)))]
            profile = self.release["profiles"][profile_id]
            timing_profile_id = str(profile["timing_profile"])
            timing = self.release["timing_profiles"][timing_profile_id]
            distribution = timing["game_quantile_distribution"]
            record = {
                "schema_version": 1,
                "contract": COLLECTOR_IDENTITY_ASSIGNMENT_CONTRACT,
                "game_id": game_id,
                "family": family,
                "identity_mode": identity_mode,
                "profile_id": profile_id,
                "renderer": profile["renderer"],
                "prompt_contract": profile["prompt_contract"],
                "timing_profile_id": timing_profile_id,
                "game_speed_quantile": round(self._sample_triangular(distribution), 9),
                "timing": {**copy.deepcopy(timing), "move_multipliers": copy.deepcopy(self.release["move_multipliers"][family])},
                "policy_revision": self.release["revision"],
                "release_sha256": self.release_sha256,
            }
            self._validate_assignment(record)
            self._append(record)
            self._assignments[game_id] = copy.deepcopy(record)
            return copy.deepcopy(record), True

    def fallback_message(self, game: Mapping[str, object], action: Mapping[str, object]) -> str | None:
        """Return collector-specific emergency wording without changing a protocol-constrained binary signal."""
        if not _message_action(game, action):
            return None
        family = str(game.get("game_family") or "")
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        if family == "persuasion":
            original = str(action.get("message") or "").strip()
            channel = str(state.get("seller_message_type") or "text").casefold()
            polarity, _act, _fingerprint = classify_persuasion_signal(original, channel=channel)
            if channel == "binary":
                return original or None
            key = f"persuasion_{polarity if polarity in {'positive', 'negative'} else 'unknown'}"
        else:
            key = family
        values = self.release["fallback_messages"][key]
        return _pick(game, "collector-fallback", key, values)

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "contract": COLLECTOR_IDENTITY_POLICY_CONTRACT,
            "revision": self.release["revision"],
            "release_path": str(self.release_path),
            "release_sha256": self.release_sha256,
            "assignment_contract": COLLECTOR_IDENTITY_ASSIGNMENT_CONTRACT,
            "assignment_count": len(self._assignments),
            "live_authority": False,
        }
