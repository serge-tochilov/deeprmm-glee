"""GLEE action schemas, normalization, deterministic fallbacks, and payoff guards."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GLEE_FAMILIES = ("bargaining", "negotiation", "persuasion")

_GLEE_CREDENTIAL = re.compile(r"\bglee_[A-Za-z0-9_-]{6,}\b")


@dataclass(frozen=True)
class AnalyticBargainingOpening:
    """A complete-information round-1 equilibrium proposal with an auditable derivation."""

    action: dict[str, float]
    method: str
    proposer: str
    delta_1: float
    delta_2: float
    responder_gain: float
    max_rounds: int | None


@dataclass(frozen=True)
class AnalyticBargainingReference:
    """A current-round complete-information equilibrium calculated outside model inference."""

    equilibrium_offer: dict[str, float]
    method: str
    round_number: int
    proposer: str
    responder: str
    delta_1: float
    delta_2: float
    responder_gain: float
    max_rounds: int | None


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BargainingOffer(_Action):
    alice_gain: float = Field(ge=0)
    bob_gain: float = Field(ge=0)
    message: str | None = Field(default=None, max_length=2000)


class BargainingDecision(_Action):
    decision: Literal["accept", "reject", "walkaway"]


class NegotiationOffer(_Action):
    product_price: float = Field(ge=0)
    message: str | None = Field(default=None, max_length=2000)


class NegotiationDecision(_Action):
    decision: Literal["AcceptOffer", "RejectOffer", "WalkAway"]
    product_price: float | None = Field(default=None, ge=0)
    message: str | None = Field(default=None, max_length=2000)


class PersuasionMessage(_Action):
    message: str = Field(min_length=1, max_length=2000)


class PersuasionDecision(_Action):
    decision: Literal["yes", "no"]


def action_model(game: dict[str, Any]) -> type[_Action]:
    family = game["game_family"]
    action_type = game["valid_actions"]["type"]
    models: dict[tuple[str, str], type[_Action]] = {
        ("bargaining", "offer"): BargainingOffer,
        ("bargaining", "decision"): BargainingDecision,
        ("negotiation", "offer"): NegotiationOffer,
        ("negotiation", "decision"): NegotiationDecision,
        ("persuasion", "seller_message"): PersuasionMessage,
        ("persuasion", "seller_recommendation"): PersuasionDecision,
        ("persuasion", "buyer_decision"): PersuasionDecision,
    }
    try:
        return models[(family, action_type)]
    except KeyError as error:
        raise ValueError(f"unsupported GLEE action contract: {family}/{action_type}") from error


def _messages_allowed(game: dict[str, Any]) -> bool:
    if game["game_family"] == "persuasion":
        return True
    state = game["game_state"]
    if state.get("messages_allowed") is False:
        return False
    fields = game["valid_actions"].get("fields")
    return not isinstance(fields, dict) or not fields or "message" in fields


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _final_round(state: dict[str, Any]) -> bool:
    maximum = state.get("max_rounds")
    return bool(state.get("horizon_known") and isinstance(maximum, int) and state.get("round", 0) >= maximum)


def _negotiation_role_value(game: dict[str, Any]) -> tuple[str, float]:
    state = game["game_state"]
    me = str(game.get("your_player") or state.get("current_player"))
    role = str(state[f"{me}_role"])
    value = _finite_nonnegative(state[f"{me}_value"], f"{me}_value")
    if role not in {"seller", "buyer"}:
        raise ValueError(f"unsupported negotiation role: {role}")
    return role, value


def _negotiation_margin(role: str, value: float, price: float) -> float:
    return price - value if role == "seller" else value - price


def _positive_surplus_tick(value: float) -> float:
    """Return the smallest generic reservation nudge used when a positive own surplus is feasible."""
    return max(0.01, abs(value) * 1e-6)


def _one_round_complete_information_offer(game: dict[str, Any]) -> dict[str, float] | None:
    """Return a positive-surplus boundary offer when both one-shot reservation values are visible."""
    state = game["game_state"]
    if state.get("complete_information") is not True or state.get("horizon_known") is not True or state.get("round") != 1 or state.get("max_rounds") != 1:
        return None
    me = str(game.get("your_player") or state.get("current_player"))
    other = "player_2" if me == "player_1" else "player_1" if me == "player_2" else None
    if other is None:
        return None
    role, value = _negotiation_role_value(game)
    other_role = str(state.get(f"{other}_role"))
    try:
        other_value = _finite_nonnegative(state.get(f"{other}_value"), f"{other}_value")
    except ValueError:
        return None
    tick = 0.01
    if role == "seller" and other_role == "buyer" and other_value - tick >= value:
        return {"product_price": other_value - tick}
    if role == "buyer" and other_role == "seller" and other_value + tick <= value:
        return {"product_price": other_value + tick}
    return None


def _bargaining_gain(offer: dict[str, Any], player: str) -> float:
    aliases = {"player_1": ("player_1_gain", "alice_gain"), "player_2": ("player_2_gain", "bob_gain")}
    for key in aliases.get(player, ()):
        if key in offer:
            return _finite_nonnegative(offer[key], key)
    return 0.0


def _discount_factor(state: dict[str, Any], key: str, *, include_boundaries: bool) -> float | None:
    raw = state.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value):
        return None
    if include_boundaries:
        return value if 0 <= value <= 1 else None
    return value if 0 < value < 1 else None


def _asymmetric_unit_discount_boundary(game: dict[str, Any]) -> bool:
    """Identify the unknown-horizon boundary where only the opponent is undiscounted."""
    state = game["game_state"]
    me = str(game.get("your_player") or state.get("current_player"))
    other = {"player_1": "player_2", "player_2": "player_1"}.get(me)
    if other is None:
        return False
    own_delta = _discount_factor(state, "delta_1" if me == "player_1" else "delta_2", include_boundaries=False)
    other_delta = _discount_factor(state, "delta_1" if other == "player_1" else "delta_2", include_boundaries=True)
    return state.get("complete_information") is True and state.get("horizon_known") is False and own_delta is not None and other_delta == 1.0


def analytic_bargaining_reference(game: dict[str, Any]) -> AnalyticBargainingReference | None:
    """Precompute the current complete-information bargaining subgame without model reasoning."""
    if game.get("game_family") != "bargaining" or game.get("valid_actions", {}).get("type") not in {"offer", "decision"}:
        return None
    state = game.get("game_state")
    if not isinstance(state, dict) or state.get("complete_information") is not True:
        return None
    round_number = state.get("round")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        return None
    action_type = str(game["valid_actions"]["type"])
    current_player = str(game.get("your_player") or state.get("current_player"))
    other = {"player_1": "player_2", "player_2": "player_1"}
    if current_player not in other:
        return None
    last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), dict) else {}
    proposer = str(last_offer.get("proposer") or state.get("proposer") or (current_player if action_type == "offer" else other[current_player]))
    if proposer not in other:
        return None
    try:
        money = _finite_nonnegative(state.get("money_to_divide"), "money_to_divide")
    except ValueError:
        return None
    if money <= 0:
        return None
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < round_number:
            return None
        delta_1 = _discount_factor(state, "delta_1", include_boundaries=True)
        delta_2 = _discount_factor(state, "delta_2", include_boundaries=True)
        if delta_1 is None or delta_2 is None:
            return None
        deltas = {"player_1": delta_1, "player_2": delta_2}
        next_shares: dict[str, float] | None = None
        for evaluated_round in range(maximum, round_number - 1, -1):
            round_proposer = proposer if (evaluated_round - round_number) % 2 == 0 else other[proposer]
            responder = other[round_proposer]
            responder_gain = 0.0 if next_shares is None else deltas[responder] * next_shares[responder]
            next_shares = {round_proposer: money - responder_gain, responder: responder_gain}
        if next_shares is None:
            return None
        shares = next_shares
        method = "finite-horizon backward induction"
        max_rounds: int | None = maximum
    elif state.get("horizon_known") is False:
        delta_1 = _discount_factor(state, "delta_1", include_boundaries=True)
        delta_2 = _discount_factor(state, "delta_2", include_boundaries=True)
        if delta_1 is None or delta_2 is None:
            return None
        denominator = 1 - delta_1 * delta_2
        if denominator <= 1e-12:
            return None
        if proposer == "player_1":
            player_1_gain = money * (1 - delta_2) / denominator
        else:
            player_1_gain = money * delta_1 * (1 - delta_2) / denominator
        shares = {"player_1": player_1_gain, "player_2": money - player_1_gain}
        method = "infinite-horizon Rubinstein equilibrium"
        max_rounds = None
    else:
        return None
    player_1_gain = min(money, max(0.0, shares["player_1"]))
    action = {"alice_gain": player_1_gain, "bob_gain": money - player_1_gain}
    responder = other[proposer]
    return AnalyticBargainingReference(
        equilibrium_offer=action,
        method=method,
        round_number=round_number,
        proposer=proposer,
        responder=responder,
        delta_1=delta_1,
        delta_2=delta_2,
        responder_gain=shares[responder],
        max_rounds=max_rounds,
    )


def analytic_bargaining_opening(game: dict[str, Any]) -> AnalyticBargainingOpening | None:
    """Return an exact cold opening only for identifiable, nondegenerate bargaining games."""
    state = game.get("game_state")
    if not isinstance(state, dict) or game.get("valid_actions", {}).get("type") != "offer":
        return None
    if state.get("round") != 1 or state.get("history") not in (None, []):
        return None
    reference = analytic_bargaining_reference(game)
    if reference is None or reference.proposer != game.get("your_player"):
        return None
    return AnalyticBargainingOpening(
        action=reference.equilibrium_offer,
        method=reference.method,
        proposer=reference.proposer,
        delta_1=reference.delta_1,
        delta_2=reference.delta_2,
        responder_gain=reference.responder_gain,
        max_rounds=reference.max_rounds,
    )


def _reservation_offer(role: str, value: float) -> dict[str, Any]:
    tick = _positive_surplus_tick(value)
    return {"product_price": value + tick if role == "seller" else max(0.0, value - tick)}


def safe_action(game: dict[str, Any]) -> dict[str, Any]:
    """Return a legal fallback that avoids a deterministically negative own payoff."""
    family = game["game_family"]
    action_type = game["valid_actions"]["type"]
    state = game["game_state"]
    if family == "bargaining":
        reference = analytic_bargaining_reference(game)
        if action_type == "offer":
            me = str(game.get("your_player") or state.get("current_player"))
            if reference is not None and reference.proposer == me:
                return dict(reference.equilibrium_offer)
            half = _finite_nonnegative(state["money_to_divide"], "money_to_divide") / 2
            return {"alice_gain": half, "bob_gain": half}
        me = str(game.get("your_player") or state.get("current_player"))
        offer = state.get("last_offer") or {}
        gain = _bargaining_gain(offer, me)
        money = _finite_nonnegative(state["money_to_divide"], "money_to_divide")
        if reference is not None and reference.responder == me:
            return {"decision": "accept" if gain >= reference.responder_gain - 1e-9 else "reject"}
        if math.isclose(gain, money) or (_final_round(state) and gain > 0):
            return {"decision": "accept"}
        return {"decision": "reject"}
    if family == "negotiation":
        role, value = _negotiation_role_value(game)
        if action_type == "offer":
            complete_information_offer = _one_round_complete_information_offer(game)
            if complete_information_offer is not None:
                return complete_information_offer
            return _reservation_offer(role, value)
        price = _finite_nonnegative((state.get("last_offer") or {}).get("price"), "last_offer.price")
        margin = _negotiation_margin(role, value, price)
        if margin > 0:
            return {"decision": "AcceptOffer"}
        if math.isclose(margin, 0.0, abs_tol=1e-12):
            return {"decision": "WalkAway"}
        if _final_round(state):
            return {"decision": "RejectOffer"}
        counter = _reservation_offer(role, value)
        if _negotiation_margin(role, value, float(counter["product_price"])) <= 0:
            return {"decision": "WalkAway"}
        return {"decision": "RejectOffer", **counter}
    if action_type == "seller_message":
        quality = state.get("current_quality")
        return {"message": "I recommend this product." if quality != "low" else "I do not recommend this product."}
    if action_type == "seller_recommendation":
        return {"decision": "no" if state.get("current_quality") == "low" else "yes"}
    expected = float(state["p"]) * float(state["v"]) + (1 - float(state["p"])) * float(state["u"])
    return {"decision": "yes" if expected > float(state["product_price"]) else "no"}


def normalize_action(game: dict[str, Any], candidate: _Action | dict[str, Any]) -> dict[str, Any]:
    """Convert a schema-valid proposal into a server-safe action or reject it before submission."""
    action = candidate.model_dump(exclude_none=True) if isinstance(candidate, BaseModel) else dict(candidate)
    family = game["game_family"]
    action_type = game["valid_actions"]["type"]
    state = game["game_state"]
    if not _messages_allowed(game):
        action.pop("message", None)
    if "message" in action:
        if not isinstance(action["message"], str):
            raise ValueError("message must be a string")
        action["message"] = action["message"].strip()
        if len(action["message"]) > 2000:
            raise ValueError("message exceeds GLEE's 2000-character limit")
        if not action["message"]:
            action.pop("message")
    if family == "bargaining" and action_type == "offer":
        alice = _finite_nonnegative(action.get("alice_gain"), "alice_gain")
        bob = _finite_nonnegative(action.get("bob_gain"), "bob_gain")
        money = _finite_nonnegative(state["money_to_divide"], "money_to_divide")
        if not math.isclose(alice + bob, money, rel_tol=0, abs_tol=1e-7):
            raise ValueError("bargaining gains must sum exactly to money_to_divide")
        action["alice_gain"] = alice
        action["bob_gain"] = money - alice
    elif family == "negotiation" and action_type == "offer":
        action["product_price"] = _finite_nonnegative(action.get("product_price"), "product_price")
    elif family == "negotiation" and action_type == "decision":
        decision = action.get("decision")
        if decision == "RejectOffer" and not _final_round(state):
            action["product_price"] = _finite_nonnegative(action.get("product_price"), "product_price")
        else:
            action.pop("product_price", None)
            if decision != "RejectOffer":
                action.pop("message", None)
    allowed_fields = game["valid_actions"].get("fields")
    if isinstance(allowed_fields, dict) and allowed_fields:
        action = {name: value for name, value in action.items() if name in allowed_fields}
    action_model(game).model_validate(action)
    if family == "negotiation" and action_type == "decision" and action.get("decision") == "RejectOffer" and not _final_round(state) and "product_price" not in action:
        raise ValueError("a non-final negotiation rejection requires a counteroffer")
    return action


def apply_deterministic_safeguards(game: dict[str, Any], candidate: dict[str, Any], *, preserve_positive_negotiation_exit: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Override only illegal, secret-bearing, or payoff-dominated parts of a normalized model action."""
    action = dict(candidate)
    applied: list[str] = []
    family = game["game_family"]
    action_type = game["valid_actions"]["type"]
    state = game["game_state"]
    message = action.get("message")
    if isinstance(message, str) and _GLEE_CREDENTIAL.search(message):
        action["message"] = _GLEE_CREDENTIAL.sub("glee_[REDACTED]", message)
        applied.append("credential_redaction")
    if family == "negotiation":
        role, value = _negotiation_role_value(game)
        if action_type == "offer":
            price = float(action["product_price"])
            if _negotiation_margin(role, value, price) <= 0:
                action.update(_reservation_offer(role, value))
                applied.append("positive_surplus_offer_floor")
            complete_information_offer = _one_round_complete_information_offer(game)
            if complete_information_offer is not None:
                boundary = complete_information_offer["product_price"]
                if (role == "seller" and float(action["product_price"]) > boundary) or (role == "buyer" and float(action["product_price"]) < boundary):
                    action["product_price"] = boundary
                    applied.append("one_round_positive_surplus_boundary")
        else:
            price = _finite_nonnegative((state.get("last_offer") or {}).get("price"), "last_offer.price")
            margin = _negotiation_margin(role, value, price)
            decision = action["decision"]
            if decision == "AcceptOffer" and math.isclose(margin, 0.0, abs_tol=1e-12):
                action = {"decision": "WalkAway"}
                applied.append("reject_zero_surplus_trade")
            elif decision == "AcceptOffer" and margin < 0:
                action = {"decision": "RejectOffer"}
                if not _final_round(state):
                    action.update(_reservation_offer(role, value))
                applied.append("reject_negative_surplus_trade")
            elif decision == "WalkAway" and margin > 0 and not preserve_positive_negotiation_exit:
                action = {"decision": "AcceptOffer"}
                applied.append("accept_positive_surplus_over_walkaway")
            elif decision == "RejectOffer" and _final_round(state) and margin > 0 and not preserve_positive_negotiation_exit:
                action = {"decision": "AcceptOffer"}
                applied.append("accept_positive_surplus_on_final_round")
            elif decision == "RejectOffer" and not _final_round(state):
                counter = float(action["product_price"])
                if _negotiation_margin(role, value, counter) <= 0:
                    positive = _reservation_offer(role, value)
                    if _negotiation_margin(role, value, float(positive["product_price"])) <= 0:
                        action = {"decision": "WalkAway"}
                    else:
                        action.update(positive)
                    applied.append("positive_surplus_counteroffer_floor")
    elif family == "bargaining":
        me = str(game.get("your_player") or state.get("current_player"))
        money = _finite_nonnegative(state["money_to_divide"], "money_to_divide")
        history = state.get("history")
        if action_type == "offer":
            gain = _bargaining_gain(action, me)
            asymmetric_token_opening = (
                state.get("round") == 1
                and history in (None, [])
                and _asymmetric_unit_discount_boundary(game)
                and money > 0
                and gain / money <= 0.1
            )
            if asymmetric_token_opening:
                half = money / 2
                action = {"alice_gain": half, "bob_gain": money - half}
                applied.append("probe_asymmetric_unit_discount_opening")
        else:
            offer = state.get("last_offer") or {}
            gain = _bargaining_gain(offer, me)
            round_one_undiscounted_lowball = (
                action["decision"] == "accept"
                and state.get("round") == 1
                and history in (None, [])
                and state.get("complete_information") is True
                and state.get("horizon_known") is False
                and _discount_factor(state, "delta_1", include_boundaries=True) == 1.0
                and _discount_factor(state, "delta_2", include_boundaries=True) == 1.0
                and money > 0
                and gain / money <= 1 / 3
            )
            first_received_offer = (state.get("round") == 1 and history in (None, [])) or (state.get("round") == 2 and isinstance(history, list) and len(history) == 1)
            first_asymmetric_token_counter = (
                action["decision"] == "accept"
                and first_received_offer
                and _asymmetric_unit_discount_boundary(game)
                and money > 0
                and gain / money <= 0.1
            )
            if round_one_undiscounted_lowball:
                action = {"decision": "reject"}
                applied.append("reject_round_one_undiscounted_lowball")
            elif first_asymmetric_token_counter:
                action = {"decision": "reject"}
                applied.append("reject_first_asymmetric_unit_discount_token_counter")
            elif action["decision"] == "walkaway" and gain > 0:
                if _final_round(state) or math.isclose(gain, money):
                    action = {"decision": "accept"}
                    applied.append("accept_dominant_bargaining_offer")
                else:
                    action = {"decision": "reject"}
                    applied.append("preserve_continuation_over_walkaway")
            elif action["decision"] == "reject" and gain > 0 and (_final_round(state) or math.isclose(gain, money)):
                action = {"decision": "accept"}
                applied.append("accept_dominant_bargaining_offer")
    elif family == "persuasion" and action_type == "buyer_decision":
        price = float(state["product_price"])
        low = min(float(state["u"]), float(state["v"]))
        high = max(float(state["u"]), float(state["v"]))
        if low >= price and high > price and action["decision"] != "yes":
            action = {"decision": "yes"}
            applied.append("buy_when_all_quality_states_are_nonnegative")
        elif high <= price and low < price and action["decision"] != "no":
            action = {"decision": "no"}
            applied.append("pass_when_no_quality_state_has_positive_surplus")
    return normalize_action(game, action), applied
