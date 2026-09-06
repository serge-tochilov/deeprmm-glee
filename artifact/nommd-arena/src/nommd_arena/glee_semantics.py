"""Explicit model-facing semantics for opaque GLEE transport fields."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

PERSUASION_INFORMATION_SEMANTICS_VERSION = "persuasion-information-v1"
BARGAINING_CANONICAL_SEMANTICS_VERSION = "bargaining-canonical-v1"
BARGAINING_HISTORY_SUMMARY_VERSION = "bargaining-plateau-summary-v1"
BARGAINING_SEMANTIC_STAIRCASE_VERSION = "bargaining-semantic-staircase-v1"
NEGOTIATION_HISTORY_SUMMARY_VERSION = "negotiation-history-summary-v1"

_BARGAINING_RAW_HISTORY_LIMIT = 8
_BARGAINING_PLATEAU_MINIMUM_OFFERS_PER_PLAYER = 3
_BARGAINING_STAIRCASE_WINDOW = 6
_BARGAINING_STAIRCASE_MINIMUM_OFFERS = 5
_BARGAINING_STAIRCASE_MAXIMUM_SHARE_STEP = 0.02
_BARGAINING_STAIRCASE_MAXIMUM_RELATIVE_STEP_DEVIATION = 0.35
_NEGOTIATION_RAW_HISTORY_LIMIT = 8

_BARGAINING_ROLES = {"player_1": "Alice", "player_2": "Bob"}
_RATIO_CLAIM = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*([/:])\s*(\d+(?:\.\d+)?)(?!\d)")
_ROLE_AMOUNT_CLAIM = re.compile(
    r"\b(Alice|Bob)\b\s*(?:=|:|gets?|receives?|is\s+offered|share(?:\s+is)?)\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(%)?",
    re.IGNORECASE,
)
_AUTHORITY_TERMS = (
    ("rubinstein", re.compile(r"\brubinstein\b", re.IGNORECASE)),
    ("subgame-perfect", re.compile(r"\bsubgame[ -]perfect\b", re.IGNORECASE)),
    ("equilibrium", re.compile(r"\bequilibrium\b", re.IGNORECASE)),
)
_CONTINUATION_TERMS = (
    ("clock", re.compile(r"\bclock\b", re.IGNORECASE)),
    ("continuation", re.compile(r"\bcontinuation\b", re.IGNORECASE)),
    ("delay", re.compile(r"\bdelay(?:ed|ing|s)?\b", re.IGNORECASE)),
    ("wait", re.compile(r"\bwait(?:ed|ing|s)?\b", re.IGNORECASE)),
    ("discount", re.compile(r"\bdiscount(?:ed|ing|s)?\b", re.IGNORECASE)),
    ("erosion", re.compile(r"\berod(?:e|ed|es|ing)\b", re.IGNORECASE)),
    ("shrink", re.compile(r"\bshrink(?:s|ing)?\b|\bshrank\b", re.IGNORECASE)),
    ("value-loss", re.compile(r"\b(?:value|worth|pot)\b[^.!?]{0,36}\b(?:falls?|drops?|declines?|shrinks?|decreases?|los(?:e|es|ing))\b|\b(?:falls?|drops?|declines?|shrinks?|decreases?)\b[^.!?]{0,36}\b(?:value|worth|pot)\b", re.IGNORECASE)),
    ("inflation", re.compile(r"\binflation\b", re.IGNORECASE)),
    ("round-by-round", re.compile(r"\bround[ -]by[ -]round\b|\b(?:each|every)\s+round\b", re.IGNORECASE)),
    ("round-cost", re.compile(r"\b(?:costs?|los(?:e|es|ing)|per)\b[^.!?]{0,36}\brounds?\b|\brounds?\b[^.!?]{0,36}\b(?:costs?|los(?:e|es|ing))\b", re.IGNORECASE)),
)
_DISCOUNT_PERCENT_CLAIM = re.compile(r"\b(I|you|Alice|Bob)\b\s+(?:lose|loses|forfeit|forfeits)\s+(?:about\s+|approximately\s+)?(\d+(?:\.\d+)?)\s*%(?:\s+(?:per|each|every)\s+round)?", re.IGNORECASE)
_BOTH_DECAY_CLAIM = re.compile(r"\b(?:costs?\s+us\s+both|both\s+(?:lose|shrink|erode)|shrinks?[^.!?]{0,36}\b(?:for|to)\s+both(?:\s+of\s+us)?)\b", re.IGNORECASE)
_RECIPIENT_MORE_DECAY_CLAIM = re.compile(r"\b(?:more\s+for\s+you\s+than\s+for\s+me|costs?\s+you[^.!?]{0,24}\bmore\s+than\s+(?:it\s+costs?\s+)?me)\b", re.IGNORECASE)
_NUMERIC_MESSAGE_FRAGMENT = re.compile(r"(?<![A-Za-z])[$€£]?\d[\d,]*(?:\.\d+)?(?:\s*%)?")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _offer_gain(offer: dict[str, Any], player: str) -> float | None:
    aliases = {"player_1": ("player_1_gain", "alice_gain"), "player_2": ("player_2_gain", "bob_gain")}
    for key in aliases[player]:
        value = _finite_number(offer.get(key))
        if value is not None:
            return value
    return None


def _message_audit(message: str, *, money: float | None, authenticated: dict[str, float | None], discounts: dict[str, float | None], proposer_role: str | None, recipient_role: str | None) -> dict[str, object]:
    authority_terms = [label for label, pattern in _AUTHORITY_TERMS if pattern.search(message)]
    continuation_terms = [label for label, pattern in _CONTINUATION_TERMS if pattern.search(message)]
    ratio_claims: list[dict[str, object]] = []
    flags: list[str] = []
    contradictions: list[str] = []
    for match in _RATIO_CLAIM.finditer(message):
        left = float(match.group(1))
        right = float(match.group(3))
        prefix = message[max(0, match.start() - 24) : match.start()].casefold()
        if re.search(r"alice\s*/\s*bob\s*$", prefix):
            order = ["Alice", "Bob"]
        elif re.search(r"bob\s*/\s*alice\s*$", prefix):
            order = ["Bob", "Alice"]
        else:
            order = None
        status = "ordered-role-label"
        conflict = False
        if order is None and math.isclose(left + right, 100.0, rel_tol=0, abs_tol=1e-6):
            status = "unlabelled-order-ambiguous"
            flags.append("unlabelled_ratio_order")
        elif order is not None and money not in (None, 0) and math.isclose(left + right, 100.0, rel_tol=0, abs_tol=1e-6):
            expected = [authenticated[role] / money * 100 if authenticated[role] is not None else None for role in order]
            conflict = any(value is not None and not math.isclose(claimed, value, rel_tol=0, abs_tol=max(1e-6, abs(value) * 1e-9)) for claimed, value in zip((left, right), expected, strict=True))
            if conflict:
                flags.append("explicit_allocation_conflict")
                contradictions.append(f"Ordered ratio {match.group(0)} for {'/'.join(order)} conflicts with the authenticated allocation.")
        ratio_claims.append({"raw": match.group(0), "left": left, "right": right, "role_order": order, "status": status, "conflict": conflict})
    explicit_claims: list[dict[str, object]] = []
    for match in _ROLE_AMOUNT_CLAIM.finditer(message):
        role = match.group(1).title()
        claimed = float(match.group(2).replace(",", ""))
        percent = bool(match.group(3))
        expected_amount = authenticated.get(role)
        expected = expected_amount / money * 100 if percent and expected_amount is not None and money not in (None, 0) else (None if percent else expected_amount)
        conflict = expected is not None and not math.isclose(claimed, expected, rel_tol=0, abs_tol=max(1e-6, abs(expected) * 1e-9))
        explicit_claims.append({"role": role, "claimed_value": claimed, "unit": "percent" if percent else "amount", "authenticated_value": expected, "conflict": conflict})
        if conflict:
            contradictions.append(f"{role} message claim {claimed:g}{'%' if percent else ''} conflicts with authenticated {'fraction' if percent else 'amount'} {expected:g}{'%' if percent else ''}.")
    discount_claims: list[dict[str, object]] = []
    subject_roles = {"i": proposer_role, "you": recipient_role, "alice": "Alice", "bob": "Bob"}
    for match in _DISCOUNT_PERCENT_CLAIM.finditer(message):
        subject = match.group(1)
        role = subject_roles.get(subject.casefold())
        claimed_loss = float(match.group(2))
        discount = discounts.get(role) if role is not None else None
        authenticated_loss = (1 - discount) * 100 if discount is not None else None
        conflict = authenticated_loss is not None and not math.isclose(claimed_loss, authenticated_loss, rel_tol=0.0, abs_tol=0.05)
        discount_claims.append({"raw": match.group(0), "subject": subject, "resolved_role": role, "claimed_loss_percent_per_round": claimed_loss, "authenticated_discount_factor": discount, "authenticated_loss_percent_per_round": authenticated_loss, "conflict": conflict})
        if conflict:
            contradictions.append(f"{subject} discount claim of {claimed_loss:g}% loss per round conflicts with authenticated {role} loss of {authenticated_loss:g}% per round.")
    joint_discount_claim: dict[str, object] | None = None
    if _BOTH_DECAY_CLAIM.search(message):
        known_losses = {role: (1 - discount) * 100 for role, discount in discounts.items() if discount is not None}
        conflict = len(known_losses) == 2 and any(math.isclose(loss, 0.0, rel_tol=0.0, abs_tol=1e-9) for loss in known_losses.values())
        joint_discount_claim = {"claim": "both-players-lose-value-through-waiting", "authenticated_loss_percent_per_round": known_losses, "conflict": conflict}
        if conflict:
            contradictions.append("The message claims that waiting reduces both players' values, but at least one authenticated discount factor is 1.")
    comparative_discount_claim: dict[str, object] | None = None
    if _RECIPIENT_MORE_DECAY_CLAIM.search(message):
        proposer_discount = discounts.get(proposer_role) if proposer_role is not None else None
        recipient_discount = discounts.get(recipient_role) if recipient_role is not None else None
        proposer_loss = (1 - proposer_discount) * 100 if proposer_discount is not None else None
        recipient_loss = (1 - recipient_discount) * 100 if recipient_discount is not None else None
        conflict = proposer_loss is not None and recipient_loss is not None and not recipient_loss > proposer_loss + 1e-9
        comparative_discount_claim = {"claim": "recipient-loses-more-than-proposer", "proposer_role": proposer_role, "recipient_role": recipient_role, "proposer_loss_percent_per_round": proposer_loss, "recipient_loss_percent_per_round": recipient_loss, "conflict": conflict}
        if conflict:
            contradictions.append("The message claims that the recipient loses more through delay than the proposer, but the authenticated discount factors do not support that ordering.")
    if authority_terms:
        flags.append("unverified_equilibrium_authority")
    if continuation_terms:
        flags.append("unverified_continuation_or_discount_claim")
    if any(claim["conflict"] for claim in discount_claims) or bool(joint_discount_claim and joint_discount_claim["conflict"]) or bool(comparative_discount_claim and comparative_discount_claim["conflict"]):
        flags.append("explicit_discount_conflict")
    if any(claim["conflict"] for claim in ratio_claims) or any(claim["conflict"] for claim in explicit_claims):
        flags.append("explicit_allocation_conflict")
    return {
        "message_present": bool(message),
        "authority_terms": authority_terms,
        "authority_status": "unverified-opponent-assertion" if authority_terms else "none",
        "continuation_terms": continuation_terms,
        "continuation_status": "unverified-opponent-framing" if continuation_terms else "none",
        "ratio_claims": ratio_claims,
        "explicit_role_claims": explicit_claims,
        "authenticated_discount_factors": discounts,
        "explicit_discount_claims": discount_claims,
        "joint_discount_claim": joint_discount_claim,
        "comparative_discount_claim": comparative_discount_claim,
        "contradictions": contradictions,
        "semantic_risk_flags": list(dict.fromkeys(flags)),
    }


def canonical_bargaining_facts(game: dict[str, Any]) -> dict[str, object] | None:
    """Project role-resolved bargaining facts before opponent-controlled prose."""
    if game.get("game_family") != "bargaining":
        return None
    state = game.get("game_state")
    if not isinstance(state, dict):
        return None
    self_player = str(game.get("your_player") or state.get("current_player") or "")
    if self_player not in _BARGAINING_ROLES:
        return None
    opponent_player = "player_2" if self_player == "player_1" else "player_1"
    money = _finite_number(state.get("money_to_divide"))
    round_number = state.get("round")
    maximum = state.get("max_rounds")
    rounds_remaining = maximum - round_number if state.get("horizon_known") is True and isinstance(maximum, int) and isinstance(round_number, int) else None
    discount = _finite_number(state.get("delta_1" if self_player == "player_1" else "delta_2"))
    opponent_discount = _finite_number(state.get("delta_2" if self_player == "player_1" else "delta_1"))
    result: dict[str, object] = {
        "version": BARGAINING_CANONICAL_SEMANTICS_VERSION,
        "source": "engine-authenticated visible state",
        "self_player": self_player,
        "self_role": _BARGAINING_ROLES[self_player],
        "opponent_player": opponent_player,
        "opponent_role": _BARGAINING_ROLES[opponent_player],
        "money_to_divide": money,
        "round": round_number,
        "horizon_known": state.get("horizon_known"),
        "max_rounds": maximum,
        "rounds_remaining_after_current": rounds_remaining,
        "self_discount_factor": discount,
        "opponent_discount_factor": opponent_discount,
    }
    offer = state.get("last_offer")
    if not isinstance(offer, dict):
        result["offer_status"] = "no-current-offer"
        return result
    player_1_gain = _offer_gain(offer, "player_1")
    player_2_gain = _offer_gain(offer, "player_2")
    gains = {"player_1": player_1_gain, "player_2": player_2_gain}
    self_amount = gains[self_player]
    opponent_amount = gains[opponent_player]
    message = str(offer.get("message") or "")
    proposer_player = str(offer.get("proposer") or state.get("proposer") or "")
    proposer_role = _BARGAINING_ROLES.get(proposer_player)
    recipient_role = _BARGAINING_ROLES.get("player_2" if proposer_player == "player_1" else "player_1") if proposer_player in _BARGAINING_ROLES else None
    discounts = {"Alice": _finite_number(state.get("delta_1")), "Bob": _finite_number(state.get("delta_2"))}
    result.update(
        {
            "offer_status": "authenticated-current-offer",
            "proposer": offer.get("proposer") or state.get("proposer"),
            "self_amount": self_amount,
            "self_fraction": self_amount / money if self_amount is not None and money not in (None, 0) else None,
            "opponent_amount": opponent_amount,
            "opponent_fraction": opponent_amount / money if opponent_amount is not None and money not in (None, 0) else None,
            "authenticated_allocation": {"Alice": player_1_gain, "Bob": player_2_gain},
            "message_audit": _message_audit(message, money=money, authenticated={"Alice": player_1_gain, "Bob": player_2_gain}, discounts=discounts, proposer_role=proposer_role, recipient_role=recipient_role),
        }
    )
    return result


def persuasion_information_semantics(seller_knows_buyer_values: object = None) -> dict[str, object]:
    """Expose a versioned, instruction-free persuasion semantics receipt."""
    semantics: dict[str, object] = {"version": PERSUASION_INFORMATION_SEMANTICS_VERSION}
    if isinstance(seller_knows_buyer_values, bool):
        semantics["seller_knows_buyer_values"] = seller_knows_buyer_values
    return semantics


def _negotiation_price_summary(entries: list[dict[str, Any]], player: str) -> dict[str, object]:
    prices = [float(entry["offer"]["price"]) for entry in entries if isinstance(entry.get("offer"), dict) and entry["offer"].get("from_player") == player and _finite_number(entry["offer"].get("price")) is not None]
    messages = [" ".join(str(entry["offer"].get("message") or "").split()) for entry in entries if isinstance(entry.get("offer"), dict) and entry["offer"].get("from_player") == player]
    nonempty_messages = [message for message in messages if message]
    deltas = [round(current - previous, 6) for previous, current in zip(prices, prices[1:])]
    modal_delta = Counter(deltas).most_common(1)[0][0] if deltas else None
    plateau_length = 0
    if prices:
        plateau_length = 1
        for price in reversed(prices[:-1]):
            if not math.isclose(price, prices[-1], rel_tol=0, abs_tol=max(1e-9, abs(prices[-1]) * 1e-9)):
                break
            plateau_length += 1
    latest_message_run = 0
    if messages:
        latest = messages[-1].casefold()
        for message in reversed(messages):
            if message.casefold() != latest:
                break
            latest_message_run += 1
    return {
        "offer_count": len(prices),
        "first_price": prices[0] if prices else None,
        "latest_price": prices[-1] if prices else None,
        "minimum_price": min(prices) if prices else None,
        "maximum_price": max(prices) if prices else None,
        "modal_price_step": modal_delta,
        "recent_price_steps": deltas[-4:],
        "latest_price_plateau_length": plateau_length,
        "nonempty_message_count": len(nonempty_messages),
        "distinct_message_count": len({message.casefold() for message in nonempty_messages}),
        "latest_message": nonempty_messages[-1] if nonempty_messages else None,
        "latest_identical_message_run": latest_message_run,
    }


def _bargaining_allocation(offer: dict[str, Any]) -> tuple[float, float] | None:
    alice = _offer_gain(offer, "player_1")
    bob = _offer_gain(offer, "player_2")
    if alice is None or bob is None:
        return None
    return alice, bob


def _same_bargaining_allocation(left: tuple[float, float], right: tuple[float, float]) -> bool:
    scale = max(1.0, abs(left[0]), abs(left[1]), abs(right[0]), abs(right[1]))
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=scale * 1e-9) for a, b in zip(left, right, strict=True))


def _bargaining_player_summary(entries: list[dict[str, Any]], player: str) -> dict[str, object]:
    offers = [entry["offer"] for entry in entries if isinstance(entry.get("offer"), dict) and str(entry.get("proposer") or entry["offer"].get("proposer") or "") == player]
    allocations = [allocation for offer in offers if (allocation := _bargaining_allocation(offer)) is not None]
    messages = [" ".join(str(offer.get("message") or "").split()) for offer in offers]
    plateau_length = 0
    if allocations:
        plateau_length = 1
        for allocation in reversed(allocations[:-1]):
            if not _same_bargaining_allocation(allocation, allocations[-1]):
                break
            plateau_length += 1
    nonempty_messages = [message for message in messages if message]
    return {
        "offer_count": len(allocations),
        "first_allocation": {"Alice": allocations[0][0], "Bob": allocations[0][1]} if allocations else None,
        "latest_allocation": {"Alice": allocations[-1][0], "Bob": allocations[-1][1]} if allocations else None,
        "latest_allocation_plateau_length": plateau_length,
        "nonempty_message_count": len(nonempty_messages),
        "distinct_message_count": len({message.casefold() for message in nonempty_messages}),
        "latest_message": nonempty_messages[-1] if nonempty_messages else None,
    }


def bargaining_plateau_summary(game: dict[str, Any]) -> dict[str, object] | None:
    """Return a deterministic bilateral-plateau receipt without changing authenticated history."""
    if game.get("game_family") != "bargaining":
        return None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    if not isinstance(history, list) or len(history) <= _BARGAINING_RAW_HISTORY_LIMIT:
        return None
    entries = [entry for entry in history if isinstance(entry, dict) and isinstance(entry.get("offer"), dict)]
    players = {player: _bargaining_player_summary(entries, player) for player in _BARGAINING_ROLES}
    if any(int(summary["latest_allocation_plateau_length"]) < _BARGAINING_PLATEAU_MINIMUM_OFFERS_PER_PLAYER for summary in players.values()):
        return None
    serialized = json.dumps(history, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "version": BARGAINING_HISTORY_SUMMARY_VERSION,
        "classification": "bilateral-numeric-plateau",
        "total_entry_count": len(history),
        "history_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "minimum_latest_offer_repetitions_per_player": _BARGAINING_PLATEAU_MINIMUM_OFFERS_PER_PLAYER,
        "players": players,
        "decision_counts": dict(sorted(Counter(str(entry.get("decision") or "unknown") for entry in entries).items())),
        "interpretation_boundary": "The summary proves repeated numeric allocations only; it does not infer an undocumented round limit, agreement probability, or strategic commitment.",
    }


def _semantic_message_template(message: object) -> str:
    normalized = " ".join(str(message or "").casefold().split())
    if not normalized:
        return "<silent>"
    return _NUMERIC_MESSAGE_FRAGMENT.sub("<n>", normalized)


def bargaining_semantic_staircase_summary(game: dict[str, Any]) -> dict[str, object] | None:
    """Detect a recent low-information opponent staircase without inferring its motive or future policy."""
    if game.get("game_family") != "bargaining":
        return None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    player = str(game.get("your_player") or state.get("current_player") or "")
    money = _finite_number(state.get("money_to_divide"))
    if not isinstance(history, list) or player not in _BARGAINING_ROLES or money is None or money <= 0:
        return None
    opponent = "player_2" if player == "player_1" else "player_1"
    observed: list[dict[str, object]] = []
    for entry in history:
        if not isinstance(entry, dict) or not isinstance(entry.get("offer"), dict):
            continue
        offer = entry["offer"]
        proposer = str(entry.get("proposer") or offer.get("proposer") or "")
        allocation = _bargaining_allocation(offer)
        if proposer != opponent or allocation is None:
            continue
        own_amount = allocation[0] if opponent == "player_1" else allocation[1]
        observed.append(
            {
                "round": int(entry.get("round") or offer.get("round") or 0),
                "opponent_own_share": own_amount / money,
                "message_template": _semantic_message_template(offer.get("message")),
            }
        )
    if len(observed) < _BARGAINING_STAIRCASE_MINIMUM_OFFERS:
        return None
    recent = observed[-_BARGAINING_STAIRCASE_WINDOW:]
    shares = [float(item["opponent_own_share"]) for item in recent]
    concessions = [previous - current for previous, current in zip(shares, shares[1:])]
    tolerance = 1e-9
    positive = [step for step in concessions if step > tolerance]
    if len(positive) < _BARGAINING_STAIRCASE_MINIMUM_OFFERS - 2:
        return None
    if any(step < -tolerance or step > _BARGAINING_STAIRCASE_MAXIMUM_SHARE_STEP + tolerance for step in concessions):
        return None
    mean_step = sum(positive) / len(positive)
    maximum_relative_deviation = max(abs(step - mean_step) / mean_step for step in positive)
    if maximum_relative_deviation > _BARGAINING_STAIRCASE_MAXIMUM_RELATIVE_STEP_DEVIATION:
        return None
    template_counts = Counter(str(item["message_template"]) for item in recent)
    modal_template, modal_count = template_counts.most_common(1)[0]
    if modal_count / len(recent) < 0.8 or len(template_counts) > 2:
        return None
    serialized = json.dumps(recent, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "version": BARGAINING_SEMANTIC_STAIRCASE_VERSION,
        "classification": "low-information-affine-opponent-staircase",
        "opponent_player": opponent,
        "observed_opponent_offer_count": len(observed),
        "recent_offer_count": len(recent),
        "recent_rounds": [int(item["round"]) for item in recent],
        "first_recent_opponent_own_share": shares[0],
        "latest_opponent_own_share": shares[-1],
        "recent_concessions_toward_us": concessions,
        "mean_positive_concession": mean_step,
        "maximum_relative_step_deviation": maximum_relative_deviation,
        "maximum_allowed_share_step": _BARGAINING_STAIRCASE_MAXIMUM_SHARE_STEP,
        "modal_message_template_sha256": hashlib.sha256(modal_template.encode("utf-8")).hexdigest(),
        "modal_message_template_fraction": modal_count / len(recent),
        "distinct_message_template_count": len(template_counts),
        "recent_evidence_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "interpretation_boundary": "This classifies a recent low-information numeric and message pattern only; it does not infer resource-exhaustion intent, a future concession, or an acceptance threshold.",
    }


def _compact_bargaining_state(game: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    summary = bargaining_plateau_summary(game)
    if summary is None:
        return state
    history = state.get("history")
    if not isinstance(history, list):
        return state
    retained = copy.deepcopy(history[-_BARGAINING_RAW_HISTORY_LIMIT:])
    state["history"] = retained
    state["history_summary"] = {
        **summary,
        "compression": "deterministic-plateau-summary-plus-recent-raw-entries",
        "omitted_early_entry_count": len(history) - len(retained),
        "retained_recent_entry_count": len(retained),
        "retained_rounds": [int(entry.get("round") or entry.get("offer", {}).get("round") or 0) for entry in retained if isinstance(entry, dict)],
    }
    return state


def _compact_negotiation_state(state: dict[str, Any]) -> dict[str, Any]:
    history = state.get("history")
    if not isinstance(history, list) or len(history) <= _NEGOTIATION_RAW_HISTORY_LIMIT:
        return state
    entries = [entry for entry in history if isinstance(entry, dict) and isinstance(entry.get("offer"), dict)]
    players = sorted({str(entry["offer"].get("from_player")) for entry in entries if entry["offer"].get("from_player")})
    decisions = Counter(str(entry.get("decision") or "unknown") for entry in entries)
    retained = copy.deepcopy(history[-_NEGOTIATION_RAW_HISTORY_LIMIT:])
    state["history"] = retained
    state["history_summary"] = {
        "version": NEGOTIATION_HISTORY_SUMMARY_VERSION,
        "compression": "deterministic-summary-plus-recent-raw-entries",
        "total_entry_count": len(history),
        "omitted_early_entry_count": len(history) - len(retained),
        "retained_recent_entry_count": len(retained),
        "retained_rounds": [int(entry.get("round") or entry.get("offer", {}).get("round") or 0) for entry in retained if isinstance(entry, dict)],
        "decision_counts": dict(sorted(decisions.items())),
        "players": {player: _negotiation_price_summary(entries, player) for player in players},
        "interpretation_boundary": "Summary statistics preserve concession, plateau, and message-repetition evidence; exact recent actions remain in history.",
    }
    return state


def model_official_prompt(game: dict[str, Any]) -> str | None:
    """Remove duplicated old history lines after a deterministic model-facing summary is available."""
    prompt = game.get("prompt")
    family = game.get("game_family")
    if not isinstance(prompt, str) or family not in {"bargaining", "negotiation"}:
        return prompt if isinstance(prompt, str) else None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history")
    marker = "Previous rounds:\n"
    bargaining_compacted = family == "bargaining" and bargaining_plateau_summary(game) is not None
    negotiation_compacted = family == "negotiation" and isinstance(history, list) and len(history) > _NEGOTIATION_RAW_HISTORY_LIMIT
    if not isinstance(history, list) or not (bargaining_compacted or negotiation_compacted) or marker not in prompt:
        return prompt
    before, after = prompt.split(marker, 1)
    lines = after.splitlines(keepends=True)
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].startswith("  Round ")):
        index += 1
    summary_kind = "deterministic plateau summary" if bargaining_compacted else "deterministic summary"
    replacement = f"Previous rounds: {summary_kind} and the latest 8 raw entries are provided in visible_game_state; omitted lines are not additional evidence.\n"
    return before + replacement + "".join(lines[index:])


def model_visible_game_state(game: dict[str, Any]) -> dict[str, Any]:
    """Copy visible engine state while replacing opaque persuasion vocabulary with explicit semantics."""
    state = copy.deepcopy(game["game_state"])
    if game.get("game_family") == "bargaining":
        return _compact_bargaining_state(game, state)
    if game.get("game_family") == "negotiation":
        return _compact_negotiation_state(state)
    if game.get("game_family") != "persuasion":
        return state
    raw = state.pop("is_seller_know_cv", None)
    if isinstance(raw, bool):
        state["seller_knows_buyer_values"] = raw
    state["seller_always_observes_current_quality"] = True
    return state


def model_visible_game(game: dict[str, Any]) -> dict[str, Any]:
    """Copy a complete engine turn while normalizing its model-facing state."""
    visible = copy.deepcopy(game)
    visible["game_state"] = model_visible_game_state(game)
    return visible


def model_game_semantics(game: dict[str, Any]) -> dict[str, object] | None:
    """Return explicit family semantics for the model-facing turn or evidence packet."""
    if game.get("game_family") != "persuasion":
        return None
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    return persuasion_information_semantics(state.get("is_seller_know_cv", state.get("seller_knows_buyer_values")))


def model_static_game_context(game: dict[str, Any]) -> dict[str, object]:
    """Project stable game configuration with semantic field names."""
    family = str(game["game_family"])
    state = model_visible_game_state(game)
    keys = {
        "bargaining": ("money_to_divide", "delta_1", "delta_2", "complete_information", "horizon_known", "max_rounds", "messages_allowed"),
        "negotiation": ("player_1_role", "player_2_role", "player_1_value", "player_2_value", "complete_information", "horizon_known", "max_rounds", "messages_allowed"),
        "persuasion": ("product_price", "p", "u", "v", "total_rounds", "seller_knows_buyer_values", "seller_always_observes_current_quality", "seller_message_type"),
    }[family]
    return {"game_family": family, "your_player": game.get("your_player"), **{key: state[key] for key in keys if key in state}}
