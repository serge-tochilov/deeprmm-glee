"""Validated identity-neutral tactic memory shared across GLEE opponents."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .glee_policy import GLEE_FAMILIES
from .glee_semantics import canonical_bargaining_facts

_SCHEMA_VERSION = 1
_LIVE_ROUTES = {
    "bargaining-message-risk",
    "bargaining-extreme-incoming",
    "negotiation-repeated-pair",
    "persuasion-terminal-reputation",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class GlobalTacticLedger:
    """Load a small researcher-curated tactic ledger and expose bounded family views."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is None:
            self._document: dict[str, Any] = {"schema_version": _SCHEMA_VERSION, "tactics": []}
        else:
            if not path.is_file():
                raise FileNotFoundError(f"global tactic ledger is missing: {path}")
            self._document = json.loads(path.read_text(encoding="utf-8"))
        self._validate()
        self.sha256 = hashlib.sha256(_canonical(self._document).encode("utf-8")).hexdigest()

    def _validate(self) -> None:
        if self._document.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported global tactic ledger schema")
        tactics = self._document.get("tactics")
        if not isinstance(tactics, list):
            raise ValueError("global tactic ledger tactics must be a list")
        seen: set[str] = set()
        for index, tactic in enumerate(tactics):
            if not isinstance(tactic, dict):
                raise ValueError(f"global tactic entry {index} must be an object")
            tactic_id = tactic.get("tactic_id")
            if not isinstance(tactic_id, str) or not tactic_id.strip() or tactic_id in seen:
                raise ValueError(f"global tactic entry {index} has an invalid or duplicate tactic_id")
            seen.add(tactic_id)
            family = tactic.get("game_family")
            if family not in {*GLEE_FAMILIES, "all"}:
                raise ValueError(f"global tactic {tactic_id} has an unsupported game_family")
            confidence = tactic.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
                raise ValueError(f"global tactic {tactic_id} confidence must be an integer from 0 to 100")
            for field in ("status", "failure_signature", "live_countermeasure"):
                value = tactic.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"global tactic {tactic_id} requires nonempty {field}")
            if len(str(tactic["live_countermeasure"])) > 1800:
                raise ValueError(f"global tactic {tactic_id} live_countermeasure exceeds 1800 characters")
            triggers = tactic.get("trigger_features")
            if not isinstance(triggers, list) or not triggers or any(not isinstance(value, str) or not value.strip() for value in triggers):
                raise ValueError(f"global tactic {tactic_id} requires nonempty trigger_features")
            evidence = tactic.get("evidence")
            if not isinstance(evidence, dict) or not isinstance(evidence.get("source"), str) or not evidence["source"].strip():
                raise ValueError(f"global tactic {tactic_id} requires an evidence source")
            live_route = tactic.get("live_route")
            if live_route is not None and live_route not in _LIVE_ROUTES:
                raise ValueError(f"global tactic {tactic_id} has an unsupported live_route")
            for field in ("live_action", "live_avoid"):
                value = tactic.get(field)
                if live_route is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"global tactic {tactic_id} requires nonempty {field} when live_route is set")
            for field in ("live_action_seller", "live_avoid_seller", "live_action_buyer", "live_avoid_buyer"):
                value = tactic.get(field)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"global tactic {tactic_id} has invalid {field}")

    @staticmethod
    def _negotiation_repeat(game: dict[str, Any]) -> str | None:
        state = game.get("game_state")
        if not isinstance(state, dict):
            return None
        sequence: list[tuple[str, float]] = []
        for record in state.get("history") or []:
            if not isinstance(record, dict):
                continue
            offer = record.get("offer")
            if not isinstance(offer, dict):
                continue
            player = offer.get("from_player")
            price = offer.get("price")
            if isinstance(player, str) and isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(float(price)):
                sequence.append((player, float(price)))
        last_offer = state.get("last_offer")
        if isinstance(last_offer, dict):
            player = last_offer.get("from_player")
            price = last_offer.get("price")
            if isinstance(player, str) and isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(float(price)):
                current = (player, float(price))
                if not sequence or sequence[-1] != current:
                    sequence.append(current)
        if len(sequence) < 4:
            return None
        first, second, third, fourth = sequence[-4:]
        same_price = lambda left, right: math.isclose(left, right, rel_tol=0, abs_tol=max(0.005, abs(left) * 1e-9, abs(right) * 1e-9))
        if first[0] == third[0] and second[0] == fourth[0] and first[0] != second[0] and same_price(first[1], third[1]) and same_price(second[1], fourth[1]):
            return f"The latest 4 alternating offers repeat the same pair ({first[1]:g}, {second[1]:g})."
        return None

    @staticmethod
    def _signal_polarity(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.casefold().split())
        if normalized in {"yes", "recommend", "positive", "high"}:
            return "positive"
        if normalized in {"no", "do not recommend", "don't recommend", "negative", "low"}:
            return "negative"
        if re.search(r"\b(?:do not|don't|not)\s+recommend\b|\blow[ -]quality\b", normalized):
            return "negative"
        if re.search(r"\brecommend\b|\bhigh[ -]quality\b", normalized):
            return "positive"
        return None

    @classmethod
    def _persuasion_trigger(cls, game: dict[str, Any]) -> str | None:
        state = game.get("game_state")
        if not isinstance(state, dict):
            return None
        round_number = state.get("round")
        total_rounds = state.get("total_rounds")
        if isinstance(round_number, bool) or not isinstance(round_number, int) or isinstance(total_rounds, bool) or not isinstance(total_rounds, int) or total_rounds < 1:
            return None
        remaining = total_rounds - round_number + 1
        if remaining > max(1, math.ceil(total_rounds * 0.25)):
            return None
        followed_accurate_positive = 0
        followed_accurate_negative = 0
        accurate_positive = 0
        accurate_negative = 0
        false_positive = 0
        for record in state.get("history") or []:
            if not isinstance(record, dict):
                continue
            polarity = cls._signal_polarity(record.get("seller_message"))
            quality = record.get("quality")
            bought = record.get("bought") is True or record.get("buyer_decision") == "yes"
            if polarity == "positive" and quality == "high":
                accurate_positive += 1
                followed_accurate_positive += int(bought)
            elif polarity == "negative" and quality == "low":
                accurate_negative += 1
                followed_accurate_negative += int(not bought)
            elif polarity == "positive" and quality == "low":
                false_positive += 1
        phase = str(game.get("phase") or state.get("phase") or "")
        if phase in {"seller_message", "seller_recommendation"}:
            if state.get("current_quality") != "low" or false_positive > 0 or followed_accurate_positive < 2 or followed_accurate_negative < 1:
                return None
            return f"Late low-quality seller turn with {followed_accurate_positive} followed accurate positives and {followed_accurate_negative} followed accurate negatives; no earlier false positive."
        if phase == "buyer_decision" and cls._signal_polarity(state.get("seller_message")) == "positive" and ((accurate_positive >= 2 and accurate_negative >= 1) or false_positive >= 1):
            return f"Late positive buyer signal after {accurate_positive} accurate positives, {accurate_negative} accurate negatives, and {false_positive} false positives."
        return None

    @classmethod
    def _route_trigger(cls, route: str, game: dict[str, Any]) -> str | None:
        if route == "bargaining-message-risk":
            facts = canonical_bargaining_facts(game)
            audit = facts.get("message_audit") if isinstance(facts, dict) else None
            flags = audit.get("semantic_risk_flags") if isinstance(audit, dict) else None
            return f"Current message flags: {', '.join(str(flag) for flag in flags)}." if isinstance(flags, list) and flags else None
        if route == "bargaining-extreme-incoming":
            if game.get("valid_actions", {}).get("type") != "decision":
                return None
            facts = canonical_bargaining_facts(game)
            fraction = facts.get("self_fraction") if isinstance(facts, dict) else None
            if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and float(fraction) <= 0.10:
                return f"Authenticated incoming self share is {float(fraction):.3%}."
            return None
        if route == "negotiation-repeated-pair":
            return cls._negotiation_repeat(game)
        if route == "persuasion-terminal-reputation":
            return cls._persuasion_trigger(game)
        return None

    def view(self, game: dict[str, Any] | str) -> dict[str, object] | None:
        """Return only tactics whose current-state route fired; a family string requests an offline catalog view."""
        live_game = game if isinstance(game, dict) else None
        game_family = str(game.get("game_family")) if live_game is not None else game
        if game_family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported GLEE family: {game_family}")
        entries: list[dict[str, object]] = []
        for tactic in self._document["tactics"]:
            if tactic["game_family"] not in {game_family, "all"}:
                continue
            route = tactic.get("live_route")
            if live_game is not None:
                if not isinstance(route, str):
                    continue
                observed = self._route_trigger(route, live_game)
                if observed is None:
                    continue
                phase = str(live_game.get("phase") or live_game.get("game_state", {}).get("phase") or "")
                role_suffix = "seller" if phase in {"seller_message", "seller_recommendation"} else "buyer" if phase == "buyer_decision" else None
                action = tactic.get(f"live_action_{role_suffix}", tactic["live_action"]) if role_suffix is not None else tactic["live_action"]
                avoid = tactic.get(f"live_avoid_{role_suffix}", tactic["live_avoid"]) if role_suffix is not None else tactic["live_avoid"]
                entries.append({"id": tactic["tactic_id"], "observed": observed, "do": action, "avoid": avoid})
                continue
            entries.append(
                {
                    "tactic_id": tactic["tactic_id"],
                    "status": tactic["status"],
                    "confidence": tactic["confidence"],
                    "failure_signature": tactic["failure_signature"],
                    "trigger_features": list(tactic["trigger_features"]),
                    "live_countermeasure": tactic["live_countermeasure"],
                }
            )
        if not entries:
            return None
        if live_game is not None:
            return {
                "provenance": {"sha256": self.sha256},
                "scope": {"activation": "current-state-triggered", "identity": "neutral", "authority": "advisory"},
                "entries": entries,
            }
        return {
            "provenance": {
                "kind": "repository-curated-global-tactic-ledger",
                "schema_version": _SCHEMA_VERSION,
                "sha256": self.sha256,
            },
            "scope": "Identity-neutral incident memory. Apply relevant countermeasures without inferring that the current opponent is the recorded opponent or shares its policy.",
            "entries": entries,
        }
