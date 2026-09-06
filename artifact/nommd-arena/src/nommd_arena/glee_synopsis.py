"""Hygiene checks and backward-compatible live projections for dossier synopses."""

from __future__ import annotations

import re
from typing import Any

_SELF_CORRECTION = re.compile(r"\?\s*(?:no|actually|correction)\b", re.IGNORECASE)
_INTERNAL_EXECUTION = re.compile(
    r"\b(?:xhigh|non-fallback|selected\s+max\s+branch|max\s+branch|model\s+calls?\s+timed\s+out|deterministic\s+(?:timeout\s+)?fallback|fallback\s+action|selection\s+branch)\b",
    re.IGNORECASE,
)
_LIVE_PROJECTION_VERSION = "opponent-evidence-v1"
_NEGOTIATION_LIVE_PROJECTION_VERSION = "opponent-evidence-v2"
_BARGAINING_LIVE_PROJECTION_VERSION = "opponent-evidence-v3"
_SELF_ACTION = re.compile(r"\bDeepRMM-01\b[^.!?]{0,120}\b(?:accept(?:ed|s|ing)?|reject(?:ed|s|ing)?|offer(?:ed|s|ing)?|counter(?:ed|s|ing)?|walk(?:ed|s|ing)?\s+away)\b", re.IGNORECASE)
_NOMINAL_MONEY = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?(?:\s*(?:[kmb]|thousand|million|billion))?\b", re.IGNORECASE)
_AMOUNT_TOKEN = r"\$?\s*\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?(?:\s*(?:[kmb]|thousand|million|billion))?"
_BARGAINING_ALLOCATION_PAIR = re.compile(rf"(?P<left>{_AMOUNT_TOKEN})\s*/\s*(?P<right>{_AMOUNT_TOKEN})(?P<label>\s*(?:split|allocation|offer))?", re.IGNORECASE)
_BARE_NOMINAL_AMOUNT = re.compile(r"(?<![\w.$])(?:\d{1,3}(?:,\d{3})+|\d{4,}(?:\.\d+)?|\d+\.\d{4,})(?![%\w])")


def normalized_synopsis(value: str) -> str:
    """Collapse transport whitespace without changing prose or arithmetic."""
    return " ".join(value.split())


def synopsis_hygiene_issues(value: str) -> list[str]:
    """Return deterministic issue labels for prose that should not enter a live prompt."""
    text = normalized_synopsis(value)
    issues: list[str] = []
    if _SELF_CORRECTION.search(text):
        issues.append("visible_self_correction")
    if _INTERNAL_EXECUTION.search(text):
        issues.append("internal_execution_detail")
    return issues


def validate_future_synopsis(value: str) -> str:
    """Normalize and reject malformed future model-authored live synopses."""
    text = normalized_synopsis(value)
    issues = synopsis_hygiene_issues(text)
    if issues:
        raise ValueError(f"prompt_synopsis failed live hygiene: {', '.join(issues)}")
    return text


def _bounded(value: str, limit: int, *, preserve_tail: bool = False) -> str:
    text = normalized_synopsis(value)
    if len(text) <= limit:
        return text
    if preserve_tail:
        marker = " … "
        remaining = limit - len(marker)
        head = remaining * 3 // 5
        return text[:head].rsplit(" ", 1)[0].rstrip() + marker + text[-(remaining - head) :].split(" ", 1)[-1].lstrip()
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip() + "…"


def _without_self_policy_precedents(value: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", normalized_synopsis(value))
    return " ".join(sentence for sentence in sentences if sentence and not _SELF_ACTION.search(sentence))


def normalize_negotiation_live_text(value: str) -> tuple[str, int]:
    """Remove nonportable cross-game currency anchors while retaining dimensionless evidence."""
    normalized, count = _NOMINAL_MONEY.subn("[prior-scale amount omitted]", normalized_synopsis(value))
    return normalized, count


def _amount_value(value: str) -> float | None:
    text = value.casefold().replace("$", "").replace(",", "").strip()
    multipliers = {"k": 1_000.0, "thousand": 1_000.0, "m": 1_000_000.0, "million": 1_000_000.0, "b": 1_000_000_000.0, "billion": 1_000_000_000.0}
    multiplier = 1.0
    for suffix, factor in multipliers.items():
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            multiplier = factor
            break
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return number if number >= 0 else None


def normalize_bargaining_live_text(value: str) -> tuple[str, int, int]:
    """Retain dimensionless allocation evidence while removing cross-game nominal anchors."""
    normalized_pairs = 0

    def normalize_pair(match: re.Match[str]) -> str:
        nonlocal normalized_pairs
        material = match.group(0)
        has_scale_signal = "$" in material or "," in material or bool(match.group("label")) or bool(re.search(r"\b(?:k|m|b|thousand|million|billion)\b", material, re.IGNORECASE))
        if not has_scale_signal:
            return material
        left = _amount_value(match.group("left"))
        right = _amount_value(match.group("right"))
        total = (left or 0.0) + (right or 0.0)
        if left is None or right is None or total <= 0:
            return material
        normalized_pairs += 1
        return f"{100 * left / total:.3f}% / {100 * right / total:.3f}% allocation"

    normalized = _BARGAINING_ALLOCATION_PAIR.sub(normalize_pair, normalized_synopsis(value))
    normalized, money_removed = _NOMINAL_MONEY.subn("[prior-scale amount omitted]", normalized)
    normalized, bare_removed = _BARE_NOMINAL_AMOUNT.subn("[prior-scale amount omitted]", normalized)
    return normalized, money_removed + bare_removed, normalized_pairs


def live_dossier_projection(dossier: dict[str, Any]) -> dict[str, object]:
    """Project opponent evidence without turning this agent's earlier choices into live policy precedents."""
    tendencies = dossier.get("stable_direct_tendencies") if isinstance(dossier.get("stable_direct_tendencies"), list) else []
    behavior = " ".join(normalized_synopsis(value) for value in tendencies if isinstance(value, str) and not _SELF_ACTION.search(value))
    behavior_source = "stable-direct-tendencies"
    if not behavior:
        behavior = _without_self_policy_precedents(str(dossier.get("executive_model") or dossier.get("prompt_synopsis") or ""))
        behavior_source = "sanitized-executive-model"
    uncertainties = dossier.get("uncertainties") if isinstance(dossier.get("uncertainties"), list) else []
    opponent_model = str(dossier.get("opponent_model_of_us") or "")
    uncertainty_text = " ".join(str(value) for value in uncertainties if isinstance(value, str))
    nominal_amounts_removed = 0
    if dossier.get("game_family") == "negotiation":
        behavior, removed = normalize_negotiation_live_text(behavior)
        nominal_amounts_removed += removed
        opponent_model, removed = normalize_negotiation_live_text(opponent_model)
        nominal_amounts_removed += removed
        uncertainty_text, removed = normalize_negotiation_live_text(uncertainty_text)
        nominal_amounts_removed += removed
    bargaining_pairs_normalized = 0
    if dossier.get("game_family") == "bargaining":
        behavior, removed, pairs = normalize_bargaining_live_text(behavior)
        nominal_amounts_removed += removed
        bargaining_pairs_normalized += pairs
        opponent_model, removed, pairs = normalize_bargaining_live_text(opponent_model)
        nominal_amounts_removed += removed
        bargaining_pairs_normalized += pairs
        uncertainty_text, removed, pairs = normalize_bargaining_live_text(uncertainty_text)
        nominal_amounts_removed += removed
        bargaining_pairs_normalized += pairs
    semantics: dict[str, object] = {
        "prior_self_actions": "opponent-model-evidence-only",
        "normative_policy_precedent": False,
        "opponent_learning_risk": True,
    }
    if dossier.get("game_family") == "negotiation":
        semantics.update(
            {
                "negotiation_price_transfer": "dimensionless reservation-value or visible-surplus relations only; unsupported nominal amounts are omitted",
                "nominal_amounts_removed": nominal_amounts_removed,
            }
        )
    if dossier.get("game_family") == "bargaining":
        semantics.update(
            {
                "bargaining_allocation_transfer": "dimensionless pool shares only; explicit allocation pairs are normalized and unsupported nominal amounts are omitted",
                "nominal_amounts_removed": nominal_amounts_removed,
                "allocation_pairs_normalized": bargaining_pairs_normalized,
            }
        )
    return {
        "version": _NEGOTIATION_LIVE_PROJECTION_VERSION if dossier.get("game_family") == "negotiation" else _BARGAINING_LIVE_PROJECTION_VERSION if dossier.get("game_family") == "bargaining" else _LIVE_PROJECTION_VERSION,
        "semantics": semantics,
        "opponent_behavior": _bounded(behavior, 1100),
        "opponent_behavior_source": behavior_source,
        "opponent_model_of_self": _bounded(opponent_model, 800, preserve_tail=True),
        "uncertainties": _bounded(uncertainty_text, 500),
    }
