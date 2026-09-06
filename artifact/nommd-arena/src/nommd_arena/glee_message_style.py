"""Per-game, anti-aligned message-style policy for hidden GLEE games."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .glee_bargaining_twin import classify_message_act
from .glee_negotiation_twin_v2 import classify_negotiation_message
from .glee_persuasion_twin_v2 import classify_persuasion_signal


MESSAGE_STYLE_POLICY_CONTRACT = "glee-message-style-policy-v1"
MESSAGE_STYLE_POLICY_POINTER_CONTRACT = "glee-message-style-policy-pointer-v1"
MESSAGE_STYLE_ASSIGNMENT_CONTRACT = "glee-message-style-assignment-v1"
TIMING_PERSONA_CONTRACT = "glee-hi-timing-persona-v1"

_FAMILIES = ("bargaining", "negotiation", "persuasion")
_IDENTITY_MODES = ("known", "hidden")
_ECONOMIC_STYLES = {
    "bargaining": {"soft", "middle", "hard", "unknown"},
    "negotiation": {"soft", "middle", "hard", "unknown"},
    "persuasion": {"positive", "negative", "unknown"},
}
_RENDERERS = {"native", "conversational-comma", "formal-expansive", "formal-numeric", "terse-lower"}
_TIMING_PERSONA_ACTIVATIONS = {"authoritative", "shadow"}
_TIMING_MOVE_KINDS = {
    "bargaining": {"proposal", "decision"},
    "negotiation": {"proposal", "decision"},
    "persuasion": {"seller-signal", "buyer-decision"},
}
_THRESHOLDS = {
    "bargaining_soft_max": (0.0, 1.0),
    "bargaining_hard_min": (0.0, 1.0),
    "negotiation_complete_soft_max": (0.0, 1.0),
    "negotiation_complete_hard_min": (0.0, 1.0),
    "negotiation_incomplete_soft_max": (0.0, 1000.0),
    "negotiation_incomplete_hard_min": (0.0, 1000.0),
}
_LEXICAL_FEATURES = ("log_words", "terse", "verbose", "multi_sentence", "numeric", "contracted", "question", "comma", "uppercase_ratio", "digit_ratio")
_MESSAGE_POLICY_MODELS = {
    "bargaining": {
        "next_proposal_hardness": {"kind": "linear-shift", "coefficients": (-0.006046937600760565, 0.0, 0.02980884726874136, 0.08128583478921737, -0.012768138825688871, 0.022750449201391396, -0.033005054055698584, 0.031059602461244736, 0.014282791994084595, -0.0024044834147051376), "rows": 37, "test_rows": 8, "heldout_delta": -0.011987, "metric": "mae"},
        "next_response_acceptance": {"kind": "standardized-logit-shift", "coefficients": (-0.5608501926909355, 0.5719986716884193, -0.2754969300085996, -0.20216844394714553, -0.5506445154615471, -0.3645301205878252, -0.12330807414583228, 0.5691168367297457, -0.007765081116574532, -0.22840607650863165), "means": (3.2016798063228222, 0.09433962264150944, 0.4528301886792453, 0.7735849056603774, 0.3018867924528302, 0.4339622641509434, 0.018867924528301886, 0.6037735849056604, 0.027202283018867925, 0.015826132075471698), "scales": (0.7998644743201706, 0.29230062990244654, 0.4977700361612422, 0.41851081156261954, 0.45907641738099775, 0.4956198315684414, 0.13605853869675433, 0.48911250554021585, 0.009869653910655973, 0.02566670099788112), "rows": 67, "test_rows": 14, "heldout_delta": -0.072858, "metric": "nll"},
    },
    "negotiation": {
        "next_proposal_hardness": {"kind": "linear-shift", "coefficients": (0.0009166704154275051, -0.056734512397809266, -0.007367065374672246, 0.027283296407121307, 0.010048227436532029, -0.028313446618779114, -0.021113163766200947, -0.0020818433922402853, -0.014731150488820306, 0.007829568027957536), "rows": 164, "test_rows": 33, "heldout_delta": -0.018514, "metric": "mae"},
        "next_response_acceptance": {"kind": "standardized-logit-shift", "coefficients": (0.013576221531690453, -0.1528689007334382, -0.06082060846700745, 0.24085938401869908, -0.19319690749561352, 0.1277917288454788, -0.06877748129653362, -0.31166641328277905, 0.10486594380230217, -0.2793719538604591), "means": (2.7980469926848874, 0.2097902097902098, 0.2517482517482518, 0.6013986013986014, 0.3986013986013986, 0.6013986013986014, 0.03496503496503497, 0.45454545454545453, 0.03336941958041958, 0.018314804195804196), "scales": (0.8615443445708135, 0.40715878679747236, 0.43401736081630493, 0.4896103794185817, 0.48961037941858176, 0.4896103794185817, 0.1836912662456461, 0.49792959773196915, 0.01479405580422886, 0.025486441793164186), "rows": 179, "test_rows": 36, "heldout_delta": -0.007315, "metric": "nll"},
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_lexical_style(message: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", message)
    letters = [character for character in message if character.isalpha()]
    digits = sum(character.isdigit() for character in message)
    sentences = len(re.findall(r"[.!?]+(?:\s|$)", message.strip())) or (1 if words else 0)
    values = {
        "log_words": math.log1p(len(words)),
        "terse": float(len(words) <= 6),
        "verbose": float(len(words) >= 24),
        "multi_sentence": float(sentences >= 2),
        "numeric": float(bool(re.search(r"(?:\d|[$€£%])", message))),
        "contracted": float(bool(re.search(r"\b[A-Za-z]+(?:n't|'(?:d|ll|m|re|s|ve))\b", message, flags=re.IGNORECASE))),
        "question": float("?" in message),
        "comma": float("," in message),
        "uppercase_ratio": sum(character.isupper() for character in letters) / len(letters) if letters else 0.0,
        "digit_ratio": digits / len(message) if message else 0.0,
    }
    return values


def _first_visible_opponent_message(game: Mapping[str, object]) -> tuple[str, str] | None:
    family = str(game.get("game_family") or "")
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    our_player = str(game.get("your_player") or "")
    history = state.get("history") if isinstance(state.get("history"), list) else []
    candidates: list[tuple[str, str]] = []
    if family in {"bargaining", "negotiation"}:
        for index, entry in enumerate(history):
            if not isinstance(entry, Mapping) or not isinstance(entry.get("offer"), Mapping):
                continue
            offer = entry["offer"]
            proposer = str(offer.get("proposer") or offer.get("from_player") or entry.get("proposer") or "")
            message = str(offer.get("message") or "").strip()
            if proposer and proposer != our_player and message:
                candidates.append((message, f"history[{index}].offer.message"))
        last_offer = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else None
        if last_offer is not None:
            proposer = str(last_offer.get("proposer") or last_offer.get("from_player") or "")
            message = str(last_offer.get("message") or "").strip()
            if proposer and proposer != our_player and message and all(message != prior for prior, _source in candidates):
                candidates.append((message, "game_state.last_offer.message"))
    elif family == "persuasion":
        our_role = str(state.get(f"{our_player}_role") or "")
        if our_role == "buyer":
            for index, entry in enumerate(history):
                if isinstance(entry, Mapping) and str(entry.get("seller_message") or "").strip():
                    candidates.append((str(entry["seller_message"]).strip(), f"history[{index}].seller_message"))
            message = str(state.get("seller_message") or "").strip()
            if message and all(message != prior for prior, _source in candidates):
                candidates.append((message, "game_state.seller_message"))
    return candidates[0] if candidates else None


def opponent_message_policy_signal(game: Mapping[str, object]) -> dict[str, object] | None:
    """Project a frozen, identity-free lexical prior without changing any action control."""
    family = str(game.get("game_family") or "")
    if family not in _MESSAGE_POLICY_MODELS:
        return None
    found = _first_visible_opponent_message(game)
    if found is None:
        return None
    message, source = found
    features = _raw_lexical_style(message)
    shifts: dict[str, object] = {}
    vector = tuple(features[name] for name in _LEXICAL_FEATURES)
    for target, model in _MESSAGE_POLICY_MODELS[family].items():
        coefficients = tuple(float(value) for value in model["coefficients"])
        if model["kind"] == "standardized-logit-shift":
            means = tuple(float(value) for value in model["means"])
            scales = tuple(float(value) for value in model["scales"])
            contribution = sum(coefficient * (value - mean) / scale for coefficient, value, mean, scale in zip(coefficients, vector, means, scales))
            direction = "higher" if contribution > 0 else "lower" if contribution < 0 else "neutral"
            unit = "acceptance-logit"
        else:
            contribution = sum(coefficient * value for coefficient, value in zip(coefficients, vector))
            direction = "harder" if contribution > 0 else "softer" if contribution < 0 else "neutral"
            unit = "normalized-opponent-demand"
        shifts[target] = {"lexical_shift": round(contribution, 6), "direction": direction, "unit": unit, "development_rows": model["rows"], "untouched_suffix_rows": model["test_rows"], "heldout_metric_delta": model["heldout_delta"], "heldout_metric": model["metric"]}
    return {
        "contract": "glee-identity-free-message-policy-signal-v1",
        "status": "available",
        "family": family,
        "source": source,
        "message_sha256": _text_sha256(message),
        "lexical_style": {name: round(features[name], 6) for name in _LEXICAL_FEATURES},
        "forecast_shifts": shifts,
        "authority": "soft fallible prior only; do not infer exact identity and do not override analytic or calibrated action controls",
        "evidence": "reports/glee-message-policy-signal-v1-20260813/analysis.json",
    }


def _append_fsynced(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _identity_mode(game: Mapping[str, object]) -> str:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
    return "hidden" if opponent.get("type") == "hidden" or not str(opponent.get("name") or "").strip() else "known"


def _messages_allowed(game: Mapping[str, object]) -> bool:
    family = str(game.get("game_family") or "")
    if family == "persuasion":
        return True
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    if state.get("messages_allowed") is False:
        return False
    valid = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    fields = valid.get("fields")
    return not isinstance(fields, Mapping) or not fields or "message" in fields


def _timing_move_kind(game: Mapping[str, object]) -> str | None:
    family = str(game.get("game_family") or "")
    valid = game.get("valid_actions") if isinstance(game.get("valid_actions"), Mapping) else {}
    action_type = str(valid.get("type") or "")
    if family in {"bargaining", "negotiation"}:
        return "proposal" if action_type == "offer" else "decision" if action_type == "decision" else None
    if family == "persuasion":
        return "seller-signal" if action_type == "seller_message" else "buyer-decision"
    return None


def _message_bearing_action(game: Mapping[str, object], action: Mapping[str, object]) -> bool:
    if not _messages_allowed(game) or not isinstance(action.get("message"), str) or not str(action["message"]).strip():
        return False
    family = str(game.get("game_family") or "")
    action_type = str((game.get("valid_actions") or {}).get("type") or "") if isinstance(game.get("valid_actions"), Mapping) else ""
    if family == "bargaining":
        return action_type == "offer"
    if family == "negotiation":
        return action_type == "offer" or action_type == "decision" and action.get("decision") == "RejectOffer" and "product_price" in action
    return family == "persuasion" and action_type == "seller_message"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _valid_quantile_seconds(points: object) -> bool:
    if not isinstance(points, list) or len(points) < 3:
        return False
    quantiles = [_finite(point.get("q")) if isinstance(point, Mapping) else None for point in points]
    seconds = [_finite(point.get("seconds")) if isinstance(point, Mapping) else None for point in points]
    if any(value is None for value in quantiles + seconds):
        return False
    return quantiles[0] == 0 and quantiles[-1] == 1 and all(left < right for left, right in zip(quantiles, quantiles[1:])) and all(value > 0 for value in seconds) and all(left <= right for left, right in zip(seconds, seconds[1:]))


def _valid_move_multipliers(family: str, multipliers: object) -> bool:
    return isinstance(multipliers, Mapping) and set(multipliers) == _TIMING_MOVE_KINDS[family] and all(_finite(value) is not None and 0.5 <= float(value) <= 2.0 for value in multipliers.values())


def _band(value: float, *, soft_max: float, hard_min: float) -> str:
    if value <= soft_max:
        return "soft"
    if value >= hard_min:
        return "hard"
    return "middle"


def economic_style(game: Mapping[str, object], action: Mapping[str, object], thresholds: Mapping[str, object]) -> str:
    """Classify the actual outward move before selecting an anti-aligned lexical profile."""
    family = str(game.get("game_family") or "")
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    player = str(game.get("your_player") or state.get("current_player") or "")
    if family == "bargaining":
        alice = _finite(action.get("alice_gain"))
        bob = _finite(action.get("bob_gain"))
        if alice is None or bob is None or alice + bob <= 0 or player not in {"player_1", "player_2"}:
            return "unknown"
        own_share = (alice if player == "player_1" else bob) / (alice + bob)
        return _band(own_share, soft_max=float(thresholds["bargaining_soft_max"]), hard_min=float(thresholds["bargaining_hard_min"]))
    if family == "negotiation":
        price = _finite(action.get("product_price"))
        role = str(state.get(f"{player}_role") or "")
        value = _finite(state.get(f"{player}_value"))
        if price is None or value is None or role not in {"buyer", "seller"}:
            return "unknown"
        own_surplus = value - price if role == "buyer" else price - value
        opponent = "player_2" if player == "player_1" else "player_1" if player == "player_2" else ""
        opponent_value = _finite(state.get(f"{opponent}_value")) if state.get("complete_information") is True else None
        opponent_role = str(state.get(f"{opponent}_role") or "")
        if opponent_value is not None and opponent_role in {"buyer", "seller"}:
            total_surplus = value - opponent_value if role == "buyer" else opponent_value - value
            if total_surplus > 0:
                return _band(own_surplus / total_surplus, soft_max=float(thresholds["negotiation_complete_soft_max"]), hard_min=float(thresholds["negotiation_complete_hard_min"]))
        normalized_surplus = own_surplus / max(1.0, abs(value))
        return _band(normalized_surplus, soft_max=float(thresholds["negotiation_incomplete_soft_max"]), hard_min=float(thresholds["negotiation_incomplete_hard_min"]))
    if family == "persuasion":
        channel = str(state.get("seller_message_type") or "text").casefold()
        polarity, _act, _fingerprint = classify_persuasion_signal(action.get("message"), channel=channel)
        return polarity if polarity in {"positive", "negative"} else "unknown"
    return "unknown"


def _message_act(game: Mapping[str, object], message: str) -> str:
    family = str(game.get("game_family") or "")
    if family == "bargaining":
        return classify_message_act(message, messages_allowed=True)
    if family == "negotiation":
        return classify_negotiation_message(message, messages_allowed=True)
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    return classify_persuasion_signal(message, channel=str(state.get("seller_message_type") or "text").casefold())[1]


def _amount(value: object) -> str:
    number = _finite(value)
    if number is None:
        return "the stated amount"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _semantic_phrase(family: str, act: str) -> str:
    phrases = {
        "bargaining": {
            "allocation": "the allocation is stated in the offer",
            "authority": "this is my firm position",
            "commitment": "I will hold to these terms",
            "fairness": "this is a fair basis for agreement",
            "fairness-urgency": "this is a fair basis and delay adds no value",
            "urgency": "delay adds no value",
        },
        "negotiation": {
            "commitment": "I will hold to this price",
            "fairness": "this is a fair basis for agreement",
            "price": "the price states my current position",
            "urgency": "we can conclude without further delay",
            "walkaway": "this is the boundary of my position",
        },
    }
    return phrases.get(family, {}).get(act, "this is my current proposal")


def _variant(game: Mapping[str, object], profile: Mapping[str, object], act: str, values: tuple[str, ...]) -> str:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    key = "|".join((str(game.get("game_id") or ""), str(profile.get("profile_id") or ""), str(state.get("round") or ""), act))
    index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(values)
    return values[index]


def _render_message(game: Mapping[str, object], action: Mapping[str, object], profile: Mapping[str, object], original: str, act: str, style: str) -> str:
    renderer = str(profile["renderer"])
    family = str(game.get("game_family") or "")
    if renderer == "native":
        return original
    if family == "persuasion":
        positive = style == "positive"
        negative = style == "negative"
        if renderer == "terse-lower":
            values = ("buy", "yes", "good choice") if positive else ("pass", "no", "skip this purchase") if negative else ("consider this carefully", "unclear choice", "weigh this one")
            return _variant(game, profile, act, values)
        if renderer == "conversational-comma":
            values = ("This one looks worthwhile, and I'd take it.", "It looks like a good choice, so I'd go for it.") if positive else ("I'd pass on this one, rather than buy it.", "I recommend skipping it, since it looks like a poor choice.") if negative else ("This one is uncertain, so I'd weigh it carefully.", "The signal looks mixed, and I'd stay cautious.")
            return _variant(game, profile, act, values)
        if renderer == "formal-numeric":
            price = _amount((game.get("game_state") or {}).get("product_price") if isinstance(game.get("game_state"), Mapping) else None)
            return f"The stated price is {price}. I recommend accepting the offer." if positive else f"The stated price is {price}. I recommend declining the offer." if negative else f"The stated price is {price}. The available signal is inconclusive."
        values = ("On balance, this product is a worthwhile choice. I recommend accepting the offer at the stated terms.", "The available considerations favor this product. I recommend accepting the offer under the stated terms.") if positive else ("On balance, this product is not a worthwhile choice. I recommend declining the offer at the stated terms.", "The available considerations weigh against this product. I recommend declining the offer under the stated terms.") if negative else ("The available evidence does not support a strong recommendation. The offer should be evaluated cautiously.", "The evidence remains inconclusive. A cautious assessment is appropriate before making a choice.")
        return _variant(game, profile, act, values)
    phrase = _semantic_phrase(family, act)
    if renderer == "terse-lower":
        terse = {
            "authority": "firm position",
            "commitment": "these terms stand",
            "fairness": "fair terms",
            "fairness-urgency": "fair terms settle now",
            "urgency": "settle now",
            "allocation": "split as shown",
            "price": "price as shown",
            "walkaway": "final boundary",
        }
        return terse.get(act, "current terms")
    if renderer == "conversational-comma":
        values = {
            "allocation": ("The split is shown in the offer, and that's what I'm proposing.", "The allocation is right there, and those are my terms."),
            "authority": ("That's my position, and it is firm.", "This is where I stand, and I'm firm on it."),
            "commitment": ("These terms work for me, and I'll hold to them.", "That's my proposal, and I'll stand by it."),
            "fairness": ("This seems fair to both sides, and I can agree on it.", "The terms look fair, and they work for both of us."),
            "fairness-urgency": ("The terms are fair, and we can settle now.", "This is a fair split, so there's no need to delay."),
            "urgency": ("We can settle now, and there's no need to wait.", "There's no gain in waiting, so let's conclude."),
            "price": ("The price is in the offer, and that's my position.", "That's the price I'm proposing, and it stands."),
            "walkaway": ("That's my limit, and I won't move past it.", "This is my boundary, and I can't go beyond it."),
        }
        return _variant(game, profile, act, values.get(act, (f"{phrase.capitalize()}, and that is my position.", f"{phrase.capitalize()}, which is where I stand.")))
    if renderer == "formal-numeric":
        if family == "bargaining":
            numbers = f"Alice {_amount(action.get('alice_gain'))}; Bob {_amount(action.get('bob_gain'))}"
            return f"The proposed allocation is {numbers}. {_semantic_phrase(family, act).capitalize()}."
        return f"The proposed price is {_amount(action.get('product_price'))}. {_semantic_phrase(family, act).capitalize()}."
    expansions = {
        "allocation": ("The allocation is stated directly in the offer. Its terms define the proposed division without requiring an additional interpretation.", "The proposed allocation is explicit. Each side can evaluate the stated division on its own terms."),
        "authority": ("This is my firm position. The proposal expresses the terms on which I am prepared to proceed.", "My position on these terms is firm. The proposal accurately states the outcome I am prepared to support."),
        "commitment": ("I will hold to these terms. The proposal states the position to which I am prepared to remain committed.", "These terms will remain my position. I am prepared to stand behind the proposal as stated."),
        "fairness": ("This is a fair basis for agreement. The proposal gives both sides a reasonable foundation for assessing the division.", "The terms provide a fair basis for agreement. They can be evaluated as a balanced treatment of both sides."),
        "fairness-urgency": ("This is a fair basis for agreement. Further delay would not improve the balance already expressed in the proposal.", "The proposal is fair to both sides. It also provides a sufficient basis for concluding without further delay."),
        "urgency": ("Further delay adds no value. The present proposal provides a sufficient basis for reaching a decision now.", "There is no benefit in extending the exchange. The current terms are ready for an immediate decision."),
        "price": ("The stated price represents my current position. It is the amount on which I am prepared to continue the negotiation.", "This price accurately states my present position. The proposal should be evaluated on that explicit amount."),
        "walkaway": ("This is the boundary of my position. I am not prepared to proceed beyond the stated terms.", "The proposal states my final boundary. Terms beyond it would not support an agreement from my side."),
    }
    return _variant(game, profile, act, expansions.get(act, (f"{phrase.capitalize()}. The proposal states my present position without changing the underlying terms.", f"{phrase.capitalize()}. These are the terms on which I am prepared to proceed.")))


def realize_message_style(game: Mapping[str, object], action: Mapping[str, object], profile: Mapping[str, object]) -> tuple[dict[str, Any], dict[str, object]]:
    """Replace only message prose while preserving the selected legal and economic action."""
    styled = copy.deepcopy(dict(action))
    original = str(styled.get("message") or "").strip()
    if not _message_bearing_action(game, styled):
        return styled, {"status": "not-applicable", "profile_id": profile.get("profile_id"), "action_unchanged": True}
    family = str(game.get("game_family") or "")
    act = _message_act(game, original)
    assigned_style = str(profile.get("economic_style") or "unknown")
    current_style = assigned_style
    if family == "persuasion":
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        current_style = classify_persuasion_signal(original, channel=str(state.get("seller_message_type") or "text").casefold())[0]
    rendered = _render_message(game, styled, profile, original, act, current_style).strip()
    if not rendered or len(rendered) > 2000:
        raise RuntimeError("message-style renderer produced an invalid message")
    if family == "persuasion" and current_style in {"positive", "negative"}:
        state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
        rendered_polarity = classify_persuasion_signal(rendered, channel=str(state.get("seller_message_type") or "text").casefold())[0]
        if rendered_polarity != current_style:
            raise RuntimeError("message-style renderer changed Persuasion signal polarity")
    styled["message"] = rendered
    before_nonmessage = {key: value for key, value in action.items() if key != "message"}
    after_nonmessage = {key: value for key, value in styled.items() if key != "message"}
    if before_nonmessage != after_nonmessage:
        raise RuntimeError("message-style renderer changed a non-message action field")
    return styled, {
        "status": "styled",
        "contract": MESSAGE_STYLE_POLICY_CONTRACT,
        "profile_id": profile.get("profile_id"),
        "policy_revision": profile.get("policy_revision"),
        "assigned_economic_style": assigned_style,
        "current_economic_style": current_style,
        "original_message_act": act,
        "rendered_message_act": _message_act(game, rendered),
        "original_message_sha256": _text_sha256(original),
        "rendered_message_sha256": _text_sha256(rendered),
        "message_changed": rendered != original,
        "nonmessage_action_unchanged": True,
    }


def _log_quantile(points: list[Mapping[str, object]], quantile: float) -> float:
    bounded = min(1.0, max(0.0, quantile))
    for left, right in zip(points, points[1:]):
        left_q = float(left["q"])
        right_q = float(right["q"])
        if bounded > right_q:
            continue
        fraction = 0.0 if right_q == left_q else (bounded - left_q) / (right_q - left_q)
        left_log = math.log(float(left["seconds"]))
        right_log = math.log(float(right["seconds"]))
        return math.exp(left_log + fraction * (right_log - left_log))
    return float(points[-1]["seconds"])


def sample_timing_persona_target(game: Mapping[str, object], profile: Mapping[str, object] | None, random_source: object) -> tuple[float | None, dict[str, object]]:
    """Sample one HI target around the immutable game-speed quantile without changing KI or legacy timing."""
    persona = profile.get("timing_persona") if isinstance(profile, Mapping) and isinstance(profile.get("timing_persona"), Mapping) else None
    if not isinstance(persona, Mapping):
        return None, {"contract": TIMING_PERSONA_CONTRACT, "status": "legacy-or-unassigned"}
    activation = str(persona.get("activation") or "")
    if activation == "ki-stable":
        return None, {"contract": TIMING_PERSONA_CONTRACT, "status": "ki-stable", "timing_profile_id": persona.get("timing_profile_id")}
    move_kind = _timing_move_kind(game)
    multipliers = persona.get("move_multipliers") if isinstance(persona.get("move_multipliers"), Mapping) else {}
    points = persona.get("quantile_seconds") if isinstance(persona.get("quantile_seconds"), list) else None
    game_quantile = _finite(persona.get("game_speed_quantile"))
    residual = _finite(persona.get("residual_quantile_half_width"))
    family = str(game.get("game_family") or "")
    if activation not in _TIMING_PERSONA_ACTIVATIONS or move_kind not in _TIMING_MOVE_KINDS.get(family, set()) or not _valid_quantile_seconds(points) or game_quantile is None or not 0 <= game_quantile <= 1 or residual is None or not 0 <= residual <= 0.5 or not _valid_move_multipliers(family, multipliers):
        return None, {"contract": TIMING_PERSONA_CONTRACT, "status": "invalid-or-inapplicable", "timing_profile_id": persona.get("timing_profile_id")}
    jitter = float(random_source.uniform(-residual, residual))
    sampled_quantile = min(1.0, max(0.0, game_quantile + jitter))
    multiplier = float(multipliers[move_kind])
    target = _log_quantile(points, sampled_quantile) * multiplier
    return target, {
        "contract": TIMING_PERSONA_CONTRACT,
        "status": "sampled",
        "activation": activation,
        "timing_profile_id": persona.get("timing_profile_id"),
        "lexical_profile_id": profile.get("profile_id") if isinstance(profile, Mapping) else None,
        "game_speed_quantile": round(game_quantile, 6),
        "residual_quantile_jitter": round(jitter, 6),
        "sampled_quantile": round(sampled_quantile, 6),
        "move_kind": move_kind,
        "move_multiplier": multiplier,
        "unbounded_target_elapsed_s": round(target, 6),
    }


class MessageStylePolicyStore:
    """Pin one empirically anti-aligned joint lexical and timing profile per game."""

    def __init__(self, *, root: Path, assignments_path: Path, random_source: object | None = None) -> None:
        self.root = root.resolve()
        self.current_path = self.root / "current.json"
        self.assignments_path = assignments_path
        self._lock = threading.Lock()
        self._random = random_source or secrets.SystemRandom()
        self._assignments: dict[str, dict[str, object]] = {}
        self._release_cache: dict[str, dict[str, object]] = {}
        self._last_good: tuple[dict[str, object], str, Path] | None = None
        if assignments_path.is_file():
            for line in assignments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_assignment(record)
                game_id = str(record["game_id"])
                prior = self._assignments.get(game_id)
                if prior is not None and prior != record:
                    raise RuntimeError(f"message-style game {game_id} has conflicting assignments")
                self._assignments[game_id] = record
        self._last_good = self._load_current()

    @staticmethod
    def _validate_release(release: object) -> dict[str, object]:
        if not isinstance(release, dict) or release.get("schema_version") != 1 or release.get("contract") != MESSAGE_STYLE_POLICY_CONTRACT:
            raise RuntimeError("message-style release has an incompatible contract")
        revision = release.get("revision")
        profiles = release.get("profiles")
        routing = release.get("routing")
        thresholds = release.get("economic_style_thresholds")
        if not isinstance(revision, str) or not revision or not isinstance(profiles, dict) or "native-strategic" not in profiles or not isinstance(routing, dict) or set(routing) != set(_IDENTITY_MODES) or not isinstance(thresholds, dict) or set(thresholds) != set(_THRESHOLDS):
            raise RuntimeError("message-style release is incomplete")
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not isinstance(profile, dict) or profile.get("renderer") not in _RENDERERS or not isinstance(profile.get("prompt_contract"), str) or not str(profile["prompt_contract"]).strip():
                raise RuntimeError(f"message-style profile {profile_id!r} is invalid")
        for name, bounds in _THRESHOLDS.items():
            value = thresholds[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not bounds[0] <= float(value) <= bounds[1]:
                raise RuntimeError(f"message-style threshold {name} is invalid")
        if float(thresholds["bargaining_soft_max"]) >= float(thresholds["bargaining_hard_min"]) or float(thresholds["negotiation_complete_soft_max"]) >= float(thresholds["negotiation_complete_hard_min"]) or float(thresholds["negotiation_incomplete_soft_max"]) >= float(thresholds["negotiation_incomplete_hard_min"]):
            raise RuntimeError("message-style soft and hard thresholds overlap")
        known = routing["known"]
        hidden = routing["hidden"]
        if not isinstance(known, dict) or set(known) != set(_FAMILIES) or any(value != "native-strategic" for value in known.values()) or not isinstance(hidden, dict) or set(hidden) != set(_FAMILIES):
            raise RuntimeError("message-style identity routing is invalid")
        for family in _FAMILIES:
            routes = hidden[family]
            if not isinstance(routes, dict) or set(routes) != _ECONOMIC_STYLES[family] or any(profile_id not in profiles for profile_id in routes.values()):
                raise RuntimeError(f"message-style hidden routing is invalid for {family}")
        timing = release.get("timing_persona_policy")
        if timing is not None:
            if not isinstance(timing, dict) or timing.get("contract") != TIMING_PERSONA_CONTRACT or timing.get("activation") not in _TIMING_PERSONA_ACTIVATIONS:
                raise RuntimeError("message-style timing-persona policy is invalid")
            beta = timing.get("game_quantile_beta")
            residual = timing.get("residual_quantile_half_width")
            routes = timing.get("lexical_profile_routes")
            timing_profiles = timing.get("profiles")
            if not isinstance(beta, dict) or any(_finite(beta.get(key)) is None or float(beta[key]) <= 0 for key in ("alpha", "beta")) or _finite(residual) is None or not 0 <= float(residual) <= 0.5 or not isinstance(routes, dict) or not isinstance(timing_profiles, dict):
                raise RuntimeError("message-style timing-persona parameters are invalid")
            hidden_profile_ids = {profile_id for family in _FAMILIES for profile_id in routing["hidden"][family].values()}
            if set(routes) != hidden_profile_ids or any(timing_profile_id not in timing_profiles for timing_profile_id in routes.values()):
                raise RuntimeError("message-style timing-persona routing is incomplete")
            for timing_profile_id, timing_profile in timing_profiles.items():
                points = timing_profile.get("quantile_seconds") if isinstance(timing_profile, dict) else None
                multipliers = timing_profile.get("move_multipliers") if isinstance(timing_profile, dict) else None
                if not _valid_quantile_seconds(points) or not isinstance(multipliers, dict) or set(multipliers) != set(_FAMILIES):
                    raise RuntimeError(f"timing persona {timing_profile_id!r} is invalid")
                for family in _FAMILIES:
                    family_multipliers = multipliers[family]
                    if not _valid_move_multipliers(family, family_multipliers):
                        raise RuntimeError(f"timing persona {timing_profile_id!r} move multipliers are invalid for {family}")
        return release

    @staticmethod
    def _validate_assignment(record: object) -> None:
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("contract") != MESSAGE_STYLE_ASSIGNMENT_CONTRACT or not isinstance(record.get("game_id"), str) or record.get("family") not in _FAMILIES or record.get("identity_mode") not in _IDENTITY_MODES or not isinstance(record.get("release_sha256"), str) or not isinstance(record.get("profile"), dict):
            raise RuntimeError("message-style assignment is invalid")
        profile = record["profile"]
        if profile.get("renderer") not in _RENDERERS or profile.get("profile_id") != record.get("profile_id"):
            raise RuntimeError("message-style assignment profile is invalid")
        timing = profile.get("timing_persona")
        if timing is not None and (not isinstance(timing, dict) or timing.get("contract") != TIMING_PERSONA_CONTRACT or timing.get("activation") not in {*_TIMING_PERSONA_ACTIVATIONS, "ki-stable"}):
            raise RuntimeError("message-style assignment timing persona is invalid")
        if isinstance(timing, dict) and timing.get("activation") in _TIMING_PERSONA_ACTIVATIONS:
            quantile = _finite(timing.get("game_speed_quantile"))
            residual = _finite(timing.get("residual_quantile_half_width"))
            if not isinstance(timing.get("timing_profile_id"), str) or quantile is None or not 0 <= quantile <= 1 or residual is None or not 0 <= residual <= 0.5 or not _valid_quantile_seconds(timing.get("quantile_seconds")) or not _valid_move_multipliers(str(record["family"]), timing.get("move_multipliers")):
                raise RuntimeError("message-style assignment timing persona parameters are invalid")

    def _load_current(self) -> tuple[dict[str, object], str, Path]:
        if not self.current_path.is_file():
            raise RuntimeError(f"message-style pointer is missing: {self.current_path}")
        pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema_version") != 1 or pointer.get("contract") != MESSAGE_STYLE_POLICY_POINTER_CONTRACT or not isinstance(pointer.get("release"), str) or not isinstance(pointer.get("release_sha256"), str):
            raise RuntimeError("message-style pointer is invalid")
        release_path = (self.root / str(pointer["release"])).resolve()
        try:
            release_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("message-style pointer escapes its root") from error
        actual_sha256 = _file_sha256(release_path)
        if actual_sha256 != pointer["release_sha256"]:
            raise RuntimeError("message-style release hash differs from its pointer")
        if actual_sha256 not in self._release_cache:
            self._release_cache[actual_sha256] = self._validate_release(json.loads(release_path.read_text(encoding="utf-8")))
        return self._release_cache[actual_sha256], actual_sha256, release_path

    def assigned_profile(self, game: Mapping[str, object]) -> dict[str, object] | None:
        game_id = str(game.get("game_id") or "")
        with self._lock:
            record = self._assignments.get(game_id)
            if record is None:
                return None
            if record["family"] != game.get("game_family") or record["identity_mode"] != _identity_mode(game):
                raise RuntimeError(f"message-style game identity changed after assignment: {game_id}")
            return copy.deepcopy(record["profile"])

    def profile_for_action(self, game: Mapping[str, object], action: Mapping[str, object]) -> tuple[dict[str, object] | None, bool, str | None]:
        game_id = str(game.get("game_id") or "")
        family = str(game.get("game_family") or "")
        if not game_id or family not in _FAMILIES:
            raise RuntimeError("message-style assignment requires a supported game with an ID")
        with self._lock:
            prior = self._assignments.get(game_id)
            if prior is not None:
                if prior["family"] != family or prior["identity_mode"] != _identity_mode(game):
                    raise RuntimeError(f"message-style game identity changed after assignment: {game_id}")
                return copy.deepcopy(prior["profile"]), False, str(prior.get("pointer_error")) if prior.get("pointer_error") else None
            pointer_error = None
            try:
                release, release_sha256, release_path = self._load_current()
                self._last_good = release, release_sha256, release_path
            except Exception as error:
                if self._last_good is None:
                    raise
                release, release_sha256, release_path = self._last_good
                pointer_error = f"{type(error).__name__}: {error}"
            identity_mode = _identity_mode(game)
            style = economic_style(game, action, release["economic_style_thresholds"])
            profile_id = str(release["routing"][identity_mode][family] if identity_mode == "known" else release["routing"]["hidden"][family][style])
            profile = copy.deepcopy(release["profiles"][profile_id])
            profile.update({"profile_id": profile_id, "economic_style": style, "identity_mode": identity_mode, "policy_revision": release["revision"], "release_sha256": release_sha256})
            timing_policy = release.get("timing_persona_policy") if isinstance(release.get("timing_persona_policy"), dict) else None
            if timing_policy is not None and identity_mode == "hidden":
                timing_profile_id = str(timing_policy["lexical_profile_routes"][profile_id])
                timing_profile = timing_policy["profiles"][timing_profile_id]
                beta = timing_policy["game_quantile_beta"]
                profile["timing_persona"] = {
                    "contract": TIMING_PERSONA_CONTRACT,
                    "activation": timing_policy["activation"],
                    "timing_profile_id": timing_profile_id,
                    "game_speed_quantile": round(float(self._random.betavariate(float(beta["alpha"]), float(beta["beta"]))), 9),
                    "residual_quantile_half_width": timing_policy["residual_quantile_half_width"],
                    "quantile_seconds": copy.deepcopy(timing_profile["quantile_seconds"]),
                    "move_multipliers": copy.deepcopy(timing_profile["move_multipliers"][family]),
                }
            elif timing_policy is not None:
                profile["timing_persona"] = {"contract": TIMING_PERSONA_CONTRACT, "activation": "ki-stable", "timing_profile_id": "ki-beta-2-2-v1"}
            record = {
                "schema_version": 1,
                "contract": MESSAGE_STYLE_ASSIGNMENT_CONTRACT,
                "game_id": game_id,
                "family": family,
                "identity_mode": identity_mode,
                "economic_style": style,
                "profile_id": profile_id,
                "revision": release["revision"],
                "release_sha256": release_sha256,
                "release_path": str(release_path),
                "profile": profile,
                "pointer_error": pointer_error,
            }
            _append_fsynced(self.assignments_path, record)
            self._assignments[game_id] = record
            return copy.deepcopy(profile), True, pointer_error

    def _manifest_receipt(self, release: Mapping[str, object], release_sha256: str, release_path: Path) -> dict[str, object]:
        return {
            "contract": MESSAGE_STYLE_POLICY_CONTRACT,
            "pointer_contract": MESSAGE_STYLE_POLICY_POINTER_CONTRACT,
            "root": str(self.root),
            "current_revision_at_startup": release["revision"],
            "current_release_sha256_at_startup": release_sha256,
            "current_release_path_at_startup": str(release_path),
            "assignment_contract": MESSAGE_STYLE_ASSIGNMENT_CONTRACT,
            "assignments_path": str(self.assignments_path),
            "promotion_scope": "new games only; hidden lexical and timing profiles are selected from the actual first outward move and remain pinned for the game",
        }

    def manifest_receipt(self) -> dict[str, object]:
        release, release_sha256, release_path = self._load_current()
        return self._manifest_receipt(release, release_sha256, release_path)

    def status(self) -> dict[str, object]:
        pointer_error = None
        try:
            release, release_sha256, release_path = self._load_current()
            self._last_good = release, release_sha256, release_path
        except Exception as error:
            if self._last_good is None:
                raise
            release, release_sha256, release_path = self._last_good
            pointer_error = f"{type(error).__name__}: {error}"
        counts = Counter(f"{record['family']}:{record['identity_mode']}:{record['profile_id']}" for record in self._assignments.values())
        return {
            **self._manifest_receipt(release, release_sha256, release_path),
            "current_revision": release["revision"],
            "current_release_sha256": release_sha256,
            "current_release_path": str(release_path),
            "current_pointer_error": pointer_error,
            "assigned_games": len(self._assignments),
            "assigned_games_by_route": dict(sorted(counts.items())),
        }
