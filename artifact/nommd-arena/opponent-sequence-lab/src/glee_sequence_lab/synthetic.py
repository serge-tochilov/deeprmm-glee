"""Mechanics-respecting synthetic policy zoo for population pretraining only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import polars as pl

from .corpus import CORPUS_CONTRACT, EVENT_SCHEMA, GAME_SCHEMA, GLEE_FAMILIES, TARGET_SCHEMA, _artifact, _message_features, canonical_json, file_sha256


SYNTHETIC_CONTRACT = "glee-synthetic-policy-zoo-v2"
POLICY_KINDS = ("threshold", "reciprocal", "change-point", "order-2", "stochastic")


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _event(*, game_id: str, index: int, round_number: int, maximum: int, horizon_known: bool, actor: str, kind: str, action: str, value: float | None = None, auxiliary: float | None = None, delay: float | None = None, quality: str | None = None, message: str = "", message_act: str = "none") -> dict[str, object]:
    return {
        "game_id": game_id,
        "event_index": index,
        "round_number": round_number,
        "round_phase": (round_number - 1) / max(1, maximum - 1) if horizon_known else 1.0 - math.exp(-max(0, round_number - 1) / 12.0),
        "actor": actor,
        "kind": kind,
        "action_label": action,
        "action_value": value,
        "action_aux_value": auxiliary,
        "response_time_ms": delay,
        "visible_quality": quality,
        **_message_features(message, family_act=message_act),
    }


def _message(rng: random.Random, family: str, act: str, value: float | None = None) -> str:
    if act in {"none", "silence"}:
        return ""
    amount = int(round(100 * (value if value is not None else 0.5)))
    templates = {
        "fairness": (f"A {amount}/{100 - amount} split is fair.", f"This leaves both sides a reasonable share: {amount} percent."),
        "urgency": (f"Accept {amount} now; time is short.", f"We should settle at {amount} before the deadline."),
        "constraint": (f"I cannot move beyond {amount}.", f"My limit is {amount}; there is no room past it."),
        "reciprocity": (f"You moved, so I can offer {amount} in return.", f"Match this concession and we can close at {amount}."),
        "anchoring": (f"The right number is {amount}.", f"Start from {amount}; that reflects the value here."),
        "commitment": (f"I commit to {amount}.", f"My final commitment is {amount}."),
        "allocation": (f"Allocate {amount} percent to this side.", f"The allocation should be {amount} percent."),
        "conditional": (f"If you accept now, I can agree to {amount}.", f"I will settle at {amount} if you reciprocate."),
        "price": (f"The price is {amount}.", f"I propose a price of {amount}."),
        "authority": (f"The stated rule supports {amount}.", f"The official benchmark is {amount}."),
        "walkaway": (f"Below {amount}, I will leave.", f"No agreement is better than less than {amount}."),
        "recommend": ("I recommend buying.", "Buy; this is worthwhile."),
        "binary-recommendation": ("yes", "no"),
        "quality-claim-high": ("The product is high quality.", "This is a strong item; buy it."),
        "quality-claim-low": ("The product is poor quality.", "I cannot recommend this item."),
        "other": ("Consider the proposal.", "That is my position."),
    }
    return rng.choice(templates.get(act, templates["other"]))


def _policy(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(POLICY_KINDS)
    return {
        "kind": kind,
        "opening": rng.uniform(0.50, 0.92),
        "floor": rng.uniform(0.28, 0.56),
        "concession": rng.uniform(0.005, 0.075),
        "reciprocity": rng.uniform(-0.15, 0.55),
        "noise": rng.uniform(0.005, 0.09),
        "change_fraction": rng.uniform(0.30, 0.75),
        "post_change_shift": rng.uniform(-0.20, 0.20),
        "order2_strength": rng.uniform(0.10, 0.45),
        "truthfulness": rng.uniform(0.10, 0.95),
        "trust": rng.uniform(0.20, 0.80),
        "delay_location": rng.uniform(6.1, 10.7),
        "message_tendency": rng.uniform(0.15, 0.90),
    }


def _demand(policy: Mapping[str, object], *, phase: float, counterpart_concession: float, inferred_stubbornness: float, rng: random.Random) -> float:
    opening = float(policy["opening"])
    floor = min(opening, float(policy["floor"]))
    demand = opening - float(policy["concession"]) * phase * 10.0
    kind = str(policy["kind"])
    if kind == "reciprocal":
        demand -= float(policy["reciprocity"]) * counterpart_concession
    elif kind == "change-point" and phase >= float(policy["change_fraction"]):
        demand += float(policy["post_change_shift"])
    elif kind == "order-2":
        demand += float(policy["order2_strength"]) * (inferred_stubbornness - 0.5)
    elif kind == "stochastic":
        demand += rng.gauss(0.0, float(policy["noise"]) * 2.0)
    demand += rng.gauss(0.0, float(policy["noise"]))
    return _clip(demand, floor, 0.97)


def _accept(policy: Mapping[str, object], *, offered_share: float, phase: float, inferred_stubbornness: float, rng: random.Random) -> bool:
    threshold = float(policy["floor"]) - float(policy["concession"]) * phase * 4.0
    kind = str(policy["kind"])
    if kind == "change-point" and phase >= float(policy["change_fraction"]):
        threshold += float(policy["post_change_shift"]) * 0.5
    elif kind == "order-2":
        threshold -= float(policy["order2_strength"]) * inferred_stubbornness * 0.2
    probability = 1.0 / (1.0 + math.exp(-(offered_share - threshold) / max(0.015, float(policy["noise"]))))
    return rng.random() < probability


def _delay(policy: Mapping[str, object], *, surprise: float, rng: random.Random) -> float:
    location = float(policy["delay_location"]) + min(1.4, abs(surprise) * 2.5)
    return min(115_000.0, max(70.0, math.exp(rng.gauss(location, 0.35))))


def _utterance(rng: random.Random, family: str, policy: Mapping[str, object], *, value: float | None, positive: bool | None = None) -> tuple[str, str]:
    if rng.random() > float(policy["message_tendency"]):
        return "", "none"
    if family == "persuasion":
        if positive is None:
            act = "binary-recommendation"
        else:
            act = "quality-claim-high" if positive else "quality-claim-low"
    else:
        choices = ("fairness", "urgency", "commitment", "allocation", "other", "authority") if family == "bargaining" else ("commitment", "conditional", "fairness", "other", "price", "urgency", "walkaway")
        act = rng.choice(choices)
    return _message(rng, family, act, value), act


def _bargaining(game_id: str, rng: random.Random, opponent: Mapping[str, object], self_policy: Mapping[str, object], *, maximum: int, horizon_known: bool, messages_allowed: bool) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    proposer = rng.choice(("self", "opponent"))
    last_demand = {"self": float(self_policy["opening"]), "opponent": float(opponent["opening"])}
    opening = dict(last_demand)
    for round_number in range(1, maximum + 1):
        phase = (round_number - 1) / max(1, maximum - 1) if horizon_known else 1.0 - math.exp(-max(0, round_number - 1) / 12.0)
        policy = self_policy if proposer == "self" else opponent
        counterpart = "opponent" if proposer == "self" else "self"
        counterpart_policy = opponent if proposer == "self" else self_policy
        counterpart_concession = max(0.0, opening[counterpart] - last_demand[counterpart])
        inferred_stubbornness = _clip(last_demand[counterpart])
        demand = _demand(policy, phase=phase, counterpart_concession=counterpart_concession, inferred_stubbornness=inferred_stubbornness, rng=rng)
        last_demand[proposer] = demand
        message, message_act = _utterance(rng, "bargaining", policy, value=demand) if messages_allowed else ("", "none")
        opponent_share = demand if proposer == "opponent" else 1.0 - demand
        self_share = 1.0 - opponent_share
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=horizon_known, actor=proposer, kind="proposal", action="proposal", value=opponent_share, auxiliary=self_share, message=message, message_act=message_act))
        offered_share = 1.0 - demand
        accepted = _accept(counterpart_policy, offered_share=offered_share, phase=phase, inferred_stubbornness=_clip(demand), rng=rng)
        surprise = offered_share - float(counterpart_policy["floor"])
        decision = "accept" if accepted else "reject"
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=horizon_known, actor=counterpart, kind="response", action=decision, value=opponent_share, auxiliary=self_share, delay=_delay(counterpart_policy, surprise=surprise, rng=rng)))
        if accepted:
            break
        proposer = counterpart
    return events


def _negotiation(game_id: str, rng: random.Random, opponent: Mapping[str, object], self_policy: Mapping[str, object], *, maximum: int, horizon_known: bool, messages_allowed: bool) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    proposer = rng.choice(("self", "opponent"))
    last_demand = {"self": float(self_policy["opening"]), "opponent": float(opponent["opening"])}
    opening = dict(last_demand)
    for round_number in range(1, maximum + 1):
        phase = (round_number - 1) / max(1, maximum - 1) if horizon_known else 1.0 - math.exp(-max(0, round_number - 1) / 12.0)
        policy = self_policy if proposer == "self" else opponent
        counterpart = "opponent" if proposer == "self" else "self"
        counterpart_policy = opponent if proposer == "self" else self_policy
        concession = max(0.0, opening[counterpart] - last_demand[counterpart])
        demand = _demand(policy, phase=phase, counterpart_concession=concession, inferred_stubbornness=_clip(last_demand[counterpart]), rng=rng)
        last_demand[proposer] = demand
        message, message_act = _utterance(rng, "negotiation", policy, value=demand) if messages_allowed else ("", "none")
        opponent_share = demand if proposer == "opponent" else 1.0 - demand
        self_share = 1.0 - opponent_share
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=horizon_known, actor=proposer, kind="proposal", action="proposal", value=opponent_share, auxiliary=self_share, message=message, message_act=message_act))
        offered_share = 1.0 - demand
        accepted = _accept(counterpart_policy, offered_share=offered_share, phase=phase, inferred_stubbornness=demand, rng=rng)
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=horizon_known, actor=counterpart, kind="response", action="accept" if accepted else "reject", value=opponent_share, auxiliary=self_share, delay=_delay(counterpart_policy, surprise=offered_share - float(counterpart_policy["floor"]), rng=rng)))
        if accepted:
            break
        proposer = counterpart
    return events


def _persuasion(game_id: str, rng: random.Random, opponent: Mapping[str, object], self_policy: Mapping[str, object], *, maximum: int, our_role: str, probability_high: float, seller_knows: bool, price: float) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    opponent_is_seller = our_role == "buyer"
    seller_policy = opponent if opponent_is_seller else self_policy
    buyer_policy = self_policy if opponent_is_seller else opponent
    observed_truth = 0.5
    purchases = 0
    for round_number in range(1, maximum + 1):
        quality_high = rng.random() < probability_high
        truthfulness = float(seller_policy["truthfulness"])
        if seller_knows:
            truthful_signal = quality_high if rng.random() < truthfulness else not quality_high
        else:
            truthful_signal = rng.random() < probability_high
        seller_actor = "opponent" if opponent_is_seller else "self"
        buyer_actor = "self" if opponent_is_seller else "opponent"
        message, message_act = _utterance(rng, "persuasion", seller_policy, value=None, positive=truthful_signal)
        signal_label = "signal_positive" if truthful_signal else "signal_negative"
        if rng.random() < 0.08:
            signal_label = "signal_unknown"
            message_act = "binary-recommendation"
            message = _message(rng, "persuasion", message_act)
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=True, actor=seller_actor, kind="signal", action=signal_label, message=message, message_act=message_act))
        signal_value = 0.5 if signal_label == "signal_unknown" else float(signal_label == "signal_positive")
        inferred_high = _clip(0.25 * probability_high + 0.45 * signal_value + 0.30 * observed_truth)
        threshold = _clip(0.30 + price * 0.45 + (0.5 - float(buyer_policy["trust"])) * 0.25)
        if str(buyer_policy["kind"]) == "order-2":
            threshold += float(buyer_policy["order2_strength"]) * (0.5 - observed_truth) * 0.2
        buy_probability = 1.0 / (1.0 + math.exp(-(inferred_high - threshold) / max(0.03, float(buyer_policy["noise"]))))
        bought = rng.random() < buy_probability
        events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=True, actor=buyer_actor, kind="response", action="buy" if bought else "pass", delay=_delay(buyer_policy, surprise=inferred_high - threshold, rng=rng)))
        if bought:
            purchases += 1
            events.append(_event(game_id=game_id, index=len(events), round_number=round_number, maximum=maximum, horizon_known=True, actor="environment", kind="quality_reveal", action="quality_high" if quality_high else "quality_low", quality="high" if quality_high else "low"))
            correctness = float(truthful_signal == quality_high)
            observed_truth = (observed_truth * (purchases - 1) + correctness) / purchases
    return events


def _game_row(*, game_id: str, family: str, index: int, rng: random.Random, opponent: Mapping[str, object], maximum: int, horizon_known: bool, messages_allowed: bool, complete_information: bool, our_role: str, opponent_role: str, static_self_value: float | None, static_visible_opponent_value: float | None, probability: float | None = None, auxiliary: float | None = None, seller_knows: bool = False) -> dict[str, object]:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
    return {
        "game_id": game_id,
        "family": family,
        "source_type": "synthetic",
        "generator_id": f"policy-zoo-v2:{opponent['kind']}",
        "started_at": timestamp.isoformat(),
        "completed_at": (timestamp + timedelta(milliseconds=1)).isoformat(),
        "chronological_split": "train",
        "identity_scope": "hidden",
        "opponent_name_hash": hashlib.sha256(b"synthetic-hidden").hexdigest(),
        "account_key": None,
        "account_confidence": None,
        "account_fold": -1,
        "our_player": "player_1",
        "our_role": our_role,
        "opponent_role": opponent_role,
        "complete_information": complete_information,
        "horizon_known": horizon_known,
        "messages_allowed": messages_allowed,
        "max_rounds": maximum,
        "static_scale_log": auxiliary if family == "persuasion" and auxiliary is not None else math.log1p(rng.uniform(100.0, 1_000_000.0)) if family == "bargaining" else math.log1p(max(0.0, static_self_value or 0.0)),
        "static_self_value": static_self_value,
        "static_visible_opponent_value": static_visible_opponent_value if complete_information else None,
        "static_environment_probability": probability,
        "static_aux_value": auxiliary,
        "static_seller_knows_quality": seller_knows,
        "engine_version": "synthetic-policy-zoo-v2",
        "advisor_version": "",
        "policy_revision": str(opponent["kind"]),
        "archive_path": f"synthetic://{game_id}",
        "archive_sha256": hashlib.sha256(game_id.encode("utf-8")).hexdigest(),
    }


def _targets(game: Mapping[str, object], events: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        if event["actor"] != "opponent":
            continue
        value_present = event["kind"] == "proposal" and event["action_value"] is not None
        message_present = event["kind"] in {"proposal", "signal"} and game["messages_allowed"] is True
        delay = event.get("response_time_ms")
        rows.append(
            {
                "sample_id": hashlib.sha256(f"{game['game_id']}:{event['event_index']}".encode("utf-8")).hexdigest(),
                "game_id": game["game_id"],
                "source_type": "synthetic",
                "target_event_index": event["event_index"],
                "prefix_length": event["event_index"],
                "target_kind": event["kind"],
                "target_label": event["action_label"],
                "target_value": event["action_value"] if value_present else None,
                "target_value_present": value_present,
                "target_message_act": event["message_family_act"] if message_present else None,
                "target_message_present": message_present,
                "target_delay_log_ms": math.log1p(float(delay)) if isinstance(delay, (int, float)) and delay >= 0 else None,
                "target_delay_present": isinstance(delay, (int, float)) and delay >= 0,
                "chronological_split": "train",
                "identity_scope": "hidden",
                "account_key": None,
                "account_confidence": None,
                "account_fold": -1,
            }
        )
    return rows


class SyntheticPolicyZooBuilder:
    def __init__(self, *, output_dir: Path, games_per_family: int = 1_500, seed: int = 271_828) -> None:
        if games_per_family <= 0:
            raise ValueError("games_per_family must be positive")
        self.output_dir = output_dir.resolve()
        self.games_per_family = games_per_family
        self.seed = seed

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"synthetic corpus output already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        rng = random.Random(self.seed)
        game_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []
        target_rows: list[dict[str, object]] = []
        latent_rows: list[dict[str, object]] = []
        global_index = 0
        for family in GLEE_FAMILIES:
            for family_index in range(self.games_per_family):
                game_id = f"synthetic-{family}-{self.seed}-{family_index:06d}"
                opponent = _policy(rng)
                self_policy = _policy(rng)
                messages_allowed = True if family == "persuasion" else rng.random() < 0.55
                complete_information = False if family == "persuasion" else rng.random() < 0.55
                horizon_known = family == "persuasion" or rng.random() < 0.65
                if family == "bargaining":
                    maximum = rng.randint(5, 24)
                    events = _bargaining(game_id, rng, opponent, self_policy, maximum=maximum, horizon_known=horizon_known, messages_allowed=messages_allowed)
                    our_role, opponent_role = "player_1", "player_2"
                    self_value = rng.uniform(0.82, 1.0)
                    opponent_value = rng.uniform(0.82, 1.0)
                    game = _game_row(game_id=game_id, family=family, index=global_index, rng=rng, opponent=opponent, maximum=maximum, horizon_known=horizon_known, messages_allowed=messages_allowed, complete_information=complete_information, our_role=our_role, opponent_role=opponent_role, static_self_value=self_value, static_visible_opponent_value=opponent_value)
                elif family == "negotiation":
                    maximum = rng.randint(5, 20)
                    events = _negotiation(game_id, rng, opponent, self_policy, maximum=maximum, horizon_known=horizon_known, messages_allowed=messages_allowed)
                    our_role = rng.choice(("buyer", "seller"))
                    opponent_role = "seller" if our_role == "buyer" else "buyer"
                    seller_value = rng.uniform(20.0, 120.0)
                    buyer_value = seller_value + rng.uniform(15.0, 140.0)
                    self_value = buyer_value if our_role == "buyer" else seller_value
                    opponent_value = seller_value if our_role == "buyer" else buyer_value
                    game = _game_row(game_id=game_id, family=family, index=global_index, rng=rng, opponent=opponent, maximum=maximum, horizon_known=horizon_known, messages_allowed=messages_allowed, complete_information=complete_information, our_role=our_role, opponent_role=opponent_role, static_self_value=self_value, static_visible_opponent_value=opponent_value)
                else:
                    maximum = rng.randint(5, 22)
                    our_role = rng.choice(("buyer", "seller"))
                    opponent_role = "seller" if our_role == "buyer" else "buyer"
                    probability = rng.uniform(0.10, 0.90)
                    seller_knows = rng.random() < 0.55
                    price = rng.uniform(0.10, 0.90)
                    events = _persuasion(game_id, rng, opponent, self_policy, maximum=maximum, our_role=our_role, probability_high=probability, seller_knows=seller_knows, price=price)
                    auxiliary = math.log1p(price * 100.0)
                    game = _game_row(game_id=game_id, family=family, index=global_index, rng=rng, opponent=opponent, maximum=maximum, horizon_known=True, messages_allowed=True, complete_information=False, our_role=our_role, opponent_role=opponent_role, static_self_value=None, static_visible_opponent_value=None, probability=probability, auxiliary=auxiliary, seller_knows=seller_knows)
                global_index += 1
                if not events or not any(event["actor"] == "opponent" for event in events):
                    continue
                game_rows.append(game)
                event_rows.extend(events)
                target_rows.extend(_targets(game, events))
                latent_rows.append({"game_id": game_id, "family": family, "opponent_policy": canonical_json(opponent), "self_policy": canonical_json(self_policy)})
        games = pl.DataFrame(game_rows, schema=GAME_SCHEMA, strict=False).sort(["family", "game_id"])
        events = pl.DataFrame(event_rows, schema=EVENT_SCHEMA, strict=False).sort(["game_id", "event_index"])
        targets = pl.DataFrame(target_rows, schema=TARGET_SCHEMA, strict=False).sort(["game_id", "target_event_index"])
        latents = pl.DataFrame(latent_rows, schema={"game_id": pl.String, "family": pl.String, "opponent_policy": pl.String, "self_policy": pl.String}, strict=False).sort(["family", "game_id"])
        artifacts = {
            "games.parquet": _artifact(games, staging / "games.parquet"),
            "events.parquet": _artifact(events, staging / "events.parquet"),
            "targets.parquet": _artifact(targets, staging / "targets.parquet"),
            "latents.parquet": _artifact(latents, staging / "latents.parquet"),
        }
        manifest = {
            "schema_version": 1,
            "contract": CORPUS_CONTRACT,
            "source": {"kind": "synthetic-policy-zoo", "synthetic_contract": SYNTHETIC_CONTRACT},
            "parameters": {"seed": self.seed, "games_per_family": self.games_per_family, "policy_kinds": list(POLICY_KINDS), "account_labels": "forbidden", "split": "train-only"},
            "inventory": {
                "games": games.height,
                "events": events.height,
                "targets": targets.height,
                "policy_kinds": dict(sorted(Counter(str(json.loads(value)["kind"]) for value in latents["opponent_policy"].to_list()).items())),
                "by_family": {family: {"games": games.filter(pl.col("family") == family).height, "events": events.join(games.select("game_id", "family"), on="game_id").filter(pl.col("family") == family).height, "targets": targets.join(games.select("game_id", "family"), on="game_id").filter(pl.col("family") == family).height} for family in GLEE_FAMILIES},
            },
            "artifacts": artifacts,
            "implementation_sha256": file_sha256(Path(__file__)),
            "scientific_boundary": "Synthetic rows pretrain population dynamics only and never label a real account or enter real validation or test evaluation.",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha256 = file_sha256(staging / "manifest.json")
        os.replace(staging, self.output_dir)
        return {"contract": SYNTHETIC_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": manifest_sha256, "inventory": manifest["inventory"]}
