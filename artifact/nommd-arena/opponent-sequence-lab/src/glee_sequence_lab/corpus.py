"""Build a normalized, reference-based causal sequence corpus from GLEE receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from nommd_arena.glee_analytics_online import GleeOnlineAnalyticsLakeReader
from nommd_arena.glee_bargaining_twin import classify_message_act
from nommd_arena.glee_behavior_corpus import _lexical_signature
from nommd_arena.glee_negotiation_twin_v2 import classify_negotiation_message, opponent_demand, opponent_surplus_share
from nommd_arena.glee_persuasion_twin_v2 import classify_persuasion_signal


CORPUS_CONTRACT = "glee-hierarchical-sequence-corpus-v1"
HASH_BINS = 4096
MAX_MESSAGE_HASHES = 192
ACCEPTED_ACCOUNT_CONFIDENCE = frozenset({"high", "very-high"})
GLEE_FAMILIES = ("bargaining", "negotiation", "persuasion")


GAME_SCHEMA = {
    "game_id": pl.String,
    "family": pl.String,
    "source_type": pl.String,
    "generator_id": pl.String,
    "started_at": pl.String,
    "completed_at": pl.String,
    "chronological_split": pl.String,
    "identity_scope": pl.String,
    "opponent_name_hash": pl.String,
    "account_key": pl.String,
    "account_confidence": pl.String,
    "account_fold": pl.Int32,
    "our_player": pl.String,
    "our_role": pl.String,
    "opponent_role": pl.String,
    "complete_information": pl.Boolean,
    "horizon_known": pl.Boolean,
    "messages_allowed": pl.Boolean,
    "max_rounds": pl.Int32,
    "static_scale_log": pl.Float64,
    "static_self_value": pl.Float64,
    "static_visible_opponent_value": pl.Float64,
    "static_environment_probability": pl.Float64,
    "static_aux_value": pl.Float64,
    "static_seller_knows_quality": pl.Boolean,
    "engine_version": pl.String,
    "advisor_version": pl.String,
    "policy_revision": pl.String,
    "archive_path": pl.String,
    "archive_sha256": pl.String,
}

EVENT_SCHEMA = {
    "game_id": pl.String,
    "event_index": pl.Int32,
    "round_number": pl.Int32,
    "round_phase": pl.Float64,
    "actor": pl.String,
    "kind": pl.String,
    "action_label": pl.String,
    "action_value": pl.Float64,
    "action_aux_value": pl.Float64,
    "response_time_ms": pl.Float64,
    "visible_quality": pl.String,
    "message_present": pl.Boolean,
    "message_family_act": pl.String,
    "message_discourse_acts": pl.List(pl.String),
    "message_hash_bins": pl.List(pl.Int32),
    "message_chars": pl.Int32,
    "message_words": pl.Int32,
    "message_uppercase_ratio": pl.Float64,
    "message_digit_ratio": pl.Float64,
    "message_question_marks": pl.Int32,
    "message_exclamation_marks": pl.Int32,
    "message_commas": pl.Int32,
    "message_semicolons": pl.Int32,
    "message_currency_marks": pl.Int32,
    "message_percent_marks": pl.Int32,
    "message_decimal_numbers": pl.Int32,
    "message_sha256": pl.String,
}

TARGET_SCHEMA = {
    "sample_id": pl.String,
    "game_id": pl.String,
    "source_type": pl.String,
    "target_event_index": pl.Int32,
    "prefix_length": pl.Int32,
    "target_kind": pl.String,
    "target_label": pl.String,
    "target_value": pl.Float64,
    "target_value_present": pl.Boolean,
    "target_message_act": pl.String,
    "target_message_present": pl.Boolean,
    "target_delay_log_ms": pl.Float64,
    "target_delay_present": pl.Boolean,
    "chronological_split": pl.String,
    "identity_scope": pl.String,
    "account_key": pl.String,
    "account_confidence": pl.String,
    "account_fold": pl.Int32,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player: {player!r}")


def _number(value: object, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _round_phase(round_number: int, state: Mapping[str, object]) -> float:
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True and isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 1:
        return min(1.0, max(0.0, (round_number - 1) / (maximum - 1)))
    total = state.get("total_rounds")
    persuasion_setting = "p" in state and "seller_message_type" in state
    if persuasion_setting and isinstance(total, int) and not isinstance(total, bool) and total > 1:
        return min(1.0, max(0.0, (round_number - 1) / (total - 1)))
    return 1.0 - math.exp(-max(0, round_number - 1) / 12.0)


def _message_features(message: object, *, family_act: str) -> dict[str, object]:
    signature = _lexical_signature(message, family_act=family_act)
    style = signature.get("style") if isinstance(signature.get("style"), Mapping) else {}
    hashed = signature.get("hashed_lexemes") if isinstance(signature.get("hashed_lexemes"), Mapping) else {}
    bins: list[int] = []
    for token, raw_count in sorted(hashed.items()):
        try:
            bin_id = int(str(token), 16) % HASH_BINS
            count = max(1, min(3, int(raw_count)))
        except (TypeError, ValueError):
            continue
        bins.extend([bin_id] * count)
        if len(bins) >= MAX_MESSAGE_HASHES:
            bins = bins[:MAX_MESSAGE_HASHES]
            break
    return {
        "message_present": bool(signature.get("present")),
        "message_family_act": str(signature.get("family_act") or family_act or "none"),
        "message_discourse_acts": [str(value) for value in signature.get("discourse_acts", ["silence"])],
        "message_hash_bins": bins,
        "message_chars": int(style.get("chars") or 0),
        "message_words": int(style.get("words") or 0),
        "message_uppercase_ratio": float(style.get("uppercase_ratio") or 0.0),
        "message_digit_ratio": float(style.get("digit_ratio") or 0.0),
        "message_question_marks": int(style.get("question_marks") or 0),
        "message_exclamation_marks": int(style.get("exclamation_marks") or 0),
        "message_commas": int(style.get("commas") or 0),
        "message_semicolons": int(style.get("semicolons") or 0),
        "message_currency_marks": int(style.get("currency_marks") or 0),
        "message_percent_marks": int(style.get("percent_marks") or 0),
        "message_decimal_numbers": int(style.get("decimal_numbers") or 0),
        "message_sha256": str(signature.get("message_sha256") or hashlib.sha256(b"").hexdigest()),
    }


def _event(*, game_id: str, event_index: int, round_number: int, state: Mapping[str, object], actor: str, kind: str, action_label: str, action_value: float | None = None, action_aux_value: float | None = None, response_time_ms: float | None = None, visible_quality: str | None = None, message: object = "", family_act: str = "none") -> dict[str, object]:
    return {
        "game_id": game_id,
        "event_index": event_index,
        "round_number": round_number,
        "round_phase": _round_phase(round_number, state),
        "actor": actor,
        "kind": kind,
        "action_label": action_label,
        "action_value": action_value,
        "action_aux_value": action_aux_value,
        "response_time_ms": response_time_ms,
        "visible_quality": visible_quality,
        **_message_features(message, family_act=family_act),
    }


def _bargaining_events(game: Mapping[str, Any]) -> list[dict[str, object]]:
    game_id = str(game["game_id"])
    state = game["game_state"]
    if not isinstance(state, Mapping):
        raise ValueError("bargaining archive has no state")
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("bargaining archive has no history")
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    pool = _number(state.get("money_to_divide"))
    if pool is None or pool <= 0:
        raise ValueError("bargaining pool must be positive")
    events: list[dict[str, object]] = []
    for offset, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed bargaining history")
        offer = raw.get("offer")
        if not isinstance(offer, Mapping):
            if str(raw.get("decision") or "").casefold() == "timeout":
                continue
            raise ValueError("bargaining history entry has no offer")
        proposer = str(raw.get("proposer") or offer.get("proposer") or "")
        responder = _other_player(proposer)
        round_number = int(raw.get("round") or offer.get("round") or offset + 1)
        opponent_gain = _number(offer.get(f"{opponent_player}_gain"))
        self_gain = _number(offer.get(f"{our_player}_gain"))
        if opponent_gain is None or self_gain is None:
            raise ValueError("bargaining offer has incomplete gains")
        message = str(offer.get("message") or "")
        family_act = classify_message_act(message, messages_allowed=state.get("messages_allowed") is True)
        events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if proposer == our_player else "opponent", kind="proposal", action_label="proposal", action_value=opponent_gain / pool, action_aux_value=self_gain / pool, message=message, family_act=family_act))
        decision = str(raw.get("decision") or "").casefold()
        if decision:
            if decision == "timeout":
                continue
            if decision not in {"accept", "reject", "walkaway", "walk_away"}:
                raise ValueError(f"unsupported bargaining decision: {decision!r}")
            standardized = "walkaway" if decision == "walk_away" else decision
            delay = _number(raw.get("response_time_ms"))
            events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if responder == our_player else "opponent", kind="response", action_label=standardized, action_value=opponent_gain / pool, action_aux_value=self_gain / pool, response_time_ms=delay))
    return events


def _negotiation_events(game: Mapping[str, Any]) -> list[dict[str, object]]:
    game_id = str(game["game_id"])
    state = game["game_state"]
    if not isinstance(state, Mapping):
        raise ValueError("negotiation archive has no state")
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("negotiation archive has no history")
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    our_value = _number(state.get(f"{our_player}_value"))
    if our_value is None or our_value <= 0 or {our_role, opponent_role} != {"buyer", "seller"}:
        raise ValueError("negotiation archive has invalid roles or own value")
    visible_opponent_value = _number(state.get(f"{opponent_player}_value")) if state.get("complete_information") is True else None
    events: list[dict[str, object]] = []
    response_labels = {"AcceptOffer": "accept", "RejectOffer": "reject", "WalkAway": "walkaway"}
    for offset, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed negotiation history")
        offer = raw.get("offer")
        if not isinstance(offer, Mapping):
            raise ValueError("negotiation history entry has no offer")
        proposer = str(offer.get("from_player") or "")
        responder = str(raw.get("decided_by") or _other_player(proposer))
        round_number = int(raw.get("round") or offer.get("round") or offset + 1)
        price = _number(offer.get("price"))
        if price is None or price < 0:
            raise ValueError("negotiation price must be nonnegative")
        demand = opponent_demand(price, opponent_role=opponent_role, our_value=our_value)
        share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=visible_opponent_value)
        message = str(offer.get("message") or "")
        family_act = classify_negotiation_message(message, messages_allowed=state.get("messages_allowed") is not False)
        events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if proposer == our_player else "opponent", kind="proposal", action_label="proposal", action_value=math.tanh(demand / 3.0), action_aux_value=share if share is not None else demand, message=message, family_act=family_act))
        decision = str(raw.get("decision") or "")
        if decision:
            if decision == "timeout":
                continue
            if decision not in response_labels:
                raise ValueError(f"unsupported negotiation decision: {decision!r}")
            delay = _number(raw.get("response_time_ms"))
            events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if responder == our_player else "opponent", kind="response", action_label=response_labels[decision], action_value=math.tanh(demand / 3.0), action_aux_value=share if share is not None else demand, response_time_ms=delay))
    return events


def _persuasion_events(game: Mapping[str, Any]) -> list[dict[str, object]]:
    game_id = str(game["game_id"])
    state = game["game_state"]
    if not isinstance(state, Mapping):
        raise ValueError("persuasion archive has no state")
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("persuasion archive has no history")
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    if {our_role, opponent_role} != {"buyer", "seller"}:
        raise ValueError("persuasion archive has invalid roles")
    seller = our_player if our_role == "seller" else opponent_player
    buyer = _other_player(seller)
    channel = str(state.get("seller_message_type") or "text").casefold()
    seller_knows = state.get("is_seller_know_cv") is True
    events: list[dict[str, object]] = []
    for offset, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed persuasion history")
        round_number = int(raw.get("round") or offset + 1)
        message = str(raw.get("seller_message") or "")
        polarity, family_act, _fingerprint = classify_persuasion_signal(message, channel=channel)
        quality = str(raw.get("quality") or "").casefold()
        signal_value = 1.0 if polarity == "positive" else -1.0 if polarity == "negative" else 0.0
        visible_to_self = quality if seller == our_player and seller_knows and quality in {"high", "low"} else None
        events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if seller == our_player else "opponent", kind="signal", action_label=f"signal_{polarity}", action_value=signal_value, visible_quality=visible_to_self, message=message, family_act=family_act))
        response_present = isinstance(raw.get("bought"), bool) or bool(str(raw.get("buyer_decision") or "").strip())
        if response_present:
            bought = raw.get("bought") is True or str(raw.get("buyer_decision") or "").casefold() == "yes"
            delay = _number(raw.get("response_time_ms"))
            events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self" if buyer == our_player else "opponent", kind="response", action_label="buy" if bought else "pass", action_value=1.0 if bought else 0.0, response_time_ms=delay))
            if bought and quality in {"high", "low"}:
                events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="environment", kind="quality_reveal", action_label=f"quality_{quality}", action_value=1.0 if quality == "high" else -1.0, visible_quality=quality))
    return events


def extract_events(game: Mapping[str, Any]) -> list[dict[str, object]]:
    family = str(game.get("game_family") or "")
    if family == "bargaining":
        return _bargaining_events(game)
    if family == "negotiation":
        return _negotiation_events(game)
    if family == "persuasion":
        return _persuasion_events(game)
    raise ValueError(f"unsupported game family: {family!r}")


def load_account_map(path: Path) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("account map has no groups")
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raw in groups:
        if not isinstance(raw, Mapping):
            continue
        account = str(raw.get("key") or "")
        confidence = str(raw.get("confidence") or "unknown")
        members = raw.get("members")
        if not account or confidence not in ACCEPTED_ACCOUNT_CONFIDENCE or not isinstance(members, Mapping):
            continue
        for label in members:
            normalized = normalized_label(label)
            if normalized:
                candidates[normalized].append((account, confidence))
    mapping: dict[str, tuple[str, str]] = {}
    collisions: dict[str, list[str]] = {}
    for label, values in candidates.items():
        unique = sorted(set(values))
        if len(unique) == 1:
            mapping[label] = unique[0]
        else:
            collisions[label] = sorted({account for account, _confidence in unique})
    return mapping, collisions


def _static_game(row: Mapping[str, object], game: Mapping[str, Any], *, split: str, account_map: Mapping[str, tuple[str, str]]) -> dict[str, object]:
    family = str(row["family"])
    state = game["game_state"]
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    identity_scope = str(row.get("identity_scope") or "archive-missing")
    normalized_name = normalized_label(row.get("opponent_name")) if identity_scope == "known" else ""
    account = account_map.get(normalized_name)
    account_key = account[0] if account else None
    account_confidence = account[1] if account else None
    account_fold = int(hashlib.sha256(account_key.encode("utf-8")).hexdigest()[:8], 16) % 5 if account_key else -1
    complete_information = state.get("complete_information") is True
    horizon_known = state.get("horizon_known") is True or family == "persuasion"
    maximum = state.get("max_rounds") if family != "persuasion" else state.get("total_rounds")
    max_rounds = int(maximum) if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 else None
    static_scale_log = 0.0
    static_self_value = None
    static_visible_opponent_value = None
    static_environment_probability = None
    static_aux_value = None
    seller_knows_quality = False
    our_role = our_player
    opponent_role = opponent_player
    messages_allowed = state.get("messages_allowed") is True
    if family == "bargaining":
        pool = _number(state.get("money_to_divide"), 0.0) or 0.0
        static_scale_log = math.log1p(max(0.0, pool))
        static_self_value = _number(state.get("delta_1" if our_player == "player_1" else "delta_2"))
        static_visible_opponent_value = _number(state.get("delta_2" if our_player == "player_1" else "delta_1")) if complete_information else None
    elif family == "negotiation":
        our_role = str(state.get(f"{our_player}_role") or "")
        opponent_role = str(state.get(f"{opponent_player}_role") or "")
        static_self_value = _number(state.get(f"{our_player}_value"))
        static_visible_opponent_value = _number(state.get(f"{opponent_player}_value")) if complete_information else None
        static_scale_log = math.log1p(max(0.0, static_self_value or 0.0))
        messages_allowed = state.get("messages_allowed") is not False
    elif family == "persuasion":
        our_role = str(state.get(f"{our_player}_role") or "")
        opponent_role = str(state.get(f"{opponent_player}_role") or "")
        static_environment_probability = _number(state.get("p"))
        product_price = _number(state.get("product_price"), 0.0) or 0.0
        static_aux_value = math.log1p(max(0.0, product_price))
        static_scale_log = static_aux_value
        seller_knows_quality = state.get("is_seller_know_cv") is True
        messages_allowed = str(state.get("seller_message_type") or "text").casefold() in {"binary", "text"}
    return {
        "game_id": str(row["game_id"]),
        "family": family,
        "source_type": "real",
        "generator_id": None,
        "started_at": str(row["started_at"]),
        "completed_at": str(row["completed_at"]),
        "chronological_split": split,
        "identity_scope": identity_scope,
        "opponent_name_hash": hashlib.sha256(normalized_name.encode("utf-8")).hexdigest() if normalized_name else hashlib.sha256(b"hidden").hexdigest(),
        "account_key": account_key,
        "account_confidence": account_confidence,
        "account_fold": account_fold,
        "our_player": our_player,
        "our_role": our_role,
        "opponent_role": opponent_role,
        "complete_information": complete_information,
        "horizon_known": horizon_known,
        "messages_allowed": messages_allowed,
        "max_rounds": max_rounds,
        "static_scale_log": static_scale_log,
        "static_self_value": static_self_value,
        "static_visible_opponent_value": static_visible_opponent_value,
        "static_environment_probability": static_environment_probability,
        "static_aux_value": static_aux_value,
        "static_seller_knows_quality": seller_knows_quality,
        "engine_version": str(row.get("engine_version") or "engine-unknown"),
        "advisor_version": str(row.get("advisor_version") or ""),
        "policy_revision": str(row.get("policy_revision") or ""),
        "archive_path": str(row["archive_path"]),
        "archive_sha256": str(row["archive_sha256"]),
    }


def _chronological_splits(rows: Sequence[Mapping[str, object]], *, train_fraction: float, validation_fraction: float) -> dict[str, str]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("invalid chronological split fractions")
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    assignments: dict[str, str] = {}
    for family in GLEE_FAMILIES:
        selected = sorted(by_family[family], key=lambda row: (str(row["completed_at"]), str(row["game_id"])))
        train_end = int(len(selected) * train_fraction)
        validation_end = int(len(selected) * (train_fraction + validation_fraction))
        for index, row in enumerate(selected):
            assignments[str(row["game_id"])] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return assignments


def _artifact(frame: pl.DataFrame, path: Path) -> dict[str, object]:
    frame.write_parquet(path, compression="zstd", compression_level=7, statistics=True, row_group_size=4096)
    reopened = pl.scan_parquet(path).select(pl.len()).collect().item(0, 0)
    if reopened != frame.height:
        raise RuntimeError(f"Parquet row-count mismatch: {path}")
    return {"path": path.name, "rows": frame.height, "columns": frame.width, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SequenceCorpusBuilder:
    """Freeze a verified analytics frontier into one game/event/target corpus."""

    def __init__(self, *, analytics_root: Path, archive_root: Path, account_groups: Path, output_dir: Path, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> None:
        self.analytics_root = analytics_root.resolve()
        self.archive_root = archive_root.resolve()
        self.account_groups = account_groups.resolve()
        self.output_dir = output_dir.resolve()
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"corpus output already exists: {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        reader = GleeOnlineAnalyticsLakeReader(self.analytics_root)
        scan, selection = reader.scan("games.parquet")
        source_rows = scan.collect().sort(["completed_at", "game_id"]).to_dicts()
        assignments = _chronological_splits(source_rows, train_fraction=self.train_fraction, validation_fraction=self.validation_fraction)
        account_map, account_collisions = load_account_map(self.account_groups)
        game_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []
        target_rows: list[dict[str, object]] = []
        exclusions: Counter[str] = Counter()
        for row in source_rows:
            archive_value = row.get("archive_path")
            if not isinstance(archive_value, str) or not archive_value:
                exclusions["archive-missing"] += 1
                continue
            path = (self.archive_root / archive_value).resolve()
            if self.archive_root not in path.parents:
                exclusions["archive-path-escape"] += 1
                continue
            try:
                game = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                exclusions["archive-unreadable"] += 1
                continue
            if not isinstance(game, Mapping) or object_sha256(game) != str(row.get("archive_sha256") or ""):
                exclusions["archive-hash-mismatch"] += 1
                continue
            if game.get("game_id") != row["game_id"] or game.get("game_family") != row["family"]:
                exclusions["archive-identity-mismatch"] += 1
                continue
            try:
                events = extract_events(game)
                game_row = _static_game(row, game, split=assignments[str(row["game_id"])], account_map=account_map)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                reason = re.sub(r"\s+", " ", str(error)).strip()[:160] or "unspecified"
                exclusions[f"feature-extraction-invalid:{row['family']}:{type(error).__name__}:{reason}"] += 1
                continue
            if not events or not any(event["actor"] == "opponent" for event in events):
                exclusions["no-opponent-target"] += 1
                continue
            game_rows.append(game_row)
            event_rows.extend(events)
            for event in events:
                if event["actor"] != "opponent":
                    continue
                target_value_present = event["kind"] == "proposal" and event["action_value"] is not None
                target_message_present = event["kind"] in {"proposal", "signal"} and game_row["messages_allowed"] is True
                delay = event.get("response_time_ms")
                target_rows.append(
                    {
                        "sample_id": hashlib.sha256(f"{row['game_id']}:{event['event_index']}".encode("utf-8")).hexdigest(),
                        "game_id": str(row["game_id"]),
                        "source_type": "real",
                        "target_event_index": int(event["event_index"]),
                        "prefix_length": int(event["event_index"]),
                        "target_kind": str(event["kind"]),
                        "target_label": str(event["action_label"]),
                        "target_value": float(event["action_value"]) if target_value_present else None,
                        "target_value_present": target_value_present,
                        "target_message_act": str(event["message_family_act"]) if target_message_present else None,
                        "target_message_present": target_message_present,
                        "target_delay_log_ms": math.log1p(float(delay)) if isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay >= 0 else None,
                        "target_delay_present": isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay >= 0,
                        "chronological_split": game_row["chronological_split"],
                        "identity_scope": game_row["identity_scope"],
                        "account_key": game_row["account_key"],
                        "account_confidence": game_row["account_confidence"],
                        "account_fold": game_row["account_fold"],
                    }
                )
        games = pl.DataFrame(game_rows, schema=GAME_SCHEMA, strict=False).sort(["completed_at", "game_id"])
        events = pl.DataFrame(event_rows, schema=EVENT_SCHEMA, strict=False).sort(["game_id", "event_index"])
        targets = pl.DataFrame(target_rows, schema=TARGET_SCHEMA, strict=False).sort(["game_id", "target_event_index"])
        if games["game_id"].n_unique() != games.height:
            raise RuntimeError("corpus contains duplicate games")
        if events.select(pl.struct("game_id", "event_index").n_unique()).item(0, 0) != events.height:
            raise RuntimeError("corpus contains duplicate event coordinates")
        if targets["sample_id"].n_unique() != targets.height:
            raise RuntimeError("corpus contains duplicate targets")
        artifacts = {
            "games.parquet": _artifact(games, staging / "games.parquet"),
            "events.parquet": _artifact(events, staging / "events.parquet"),
            "targets.parquet": _artifact(targets, staging / "targets.parquet"),
        }
        inventory = {
            "source_games": len(source_rows),
            "games": games.height,
            "events": events.height,
            "targets": targets.height,
            "account_labeled_games": games.filter(pl.col("account_key").is_not_null()).height,
            "account_keys": games["account_key"].drop_nulls().n_unique(),
            "by_family": {
                family: {
                    "games": games.filter(pl.col("family") == family).height,
                    "events": events.join(games.select("game_id", "family"), on="game_id").filter(pl.col("family") == family).height,
                    "targets": targets.join(games.select("game_id", "family"), on="game_id").filter(pl.col("family") == family).height,
                }
                for family in GLEE_FAMILIES
            },
            "by_split": dict(sorted(Counter(games["chronological_split"].to_list()).items())),
            "exclusions": dict(sorted(exclusions.items())),
            "account_label_collisions": account_collisions,
        }
        manifest = {
            "schema_version": 1,
            "contract": CORPUS_CONTRACT,
            "status": "frozen-retrospective-core-corpus",
            "source": {
                "analytics_root": str(self.analytics_root),
                "selection": {key: selection.get(key) for key in ("release", "manifest_sha256", "frontier_sequence", "history_games", "history_revision_rowid", "reporter_change_sequence", "release_contract", "profile")},
                "archive_root": str(self.archive_root),
                "account_groups": str(self.account_groups),
                "account_groups_sha256": file_sha256(self.account_groups),
            },
            "parameters": {
                "train_fraction": self.train_fraction,
                "validation_fraction": self.validation_fraction,
                "test_fraction": 1.0 - self.train_fraction - self.validation_fraction,
                "accepted_account_confidence": sorted(ACCEPTED_ACCOUNT_CONFIDENCE),
                "hash_bins": HASH_BINS,
                "maximum_message_hashes": MAX_MESSAGE_HASHES,
                "persuasion_quality_boundary": "quality appears only as self-visible seller context or after a purchase reveal",
            },
            "inventory": inventory,
            "artifacts": artifacts,
            "implementation_sha256": file_sha256(Path(__file__)),
        }
        _write_json(staging / "manifest.json", manifest)
        manifest_sha256 = file_sha256(staging / "manifest.json")
        os.replace(staging, self.output_dir)
        return {"contract": CORPUS_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": manifest_sha256, "inventory": inventory, "source": manifest["source"]}
