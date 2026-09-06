"""Build a public-information sequence corpus that predicts DeepRMM-01's next move."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping

import polars as pl

from nommd_arena.glee_negotiation_twin_v2 import price_from_opponent_demand

from .corpus import EVENT_SCHEMA, GAME_SCHEMA, GLEE_FAMILIES, TARGET_SCHEMA, _artifact, file_sha256
from .data import TARGET_LABELS


SELF_MIRROR_CORPUS_CONTRACT = "glee-public-information-self-mirror-corpus-v1"
SELF_MIRROR_OBJECTIVE_CONTRACT = "glee-public-information-self-move-forecast-v1"
SUPPORTED_SOURCE_CONTRACTS = frozenset({"glee-hierarchical-sequence-corpus-v1", "glee-pre-terra-feature-corpus-v3", "glee-post-planner-action-conditional-corpus-v3"})
PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR = 20.0


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_artifact(source_dir: Path, manifest: Mapping[str, object], name: str) -> Path:
    artifacts = manifest.get("artifacts")
    receipt = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    path = source_dir / name
    if not isinstance(receipt, Mapping) or not path.is_file() or file_sha256(path) != receipt.get("sha256"):
        raise RuntimeError(f"self-mirror source artifact hash mismatch: {name}")
    return path


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _public_negotiation_price(event: Mapping[str, object], game: Mapping[str, object]) -> float | None:
    action_value = _finite_number(event.get("action_value"))
    if action_value is None:
        return None
    our_value = _finite_number(game.get("static_self_value"))
    opponent_role = str(game.get("opponent_role") or "")
    if our_value is None or our_value <= 0 or opponent_role not in {"buyer", "seller"}:
        raise ValueError("negotiation source lacks the private normalization needed only to recover public price")
    auxiliary = _finite_number(event.get("action_aux_value"))
    if game.get("complete_information") is not True and auxiliary is not None:
        demand = auxiliary
    else:
        bounded = min(1.0 - 1e-12, max(-1.0 + 1e-12, action_value))
        demand = 3.0 * math.atanh(bounded)
    price = price_from_opponent_demand(demand, opponent_role=opponent_role, our_value=our_value)
    return math.log1p(price) / PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR


def _sanitize_game(row: Mapping[str, object]) -> dict[str, object]:
    game = dict(row)
    if game.get("complete_information") is not True:
        game["static_self_value"] = None
        game["static_visible_opponent_value"] = None
    if game.get("family") == "negotiation":
        game["static_scale_log"] = 0.0
    if game.get("identity_scope") != "known":
        game["account_key"] = None
        game["account_confidence"] = None
        game["account_fold"] = -1
    game["engine_version"] = ""
    game["advisor_version"] = ""
    game["policy_revision"] = ""
    return game


def _sanitize_event(row: Mapping[str, object], *, game: Mapping[str, object]) -> dict[str, object]:
    event = dict(row)
    if event.get("actor") != "environment":
        event["visible_quality"] = None
    if game.get("family") == "negotiation" and event.get("kind") in {"proposal", "response"}:
        event["action_value"] = _public_negotiation_price(event, game)
        event["action_aux_value"] = None
    return event


def _target(event: Mapping[str, object], *, game: Mapping[str, object]) -> dict[str, object]:
    target_value = _finite_number(event.get("action_value"))
    delay = _finite_number(event.get("response_time_ms"))
    game_id = str(game["game_id"])
    event_index = int(event["event_index"])
    return {
        "sample_id": hashlib.sha256(f"self-mirror-v1:{game_id}:{event_index}".encode("utf-8")).hexdigest(),
        "game_id": game_id,
        "source_type": "real",
        "target_event_index": event_index,
        "prefix_length": event_index,
        "target_kind": str(event["kind"]),
        "target_label": str(event["action_label"]),
        "target_value": target_value if event.get("kind") == "proposal" else None,
        "target_value_present": event.get("kind") == "proposal" and target_value is not None,
        "target_message_act": None,
        "target_message_present": False,
        "target_delay_log_ms": math.log1p(delay) if delay is not None and delay >= 0 else None,
        "target_delay_present": delay is not None and delay >= 0,
        "chronological_split": str(game["chronological_split"]),
        "identity_scope": str(game["identity_scope"]),
        "account_key": game.get("account_key"),
        "account_confidence": game.get("account_confidence"),
        "account_fold": int(game.get("account_fold") if game.get("account_fold") is not None else -1),
    }


class PublicSelfMirrorCorpusBuilder:
    """Derive self-action targets from a frozen corpus while retaining only opponent-observable model inputs."""

    def __init__(self, *, source_corpus: Path, output_dir: Path) -> None:
        self.source_corpus = source_corpus.resolve()
        self.output_dir = output_dir.resolve()

    def run(self) -> dict[str, object]:
        if self.output_dir.exists():
            raise FileExistsError(f"self-mirror corpus output already exists: {self.output_dir}")
        source_manifest_path = self.source_corpus / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("contract") not in SUPPORTED_SOURCE_CONTRACTS or source_manifest.get("status") != "frozen-retrospective-core-corpus":
            raise ValueError("self-mirror source corpus is unsupported or not frozen")
        games_path = _source_artifact(self.source_corpus, source_manifest, "games.parquet")
        events_path = _source_artifact(self.source_corpus, source_manifest, "events.parquet")
        source_games = pl.read_parquet(games_path).sort(["completed_at", "game_id"]).to_dicts()
        source_events = pl.read_parquet(events_path).sort(["game_id", "event_index"]).to_dicts()
        original_games = {str(row["game_id"]): row for row in source_games}
        sanitized_games = {game_id: _sanitize_game(row) for game_id, row in original_games.items()}
        event_rows: list[dict[str, object]] = []
        target_rows: list[dict[str, object]] = []
        games_with_targets: set[str] = set()
        for source_event in source_events:
            game_id = str(source_event["game_id"])
            original_game = original_games.get(game_id)
            if original_game is None:
                raise RuntimeError(f"self-mirror event references an absent game: {game_id}")
            event = _sanitize_event(source_event, game=original_game)
            event_rows.append(event)
            if event.get("actor") == "self":
                target_rows.append(_target(event, game=sanitized_games[game_id]))
                games_with_targets.add(game_id)
        game_rows = [sanitized_games[str(row["game_id"])] for row in source_games if str(row["game_id"]) in games_with_targets]
        event_rows = [row for row in event_rows if str(row["game_id"]) in games_with_targets]
        games = pl.DataFrame(game_rows, schema=GAME_SCHEMA, strict=False).sort(["completed_at", "game_id"])
        events = pl.DataFrame(event_rows, schema=EVENT_SCHEMA, strict=False).sort(["game_id", "event_index"])
        targets = pl.DataFrame(target_rows, schema=TARGET_SCHEMA, strict=False).sort(["game_id", "target_event_index"])
        if games["game_id"].n_unique() != games.height:
            raise RuntimeError("self-mirror corpus contains duplicate games")
        if events.select(pl.struct("game_id", "event_index").n_unique()).item(0, 0) != events.height:
            raise RuntimeError("self-mirror corpus contains duplicate event coordinates")
        if targets["sample_id"].n_unique() != targets.height or targets.filter(pl.col("prefix_length") != pl.col("target_event_index")).height:
            raise RuntimeError("self-mirror target identity or causal prefix contract failed")
        if targets.join(events.select("game_id", "event_index", "actor"), left_on=["game_id", "target_event_index"], right_on=["game_id", "event_index"]).filter(pl.col("actor") != "self").height:
            raise RuntimeError("self-mirror corpus contains a non-self target")
        if set(games["family"].unique().to_list()) != set(GLEE_FAMILIES) or set(games["chronological_split"].unique().to_list()) != {"train", "validation", "test"}:
            raise RuntimeError("self-mirror corpus lacks a family or chronological split")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.with_name(f".{self.output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            artifacts = {
                "games.parquet": _artifact(games, staging / "games.parquet"),
                "events.parquet": _artifact(events, staging / "events.parquet"),
                "targets.parquet": _artifact(targets, staging / "targets.parquet"),
            }
            joined = targets.join(games.select("game_id", "family", "our_role"), on="game_id")
            inventory = {
                "games": games.height,
                "events": events.height,
                "targets": targets.height,
                "by_family": {family: {"games": games.filter(pl.col("family") == family).height, "targets": joined.filter(pl.col("family") == family).height} for family in GLEE_FAMILIES},
                "by_split": dict(sorted(Counter(targets["chronological_split"].to_list()).items())),
                "by_target_kind": dict(sorted(Counter(targets["target_kind"].to_list()).items())),
                "known_account_targets": targets.filter(pl.col("account_key").is_not_null()).height,
            }
            manifest = {
                "schema_version": 1,
                "contract": SELF_MIRROR_CORPUS_CONTRACT,
                "status": "frozen-retrospective-shadow-corpus",
                "objective": {
                    "contract": SELF_MIRROR_OBJECTIVE_CONTRACT,
                    "target_actor": "self",
                    "prediction_frontier": "immediately before DeepRMM-01's authenticated move",
                    "predicted": ["categorical strategic action", "public numeric proposal coordinate", "log response delay"],
                    "message_reconstruction": False,
                },
                "boundary": "Opponent-observable game mechanics and authenticated prior events only; DeepRMM-01 private values, unrevealed quality, internal advisor state, policy version, Terra traces, and the target move are absent from model inputs.",
                "public_information_masks": {
                    "incomplete_information_player_values": True,
                    "unrevealed_persuasion_quality": True,
                    "hidden_identity_account_key": True,
                    "internal_engine_advisor_policy_versions": True,
                    "target_message": True,
                },
                "negotiation_price_coordinate": {"transform": "log1p(public price) / 20", "divisor": PUBLIC_NEGOTIATION_LOG_PRICE_DIVISOR, "private_source_normalization_used_only_for_inversion_before_masking": True},
                "source_corpus": {"path": str(self.source_corpus), "manifest_sha256": file_sha256(source_manifest_path), "contract": source_manifest.get("contract"), "artifacts": {name: source_manifest["artifacts"][name] for name in ("games.parquet", "events.parquet")}},
                "target_labels": {family: list(TARGET_LABELS[family]) for family in GLEE_FAMILIES},
                "inventory": inventory,
                "artifacts": artifacts,
                "implementation_sha256": file_sha256(Path(__file__)),
                "promotion": "offline shadow evidence only; the corpus and any trained mirror have no live policy authority",
            }
            _write_json(staging / "manifest.json", manifest)
            manifest_sha256 = file_sha256(staging / "manifest.json")
            os.replace(staging, self.output_dir)
        except BaseException:
            if staging.exists():
                for child in staging.iterdir():
                    child.unlink()
                staging.rmdir()
            raise
        return {"contract": SELF_MIRROR_CORPUS_CONTRACT, "output_dir": str(self.output_dir), "manifest_sha256": manifest_sha256, "inventory": inventory}
