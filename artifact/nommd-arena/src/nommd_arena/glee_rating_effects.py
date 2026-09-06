"""Validate and apply auditable corrections from gross history deltas to per-game rating effects."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Mapping


RATING_EFFECT_CORRECTIONS_CONTRACT = "glee-rating-effect-corrections-v1"
DEFAULT_RATING_EFFECT_CORRECTIONS_PATH = Path(__file__).resolve().parents[2] / "config" / "glee-rating-effect-corrections-v1.json"
_GLEE_FAMILIES = frozenset({"bargaining", "negotiation", "persuasion"})


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("rating-effect correction timestamp is missing")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"rating-effect correction {field} must be finite")
    return float(value)


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"rating-effect correction {field} must be a lowercase SHA-256 digest")
    return value


def _sensor_observation(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"rating-effect correction {label} sensor observation is missing")
    event_sequence = value.get("event_sequence")
    games_played = value.get("games_played")
    if isinstance(event_sequence, bool) or not isinstance(event_sequence, int) or event_sequence < 0:
        raise ValueError(f"rating-effect correction {label} event sequence is invalid")
    if isinstance(games_played, bool) or not isinstance(games_played, int) or games_played < 0:
        raise ValueError(f"rating-effect correction {label} game count is invalid")
    return {
        "event_sequence": event_sequence,
        "observed_at": str(value.get("observed_at") or ""),
        "games_played": games_played,
        "rating": _finite(value.get("rating"), field=f"{label}.rating"),
        "stats_sha256": _digest(value.get("stats_sha256"), field=f"{label}.stats_sha256"),
    }


def _validate_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("rating-effect correction record must be an object")
    game_id = str(value.get("game_id") or "")
    family = str(value.get("family") or "")
    reason = str(value.get("reason") or "")
    source = str(value.get("source") or "")
    if not game_id:
        raise ValueError("rating-effect correction game ID is missing")
    if family not in _GLEE_FAMILIES:
        raise ValueError(f"rating-effect correction family is invalid: {family}")
    if reason != "authenticated history delta includes pre-game inactivity decay":
        raise ValueError(f"rating-effect correction reason is unsupported: {reason}")
    if source != "adjacent authenticated sensor observations with one family-game increment":
        raise ValueError(f"rating-effect correction source is unsupported: {source}")
    completed_at = str(value.get("completed_at") or "")
    completed_timestamp = _timestamp(completed_at)
    raw_delta = _finite(value.get("raw_history_rating_delta"), field="raw_history_rating_delta")
    effect_delta = _finite(value.get("game_effect_rating_delta"), field="game_effect_rating_delta")
    history_record_sha256 = _digest(value.get("history_record_sha256"), field="history_record_sha256")
    previous = _sensor_observation(value.get("previous_sensor_observation"), label="previous")
    current = _sensor_observation(value.get("current_sensor_observation"), label="current")
    if int(current["event_sequence"]) <= int(previous["event_sequence"]):
        raise ValueError(f"rating-effect correction sensor sequence does not advance: {game_id}")
    if _timestamp(current["observed_at"]) <= _timestamp(previous["observed_at"]):
        raise ValueError(f"rating-effect correction sensor time does not advance: {game_id}")
    if completed_timestamp > _timestamp(current["observed_at"]):
        raise ValueError(f"rating-effect correction completion follows its confirming observation: {game_id}")
    if int(current["games_played"]) != int(previous["games_played"]) + 1:
        raise ValueError(f"rating-effect correction is not a single-game transition: {game_id}")
    observed_effect = float(current["rating"]) - float(previous["rating"])
    if not math.isclose(effect_delta, observed_effect, abs_tol=1e-9):
        raise ValueError(f"rating-effect correction disagrees with its sensor transition: {game_id}")
    if math.isclose(raw_delta, effect_delta, abs_tol=1e-9):
        raise ValueError(f"rating-effect correction does not change the gross history label: {game_id}")
    record = {
        "game_id": game_id,
        "family": family,
        "completed_at": completed_at,
        "raw_history_rating_delta": raw_delta,
        "game_effect_rating_delta": effect_delta,
        "history_record_sha256": history_record_sha256,
        "reason": reason,
        "source": source,
        "previous_sensor_observation": previous,
        "current_sensor_observation": current,
    }
    record["record_sha256"] = _sha(record)
    return record


def load_rating_effect_corrections(path: Path = DEFAULT_RATING_EFFECT_CORRECTIONS_PATH) -> dict[str, dict[str, object]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("contract") != RATING_EFFECT_CORRECTIONS_CONTRACT or payload.get("schema_version") != 1:
        raise ValueError("unsupported rating-effect corrections contract")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("rating-effect corrections records must be a list")
    result: dict[str, dict[str, object]] = {}
    for value in records:
        record = _validate_record(value)
        game_id = str(record["game_id"])
        if game_id in result:
            raise ValueError(f"duplicate rating-effect correction: {game_id}")
        result[game_id] = record
    return result


def effective_rating_delta(*, game_id: str, raw_rating_delta: float, history_record_sha256: str, corrections: Mapping[str, Mapping[str, object]]) -> tuple[float, Mapping[str, object] | None]:
    raw = _finite(raw_rating_delta, field="raw_rating_delta")
    correction = corrections.get(game_id)
    if correction is None:
        return raw, None
    expected_raw = float(correction["raw_history_rating_delta"])
    if not math.isclose(raw, expected_raw, abs_tol=1e-9):
        raise RuntimeError(f"rating-effect correction raw delta mismatch: {game_id}")
    if str(correction["history_record_sha256"]) != history_record_sha256:
        raise RuntimeError(f"rating-effect correction history receipt mismatch: {game_id}")
    return float(correction["game_effect_rating_delta"]), correction
