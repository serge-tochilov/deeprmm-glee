"""Append-only prospective registry for GLEE rating model v3 forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .glee_joint_rating_analysis import prediction_metrics
from .glee_rating_effects import effective_rating_delta, load_rating_effect_corrections
from .glee_rating_v3 import RATING_V3_CONTRACT, RATING_V3_MODEL_VERSION


RATING_V3_SHADOW_CONTRACT = "glee-rating-model-v3-prospective-shadow-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("prospective rating outcome must be finite")
    return float(value)


class RatingV3ShadowRegistry:
    """Store one immutable pre-outcome forecast and one immutable outcome per game."""

    def __init__(self, path: Path, *, model_sha256: str, model_cutoff: str, corrections_path: Path | None = None) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_sha256 = str(model_sha256)
        self.model_cutoff = str(model_cutoff)
        self.corrections_path = corrections_path.resolve() if corrections_path is not None else None
        self.corrections = load_rating_effect_corrections(self.corrections_path) if self.corrections_path is not None else {}
        self._initialize()
        self._import_corrections()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS game_contexts (
                    game_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    target_player TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_advisories (
                    turn_id TEXT PRIMARY KEY,
                    game_id TEXT NOT NULL REFERENCES game_contexts(game_id),
                    observed_at TEXT NOT NULL,
                    advisory_json TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecasts (
                    game_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    target_player TEXT NOT NULL,
                    terminal_at TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    terminal_sha256 TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    forecast_json TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    game_id TEXT PRIMARY KEY REFERENCES forecasts(game_id),
                    rating_delta REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    history_record_sha256 TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS causally_ineligible_outcomes (
                    game_id TEXT PRIMARY KEY REFERENCES forecasts(game_id),
                    reason TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    history_record_sha256 TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcome_corrections (
                    game_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    raw_rating_delta REAL NOT NULL,
                    game_effect_delta REAL NOT NULL,
                    correction_json TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS game_contexts_no_update BEFORE UPDATE ON game_contexts BEGIN SELECT RAISE(ABORT, 'rating v3 game contexts are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS game_contexts_no_delete BEFORE DELETE ON game_contexts BEGIN SELECT RAISE(ABORT, 'rating v3 game contexts are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS turn_advisories_no_update BEFORE UPDATE ON turn_advisories BEGIN SELECT RAISE(ABORT, 'rating v3 turn advisories are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS turn_advisories_no_delete BEFORE DELETE ON turn_advisories BEGIN SELECT RAISE(ABORT, 'rating v3 turn advisories are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts BEGIN SELECT RAISE(ABORT, 'rating v3 forecasts are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts BEGIN SELECT RAISE(ABORT, 'rating v3 forecasts are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS outcomes_no_update BEFORE UPDATE ON outcomes BEGIN SELECT RAISE(ABORT, 'rating v3 outcomes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS outcomes_no_delete BEFORE DELETE ON outcomes BEGIN SELECT RAISE(ABORT, 'rating v3 outcomes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS causally_ineligible_outcomes_no_update BEFORE UPDATE ON causally_ineligible_outcomes BEGIN SELECT RAISE(ABORT, 'rating v3 causally ineligible outcomes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS causally_ineligible_outcomes_no_delete BEFORE DELETE ON causally_ineligible_outcomes BEGIN SELECT RAISE(ABORT, 'rating v3 causally ineligible outcomes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS outcome_corrections_no_update BEFORE UPDATE ON outcome_corrections BEGIN SELECT RAISE(ABORT, 'rating v3 outcome corrections are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS outcome_corrections_no_delete BEFORE DELETE ON outcome_corrections BEGIN SELECT RAISE(ABORT, 'rating v3 outcome corrections are append-only'); END;
                """
            )
            expected = {"contract": RATING_V3_SHADOW_CONTRACT, "model_sha256": self.model_sha256, "model_cutoff": self.model_cutoff}
            connection.execute("BEGIN IMMEDIATE")
            for key, value in expected.items():
                existing = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
                if existing is not None and str(existing["value"]) != value:
                    raise RuntimeError(f"rating v3 shadow metadata conflict: {key}")
                connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value))
            connection.commit()
        finally:
            connection.close()
        self.path.chmod(0o600)

    def _import_corrections(self) -> None:
        for correction in self.corrections.values():
            self.register_outcome_correction(correction)

    def register_outcome_correction(self, correction: Mapping[str, object]) -> bool:
        game_id = str(correction["game_id"])
        payload = {key: value for key, value in correction.items() if key != "record_sha256"}
        values = {
            "game_id": game_id,
            "family": str(correction["family"]),
            "raw_rating_delta": float(correction["raw_history_rating_delta"]),
            "game_effect_delta": float(correction["game_effect_rating_delta"]),
            "correction_json": _canonical(payload),
            "record_sha256": str(correction["record_sha256"]),
        }
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outcome = connection.execute("SELECT rating_delta, history_record_sha256 FROM outcomes WHERE game_id = ?", (game_id,)).fetchone()
            if outcome is not None:
                effective_rating_delta(game_id=game_id, raw_rating_delta=float(outcome["rating_delta"]), history_record_sha256=str(outcome["history_record_sha256"]), corrections={game_id: correction})
            inserted = self._insert_immutable(connection, "outcome_corrections", game_id, values)
            connection.commit()
            return inserted
        finally:
            connection.close()

    def register_game_context(self, *, game_id: str, family: str, target_player: str, observed_at: str, context: Mapping[str, object]) -> bool:
        payload = {"game_id": game_id, "family": family, "target_player": target_player, "observed_at": observed_at, "context": dict(context)}
        values = {"game_id": game_id, "family": family, "target_player": target_player, "observed_at": observed_at, "context_json": _canonical(context), "record_sha256": _sha(payload)}
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            inserted = self._insert_immutable(connection, "game_contexts", game_id, values)
            connection.commit()
            return inserted
        finally:
            connection.close()

    def game_context(self, game_id: str) -> dict[str, object] | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT context_json FROM game_contexts WHERE game_id = ?", (game_id,)).fetchone()
        finally:
            connection.close()
        return json.loads(str(row["context_json"])) if row is not None else None

    def register_turn_advisory(self, *, turn_id: str, game_id: str, observed_at: str, advisory: Mapping[str, object]) -> bool:
        payload = {"turn_id": turn_id, "game_id": game_id, "observed_at": observed_at, "advisory": dict(advisory)}
        values = {"turn_id": turn_id, "game_id": game_id, "observed_at": observed_at, "advisory_json": _canonical(advisory), "record_sha256": _sha(payload)}
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT record_sha256 FROM turn_advisories WHERE turn_id = ?", (turn_id,)).fetchone()
            if existing is not None:
                if str(existing["record_sha256"]) != values["record_sha256"]:
                    raise RuntimeError(f"conflicting immutable rating v3 advisory record: {turn_id}")
                connection.commit()
                return False
            connection.execute(
                "INSERT INTO turn_advisories(turn_id, game_id, observed_at, advisory_json, record_sha256) VALUES (?, ?, ?, ?, ?)",
                (turn_id, game_id, observed_at, values["advisory_json"], values["record_sha256"]),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def turn_advisory(self, turn_id: str) -> dict[str, object] | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT advisory_json FROM turn_advisories WHERE turn_id = ?", (turn_id,)).fetchone()
        finally:
            connection.close()
        return json.loads(str(row["advisory_json"])) if row is not None else None

    def terminal_forecast(self, game_id: str) -> dict[str, object] | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT family, target_player, terminal_at, terminal_sha256, forecast_json FROM forecasts WHERE game_id = ?", (game_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {"game_id": game_id, "family": str(row["family"]), "target_player": str(row["target_player"]), "terminal_at": str(row["terminal_at"]), "terminal_sha256": str(row["terminal_sha256"]), "forecast": json.loads(str(row["forecast_json"]))}

    def terminal_forecasts(self) -> list[dict[str, object]]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT game_id, family, target_player, terminal_at, forecast_json FROM forecasts ORDER BY terminal_at, game_id").fetchall()
        finally:
            connection.close()
        return [{"game_id": str(row["game_id"]), "family": str(row["family"]), "target_player": str(row["target_player"]), "terminal_at": str(row["terminal_at"]), "forecast": json.loads(str(row["forecast_json"]))} for row in rows]

    @staticmethod
    def _insert_immutable(connection: sqlite3.Connection, table: str, key: str, values: Mapping[str, object]) -> bool:
        existing = connection.execute(f"SELECT record_sha256 FROM {table} WHERE game_id = ?", (key,)).fetchone()
        if existing is not None:
            if str(existing["record_sha256"]) != values["record_sha256"]:
                raise RuntimeError(f"conflicting immutable rating v3 shadow record: {table}/{key}")
            return False
        columns = tuple(values)
        connection.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _column in columns)})", tuple(values[column] for column in columns))
        return True

    def register(self, *, game_id: str, family: str, target_player: str, terminal_at: str, registered_at: str, terminal_sha256: str, forecast: Mapping[str, object]) -> bool:
        if forecast.get("contract") != RATING_V3_CONTRACT or forecast.get("model_version") != RATING_V3_MODEL_VERSION:
            raise ValueError("prospective registration requires a rating v3 forecast")
        if str(forecast.get("family")) != family or str(forecast.get("target_player")) != target_player:
            raise ValueError("prospective registration target differs from its forecast")
        if _timestamp(terminal_at) <= _timestamp(self.model_cutoff):
            raise ValueError("prospective target is not later than the model cutoff")
        if _timestamp(registered_at) < _timestamp(terminal_at):
            raise ValueError("prospective registration predates its terminal observation")
        payload = {"game_id": game_id, "family": family, "target_player": target_player, "terminal_at": terminal_at, "registered_at": registered_at, "terminal_sha256": terminal_sha256, "model_sha256": self.model_sha256, "forecast": dict(forecast)}
        values = {"game_id": game_id, "family": family, "target_player": target_player, "terminal_at": terminal_at, "registered_at": registered_at, "terminal_sha256": terminal_sha256, "model_sha256": self.model_sha256, "forecast_json": _canonical(forecast), "record_sha256": _sha(payload)}
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            inserted = self._insert_immutable(connection, "forecasts", game_id, values)
            connection.commit()
            return inserted
        finally:
            connection.close()

    def mature(self, *, game_id: str, rating_delta: float, observed_at: str, history_record_sha256: str) -> bool:
        numeric_delta = _finite(rating_delta)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            forecast = connection.execute("SELECT registered_at FROM forecasts WHERE game_id = ?", (game_id,)).fetchone()
            if forecast is None:
                raise KeyError(f"rating v3 shadow has no forecast for {game_id}")
            if _timestamp(observed_at) <= _timestamp(str(forecast["registered_at"])):
                raise ValueError("rating outcome was observable before or at forecast registration")
            correction_row = connection.execute("SELECT correction_json FROM outcome_corrections WHERE game_id = ?", (game_id,)).fetchone()
            if correction_row is not None:
                correction = json.loads(str(correction_row["correction_json"]))
                effective_rating_delta(game_id=game_id, raw_rating_delta=numeric_delta, history_record_sha256=history_record_sha256, corrections={game_id: correction})
            payload = {"game_id": game_id, "rating_delta": numeric_delta, "observed_at": observed_at, "history_record_sha256": history_record_sha256}
            values = {**payload, "record_sha256": _sha(payload)}
            inserted = self._insert_immutable(connection, "outcomes", game_id, values)
            connection.commit()
            return inserted
        finally:
            connection.close()

    def quarantine_causally_ineligible(self, *, game_id: str, observed_at: str, history_record_sha256: str) -> bool:
        reason = "rating outcome was observable before or at forecast registration"
        payload = {"game_id": game_id, "reason": reason, "observed_at": observed_at, "history_record_sha256": history_record_sha256}
        values = {**payload, "record_sha256": _sha(payload)}
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            inserted = self._insert_immutable(connection, "causally_ineligible_outcomes", game_id, values)
            connection.commit()
            return inserted
        finally:
            connection.close()

    def mature_from_history(self, history_database: Path) -> dict[str, int]:
        history = sqlite3.connect(f"file:{history_database.resolve()}?mode=ro", uri=True)
        history.row_factory = sqlite3.Row
        registry = self._connect()
        matured = 0
        pending = 0
        causally_ineligible = 0
        try:
            forecasts = registry.execute("SELECT game_id, registered_at FROM forecasts WHERE game_id NOT IN (SELECT game_id FROM outcomes) AND game_id NOT IN (SELECT game_id FROM causally_ineligible_outcomes) ORDER BY registered_at, game_id").fetchall()
            for forecast in forecasts:
                row = history.execute("SELECT rating_delta, first_seen_at, record_sha256 FROM games WHERE game_id = ? AND rating_delta IS NOT NULL", (str(forecast["game_id"]),)).fetchone()
                if row is None:
                    pending += 1
                    continue
                if _timestamp(str(row["first_seen_at"])) <= _timestamp(str(forecast["registered_at"])):
                    causally_ineligible += int(self.quarantine_causally_ineligible(game_id=str(forecast["game_id"]), observed_at=str(row["first_seen_at"]), history_record_sha256=str(row["record_sha256"])))
                    continue
                matured += int(self.mature(game_id=str(forecast["game_id"]), rating_delta=float(row["rating_delta"]), observed_at=str(row["first_seen_at"]), history_record_sha256=str(row["record_sha256"])))
        finally:
            registry.close()
            history.close()
        return {"matured": matured, "pending": pending, "causally_ineligible": causally_ineligible}

    def summary(self) -> dict[str, object]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT f.game_id, f.family, f.forecast_json, o.rating_delta, o.history_record_sha256, o.observed_at, c.correction_json FROM forecasts AS f JOIN outcomes AS o USING(game_id) LEFT JOIN outcome_corrections AS c USING(game_id) ORDER BY o.observed_at, f.game_id").fetchall()
            registered = int(connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0])
            causally_ineligible = int(connection.execute("SELECT COUNT(*) FROM causally_ineligible_outcomes").fetchone()[0])
            contexts = int(connection.execute("SELECT COUNT(*) FROM game_contexts").fetchone()[0])
            advisories = int(connection.execute("SELECT COUNT(*) FROM turn_advisories").fetchone()[0])
        finally:
            connection.close()
        values = []
        for row in rows:
            forecast = json.loads(str(row["forecast_json"]))
            correction = json.loads(str(row["correction_json"])) if row["correction_json"] is not None else None
            actual, applied = effective_rating_delta(game_id=str(row["game_id"]), raw_rating_delta=float(row["rating_delta"]), history_record_sha256=str(row["history_record_sha256"]), corrections={str(row["game_id"]): correction} if correction is not None else {})
            values.append({"game_id": str(row["game_id"]), "family": str(row["family"]), "actual": actual, "raw_actual": float(row["rating_delta"]), "predicted": float(forecast["predicted_delta"]), "corrected": applied is not None})
        by_family = {}
        for family in ("bargaining", "negotiation", "persuasion"):
            family_rows = [row for row in values if row["family"] == family]
            by_family[family] = prediction_metrics([row["actual"] for row in family_rows], [row["predicted"] for row in family_rows]) if family_rows else {"count": 0}
            by_family[family]["corrected_outcomes"] = sum(bool(row["corrected"]) for row in family_rows)
        corrected = sum(bool(row["corrected"]) for row in values)
        return {"contract": RATING_V3_SHADOW_CONTRACT, "model_sha256": self.model_sha256, "model_cutoff": self.model_cutoff, "game_contexts": contexts, "turn_advisories": advisories, "registered": registered, "matured": len(values), "corrected_outcomes": corrected, "gross_history_delta_sum": sum(float(row["raw_actual"]) for row in values), "game_effect_delta_sum": sum(float(row["actual"]) for row in values), "causally_ineligible": causally_ineligible, "pending": registered - len(values) - causally_ineligible, "families": by_family}
